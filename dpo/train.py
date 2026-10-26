#!/usr/bin/env python3
"""Single-device LoRA DPO for Qwen3.5-9B, Gemma-4-12B and Ministral-3-14B (`family` in the config).

Loop is the old torchtune recipe (concatenated chosen/rejected, adapter-off
reference, DPO + chosen NLL). Model/tokenizer/checkpointer are Hugging Face
because Qwen3.5 is not a torchtune Llama.

    python dpo/train.py --config dpo/configs/qwen_value.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from format_calls import apply_prompt_template, encode_completion, prompt_messages


def load_cfg(path: str, overrides: list[str]):
    import yaml

    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cfg = _wrap(raw)
    for item in overrides:
        if "=" not in item:
            raise SystemExit(f"override must be key=value, got {item!r}")
        dotted, _, value = item.partition("=")
        _set_dotted(cfg, dotted, yaml.safe_load(value))
    return cfg


class _Dot(dict):
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e

    def __setattr__(self, k, v):
        self[k] = v

    def get(self, k, default=None):
        return dict.get(self, k, default)


def _wrap(obj):
    if isinstance(obj, dict):
        return _Dot({k: _wrap(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_wrap(x) for x in obj]
    return obj


def _set_dotted(cfg: _Dot, dotted: str, value) -> None:
    keys = dotted.split(".")
    cur = cfg
    for k in keys[:-1]:
        nxt = cur.get(k)
        if not isinstance(nxt, _Dot):
            nxt = _Dot()
            cur[k] = nxt
        cur = nxt
    cur[keys[-1]] = _wrap(value) if isinstance(value, dict) else value


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def resolve_hf_tokenizer(processor):
    """Use the HF-facing tokenizer, not mistral_common's inner MistralTokenizer.

    AutoProcessor.tokenizer is the right object (Qwen/Gemma). Ministral's
    AutoTokenizer is MistralCommonBackend: it has pad/eos, but .tokenizer is
    the non-callable mistral_common wrapper that lacks those attributes.
    """
    inner = getattr(processor, "tokenizer", None)
    if (
        inner is not None
        and callable(inner)
        and (hasattr(inner, "pad_token_id") or hasattr(inner, "eos_token_id"))
    ):
        return inner
    if hasattr(processor, "pad_token_id") or hasattr(processor, "eos_token_id"):
        return processor
    return inner if inner is not None else processor


class PreferenceDataset(Dataset):
    def __init__(self, rows: list[dict], processor, max_seq_len: int, family: str):
        self.processor = processor
        self.tokenizer = resolve_hf_tokenizer(processor)
        self.max_seq_len = max_seq_len
        self.family = family
        self.examples = []
        skipped = 0
        for row in rows:
            ex = self._encode(row)
            if ex is None:
                skipped += 1
                continue
            self.examples.append(ex)
        if skipped:
            print(f"skipped {skipped} pairs over max_seq_len={max_seq_len}", flush=True)
        if not self.examples:
            raise RuntimeError("no training pairs fit max_seq_len")

    def _encode(self, row: dict) -> dict | None:
        tok = self.tokenizer
        messages = prompt_messages(row["prompt"]["system"], row["prompt"]["user"])
        prompt = apply_prompt_template(
            self.processor, messages, row["prompt"]["tools"], self.family
        )
        if isinstance(prompt, list):
            prompt_ids = prompt
        else:
            prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        chosen = encode_completion(
            tok, self.family, row["chosen_tool"], row["chosen_rationale"]
        )
        rejected = encode_completion(
            tok, self.family, row["rejected_tool"], row["rejected_rationale"]
        )

        def pack(comp):
            ids = prompt_ids + comp
            if len(ids) > self.max_seq_len:
                return None
            labels = [-100] * len(prompt_ids) + list(comp)
            return ids, labels

        ch = pack(chosen)
        rj = pack(rejected)
        if ch is None or rj is None:
            return None
        return {
            "chosen_ids": ch[0],
            "chosen_labels": ch[1],
            "rejected_ids": rj[0],
            "rejected_labels": rj[1],
            "id": row.get("id"),
            "bucket": row.get("bucket"),
        }

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


def collate(batch, pad_id: int):
    def as_rows(key):
        return [b[key] for b in batch]

    c_ids, c_lab = as_rows("chosen_ids"), as_rows("chosen_labels")
    r_ids, r_lab = as_rows("rejected_ids"), as_rows("rejected_labels")
    max_len = max(len(s) for s in c_ids + r_ids)

    def pad(seqs, fill):
        out, mask = [], []
        for s in seqs:
            pad_n = max_len - len(s)
            out.append(s + [fill] * pad_n)
            mask.append([1] * len(s) + [0] * pad_n)
        return torch.tensor(out, dtype=torch.long), torch.tensor(mask, dtype=torch.long)

    c_ids_t, c_mask = pad(c_ids, pad_id)
    c_lab_t, _ = pad(c_lab, -100)
    r_ids_t, r_mask = pad(r_ids, pad_id)
    r_lab_t, _ = pad(r_lab, -100)
    return {
        "input_ids": torch.cat([c_ids_t, r_ids_t], dim=0),
        "attention_mask": torch.cat([c_mask, r_mask], dim=0),
        "labels": torch.cat([c_lab_t, r_lab_t], dim=0),
    }


def sequence_logps(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    logp = F.log_softmax(shift_logits.float(), dim=-1)
    gather = torch.gather(logp, 2, shift_labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    mask = shift_labels.ne(-100)
    return (gather * mask).sum(dim=-1)


def completion_lengths(labels: torch.Tensor) -> torch.Tensor:
    return (labels[:, 1:].ne(-100)).sum(dim=-1).clamp(min=1).float()


def dpo_loss(policy_c, policy_r, ref_c, ref_r, beta: float):
    pi = policy_c - policy_r
    ref = ref_c - ref_r
    logits = beta * (pi - ref)
    loss = -F.logsigmoid(logits)
    chosen_r = (beta * (policy_c - ref_c)).detach()
    rejected_r = (beta * (policy_r - ref_r)).detach()
    return loss, chosen_r, rejected_r


def append_metrics(path: str, step: int, row: dict) -> None:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    out = {"step": int(step)}
    for k, v in row.items():
        if isinstance(v, torch.Tensor):
            v = v.detach().float().mean().cpu().item()
        if isinstance(v, (int, float)):
            out[k] = float(v)
    with dest.open("a", encoding="utf-8") as f:
        f.write(json.dumps(out) + "\n")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model(cfg):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

    family = str(cfg.get("family", "qwen"))
    dtype = torch.bfloat16 if str(cfg.dtype) == "bf16" else torch.float32
    if family == "ministral":
        processor = AutoTokenizer.from_pretrained(cfg.model_path)
    else:
        processor = AutoProcessor.from_pretrained(cfg.model_path, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        cfg.model_path,
        dtype=dtype,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    tgt = cfg.lora.target_modules
    if isinstance(tgt, str):
        target_modules = tgt
    else:
        target_modules = list(tgt)
    lora = LoraConfig(
        r=int(cfg.lora.rank),
        lora_alpha=int(cfg.lora.alpha),
        lora_dropout=float(cfg.lora.dropout),
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.to(cfg.device)
    if bool(cfg.training.gradient_checkpointing):
        try:
            model.enable_input_require_grads()
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            model.gradient_checkpointing_enable()
    model.print_trainable_parameters()
    return processor, model


def concatenated_forward(model, batch, device):
    ids = batch["input_ids"].to(device)
    mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    out = model(input_ids=ids, attention_mask=mask)
    logits = out.logits if hasattr(out, "logits") else out[0]
    logps = sequence_logps(logits, labels)
    n = ids.size(0) // 2
    return logps[:n], logps[n:], completion_lengths(labels)[:n]


def run_validation(model, loader, device, beta: float) -> float:
    was = model.training
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in loader:
            pol_c, pol_r, _ = concatenated_forward(model, batch, device)
            with model.disable_adapter():
                ref_c, ref_r, _ = concatenated_forward(model, batch, device)
            loss, _, _ = dpo_loss(pol_c, pol_r, ref_c, ref_r, beta)
            losses.append(float(loss.mean().cpu()))
    model.train(was)
    return sum(losses) / max(len(losses), 1)


def save_adapter(model, processor, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    processor.save_pretrained(path)
    print(f"saved adapter {path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("overrides", nargs="*", help="dotted.key=value")
    args = ap.parse_args()
    cfg = load_cfg(args.config, args.overrides)
    set_seed(int(cfg.training.seed))

    family = str(cfg.get("family", "qwen"))
    processor, model = load_model(cfg)
    tok = resolve_hf_tokenizer(processor)
    pad_id = getattr(tok, "pad_token_id", None)
    if pad_id is None:
        if getattr(tok, "pad_token", None) is None and getattr(tok, "eos_token", None) is not None:
            tok.pad_token = tok.eos_token
        pad_id = getattr(tok, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tok, "eos_token_id", None)
    device = cfg.device

    train_rows = load_jsonl(str(cfg.dataset.train))
    val_rows = load_jsonl(str(cfg.dataset.val)) if cfg.dataset.get("val") else []
    max_len = int(cfg.dataset.max_seq_len)
    train_ds = PreferenceDataset(train_rows, processor, max_len, family)
    val_ds = (
        PreferenceDataset(val_rows, processor, max_len, family) if val_rows else None
    )

    bs = int(cfg.training.batch_size)
    accum = int(cfg.training.gradient_accumulation_steps)
    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=True,
        collate_fn=lambda b: collate(b, pad_id),
        drop_last=True,
    )
    val_loader = None
    if val_ds:
        val_loader = DataLoader(
            val_ds,
            batch_size=bs,
            shuffle=False,
            collate_fn=lambda b: collate(b, pad_id),
            drop_last=False,
        )

    steps_per_epoch = max(1, len(train_loader) // accum)
    epochs = int(cfg.training.epochs)
    max_steps = cfg.training.get("max_steps")
    total_steps = steps_per_epoch * epochs
    if max_steps is not None:
        total_steps = min(total_steps, int(max_steps))

    opt_kwargs = dict(
        lr=float(cfg.training.lr),
        weight_decay=float(cfg.training.weight_decay),
    )
    try:
        opt = torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad),
            fused=str(device).startswith("cuda"),
            **opt_kwargs,
        )
    except TypeError:
        opt = torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad),
            **opt_kwargs,
        )
    warmup = int(cfg.training.warmup_steps)

    def lr_at(step: int) -> float:
        base = float(cfg.training.lr)
        if step < warmup:
            return base * (step + 1) / max(warmup, 1)
        t = (step - warmup) / max(total_steps - warmup, 1)
        return base * 0.5 * (1.0 + math.cos(math.pi * min(t, 1.0)))

    beta = float(cfg.training.beta)
    rpo = float(cfg.training.get("rpo_alpha", 0.0) or 0.0)
    objective = str(cfg.training.get("objective", "dpo")).lower()
    if objective not in ("dpo", "sft"):
        raise SystemExit(f"training.objective must be dpo or sft, got {objective!r}")
    val_every = cfg.training.get("val_every_n_steps")
    save_every = cfg.training.get("save_every_n_steps")
    log_every = int(cfg.training.log_every_n_steps)
    max_grad = float(cfg.training.max_grad_norm)
    out_dir = Path(str(cfg.output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = str(cfg.metric_log)

    print(
        f"train={len(train_ds)} val={len(val_ds) if val_ds else 0} "
        f"steps/epoch={steps_per_epoch} total_steps={total_steps} "
        f"objective={objective} beta={beta} rpo_alpha={rpo}",
        flush=True,
    )

    model.train()
    global_step = 0
    opt.zero_grad(set_to_none=True)
    running = 0.0
    t0 = time.perf_counter()
    stop = False

    for epoch in range(epochs):
        pbar = tqdm(total=steps_per_epoch, desc=f"epoch {epoch}")
        for idx, batch in enumerate(train_loader):
            if global_step >= total_steps:
                stop = True
                break
            pol_c, pol_r, n_tok = concatenated_forward(model, batch, device)
            if objective == "sft":
                loss = (-pol_c / n_tok).mean()
                ch_r = pol_c.detach()
                rj_r = pol_r.detach()
            else:
                with torch.no_grad(), model.disable_adapter():
                    ref_c, ref_r, _ = concatenated_forward(model, batch, device)
                loss_vec, ch_r, rj_r = dpo_loss(pol_c, pol_r, ref_c, ref_r, beta)
                loss = loss_vec.mean()
                if rpo:
                    loss = loss + rpo * (-pol_c / n_tok).mean()
            loss = loss / accum
            loss.backward()
            running += float(loss.detach())

            if (idx + 1) % accum != 0:
                continue

            torch.nn.utils.clip_grad_norm_(
                (p for p in model.parameters() if p.requires_grad), max_grad
            )
            for g in opt.param_groups:
                g["lr"] = lr_at(global_step)
            opt.step()
            opt.zero_grad(set_to_none=True)
            global_step += 1
            pbar.update(1)
            pbar.set_postfix(loss=running)

            if global_step % log_every == 0:
                acc = (ch_r > rj_r).float().mean()
                row = {
                    "loss": running,
                    "lr": opt.param_groups[0]["lr"],
                    "rewards/chosen": ch_r.mean(),
                    "rewards/rejected": rj_r.mean(),
                    "rewards/accuracies": acc,
                    "rewards/margins": (ch_r - rj_r).mean(),
                    "log_probs/chosen": pol_c.mean(),
                    "log_probs/rejected": pol_r.mean(),
                    "epoch": epoch,
                }
                append_metrics(metrics_path, global_step, row)
            running = 0.0

            if (
                val_loader is not None
                and val_every
                and global_step % int(val_every) == 0
            ):
                vloss = run_validation(model, val_loader, device, beta)
                append_metrics(metrics_path, global_step, {"val_loss": vloss})
                print(f"\nval_loss={vloss:.4f} step={global_step}", flush=True)

            if save_every and global_step % int(save_every) == 0:
                save_adapter(model, processor, out_dir / f"step_{global_step}")

        pbar.close()
        save_adapter(model, processor, out_dir / f"epoch_{epoch}")
        if stop:
            break

    save_adapter(model, processor, out_dir / "final")
    print(f"done in {time.perf_counter() - t0:.1f}s  adapter={out_dir / 'final'}", flush=True)


if __name__ == "__main__":
    main()
