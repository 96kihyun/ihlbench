"""Run directories: one manifest, one append-only record file.

    results/<run>/manifest.json   what produced the records
    results/<run>/records.jsonl   one line per trial, never rewritten

Resuming a run is allowed only when the identity of the run is unchanged
(model, condition, sampling settings, data fingerprint). Anything else is a
different run and must go to a different directory; mixing them silently is
how a table ends up with numbers no file can reproduce.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"

MANIFEST = "manifest.json"
RECORDS = "records.jsonl"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_state() -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()

    commit = run("rev-parse", "HEAD") or None
    dirty = bool(run("status", "--porcelain", "--", "ihl", "scripts", "data", "serving", "dpo"))
    return {"commit": commit, "dirty": dirty}


def trial_seed(trial_id: str, base_seed: int) -> int:
    """Deterministic per-trial sampling seed, sent to the server."""
    digest = hashlib.sha256(f"{base_seed}:{trial_id}".encode()).hexdigest()
    return int(digest[:8], 16)


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for n, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{n}: {e}") from e


class Run:
    def __init__(self, run_dir: Path, identity: dict[str, Any], extra: dict[str, Any] | None = None):
        self.dir = run_dir
        self.identity = identity
        self.manifest_path = run_dir / MANIFEST
        self.records_path = run_dir / RECORDS
        run_dir.mkdir(parents=True, exist_ok=True)
        state = {"ts": now(), "git": git_state(), **(extra or {})}
        if self.manifest_path.exists():
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if manifest["identity"] != identity:
                diff = {
                    k: (manifest["identity"].get(k), identity.get(k))
                    for k in set(manifest["identity"]) | set(identity)
                    if manifest["identity"].get(k) != identity.get(k)
                }
                raise SystemExit(f"{run_dir}: run identity changed, refusing to resume: {diff}")
            manifest["sessions"].append(state)
        else:
            manifest = {"identity": identity, "created": state["ts"], "sessions": [state]}
        self.manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        self.done = {r["trial_id"] for r in read_jsonl(self.records_path)} if self.records_path.exists() else set()

    def write(self, record: dict[str, Any]) -> None:
        if record["trial_id"] in self.done:
            raise ValueError(f"duplicate trial {record['trial_id']}")
        with self.records_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.done.add(record["trial_id"])


def load_run(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads((run_dir / MANIFEST).read_text(encoding="utf-8"))
    path = run_dir / RECORDS
    records = list(read_jsonl(path)) if path.exists() else []
    return manifest, records
