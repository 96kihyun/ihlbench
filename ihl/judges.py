"""LLM judges for the parts of a record that are not scored by tool name.

A judge task is a YAML file in data/judges/: system prompt, rubric template
(formatted with fields of the record), and the label set. Judgments are
stored next to the run, one file per (task, judge model), so several judges
can grade the same run and be compared for agreement.
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import yaml

from ihl.catalog import DATA_DIR


@dataclass(frozen=True)
class JudgeTask:
    name: str
    applies_to: str
    system: str
    rubric: str
    labels: tuple[str, ...]

    @property
    def fields(self) -> tuple[str, ...]:
        return tuple(f for _, f, _, _ in string.Formatter().parse(self.rubric) if f)

    def render(self, record: dict[str, Any]) -> str:
        return self.rubric.format(**{f: record.get(f) or "" for f in self.fields})

    def parse(self, text: str | None) -> str | None:
        if not text or not text.strip():
            return None
        pattern = re.compile(r"\b(" + "|".join(map(re.escape, self.labels)) + r")\b", re.I)
        first = text.strip().splitlines()[0]
        m = pattern.search(first) or pattern.search(text)
        return m.group(1).lower() if m else None


@cache
def load_task(name: str) -> JudgeTask:
    path = DATA_DIR / "judges" / f"{name}.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if set(raw) != {"applies_to", "system", "rubric", "labels"}:
        raise ValueError(f"{path}: needs exactly applies_to, system, rubric, labels")
    return JudgeTask(name, raw["applies_to"], raw["system"], raw["rubric"], tuple(raw["labels"]))


def judgments_path(run_dir: Path, task: str, judge_model: str) -> Path:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", judge_model).strip("_")
    return run_dir / "judgments" / f"{task}__{slug}.jsonl"
