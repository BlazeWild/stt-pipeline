#!/usr/bin/env python3
"""Dependency-free normalization and edit-distance metrics for Nepali ASR."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher

DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower().translate(DEVANAGARI_DIGITS)
    text = "".join(" " if unicodedata.category(ch).startswith("P") else ch for ch in text)
    return re.sub(r"\s+", " ", text).strip()


@dataclass
class Edits:
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0
    reference_units: int = 0

    @property
    def error_rate(self) -> float:
        errors = self.substitutions + self.deletions + self.insertions
        if self.reference_units == 0:
            return 0.0 if errors == 0 else 1.0
        return errors / self.reference_units

    def add(self, other: "Edits") -> None:
        self.substitutions += other.substitutions
        self.deletions += other.deletions
        self.insertions += other.insertions
        self.reference_units += other.reference_units


def edit_counts(reference: list[str], hypothesis: list[str]) -> Edits:
    rows, cols = len(reference) + 1, len(hypothesis) + 1
    dp = [[None] * cols for _ in range(rows)]
    dp[0][0] = (0, 0, 0)
    for i in range(1, rows):
        s, d, ins = dp[i - 1][0]
        dp[i][0] = (s, d + 1, ins)
    for j in range(1, cols):
        s, d, ins = dp[0][j - 1]
        dp[0][j] = (s, d, ins + 1)
    for i in range(1, rows):
        for j in range(1, cols):
            if reference[i - 1] == hypothesis[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                continue
            candidates = []
            s, d, ins = dp[i - 1][j - 1]
            candidates.append((s + 1, d, ins))
            s, d, ins = dp[i - 1][j]
            candidates.append((s, d + 1, ins))
            s, d, ins = dp[i][j - 1]
            candidates.append((s, d, ins + 1))
            dp[i][j] = min(candidates, key=lambda x: (sum(x), x[0], x[2]))
    s, d, ins = dp[-1][-1]
    return Edits(s, d, ins, len(reference))


def word_edits(reference: str, hypothesis: str) -> Edits:
    return edit_counts(normalize(reference).split(), normalize(hypothesis).split())


def char_edits(reference: str, hypothesis: str, normalized: bool = True) -> Edits:
    if normalized:
        reference, hypothesis = normalize(reference), normalize(hypothesis)
    return edit_counts(list(reference), list(hypothesis))


def entity_score(entity: str, hypothesis: str) -> tuple[bool, float]:
    entity_n, hypothesis_n = normalize(entity), normalize(hypothesis)
    entity_words, hypothesis_words = entity_n.split(), hypothesis_n.split()
    exact = bool(entity_words) and any(
        hypothesis_words[i : i + len(entity_words)] == entity_words
        for i in range(len(hypothesis_words) - len(entity_words) + 1)
    )
    if exact:
        return True, 1.0
    # Best approximate match over word windows, useful alongside strict recall.
    words = hypothesis_words
    width = max(1, len(entity_words))
    candidates = [" ".join(words[i : i + width]) for i in range(max(1, len(words) - width + 1))]
    best = max((SequenceMatcher(None, entity_n, value).ratio() for value in candidates), default=0.0)
    return False, best
