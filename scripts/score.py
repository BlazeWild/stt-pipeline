#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from stt_metrics import Edits, char_edits, entity_score, normalize, word_edits


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def summarize(rows: list[tuple[dict, dict]]) -> dict:
    words, chars, raw_chars = Edits(), Edits(), Edits()
    entities = defaultdict(lambda: {"total": 0, "exact": 0, "similarity_sum": 0.0})
    latency = cost = audio_seconds = 0.0
    peak_vram = 0.0
    silence_total = silence_nonempty = 0
    for ref, pred in rows:
        hypothesis = pred.get("text", "") if pred.get("status", "ok") == "ok" else ""
        words.add(word_edits(ref.get("reference", ""), hypothesis))
        chars.add(char_edits(ref.get("reference", ""), hypothesis))
        raw_chars.add(char_edits(ref.get("reference", ""), hypothesis, normalized=False))
        latency += float(pred.get("latency_seconds") or 0)
        cost += float(pred.get("api_cost_usd") or 0)
        audio_seconds += float(ref.get("duration_seconds") or 0)
        peak_vram = max(peak_vram, float(pred.get("peak_vram_mb") or 0))
        if ref.get("primary_stratum") == "silence":
            silence_total += 1
            silence_nonempty += bool(normalize(hypothesis))
        for entity in ref.get("entities", []):
            exact, similarity = entity_score(entity["text"], hypothesis)
            bucket = entities[entity["type"]]
            bucket["total"] += 1
            bucket["exact"] += exact
            bucket["similarity_sum"] += similarity
    entity_output = {}
    for kind, value in sorted(entities.items()):
        entity_output[kind] = {
            "total": value["total"],
            "exact_accuracy": value["exact"] / value["total"],
            "mean_similarity": value["similarity_sum"] / value["total"],
        }
    denom = words.reference_units
    return {
        "clips": len(rows),
        "wer": words.error_rate,
        "cer": chars.error_rate,
        "raw_cer": raw_chars.error_rate,
        "word_edits": words.__dict__,
        "missing_word_rate": words.deletions / denom if denom else 0.0,
        "insertion_rate": words.insertions / denom if denom else 0.0,
        "entities": entity_output,
        "silence_hallucination_rate": silence_nonempty / silence_total if silence_total else None,
        "latency_seconds": latency,
        "audio_seconds": audio_seconds,
        "real_time_factor": latency / audio_seconds if audio_seconds else None,
        "peak_vram_mb": peak_vram,
        "api_cost_usd": cost,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--allow-partial", action="store_true", help="score only predicted IDs; intended for quota-limited pilots")
    parser.add_argument("--include-ids-from", type=Path, help="restrict manifest to successful IDs in another predictions JSONL")
    args = parser.parse_args()
    refs = read_jsonl(args.manifest)
    if args.include_ids_from:
        include = {
            row["id"]
            for row in read_jsonl(args.include_ids_from)
            if row.get("status", "ok") == "ok"
        }
        refs = [row for row in refs if row["id"] in include]
    predictions = read_jsonl(args.predictions)
    pred_by_id = {row["id"]: row for row in predictions}
    ref_ids = [row["id"] for row in refs]
    duplicate_count = len(predictions) - len(pred_by_id)
    attempted_ids = {clip_id for clip_id, row in pred_by_id.items() if row.get("status", "ok") == "ok"}
    missing = [clip_id for clip_id in ref_ids if clip_id not in attempted_ids]
    # When scoring a matched subset, predictions for deliberately excluded
    # manifest IDs are expected and must not invalidate the run.
    extra = [] if args.include_ids_from else sorted(set(pred_by_id) - set(ref_ids))
    if duplicate_count or extra or (missing and not args.allow_partial):
        raise SystemExit(f"prediction ID mismatch: duplicates={duplicate_count}, missing={missing}, extra={extra}")
    # A request failure is operational evidence, not an empty transcription.
    # Exclude it from accuracy denominators and surface it in coverage/errors.
    pairs = [(row, pred_by_id[row["id"]]) for row in refs if row["id"] in attempted_ids]
    output = {
        "overall": summarize(pairs),
        "by_stratum": {},
        "coverage": {
            "successful": len(pairs),
            "attempted": len(pred_by_id),
            "manifest": len(refs),
            "missing_or_failed_ids": missing,
        },
    }
    for stratum in sorted({row["primary_stratum"] for row in refs}):
        output["by_stratum"][stratum] = summarize([(r, p) for r, p in pairs if r["primary_stratum"] == stratum])
    output["errors"] = Counter(pred_by_id[x].get("status", "ok") for x in ref_ids if x in pred_by_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2, default=dict) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
