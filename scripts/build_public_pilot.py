#!/usr/bin/env python3
"""Build a deterministic 150-clip public pilot from FLEURS Nepali validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import subprocess
import tempfile
from pathlib import Path

import pyarrow.parquet as pq

LATIN = re.compile(r"[A-Za-z]")
DIGITS = re.compile(r"[0-9०-९][0-9०-९,\.]*")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, default=Path("data/public_sources/fleurs_ne_np/validation.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/public_pilot"))
    parser.add_argument("--clips", type=int, default=150)
    parser.add_argument("--seed", type=int, default=20261025)
    args = parser.parse_args()

    table = pq.read_table(args.parquet, columns=["id", "num_samples", "audio", "transcription", "raw_transcription"])
    rows = list(enumerate(table.to_pylist()))
    if args.clips > len(rows):
        raise SystemExit(f"requested {args.clips} clips but source has only {len(rows)}")

    # Prefer all code-switch and numeric examples, then fill deterministically.
    priority = [item for item in rows if LATIN.search(item[1]["raw_transcription"]) or DIGITS.search(item[1]["raw_transcription"])]
    remainder = [item for item in rows if item not in priority]
    random.Random(args.seed).shuffle(remainder)
    selected = (priority + remainder)[: args.clips]
    selected.sort(key=lambda item: item[0])

    audio_dir = args.output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.jsonl"
    manifest_rows = []
    for source_index, row in selected:
        clip_id = f"fleurs-ne-val-{source_index:04d}"
        target = audio_dir / f"{clip_id}.wav"
        with tempfile.NamedTemporaryFile(suffix=".wav") as source:
            source.write(row["audio"]["bytes"])
            source.flush()
            subprocess.run(
                ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", source.name, "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(target)],
                check=True,
            )
        raw = row["raw_transcription"]
        has_latin, numbers = bool(LATIN.search(raw)), DIGITS.findall(raw)
        stratum = "code_switch" if has_latin else ("names_numbers" if numbers else "normal_nepali")
        tags = [stratum, "public_pilot", "read_speech", "fleurs_validation"]
        if has_latin and numbers:
            tags.append("names_numbers")
        duration = row["num_samples"] / 16000
        manifest_rows.append(
            {
                "id": clip_id,
                "audio_path": f"audio/{target.name}",
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                "duration_seconds": round(duration, 6),
                "speaker_id": None,
                "primary_stratum": stratum,
                "tags": tags,
                "reference": row["transcription"],
                "entities": [{"type": "number", "text": value} for value in numbers],
                "transcript_verified": False,
                "reference_status": "source_provided_fleurs_validation",
                "reviewer": "google-fleurs",
                "review_date": "source_release",
                "external_approved": True,
                "source": {
                    "dataset": "google/fleurs",
                    "config": "ne_np",
                    "split": "validation",
                    "source_row": source_index,
                    "source_id": row["id"],
                    "license": "CC-BY-4.0",
                },
            }
        )
    manifest_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in manifest_rows), encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "clips": len(manifest_rows), "strata": {s: sum(r["primary_stratum"] == s for r in manifest_rows) for s in sorted({r["primary_stratum"] for r in manifest_rows})}}, indent=2))


if __name__ == "__main__":
    main()
