#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import wave
from collections import Counter
from pathlib import Path

ALLOWED = {"normal_nepali", "code_switch", "names_numbers", "noisy_telephone", "silence"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--allow-template-hashes", action="store_true")
    parser.add_argument("--allow-source-transcripts", action="store_true", help="pilot only: accept published references that were not manually rechecked")
    args = parser.parse_args()
    errors, ids, counts = [], set(), Counter()
    base = args.manifest.parent
    rows = []
    for line_no, line in enumerate(args.manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"line {line_no}: invalid JSON: {exc}")
            continue
        rows.append(row)
        clip_id = row.get("id")
        if not clip_id or clip_id in ids:
            errors.append(f"line {line_no}: missing/duplicate id {clip_id!r}")
        ids.add(clip_id)
        stratum = row.get("primary_stratum")
        counts[stratum] += 1
        if stratum not in ALLOWED:
            errors.append(f"{clip_id}: invalid primary_stratum {stratum!r}")
        source_reference = str(row.get("reference_status", "")).startswith("source_provided_")
        if row.get("transcript_verified") is not True and not (args.allow_source_transcripts and source_reference):
            errors.append(f"{clip_id}: transcript_verified must be true")
        if not row.get("reviewer") or not row.get("review_date"):
            errors.append(f"{clip_id}: reviewer and review_date are required")
        if stratum == "silence" and row.get("reference", "").strip():
            errors.append(f"{clip_id}: silence reference must be empty")
        path = base / row.get("audio_path", "")
        if not path.is_file():
            errors.append(f"{clip_id}: audio missing: {path}")
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if row.get("sha256") != digest and not (args.allow_template_hashes and row.get("sha256") == "REPLACE"):
            errors.append(f"{clip_id}: sha256 mismatch (actual {digest})")
        if path.suffix.lower() == ".wav":
            try:
                with wave.open(str(path), "rb") as audio:
                    measured = audio.getnframes() / audio.getframerate()
                if abs(float(row.get("duration_seconds", -1)) - measured) > 0.05:
                    errors.append(f"{clip_id}: duration mismatch (manifest={row.get('duration_seconds')}, actual={measured:.3f})")
            except (wave.Error, EOFError) as exc:
                errors.append(f"{clip_id}: unreadable WAV: {exc}")
    if not 100 <= len(rows) <= 200:
        errors.append(f"expected 100–200 clips, found {len(rows)}")
    print(json.dumps({"clips": len(rows), "strata": counts, "errors": errors}, ensure_ascii=False, indent=2, default=dict))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
