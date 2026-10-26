#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def pct(value):
    return "—" if value is None else f"{100 * value:.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    metrics_files = sorted(args.results.glob("*/metrics.json"))
    if not metrics_files:
        raise SystemExit(f"no */metrics.json files found under {args.results}")
    lines = [
        "# Nagarik Nepali STT comparison",
        "",
        "All values below are measured on the same frozen Nagarik manifest.",
        "",
        "| Model | WER | CER | Names | Locations | Numbers | Missing words | Silence hallucination | Time (s) | RTF | Peak VRAM (MB) | API cost (USD) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for path in metrics_files:
        value = json.loads(path.read_text(encoding="utf-8"))["overall"]
        entity = value.get("entities", {})
        get_entity = lambda kind: pct(entity.get(kind, {}).get("exact_accuracy"))
        lines.append(
            f"| {path.parent.name} | {pct(value['wer'])} | {pct(value['cer'])} | "
            f"{get_entity('name')} | {get_entity('location')} | {get_entity('number')} | "
            f"{pct(value.get('missing_word_rate'))} | {pct(value.get('silence_hallucination_rate'))} | "
            f"{value.get('latency_seconds', 0):.2f} | {value.get('real_time_factor') or 0:.3f} | "
            f"{value.get('peak_vram_mb', 0):.0f} | {value.get('api_cost_usd', 0):.4f} |"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()

