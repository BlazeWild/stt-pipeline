#!/usr/bin/env python3
"""Run Gemini's dedicated transcription model on externally approved clips."""

from __future__ import annotations

import argparse
import json
import os
import signal
import time
from contextlib import contextmanager
from pathlib import Path

from google import genai


@contextmanager
def wall_clock_timeout(seconds: int):
    """Hard Unix timeout, including time hidden inside SDK retry loops."""
    def handle_timeout(_signum, _frame):
        raise TimeoutError(f"operation exceeded {seconds} seconds")

    previous = signal.signal(signal.SIGALRM, handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def load_env(path: Path) -> None:
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key.strip(), value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--model", default="gemini-3.5-transcribe")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--free-tier", action="store_true", help="record zero estimated API cost; does not verify billing configuration")
    parser.add_argument("--resume", action="store_true", help="retain successful rows already present in the output")
    parser.add_argument("--timeout-seconds", type=int, default=45, help="fail a request instead of waiting through long SDK retries")
    args = parser.parse_args()

    load_env(args.env_file)
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY is missing")
    refs = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        refs = refs[: args.limit]
    unsafe = [row["id"] for row in refs if row.get("external_approved") is not True]
    if unsafe:
        raise SystemExit(f"refusing to upload clips without external_approved=true: {unsafe[:5]}")

    client = genai.Client(
        api_key=api_key,
        http_options={
            "timeout": args.timeout_seconds * 1000,
            "retry_options": {"attempts": 1},
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions = []
    if args.resume and args.output.exists():
        predictions = [json.loads(line) for line in args.output.read_text(encoding="utf-8").splitlines() if line.strip()]
    completed_ids = {row["id"] for row in predictions if row.get("status") == "ok"}
    pending = [row for row in refs if row["id"] not in completed_ids]
    for offset, ref in enumerate(pending, 1):
        index = len(completed_ids) + offset
        path = args.manifest.parent / ref["audio_path"]
        started = time.monotonic()
        uploaded = None
        try:
            with wall_clock_timeout(args.timeout_seconds):
                uploaded = client.files.upload(file=str(path))
                interaction = client.interactions.create(
                    model=args.model,
                    input=[{"type": "audio", "uri": uploaded.uri, "mime_type": uploaded.mime_type}],
                    generation_config={
                        "transcription_config": {
                            "language_codes": ["ne-NP"],
                            "mode": {"type": "verbatim"},
                        }
                    },
                )
            row = {"id": ref["id"], "text": interaction.output_text or "", "status": "ok"}
            usage = getattr(interaction, "usage", None) or getattr(interaction, "usage_metadata", None)
            if usage is not None:
                row["usage"] = usage.model_dump(mode="json") if hasattr(usage, "model_dump") else str(usage)
        except Exception as exc:
            row = {"id": ref["id"], "text": "", "status": "error", "error": f"{type(exc).__name__}: {exc}"}
        finally:
            if uploaded is not None and getattr(uploaded, "name", None):
                try:
                    with wall_clock_timeout(min(10, args.timeout_seconds)):
                        client.files.delete(name=uploaded.name)
                except Exception as exc:
                    row["cleanup_error"] = f"{type(exc).__name__}: {exc}"
        row.update(
            {
                "latency_seconds": time.monotonic() - started,
                "peak_vram_mb": None,
                "api_cost_usd": 0.0 if args.free_tier else None,
            }
        )
        # Replace a previous failed attempt for this ID; never create duplicate IDs.
        predictions = [item for item in predictions if item["id"] != ref["id"]]
        predictions.append(row)
        args.output.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in predictions), encoding="utf-8")
        print(f"[{index}/{len(refs)}] {ref['id']} {row['status']} {row['latency_seconds']:.2f}s", flush=True)
        if row["status"] == "error":
            print(row["error"], flush=True)
            if "429" in row["error"]:
                print("Quota/rate limit reached. Stop now and resume after the quota resets.", flush=True)
            break

    metadata = {
        "model": args.model,
        "mode": "verbatim",
        "language_codes": ["ne-NP"],
        "clips_requested": len(refs),
        "clips_completed": sum(row.get("status") == "ok" for row in predictions),
        "billing_assumption": "user_indicated_free_tier" if args.free_tier else "unknown",
        "timeout_seconds": args.timeout_seconds,
        "sdk_retry_attempts": 1,
    }
    args.output.with_name("run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
