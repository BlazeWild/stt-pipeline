#!/usr/bin/env python3
"""Validate Gemini word-timestamp output on a small approved audio batch."""

from __future__ import annotations

import argparse
import json
import os
import wave
from pathlib import Path

from google import genai


def load_env(path: Path) -> None:
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--clips", type=int, default=2)
    parser.add_argument("--gap-seconds", type=float, default=1.0)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    args = parser.parse_args()

    load_env(args.env_file)
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY is missing")
    refs = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()][: args.clips]
    if not refs or any(row.get("external_approved") is not True for row in refs):
        raise SystemExit("all selected clips must be externally approved")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    batch_path = args.output_dir / "batch.wav"
    boundaries = []
    with wave.open(str(args.manifest.parent / refs[0]["audio_path"]), "rb") as first:
        params = first.getparams()
        audio_format = (params.nchannels, params.sampwidth, params.framerate, params.comptype)
    position = 0
    with wave.open(str(batch_path), "wb") as output:
        output.setparams(params)
        for index, ref in enumerate(refs):
            with wave.open(str(args.manifest.parent / ref["audio_path"]), "rb") as source:
                source_params = source.getparams()
                if (source_params.nchannels, source_params.sampwidth, source_params.framerate, source_params.comptype) != audio_format:
                    raise SystemExit(f"incompatible WAV parameters for {ref['id']}")
                frames = source.readframes(source.getnframes())
                start = position / params.framerate
                output.writeframes(frames)
                position += source.getnframes()
                boundaries.append({"id": ref["id"], "start_seconds": start, "end_seconds": position / params.framerate})
            if index < len(refs) - 1:
                silence = b"\0" * int(params.framerate * args.gap_seconds) * params.nchannels * params.sampwidth
                output.writeframes(silence)
                position += int(params.framerate * args.gap_seconds)
    (args.output_dir / "boundaries.json").write_text(json.dumps(boundaries, indent=2) + "\n", encoding="utf-8")

    client = genai.Client(api_key=api_key)
    uploaded = client.files.upload(file=str(batch_path))
    try:
        interaction = client.interactions.create(
            model="gemini-3.5-transcribe",
            input=[{"type": "audio", "uri": uploaded.uri, "mime_type": uploaded.mime_type}],
            generation_config={
                "transcription_config": {
                    "language_codes": ["ne-NP"],
                    "mode": {"type": "verbatim", "timestamp_granularities": ["word"]},
                }
            },
        )
        payload = interaction.model_dump(mode="json") if hasattr(interaction, "model_dump") else {"output_text": interaction.output_text}
        (args.output_dir / "response.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(interaction.output_text or "")
    finally:
        client.files.delete(name=uploaded.name)


if __name__ == "__main__":
    main()
