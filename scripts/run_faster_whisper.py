#!/usr/bin/env python3
"""Run a faster-whisper checkpoint on a manifest with timing and VRAM polling."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from faster_whisper import WhisperModel


def configure_cuda_libraries() -> None:
    """Expose pip-installed CUDA 12 libraries to CTranslate2."""
    site_packages = Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    cuda_libs = [
        site_packages / "nvidia" / "cublas" / "lib",
        site_packages / "nvidia" / "cudnn" / "lib",
    ]
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = ":".join([*(str(path) for path in cuda_libs if path.is_dir()), existing])
    # glibc only reads LD_LIBRARY_PATH at process startup.  Load the pip wheels
    # explicitly so CTranslate2 can resolve their SONAMEs in this process.
    for filename in ("libcublasLt.so.12", "libcublas.so.12", "libcudnn.so.9"):
        for directory in cuda_libs:
            candidate = directory / filename
            if candidate.is_file():
                ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
                break


class VramMonitor:
    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.peak_mb = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            try:
                value = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    text=True,
                    stderr=subprocess.DEVNULL,
                ).splitlines()[0]
                self.peak_mb = max(self.peak_mb, float(value.strip()))
            except Exception:
                pass
            self._stop.wait(self.interval)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._thread.join()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default="large-v3")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--compute-type", default="int8_float16")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.device == "cuda":
        configure_cuda_libraries()

    refs = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        refs = refs[: args.limit]
    predictions = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with VramMonitor() as monitor:
        load_started = time.monotonic()
        model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)
        load_seconds = time.monotonic() - load_started
        for index, ref in enumerate(refs, 1):
            audio = args.manifest.parent / ref["audio_path"]
            started = time.monotonic()
            try:
                segments, info = model.transcribe(
                    str(audio),
                    language="ne",
                    task="transcribe",
                    beam_size=args.beam_size,
                    vad_filter=False,
                    condition_on_previous_text=False,
                )
                text = " ".join(segment.text.strip() for segment in segments).strip()
                row = {"id": ref["id"], "text": text, "status": "ok"}
            except Exception as exc:
                row = {"id": ref["id"], "text": "", "status": "error", "error": f"{type(exc).__name__}: {exc}"}
            row.update({"latency_seconds": time.monotonic() - started, "peak_vram_mb": monitor.peak_mb, "api_cost_usd": 0.0})
            predictions.append(row)
            args.output.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in predictions), encoding="utf-8")
            print(f"[{index}/{len(refs)}] {ref['id']} {row['status']} {row['latency_seconds']:.2f}s", flush=True)
    metadata = {
        "model": args.model,
        "device": args.device,
        "compute_type": args.compute_type,
        "beam_size": args.beam_size,
        "load_seconds": load_seconds,
        "peak_vram_mb": monitor.peak_mb,
        "clips": len(refs),
    }
    args.output.with_name("run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
