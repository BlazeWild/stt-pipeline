# Lightning AI setup (local models only)

This project is prepared to run **local Faster-Whisper models only** on Lightning AI. Gemini and other hosted transcription services are out of scope for this setup.

## 1. Create the Studio

Create a Lightning Studio with a GPU. Use at least **12 GB VRAM** for the comparison; **16 GB is preferable** for the Turbo model and for later LoRA feasibility work. Use a fresh terminal at the repository root.

## 2. Create the environment

```bash
sudo apt-get update && sudo apt-get install -y ffmpeg
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Confirm that the GPU is exposed:

```bash
nvidia-smi
python -c "import ctranslate2; print('CTranslate2:', ctranslate2.__version__)"
```

The runner explicitly loads the CUDA libraries bundled with the pip NVIDIA packages, so no manual `LD_LIBRARY_PATH` change is normally required.

## 3. Obtain the public-pilot source and build its audio

Download the `ne_np` validation Parquet from the Google FLEURS dataset into:

```text
data/public_sources/fleurs_ne_np/validation.parquet
```

Then build and validate the deterministic 150-clip pilot:

```bash
python scripts/build_public_pilot.py
python scripts/validate_manifest.py data/public_pilot/manifest.jsonl --allow-source-transcripts
```

The generated audio is ignored by Git. Do not upload private Nagarik audio or manifests to a public repository.

## 4. Run the best completed local model

The current local leader is `Dragneel/whisper-medium-nepali-openslr-ct2`. Faster-Whisper downloads it on first use:

```bash
python scripts/run_faster_whisper.py \
  --manifest data/public_pilot/manifest.jsonl \
  --model Dragneel/whisper-medium-nepali-openslr-ct2 \
  --device cuda \
  --compute-type int8_float16 \
  --output results/public-pilot-whisper-medium-nepali-int8/predictions.jsonl

python scripts/score.py \
  --manifest data/public_pilot/manifest.jsonl \
  --predictions results/public-pilot-whisper-medium-nepali-int8/predictions.jsonl \
  --output results/public-pilot-whisper-medium-nepali-int8/metrics.json
```

## 5. Run the leading larger-GPU candidate

On the larger Lightning GPU, evaluate the Nepali fine-tuned Turbo candidate on the same frozen pilot:

```bash
python scripts/run_faster_whisper.py \
  --manifest data/public_pilot/manifest.jsonl \
  --model kiranpantha/whisper-large-v3-turbo-nepali-ctranslate2 \
  --device cuda \
  --compute-type int8_float16 \
  --output results/public-pilot-whisper-large-v3-turbo-nepali-int8/predictions.jsonl

python scripts/score.py \
  --manifest data/public_pilot/manifest.jsonl \
  --predictions results/public-pilot-whisper-large-v3-turbo-nepali-int8/predictions.jsonl \
  --output results/public-pilot-whisper-large-v3-turbo-nepali-int8/metrics.json

python scripts/compare.py --results results --output reports/comparison.md
```

Keep the manifest fixed across every run. Record GPU type, CUDA/runtime versions, command, elapsed time, and peak VRAM with the results.
