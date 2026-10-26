# Nagarik Nepali STT benchmark

This repository is a reproducible evaluation harness for comparing speech-to-text systems on the **same private Nagarik test clips**. The GitHub version is deliberately set up for **local-model execution in Lightning AI**; no Gemini configuration or API key is needed.

1. Google Cloud Speech-to-Text V2 `chirp_3` (`ne-NP`)
2. Gemini transcription
3. `openai/whisper-large-v3` with INT8 inference
4. `Dragneel/whisper-large-v3-nepali-openslr`
5. `ai4bharat/indic-conformer-600m-multilingual`

The final Nagarik test set has not yet been supplied. A 150-clip approved public FLEURS Nepali pilot is available under `data/public_pilot/`; its preliminary measurements and limitations are documented in `reports/first-deliverable.md`. Do not populate the final comparison table with model-card scores: those scores use different datasets and are not comparable.

## Quick start

1. Copy `data/manifest.template.jsonl` to `data/manifest.jsonl` and add 100–200 approved clips.
2. Have a Nepali-speaking reviewer check every reference transcript and entity annotation.
3. Validate the test set:

   ```bash
   python3 scripts/validate_manifest.py data/manifest.jsonl
   ```

4. Run each model, writing one prediction JSONL in the schema below.
5. Score a run:

   ```bash
   python3 scripts/score.py \
     --manifest data/manifest.jsonl \
     --predictions results/whisper-large-v3-int8/predictions.jsonl \
     --output results/whisper-large-v3-int8/metrics.json
   ```

6. Generate the comparison table after all runs:

   ```bash
   python3 scripts/compare.py --results results --output reports/comparison.md
   ```

See [docs/protocol.md](docs/protocol.md) for sampling, transcription, privacy, timing, and selection rules. See [reports/first-deliverable.md](reports/first-deliverable.md) for the current evidence and the exact blockers to a defensible recommendation.

## Current leader and Lightning AI

The best completed local run is the Nepali fine-tuned Whisper Medium INT8 model: **34.1% WER / 13.4% CER**, with 148/150 public-pilot clips completed. This is preliminary public-pilot evidence, not the final Nagarik result.

For an end-to-end Lightning AI setup and the commands to evaluate local models, see [docs/lightning-ai-setup.md](docs/lightning-ai-setup.md). The ordered evaluation plan is in [docs/next-steps.md](docs/next-steps.md).

## Prediction schema

One JSON object per line, with the exact same `id` values as the manifest:

```json
{"id":"nkr-0001","text":"...","latency_seconds":1.23,"peak_vram_mb":null,"api_cost_usd":0.0004,"status":"ok"}
```

For failed items use `status: "error"` and an `error` field. Never silently omit a clip. Hosted-model runners must reject rows whose `external_approved` field is not `true`.
