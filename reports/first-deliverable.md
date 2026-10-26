# First deliverable: Nagarik Nepali STT benchmark

**Status (2026-10-25): preliminary public pilot complete for Gemini (26 clips) and Whisper large-v3 INT8 (150 clips). This is not yet a Nagarik test-set result.**

The frozen public pilot contains 150 approved FLEURS Nepali validation clips (28.18 minutes; CC-BY-4.0). They are source transcripts, not yet manually verified Nagarik references. It has 83 normal-Nepali, 35 code-switch and 32 names/numbers clips. It has no telephone/noise/silence stratum and no reliable speaker IDs, so it is an engineering pilot only; do not present it as the final Nagarik benchmark.

## Comparison table

| System | Evaluated clips | WER | CER | Numbers exact | RTF | Peak VRAM | Cost | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Gemini transcription | 26/150 successful | 23.0% | 8.5% | 9/9 (100%) | 1.56 | N/A | $0 free tier | Partial reference only; 1 request timed out and the free tier has 25 RPD |
| Whisper large-v3, faster-whisper INT8-FP16 | 150/150 | 92.0% | 27.9% | 49/80 (61.3%) | 0.30 | **3,374 MiB** | $0 | Completed locally; matched 26-clip Gemini subset: 94.9% WER / 25.2% CER |
| Nepali fine-tuned Whisper Medium, faster-whisper INT8 | 148/150 successful | **34.1%** | **13.4%** | 52/77 (67.5%) | 0.34 | **3,763 MiB** | $0 | Best completed local result; 2 long-clip GPU failures on 4 GB |
| Nepali fine-tuned Whisper large-v3 Turbo, faster-whisper INT8 | 0/150 | — | — | — | — | **does not fit** | $0 | CUDA OOM during model load on 4 GB GPU |
| Google Cloud Chirp 3 (`ne-NP`) | 0/150 | — | — | — | — | N/A | Not run | Credentials/project setup still required |
| IndicConformer-600M | 0/150 | — | — | — | — | Not measured | $0 | Hugging Face gated terms must be accepted; no token/access in workspace |
| Shruti Nepali ASR v6b (Nemotron-0.6B adaptation) | 0/150 | — | — | — | — | Not measured | $0 | Checkpoint downloaded, but its required `nemotron_asr` runtime/source is not released; cannot reproduce inference |
| Qwen3-ASR-0.6B | 0/150 | — | — | — | — | Not measured | $0 | Exploratory only: Nepali is absent from its documented ASR language list; needs a larger-memory Transformers runtime |

WER uses whitespace-delimited normalized tokens; CER is often more diagnostic for this pilot because Whisper makes many near-spelling and word-boundary errors in Devanagari. Both are reported, with raw CER retained in result JSON. Gemini's score is calculated only on successful requests; the failure is recorded in coverage rather than treated as an empty transcript.

### Error signals observed

| System | Evidence from pilot |
|---|---|
| Gemini | Code-switch WER is 31.8% (7 clips), versus 12.0% on names/numbers (4 clips). It had 4.2% missing-word rate and 0.7% insertion rate. |
| Whisper large-v3 INT8 | Code-switch CER is 35.4%, normal Nepali CER 25.7%, names/numbers CER 24.4%. It has a high 21.4% insertion rate, so manual review must inspect invented/extra words as well as omissions. |

## Recommendation today

Use **Gemini transcription as the quality reference target**, but do not yet call it the measured SOTA for Nagarik. On the matched 26-clip public subset Gemini is 23.0% WER / 8.5% CER against Whisper's 94.9% / 25.2%; the free tier's 25 RPD limit prevents a 150-clip Gemini conclusion today.

For local work, **Whisper large-v3 INT8-FP16 is the only completed 4 GB candidate and is the current operational baseline**. It is not recommended for deployment based on this public pilot—the accuracy gap to Gemini is substantial. The leading fine-tuning candidate is the Nepali-fine-tuned Whisper Turbo checkpoint, but it must first be evaluated on a GPU with more than 4 GB VRAM. Do not select local finalists until Chirp, IndicConformer, and the fine-tuned checkpoint have either completed the same frozen clips or have documented access/fit exclusions.

## Measured GPU requirements

| Check | Result |
|---|---|
| Laptop GPU | NVIDIA GeForce RTX 3050 Laptop GPU, 4,096 MiB |
| Whisper large-v3 INT8-FP16 inference | Fits; 3,374 MiB peak VRAM on full 150-clip run |
| Nepali fine-tuned large-v3 Turbo INT8-FP16 | CUDA out-of-memory while loading (before first clip) |
| Nepali fine-tuned large-v3 Turbo INT8 | CUDA out-of-memory while loading (before first clip) |
| Nepali fine-tuned large-v3 Turbo CPU INT8 | Did not load on this laptop: 7.1 GiB RAM total with only ~0.94 GiB available and ~0.83 GiB swap free at test time |
| Training feasibility | Not run; 4 GB is unsuitable for full fine-tuning and has no demonstrated LoRA headroom |

Request a **12 GB GPU minimum for inference comparison plus a short LoRA/QLoRA feasibility run; 16 GB is the safer request for reproducible training experiments**. This is a planning recommendation, not a fabricated training-memory measurement. Before a longer rental, run 50–100 optimizer steps and capture peak VRAM, step time, loss, and held-out development CER/WER.

## What remains

1. Build the final 100–200 clip Nagarik test set, manually check every reference, and include the currently missing noisy/telephone and silence strata. Keep it untouched by training.
2. Run all open models on all 150 public-pilot clips now that the Whisper baseline is complete. IndicConformer needs Hugging Face acceptance plus an access token.
3. For Gemini, either wait for the daily quota reset and send a few timestamped audio batches (rather than one API request per clip), or enable billing. The measured 28.18 minutes would be roughly $0.14 at the published $0.005/min Transcribe paid rate; confirm current billing before spending.
4. Configure and run Chirp 3 with approved audio and Google Cloud credentials. Record the actual region, model revision, billable minutes, and cost.
5. On the larger GPU, run the fine-tuned Whisper and then perform the required manual audit for the best two local candidates: missing words, locations, English mixing, numbers, and silence hallucinations.

## Evidence and provenance

- Public-pilot manifest: `data/public_pilot/manifest.jsonl`; validation command: `python3 scripts/validate_manifest.py data/public_pilot/manifest.jsonl --allow-source-transcripts`.
- Full local metrics: `results/public-pilot-whisper-large-v3-int8/metrics.json`.
- Exact matched subset: `results/public-pilot-whisper-large-v3-int8/metrics-gemini-matched-26.json`.
- Gemini partial metrics: `results/public-pilot-gemini-transcribe/metrics-partial.json`.
- Google lists `chirp_3` with Nepali (`ne-NP`) support: <https://docs.cloud.google.com/speech-to-text/docs/models/chirp-3>
- Gemini transcription documentation and pricing: <https://ai.google.dev/gemini-api/docs/transcribe>, <https://ai.google.dev/gemini-api/docs/pricing>
- IndicConformer’s model card lists Nepali (`ne`), CTC/RNNT decoding, and its gated access condition: <https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual>
