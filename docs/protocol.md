# Benchmark protocol

## 1. Freeze one test set

Target 150 clips (acceptable range 100–200) from distinct source segments and as many speakers as practical. Do not put overlapping audio, repeated sentences, or the same speaker session in both training and test data.

Suggested quotas:

| Stratum | Target | What to include |
|---|---:|---|
| Normal Nepali | 50 | conversational and read speech, varied speakers |
| Nepali–English mixed | 30 | natural code-switching, product/organization terms |
| Names and numbers | 30 | people, wards, districts, dates, money, phone-like digit strings |
| Noisy/telephone | 30 | 8 kHz calls, traffic, rooms, compression |
| Silence/non-speech | 10 | leading/trailing silence and representative background noise |

A clip may carry multiple tags, but assign exactly one `primary_stratum` for balanced reporting. Aim for 3–30 seconds per speech clip. Keep the ten silence clips because invented speech during silence is a required failure measure.

## 2. Reference transcript policy

- Transcribe what was spoken, not a grammatical correction.
- Use Devanagari for Nepali and Latin script for clearly spoken English.
- Store a single canonical form for numbers in `reference`; annotate the spoken semantic item in `entities`.
- Mark unintelligible audio explicitly during review and exclude it before freezing the set.
- Require `transcript_verified: true`, reviewer initials, and a review date.
- Compute and store the SHA-256 digest so the frozen test set can be audited.

The scorer reports a documented, shared normalization: Unicode NFC, lowercase Latin, Devanagari-digit conversion to ASCII, punctuation removal, and whitespace collapse. It also reports raw CER so normalization does not hide spelling/script differences.

## 3. Identical inference conditions

- Use exactly the manifest IDs for every model.
- Do not give any model the reference text, annotated entities, or model-specific phrase hints.
- Fix decoding parameters before looking at model errors.
- Use `ne-NP`/Nepali transcription mode where supported.
- Record model repository/revision, runtime version, quantization, device, and command.
- Warm up local models once, reset peak-memory counters, then time every clip with a monotonic clock. Report total wall time and real-time factor (total inference seconds / total audio seconds).
- Peak VRAM means the maximum allocated/reserved GPU memory observed for the complete process. If inference is CPU-only, report zero GPU VRAM and record peak RAM separately. If loading fails, preserve the error and report `does not fit`; do not substitute another quantization silently.
- Hosted cost must come from billable units returned by the service or the provider's current price multiplied by measured audio/tokens. Preserve the price date and calculation.

Only rows with `external_approved: true` may be sent to Chirp or Gemini. Approval covers both audio and transcript handling under Nagarik policy. Local models may evaluate all frozen rows.

## 4. Metrics

Primary metrics are corpus WER and CER. Also report:

- normalized and raw CER;
- WER/CER by `primary_stratum`;
- exact entity accuracy and entity character similarity for `name`, `location`, and `number` annotations;
- missing-word rate (deletions/reference words);
- insertion rate (insertions/reference words);
- silence hallucination rate: non-empty normalized predictions on silence clips;
- total time, real-time factor, peak VRAM, and hosted API cost.

Review every error from the best two local candidates and label at least: missing word, wrong location, English/code-switch error, number error, hallucination on silence, and other. The quantitative table and this qualitative audit jointly determine selection.

## 5. Decision gate and fine-tuning

Select two local candidates only after all five systems have predictions on the frozen set. A candidate should not win solely on mean WER: reject a model with materially worse named-location recall, code-switch behavior, silence hallucination, or operational fit unless Nagarik explicitly accepts that trade-off.

Keep the benchmark manifest and audio immutable. Build a separate training collection targeted to the observed errors. Before requesting a larger GPU, run a small feasibility experiment (for example, 30–60 minutes of training audio and 50–100 optimizer steps) and measure:

- maximum allocated/reserved VRAM;
- step time and examples/hour;
- training loss and a tiny held-out development WER/CER;
- exact precision, quantization, LoRA rank, batch size, gradient accumulation, checkpointing, and clip duration.

The 4 GB laptop is for data preparation and inference experiments. If a candidate cannot load, record that result. Use a short larger-GPU run to establish accuracy and training memory; only then size a longer rental. Qwen/Nemotron adaptation is outside phase one.

