# Next steps

## Current best result

The best **completed local** model is **Dragneel Whisper Medium Nepali, Faster-Whisper INT8**:

| Measure | Result |
|---|---:|
| Pilot coverage | 148 / 150 clips |
| WER | 34.1% |
| CER | 13.4% |
| Number exact accuracy | 67.5% (52 / 77) |
| Real-time factor | 0.34 |
| Peak VRAM | 3,763 MiB |

This is only a public FLEURS engineering pilot, not a Nagarik acceptance result. Two long clips failed on the 4 GB GPU.

## Ordered work plan

1. In Lightning AI, rerun Whisper Medium on a larger GPU and confirm all 150 clips complete.
2. Run `kiranpantha/whisper-large-v3-turbo-nepali-ctranslate2` on the exact same frozen public-pilot manifest. It could not fit on the 4 GB laptop and is the next leading local candidate to test.
3. Run IndicConformer only after accepting its Hugging Face gated terms and using an authorized access token. Do not place tokens in Git.
4. Generate `reports/comparison.md`, then manually audit the two best local candidates for missing words, locations, English mixing, numbers, and silence hallucinations.
5. Build the separate final Nagarik test set (100–200 clips) with reviewed references and the currently missing telephone/noise and silence strata. Keep it private and immutable; never train on it.
6. Before any extended fine-tuning, perform a 50–100-step LoRA/QLoRA feasibility run on a 16 GB GPU and record peak VRAM, step time, loss, and held-out WER/CER.

## Decision rule

Do not select a production model from the public pilot alone. Choose only after all viable local models have run on the same frozen, reviewed Nagarik test set and operational constraints have been measured.
