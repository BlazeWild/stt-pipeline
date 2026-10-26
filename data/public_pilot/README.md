# Public Nepali pilot set

This directory is generated from the `ne_np` validation split of Google FLEURS, licensed CC BY 4.0. Run:

```bash
.venv/bin/python scripts/build_public_pilot.py
python3 scripts/validate_manifest.py data/public_pilot/manifest.jsonl --allow-source-transcripts
```

This is a **pipeline/model smoke benchmark**, not the Nagarik acceptance benchmark. FLEURS contains read speech, lacks Nagarik's newsroom/domain distribution, does not supply useful speaker IDs in the downloaded Parquet conversion, and does not cover telephone/noise or silence hallucination testing. Its published transcripts have not been manually rechecked in this project. Do not use public-pilot scores as the final model selection evidence.

Source: https://huggingface.co/datasets/google/fleurs

