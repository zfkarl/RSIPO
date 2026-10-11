# Skill SFT

The SFT stage follows SEED: the base model collects its own trajectories on 500 training tasks (8 rollouts each), an
external LLM (GLM-5.2 in the paper, any OpenAI-compatible endpoint set in `.env`) writes an episode summary and an
episode skill for every trajectory with the analysis template of `seed/analysis.py`, and the base model is fine-tuned
to write these analyses. Every benchmark has two entry points:

```bash
bash scripts/sft/<alfworld|search>/prepare_data.sh   # rollouts, skills, sft_episode_skill_{train,val}.parquet
bash scripts/sft/<alfworld|search>/train_sft.sh      # 3 epochs, lr 5e-6, batch 8; exports a Hugging Face model
```

The task lists of the paper:

- ALFWorld: `python scripts/sft/alfworld/sample_tasks.py --output-dir OUT` writes the 500 games (84, 84, 83, 83, 83, 83
  per task type); then run `prepare_data.sh` with `OUTPUT_DIR=OUT RESUME=true`.
- Search-based tasks: run `prepare_data.sh` with `TRAIN_DATA=<sft_tasks.parquet from scripts/data/> NUM_TASKS=500`.

Common overrides: `MODEL_PATH` (base model), `OUTPUT_DIR`/`DATA_DIR` (data directory), `EXPORT_MODEL_DIR` (exported
model), `MAX_LENGTH` (raise it if a sample is longer than the default 8192 for ALFWorld or 12288 for search),
`NPROC_PER_NODE`, `TOTAL_EPOCHS`, `TRAIN_BATCH_SIZE`. For Qwen3 models, collect rollouts in non-thinking mode with
`POLICY_EXTRA_BODY_JSON='{"chat_template_kwargs": {"enable_thinking": false}}'` and append
`+data.apply_chat_template_kwargs.enable_thinking=False` to `train_sft.sh`.
