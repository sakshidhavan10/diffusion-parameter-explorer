# Model Card — Diffusion Parameter Explorer (P33)

## Models used
| Role | Model | Notes |
|---|---|---|
| Text-to-image | `stabilityai/sd-turbo` (default) or `stable-diffusion-v1-5/stable-diffusion-v1-5` via `SD_MODEL_ID` | Distilled for 1–4 steps, CFG ≈ 0 |
| Scoring + NSFW | `openai/clip-vit-base-patch32` | CLIPScore = 100·max(cos, 0); zero-shot safety labels |
| Baseline | DCGAN (64×64), trained locally with `train_dcgan.py` | Unconditional |

## Intended use
Educational exploration of how steps, CFG, seed and negative prompts affect diffusion output. Not for production moderation, identity/likeness generation, or deceptive media.

## Safety
Layer 1: prompt term blocklist. Layer 2: CLIP zero-shot classifier over each image (threshold `NSFW_THRESHOLD`, default 0.5). Blocked outputs show a placeholder and are logged to `logs/nsfw_log.csv`. A CLIP zero-shot filter is coarse: expect false positives (e.g. classical statues) and false negatives.

## Licence
Verify the licence on each model page before reuse or commercial deployment (sd-turbo is under Stability AI terms, which have restricted commercial use). CLIP: MIT. Code in this repo: MIT.

## Limitations & bias
Trained on web-scale data with Western/English skew; stereotypes and under-representation can appear. CLIPScore rewards caption-like matches, not aesthetics or truthfulness. Seeds reproduce only on the same hardware/library versions.
