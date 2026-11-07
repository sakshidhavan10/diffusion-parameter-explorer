# 🌌 Diffusion Parameter Explorer (P33)
Streamlit dark-theme dashboard showing how **steps, CFG, seed and negative prompts** change a diffusion image — with CLIP scoring, NSFW filtering, a 4×3 parameter grid and a DCGAN comparison.

## Structure
```
app.py  train_dcgan.py  requirements.txt  MODEL_CARD.md  REPORT.md
.streamlit/config.toml   src/{generation,clip_score,nsfw,dcgan,utils,styles}.py
weights/ (DCGAN weights)  logs/ (NSFW log)
```
## Run locally
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python train_dcgan.py --data cifar10 --cls 3 --epochs 30   # optional but needed for the DCGAN tab
streamlit run app.py
```
First run downloads sd-turbo (~2 GB) and CLIP. A GPU is strongly recommended (CPU ≈ several s/step).

## Secrets
Optional `HF_TOKEN` only if a model needs auth: `export HF_TOKEN=...` or `.streamlit/secrets.toml` (git-ignored). No keys live in code.

## Deploy — Hugging Face Spaces (recommended; GPU for speed)
1. Create a Space → SDK **Streamlit** → hardware GPU (T4) if available.
2. `git remote add space https://huggingface.co/spaces/<user>/diffusion-explorer && git push space main`
3. Add `HF_TOKEN` under Settings → Secrets if needed. Commit `weights/dcgan_netG.pth` (small, ~14 MB) or set `DCGAN_WEIGHTS`.
4. Add this YAML header at the top of README on the Space:
`---\ntitle: Diffusion Parameter Explorer\nsdk: streamlit\napp_file: app.py\n---`

## Deploy — Streamlit Community Cloud
Push to GitHub → share.streamlit.io → New app → select repo, `app.py`. Free tier is CPU-only (1 GB RAM), so sd-turbo may be too heavy; prefer HF Spaces.
