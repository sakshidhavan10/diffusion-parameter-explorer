# Diffusion Parameter Explorer (P33)
**Project report** · Student: _<name, roll no.>_ · Course: _<course>_ · Date: _<date>_
Live app: _<HF Spaces URL>_ · Repo: _<GitHub URL>_

---
## 1. Introduction
**Background.** Text-to-image diffusion models generate images by iteratively removing Gaussian noise under text guidance. Users control the result through a few parameters whose effects are rarely visible side by side.
**Problem statement.** Build an interactive tool showing how *steps, guidance scale (CFG), seed and negative prompts* change a diffusion image, with quantitative scoring and safety checks.
**Objectives.**
| ID | Objective | Where in the app |
|---|---|---|
| U1 | Explain noise as a probability distribution; prompt variants | Prompt Lab |
| U2 | DCGAN samples to compare with diffusion | DCGAN vs Diffusion |
| U3 | Run sd-turbo; score with CLIP | Explorer, Grid Lab |
| U4 | Grid generator with diffusers (PyTorch) | Parameter Grid Lab |
| U5 | NSFW filter; licence and representation | Safety & Evaluation, MODEL_CARD |

**Core feature.** One-click 4 (steps) × 3 (CFG) grid at a fixed seed with CLIP score per cell.

## 2. Application description
Dark glassmorphism dashboard with six tabs. _[Screenshot 1: Explorer] [Screenshot 2: Grid Lab] [Screenshot 3: DCGAN vs Diffusion] [Screenshot 4: Prompt Lab] [Screenshot 5: Safety tab]_
Features: parameter cards, seed lock/random, time estimate, per-image badges ("Grounded on", "Safety status"), PNG/collage/zip download, seed-history gallery, mobile-responsive CSS.

## 3. System design
**Architecture.** `app.py` (UI) → `generation.py` (diffusers pipeline) → `nsfw.py` (prompt + image filter) → `clip_score.py` → UI badges/history. `dcgan.py` is independent. Models are cached with `st.cache_resource`.
| Layer | Technology |
|---|---|
| UI | Streamlit + custom CSS |
| Diffusion | diffusers, PyTorch, `stabilityai/sd-turbo` |
| Scoring/safety | transformers, CLIP ViT-B/32 |
| Baseline | PyTorch DCGAN (64×64) |
| Deployment | Hugging Face Spaces / Streamlit Cloud |

**Data.** No dataset for diffusion (pre-trained). DCGAN trained on a CIFAR-10 class or CelebA subset _(state which)_.
**Request flow.** Inputs → prompt filter → pipeline(prompt, steps, cfg, seed, negative) → image filter → CLIPScore → badges, history, download.

## 4. Layer I — Foundations (U1)
Forward process (noise as a Gaussian):
`q(x_t | x_0) = N(√ᾱ_t x_0, (1−ᾱ_t) I)`, `x_t = √ᾱ_t x_0 + √(1−ᾱ_t) ε`, with ᾱ_t = ∏(1−β_s), linear β from 1e-4 to 0.02, T=1000.
Training loss: `L = E‖ε − ε_θ(x_t, t, c)‖²`. At t→T, x_t ≈ N(0, I), so sampling starts from pure noise; the **seed** fixes that sample.
Prompt variants tested: _(base / +style / +photographic detail — fill in Appendix A with CLIP scores)_.

## 5. Layer II — Generative model comparison (U2)
GAN objective: `min_G max_D E[log D(x)] + E[log(1 − D(G(z)))]`. DCGAN: transposed-conv generator, strided-conv discriminator, BatchNorm, Adam(2e-4, β1=0.5).
| | DCGAN | Diffusion |
|---|---|---|
| Sampling | 1 pass | 1–50 steps |
| Conditioning | none | text + CFG |
| Training stability | fragile | stable |
| Failure | mode collapse | slow, prompt errors |
Observed results: _<describe your samples; training epochs, loss curves>_

## 6. Layer III — Diffusion control (U3, U4)
Sampler steps trade quality for latency; sd-turbo is distilled for 1–4 steps. CFG: `ε̂ = ε_u + w(ε_c − ε_u)`; negative prompt replaces ε_u, so it only acts when w > 1. Same seed ⇒ same initial noise ⇒ differences in the grid are attributable to steps/CFG only.
## 7. Layer IV — Evaluation metric
CLIPScore = 100·max(cos(E_img, E_txt), 0) with CLIP ViT-B/32. Limitation: measures prompt-image agreement, not realism or aesthetics.
## 8. Layer V — Responsible AI (U5)
Two-layer filter (blocklist + CLIP zero-shot `{safe, explicit, violent}`; threshold 0.5), blocked outputs replaced and logged. Licence: _<verify sd-turbo licence at submission>_; CLIP MIT. Representation: test prompts such as "a doctor", "a CEO", "a nurse" over multiple seeds and report skews _(Appendix B)_.

## 9. Evaluation and results
**Functional tests** _(mark pass/fail after running)_
| Test | Expected | Result |
|---|---|---|
| Same seed twice | identical image | |
| Different seed | different image | |
| Steps 1→8 | more detail | |
| CFG ≤ 1 with negative prompt | negative has no effect | |
| Blocked-term prompt | placeholder + log entry | |
| Grid 4×3 | 12 images + zip | |

**Quantitative table** (fill from Grid Lab; prompt, seed 42)
| CFG \ steps | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| 1 | | | | |
| 7.5 | | | | |
| 15 | | | | |
Expected pattern to verify (not yet measured): sd-turbo degrades or oversaturates at high CFG; CLIP changes little beyond 4 steps.
**Failure case** _(one measured example: prompt, seed, steps, CFG, score, what went wrong, e.g. filter false-positive on a statue)_.

## 10. Deployment
HF Spaces (Streamlit SDK, GPU T4) per README; secrets via Space settings; DCGAN weights committed or via `DCGAN_WEIGHTS`. Cold start ≈ model download time.
## 11. Compliance checklist
- [ ] NSFW filter enabled and logged  - [ ] Licences checked and cited  - [ ] No secrets in repo
- [ ] MODEL_CARD.md present  - [ ] Representation test documented  - [ ] AI-generated images labelled
## 12. Limitations & future work
CPU latency; CLIP filter is coarse (use a dedicated safety classifier); DCGAN is low-resolution; add SDXL-Turbo, LoRA, schedulers comparison, FID.
## 13. Conclusion
The tool makes parameter effects visible and measurable. _<summarise your findings once measured>_

## References
Ho et al. 2020, *DDPM*; Radford et al. 2015, *DCGAN*; Ho & Salimans 2022, *Classifier-Free Guidance*; Radford et al. 2021, *CLIP*; Sauer et al. 2023, *Adversarial Diffusion Distillation* (sd-turbo); Hugging Face diffusers docs; Streamlit docs.
## Appendices
A. Prompts and CLIP scores · B. Representation test · C. MODEL_CARD.md · D. Sample images
