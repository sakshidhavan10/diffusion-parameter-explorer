"""Diffusion generation (sd-turbo by default)."""
import os
import threading
import torch
import streamlit as st

MODEL_ID = os.getenv("SD_MODEL_ID", "stabilityai/sd-turbo")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
_LOCK = threading.Lock()


@st.cache_resource(show_spinner="Loading diffusion model (first run downloads ~2 GB)...")
def load_pipe():
    from diffusers import AutoPipelineForText2Image
    kw = {"torch_dtype": DTYPE}
    if DEVICE == "cuda":
        kw["variant"] = "fp16"
    pipe = AutoPipelineForText2Image.from_pretrained(MODEL_ID, **kw)
    pipe.set_progress_bar_config(disable=True)
    # sd-turbo ships without a safety checker -> our CLIP filter (src/nsfw.py) is applied to every output.
    return pipe.to(DEVICE)


def generate(prompt, negative="", steps=1, cfg=0.0, seed=0, size=512):
    """Reproducible: the seed drives a CPU generator so the same seed gives the same initial noise."""
    pipe = load_pipe()
    g = torch.Generator("cpu").manual_seed(int(seed))
    kw = dict(prompt=prompt, num_inference_steps=int(steps), guidance_scale=float(cfg),
              generator=g, height=int(size), width=int(size))
    # Negative prompts only act when CFG > 1 (they replace the "unconditional" branch).
    if negative.strip() and cfg > 1.0:
        kw["negative_prompt"] = negative
    with _LOCK:  # one generation at a time on the shared pipeline
        pipe.scheduler = pipe.scheduler.from_config(pipe.scheduler.config)  # fresh step counter
        return pipe(**kw).images[0]