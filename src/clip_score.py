"""CLIP image-text similarity (CLIPScore = 100 * max(cos, 0))."""
import torch
import streamlit as st
from transformers import CLIPModel, CLIPProcessor

CLIP_ID = "openai/clip-vit-base-patch32"


@st.cache_resource(show_spinner="Loading CLIP...")
def load_clip():
    return CLIPModel.from_pretrained(CLIP_ID).eval(), CLIPProcessor.from_pretrained(CLIP_ID)


@torch.no_grad()
def clip_score(image, text):
    model, proc = load_clip()
    inp = proc(text=[text], images=image, return_tensors="pt", padding=True, truncation=True, max_length=77)
    out = model(**inp)
    cos = torch.nn.functional.cosine_similarity(out.image_embeds, out.text_embeds).item()
    return round(max(0.0, 100 * cos), 2)
