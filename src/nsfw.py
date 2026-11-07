"""Two-layer NSFW filter: prompt blocklist + CLIP zero-shot image classifier. Blocked items are logged."""
import csv, os, time
import torch
from src.clip_score import load_clip

LOG = os.path.join("logs", "nsfw_log.csv")
BLOCKLIST = {
    "nude", "naked", "nsfw", "explicit", "porn", "sex", "gore", "bloody",
    "hentai", "erotic", "topless"
}

SAFE = [
    "a photo of an animal", "a landscape or nature scene", "a photo of a fully clothed person",
    "a painting or illustration", "a city, building or street", "an everyday object or food",
    "abstract art or a pattern", "a blurry or low-quality image"
]
UNSAFE = ["a nude person, explicit sexual content", "graphic violence, blood and gore"]
LABELS = SAFE + UNSAFE
THRESHOLD = float(os.getenv("NSFW_THRESHOLD", "0.6"))


def _log(kind, prompt, score):
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    new = not os.path.exists(LOG)
    with open(LOG, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "kind", "prompt", "unsafe_prob"])
        w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), kind, prompt, score])


def check_prompt(prompt):
    if not prompt:
        return False
    hit = BLOCKLIST & set(prompt.lower().replace(",", " ").split())
    if hit:
        _log("prompt", prompt, 1.0)
    return bool(hit)


@torch.no_grad()
def check_image(image, prompt=""):
    model, proc = load_clip()
    inp = proc(text=LABELS, images=image, return_tensors="pt", padding=True)
    probs = model(**inp).logits_per_image.softmax(-1)[0]
    unsafe = float(probs[len(SAFE):].sum())
    flagged = unsafe >= THRESHOLD
    if flagged:
        _log("image", prompt, round(unsafe, 4))
    return {"flagged": flagged, "unsafe_p": round(unsafe, 4)}
