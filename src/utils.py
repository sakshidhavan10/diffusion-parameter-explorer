import io, zipfile
from PIL import Image, ImageDraw


def png_bytes(img):
    b = io.BytesIO(); img.save(b, "PNG"); return b.getvalue()


def placeholder(size=512, text="NSFW filtered"):
    img = Image.new("RGB", (size, size), (17, 17, 28))
    d = ImageDraw.Draw(img)
    d.rectangle([8, 8, size - 9, size - 9], outline=(124, 92, 255), width=3)
    d.text((size // 2 - 45, size // 2 - 6), text, fill=(0, 229, 255))
    return img


def collage(grid, steps, cfgs, tile=256):
    """grid[(steps, cfg)] -> PIL image. Rows = CFG, columns = steps, with header labels."""
    pad = 28
    W, H = pad + tile * len(steps), pad + tile * len(cfgs)
    out = Image.new("RGB", (W, H), (7, 7, 13)); d = ImageDraw.Draw(out)
    for j, s in enumerate(steps):
        d.text((pad + j * tile + tile // 2 - 20, 8), f"steps={s}", fill=(0, 229, 255))
    for i, c in enumerate(cfgs):
        d.text((2, pad + i * tile + tile // 2), f"cfg\n{c}", fill=(124, 92, 255))
        for j, s in enumerate(steps):
            out.paste(grid[(s, c)]["image"].resize((tile, tile)), (pad + j * tile, pad + i * tile))
    return out


def zip_grid(grid, collage_img):
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w") as z:
        z.writestr("collage.png", png_bytes(collage_img))
        for (s, c), r in grid.items():
            z.writestr(f"steps{s}_cfg{c}_seed{r['seed']}.png", png_bytes(r["image"]))
    return b.getvalue()
