"""DCGAN generator (64x64) + sampling. Train weights with train_dcgan.py -> weights/dcgan_netG.pth"""
import os
import torch, torch.nn as nn
from PIL import Image

WEIGHTS = os.getenv("DCGAN_WEIGHTS", os.path.join(os.path.dirname(__file__), "..", "weights", "dcgan_netG.pth"))
NZ = 100


class Generator(nn.Module):
    def __init__(self, nz=NZ, ngf=64, nc=3):
        super().__init__()
        def up(i, o, k=4, s=2, p=1):
            return [nn.ConvTranspose2d(i, o, k, s, p, bias=False), nn.BatchNorm2d(o), nn.ReLU(True)]
        self.main = nn.Sequential(*up(nz, ngf * 8, 4, 1, 0), *up(ngf * 8, ngf * 4), *up(ngf * 4, ngf * 2),
                                  *up(ngf * 2, ngf), nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False), nn.Tanh())

    def forward(self, z):
        return self.main(z)


class Discriminator(nn.Module):
    def __init__(self, ndf=64, nc=3):
        super().__init__()
        def down(i, o, bn=True):
            return [nn.Conv2d(i, o, 4, 2, 1, bias=False)] + ([nn.BatchNorm2d(o)] if bn else []) + [nn.LeakyReLU(0.2, True)]
        self.main = nn.Sequential(*down(nc, ndf, False), *down(ndf, ndf * 2), *down(ndf * 2, ndf * 4),
                                  *down(ndf * 4, ndf * 8), nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False), nn.Sigmoid())

    def forward(self, x):
        return self.main(x).view(-1)


def weights_available():
    return os.path.exists(WEIGHTS)


def sample(n=6, seed=0, out_size=256):
    """Returns (list[PIL], trained_flag). Untrained nets give noise - the UI labels this clearly."""
    g = Generator().eval()
    trained = weights_available()
    if trained:
        g.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
    z = torch.randn(n, NZ, 1, 1, generator=torch.Generator().manual_seed(int(seed)))
    with torch.no_grad():
        x = ((g(z) + 1) / 2).clamp(0, 1)
    imgs = [Image.fromarray((t.permute(1, 2, 0).numpy() * 255).astype("uint8")).resize((out_size, out_size), Image.BICUBIC) for t in x]
    return imgs, trained
