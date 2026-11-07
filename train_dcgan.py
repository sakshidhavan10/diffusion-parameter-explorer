"""Train a small DCGAN. Usage:
  python train_dcgan.py --data cifar10 --cls 3 --epochs 30        # CIFAR-10 'cat' class
  python train_dcgan.py --data folder --path ./data/celeba --epochs 10   # faces (ImageFolder layout)
Saves weights/dcgan_netG.pth  (GPU strongly recommended)."""
import argparse, os, torch, torch.nn as nn
import torchvision.transforms as T, torchvision.datasets as D
from torch.utils.data import DataLoader, Subset
from src.dcgan import Generator, Discriminator, NZ

p = argparse.ArgumentParser()
p.add_argument("--data", default="cifar10"); p.add_argument("--path", default="./data")
p.add_argument("--cls", type=int, default=3); p.add_argument("--epochs", type=int, default=30)
p.add_argument("--bs", type=int, default=128)
a = p.parse_args()

tf = T.Compose([T.Resize(64), T.CenterCrop(64), T.ToTensor(), T.Normalize((.5,) * 3, (.5,) * 3)])
if a.data == "cifar10":
    ds = D.CIFAR10(a.path, train=True, download=True, transform=tf)
    ds = Subset(ds, [i for i, y in enumerate(ds.targets) if y == a.cls])
else:
    ds = D.ImageFolder(a.path, transform=tf)
dl = DataLoader(ds, batch_size=a.bs, shuffle=True, num_workers=2, drop_last=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
G, Dn = Generator().to(dev), Discriminator().to(dev)
init = lambda m: nn.init.normal_(m.weight, 0, 0.02) if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)) else None
G.apply(init); Dn.apply(init)
oG = torch.optim.Adam(G.parameters(), 2e-4, betas=(.5, .999)); oD = torch.optim.Adam(Dn.parameters(), 2e-4, betas=(.5, .999))
bce = nn.BCELoss()
for ep in range(a.epochs):
    for x, _ in dl:
        x = x.to(dev); b = x.size(0)
        z = torch.randn(b, NZ, 1, 1, device=dev); fake = G(z)
        oD.zero_grad()
        lD = bce(Dn(x), torch.ones(b, device=dev)) + bce(Dn(fake.detach()), torch.zeros(b, device=dev))
        lD.backward(); oD.step()
        oG.zero_grad(); lG = bce(Dn(fake), torch.ones(b, device=dev)); lG.backward(); oG.step()
    print(f"epoch {ep+1}/{a.epochs}  lossD={lD.item():.3f} lossG={lG.item():.3f}")
os.makedirs("weights", exist_ok=True)
torch.save(G.state_dict(), "weights/dcgan_netG.pth")
print("saved weights/dcgan_netG.pth")
