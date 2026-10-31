"""Generate samples and reconstructions with the trained VAE,
and plot the training loss curve.

Outputs:
  results/samples_final.png     8x8 grid of generated images
  results/reconstructions.png   top: real test images, bottom: VAE reconstructions
  results/loss_curve.png        training loss curves (total / recon / KL)
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torchvision.utils as vutils

from dataset import get_eval_loader
from vae import VAE

SEED = 42
CKPT = "results/vae.pth"


@torch.no_grad()
def main():
    torch.manual_seed(SEED)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ckpt = torch.load(CKPT, map_location="cpu")
    cfg = ckpt["config"]
    model = VAE(1, cfg["latent_dim"]).to(device).eval()
    model.load_state_dict(ckpt["model"])
    print(f"loaded VAE from epoch {ckpt['epoch']}")

    # 1) generated sample grid
    samples = model.sample(64, device)
    vutils.save_image(samples, "results/samples_final.png", nrow=8, padding=2)
    print("saved results/samples_final.png")

    # 2) reconstructions of real test images
    loader = get_eval_loader("data/chest_xray", ("test",),
                             cfg["img_size"], batch_size=8)
    x, _ = next(iter(loader))
    x = x.to(device)
    logits, mu, logvar = model(x)
    recon = torch.sigmoid(logits)
    strip = torch.cat([x, recon], dim=0)
    vutils.save_image(strip, "results/reconstructions.png", nrow=8, padding=2)
    print("saved results/reconstructions.png")

    # 3) loss curves
    with open("results/losses.json") as f:
        h = json.load(f)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    axes[0].plot(h["epoch"], h["loss"], label="total loss")
    axes[0].plot(h["epoch"], h["recon"], label="reconstruction (BCE)")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss / image")
    axes[0].legend()
    axes[0].set_title("Total & reconstruction loss")
    axes[1].plot(h["epoch"], h["kl"], color="tab:green", label="KL")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("nats / image")
    axes[1].legend()
    axes[1].set_title("KL divergence")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("results/loss_curve.png", dpi=200)
    print("saved results/loss_curve.png")


if __name__ == "__main__":
    main()
