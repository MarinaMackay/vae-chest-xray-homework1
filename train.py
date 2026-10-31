"""Train the VAE on the chest X-ray dataset.

Loss: ELBO (negative), L = BCE_recon + beta * KL, with linear KL warm-up.
"""
import json
import math
import os
import time

import torch
import torch.nn.functional as F
import torchvision.utils as vutils

from dataset import get_train_loader
from vae import VAE

# ------------------------- hyperparameters -------------------------
SEED = 42
IMG_SIZE = 128
BATCH_SIZE = 64
LATENT_DIM = 128
LR = 2e-4
EPOCHS = 100
BETA = 1.0
WARMUP_EPOCHS = 10          # KL term ramps linearly 0 -> BETA over this span
OUT_DIR = "results"
DATA_ROOT = "data/chest_xray"
# -------------------------------------------------------------------


def main():
    torch.manual_seed(SEED)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    os.makedirs(os.path.join(OUT_DIR, "samples"), exist_ok=True)
    print(f"device: {device}")

    loader = get_train_loader(DATA_ROOT, IMG_SIZE, BATCH_SIZE)
    print(f"training images: {len(loader.dataset)}")

    model = VAE(1, LATENT_DIM).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model parameters: {n_params:,}")
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999))

    fixed_z = torch.randn(64, LATENT_DIM, device=device)
    history = {"epoch": [], "loss": [], "recon": [], "kl": [], "beta": []}

    for epoch in range(1, EPOCHS + 1):
        beta = BETA * min(1.0, (epoch - 1) / max(1, WARMUP_EPOCHS))
        model.train()
        t0 = time.time()
        sums = {"loss": 0.0, "recon": 0.0, "kl": 0.0}
        n_batches = 0
        for x, _ in loader:
            x = x.to(device)
            x_recon_logits, mu, logvar = model(x)
            recon = F.binary_cross_entropy_with_logits(
                x_recon_logits, x, reduction="sum")
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon + beta * kl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bs = x.size(0)
            sums["loss"] += loss.item() / bs
            sums["recon"] += recon.item() / bs
            sums["kl"] += kl.item() / bs
            n_batches += 1

        loss_m = sums["loss"] / n_batches
        recon_m = sums["recon"] / n_batches
        kl_m = sums["kl"] / n_batches
        history["epoch"].append(epoch)
        history["loss"].append(loss_m)
        history["recon"].append(recon_m)
        history["kl"].append(kl_m)
        history["beta"].append(beta)
        dt = time.time() - t0
        print(f"epoch {epoch:3d}/{EPOCHS} | loss {loss_m:9.1f} | "
              f"recon {recon_m:9.1f} | KL {kl_m:8.2f} | beta {beta:.2f} "
              f"| {dt:5.1f}s", flush=True)

        if epoch % 10 == 0 or epoch == EPOCHS:
            model.eval()
            with torch.no_grad():
                grid = torch.sigmoid(model.decoder(fixed_z))
            vutils.save_image(
                grid, f"{OUT_DIR}/samples/epoch_{epoch:03d}.png", nrow=8)
            torch.save(
                {"model": model.state_dict(), "epoch": epoch,
                 "config": {"latent_dim": LATENT_DIM, "img_size": IMG_SIZE}},
                f"{OUT_DIR}/vae.pth")
            with open(f"{OUT_DIR}/losses.json", "w") as f:
                json.dump(history, f)

    torch.save(
        {"model": model.state_dict(), "epoch": EPOCHS,
         "config": {"latent_dim": LATENT_DIM, "img_size": IMG_SIZE}},
        f"{OUT_DIR}/vae.pth")
    with open(f"{OUT_DIR}/losses.json", "w") as f:
        json.dump(history, f)
    print("done.")


if __name__ == "__main__":
    main()
