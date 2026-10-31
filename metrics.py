"""Evaluate VAE samples with the PyTorch-Ignite FID and Inception Score metrics.

Images are converted from grayscale to RGB and resized with PIL bilinear
interpolation to 299x299, following the evaluation procedure in the assignment's
PyTorch-Ignite example. Metrics compare prior samples against train+validation.
"""
import json
import argparse

import torch
import torch.nn as nn
import ignite
import scipy
from PIL import Image
from torch.utils.data import DataLoader, Subset
import torchvision
from torchvision import models, transforms
from ignite.metrics import FID, InceptionScore

from dataset import ChestXrayDataset
from vae import VAE

SEED = 42
BATCH = 64
DATA_ROOT = "data/chest_xray"
CKPT = "results/vae.pth"
OUT = "results/metrics.json"


def resize_pil(batch):
    """Convert (B,1,H,W) tensors in [0,1] to PIL-resized RGB tensors."""
    to_pil = transforms.ToPILImage()
    to_tensor = transforms.ToTensor()
    result = []
    for image in batch.cpu().clamp(0, 1):
        pil_image = to_pil(image).convert("RGB")
        pil_image = pil_image.resize((299, 299), resample=Image.Resampling.BILINEAR)
        result.append(to_tensor(pil_image))
    return torch.stack(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-images", type=int, default=None,
                        help="optional deterministic subset size for a faster evaluation")
    args = parser.parse_args()
    torch.manual_seed(SEED)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    ckpt = torch.load(CKPT, map_location="cpu")
    cfg = ckpt["config"]
    vae = VAE(1, cfg["latent_dim"]).to(device).eval()
    vae.load_state_dict(ckpt["model"])
    print(f"loaded VAE from epoch {ckpt['epoch']}")

    real_ds = ChestXrayDataset(DATA_ROOT, ("train", "val"), cfg["img_size"])
    n_available = len(real_ds)
    if args.max_images is not None:
        if args.max_images < 2:
            raise ValueError("--max-images must be at least 2")
        n_selected = min(args.max_images, n_available)
        subset_rng = torch.Generator().manual_seed(SEED + 1)
        indices = torch.randperm(n_available, generator=subset_rng)[:n_selected]
        real_ds = Subset(real_ds, indices.tolist())
    real_loader = DataLoader(real_ds, batch_size=BATCH, shuffle=False,
                             num_workers=0)
    n_samples = len(real_ds)
    print(f"real images for FID: {n_samples} (available: {n_available})")

    # Share one Ignite-equivalent Inception forward between both Ignite
    # metrics. Ignite's default FID and IS extractors both return the 1000-way
    # ImageNet probabilities; sharing avoids a duplicate full-model pass.
    inception = models.inception_v3(
        weights=models.Inception_V3_Weights.DEFAULT).to(device).eval()
    fid_metric = FID(num_features=1000, feature_extractor=nn.Identity(),
                     device=device)
    is_metric = InceptionScore(num_features=1000,
                               feature_extractor=nn.Identity(), device=device)

    with torch.no_grad():
        seen = 0
        for real, _ in real_loader:
            n = real.size(0)
            fake = vae.sample(n, device).cpu()
            fake_299 = resize_pil(fake).to(device)
            real_299 = resize_pil(real).to(device)

            probabilities = torch.softmax(
                inception(torch.cat((fake_299, real_299), dim=0)), dim=1)
            fake_prob, real_prob = probabilities[:n], probabilities[n:]
            fid_metric.update((fake_prob, real_prob))
            is_metric.update(fake_prob)
            seen += n
            print(f"evaluated {seen}/{n_samples} image pairs", flush=True)

    fid = float(fid_metric.compute())
    is_value = float(is_metric.compute())
    output = {
        "FID": fid,
        "IS": is_value,
        "status": "subset_evaluation" if args.max_images else "full_train_val_evaluation",
        "n_fake": seen,
        "n_real": n_samples,
        "n_real_available": n_available,
        "seed": SEED,
        "real_subset_seed": SEED + 1 if args.max_images else None,
        "metric_library": "PyTorch-Ignite FID and InceptionScore",
        "metric_library_version": ignite.__version__,
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "scipy_version": scipy.__version__,
        "device": str(device),
        "real_subset_selection": "seeded random sample without replacement" if args.max_images else "all train+validation images",
        "protocol": "Ignite default ImageNet Inception V3 1000-class probability outputs; grayscale replicated to RGB; PIL bilinear resize to 299x299; real=train+val; prior-generated fake samples",
        "warning": "ImageNet-feature proxies are not clinical-quality measures; real reference images include training data.",
    }
    with open(OUT, "w") as f:
        json.dump(output, f, indent=2)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
