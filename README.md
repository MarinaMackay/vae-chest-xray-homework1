# ECE 285 Homework 1: VAE for Chest X-ray Generation

PyTorch implementation of a convolutional variational autoencoder for the Kaggle Chest X-Ray Images (Pneumonia) dataset. The assignment report is submitted separately as a two-page PDF; this public repository hosts the implementation code.

## Source files

- `vae.py`: convolutional VAE model.
- `dataset.py`: image loading and preprocessing.
- `train.py`: training loop and checkpoint/loss-history writing.
- `generate.py`: prior samples, test-set reconstructions, and loss plot.
- `metrics.py`: PyTorch-Ignite FID and Inception Score evaluation.
- `requirements.txt`: Python dependencies.

## Setup

Use Python 3.10 or 3.11, then install dependencies:

```bash
python -m pip install -r requirements.txt
```

Download and unpack the Kaggle dataset so the folders are under `data/chest_xray/{train,val,test}/{NORMAL,PNEUMONIA}/`. The dataset is not included or redistributed here.

## Reproduce

Run commands from the repository directory:

```bash
python train.py
python generate.py
python metrics.py
python metrics.py --max-images 512
```

Training runs 100 epochs and writes a checkpoint and loss history under `results/`. Run training before generation or evaluation; generation also needs the test split, while full metric evaluation needs train and validation splits. The metrics script downloads torchvision's ImageNet Inception V3 weights on first run.

The report's recorded full evaluation used PyTorch-Ignite 0.5.2 on CPU, PIL bilinear resizing to 299×299, and 5,232 generated versus 5,232 train/validation images. Ignite's default FID here uses 1,000-class probability outputs and is not directly comparable with the common 2,048-dimensional pool3 FID. These ImageNet-based metrics are not clinical quality measures. The earlier real-data FID of 4.6 is unsupported and is not claimed.
