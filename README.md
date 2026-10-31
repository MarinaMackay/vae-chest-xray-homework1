# ECE 285 Homework 1: VAE for Chest X-ray Generation

PyTorch implementation of a convolutional variational autoencoder for the Kaggle Chest X-Ray Images (Pneumonia) dataset.

Source code repository: https://github.com/MarinaMackay/vae-chest-xray-homework1

## Files

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

Download and unpack the Kaggle dataset so the folders are under `data/chest_xray/{train,val,test}/{NORMAL,PNEUMONIA}/`. The dataset is not included here.

## Run

Run commands from the repository directory:

```bash
python train.py
python generate.py
python metrics.py
python metrics.py --max-images 512
```

Training runs 100 epochs and writes the checkpoint and loss history under `results/`. Generation uses that checkpoint and requires the test split for reconstructions. Metric evaluation uses train and validation images and downloads the torchvision ImageNet Inception V3 weights on first run.

The submitted report and saved model/results are provided in the Gradescope homework submission archive. The report documents metric settings and limitations; its reported FID and IS use PyTorch-Ignite 0.5.2, with CPU inference, PIL bilinear resizing to 299×299, and 5,232 generated versus 5,232 train/validation images. Ignite's default FID here is based on 1,000-class probabilities and is not directly comparable to the common 2,048-dimensional pool3 FID. Neither metric measures clinical quality.

The earlier reported real-data FID of 4.6 is unsupported and is not claimed. No retraining was performed for this submission.
