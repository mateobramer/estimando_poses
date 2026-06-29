"""
Script de entrenamiento Modelo B (heatmaps) optimizado para GPU en GCP.
Lee train.csv y val.csv separados.

USO EN LA VM:
    python entrenar_gcp_b.py
    python entrenar_gcp_b.py --epochs 100 --batch 32
    python entrenar_gcp_b.py --continuar

SALIDA:
    checkpoints_b/mejor_modelo_b.pth
    checkpoints_b/ultimo_modelo_b.pth
    checkpoints_b/log_b.csv
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import numpy as np
import os
import argparse
import time
import csv
import math

KP_NAMES = [
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
    "head", "neck"
]
N_KP    = len(KP_NAMES)
HM_SIZE = 64


# Generar heatmap gaussiano
def generar_heatmap(x_norm, y_norm, size=64, sigma=2.0):
    hm = np.zeros((size, size), dtype=np.float32)
    if not (0.0 <= x_norm <= 1.0 and 0.0 <= y_norm <= 1.0):
        return hm
    cx = x_norm * (size - 1)
    cy = y_norm * (size - 1)
    xs = np.arange(size)
    ys = np.arange(size)
    xx, yy = np.meshgrid(xs, ys)
    hm = np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * sigma**2))
    return hm.astype(np.float32)


# Dataset
class PoseDatasetB(Dataset):
    def __init__(self, df, frames_dir, transform):
        self.df = df.reset_index(drop=True)
        self.frames_dir = frames_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = os.path.join(self.frames_dir, row["imagen"])
        img = Image.open(path).convert("RGB")
        img = self.transform(img)

        heatmaps = np.zeros((N_KP, HM_SIZE, HM_SIZE), dtype=np.float32)
        for i, kp in enumerate(KP_NAMES):
            x = float(row[f"{kp}_x"])
            y = float(row[f"{kp}_y"])
            heatmaps[i] = generar_heatmap(x, y, size=HM_SIZE)

        return img, torch.tensor(heatmaps)


# Modelo B
class PoseModelB(nn.Module):
    def __init__(self, n_keypoints=14):
        super().__init__()
        backbone = models.mobilenet_v2(weights="IMAGENET1K_V1")
        for param in backbone.features.parameters():
            param.requires_grad = False
        self.encoder = backbone.features
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1280, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, n_keypoints, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = F.interpolate(x, size=(HM_SIZE, HM_SIZE), mode='bilinear', align_corners=False)
        return x

    def descongelar_backbone(self):
        for param in self.encoder.parameters():
            param.requires_grad = True


def entrenar(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs("checkpoints_b", exist_ok=True)

    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        transforms.RandomRotation(15),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("Cargando datasets...")
    df_train = pd.read_csv(args.train_csv)
    df_val   = pd.read_csv(args.val_csv)
    print(f"  Train: {len(df_train)} | Val: {len(df_val)}")

    train_ds = PoseDatasetB(df_train, args.frames, train_transform)
    val_ds   = PoseDatasetB(df_val,   args.frames, val_transform)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=4, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                          num_workers=4, pin_memory=True)

    model = PoseModelB(n_keypoints=N_KP).to(device)

    if args.continuar and os.path.exists("checkpoints_b/mejor_modelo_b.pth"):
        model.load_state_dict(torch.load("checkpoints_b/mejor_modelo_b.pth", map_location=device))
        model.descongelar_backbone()
        print("  Continuando desde checkpoint, backbone descongelado")

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    mejor_val_loss = float("inf")
    log_path = "checkpoints_b/log_b.csv"

    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "lr", "tiempo_s"])

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        model.train()
        train_loss = 0
        for imgs, hms in train_dl:
            imgs, hms = imgs.to(device, non_blocking=True), hms.to(device, non_blocking=True)
            optimizer.zero_grad()
            pred = model(imgs)
            loss = F.mse_loss(pred, hms)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_dl)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, hms in val_dl:
                imgs, hms = imgs.to(device, non_blocking=True), hms.to(device, non_blocking=True)
                pred = model(imgs)
                val_loss += F.mse_loss(pred, hms).item()
        val_loss /= len(val_dl)

        scheduler.step(val_loss)
        elapsed = time.time() - t0
        lr_actual = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch:3d}/{args.epochs} | train: {train_loss:.5f} | val: {val_loss:.5f} | lr: {lr_actual:.6f} | {elapsed:.1f}s")

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, round(train_loss, 6), round(val_loss, 6), lr_actual, round(elapsed, 1)])

        if val_loss < mejor_val_loss:
            mejor_val_loss = val_loss
            torch.save(model.state_dict(), "checkpoints_b/mejor_modelo_b.pth")
            print(f"  Mejor modelo guardado ({mejor_val_loss:.5f})")

        torch.save(model.state_dict(), "checkpoints_b/ultimo_modelo_b.pth")

        if epoch == args.epochs // 2:
            print("  Descongelando backbone...")
            model.descongelar_backbone()
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr * 0.1, weight_decay=1e-4)

    print(f"\nEntrenamiento terminado. Mejor val loss: {mejor_val_loss:.5f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", default="dataset_final/train.csv")
    parser.add_argument("--val_csv",   default="dataset_final/val.csv")
    parser.add_argument("--frames",    default="dataset_final/frames")
    parser.add_argument("--epochs",    type=int,   default=100)
    parser.add_argument("--batch",     type=int,   default=32)
    parser.add_argument("--lr",        type=float, default=0.001)
    parser.add_argument("--continuar", action="store_true")
    args = parser.parse_args()
    entrenar(args)


if __name__ == "__main__":
    main()