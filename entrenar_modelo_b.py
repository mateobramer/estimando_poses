"""
Modelo B — Pose estimation con heatmaps gaussianos.
MobileNetV2 como encoder + decoder con upsampling que predice 14 mapas de 64x64.

USO:
    python entrenar_modelo_b.py --epochs 3 --batch 8 --max_samples 500
    python entrenar_modelo_b.py --epochs 50 --batch 32

SALIDA:
    modelo_b_mejor.pth
    modelo_b_ultimo.pth
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

KP_NAMES = [
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
    "head", "neck"
]
N_KP    = len(KP_NAMES)   # 14
HM_SIZE = 64              # tamaño de cada heatmap


# ── Generar heatmap gaussiano ──────────────────────────────────────
def generar_heatmap(x_norm, y_norm, size=64, sigma=2.0):
    """
    Genera un mapa de calor gaussiano centrado en (x_norm, y_norm).
    Si las coordenadas son -1 (invisible) devuelve mapa en cero.
    """
    hm = np.zeros((size, size), dtype=np.float32)
    if not (0.0 <= x_norm <= 1.0 and 0.0 <= y_norm <= 1.0):
        return hm
    cx = x_norm * (size - 1)
    cy = y_norm * (size - 1)
    for y in range(size):
        for x in range(size):
            hm[y, x] = np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))
    return hm


# ── Dataset ────────────────────────────────────────────────────────
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

        # generar 14 heatmaps
        heatmaps = np.zeros((N_KP, HM_SIZE, HM_SIZE), dtype=np.float32)
        for i, kp in enumerate(KP_NAMES):
            x = float(row[f"{kp}_x"])
            y = float(row[f"{kp}_y"])
            heatmaps[i] = generar_heatmap(x, y, size=HM_SIZE)

        return img, torch.tensor(heatmaps)


# ── Modelo B ───────────────────────────────────────────────────────
class PoseModelB(nn.Module):
    """
    Encoder: MobileNetV2 (features 7x7x1280)
    Decoder: upsampling con ConvTranspose2d hasta 64x64
    Output:  14 heatmaps de 64x64
    """
    def __init__(self, n_keypoints=14):
        super().__init__()
        backbone = models.mobilenet_v2(weights="IMAGENET1K_V1")
        for param in backbone.features.parameters():
            param.requires_grad = False
        self.encoder = backbone.features  # salida: (B, 1280, 7, 7)

        self.decoder = nn.Sequential(
            # 7x7 → 14x14
            nn.ConvTranspose2d(1280, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # 14x14 → 28x28
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # 28x28 → 56x56
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 56x56 → 64x64 (via interpolation)
            nn.Conv2d(64, n_keypoints, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)       # (B, 1280, 7, 7)
        x = self.decoder(x)       # (B, 14, 56, 56)
        # redimensionar a 64x64
        x = F.interpolate(x, size=(HM_SIZE, HM_SIZE), mode='bilinear', align_corners=False)
        return x                  # (B, 14, 64, 64)


# ── Loss ───────────────────────────────────────────────────────────
def heatmap_loss(pred, target):
    """MSE entre heatmaps predichos y gaussianas objetivo."""
    return F.mse_loss(pred, target)


# ── Entrenamiento ──────────────────────────────────────────────────
def entrenar(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomRotation(15),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    print(f"Cargando dataset: {args.csv}")
    df = pd.read_csv(args.csv)
    if args.max_samples:
        df = df.head(args.max_samples)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    n_val    = int(len(df) * 0.1)
    df_val   = df[:n_val]
    df_train = df[n_val:]
    print(f"  Train: {len(df_train)} | Val: {len(df_val)}")

    train_ds = PoseDatasetB(df_train, args.frames, train_transform)
    val_ds   = PoseDatasetB(df_val,   args.frames, val_transform)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=0)

    model     = PoseModelB(n_keypoints=N_KP).to(device)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    mejor_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # train
        model.train()
        train_loss = 0
        for imgs, hms in train_dl:
            imgs, hms = imgs.to(device), hms.to(device)
            optimizer.zero_grad()
            pred = model(imgs)
            loss = heatmap_loss(pred, hms)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_dl)

        # val
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, hms in val_dl:
                imgs, hms = imgs.to(device), hms.to(device)
                pred = model(imgs)
                val_loss += heatmap_loss(pred, hms).item()
        val_loss /= len(val_dl)

        scheduler.step(val_loss)
        elapsed = time.time() - t0
        print(f"Epoch {epoch:3d}/{args.epochs} | train: {train_loss:.5f} | val: {val_loss:.5f} | {elapsed:.1f}s")

        if val_loss < mejor_val_loss:
            mejor_val_loss = val_loss
            torch.save(model.state_dict(), args.salida_mejor)
            print(f"  Mejor modelo guardado ({mejor_val_loss:.5f})")

        # descongelar backbone a mitad
        if epoch == args.epochs // 2:
            print("  Descongelando backbone...")
            for param in model.encoder.parameters():
                param.requires_grad = True
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr * 0.1, weight_decay=1e-4)

    torch.save(model.state_dict(), args.salida_ultimo)
    print(f"\nEntrenamiento terminado. Mejor val loss: {mejor_val_loss:.5f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",           default="dataset_final/keypoints.csv")
    parser.add_argument("--frames",        default="dataset_final/frames")
    parser.add_argument("--epochs",        type=int,   default=20)
    parser.add_argument("--batch",         type=int,   default=16)
    parser.add_argument("--lr",            type=float, default=0.001)
    parser.add_argument("--max_samples",   type=int,   default=None)
    parser.add_argument("--salida_mejor",  default="modelo_b_mejor.pth")
    parser.add_argument("--salida_ultimo", default="modelo_b_ultimo.pth")
    args = parser.parse_args()
    entrenar(args)


if __name__ == "__main__":
    main()