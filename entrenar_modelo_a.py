"""
Modelo A — Regresión directa de keypoints.
MobileNetV2 preentrenado + cabeza de regresión que predice 28 valores (14 keypoints x 2).

USO:
    python entrenar_modelo_a.py
    python entrenar_modelo_a.py --epochs 30 --batch 32 --lr 0.001

SALIDA:
    modelo_a_mejor.pth   ← mejor modelo según val loss
    modelo_a_ultimo.pth  ← último epoch
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import numpy as np
import os
import argparse
import time

# ── Configuración ──────────────────────────────────────────────────
KP_NAMES = [
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
    "head", "neck"
]
N_KP = len(KP_NAMES)  # 14


# ── Dataset ────────────────────────────────────────────────────────
class PoseDataset(Dataset):
    def __init__(self, df, frames_dir, transform):
        self.df = df.reset_index(drop=True)
        self.frames_dir = frames_dir
        self.transform = transform
        self.kp_cols = [f"{kp}_{c}" for kp in KP_NAMES for c in ["x", "y"]]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = os.path.join(self.frames_dir, row["imagen"])
        img = Image.open(path).convert("RGB")
        img = self.transform(img)

        kp = torch.tensor(row[self.kp_cols].values.astype(float), dtype=torch.float32)

        # máscara: 1 si el keypoint es visible (entre 0 y 1), 0 si no
        mask = torch.zeros(N_KP * 2)
        for i, col in enumerate(self.kp_cols):
            v = float(row[col])
            if 0.0 <= v <= 1.0:
                mask[i] = 1.0

        return img, kp, mask


# ── Modelo ─────────────────────────────────────────────────────────
class PoseModelA(nn.Module):
    def __init__(self, n_keypoints=14):
        super().__init__()
        backbone = models.mobilenet_v2(weights="IMAGENET1K_V1")
        # congelar backbone al principio
        for param in backbone.features.parameters():
            param.requires_grad = False
        self.backbone = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, n_keypoints * 2),
            nn.Sigmoid()  # fuerza salida entre 0 y 1
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        return self.head(x)


# ── Loss con máscara ───────────────────────────────────────────────
def masked_mse(pred, target, mask):
    loss = ((pred - target) ** 2) * mask
    n = mask.sum()
    return loss.sum() / (n + 1e-6)


# ── Entrenamiento ──────────────────────────────────────────────────
def entrenar(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    # transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomHorizontalFlip(),
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

    # cargar datos
    print(f"Cargando dataset: {args.csv}")
    df = pd.read_csv(args.csv)
    if args.max_samples:
        df = df.head(args.max_samples)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    n_val = int(len(df) * 0.1)
    df_val = df[:n_val]
    df_train = df[n_val:]
    print(f"  Train: {len(df_train)} | Val: {len(df_val)}")

    train_ds = PoseDataset(df_train, args.frames, train_transform)
    val_ds   = PoseDataset(df_val,   args.frames, val_transform)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=0)

    # modelo
    model = PoseModelA(n_keypoints=N_KP).to(device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    mejor_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # --- train ---
        model.train()
        train_loss = 0
        for imgs, kps, masks in train_dl:
            imgs, kps, masks = imgs.to(device), kps.to(device), masks.to(device)
            optimizer.zero_grad()
            pred = model(imgs)
            loss = masked_mse(pred, kps, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_dl)

        # --- val ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, kps, masks in val_dl:
                imgs, kps, masks = imgs.to(device), kps.to(device), masks.to(device)
                pred = model(imgs)
                val_loss += masked_mse(pred, kps, masks).item()
        val_loss /= len(val_dl)

        scheduler.step(val_loss)
        elapsed = time.time() - t0

        print(f"Epoch {epoch:3d}/{args.epochs} | train: {train_loss:.4f} | val: {val_loss:.4f} | {elapsed:.1f}s")

        # guardar mejor modelo
        if val_loss < mejor_val_loss:
            mejor_val_loss = val_loss
            torch.save(model.state_dict(), args.salida_mejor)
            print(f"  ✓ Mejor modelo guardado ({mejor_val_loss:.4f})")

        # descongelar backbone a mitad del entrenamiento
        if epoch == args.epochs // 2:
            print("  Descongelando backbone para fine-tuning completo...")
            for param in model.backbone.parameters():
                param.requires_grad = True
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr * 0.1)

    torch.save(model.state_dict(), args.salida_ultimo)
    print(f"\n✓ Entrenamiento terminado. Mejor val loss: {mejor_val_loss:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",          default="dataset_final/keypoints.csv")
    parser.add_argument("--frames",       default="dataset_final/frames")
    parser.add_argument("--epochs",       type=int,   default=20)
    parser.add_argument("--batch",        type=int,   default=32)
    parser.add_argument("--lr",           type=float, default=0.001)
    parser.add_argument("--salida_mejor", default="modelo_a_mejor.pth")
    parser.add_argument("--max_samples",  type=int,   default=None)
    parser.add_argument("--salida_ultimo",default="modelo_a_ultimo.pth")
    args = parser.parse_args()

    entrenar(args)


if __name__ == "__main__":
    main()