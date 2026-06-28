"""
Comparativo CON vs SIN augmentation geometrica (Modelo B).

Entrena el mismo modelo dos veces, con todo identico excepto
augmentar_geometria=True/False, y compara val loss + pixel error.

USO (en tu Mac, dentro del proyecto, junto a augmentation_geometrica.py):
    python comparar_augmentation.py --csv train_14kp.csv --frames frames_train \
        --epochs 8 --batch 16 --max_samples 800 --sigma 1.5

Si no pasas --max_samples, usa el CSV completo (mas lento, mas confiable).
Recomendado para una primera pasada: max_samples chico (500-1000) para que
corra en minutos y veas la tendencia antes de comprometer un run largo.

SALIDA: tabla comparativa impresa al final + comparacion_aug.csv
"""

import argparse
import csv
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from augmentation_geometrica import aplicar_augmentation_geometrica, kps_dict_desde_row

KP_NAMES = [
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
    "head", "neck"
]
N_KP = len(KP_NAMES)
HM_SIZE = 64


def generar_heatmap(x_norm, y_norm, size=64, sigma=2.0):
    hm = np.zeros((size, size), dtype=np.float32)
    if not (0.0 <= x_norm <= 1.0 and 0.0 <= y_norm <= 1.0):
        return hm
    cx = x_norm * (size - 1)
    cy = y_norm * (size - 1)
    xs = np.arange(size)
    ys = np.arange(size)
    xx, yy = np.meshgrid(xs, ys)
    return np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2)).astype(np.float32)


class PoseDatasetB(Dataset):
    def __init__(self, df, frames_dir, transform, sigma, augmentar_geometria):
        self.df = df.reset_index(drop=True)
        self.frames_dir = frames_dir
        self.transform = transform
        self.sigma = sigma
        self.augmentar_geometria = augmentar_geometria

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = os.path.join(self.frames_dir, row["imagen"])
        img = Image.open(path).convert("RGB")
        kps_dict = kps_dict_desde_row(row, KP_NAMES)

        if self.augmentar_geometria:
            img, kps_dict = aplicar_augmentation_geometrica(img, kps_dict, out_size=224)
        else:
            img = img.resize((224, 224), Image.BILINEAR)

        img = self.transform(img)

        heatmaps = np.zeros((N_KP, HM_SIZE, HM_SIZE), dtype=np.float32)
        coords = np.full((N_KP, 2), -1.0, dtype=np.float32)
        for i, kp in enumerate(KP_NAMES):
            x, y = kps_dict[kp]
            heatmaps[i] = generar_heatmap(x, y, size=HM_SIZE, sigma=self.sigma)
            coords[i] = (x, y)

        return img, torch.tensor(heatmaps), torch.tensor(coords)


class PoseModelB(nn.Module):
    def __init__(self, n_keypoints=14):
        super().__init__()
        backbone = models.mobilenet_v2(weights="IMAGENET1K_V1")
        for p in backbone.features.parameters():
            p.requires_grad = False
        self.encoder = backbone.features
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1280, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, n_keypoints, 1), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return F.interpolate(x, size=(HM_SIZE, HM_SIZE), mode="bilinear", align_corners=False)


def heatmap_a_coords(hm_batch):
    """De (B,14,64,64) a coords normalizadas (B,14,2), tomando el argmax."""
    B, K, H, W = hm_batch.shape
    flat = hm_batch.view(B, K, -1)
    idx = flat.argmax(dim=-1)
    ys = (idx // W).float() / (H - 1)
    xs = (idx % W).float() / (W - 1)
    return torch.stack([xs, ys], dim=-1)  # (B,14,2)


def pixel_error(pred_coords, gt_coords, crop_size=224):
    """Error en pixeles (sobre el crop de 224), solo en keypoints validos (gt != -1)."""
    valid = (gt_coords[..., 0] >= 0) & (gt_coords[..., 1] >= 0)
    if valid.sum() == 0:
        return float("nan")
    diff = (pred_coords - gt_coords) * crop_size
    dist = torch.sqrt((diff ** 2).sum(dim=-1))
    return dist[valid].mean().item()


def entrenar_una_variante(nombre, augmentar_geometria, df_train, df_val, frames, args, device):
    print(f"\n{'='*60}")
    print(f"  Variante: {nombre} (augmentar_geometria={augmentar_geometria})")
    print(f"{'='*60}")

    train_tf = transforms.Compose([
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = PoseDatasetB(df_train, frames, train_tf, args.sigma, augmentar_geometria)
    val_ds = PoseDatasetB(df_val, frames, val_tf, args.sigma, False)  # val SIEMPRE sin aug
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=2)

    torch.manual_seed(123)  # mismos pesos iniciales en ambas variantes
    model = PoseModelB(N_KP).to(device)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4
    )

    history = []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0
        for imgs, hms, _ in train_dl:
            imgs, hms = imgs.to(device), hms.to(device)
            optimizer.zero_grad()
            pred = model(imgs)
            loss = F.mse_loss(pred, hms)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_dl)

        model.eval()
        val_loss = 0
        px_errs = []
        with torch.no_grad():
            for imgs, hms, coords in val_dl:
                imgs, hms = imgs.to(device), hms.to(device)
                pred = model(imgs)
                val_loss += F.mse_loss(pred, hms).item()
                pred_coords = heatmap_a_coords(pred).cpu()
                px_errs.append(pixel_error(pred_coords, coords))
        val_loss /= len(val_dl)
        px_err = float(np.nanmean(px_errs))

        elapsed = time.time() - t0
        print(f"  Epoch {epoch:2d}/{args.epochs} | train: {train_loss:.5f} | val: {val_loss:.5f} "
              f"| pixel_err: {px_err:.1f}px | {elapsed:.1f}s")
        history.append((epoch, train_loss, val_loss, px_err))

    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="ej: train_14kp.csv")
    parser.add_argument("--frames", required=True, help="ej: frames_train")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--sigma", type=float, default=1.5)
    parser.add_argument("--max_samples", type=int, default=800,
                         help="subset chico para que corra rapido en CPU. 0 = usar todo el csv")
    parser.add_argument("--val_frac", type=float, default=0.2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    df = pd.read_csv(args.csv)
    if args.max_samples and args.max_samples > 0:
        df = df.sample(n=min(args.max_samples, len(df)), random_state=42).reset_index(drop=True)

    n_val = int(len(df) * args.val_frac)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle reproducible
    df_val = df.iloc[:n_val].reset_index(drop=True)
    df_train = df.iloc[n_val:].reset_index(drop=True)
    print(f"Train: {len(df_train)} | Val: {len(df_val)} (mismo split para ambas variantes)")

    hist_sin = entrenar_una_variante("SIN augmentation geometrica", False,
                                      df_train, df_val, args.frames, args, device)
    hist_con = entrenar_una_variante("CON augmentation geometrica", True,
                                      df_train, df_val, args.frames, args, device)

    print(f"\n{'='*60}")
    print("  RESUMEN FINAL (ultima epoca)")
    print(f"{'='*60}")
    _, tl_sin, vl_sin, px_sin = hist_sin[-1]
    _, tl_con, vl_con, px_con = hist_con[-1]
    print(f"  SIN augmentation -> val_loss: {vl_sin:.5f} | pixel_error: {px_sin:.1f}px")
    print(f"  CON augmentation -> val_loss: {vl_con:.5f} | pixel_error: {px_con:.1f}px")

    mejor_vl = "CON" if vl_con < vl_sin else "SIN"
    mejor_px = "CON" if px_con < px_sin else "SIN"
    print(f"\n  Mejor val_loss:    {mejor_vl} augmentation")
    print(f"  Mejor pixel_error: {mejor_px} augmentation")

    with open("comparacion_aug.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variante", "epoch", "train_loss", "val_loss", "pixel_error_px"])
        for ep, tl, vl, px in hist_sin:
            w.writerow(["sin_aug", ep, tl, vl, px])
        for ep, tl, vl, px in hist_con:
            w.writerow(["con_aug", ep, tl, vl, px])
    print("\nGuardado: comparacion_aug.csv (para graficar si queres)")


if __name__ == "__main__":
    main()