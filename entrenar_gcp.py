"""
Script de entrenamiento optimizado para GPU en GCP.
Lee train.csv y val.csv separados, usa batch grande y guarda checkpoints.

USO EN LA VM:
    python entrenar_gcp.py
    python entrenar_gcp.py --epochs 100 --batch 64 --backbone mobilenetv2_100
    python entrenar_gcp.py --continuar  # continuar desde checkpoint

SALIDA:
    checkpoints/mejor_modelo.pth
    checkpoints/ultimo_modelo.pth
    checkpoints/log.csv  <- historial de loss por epoch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from PIL import Image
import pandas as pd
import numpy as np
import os
import argparse
import time
import math
import csv

from augmentation_geometrica import (
    aplicar_augmentation_geometrica, kps_dict_desde_row, KP_NAMES as _KP_ORDER
)

KP_NAMES = [
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
    "head", "neck"
]
N_KP = len(KP_NAMES)
assert KP_NAMES == _KP_ORDER, "El orden de keypoints debe coincidir con augmentation_geometrica.py"


class PoseDataset(Dataset):
    def __init__(self, df, frames_dir, transform, augmentar_geometria=False):
        self.df = df.reset_index(drop=True)
        self.frames_dir = frames_dir
        self.transform = transform
        self.augmentar_geometria = augmentar_geometria
        self.kp_cols = [f"{kp}_{c}" for kp in KP_NAMES for c in ["x", "y"]]

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
            # sin augmentation geometrica: las coords no cambian (ya estan
            # normalizadas respecto al crop, que ahora se resizea sin recortar)

        img = self.transform(img)  # solo color/blur/grayscale + ToTensor + Normalize

        kp = torch.tensor(
            [kps_dict[kp][c] for kp in KP_NAMES for c in (0, 1)],
            dtype=torch.float32
        )

        mask_coords = torch.zeros(N_KP * 2)
        mask_kp     = torch.zeros(N_KP)
        for i, kp_name in enumerate(KP_NAMES):
            x, y = kps_dict[kp_name]
            if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
                mask_coords[i*2]   = 1.0
                mask_coords[i*2+1] = 1.0
                mask_kp[i]         = 1.0

        return img, kp, mask_coords, mask_kp


# Modelo
class PoseModel(nn.Module):
    def __init__(self, backbone_name="mobilenetv2_100", n_keypoints=14):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0)
        for param in self.backbone.parameters():
            param.requires_grad = False
        n_features = self.backbone.num_features
        self.shared = nn.Sequential(
            nn.Linear(n_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.head_coords = nn.Sequential(
            nn.Linear(512, n_keypoints * 2),
            nn.Sigmoid()
        )
        self.head_vis = nn.Sequential(
            nn.Linear(512, n_keypoints),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.shared(x)
        return self.head_coords(x), self.head_vis(x)

    def descongelar_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True


# Wing Loss
def wing_loss(pred, target, mask, w=10.0, epsilon=2.0):
    C = w - w * math.log(1 + w / epsilon)
    diff = torch.abs(pred - target)
    loss = torch.where(diff < w, w * torch.log(1 + diff / epsilon), diff - C)
    return (loss * mask).sum() / (mask.sum() + 1e-6)


def entrenar(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs("checkpoints", exist_ok=True)


    train_transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print(f"Cargando datasets...")
    df_train = pd.read_csv(args.train_csv)
    df_val   = pd.read_csv(args.val_csv)
    print(f"  Train: {len(df_train)} | Val: {len(df_val)}")

    frames_train_dir = args.frames if args.frames is not None else args.frames_train
    frames_val_dir   = args.frames if args.frames is not None else args.frames_val

    train_ds = PoseDataset(df_train, frames_train_dir, train_transform, augmentar_geometria=True)
    val_ds   = PoseDataset(df_val,   frames_val_dir,   val_transform,   augmentar_geometria=False)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=4, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                          num_workers=4, pin_memory=True)

    model = PoseModel(backbone_name=args.backbone, n_keypoints=N_KP).to(device)
    print(f"Backbone: {args.backbone}")

    if args.continuar and os.path.exists("checkpoints/mejor_modelo.pth"):
        model.load_state_dict(torch.load("checkpoints/mejor_modelo.pth", map_location=device))
        model.descongelar_backbone()
        print("  Continuando desde checkpoint, backbone descongelado")

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    mejor_val_loss = float("inf")
    log_path = "checkpoints/log.csv"
    epochs_sin_mejora = 0  # contador para early stopping

    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "lr", "tiempo_s"])

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # train
        model.train()
        train_loss = 0
        for imgs, kps, mask_coords, mask_kp in train_dl:
            imgs        = imgs.to(device, non_blocking=True)
            kps         = kps.to(device, non_blocking=True)
            mask_coords = mask_coords.to(device, non_blocking=True)
            mask_kp     = mask_kp.to(device, non_blocking=True)

            optimizer.zero_grad()
            coords, vis = model(imgs)
            loss = wing_loss(coords, kps, mask_coords) + 0.5 * F.binary_cross_entropy(vis, mask_kp)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_dl)

        # val
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, kps, mask_coords, mask_kp in val_dl:
                imgs        = imgs.to(device, non_blocking=True)
                kps         = kps.to(device, non_blocking=True)
                mask_coords = mask_coords.to(device, non_blocking=True)
                mask_kp     = mask_kp.to(device, non_blocking=True)
                coords, vis = model(imgs)
                val_loss += (wing_loss(coords, kps, mask_coords) + 0.5 * F.binary_cross_entropy(vis, mask_kp)).item()
        val_loss /= len(val_dl)

        scheduler.step(val_loss)
        elapsed = time.time() - t0
        lr_actual = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch:3d}/{args.epochs} | train: {train_loss:.4f} | val: {val_loss:.4f} | lr: {lr_actual:.6f} | {elapsed:.1f}s")

        # guardar log
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, round(train_loss, 5), round(val_loss, 5), lr_actual, round(elapsed, 1)])

        # guardar mejor modelo
        if val_loss < mejor_val_loss:
            mejor_val_loss = val_loss
            epochs_sin_mejora = 0
            torch.save(model.state_dict(), "checkpoints/mejor_modelo.pth")
            print(f"  Mejor modelo guardado ({mejor_val_loss:.4f})")
        else:
            epochs_sin_mejora += 1
            print(f"  Sin mejora: {epochs_sin_mejora}/{args.patience}")
            if epochs_sin_mejora >= args.patience:
                print(f"\nEarly stopping en epoch {epoch} — sin mejora por {args.patience} epocas.")
                print(f"Mejor val loss: {mejor_val_loss:.4f}")
                break

        # guardar ultimo
        torch.save(model.state_dict(), "checkpoints/ultimo_modelo.pth")

        # descongelar backbone a mitad
        if epoch == args.epochs // 2:
            print("  Descongelando backbone...")
            model.descongelar_backbone()
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr * 0.1, weight_decay=1e-4)
            epochs_sin_mejora = 0  # resetear contador al descongelar

    print(f"\nEntrenamiento terminado. Mejor val loss: {mejor_val_loss:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", default="dataset_final/train.csv")
    parser.add_argument("--val_csv",   default="dataset_final/val.csv")
    parser.add_argument("--frames_train", default="dataset_final/frames_train",
                         help="carpeta con los crops de train")
    parser.add_argument("--frames_val",   default="dataset_final/frames_val",
                         help="carpeta con los crops de val")
    parser.add_argument("--frames",       default=None,
                         help="(legacy) si se pasa, se usa para train Y val por igual")
    parser.add_argument("--backbone",  default="mobilenetv2_100")
    parser.add_argument("--epochs",    type=int,   default=100)
    parser.add_argument("--batch",     type=int,   default=64)
    parser.add_argument("--lr",        type=float, default=0.001)
    parser.add_argument("--continuar", action="store_true")
    parser.add_argument("--patience",  type=int, default=10,
                        help="epocas sin mejora antes de early stopping")
    args = parser.parse_args()
    entrenar(args)


if __name__ == "__main__":
    main()