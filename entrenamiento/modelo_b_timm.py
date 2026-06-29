"""
Modelo B (timm) — Pose estimation con heatmaps gaussianos.
Igual al Modelo B original, pero el encoder ahora es intercambiable via timm.

DIFERENCIA CON entrenar_modelo_b.py:
    - El backbone ya no es MobileNetV2 fijo (de torchvision), sino cualquier
      arquitectura de timm, elegible con --backbone.
    - El decoder se adapta automaticamente a los canales que devuelva el
      backbone elegido (feat_ch), en vez de tener 1280 hardcodeado.
    - Todo lo demas (dataset, loss, loop de entrenamiento, scheduler,
      freeze/unfreeze) es identico al original.

INSTALAR:
    pip install timm

LISTAR BACKBONES DISPONIBLES (ejemplos livianos para CPU/tiempo real):
    python -c "import timm; print(timm.list_models('mobilenet*'))"
    python -c "import timm; print(timm.list_models('efficientnet*'))"

USO:
    python entrenar_modelo_b_timm.py --epochs 3 --batch 8 --max_samples 500
    python entrenar_modelo_b_timm.py --backbone mobilenetv3_large_100 --epochs 20
    python entrenar_modelo_b_timm.py --backbone resnet18 --epochs 20
    python entrenar_modelo_b_timm.py --backbone tf_mobilenetv3_small_minimal_100 --epochs 20

SALIDA:
    modelo_b_<backbone>_mejor.pth   (si no se pasa --salida_mejor)
    modelo_b_<backbone>_ultimo.pth  (si no se pasa --salida_ultimo)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
import os
import argparse
import time

try:
    import timm
except ImportError:
    raise ImportError(
        "Falta timm. Instalalo con:  pip install timm"
    )

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


# ── Dataset (identico al original) ──────────────────────────────────
class PoseDatasetB(Dataset):
    def __init__(self, df, frames_dir, transform, sigma=2.0):
        self.df = df.reset_index(drop=True)
        self.frames_dir = frames_dir
        self.transform = transform
        self.sigma = sigma

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
            heatmaps[i] = generar_heatmap(x, y, size=HM_SIZE, sigma=self.sigma)

        return img, torch.tensor(heatmaps)


# ── Modelo B con backbone timm ──────────────────────────────────────
class PoseModelBTimm(nn.Module):
    """
    Encoder: cualquier backbone de timm, en modo features_only
             (devuelve el mapa espacial antes del pooling/clasificacion)
    Decoder: igual al original — upsampling con ConvTranspose2d hasta 64x64
    Output:  14 heatmaps de 64x64

    La diferencia clave con el PoseModelB original es que feat_ch (canales
    que entrega el backbone) se lee automaticamente de timm en vez de estar
    hardcodeado en 1280 (valor especifico de MobileNetV2).
    """
    def __init__(self, n_keypoints=14, backbone_name="mobilenetv3_large_100",
                 pretrained=True, freeze_backbone=True):
        super().__init__()

        # features_only=True: timm devuelve mapas intermedios, no el vector
        # final de clasificacion. out_indices=(-1,) = la ultima etapa,
        # equivalente a lo que antes era backbone.features de MobileNetV2.
        self.encoder = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(-1,),
        )
        feat_ch = self.encoder.feature_info.channels()[-1]
        print(f"  Backbone: {backbone_name} | canales de salida: {feat_ch}")

        if freeze_backbone:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(feat_ch, 256, kernel_size=4, stride=2, padding=1),
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
        feat = self.encoder(x)[-1]    # (B, feat_ch, H, W) — H,W dependen del backbone
        x = self.decoder(feat)
        x = F.interpolate(x, size=(HM_SIZE, HM_SIZE), mode='bilinear', align_corners=False)
        return x


# ── Loss (identica al original) ─────────────────────────────────────
def heatmap_loss(pred, target):
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

    train_ds = PoseDatasetB(df_train, args.frames, train_transform, sigma=args.sigma)
    val_ds   = PoseDatasetB(df_val,   args.frames, val_transform,   sigma=args.sigma)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=0)

    model = PoseModelBTimm(
        n_keypoints=N_KP,
        backbone_name=args.backbone,
        pretrained=True,
        freeze_backbone=True,
    ).to(device)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    mejor_val_loss = float("inf")

    # nombres de salida: si no se pasaron explicitamente, incluyen el backbone
    salida_mejor  = args.salida_mejor  or f"modelo_b_{args.backbone}_mejor.pth"
    salida_ultimo = args.salida_ultimo or f"modelo_b_{args.backbone}_ultimo.pth"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

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
            torch.save(model.state_dict(), salida_mejor)
            print(f"  Mejor modelo guardado ({mejor_val_loss:.5f}) -> {salida_mejor}")

        # descongelar backbone a mitad — igual que el original, pero ahora
        # lr*0.05 en vez de lr*0.1: el backbone preentrenado se mueve mas
        # despacio para no destruir features utiles de ImageNet.
        if epoch == args.epochs // 2:
            print("  Descongelando backbone...")
            for param in model.encoder.parameters():
                param.requires_grad = True
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr * 0.05, weight_decay=1e-4)

    torch.save(model.state_dict(), salida_ultimo)
    print(f"\nEntrenamiento terminado. Mejor val loss: {mejor_val_loss:.5f}")
    print(f"Backbone usado: {args.backbone}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",           default="dataset_final/keypoints.csv")
    parser.add_argument("--frames",        default="dataset_final/frames")
    parser.add_argument("--epochs",        type=int,   default=20)
    parser.add_argument("--batch",         type=int,   default=16)
    parser.add_argument("--lr",            type=float, default=0.001)
    parser.add_argument("--sigma",         type=float, default=2.0,
                         help="Desvio estandar del heatmap gaussiano (antes hardcodeado)")
    parser.add_argument("--backbone",      default="mobilenetv3_large_100",
                         help="Nombre del modelo en timm, ej: mobilenetv2_100, "
                              "mobilenetv3_large_100, resnet18, efficientnet_lite0")
    parser.add_argument("--max_samples",   type=int,   default=None)
    parser.add_argument("--salida_mejor",  default=None)
    parser.add_argument("--salida_ultimo", default=None)
    args = parser.parse_args()
    entrenar(args)


if __name__ == "__main__":
    main()