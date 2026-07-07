"""
Evalua el modelo con PCK (Percentage of Correct Keypoints) sobre el test set.
PCK@0.2: un keypoint es correcto si esta a menos del 20% del alto del bbox.

USO:
    python evaluar_pck.py
    python evaluar_pck.py --modelo mejor_modelo.pth --csv dataset_final/test.csv
"""

import torch
import torch.nn as nn
import timm
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
import os
import argparse

KP_NAMES = [
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
    "head", "neck"
]
N_KP = len(KP_NAMES)


class PoseModel(nn.Module):
    def __init__(self, backbone_name="mobilenetv2_100", n_keypoints=14):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0)
        n_features = self.backbone.num_features
        self.shared = nn.Sequential(
            nn.Linear(n_features, 512), nn.ReLU(), nn.Dropout(0.3)
        )
        self.head_coords = nn.Sequential(nn.Linear(512, n_keypoints * 2), nn.Sigmoid())
        self.head_vis    = nn.Sequential(nn.Linear(512, n_keypoints),     nn.Sigmoid())

    def forward(self, x):
        x = self.backbone(x)
        x = self.shared(x)
        return self.head_coords(x), self.head_vis(x)


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def evaluar(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    print(f"Cargando modelo: {args.modelo}")
    model = PoseModel(backbone_name=args.backbone).to(device)
    model.load_state_dict(torch.load(args.modelo, map_location=device))
    model.eval()

    print(f"Cargando test set: {args.csv}")
    df = pd.read_csv(args.csv)
    print(f"  {len(df)} frames\n")

    correctos = np.zeros(N_KP)
    totales   = np.zeros(N_KP)
    mae_sum   = np.zeros(N_KP)

    for idx, (_, row) in enumerate(df.iterrows()):
        if idx % 200 == 0:
            print(f"  {idx}/{len(df)} frames procesados...", flush=True)

        path = os.path.join(args.frames, row["imagen"])
        if not os.path.exists(path):
            continue

        img = Image.open(path).convert("RGB")
        w, h = img.size
        tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            coords, vis = model(tensor)

        coords = coords[0].cpu().numpy()
        umbral = args.pck_umbral * h

        for i, kp in enumerate(KP_NAMES):
            gt_x = float(row[f"{kp}_x"])
            gt_y = float(row[f"{kp}_y"])

            if not (0.0 <= gt_x <= 1.0 and 0.0 <= gt_y <= 1.0):
                continue

            pred_x = coords[i*2]
            pred_y = coords[i*2+1]

            dist = np.sqrt(((pred_x - gt_x) * w)**2 + ((pred_y - gt_y) * h)**2)
            mae_sum[i] += dist
            totales[i] += 1
            if dist <= umbral:
                correctos[i] += 1

    print(f"\n--- PCK@{args.pck_umbral} ---")
    print(f"{'Keypoint':<20} {'PCK':>8} {'MAE (px)':>10} {'N':>6}")
    print("-" * 50)

    pck_total = 0
    n_total   = 0
    for i, kp in enumerate(KP_NAMES):
        if totales[i] == 0:
            continue
        pck = 100 * correctos[i] / totales[i]
        mae = mae_sum[i] / totales[i]
        print(f"{kp:<20} {pck:>7.1f}% {mae:>10.1f}px {int(totales[i]):>6}")
        pck_total += correctos[i]
        n_total   += totales[i]

    pck_global = 100 * pck_total / n_total if n_total > 0 else 0
    print("-" * 50)
    print(f"{'TOTAL':<20} {pck_global:>7.1f}%")
    print(f"\nPCK@{args.pck_umbral} global: {pck_global:.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelo",     default="mejor_modelo.pth")
    parser.add_argument("--backbone",   default="mobilenetv2_100")
    parser.add_argument("--csv",        default="dataset_final/test.csv")
    parser.add_argument("--frames",     default="dataset_final/frames")
    parser.add_argument("--pck_umbral", type=float, default=0.2)
    args = parser.parse_args()
    evaluar(args)


if __name__ == "__main__":
    main()