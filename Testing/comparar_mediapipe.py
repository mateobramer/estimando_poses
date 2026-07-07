"""
Compara PCK del modelo propio vs MediaPipe sobre el test set.

USO:
    python comparar_mediapipe.py
    python comparar_mediapipe.py --modelo mejor_modelo_v2.pth --csv dataset_final/test.csv
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
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import cv2

KP_NAMES = [
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
    "head", "neck"
]

# mapeo MediaPipe → nuestros keypoints
MP_TO_KP = {
    "left_shoulder":  11,
    "right_shoulder": 12,
    "left_elbow":     13,
    "right_elbow":    14,
    "left_wrist":     15,
    "right_wrist":    16,
    "left_hip":       23,
    "right_hip":      24,
    "left_knee":      25,
    "right_knee":     26,
    "left_ankle":     27,
    "right_ankle":    28,
    "head":           0,
    "neck":           None,  # promedio hombros
}

N_KP = len(KP_NAMES)


class PoseModel(nn.Module):
    def __init__(self, backbone_name="mobilenetv2_100", n_keypoints=14):
        super().__init__()
        self.backbone    = timm.create_model(backbone_name, pretrained=False, num_classes=0)
        n_features       = self.backbone.num_features
        self.shared      = nn.Sequential(nn.Linear(n_features, 512), nn.ReLU(), nn.Dropout(0.3))
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


def descargar_modelo_mp(model_path="pose_landmarker_full.task"):
    if not os.path.exists(model_path):
        print("Bajando modelo de MediaPipe...")
        url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
        urllib.request.urlretrieve(url, model_path)
        print("Listo.")
    return model_path


def crear_detector_mp(model_path):
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False
    )
    return vision.PoseLandmarker.create_from_options(options)


def predecir_mediapipe(detector, img_path):
    """Devuelve coordenadas normalizadas de los 14 keypoints con MediaPipe."""
    img_mp = mp.Image.create_from_file(img_path)
    result = detector.detect(img_mp)

    coords = np.full(N_KP * 2, -1.0, dtype=np.float32)

    if not result.pose_landmarks:
        return coords

    landmarks = result.pose_landmarks[0]

    for i, kp in enumerate(KP_NAMES):
        idx = MP_TO_KP[kp]
        if kp == "neck":
            ls = landmarks[11]
            rs = landmarks[12]
            x = (ls.x + rs.x) / 2
            y = (ls.y + rs.y) / 2
            vis = (ls.visibility + rs.visibility) / 2
        else:
            lm = landmarks[idx]
            x, y, vis = lm.x, lm.y, lm.visibility

        if vis > 0.3 and 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
            coords[i*2]   = x
            coords[i*2+1] = y

    return coords


def evaluar_modelo(model, df, frames_dir, device, pck_umbral, nombre):
    """Evalúa el modelo propio y devuelve correctos, totales, mae."""
    correctos = np.zeros(N_KP)
    totales   = np.zeros(N_KP)
    mae_sum   = np.zeros(N_KP)

    for idx, (_, row) in enumerate(df.iterrows()):
        if idx % 200 == 0:
            print(f"  [{nombre}] {idx}/{len(df)}...", flush=True)

        path = os.path.join(frames_dir, row["imagen"])
        if not os.path.exists(path):
            continue

        img = Image.open(path).convert("RGB")
        w, h = img.size
        tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            coords, _ = model(tensor)
        coords = coords[0].cpu().numpy()

        umbral = pck_umbral * h

        for i, kp in enumerate(KP_NAMES):
            gt_x = float(row[f"{kp}_x"])
            gt_y = float(row[f"{kp}_y"])
            if not (0.0 <= gt_x <= 1.0 and 0.0 <= gt_y <= 1.0):
                continue
            pred_x = coords[i*2]
            pred_y = coords[i*2+1]
            dist = np.sqrt(((pred_x - gt_x) * w)**2 + ((pred_y - gt_y) * h)**2)
            mae_sum[i]   += dist
            totales[i]   += 1
            if dist <= umbral:
                correctos[i] += 1

    return correctos, totales, mae_sum


def evaluar_mediapipe(detector, df, frames_dir, pck_umbral):
    """Evalúa MediaPipe y devuelve correctos, totales, mae."""
    correctos = np.zeros(N_KP)
    totales   = np.zeros(N_KP)
    mae_sum   = np.zeros(N_KP)

    for idx, (_, row) in enumerate(df.iterrows()):
        if idx % 200 == 0:
            print(f"  [MediaPipe] {idx}/{len(df)}...", flush=True)

        path = os.path.join(frames_dir, row["imagen"])
        if not os.path.exists(path):
            continue

        img = Image.open(path).convert("RGB")
        w, h = img.size

        coords = predecir_mediapipe(detector, path)
        umbral = pck_umbral * h

        for i, kp in enumerate(KP_NAMES):
            gt_x = float(row[f"{kp}_x"])
            gt_y = float(row[f"{kp}_y"])
            if not (0.0 <= gt_x <= 1.0 and 0.0 <= gt_y <= 1.0):
                continue
            pred_x = coords[i*2]
            pred_y = coords[i*2+1]
            if pred_x < 0:
                continue
            dist = np.sqrt(((pred_x - gt_x) * w)**2 + ((pred_y - gt_y) * h)**2)
            mae_sum[i]   += dist
            totales[i]   += 1
            if dist <= umbral:
                correctos[i] += 1

    return correctos, totales, mae_sum


def imprimir_resultados(nombre, correctos, totales, mae_sum):
    print(f"\n--- {nombre} ---")
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
    return pck_global


def imprimir_comparacion(c_modelo, t_modelo, c_mp, t_mp):
    print(f"\n--- COMPARACION ---")
    print(f"{'Keypoint':<20} {'Nuestro':>10} {'MediaPipe':>12} {'Diferencia':>12}")
    print("-" * 58)
    for i, kp in enumerate(KP_NAMES):
        if t_modelo[i] == 0:
            continue
        pck_m  = 100 * c_modelo[i] / t_modelo[i]
        pck_mp = 100 * c_mp[i]     / t_mp[i] if t_mp[i] > 0 else 0
        diff   = pck_m - pck_mp
        signo  = "+" if diff >= 0 else ""
        print(f"{kp:<20} {pck_m:>9.1f}% {pck_mp:>11.1f}% {signo}{diff:>10.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelo",     default="mejor_modelo_v2.pth")
    parser.add_argument("--backbone",   default="mobilenetv2_100")
    parser.add_argument("--csv",        default="dataset_final/test.csv")
    parser.add_argument("--frames",     default="dataset_final/frames")
    parser.add_argument("--pck_umbral", type=float, default=0.2)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limitar a N muestras para prueba rapida")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    print(f"Cargando modelo: {args.modelo}...")
    model = PoseModel(backbone_name=args.backbone).to(device)
    model.load_state_dict(torch.load(args.modelo, map_location=device))
    model.eval()

    print("Cargando MediaPipe...")
    model_path = descargar_modelo_mp()
    detector   = crear_detector_mp(model_path)

    print(f"Cargando test set: {args.csv}...")
    df = pd.read_csv(args.csv)
    if args.max_samples:
        df = df.sample(args.max_samples, random_state=42)
    print(f"  {len(df)} frames\n")

    # evaluar modelo propio
    c_modelo, t_modelo, mae_modelo = evaluar_modelo(
        model, df, args.frames, device, args.pck_umbral, "Nuestro modelo"
    )

    # evaluar mediapipe
    c_mp, t_mp, mae_mp = evaluar_mediapipe(
        detector, df, args.frames, args.pck_umbral
    )

    # imprimir resultados
    pck_nuestro = imprimir_resultados("Nuestro modelo", c_modelo, t_modelo, mae_modelo)
    pck_mp      = imprimir_resultados("MediaPipe",      c_mp,     t_mp,     mae_mp)
    imprimir_comparacion(c_modelo, t_modelo, c_mp, t_mp)

    print(f"\nResumen: Nuestro modelo {pck_nuestro:.1f}% vs MediaPipe {pck_mp:.1f}%")
    diff = pck_nuestro - pck_mp
    if diff > 0:
        print(f"Nuestro modelo gana por {diff:.1f}%")
    else:
        print(f"MediaPipe gana por {abs(diff):.1f}%")


if __name__ == "__main__":
    main()