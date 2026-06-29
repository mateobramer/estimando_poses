"""
Diagnostico visual: muestra si RandomResizedCrop + RandomRotation
desalinean los keypoints respecto a la imagen transformada.

USO (en tu Mac, dentro del proyecto):
    python diagnostico_transforms.py

Por defecto busca:
    dataset_final/train.csv
    dataset_final/frames/

Si tus paths son otros, pasalos:
    python diagnostico_transforms.py --csv dataset_final/train.csv --frames dataset_final/frames

SALIDA:
    diagnostico_transforms.png  <- grilla con 6 ejemplos, original vs transformado
"""

import argparse
import os
import random

import pandas as pd
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from torchvision import transforms

KP_NAMES = [
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
    "head", "neck"
]

# Mismo transform problemático que entrenar_gcp.py / entrenar_gcp_b.py
TRANSFORM_PROBLEMATICO = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomRotation(15),
])

# Transform "fijo" propuesto, sin movimiento geometrico
TRANSFORM_FIJO = transforms.Compose([
    transforms.Resize((224, 224)),
])


def dibujar_keypoints(img, row, color="lime", radius=4):
    """Dibuja los keypoints (coordenadas normalizadas 0-1) sobre la imagen tal cual está."""
    img = img.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    for kp in KP_NAMES:
        x = float(row[f"{kp}_x"])
        y = float(row[f"{kp}_y"])
        if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
            px, py = x * w, y * h
            draw.ellipse(
                [px - radius, py - radius, px + radius, py + radius],
                fill=color, outline="black"
            )
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="dataset_final/train.csv")
    parser.add_argument("--frames", default="dataset_final/frames")
    parser.add_argument("--n", type=int, default=6, help="cantidad de ejemplos a mostrar")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"No encuentro {args.csv}. Pasá --csv y --frames correctos.")
        return

    df = pd.read_csv(args.csv)
    random.seed(args.seed)
    idxs = random.sample(range(len(df)), min(args.n, len(df)))

    fig, axes = plt.subplots(len(idxs), 3, figsize=(12, 4 * len(idxs)))
    if len(idxs) == 1:
        axes = axes.reshape(1, -1)

    for row_i, idx in enumerate(idxs):
        row = df.iloc[idx]
        path = os.path.join(args.frames, row["imagen"])
        img = Image.open(path).convert("RGB")

        # 1. Original con keypoints (referencia: acá DEBEN calzar bien)
        img_orig_kp = dibujar_keypoints(img, row, color="lime")

        # 2. Imagen pasada por el transform problemático,
        #    con los keypoints ORIGINALES dibujados encima
        #    (asi se entrena hoy: el target no se mueve con la imagen)
        img_bug = TRANSFORM_PROBLEMATICO(img)
        img_bug_kp = dibujar_keypoints(img_bug, row, color="red")

        # 3. Imagen con el transform fijo (Resize sin crop/rotate aleatorio),
        #    los keypoints siguen siendo validos porque no hubo movimiento geometrico
        img_fix = TRANSFORM_FIJO(img)
        img_fix_kp = dibujar_keypoints(img_fix, row, color="lime")

        axes[row_i, 0].imshow(img_orig_kp)
        axes[row_i, 0].set_title("Original + keypoints (referencia)")
        axes[row_i, 0].axis("off")

        axes[row_i, 1].imshow(img_bug_kp)
        axes[row_i, 1].set_title("RandomResizedCrop+Rotation\n+ keypoints SIN mover (bug)")
        axes[row_i, 1].axis("off")

        axes[row_i, 2].imshow(img_fix_kp)
        axes[row_i, 2].set_title("Resize fijo + keypoints (correcto)")
        axes[row_i, 2].axis("off")

    plt.tight_layout()
    out_path = "diagnostico_transforms.png"
    plt.savefig(out_path, dpi=120)
    print(f"Guardado: {out_path}")
    print("\nMirá la columna del medio (roja): si los puntos NO caen sobre")
    print("hombros/codos/rodillas reales, confirma el bug.")
    print("La columna derecha (verde) es como debería verse siempre.")


if __name__ == "__main__":
    main()