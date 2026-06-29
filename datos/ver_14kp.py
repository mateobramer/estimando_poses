"""
Viewer para ver los frames con los 14 keypoints dibujados encima.
Navega con ← → o A D.

USO:
    python ver_14kp.py
    python ver_14kp.py --csv kaggle_14kp.csv --frames kaggle_muy_validos
    python ver_14kp.py --csv propio_14kp.csv --frames dataset_propio/frames
"""

import tkinter as tk
from PIL import Image, ImageTk
import pandas as pd
import cv2
import os
import argparse
import random

# conexiones del esqueleto (índices en la lista de 14 keypoints)
KP_NAMES = [
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
    "head", "neck"
]

SKELETON = [
    (12, 13),  # head - neck
    (13, 0),   # neck - left_shoulder
    (13, 1),   # neck - right_shoulder
    (0, 2),    # left_shoulder - left_elbow
    (2, 4),    # left_elbow - left_wrist
    (1, 3),    # right_shoulder - right_elbow
    (3, 5),    # right_elbow - right_wrist
    (0, 6),    # left_shoulder - left_hip
    (1, 7),    # right_shoulder - right_hip
    (6, 8),    # left_hip - left_knee
    (8, 10),   # left_knee - left_ankle
    (7, 9),    # right_hip - right_knee
    (9, 11),   # right_knee - right_ankle
]

COLORS = {
    "head":  (0, 255, 255),
    "neck":  (0, 200, 255),
    "default": (0, 255, 0)
}


def dibujar_keypoints(img, row):
    h, w = img.shape[:2]
    pts = []
    visible = []
    for kp in KP_NAMES:
        x_norm = float(row[f"{kp}_x"])
        y_norm = float(row[f"{kp}_y"])
        en_imagen = 0.0 <= x_norm <= 1.0 and 0.0 <= y_norm <= 1.0
        pts.append((int(x_norm * w), int(y_norm * h)))
        visible.append(en_imagen)

    # skeleton — solo si ambos extremos son visibles
    for a, b in SKELETON:
        if visible[a] and visible[b]:
            cv2.line(img, pts[a], pts[b], (0, 200, 255), 2)

    # puntos — solo los visibles
    for i, (x, y) in enumerate(pts):
        if not visible[i]:
            continue
        color = COLORS.get(KP_NAMES[i], COLORS["default"])
        cv2.circle(img, (x, y), 5, color, -1)
        cv2.putText(img, KP_NAMES[i], (x+4, y-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",    default=None)
    parser.add_argument("--frames", default=None)
    args = parser.parse_args()

    # si no se pasan argumentos, combinar los dos datasets
    if args.csv is None:
        dfs = []
        if os.path.exists("kaggle_14kp.csv"):
            df_k = pd.read_csv("kaggle_14kp.csv")
            df_k["_frames_dir"] = "kaggle_muy_validos"
            dfs.append(df_k)
        if os.path.exists("propio_14kp.csv"):
            df_p = pd.read_csv("propio_14kp.csv")
            df_p["_frames_dir"] = "dataset_propio/frames"
            dfs.append(df_p)
        df = pd.concat(dfs).reset_index(drop=True)
        df = df.sample(frac=1).reset_index(drop=True)  # shuffle
    else:
        df = pd.read_csv(args.csv)
        df["_frames_dir"] = args.frames

    print(f"{len(df)} frames para ver")

    estado = {"idx": 0}

    root = tk.Tk()
    root.title("Viewer 14 Keypoints")
    root.configure(bg="#1a1a2e")
    root.geometry("900x640")

    label_img = tk.Label(root, bg="#1a1a2e")
    label_img.pack(pady=10)

    label_info = tk.Label(root, text="", bg="#1a1a2e", fg="white", font=("Arial", 11))
    label_info.pack()

    label_nombre = tk.Label(root, text="", bg="#1a1a2e", fg="#888", font=("Arial", 9))
    label_nombre.pack()

    def mostrar(idx):
        row = df.iloc[idx]
        frames_dir = row["_frames_dir"]
        path = os.path.join(frames_dir, row["imagen"])

        img = cv2.imread(path)
        if img is None:
            label_info.configure(text=f"No encontrado: {path}")
            return

        img = dibujar_keypoints(img, row)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(img_rgb)
        pil.thumbnail((860, 520))
        foto = ImageTk.PhotoImage(pil)
        label_img.configure(image=foto)
        label_img.image = foto
        label_info.configure(text=f"Frame {idx+1}/{len(df)}  |  {frames_dir}")
        label_nombre.configure(text=row["imagen"])

    def siguiente(e=None):
        if estado["idx"] < len(df) - 1:
            estado["idx"] += 1
            mostrar(estado["idx"])

    def anterior(e=None):
        if estado["idx"] > 0:
            estado["idx"] -= 1
            mostrar(estado["idx"])

    frame_btn = tk.Frame(root, bg="#1a1a2e")
    frame_btn.pack(pady=10)

    tk.Button(frame_btn, text="← Anterior [A]", bg="#555", fg="white",
              font=("Arial", 12), width=14, height=2, command=anterior
              ).grid(row=0, column=0, padx=20)

    tk.Button(frame_btn, text="Siguiente → [D]", bg="#2980b9", fg="white",
              font=("Arial", 12), width=14, height=2, command=siguiente
              ).grid(row=0, column=1, padx=20)

    root.bind("d", siguiente)
    root.bind("D", siguiente)
    root.bind("a", anterior)
    root.bind("A", anterior)
    root.bind("<Right>", siguiente)
    root.bind("<Left>", anterior)

    mostrar(0)
    root.mainloop()


if __name__ == "__main__":
    main()