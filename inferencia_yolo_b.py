"""
Inferencia en tiempo real con Modelo B (heatmaps).
YOLO detecta personas, el modelo predice heatmaps y extrae el pico de cada uno.

USO:
    python inferencia_yolo_b.py                          # webcam
    python inferencia_yolo_b.py --imagen foto.jpg        # imagen estatica
    python inferencia_yolo_b.py --modelo modelo_b_mejor.pth
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import argparse

KP_NAMES = [
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
    "head", "neck"
]

SKELETON = [
    (12, 13), (13, 0), (13, 1),
    (0, 2), (2, 4), (1, 3), (3, 5),
    (0, 6), (1, 7),
    (6, 8), (8, 10), (7, 9), (9, 11),
]

HM_SIZE = 64
UMBRAL_CONFIANZA = 0.1  # si el pico del heatmap es menor a esto, el keypoint no es visible


# Modelo B
class PoseModelB(nn.Module):
    def __init__(self, n_keypoints=14):
        super().__init__()
        backbone = models.mobilenet_v2(weights=None)
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


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def heatmaps_a_coords(heatmaps):
    """
    Convierte 14 heatmaps a coordenadas normalizadas.
    Devuelve lista de (x_norm, y_norm, confianza) por keypoint.
    """
    coords = []
    for i in range(len(KP_NAMES)):
        hm = heatmaps[i]  # 64x64
        confianza = float(hm.max())
        if confianza < UMBRAL_CONFIANZA:
            coords.append((-1, -1, 0.0))
        else:
            idx = hm.argmax()
            y = int(idx // HM_SIZE)
            x = int(idx % HM_SIZE)
            coords.append((x / HM_SIZE, y / HM_SIZE, confianza))
    return coords


def predecir(modelo, recorte, device):
    img = Image.fromarray(cv2.cvtColor(recorte, cv2.COLOR_BGR2RGB))
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        heatmaps = modelo(tensor)[0].cpu().numpy()
    return heatmaps_a_coords(heatmaps)


def dibujar_persona(frame, coords, x1, y1, x2, y2):
    w_box = x2 - x1
    h_box = y2 - y1

    pts = []
    visible = []
    for x_norm, y_norm, conf in coords:
        if x_norm < 0:
            pts.append((0, 0))
            visible.append(False)
        else:
            kp_x = int(x_norm * w_box + x1)
            kp_y = int(y_norm * h_box + y1)
            pts.append((kp_x, kp_y))
            visible.append(True)

    # bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 100, 255), 2)

    # skeleton
    for a, b in SKELETON:
        if visible[a] and visible[b]:
            cv2.line(frame, pts[a], pts[b], (0, 200, 255), 2)

    # keypoints con color segun confianza
    for i, (x, y) in enumerate(pts):
        if not visible[i]:
            continue
        conf = coords[i][2]
        # verde brillante = alta confianza, amarillo = baja confianza
        color = (0, 255, 0) if conf > 0.5 else (0, 255, 255)
        if KP_NAMES[i] in ("head", "neck"):
            color = (255, 200, 0)
        cv2.circle(frame, (x, y), 5, color, -1)

    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelo",  default="modelo_b_mejor.pth")
    parser.add_argument("--imagen",  default=None)
    parser.add_argument("--conf",    type=float, default=0.5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    print("Cargando YOLO...")
    yolo = YOLO("yolov8n.pt")

    print(f"Cargando modelo B: {args.modelo}...")
    modelo = PoseModelB().to(device)
    modelo.load_state_dict(torch.load(args.modelo, map_location=device))
    modelo.eval()
    print("Listo.\n")

    if args.imagen:
        frame = cv2.imread(args.imagen)
        resultados = yolo(frame, classes=[0], conf=args.conf, verbose=False)
        for box in resultados[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            recorte = frame[y1:y2, x1:x2]
            if recorte.size == 0:
                continue
            coords = predecir(modelo, recorte, device)
            frame = dibujar_persona(frame, coords, x1, y1, x2, y2)
        cv2.imshow("Modelo B - Heatmaps", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        cap = cv2.VideoCapture(0)
        print("Webcam abierta. Presiona Q para salir.")

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            resultados = yolo(frame, classes=[0], conf=args.conf, verbose=False)
            personas = resultados[0].boxes
            n = len(personas)

            for box in personas:
                x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                margen = int(max(x2-x1, y2-y1) * 0.1)
                x1 = max(0, x1 - margen)
                y1 = max(0, y1 - margen)
                x2 = min(frame.shape[1], x2 + margen)
                y2 = min(frame.shape[0], y2 + margen)
                recorte = frame[y1:y2, x1:x2]
                if recorte.size == 0:
                    continue
                coords = predecir(modelo, recorte, device)
                frame = dibujar_persona(frame, coords, x1, y1, x2, y2)

            cv2.putText(frame, f"Modelo B (heatmaps) | Personas: {n}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("Pose Estimation - Modelo B", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()