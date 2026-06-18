"""
Inferencia en tiempo real: YOLO detecta personas, tu modelo predice keypoints.

USO:
    python inferencia_yolo.py                          # webcam
    python inferencia_yolo.py --imagen foto.jpg        # imagen estatica
    python inferencia_yolo.py --modelo modelo_a_mejor.pth
"""

import torch
import torch.nn as nn
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


class PoseModelA(nn.Module):
    def __init__(self, n_keypoints=14):
        super().__init__()
        backbone = models.mobilenet_v2(weights=None)
        self.backbone = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, n_keypoints * 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        return self.head(x)


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def predecir_keypoints(modelo, recorte, device):
    img = Image.fromarray(cv2.cvtColor(recorte, cv2.COLOR_BGR2RGB))
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = modelo(tensor)[0].cpu().numpy()
    return pred


def dibujar_persona(frame, pred, x1, y1, x2, y2):
    w_box = x2 - x1
    h_box = y2 - y1

    # convertir coordenadas del recorte al frame original
    pts = []
    visible = []
    for i in range(14):
        kp_x = int(pred[i*2]   * w_box + x1)
        kp_y = int(pred[i*2+1] * h_box + y1)
        en_frame = (x1 <= kp_x <= x2) and (y1 <= kp_y <= y2)
        pts.append((kp_x, kp_y))
        visible.append(en_frame)

    # dibujar bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 100, 255), 2)

    # skeleton
    for a, b in SKELETON:
        if visible[a] and visible[b]:
            cv2.line(frame, pts[a], pts[b], (0, 200, 255), 2)

    # keypoints
    for i, (x, y) in enumerate(pts):
        if not visible[i]:
            continue
        color = (0, 255, 255) if KP_NAMES[i] in ("head", "neck") else (0, 255, 0)
        cv2.circle(frame, (x, y), 5, color, -1)

    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelo",  default="modelo_a_mejor.pth")
    parser.add_argument("--imagen",  default=None)
    parser.add_argument("--conf",    type=float, default=0.5, help="Confianza minima YOLO")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    # cargar modelos
    print("Cargando YOLO...")
    yolo = YOLO("yolov8n.pt")

    print(f"Cargando modelo de pose: {args.modelo}...")
    modelo = PoseModelA().to(device)
    modelo.load_state_dict(torch.load(args.modelo, map_location=device))
    modelo.eval()
    print("Listo.\n")

    if args.imagen:
        frame = cv2.imread(args.imagen)
        if frame is None:
            print(f"No se encontro {args.imagen}")
            return

        resultados = yolo(frame, classes=[0], conf=args.conf, verbose=False)
        for box in resultados[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            recorte = frame[y1:y2, x1:x2]
            if recorte.size == 0:
                continue
            pred = predecir_keypoints(modelo, recorte, device)
            frame = dibujar_persona(frame, pred, x1, y1, x2, y2)

        cv2.imshow("Resultado", frame)
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
                # agregar margen
                margen = int(max(x2-x1, y2-y1) * 0.1)
                x1 = max(0, x1 - margen)
                y1 = max(0, y1 - margen)
                x2 = min(frame.shape[1], x2 + margen)
                y2 = min(frame.shape[0], y2 + margen)
                recorte = frame[y1:y2, x1:x2]
                if recorte.size == 0:
                    continue
                pred = predecir_keypoints(modelo, recorte, device)
                frame = dibujar_persona(frame, pred, x1, y1, x2, y2)

            cv2.putText(frame, f"Personas: {n}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Pose Estimation", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()