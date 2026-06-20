"""
Inferencia en tiempo real con Modelo A (timm backbone).
YOLO detecta personas, el modelo predice keypoints y su confianza.

USO:
    python inferencia_yolo.py                          # webcam
    python inferencia_yolo.py --imagen foto.jpg
    python inferencia_yolo.py --modelo mejor_modelo.pth
"""

import torch
import torch.nn as nn
import timm
from torchvision import transforms
from ultralytics import YOLO
from PIL import Image
import cv2
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

UMBRAL_VIS = 0.5


class PoseModel(nn.Module):
    def __init__(self, backbone_name="mobilenetv2_100", n_keypoints=14):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0)
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


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def predecir(modelo, recorte, device):
    img = Image.fromarray(cv2.cvtColor(recorte, cv2.COLOR_BGR2RGB))
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        coords, vis = modelo(tensor)
    return coords[0].cpu().numpy(), vis[0].cpu().numpy()


def dibujar_persona(frame, coords, vis, x1, y1, x2, y2):
    w_box = x2 - x1
    h_box = y2 - y1
    pts     = []
    visible = []
    for i in range(14):
        kp_x = int(coords[i*2]   * w_box + x1)
        kp_y = int(coords[i*2+1] * h_box + y1)
        pts.append((kp_x, kp_y))
        visible.append(float(vis[i]) >= UMBRAL_VIS)

    cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 100, 255), 2)

    for a, b in SKELETON:
        if visible[a] and visible[b]:
            cv2.line(frame, pts[a], pts[b], (0, 200, 255), 2)

    for i, (x, y) in enumerate(pts):
        if not visible[i]:
            continue
        conf = float(vis[i])
        color = (0, 255, 0) if conf > 0.8 else (0, 255, 255)
        if KP_NAMES[i] in ("head", "neck"):
            color = (255, 200, 0)
        cv2.circle(frame, (x, y), 5, color, -1)

    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelo",   default="mejor_modelo.pth")
    parser.add_argument("--backbone", default="mobilenetv2_100")
    parser.add_argument("--imagen",   default=None)
    parser.add_argument("--conf",     type=float, default=0.5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    print("Cargando YOLO...")
    yolo = YOLO("yolov8n.pt")

    print(f"Cargando modelo: {args.modelo}...")
    modelo = PoseModel(backbone_name=args.backbone).to(device)
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
            coords, vis = predecir(modelo, recorte, device)
            frame = dibujar_persona(frame, coords, vis, x1, y1, x2, y2)
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
            n = len(resultados[0].boxes)
            for box in resultados[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                margen = int(max(x2-x1, y2-y1) * 0.1)
                x1 = max(0, x1 - margen)
                y1 = max(0, y1 - margen)
                x2 = min(frame.shape[1], x2 + margen)
                y2 = min(frame.shape[0], y2 + margen)
                recorte = frame[y1:y2, x1:x2]
                if recorte.size == 0:
                    continue
                coords, vis = predecir(modelo, recorte, device)
                frame = dibujar_persona(frame, coords, vis, x1, y1, x2, y2)
            cv2.putText(frame, f"Personas: {n}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Pose Estimation", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()