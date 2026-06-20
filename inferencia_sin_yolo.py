"""
Inferencia directa sin YOLO usando timm backbone.
Pasa el frame completo de la webcam al modelo.

USO:
    python inferencia_sin_yolo.py
    python inferencia_sin_yolo.py --imagen foto.jpg
    python inferencia_sin_yolo.py --modelo mejor_modelo.pth
"""

import torch
import torch.nn as nn
import timm
from torchvision import transforms
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


def predecir(modelo, frame, device):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        coords, vis = modelo(tensor)
    return coords[0].cpu().numpy(), vis[0].cpu().numpy()


def dibujar(frame, coords, vis):
    h, w = frame.shape[:2]
    pts     = []
    visible = []
    for i in range(14):
        x = int(coords[i*2]   * w)
        y = int(coords[i*2+1] * h)
        pts.append((x, y))
        visible.append(float(vis[i]) >= UMBRAL_VIS)

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
        cv2.circle(frame, (x, y), 6, color, -1)
        cv2.putText(frame, KP_NAMES[i], (x+4, y-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelo",   default="mejor_modelo.pth")
    parser.add_argument("--backbone", default="mobilenetv2_100")
    parser.add_argument("--imagen",   default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    modelo = PoseModel(backbone_name=args.backbone).to(device)
    modelo.load_state_dict(torch.load(args.modelo, map_location=device))
    modelo.eval()
    print("Modelo cargado. Presiona Q o Escape para salir.")

    if args.imagen:
        frame = cv2.imread(args.imagen)
        coords, vis = predecir(modelo, frame, device)
        frame = dibujar(frame, coords, vis)
        cv2.imshow("Resultado", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        cap = cv2.VideoCapture(0)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            coords, vis = predecir(modelo, frame, device)
            frame = dibujar(frame, coords, vis)
            cv2.imshow("Pose - sin YOLO", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()