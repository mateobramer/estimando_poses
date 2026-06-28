"""
inferencia_yolo.py
==================
Loop principal. Orquesta YOLO + modelo de pose + One Euro Filter +
análisis de ángulos + visualización.

USO:
    cd ~/Documents/GitHub/estimando_poses
    python3 inferencia_yolo.py
    python3 inferencia_yolo.py --modelo modelo_a_fix_mejor.pth

CONTROLES durante el ejercicio:
    Q / ESC  → salir
    R        → resetear contador de reps
    F        → toggle One Euro Filter on/off
    M        → volver al menú de selección
    1-5      → cambiar ejercicio en caliente
"""

import torch
import torch.nn as nn
import timm
from torchvision import transforms
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import argparse

from one_euro_filter import OneEuroFilterArray
from analizador import Analizador, calcular_angulo
from angulos_referencia import EJERCICIOS
from menu import mostrar_menu
from visualizador import Visualizador

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

KP_NAMES = [
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist",    "right_wrist",    "left_hip",   "right_hip",
    "left_knee",     "right_knee",     "left_ankle", "right_ankle",
    "head",          "neck",
]
N_KP           = len(KP_NAMES)
EJERCICIO_KEYS = list(EJERCICIOS.keys())
C_DIM          = (100, 100, 100)

# ---------------------------------------------------------------------------
# Modelo — copia exacta de entrenar_gcp.py
# ---------------------------------------------------------------------------

class PoseModel(nn.Module):
    def __init__(self, backbone_name="mobilenetv2_100", n_keypoints=14):
        super().__init__()
        self.backbone    = timm.create_model(backbone_name, pretrained=False, num_classes=0)
        n_features       = self.backbone.num_features
        self.shared      = nn.Sequential(
            nn.Linear(n_features, 512), nn.ReLU(), nn.Dropout(0.3))
        self.head_coords = nn.Sequential(
            nn.Linear(512, n_keypoints * 2), nn.Sigmoid())
        self.head_vis    = nn.Sequential(
            nn.Linear(512, n_keypoints), nn.Sigmoid())

    def forward(self, x):
        x = self.backbone(x)
        x = self.shared(x)
        return self.head_coords(x), self.head_vis(x)


_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ---------------------------------------------------------------------------
# Funciones de inferencia
# ---------------------------------------------------------------------------

def predecir(modelo, recorte, device):
    img    = Image.fromarray(cv2.cvtColor(recorte, cv2.COLOR_BGR2RGB))
    tensor = _transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        coords, vis = modelo(tensor)
    return coords[0].cpu().numpy(), vis[0].cpu().numpy()


def coords_a_pixeles(coords, vis, x1, y1, x2, y2, umbral_vis):
    w, h = x2 - x1, y2 - y1
    kps  = np.full((N_KP, 2), -1.0, dtype=np.float32)
    for i in range(N_KP):
        if float(vis[i]) >= umbral_vis:
            kps[i, 0] = coords[i*2]   * w + x1
            kps[i, 1] = coords[i*2+1] * h + y1
    return kps


def determinar_fase(keypoints, ejercicio):
    tracking   = EJERCICIOS[ejercicio].get("rep_tracking", {})
    art_nombre = tracking.get("articulacion", "rodilla_izq")
    art_config = EJERCICIOS[ejercicio]["articulaciones"].get(art_nombre, {})
    kps_idx    = art_config.get("keypoints")
    if kps_idx is None:
        return "baja"
    a, b, c = keypoints[kps_idx[0]], keypoints[kps_idx[1]], keypoints[kps_idx[2]]
    ang     = calcular_angulo(a, b, c)
    umbral  = tracking.get("umbral_inicio", 120)
    return "baja" if 0 < ang < umbral else "alta"

# ---------------------------------------------------------------------------
# Loop de video
# ---------------------------------------------------------------------------

def loop_video(args, modelo, yolo, device, ejercicio_inicial: str) -> str | None:
    """
    Corre el loop de captura y análisis.

    Retorna
    -------
    str  — ejercicio actual si el usuario presionó M (volver al menú).
    None — si el usuario salió con Q/ESC.
    """
    analizador    = Analizador(ejercicio_inicial)
    filtro        = OneEuroFilterArray(size=N_KP*2, freq=30.0,
                                       mincutoff=1.0, beta=0.05)
    filtro_activo = True
    viz           = Visualizador()

    cap = cv2.VideoCapture(args.camara)
    if not cap.isOpened():
        print(f"No se pudo abrir la cámara {args.camara}")
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print(f"Ejercicio: {analizador.ejercicio}")

    pedir_menu = False

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        resultados = yolo(frame, classes=[0], conf=args.conf, verbose=False)
        boxes      = resultados[0].boxes
        n_personas = len(boxes)
        analisis   = None

        # Primera persona — analizar
        for box in boxes[:1]:
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            margen = int(max(x2-x1, y2-y1) * 0.1)
            x1 = max(0, x1-margen);  y1 = max(0, y1-margen)
            x2 = min(frame.shape[1], x2+margen)
            y2 = min(frame.shape[0], y2+margen)

            recorte = frame[y1:y2, x1:x2]
            if recorte.size == 0:
                continue

            coords, vis = predecir(modelo, recorte, device)
            kps = coords_a_pixeles(coords, vis, x1, y1, x2, y2,
                                   args.umbral_vis)

            if filtro_activo:
                kps = filtro.apply(kps.flatten()).reshape(N_KP, 2)

            fase     = determinar_fase(kps, analizador.ejercicio)
            analisis = analizador.analizar(kps, fase)

            viz.dibujar(frame, kps, vis, analisis, filtro_activo, n_personas)

        # Resto de personas — solo esqueleto
        for box in boxes[1:]:
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            recorte = frame[y1:y2, x1:x2]
            if recorte.size == 0:
                continue
            coords, vis = predecir(modelo, recorte, device)
            kps = coords_a_pixeles(coords, vis, x1, y1, x2, y2,
                                   args.umbral_vis)
            viz._dibujar_skeleton(frame, kps, vis)

        if analisis is None:
            cv2.putText(frame, "Posicionáte frente a la cámara",
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, C_DIM, 2, cv2.LINE_AA)

        cv2.imshow("Pose Analysis", frame)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):
            break
        elif key == ord("r"):
            analizador.reset_reps()
            viz.limpiar_graficos()
            print("Reps reseteadas")
        elif key == ord("f"):
            filtro_activo = not filtro_activo
            filtro.reset()
            print(f"Filtro: {'ON' if filtro_activo else 'OFF'}")
        elif key == ord("m"):
            pedir_menu = True
            break
        elif ord("1") <= key <= ord("5"):
            idx = key - ord("1")
            if idx < len(EJERCICIO_KEYS):
                nuevo = EJERCICIO_KEYS[idx]
                analizador.cambiar_ejercicio(nuevo)
                filtro.reset()
                viz.limpiar_graficos()
                print(f"Ejercicio: {nuevo}")

    cap.release()
    cv2.destroyAllWindows()
    return analizador.ejercicio if pedir_menu else None

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelo",     default="modelo_a_fix_mejor.pth")
    parser.add_argument("--backbone",   default="mobilenetv2_100")
    parser.add_argument("--camara",     type=int,   default=0)
    parser.add_argument("--conf",       type=float, default=0.5)
    parser.add_argument("--umbral_vis", type=float, default=0.3)
    args = parser.parse_args()

    # Cargar modelos una sola vez
    device = torch.device("mps"  if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    print("Cargando YOLO...")
    yolo = YOLO("yolov8n.pt")

    print(f"Cargando modelo de pose: {args.modelo}...")
    modelo = PoseModel(backbone_name=args.backbone).to(device)
    modelo.load_state_dict(torch.load(args.modelo, map_location=device))
    modelo.eval()
    print("Listo.\n")

    ejercicio = args.ejercicio if hasattr(args, "ejercicio") else "sentadilla"

    # Ciclo menú → video → menú
    while True:
        elegido = mostrar_menu(ejercicio)
        if elegido is None:
            print("Saliendo.")
            break

        ejercicio = elegido
        print(f"Ejercicio: {ejercicio}")

        resultado = loop_video(args, modelo, yolo, device, ejercicio)

        if resultado is None:
            break          # Q/ESC → salir
        else:
            ejercicio = resultado  # M → volver al menú


if __name__ == "__main__":
    main()