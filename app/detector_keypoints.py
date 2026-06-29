"""
detector_keypoints.py
=====================
Recibe un frame de video y devuelve los 14 keypoints del modelo propio.

Interfaz pública
----------------
  detector = DetectorKeypoints("modelo_a_fix_mejor.pth")
  resultado = detector.detectar(frame)
  # resultado["keypoints"]  → np.ndarray (14, 2) en píxeles
  # resultado["visibilidad"] → np.ndarray (14,)  scores [0,1]

Orden de keypoints (igual que en entrenar_gcp.py)
--------------------------------------------------
  0  left_shoulder    1  right_shoulder
  2  left_elbow       3  right_elbow
  4  left_wrist       5  right_wrist
  6  left_hip         7  right_hip
  8  left_knee        9  right_knee
  10 left_ankle       11 right_ankle
  12 head             13 neck
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import timm
from PIL import Image
import cv2


# ---------------------------------------------------------------------------
# Constantes — deben coincidir exactamente con entrenar_gcp.py
# ---------------------------------------------------------------------------

KP_NAMES = [
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist",    "right_wrist",    "left_hip",   "right_hip",
    "left_knee",     "right_knee",     "left_ankle", "right_ankle",
    "head",          "neck",
]
N_KP       = len(KP_NAMES)   # 14
INPUT_SIZE = 224              # el dataset se resizea a 224×224


# ---------------------------------------------------------------------------
# Arquitectura — copia exacta de PoseModel en entrenar_gcp.py
# ---------------------------------------------------------------------------

class PoseModel(nn.Module):
    def __init__(self, backbone_name: str = "mobilenetv2_100", n_keypoints: int = 14):
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
            nn.Sigmoid(),
        )
        self.head_vis = nn.Sequential(
            nn.Linear(512, n_keypoints),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        x = self.backbone(x)
        x = self.shared(x)
        return self.head_coords(x), self.head_vis(x)


# ---------------------------------------------------------------------------
# Transform de inferencia (igual que val_transform en entrenar_gcp.py)
# ---------------------------------------------------------------------------

_transform = T.Compose([
    T.Resize((INPUT_SIZE, INPUT_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])


# ---------------------------------------------------------------------------
# Clase principal
# ---------------------------------------------------------------------------

class DetectorKeypoints:
    """
    Carga el checkpoint y expone .detectar(frame).

    Parámetros
    ----------
    ruta_modelo  : str   — path al .pth guardado con torch.save(state_dict).
    backbone     : str   — debe coincidir con el usado en entrenamiento.
    device       : str   — 'cuda', 'mps', 'cpu', o None para autodetectar.
    umbral_vis   : float — keypoints con score < umbral se marcan como (-1,-1).
    """

    def __init__(self, ruta_modelo: str,
                 backbone: str = "mobilenetv2_100",
                 device: str | None = None,
                 umbral_vis: float = 0.3):
        self.umbral_vis = umbral_vis
        self.device     = self._elegir_device(device)

        self.modelo = PoseModel(backbone_name=backbone, n_keypoints=N_KP)
        state = torch.load(ruta_modelo, map_location=self.device)
        self.modelo.load_state_dict(state)
        self.modelo.to(self.device)
        self.modelo.eval()
        print(f"Modelo cargado en {self.device} desde {ruta_modelo}")

    # ------------------------------------------------------------------
    # Método principal
    # ------------------------------------------------------------------

    def detectar(self, frame: np.ndarray) -> dict | None:
        """
        Recibe un frame BGR (OpenCV) y devuelve keypoints en píxeles.

        Retorna
        -------
        dict con:
          "keypoints"   : np.ndarray (14, 2) — (x, y) en píxeles.
                          Keypoints con baja visibilidad → (-1, -1).
          "visibilidad" : np.ndarray (14,)   — scores [0, 1].
        Retorna None si el frame es inválido.
        """
        if frame is None or frame.size == 0:
            return None

        h, w   = frame.shape[:2]
        tensor = self._preprocesar(frame)

        with torch.no_grad():
            coords, vis = self.modelo(tensor)

        coords_np = coords.squeeze(0).cpu().numpy()   # (28,)
        vis_np    = vis.squeeze(0).cpu().numpy()       # (14,)

        keypoints = self._desnormalizar(coords_np, w, h)

        # Marcar keypoints de baja confianza como no detectados
        for i in range(N_KP):
            if vis_np[i] < self.umbral_vis:
                keypoints[i] = [-1.0, -1.0]

        return {"keypoints": keypoints, "visibilidad": vis_np}

    def detectar_en_crop(self, frame: np.ndarray,
                         bbox: tuple[int, int, int, int]) -> dict | None:
        """
        Versión top-down: extrae el crop de la persona y mapea keypoints
        de vuelta al frame original.

        bbox : (x1, y1, x2, y2) en píxeles del frame.
        """
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1  = max(0, x1);  y1  = max(0, y1)
        x2  = min(frame.shape[1], x2);  y2  = min(frame.shape[0], y2)

        if x2 <= x1 or y2 <= y1:
            return None

        crop    = frame[y1:y2, x1:x2]
        result  = self.detectar(crop)
        if result is None:
            return None

        kps      = result["keypoints"]
        crop_w   = x2 - x1
        crop_h   = y2 - y1
        kps_out  = kps.copy()

        validos = ~np.all(kps == -1, axis=1)
        # Las coords ya están en píxeles del crop (224×224 → crop_w×crop_h)
        kps_out[validos, 0] = kps[validos, 0] / INPUT_SIZE * crop_w + x1
        kps_out[validos, 1] = kps[validos, 1] / INPUT_SIZE * crop_h + y1

        return {"keypoints": kps_out, "visibilidad": result["visibilidad"]}

    # ------------------------------------------------------------------
    # Métodos privados
    # ------------------------------------------------------------------

    def _preprocesar(self, frame: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        t   = _transform(pil)
        return t.unsqueeze(0).to(self.device)

    @staticmethod
    def _desnormalizar(coords_norm: np.ndarray, w: int, h: int) -> np.ndarray:
        kps = coords_norm.reshape(N_KP, 2).copy()
        kps[:, 0] *= w
        kps[:, 1] *= h
        return kps

    @staticmethod
    def _elegir_device(device: str | None) -> torch.device:
        if device is not None:
            return torch.device(device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")