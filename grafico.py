"""
grafico.py
==========
Gráfico de línea en tiempo real del ángulo articular.
Dibujado directamente sobre un numpy array BGR (OpenCV).

Interfaz pública
----------------
  from grafico import GraficoAngulo

  g = GraficoAngulo(width=320, height=130, hist_len=120)
  g.agregar(angulo)
  img = g.renderizar(rango=(55, 80), label="rodilla_izq")
  # img es np.ndarray (H, W, 3) BGR — pegalo donde quieras en el frame
"""

import numpy as np
import cv2
from collections import deque

# Colores (BGR)
_BG       = (35,  25,  25)
_ZONA_OK  = (20,  60,  20)
_LINEA_OK = (60, 160,  60)
_CURVA_OK = (80, 210,  80)
_CURVA_MAL= (80,  80, 210)
_DIM      = (100, 100, 100)


class GraficoAngulo:
    """
    Mantiene el historial de un ángulo y lo renderiza como gráfico de línea.

    Parámetros
    ----------
    width    : int — ancho en píxeles del gráfico.
    height   : int — alto en píxeles del gráfico.
    hist_len : int — cuántos frames de historial mantener.
    ang_min  : float — mínimo esperado en el eje Y (para la escala).
    ang_max  : float — máximo esperado en el eje Y (para la escala).
    """

    def __init__(self, width: int = 320, height: int = 130,
                 hist_len: int = 120,
                 ang_min: float = 0.0, ang_max: float = 200.0):
        self.w        = width
        self.h        = height
        self.hist_len = hist_len
        self.y_min    = ang_min
        self.y_max    = ang_max
        self._hist    = deque(maxlen=hist_len)

    def agregar(self, angulo: float):
        """Agrega un valor al historial. Pasar -1 para frames sin detección."""
        self._hist.append(angulo)

    def limpiar(self):
        """Limpia el historial."""
        self._hist.clear()

    def renderizar(self, rango: tuple[float, float], label: str = "") -> np.ndarray:
        """
        Genera la imagen BGR del gráfico.

        Parámetros
        ----------
        rango : (min, max) — rango correcto del ángulo (zona verde).
        label : str        — texto descriptivo en la esquina inferior.

        Retorna
        -------
        np.ndarray shape (H, W, 3) dtype uint8.
        """
        img    = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        img[:] = _BG

        margen   = 18
        ang_min_r, ang_max_r = rango

        def _y_px(ang):
            """Convierte ángulo a coordenada Y en píxeles."""
            ratio = (ang - self.y_min) / max(self.y_max - self.y_min, 1)
            return int(self.h - margen - ratio * (self.h - 2 * margen))

        # Zona verde (rango correcto)
        y_top = max(margen, _y_px(ang_max_r))
        y_bot = min(self.h - margen, _y_px(ang_min_r))
        cv2.rectangle(img, (0, y_top), (self.w, y_bot), _ZONA_OK, -1)
        cv2.line(img, (0, y_top), (self.w, y_top), _LINEA_OK, 1)
        cv2.line(img, (0, y_bot), (self.w, y_bot), _LINEA_OK, 1)

        # Labels del rango
        cv2.putText(img, f"{ang_max_r:.0f}°",
                    (self.w - 36, y_top + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, _LINEA_OK, 1)
        cv2.putText(img, f"{ang_min_r:.0f}°",
                    (self.w - 36, max(y_bot - 3, margen + 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, _LINEA_OK, 1)

        # Curva del historial
        hist = list(self._hist)
        n    = len(hist)
        if n >= 2:
            pts = []
            for i, ang in enumerate(hist):
                if ang < 0:
                    pts.append(None)
                    continue
                x = int(i / max(n - 1, 1) * (self.w - 1))
                y = max(margen, min(self.h - margen, _y_px(ang)))
                pts.append((x, y))

            for i in range(1, len(pts)):
                if pts[i-1] is None or pts[i] is None:
                    continue
                ang_val  = hist[i] if hist[i] >= 0 else 0
                en_rango = ang_min_r <= ang_val <= ang_max_r
                color    = _CURVA_OK if en_rango else _CURVA_MAL
                cv2.line(img, pts[i-1], pts[i], color, 2, cv2.LINE_AA)

        # Valor actual (último válido)
        ultimo = next((v for v in reversed(hist) if v >= 0), -1)
        if ultimo >= 0:
            en_rango  = ang_min_r <= ultimo <= ang_max_r
            color_val = _CURVA_OK if en_rango else _CURVA_MAL
            cv2.putText(img, f"{ultimo:.0f}°",
                        (6, 18), cv2.FONT_HERSHEY_SIMPLEX,
                        0.58, color_val, 2, cv2.LINE_AA)

        # Label inferior
        if label:
            cv2.putText(img, label,
                        (6, self.h - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.32, _DIM, 1)

        return img
