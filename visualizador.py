"""
visualizador.py
===============
Todo el dibujo sobre el frame de video: esqueleto, panel de ángulos,
gráficos en tiempo real y overlay de información.

Interfaz pública
----------------
  from visualizador import Visualizador

  viz = Visualizador()
  viz.dibujar(frame, keypoints, visibilidad, resultado, historiales,
              filtro_activo, n_personas)
  # Modifica frame in-place.
"""

import cv2
import numpy as np
from angulos_referencia import EJERCICIOS
from grafico import GraficoAngulo

# Índices y labels de keypoints (mismo orden que entrenar_gcp.py)
KP_NAMES = [
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist",    "right_wrist",    "left_hip",   "right_hip",
    "left_knee",     "right_knee",     "left_ankle", "right_ankle",
    "head",          "neck",
]
KP_LABELS = [
    "L.SH", "R.SH", "L.EL", "R.EL",
    "L.WR", "R.WR", "L.HI", "R.HI",
    "L.KN", "R.KN", "L.AN", "R.AN",
    "HEAD", "NECK",
]
SKELETON = [
    (12, 13), (13, 0), (13, 1),
    (0,  2),  (2,  4), (1,  3), (3, 5),
    (0,  6),  (1,  7), (6,  7),
    (6,  8),  (8, 10), (7,  9), (9, 11),
]

# Colores (BGR)
C_OK     = (80,  200,  80)
C_MAL    = (60,   60, 220)
C_INFO   = (230, 230, 230)
C_DIM    = (100, 100, 100)
C_HUESO  = (200, 160,  60)
C_KP_HI  = (80,  220,  80)
C_KP_LO  = (80,  180, 200)

# Tamaño de cada gráfico
PLOT_W = 300
PLOT_H = 120


class Visualizador:
    """
    Dibuja el esqueleto, panel de feedback y gráficos sobre el frame.

    Los GraficoAngulo se crean internamente por articulación y se
    reutilizan entre frames para mantener el historial.
    """

    def __init__(self):
        self._graficos: dict[str, GraficoAngulo] = {}

    # ------------------------------------------------------------------
    # Método principal
    # ------------------------------------------------------------------

    def dibujar(self, frame: np.ndarray,
                keypoints: np.ndarray,
                visibilidad: np.ndarray,
                resultado: dict,
                filtro_activo: bool,
                n_personas: int):
        """
        Dibuja todo el overlay sobre frame (in-place).

        Parámetros
        ----------
        frame        : np.ndarray BGR — frame del video.
        keypoints    : np.ndarray (14, 2) — keypoints en píxeles.
        visibilidad  : np.ndarray (14,)  — scores de visibilidad.
        resultado    : dict — salida de Analizador.analizar().
        filtro_activo: bool — si el One Euro Filter está activo.
        n_personas   : int  — personas detectadas por YOLO.
        """
        self._dibujar_skeleton(frame, keypoints, visibilidad)
        self._dibujar_graficos(frame, resultado)
        self._dibujar_panel_izq(frame, resultado)
        self._dibujar_panel_inf(frame, resultado, filtro_activo, n_personas)

    def limpiar_graficos(self):
        """Limpia el historial de todos los gráficos (al cambiar ejercicio)."""
        for g in self._graficos.values():
            g.limpiar()

    # ------------------------------------------------------------------
    # Métodos privados
    # ------------------------------------------------------------------

    def _dibujar_skeleton(self, frame, kps, vis):
        for a, b in SKELETON:
            xa, ya = kps[a]; xb, yb = kps[b]
            if xa < 0 or ya < 0 or xb < 0 or yb < 0:
                continue
            cv2.line(frame, (int(xa), int(ya)), (int(xb), int(yb)),
                     C_HUESO, 2, cv2.LINE_AA)

        for i, (x, y) in enumerate(kps):
            if x < 0 or y < 0:
                continue
            color = C_KP_HI if vis[i] > 0.6 else C_KP_LO
            cv2.circle(frame, (int(x), int(y)), 5, color, -1, cv2.LINE_AA)
            label = KP_LABELS[i]
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.32, 1)
            cv2.rectangle(frame,
                          (int(x)+5, int(y)-th-5),
                          (int(x)+5+tw, int(y)-1), (0, 0, 0), -1)
            cv2.putText(frame, label, (int(x)+5, int(y)-3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                        (255, 255, 255), 1)

    def _dibujar_panel_izq(self, frame, resultado):
        n_arts   = len(resultado["evaluaciones"])
        panel_h  = n_arts * 28 + 16
        overlay  = frame.copy()
        cv2.rectangle(overlay, (0, 0), (370, panel_h), (20, 20, 30), -1)
        cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

        y = 22
        for nombre, eval_info in resultado["evaluaciones"].items():
            ang    = resultado["angulos"].get(nombre, -1)
            estado = eval_info["estado"]
            msg    = eval_info["mensaje"]
            color  = (C_OK  if estado == "ok" else
                      C_MAL if estado in ("bajo", "alto") else C_DIM)

            # Indicador circular
            ind_c = (C_OK  if estado == "ok" else
                     C_MAL if estado != "no_detectado" else C_DIM)
            cv2.circle(frame, (10, y - 4), 5, ind_c, -1)

            ang_str = f"{ang:.0f}°" if ang >= 0 else "n/d"
            cv2.putText(frame, f"{nombre}: {ang_str}",
                        (22, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.48, color, 1, cv2.LINE_AA)
            cv2.putText(frame, msg,
                        (22, y + 14), cv2.FONT_HERSHEY_SIMPLEX,
                        0.34, C_DIM, 1, cv2.LINE_AA)
            y += 28

    def _dibujar_graficos(self, frame, resultado):
        h, w     = frame.shape[:2]
        ejercicio = resultado["ejercicio"]
        fase      = resultado["fase"]
        config    = EJERCICIOS[ejercicio]["articulaciones"]

        x_g = w - PLOT_W - 8
        y_g = 8

        for nombre, ang in resultado["angulos"].items():
            # Obtener o crear el gráfico para esta articulación
            if nombre not in self._graficos:
                self._graficos[nombre] = GraficoAngulo(
                    width=PLOT_W, height=PLOT_H)
            g = self._graficos[nombre]
            g.agregar(ang)

            # Rango según la fase actual
            art_cfg   = config.get(nombre, {})
            fase_cfg  = art_cfg.get("fases", {}).get(fase)
            if fase_cfg is None:
                fases = art_cfg.get("fases", {})
                fase_cfg = next(iter(fases.values()), None) if fases else None
            if fase_cfg is None:
                continue

            img_g = g.renderizar(rango=fase_cfg["rango"], label=nombre)

            # Pegar en el frame si hay espacio
            if y_g + PLOT_H <= h - 80:
                frame[y_g:y_g+PLOT_H, x_g:x_g+PLOT_W] = img_g
                y_g += PLOT_H + 6

    def _dibujar_panel_inf(self, frame, resultado, filtro_activo, n_personas):
        h, w = frame.shape[:2]

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h-72), (w, h), (15, 15, 25), -1)
        cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

        desc = EJERCICIOS[resultado["ejercicio"]]["descripcion"]
        cv2.putText(frame, desc, (10, h-52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_INFO, 1)

        filtro_str = "Filtro 1€: ON" if filtro_activo else "Filtro 1€: OFF"
        cv2.putText(frame, filtro_str, (10, h-32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44,
                    C_OK if filtro_activo else C_DIM, 1)

        cv2.putText(frame,
                    f"Personas: {n_personas}  |  Fase: {resultado['fase']}",
                    (10, h-14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_DIM, 1)

        # Reps
        reps_color = C_OK if resultado["rep_completa"] else C_INFO
        cv2.putText(frame, str(resultado["num_reps"]),
                    (w-90, h-14), cv2.FONT_HERSHEY_SIMPLEX,
                    2.0, reps_color, 3, cv2.LINE_AA)
        cv2.putText(frame, "reps", (w-88, h-46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_DIM, 1)

        # Flash rep completa
        if resultado["rep_completa"]:
            cv2.putText(frame, "REP!", (w//2-70, h//2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2.5, C_OK, 4, cv2.LINE_AA)

        # Ayuda de teclas
        cv2.putText(frame,
                    "Q=salir  R=reset  F=filtro  M=menú  1-5=ejercicio",
                    (160, h-3), cv2.FONT_HERSHEY_SIMPLEX,
                    0.32, C_DIM, 1)
