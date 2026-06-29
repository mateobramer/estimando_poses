"""
correr_camara.py
================
Loop principal para probar el sistema con la webcam.

USO:
    cd ~/Documents/GitHub/estimando_poses
    python correr_camara.py
    python correr_camara.py --ejercicio flexion_brazos
    python correr_camara.py --modelo modelo_b_fix_mejor.pth

CONTROLES:
    Q          → salir
    R          → resetear contador de reps
    1-5        → cambiar ejercicio en caliente
"""

import cv2
import numpy as np
import argparse
from detector_keypoints import DetectorKeypoints, KP_NAMES
from analizador import Analizador, calcular_angulo
from angulos_referencia import EJERCICIOS

# Esqueleto: pares de índices a conectar visualmente
SKELETON = [
    (0, 1),   # hombro izq - hombro der
    (0, 2),   # hombro izq - codo izq
    (2, 4),   # codo izq - muñeca izq
    (1, 3),   # hombro der - codo der
    (3, 5),   # codo der - muñeca der
    (0, 6),   # hombro izq - cadera izq
    (1, 7),   # hombro der - cadera der
    (6, 7),   # cadera izq - cadera der
    (6, 8),   # cadera izq - rodilla izq
    (8, 10),  # rodilla izq - tobillo izq
    (7, 9),   # cadera der - rodilla der
    (9, 11),  # rodilla der - tobillo der
    (12, 13), # cabeza - cuello
    (13, 0),  # cuello - hombro izq
    (13, 1),  # cuello - hombro der
]

EJERCICIO_KEYS = list(EJERCICIOS.keys())

COLOR_OK    = (0, 210, 0)
COLOR_MAL   = (0, 60, 255)
COLOR_INFO  = (255, 255, 255)
COLOR_HUESO = (180, 180, 180)


def determinar_fase(keypoints: np.ndarray, ejercicio: str) -> str:
    """
    Detecta automáticamente si la persona está en fase baja o alta
    usando el ángulo de la articulación de tracking del ejercicio.
    """
    tracking = EJERCICIOS[ejercicio].get("rep_tracking", {})
    art_nombre = tracking.get("articulacion", "rodilla_izq")

    # Obtener los keypoints de esa articulación
    art_config = EJERCICIOS[ejercicio]["articulaciones"].get(art_nombre, {})
    kps_idx = art_config.get("keypoints")
    if kps_idx is None:
        return "baja"

    a, b, c = keypoints[kps_idx[0]], keypoints[kps_idx[1]], keypoints[kps_idx[2]]
    ang = calcular_angulo(a, b, c)

    umbral = tracking.get("umbral_inicio", 120)
    return "baja" if 0 < ang < umbral else "alta"


def dibujar(frame: np.ndarray, keypoints: np.ndarray,
            visibilidad: np.ndarray, resultado: dict,
            ejercicio_idx: int):
    h, w = frame.shape[:2]

    # --- Esqueleto ---
    for i, j in SKELETON:
        xi, yi = keypoints[i]
        xj, yj = keypoints[j]
        if xi < 0 or yi < 0 or xj < 0 or yj < 0:
            continue
        cv2.line(frame, (int(xi), int(yi)), (int(xj), int(yj)),
                 COLOR_HUESO, 2, cv2.LINE_AA)

    # --- Keypoints ---
    for i, (x, y) in enumerate(keypoints):
        if x < 0 or y < 0:
            continue
        color = COLOR_OK if visibilidad[i] > 0.5 else (100, 100, 100)
        cv2.circle(frame, (int(x), int(y)), 5, color, -1, cv2.LINE_AA)

    # --- Panel superior: ángulos y feedback ---
    y_txt = 28
    for nombre, eval_info in resultado["evaluaciones"].items():
        ang    = resultado["angulos"].get(nombre, -1)
        estado = eval_info["estado"]
        msg    = eval_info["mensaje"]
        color  = COLOR_OK if estado == "ok" else (
                 COLOR_MAL if estado in ("bajo", "alto") else (120, 120, 120))

        ang_str = f"{ang:.0f}°" if ang >= 0 else "n/d"
        texto   = f"{nombre}: {ang_str}  {msg}"
        cv2.putText(frame, texto, (10, y_txt),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)
        y_txt += 24

    # --- Panel inferior: reps, fase, ejercicio ---
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 70), (w, h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    ejercicio_nombre = EJERCICIOS[resultado["ejercicio"]]["descripcion"]
    cv2.putText(frame, f"Ejercicio: {ejercicio_nombre}  [{ejercicio_idx+1}]",
                (10, h - 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_INFO, 1)
    cv2.putText(frame, f"Fase: {resultado['fase']}",
                (10, h - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_INFO, 1)

    reps_color = COLOR_OK if resultado["rep_completa"] else COLOR_INFO
    cv2.putText(frame, f"Reps: {resultado['num_reps']}",
                (w - 130, h - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                reps_color, 2, cv2.LINE_AA)

    if resultado["rep_completa"]:
        cv2.putText(frame, "REP!", (w // 2 - 40, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, COLOR_OK, 4, cv2.LINE_AA)

    # --- Teclas disponibles ---
    cv2.putText(frame, "Q=salir  R=reset reps  1-5=ejercicio",
                (10, h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                (150, 150, 150), 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelo",    default="modelo_a_fix_mejor.pth")
    parser.add_argument("--ejercicio", default="sentadilla",
                        choices=EJERCICIO_KEYS)
    parser.add_argument("--camara",    type=int, default=0)
    parser.add_argument("--umbral_vis", type=float, default=0.3)
    args = parser.parse_args()

    detector   = DetectorKeypoints(args.modelo, umbral_vis=args.umbral_vis)
    analizador = Analizador(args.ejercicio)
    ej_idx     = EJERCICIO_KEYS.index(args.ejercicio)

    cap = cv2.VideoCapture(args.camara)
    if not cap.isOpened():
        print(f"No se pudo abrir la cámara {args.camara}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Sistema iniciado.")
    print(f"Ejercicio: {args.ejercicio}")
    print(f"Ejercicios disponibles: {EJERCICIO_KEYS}")
    print("Teclas: Q=salir  R=reset reps  1-5=cambiar ejercicio")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        resultado = detector.detectar(frame)

        if resultado is not None:
            kps = resultado["keypoints"]
            vis = resultado["visibilidad"]

            fase = determinar_fase(kps, analizador.ejercicio)
            analisis = analizador.analizar(kps, fase)

            dibujar(frame, kps, vis, analisis, ej_idx)

        cv2.imshow("Pose Analysis", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("r"):
            analizador.reset_reps()
            print("Reps reseteadas")
        elif ord("1") <= key <= ord("5"):
            ej_idx = key - ord("1")
            if ej_idx < len(EJERCICIO_KEYS):
                nuevo = EJERCICIO_KEYS[ej_idx]
                analizador.cambiar_ejercicio(nuevo)
                print(f"Ejercicio cambiado a: {nuevo}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()