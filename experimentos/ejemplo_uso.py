"""
ejemplo_uso.py
==============
Ejemplo de cómo conectar los tres módulos en un loop de video real.
Reemplazá "ruta/modelo_a_fix_mejor.pth" con el path real al checkpoint.
"""

import cv2
import numpy as np
from detector_keypoints import DetectorKeypoints
from analizador import Analizador

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

RUTA_MODELO  = "ruta/modelo_a_fix_mejor.pth"
EJERCICIO    = "sentadilla"
UMBRAL_BAJA  = 100   # ángulo de rodilla por debajo del cual es "fase baja"

# ---------------------------------------------------------------------------
# Inicialización
# ---------------------------------------------------------------------------

detector   = DetectorKeypoints(RUTA_MODELO)
analizador = Analizador(EJERCICIO)

cap = cv2.VideoCapture(0)
print(f"Ejercicio: {analizador.ejercicio}")
print(f"Ejercicios disponibles: {Analizador.ejercicios_disponibles()}")

# ---------------------------------------------------------------------------
# Loop principal
# ---------------------------------------------------------------------------

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break

    # 1. Detectar keypoints
    keypoints = detector.detectar(frame)

    if keypoints is not None:

        # 2. Determinar fase según ángulo de rodilla izquierda (kp 8,10,12)
        from analizador import calcular_angulo
        ang_rodilla = calcular_angulo(
            keypoints[8], keypoints[10], keypoints[12]
        )
        fase = "baja" if ang_rodilla < UMBRAL_BAJA else "alta"

        # 3. Analizar
        resultado = analizador.analizar(keypoints, fase)

        # 4. Dibujar en pantalla
        _dibujar(frame, keypoints, resultado)

    cv2.imshow("Pose Analysis", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Función auxiliar de visualización
# ---------------------------------------------------------------------------

def _dibujar(frame: np.ndarray, keypoints: np.ndarray, resultado: dict):
    """Dibuja keypoints, ángulos y feedback sobre el frame."""

    # Colores por estado
    COLOR = {"ok": (0, 200, 0), "bajo": (0, 100, 255),
             "alto": (0, 0, 255), "no_detectado": (120, 120, 120)}

    # Keypoints
    for i, (x, y) in enumerate(keypoints):
        if x < 0 or y < 0:
            continue
        cv2.circle(frame, (int(x), int(y)), 5, (255, 255, 0), -1)
        cv2.putText(frame, str(i), (int(x)+6, int(y)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    # Ángulos y feedback
    y_texto = 30
    for nombre, eval_info in resultado["evaluaciones"].items():
        angulo  = resultado["angulos"].get(nombre, -1)
        estado  = eval_info["estado"]
        mensaje = eval_info["mensaje"]
        color   = COLOR[estado]

        texto = f"{nombre}: {angulo:.1f}°  {mensaje}"
        cv2.putText(frame, texto, (10, y_texto),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        y_texto += 28

    # Reps y estado global
    color_global = (0, 200, 0) if resultado["ok_global"] else (0, 0, 255)
    cv2.putText(frame, f"Reps: {resultado['num_reps']}",
                (10, frame.shape[0] - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    if resultado["rep_completa"]:
        cv2.putText(frame, "REP COMPLETA", (10, frame.shape[0] - 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 0), 3)
