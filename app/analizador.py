
from __future__ import annotations

import numpy as np
from angulos_referencia import EJERCICIOS


# ---------------------------------------------------------------------------
# Función de cálculo de ángulo 
# ---------------------------------------------------------------------------

def calcular_angulo(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Ángulo interior en el vértice b, formado por los segmentos b→a y b→c.

    Parámetros
    ----------
    a, b, c : np.ndarray shape (2,) — coordenadas (x, y).
              b es el vértice (la articulación que se mide).

    Retorna
    -------
    float en grados [0°, 180°]. Devuelve 0.0 si algún punto es inválido.
    """
    # Ignorar keypoints marcados como no detectados
    if np.any(np.array([a, b, c]) < 0):
        return -1.0

    ba    = a - b
    bc    = c - b
    norma = np.linalg.norm(ba) * np.linalg.norm(bc)

    if norma < 1e-6:
        return 0.0

    cos_ang = np.clip(np.dot(ba, bc) / norma, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_ang)))


# ---------------------------------------------------------------------------
# Evaluador de ángulo contra un rango
# ---------------------------------------------------------------------------

def _evaluar(angulo: float, rango: tuple[float, float],
             feedback_bajo: str | None,
             feedback_alto: str | None) -> dict:
    """
    Compara un ángulo con su rango aceptable.

    Retorna dict con "estado" ('ok'|'bajo'|'alto'|'no_detectado') y "mensaje".
    """
    if angulo < 0:
        return {"estado": "no_detectado", "mensaje": "Keypoint no visible"}

    min_ang, max_ang = rango
    if angulo < min_ang:
        return {"estado": "bajo",  "mensaje": feedback_bajo or "Ángulo demasiado cerrado"}
    if angulo > max_ang:
        return {"estado": "alto",  "mensaje": feedback_alto or "Ángulo demasiado abierto"}
    return {"estado": "ok", "mensaje": "✓ Buen ángulo"}


# ---------------------------------------------------------------------------
# Detector de repeticiones
# ---------------------------------------------------------------------------

class _ContadorReps:
    """
    Máquina de estados de 2 estados: ARRIBA → ABAJO → ARRIBA = 1 rep.

    umbral_inicio : el ángulo baja por debajo de este valor → entra en fase baja.
    umbral_fin    : el ángulo sube por encima de este valor → rep completada.
    """

    def __init__(self, umbral_inicio: float, umbral_fin: float):
        self._umbral_inicio = umbral_inicio
        self._umbral_fin    = umbral_fin
        self._en_fase_baja  = False
        self.reps           = 0

    def actualizar(self, angulo: float) -> bool:
        """Retorna True si en este frame se completó una rep."""
        if angulo < 0:
            return False   

        # Soporte para ejercicios donde el ángulo empieza alto y baja
        invertido = self._umbral_inicio > self._umbral_fin

        completada = False
        if not invertido:
            if not self._en_fase_baja and angulo < self._umbral_inicio:
                self._en_fase_baja = True
            elif self._en_fase_baja and angulo > self._umbral_fin:
                self._en_fase_baja = False
                self.reps += 1
                completada = True
        else:
            if not self._en_fase_baja and angulo > self._umbral_inicio:
                self._en_fase_baja = True
            elif self._en_fase_baja and angulo < self._umbral_fin:
                self._en_fase_baja = False
                self.reps += 1
                completada = True

        return completada

    def reset(self):
        self._en_fase_baja = False
        self.reps = 0


# ---------------------------------------------------------------------------
# Clase principal
# ---------------------------------------------------------------------------

class Analizador:
    """
    Analiza keypoints frame a frame para un ejercicio dado.

    Uso
    ---
    analizador = Analizador("sentadilla")
    resultado  = analizador.analizar(keypoints, fase="baja")
    """

    def __init__(self, ejercicio: str):
        if ejercicio not in EJERCICIOS:
            raise ValueError(
                f"Ejercicio '{ejercicio}' no encontrado. "
                f"Disponibles: {list(EJERCICIOS.keys())}"
            )
        self._nombre    = ejercicio
        self._config    = EJERCICIOS[ejercicio]
        self._contador  = self._crear_contador()

    # ------------------------------------------------------------------
    # Método principal
    # ------------------------------------------------------------------

    def analizar(self, keypoints: np.ndarray, fase: str) -> dict:
        """
        Evalúa los keypoints del frame actual.

        Parámetros
        ----------
        keypoints : np.ndarray shape (14, 2) — salida de DetectorKeypoints.detectar().
                    Keypoints no detectados tienen valor (-1, -1).
        fase      : str — nombre de la fase a evaluar ("baja", "alta", etc.)
                    Debe coincidir con una clave en la config del ejercicio.

        Retorna
        -------
        dict con claves: ejercicio, fase, angulos, evaluaciones, ok_global,
                         rep_completa, num_reps.
        """
        angulos     = self._calcular_angulos(keypoints, fase)
        evaluaciones = self._evaluar_angulos(angulos, fase)

        # ok_global solo considera articulaciones visibles
        estados_validos = [
            e["estado"]
            for e in evaluaciones.values()
            if e["estado"] != "no_detectado"
        ]
        ok_global = len(estados_validos) > 0 and all(
            s == "ok" for s in estados_validos
        )

        # Actualizar contador de reps con la articulación de tracking
        rep_completa = self._actualizar_reps(angulos)

        return {
            "ejercicio":    self._nombre,
            "fase":         fase,
            "angulos":      angulos,
            "evaluaciones": evaluaciones,
            "ok_global":    ok_global,
            "rep_completa": rep_completa,
            "num_reps":     self._contador.reps,
        }

    # ------------------------------------------------------------------
    # Utilidades de instancia
    # ------------------------------------------------------------------

    def reset_reps(self):
        """Reinicia el contador de repeticiones."""
        self._contador.reset()

    def cambiar_ejercicio(self, ejercicio: str):
        """Cambia el ejercicio en caliente sin crear un nuevo objeto."""
        if ejercicio not in EJERCICIOS:
            raise ValueError(f"Ejercicio '{ejercicio}' no encontrado.")
        self._nombre   = ejercicio
        self._config   = EJERCICIOS[ejercicio]
        self._contador = self._crear_contador()

    @property
    def ejercicio(self) -> str:
        return self._nombre

    @property
    def num_reps(self) -> int:
        return self._contador.reps

    @staticmethod
    def ejercicios_disponibles() -> list[str]:
        return list(EJERCICIOS.keys())

    # ------------------------------------------------------------------
    # Métodos privados
    # ------------------------------------------------------------------

    def _calcular_angulos(self, keypoints: np.ndarray, fase: str) -> dict[str, float]:
        """
        Calcula el ángulo de cada articulación definida para la fase dada.
        Retorna dict {nombre_articulacion: angulo_en_grados}.
        """
        angulos = {}
        for nombre, art_config in self._config["articulaciones"].items():
            if fase not in art_config["fases"]:
                continue   # esta articulación no se evalúa en esta fase
            a_idx, b_idx, c_idx = art_config["keypoints"]
            a = keypoints[a_idx]
            b = keypoints[b_idx]
            c = keypoints[c_idx]
            angulos[nombre] = calcular_angulo(a, b, c)
        return angulos

    def _evaluar_angulos(self, angulos: dict[str, float],
                         fase: str) -> dict[str, dict]:
        """
        Compara cada ángulo calculado con su rango de referencia.
        """
        evaluaciones = {}
        for nombre, angulo in angulos.items():
            fase_config = self._config["articulaciones"][nombre]["fases"][fase]
            evaluaciones[nombre] = _evaluar(
                angulo=angulo,
                rango=fase_config["rango"],
                feedback_bajo=fase_config.get("feedback_bajo"),
                feedback_alto=fase_config.get("feedback_alto"),
            )
        return evaluaciones

    def _actualizar_reps(self, angulos: dict[str, float]) -> bool:
        tracking = self._config.get("rep_tracking")
        if not tracking:
            return False

        # Soporte para uno o varios articulaciones de tracking
        if "articulaciones" in tracking:
            valores = [angulos.get(a, -1.0) for a in tracking["articulaciones"]]
            valores_validos = [v for v in valores if v >= 0]
            if not valores_validos:
                return False
            angulo_ref = min(valores_validos)  # la que más se movió
        else:
            angulo_ref = angulos.get(tracking["articulacion"], -1.0)

        return self._contador.actualizar(angulo_ref)

    def _crear_contador(self) -> _ContadorReps:
        tracking = self._config.get("rep_tracking", {})
        return _ContadorReps(
            umbral_inicio=tracking.get("umbral_inicio", 120),
            umbral_fin=tracking.get("umbral_fin",    160),
        )