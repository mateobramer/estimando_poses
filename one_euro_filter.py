"""
one_euro_filter.py
==================
Implementación desde cero del One Euro Filter para suavizar keypoints
en tiempo real. Elimina jitter estático sin introducir lag en movimiento.

Referencia: Casiez et al. (2012) — "1€ Filter: A Simple Speed-based
Low-pass Filter for Noisy Input in Interactive Systems"

Idea central
------------
Un low-pass filter simple suaviza bien el ruido pero introduce lag.
El One Euro Filter resuelve esto adaptando la frecuencia de corte
según la velocidad de la señal:
  - Movimiento lento  → frecuencia de corte baja  → más suavizado
  - Movimiento rápido → frecuencia de corte alta  → menos suavizado (menos lag)

Parámetros
----------
freq   : float — frecuencia de muestreo estimada (fps). Default: 30.
mincutoff : float — frecuencia de corte mínima (Hz). Más bajo = más suavizado
                    en reposo. Default: 1.0
beta   : float — coeficiente de velocidad. Más alto = menos lag en movimiento.
                 Default: 0.007
dcutoff : float — frecuencia de corte del derivador (velocidad). Default: 1.0

Valores recomendados para keypoints de pose estimation:
  mincutoff=1.0, beta=0.05, dcutoff=1.0
  → suaviza bien el jitter estático, sigue rápido el movimiento de ejercicio.

Uso
---
  from one_euro_filter import OneEuroFilterArray

  # Una instancia por keypoint array (14 keypoints × 2 coords = 28 valores)
  filtro = OneEuroFilterArray(size=28, freq=30, mincutoff=1.0, beta=0.05)

  # En cada frame:
  kps_suavizados = filtro.apply(kps.flatten()).reshape(14, 2)
"""

from __future__ import annotations
import numpy as np


# ---------------------------------------------------------------------------
# Low-pass filter de un solo valor
# ---------------------------------------------------------------------------

class _LowPassFilter:
    """Exponential moving average con alpha variable."""

    def __init__(self):
        self._initialized = False
        self._value       = 0.0

    def apply(self, x: float, alpha: float) -> float:
        if not self._initialized:
            self._value       = x
            self._initialized = True
            return x
        self._value = alpha * x + (1.0 - alpha) * self._value
        return self._value

    @property
    def value(self) -> float:
        return self._value

    @property
    def initialized(self) -> bool:
        return self._initialized


# ---------------------------------------------------------------------------
# One Euro Filter de un solo valor
# ---------------------------------------------------------------------------

class _OneEuroFilter:
    """
    One Euro Filter escalar.
    Implementa la ecuación completa de Casiez et al. (2012).
    """

    def __init__(self, freq: float = 30.0, mincutoff: float = 1.0,
                 beta: float = 0.007, dcutoff: float = 1.0):
        self._freq      = freq
        self._mincutoff = mincutoff
        self._beta      = beta
        self._dcutoff   = dcutoff
        self._x         = _LowPassFilter()   # valor suavizado
        self._dx        = _LowPassFilter()   # velocidad suavizada

    def apply(self, x: float) -> float:
        # Alpha para el derivador (velocidad) — frecuencia fija dcutoff
        alpha_d = self._alpha(self._dcutoff)

        # Velocidad estimada
        if self._x.initialized:
            dx_raw = (x - self._x.value) * self._freq
        else:
            dx_raw = 0.0
        dx = self._dx.apply(dx_raw, alpha_d)

        # Frecuencia de corte adaptativa: sube con la velocidad
        cutoff = self._mincutoff + self._beta * abs(dx)

        # Alpha para el valor
        alpha = self._alpha(cutoff)
        return self._x.apply(x, alpha)

    def _alpha(self, cutoff: float) -> float:
        """Convierte frecuencia de corte en coeficiente alpha."""
        tau = 1.0 / (2.0 * np.pi * cutoff)
        dt  = 1.0 / self._freq
        return 1.0 / (1.0 + tau / dt)


# ---------------------------------------------------------------------------
# Wrapper vectorial para arrays de keypoints
# ---------------------------------------------------------------------------

class OneEuroFilterArray:
    """
    Aplica un One Euro Filter independiente a cada elemento de un array.

    Diseñado para suavizar arrays de keypoints (14×2 = 28 valores),
    pero funciona con cualquier tamaño.

    Parámetros
    ----------
    size      : int   — número de valores a filtrar (ej: 14*2 = 28).
    freq      : float — FPS estimados de la cámara.
    mincutoff : float — suavizado en reposo (más bajo = más suave).
    beta      : float — respuesta al movimiento (más alto = menos lag).
    dcutoff   : float — frecuencia del filtro de velocidad (dejar en 1.0).
    """

    def __init__(self, size: int, freq: float = 30.0,
                 mincutoff: float = 1.0, beta: float = 0.05,
                 dcutoff: float = 1.0):
        self._filters = [
            _OneEuroFilter(freq=freq, mincutoff=mincutoff,
                           beta=beta, dcutoff=dcutoff)
            for _ in range(size)
        ]
        self._size = size

    def apply(self, valores: np.ndarray) -> np.ndarray:
        """
        Suaviza un array de valores.

        Parámetros
        ----------
        valores : np.ndarray shape (size,) — valores crudos del frame actual.
                  Los valores -1 (keypoints no detectados) se pasan sin filtrar.

        Retorna
        -------
        np.ndarray shape (size,) — valores suavizados.
        """
        assert len(valores) == self._size, \
            f"Se esperaban {self._size} valores, llegaron {len(valores)}"

        resultado = np.empty(self._size, dtype=np.float32)
        for i, (v, f) in enumerate(zip(valores, self._filters)):
            # No filtrar keypoints no detectados (-1)
            resultado[i] = v if v < 0 else f.apply(float(v))
        return resultado

    def reset(self):
        """Reinicia todos los filtros (útil al cambiar de ejercicio)."""
        for f in self._filters:
            f._x = _LowPassFilter()
            f._dx = _LowPassFilter()
