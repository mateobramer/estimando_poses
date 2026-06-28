"""
angulos_referencia.py
=====================
Fuente única de verdad sobre qué ángulos son correctos para cada ejercicio.
Este archivo NO contiene lógica — solo datos.

Orden de keypoints (igual que entrenar_gcp.py)
----------------------------------------------
  0  left_shoulder    1  right_shoulder
  2  left_elbow       3  right_elbow
  4  left_wrist       5  right_wrist
  6  left_hip         7  right_hip
  8  left_knee        9  right_knee
  10 left_ankle       11 right_ankle
  12 head             13 neck

Para agregar un ejercicio: agregar una entrada al dict EJERCICIOS.
No tocar ningún otro archivo.

Fuentes biomecánicas
--------------------
- Kotiuk et al. (2022) — CEJSSM — knee/hip/ankle angles in squats
- Frontiers in Sports (2024) — deep squat review
- ETH Zurich / PubMed (2012) — restricted vs unrestricted squat kinematics
"""

# Alias legibles para los índices
L_SHOULDER, R_SHOULDER = 0, 1
L_ELBOW,    R_ELBOW    = 2, 3
L_WRIST,    R_WRIST    = 4, 5
L_HIP,      R_HIP      = 6, 7
L_KNEE,     R_KNEE     = 8, 9
L_ANKLE,    R_ANKLE    = 10, 11
HEAD,       NECK       = 12, 13

EJERCICIOS: dict = {

    # ------------------------------------------------------------------
    # SENTADILLA PARALELA
    # Rodilla baja: 55°–80°  |  Cadera baja: 90°–110°
    # Rodilla arriba: 160°–180°
    # ------------------------------------------------------------------
    "sentadilla": {
        "descripcion": "Sentadilla (parallel squat)",
        "articulaciones": {
            "rodilla_izq": {
                "keypoints": (L_HIP, L_KNEE, L_ANKLE),
                "fases": {
                    "baja": {
                        "rango": (55, 80),
                        "feedback_bajo": "Rodilla izquierda muy flexionada",
                        "feedback_alto": "Rodilla izquierda no llegó a profundidad paralela",
                    },
                    "alta": {
                        "rango": (160, 180),
                        "feedback_bajo": "Extendé la rodilla izquierda al subir",
                        "feedback_alto": None,
                    },
                },
            },
            "rodilla_der": {
                "keypoints": (R_HIP, R_KNEE, R_ANKLE),
                "fases": {
                    "baja": {
                        "rango": (55, 80),
                        "feedback_bajo": "Rodilla derecha muy flexionada",
                        "feedback_alto": "Rodilla derecha no llegó a profundidad paralela",
                    },
                    "alta": {
                        "rango": (160, 180),
                        "feedback_bajo": "Extendé la rodilla derecha al subir",
                        "feedback_alto": None,
                    },
                },
            },
            "cadera_izq": {
                "keypoints": (L_SHOULDER, L_HIP, L_KNEE),
                "fases": {
                    "baja": {
                        "rango": (90, 110),
                        "feedback_bajo": "Torso muy inclinado — mantené la espalda",
                        "feedback_alto": "Cadera izquierda no bajó suficiente",
                    },
                },
            },
            "cadera_der": {
                "keypoints": (R_SHOULDER, R_HIP, R_KNEE),
                "fases": {
                    "baja": {
                        "rango": (90, 110),
                        "feedback_bajo": "Torso muy inclinado — mantené la espalda",
                        "feedback_alto": "Cadera derecha no bajó suficiente",
                    },
                },
            },
        },
        "rep_tracking": {
            "articulacion": "rodilla_izq",
            "umbral_inicio": 100,
            "umbral_fin":    150,
        },
    },

    # ------------------------------------------------------------------
    # MEDIA SENTADILLA  — rodilla baja: 70°–100°
    # ------------------------------------------------------------------
    "media_sentadilla": {
        "descripcion": "Media sentadilla (half squat)",
        "articulaciones": {
            "rodilla_izq": {
                "keypoints": (L_HIP, L_KNEE, L_ANKLE),
                "fases": {
                    "baja": {
                        "rango": (70, 100),
                        "feedback_bajo": "Más abajo de media sentadilla",
                        "feedback_alto": "Bajá más para llegar a media sentadilla",
                    },
                    "alta": {
                        "rango": (160, 180),
                        "feedback_bajo": "Extendé la rodilla izquierda al subir",
                        "feedback_alto": None,
                    },
                },
            },
            "rodilla_der": {
                "keypoints": (R_HIP, R_KNEE, R_ANKLE),
                "fases": {
                    "baja": {
                        "rango": (70, 100),
                        "feedback_bajo": "Más abajo de media sentadilla",
                        "feedback_alto": "Bajá más para llegar a media sentadilla",
                    },
                    "alta": {
                        "rango": (160, 180),
                        "feedback_bajo": "Extendé la rodilla derecha al subir",
                        "feedback_alto": None,
                    },
                },
            },
        },
        "rep_tracking": {
            "articulacion": "rodilla_izq",
            "umbral_inicio": 110,
            "umbral_fin":    155,
        },
    },

    # ------------------------------------------------------------------
    # SENTADILLA PROFUNDA  — rodilla baja: 40°–55°
    # ------------------------------------------------------------------
    "sentadilla_profunda": {
        "descripcion": "Sentadilla profunda (deep squat)",
        "articulaciones": {
            "rodilla_izq": {
                "keypoints": (L_HIP, L_KNEE, L_ANKLE),
                "fases": {
                    "baja": {
                        "rango": (40, 60),
                        "feedback_bajo": "Posición extrema de rodilla",
                        "feedback_alto": "No llegaste a profundidad de deep squat",
                    },
                    "alta": {
                        "rango": (160, 180),
                        "feedback_bajo": "Extendé la rodilla al subir",
                        "feedback_alto": None,
                    },
                },
            },
            "rodilla_der": {
                "keypoints": (R_HIP, R_KNEE, R_ANKLE),
                "fases": {
                    "baja": {
                        "rango": (40, 60),
                        "feedback_bajo": "Posición extrema de rodilla",
                        "feedback_alto": "No llegaste a profundidad de deep squat",
                    },
                    "alta": {
                        "rango": (160, 180),
                        "feedback_bajo": "Extendé la rodilla al subir",
                        "feedback_alto": None,
                    },
                },
            },
        },
        "rep_tracking": {
            "articulacion": "rodilla_izq",
            "umbral_inicio": 80,
            "umbral_fin":    155,
        },
    },

    # ------------------------------------------------------------------
    # ESTOCADA  — rodilla delantera baja: 80°–105°
    # ------------------------------------------------------------------
    "estocada": {
        "descripcion": "Estocada (lunge)",
        "articulaciones": {
            "rodilla_delantera": {
                "keypoints": (L_HIP, L_KNEE, L_ANKLE),
                "fases": {
                    "baja": {
                        "rango": (80, 105),
                        "feedback_bajo": "Rodilla delantera muy cerrada",
                        "feedback_alto": "La rodilla delantera debe llegar a ~90°",
                    },
                    "alta": {
                        "rango": (155, 180),
                        "feedback_bajo": "Extendé la pierna delantera al volver",
                        "feedback_alto": None,
                    },
                },
            },
            "cadera_delantera": {
                "keypoints": (L_SHOULDER, L_HIP, L_KNEE),
                "fases": {
                    "baja": {
                        "rango": (85, 110),
                        "feedback_bajo": "Torso muy inclinado hacia adelante",
                        "feedback_alto": "Cadera no bajó suficiente",
                    },
                },
            },
        },
        "rep_tracking": {
            "articulacion": "rodilla_delantera",
            "umbral_inicio": 110,
            "umbral_fin":    160,
        },
    },

    # ------------------------------------------------------------------
    # FLEXIÓN DE BRAZOS  — codo baja: 75°–105°  |  sube: 155°–180°
    # ------------------------------------------------------------------
    "flexion_brazos": {
        "descripcion": "Flexión de brazos (push-up)",
        "articulaciones": {
            "codo_izq": {
                "keypoints": (L_SHOULDER, L_ELBOW, L_WRIST),
                "fases": {
                    "baja": {
                        "rango": (75, 105),
                        "feedback_bajo": "Codo izquierdo muy cerrado",
                        "feedback_alto": "Bajá más — el codo debe llegar a ~90°",
                    },
                    "alta": {
                        "rango": (155, 180),
                        "feedback_bajo": "Extendé completamente el brazo izquierdo",
                        "feedback_alto": None,
                    },
                },
            },
            "codo_der": {
                "keypoints": (R_SHOULDER, R_ELBOW, R_WRIST),
                "fases": {
                    "baja": {
                        "rango": (75, 105),
                        "feedback_bajo": "Codo derecho muy cerrado",
                        "feedback_alto": "Bajá más — el codo debe llegar a ~90°",
                    },
                    "alta": {
                        "rango": (155, 180),
                        "feedback_bajo": "Extendé completamente el brazo derecho",
                        "feedback_alto": None,
                    },
                },
            },
            "columna": {
                "keypoints": (L_SHOULDER, L_HIP, L_KNEE),
                "fases": {
                    "baja": {
                        "rango": (165, 180),
                        "feedback_bajo": "Cadera caída — mantené el cuerpo recto",
                        "feedback_alto": "Cadera elevada — bajá las caderas",
                    },
                    "alta": {
                        "rango": (165, 180),
                        "feedback_bajo": "Cadera caída — mantené el cuerpo recto",
                        "feedback_alto": "Cadera elevada — bajá las caderas",
                    },
                },
            },
        },
        "rep_tracking": {
            "articulacion": "codo_izq",
            "umbral_inicio": 120,
            "umbral_fin":    150,
        },
    },
}