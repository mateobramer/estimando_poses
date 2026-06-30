L_SHOULDER, R_SHOULDER = 0, 1
L_ELBOW,    R_ELBOW    = 2, 3
L_WRIST,    R_WRIST    = 4, 5
L_HIP,      R_HIP      = 6, 7
L_KNEE,     R_KNEE     = 8, 9
L_ANKLE,    R_ANKLE    = 10, 11
HEAD,       NECK       = 12, 13

EJERCICIOS: dict = {

    # ------------------------------------------------------------------
    # MEDIA SENTADILLA  — rodilla baja: 70°–100°
    # ------------------------------------------------------------------
    "media_sentadilla": {
        "descripcion": "Sentadilla",
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

    # ------------------------------------------------------------------
    # JUMPING JACKS  — hombro cerrado: 10°–25°  |  hombro abierto: 150°–180°
    # Cámara: de frente
    # ------------------------------------------------------------------
    "jumping_jacks": {
        "descripcion": "Jumping jacks",
        "camara": "frente",
        "articulaciones": {
            "hombro_izq": {
                "keypoints": (L_ELBOW, L_SHOULDER, L_HIP),
                "fases": {
                    "cerrada": {
                        "rango": (10, 25),
                        "feedback_bajo": None,
                        "feedback_alto": "Bajá más el brazo izquierdo al cerrar",
                    },
                    "abierta": {
                        "rango": (150, 180),
                        "feedback_bajo": "Levantá más el brazo izquierdo",
                        "feedback_alto": None,
                    },
                },
            },
            "hombro_der": {
                "keypoints": (R_ELBOW, R_SHOULDER, R_HIP),
                "fases": {
                    "cerrada": {
                        "rango": (10, 25),
                        "feedback_bajo": None,
                        "feedback_alto": "Bajá más el brazo derecho al cerrar",
                    },
                    "abierta": {
                        "rango": (150, 180),
                        "feedback_bajo": "Levantá más el brazo derecho",
                        "feedback_alto": None,
                    },
                },
            },
        },
        "rep_tracking": {
            "articulacion": "hombro_izq",
            "umbral_inicio": 130,
            "umbral_fin":    40,
        },
    },

    # ------------------------------------------------------------------
    # PRESS DE HOMBROS  — hombro abajo: 70°–100°  |  hombro arriba: 160°–180°
    # codo abajo: 75°–100°  |  codo extendido arriba: 160°–180°
    # Cámara: de frente o perfil
    # ------------------------------------------------------------------
    "press_hombros": {
        "descripcion": "Press de hombros",
        "camara": "frente_o_perfil",
        "articulaciones": {
            "hombro_izq": {
                "keypoints": (L_ELBOW, L_SHOULDER, L_HIP),
                "fases": {
                    "abajo": {
                        "rango": (70, 100),
                        "feedback_bajo": "Bajá más el brazo izquierdo a la altura de hombro",
                        "feedback_alto": "Brazo izquierdo demasiado bajo para iniciar",
                    },
                    "arriba": {
                        "rango": (160, 180),
                        "feedback_bajo": "Extendé completamente el brazo izquierdo arriba",
                        "feedback_alto": None,
                    },
                },
            },
            "hombro_der": {
                "keypoints": (R_ELBOW, R_SHOULDER, R_HIP),
                "fases": {
                    "abajo": {
                        "rango": (70, 100),
                        "feedback_bajo": "Bajá más el brazo derecho a la altura de hombro",
                        "feedback_alto": "Brazo derecho demasiado bajo para iniciar",
                    },
                    "arriba": {
                        "rango": (160, 180),
                        "feedback_bajo": "Extendé completamente el brazo derecho arriba",
                        "feedback_alto": None,
                    },
                },
            },
            "codo_izq": {
                "keypoints": (L_SHOULDER, L_ELBOW, L_WRIST),
                "fases": {
                    "abajo": {
                        "rango": (75, 100),
                        "feedback_bajo": "Codo izquierdo muy cerrado al iniciar",
                        "feedback_alto": None,
                    },
                    "arriba": {
                        "rango": (160, 180),
                        "feedback_bajo": "Extendé completamente el codo izquierdo arriba",
                        "feedback_alto": None,
                    },
                },
            },
            "codo_der": {
                "keypoints": (R_SHOULDER, R_ELBOW, R_WRIST),
                "fases": {
                    "abajo": {
                        "rango": (75, 100),
                        "feedback_bajo": "Codo derecho muy cerrado al iniciar",
                        "feedback_alto": None,
                    },
                    "arriba": {
                        "rango": (160, 180),
                        "feedback_bajo": "Extendé completamente el codo derecho arriba",
                        "feedback_alto": None,
                    },
                },
            },
        },
        "rep_tracking": {
        "articulacion": "codo_izq",
        "umbral_inicio": 60,   # solo entra si el codo está muy cerrado
        "umbral_fin":    160,  # solo completa si el brazo está casi completamente extendido
        },
    },

    # ------------------------------------------------------------------
    # PESO MUERTO A UNA PIERNA  — cadera parado: 160°–180°  |  inclinado: 80°–100°
    # rodilla apoyo: leve flexión 155°–175°  |  columna: 165°–180° (espalda recta)
    # Cámara: perfil
    # ------------------------------------------------------------------
    "peso_muerto_una_pierna": {
        "descripcion": "Peso muerto a una pierna",
        "camara": "perfil",
        "articulaciones": {
            "cadera": {
                "keypoints": (L_SHOULDER, L_HIP, L_KNEE),
                "fases": {
                    "parado": {
                        "rango": (160, 180),
                        "feedback_bajo": "Terminá de extender la cadera al subir",
                        "feedback_alto": None,
                    },
                    "inclinado": {
                        "rango": (80, 100),
                        "feedback_bajo": "Te inclinaste demasiado — cuidado con la espalda",
                        "feedback_alto": "Inclinate más desde la cadera",
                    },
                },
            },
            "rodilla_apoyo": {
                "keypoints": (L_HIP, L_KNEE, L_ANKLE),
                "fases": {
                    "inclinado": {
                        "rango": (155, 175),
                        "feedback_bajo": "Rodilla de apoyo muy flexionada — mantenela casi recta",
                        "feedback_alto": None,
                    },
                },
            },
            "columna": {
                "keypoints": (NECK, L_SHOULDER, L_HIP),
                "fases": {
                    "inclinado": {
                        "rango": (165, 180),
                        "feedback_bajo": "Espalda redondeada — mantenela recta",
                        "feedback_alto": None,
                    },
                },
            },
        },
        "rep_tracking": {
            "articulacion": "cadera",
            "umbral_inicio": 140,
            "umbral_fin":    170,
        },
    }

}