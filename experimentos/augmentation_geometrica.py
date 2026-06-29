"""
Augmentation geometrica para pose estimation: flip horizontal (con swap L/R)
+ shift/scale leve, aplicados correctamente tanto a la imagen como a los keypoints.

Por que existe este archivo:
torchvision.transforms.RandomResizedCrop / RandomRotation NO saben que existen
keypoints asociados a la imagen: transforman la imagen pero dejan el target
intacto, lo que descalibra sistematicamente el entrenamiento (confirmado con
diagnostico_transform.py). Esta funcion reemplaza esas dos transforms,
aplicando la MISMA transformacion geometrica a imagen y coordenadas.

Sin rotacion: shift/scale + flip cubren la mayor parte del beneficio de
augmentation geometrica con mucha menos superficie de bugs.
"""

import random
from PIL import Image

KP_NAMES = [
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
    "head", "neck"
]

# Pares que deben intercambiarse al hacer flip horizontal.
# head y neck no se swapean (son centrales).
FLIP_PAIRS = [
    ("left_shoulder", "right_shoulder"),
    ("left_elbow", "right_elbow"),
    ("left_wrist", "right_wrist"),
    ("left_hip", "right_hip"),
    ("left_knee", "right_knee"),
    ("left_ankle", "right_ankle"),
]


def aplicar_augmentation_geometrica(img, kps_dict, out_size=224,
                                     p_flip=0.5,
                                     scale_range=(0.85, 1.0),
                                     shift_jitter=0.05):
    """
    img: PIL Image RGB (el crop de persona ya generado por crowdpose_a_csv.py)
    kps_dict: dict {kp_name: (x_norm, y_norm)}, normalizados [0,1] respecto a img.
              Usar (-1.0, -1.0) para keypoints invisibles/ausentes.
    out_size: tamaño final cuadrado (224 para que coincida con el resto del pipeline)
    p_flip: probabilidad de flip horizontal
    scale_range: rango de tamaño del sub-recorte, como fraccion del crop original.
                 1.0 = sin zoom in, 0.85 = hasta 15% mas cerca.
    shift_jitter: cuanto puede desplazarse el centro del sub-recorte respecto
                  al centro geometrico disponible (fraccion del margen libre).

    Devuelve: (img_aug PIL Image out_size x out_size, kps_dict_aug)
    Los keypoints que quedan fuera del nuevo recorte se marcan (-1.0, -1.0).
    """
    w, h = img.size
    kps = dict(kps_dict)

    # --- 1. Flip horizontal ---
    if random.random() < p_flip:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        flipped = {}
        for kp, (x, y) in kps.items():
            if x == -1.0 and y == -1.0:
                flipped[kp] = (-1.0, -1.0)
            else:
                flipped[kp] = (1.0 - x, y)
        for left, right in FLIP_PAIRS:
            flipped[left], flipped[right] = flipped[right], flipped[left]
        kps = flipped

    # --- 2. Shift + scale (reemplaza RandomResizedCrop, sin desalinear targets) ---
    scale = random.uniform(*scale_range)
    max_margin = 1.0 - scale  # cuanto margen libre hay para mover el recuadro
    if max_margin > 0:
        shift_x = max_margin / 2 + random.uniform(-shift_jitter, shift_jitter) * max_margin
        shift_y = max_margin / 2 + random.uniform(-shift_jitter, shift_jitter) * max_margin
        shift_x = min(max(shift_x, 0.0), max_margin)
        shift_y = min(max(shift_y, 0.0), max_margin)
    else:
        shift_x = shift_y = 0.0

    x0, y0 = shift_x, shift_y
    x1, y1 = x0 + scale, y0 + scale

    px0, py0, px1, py1 = x0 * w, y0 * h, x1 * w, y1 * h
    img = img.crop((px0, py0, px1, py1)).resize((out_size, out_size), Image.BILINEAR)

    kps_out = {}
    for kp, (x, y) in kps.items():
        if x == -1.0 and y == -1.0:
            kps_out[kp] = (-1.0, -1.0)
            continue
        new_x = (x - x0) / scale
        new_y = (y - y0) / scale
        if 0.0 <= new_x <= 1.0 and 0.0 <= new_y <= 1.0:
            kps_out[kp] = (new_x, new_y)
        else:
            kps_out[kp] = (-1.0, -1.0)

    return img, kps_out


def kps_dict_desde_row(row, kp_names=KP_NAMES):
    """Helper: arma el dict {kp_name: (x,y)} a partir de una fila de pandas
    con columnas {kp}_x, {kp}_y, como las que genera crowdpose_a_csv.py."""
    return {
        kp: (float(row[f"{kp}_x"]), float(row[f"{kp}_y"]))
        for kp in kp_names
    }