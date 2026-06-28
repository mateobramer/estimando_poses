"""
crowdpose_a_csv.py — Convierte anotaciones de CrowdPose al formato CSV
que esperan entrenar_modelo_a.py / entrenar_modelo_b.py.

QUE HACE, paso a paso, por cada anotacion (persona) del JSON:
    1. Lee el bbox [x, y, w, h] de esa persona (coordenadas absolutas
       sobre la imagen original).
    2. Agranda el bbox con un margen (igual que hace inferencia_yolo.py
       al recortar con YOLO) para no cortar manos/pies justo en el borde.
    3. Recorta esa region de la imagen original y la guarda como un
       nuevo JPG (una imagen por persona, no por foto original).
    4. Convierte los 14 keypoints de pixeles-absolutos (sobre la foto
       completa) a coordenadas normalizadas 0-1 RELATIVAS AL RECORTE,
       que es el formato que esperan tus modelos (mismo sistema que usa
       inferencia_yolo.py para dibujar sobre cada bounding box).
    5. Si un keypoint tiene v=0 (no etiquetado), lo guarda como -1,-1
       para que PoseDataset/PoseDatasetB lo detecten como invisible
       (igual chequeo que ya hacen: "0.0 <= v <= 1.0").
    6. Si la persona tiene 0 keypoints anotados (num_keypoints=0) o el
       bbox cae fuera de la imagen, se descarta esa anotacion entera.

El orden de KP_NAMES coincide EXACTO con categories[0]["keypoints"] de
CrowdPose, así que no hace falta ningun mapeo/reordenamiento.

USO:
    python crowdpose_a_csv.py --json crowdpose_val.json   --images images --out_csv val_14kp.csv   --out_frames frames_val
    python crowdpose_a_csv.py --json crowdpose_train.json --images images --out_csv train_14kp.csv --out_frames frames_train
    python crowdpose_a_csv.py --json crowdpose_test.json  --images images --out_csv test_14kp.csv  --out_frames frames_test

    # para probar rapido con pocas imagenes antes de correr todo:
    python crowdpose_a_csv.py --json crowdpose_val.json --images images --out_csv val_14kp.csv --out_frames frames_val --max_samples 500

SALIDA:
    <out_frames>/<image_id>_<ann_id>.jpg   (un recorte por persona)
    <out_csv>                              (una fila por persona, columnas:
                                             imagen, left_shoulder_x, left_shoulder_y, ...)
"""

import json
import os
import argparse
import csv as csv_module

import cv2

KP_NAMES = [
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
    "head", "neck"
]
N_KP = len(KP_NAMES)  # 14

MARGEN_FRAC = 0.10  # mismo margen que usa inferencia_yolo.py al recortar


def cargar_json(path):
    with open(path) as f:
        data = json.load(f)
    # nombre del archivo de categories debe matchear KP_NAMES; lo chequeamos
    # para frenar temprano si algun dia cambian el JSON.
    nombres_json = data["categories"][0]["keypoints"]
    if nombres_json != KP_NAMES:
        raise ValueError(
            f"El orden de keypoints del JSON no coincide con KP_NAMES.\n"
            f"JSON:  {nombres_json}\n"
            f"Local: {KP_NAMES}"
        )
    return data


def indexar_imagenes(data):
    """image_id -> dict con file_name, height, width"""
    return {img["id"]: img for img in data["images"]}


def expandir_bbox(x, y, w, h, img_w, img_h, margen_frac=MARGEN_FRAC):
    """Agranda el bbox con margen y lo recorta a los limites de la imagen."""
    margen = int(max(w, h) * margen_frac)
    x1 = max(0, int(x) - margen)
    y1 = max(0, int(y) - margen)
    x2 = min(img_w,  int(x + w) + margen)
    y2 = min(img_h,  int(y + h) + margen)
    return x1, y1, x2, y2


def convertir(args):
    print(f"Cargando {args.json}...")
    data = cargar_json(args.json)
    imagenes_por_id = indexar_imagenes(data)
    print(f"  {len(data['images'])} imagenes, {len(data['annotations'])} anotaciones")

    os.makedirs(args.out_frames, exist_ok=True)

    columnas = ["imagen"] + [f"{kp}_{c}" for kp in KP_NAMES for c in ("x", "y")]

    n_escritas = 0
    n_sin_imagen = 0
    n_sin_keypoints = 0
    n_bbox_invalido = 0
    cache_img = {}  # evita releer el mismo jpg para multiples personas

    with open(args.out_csv, "w", newline="") as f_csv:
        writer = csv_module.writer(f_csv)
        writer.writerow(columnas)

        for ann in data["annotations"]:
            if args.max_samples and n_escritas >= args.max_samples:
                break

            if ann.get("num_keypoints", 0) == 0:
                n_sin_keypoints += 1
                continue

            info_img = imagenes_por_id.get(ann["image_id"])
            if info_img is None:
                n_sin_imagen += 1
                continue

            file_name = info_img["file_name"]
            img_path = os.path.join(args.images, file_name)

            if img_path not in cache_img:
                img = cv2.imread(img_path)
                cache_img[img_path] = img
                if len(cache_img) > 50:  # cache chico, no acumular toda la RAM
                    cache_img.pop(next(iter(cache_img)))
            img = cache_img[img_path]

            if img is None:
                n_sin_imagen += 1
                continue

            img_h, img_w = img.shape[:2]
            bx, by, bw, bh = ann["bbox"]
            if bw <= 0 or bh <= 0:
                n_bbox_invalido += 1
                continue

            x1, y1, x2, y2 = expandir_bbox(bx, by, bw, bh, img_w, img_h)
            if x2 <= x1 or y2 <= y1:
                n_bbox_invalido += 1
                continue

            recorte = img[y1:y2, x1:x2]
            if recorte.size == 0:
                n_bbox_invalido += 1
                continue

            recorte_w = x2 - x1
            recorte_h = y2 - y1

            kp_flat = ann["keypoints"]  # [x0,y0,v0, x1,y1,v1, ...]
            fila_coords = []
            for i in range(N_KP):
                px, py, v = kp_flat[i*3], kp_flat[i*3 + 1], kp_flat[i*3 + 2]
                if v == 0:
                    # no etiquetado -> invalido, igual que hace PoseDataset
                    # al detectar valores fuera de [0,1]
                    fila_coords.extend([-1.0, -1.0])
                else:
                    # pixel absoluto (foto completa) -> normalizado relativo al recorte
                    x_norm = (px - x1) / recorte_w
                    y_norm = (py - y1) / recorte_h
                    fila_coords.extend([x_norm, y_norm])

            nombre_salida = f"{ann['image_id']}_{ann['id']}.jpg"
            cv2.imwrite(os.path.join(args.out_frames, nombre_salida), recorte)

            writer.writerow([nombre_salida] + fila_coords)
            n_escritas += 1

            if n_escritas % 1000 == 0:
                print(f"  ...{n_escritas} personas procesadas")

    print(f"\nListo: {args.out_csv}")
    print(f"  Personas escritas:        {n_escritas}")
    print(f"  Saltadas (sin keypoints): {n_sin_keypoints}")
    print(f"  Saltadas (sin imagen):    {n_sin_imagen}")
    print(f"  Saltadas (bbox invalido): {n_bbox_invalido}")
    print(f"  Recortes guardados en:    {args.out_frames}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json",        required=True, help="ej: crowdpose_val.json")
    parser.add_argument("--images",      required=True, help="carpeta con los jpg originales de CrowdPose")
    parser.add_argument("--out_csv",     required=True, help="ej: val_14kp.csv")
    parser.add_argument("--out_frames",  required=True, help="carpeta donde guardar los recortes por persona")
    parser.add_argument("--max_samples", type=int, default=None,
                         help="cortar despues de N personas (para pruebas rapidas)")
    args = parser.parse_args()
    convertir(args)


if __name__ == "__main__":
    main()