"""
Avatar 3D con RealSense D435i.
Usa keypoints del modelo para definir segmentos del cuerpo
y agarra los pixeles de profundidad cerca de cada segmento.

USO:
    python avatar_3d_realsense.py --modelo modelo_a_fix_mejor.pth
    python avatar_3d_realsense.py --modelo modelo_a_fix_mejor.pth --paso 8
"""

import torch
import torch.nn as nn
import timm
from torchvision import transforms
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import pyrealsense2 as rs
import argparse
import threading
import time
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import io
import open3d as o3d


# One Euro Filter
class OneEuroFilter:
    def __init__(self, freq=30.0, min_cutoff=1.0, beta=0.01, d_cutoff=1.0):
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    def _alpha(self, cutoff):
        te = 1.0 / self.freq
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x, t=None):
        if t is None:
            t = time.time()
        if self.x_prev is None:
            self.x_prev = x
            self.t_prev = t
            return x
        dt = t - self.t_prev
        if dt > 0:
            self.freq = 1.0 / dt
        self.t_prev = t
        dx = (x - self.x_prev) * self.freq
        a_d = self._alpha(self.d_cutoff)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat


class PoseFilter:
    def __init__(self, n_kp=14, freq=30.0, min_cutoff=1.5, beta=0.05):
        self.filters = [OneEuroFilter(freq=freq, min_cutoff=min_cutoff, beta=beta)
                        for _ in range(n_kp * 2)]

    def __call__(self, coords):
        t = time.time()
        return np.array([f(coords[i], t) for i, f in enumerate(self.filters)],
                        dtype=np.float32)

KP_NAMES = [
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
    "head", "neck"
]

SKELETON = [
    (12, 13), (13, 0), (13, 1),
    (0, 2), (2, 4), (1, 3), (3, 5),
    (0, 6), (1, 7), (6, 7),
    (6, 8), (8, 10), (7, 9), (9, 11),
]

# segmentos: (kp_a, kp_b, radio_metros, color_rgb)
SEGMENTOS = [
    (12, 13, 0.10, (1.0, 0.8, 0.0)),   # cabeza-cuello
    (13,  0, 0.07, (0.0, 0.8, 1.0)),   # cuello-hombro izq
    (13,  1, 0.07, (0.0, 0.8, 1.0)),   # cuello-hombro der
    ( 0,  2, 0.06, (0.0, 1.0, 0.4)),   # hombro-codo izq
    ( 1,  3, 0.06, (0.0, 1.0, 0.4)),   # hombro-codo der
    ( 2,  4, 0.05, (0.2, 0.6, 1.0)),   # codo-muñeca izq
    ( 3,  5, 0.05, (0.2, 0.6, 1.0)),   # codo-muñeca der
    ( 0,  6, 0.09, (0.8, 0.2, 1.0)),   # hombro-cadera izq
    ( 1,  7, 0.09, (0.8, 0.2, 1.0)),   # hombro-cadera der
    ( 6,  7, 0.10, (0.8, 0.2, 1.0)),   # cadera-cadera
    ( 6,  8, 0.08, (1.0, 0.2, 0.2)),   # cadera-rodilla izq
    ( 7,  9, 0.08, (1.0, 0.2, 0.2)),   # cadera-rodilla der
    ( 8, 10, 0.06, (1.0, 0.6, 0.2)),   # rodilla-tobillo izq
    ( 9, 11, 0.06, (1.0, 0.6, 0.2)),   # rodilla-tobillo der
]

N_KP = 14
UMBRAL_VIS = 0.0


# ── Modelo ────────────────────────────────────────────────────────
class PoseModel(nn.Module):
    def __init__(self, backbone_name="mobilenetv2_100", n_keypoints=14):
        super().__init__()
        self.backbone    = timm.create_model(backbone_name, pretrained=False, num_classes=0)
        n_features       = self.backbone.num_features
        self.shared      = nn.Sequential(nn.Linear(n_features, 512), nn.ReLU(), nn.Dropout(0.3))
        self.head_coords = nn.Sequential(nn.Linear(512, n_keypoints * 2), nn.Sigmoid())
        self.head_vis    = nn.Sequential(nn.Linear(512, n_keypoints),     nn.Sigmoid())

    def forward(self, x):
        x = self.backbone(x)
        x = self.shared(x)
        return self.head_coords(x), self.head_vis(x)


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def predecir(modelo, recorte, device):
    img    = Image.fromarray(cv2.cvtColor(recorte, cv2.COLOR_BGR2RGB))
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        coords, vis = modelo(tensor)
    return coords[0].cpu().numpy(), vis[0].cpu().numpy()


def pixel_a_3d(x_px, y_px, depth_frame, intrinsics):
    x_px = int(min(max(x_px, 0), intrinsics.width  - 1))
    y_px = int(min(max(y_px, 0), intrinsics.height - 1))
    z = depth_frame.get_distance(x_px, y_px)
    if z < 0.1 or z > 5.0:
        return None
    return rs.rs2_deproject_pixel_to_point(intrinsics, [float(x_px), float(y_px)], z)


def dist_punto_segmento(p, a, b):
    """Distancia de punto p al segmento a-b en 3D."""
    pa = np.array(p) - np.array(a)
    ba = np.array(b) - np.array(a)
    h  = np.clip(np.dot(pa, ba) / (np.dot(ba, ba) + 1e-9), 0.0, 1.0)
    return np.linalg.norm(pa - h * ba), h


def construir_nube(depth_frame, color_frame_np, intrinsics, pts_3d, radio_base, paso):
    """Construye nube de puntos por segmento con colores reales del pixel."""
    h_img = intrinsics.height
    w_img = intrinsics.width
    fx, fy = intrinsics.fx, intrinsics.fy
    cx, cy = intrinsics.ppx, intrinsics.ppy

    scale       = depth_frame.get_units()
    depth_array = np.asanyarray(depth_frame.get_data()).astype(np.float32) * scale

    # subsampling
    ys = np.arange(0, h_img, paso)
    xs = np.arange(0, w_img, paso)
    xx, yy = np.meshgrid(xs, ys)
    zz = depth_array[yy, xx]

    # filtrar profundidades invalidas
    mask = (zz > 0.1) & (zz < 5.0)
    xx, yy, zz = xx[mask], yy[mask], zz[mask]

    # colores reales de cada pixel (BGR -> RGB, normalizado 0-1)
    colores_rgb = color_frame_np[yy, xx, ::-1].astype(np.float32) / 255.0

    # proyectar todos los pixeles a 3D de una vez (vectorizado)
    px = (xx - cx) / fx * zz
    py = (yy - cy) / fy * zz
    pz = zz

    nube_xyz    = np.stack([px, py, pz], axis=1)

    nubes = []
    for a_idx, b_idx, radio, color in SEGMENTOS:
        pa = pts_3d[a_idx]
        pb = pts_3d[b_idx]
        if pa is None or pb is None:
            continue

        pa_np = np.array(pa, dtype=np.float32)
        pb_np = np.array(pb, dtype=np.float32)
        ab = pb_np - pa_np
        largo = np.linalg.norm(ab)
        if largo < 0.01:
            continue

        ap = nube_xyz - pa_np
        t  = np.clip(ap @ ab / (ab @ ab + 1e-9), 0.0, 1.0)
        proj = pa_np + t[:, None] * ab
        dist = np.linalg.norm(nube_xyz - proj, axis=1)

        radio_local = radio * (1.0 + 0.3 * (1.0 - np.abs(2*t - 1)))
        sel = dist < radio_local

        z_min = min(pa_np[2], pb_np[2]) - 0.20
        z_max = max(pa_np[2], pb_np[2]) + 0.20
        sel = sel & (nube_xyz[:, 2] > z_min) & (nube_xyz[:, 2] < z_max)

        if sel.sum() > 0:
            # usar colores reales de los pixeles seleccionados
            nubes.append((nube_xyz[sel], colores_rgb[sel]))

    # torso solido: rectangulo 3D entre hombros y caderas
    p_hl = pts_3d[0]   # hombro izq
    p_hr = pts_3d[1]   # hombro der
    p_cl = pts_3d[6]   # cadera izq
    p_cr = pts_3d[7]   # cadera der

    if all(p is not None for p in [p_hl, p_hr, p_cl, p_cr]):
        p_hl = np.array(p_hl, dtype=np.float32)
        p_hr = np.array(p_hr, dtype=np.float32)
        p_cl = np.array(p_cl, dtype=np.float32)
        p_cr = np.array(p_cr, dtype=np.float32)

        # ejes del rectangulo del torso
        eje_h = p_hr - p_hl  # hombro izq → hombro der
        eje_v = p_cl - p_hl  # hombro izq → cadera izq

        largo_h = np.linalg.norm(eje_h)
        largo_v = np.linalg.norm(eje_v)

        if largo_h > 0.01 and largo_v > 0.01:
            eje_h_n = eje_h / largo_h
            eje_v_n = eje_v / largo_v

            # proyectar todos los puntos en el sistema de coordenadas del torso
            rel = nube_xyz - p_hl
            u   = rel @ eje_h_n  # coordenada horizontal (0=hombro izq, largo_h=hombro der)
            v   = rel @ eje_v_n  # coordenada vertical   (0=hombros, largo_v=caderas)

            # profundidad del torso: promedio de los 4 keypoints ± margen
            z_torso = np.mean([p_hl[2], p_hr[2], p_cl[2], p_cr[2]])
            margen_z = 0.18  # 18cm de profundidad

            sel_torso = (
                (u >= -0.02) & (u <= largo_h + 0.02) &
                (v >= -0.02) & (v <= largo_v + 0.02) &
                (nube_xyz[:, 2] > z_torso - margen_z) &
                (nube_xyz[:, 2] < z_torso + margen_z)
            )

            if sel_torso.sum() > 0:
                nubes.append((nube_xyz[sel_torso], colores_rgb[sel_torso]))

    return nubes


def construir_pcd_o3d(nubes, pts_3d):
    """Convierte la nube de puntos a formato Open3D."""
    todos_pts    = []
    todos_colores = []

    for puntos, colores in nubes:
        if len(puntos) == 0:
            continue
        # subsamplear para no saturar
        idx = np.random.choice(len(puntos), min(len(puntos), 500), replace=False)
        todos_pts.append(puntos[idx])
        todos_colores.append(colores[idx])

    # agregar keypoints como puntos amarillos grandes
    if pts_3d:
        for p in pts_3d:
            if p is not None:
                todos_pts.append(np.array([[p[0], p[1], p[2]]]))
                todos_colores.append(np.array([[1.0, 1.0, 0.0]]))

    if not todos_pts:
        return None

    pts_np    = np.vstack(todos_pts)
    colores_np = np.vstack(todos_colores)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_np * np.array([1, -1, -1]))
    pcd.colors = o3d.utility.Vector3dVector(np.clip(colores_np, 0, 1))
    return pcd



def dibujar_2d(frame, coords, vis, x1, y1, x2, y2):
    w_box = x2 - x1
    h_box = y2 - y1
    pts, visible = [], []
    for i in range(N_KP):
        kp_x = int(coords[i*2]   * w_box + x1)
        kp_y = int(coords[i*2+1] * h_box + y1)
        pts.append((kp_x, kp_y))
        visible.append(float(vis[i]) >= UMBRAL_VIS)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 100, 255), 2)
    for a, b in SKELETON:
        if visible[a] and visible[b]:
            cv2.line(frame, pts[a], pts[b], (0, 200, 255), 2)
    for i, (x, y) in enumerate(pts):
        if visible[i]:
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelo",   default="modelo_a_fix_mejor.pth")
    parser.add_argument("--backbone", default="mobilenetv2_100")
    parser.add_argument("--conf",     type=float, default=0.5)
    parser.add_argument("--radio",    type=float, default=0.08)
    parser.add_argument("--paso",     type=int,   default=6)
    args = parser.parse_args()

    device = torch.device("cpu")
    pose_filter = PoseFilter()

    print("Cargando YOLO...")
    yolo = YOLO("yolov8n.pt")

    print(f"Cargando modelo: {args.modelo}...")
    modelo = PoseModel(backbone_name=args.backbone).to(device)
    modelo.load_state_dict(torch.load(args.modelo, map_location=device))
    modelo.eval()

    print("Iniciando RealSense...")
    pipeline = rs.pipeline()
    config   = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16,  30)
    profile  = pipeline.start(config)
    align    = rs.align(rs.stream.color)

    color_stream = profile.get_stream(rs.stream.color)
    intrinsics   = color_stream.as_video_stream_profile().get_intrinsics()
    print(f"RealSense OK: {intrinsics.width}x{intrinsics.height}")
    print(f"Paso: {args.paso} | Radio: {args.radio}m")
    print("Q o Escape para salir.")

    img_3d = np.zeros((500, 500, 3), dtype=np.uint8)

    # visualizador Open3D
    vis3d = o3d.visualization.Visualizer()
    vis3d.create_window("Nube de Puntos 3D - RealSense", width=640, height=640)
    vis3d.get_render_option().background_color = np.array([0.05, 0.05, 0.1])
    vis3d.get_render_option().point_size = 3.0
    pcd_actual  = None
    primera_vez = True
    frame_count = 0

    try:
        while True:
            t0     = time.time()
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())
            resultados = yolo(frame, classes=[0], conf=args.conf, verbose=False)

            for box in resultados[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                margen = int(max(x2-x1, y2-y1) * 0.1)
                x1 = max(0, x1 - margen)
                y1 = max(0, y1 - margen)
                x2 = min(frame.shape[1], x2 + margen)
                y2 = min(frame.shape[0], y2 + margen)
                recorte = frame[y1:y2, x1:x2]
                if recorte.size == 0:
                    continue

                coords, vis = predecir(modelo, recorte, device)
                coords = pose_filter(coords)  # suavizar con One Euro Filter

                # keypoints 3D
                pts_3d = []
                w_box, h_box = x2-x1, y2-y1
                for i in range(N_KP):
                    if float(vis[i]) < UMBRAL_VIS:
                        pts_3d.append(None)
                        continue
                    kp_x = coords[i*2]   * w_box + x1
                    kp_y = coords[i*2+1] * h_box + y1
                    pts_3d.append(pixel_a_3d(kp_x, kp_y, depth_frame, intrinsics))

                # nube de puntos
                nubes = construir_nube(depth_frame, frame, intrinsics, pts_3d,
                                       args.radio, args.paso)

                # actualizar Open3D
                nueva_pcd = construir_pcd_o3d(nubes, pts_3d)
                if nueva_pcd is not None:
                    if pcd_actual is not None:
                        vis3d.remove_geometry(pcd_actual, reset_bounding_box=False)

                    reset = primera_vez or (frame_count % 50 == 0)
                    vis3d.add_geometry(nueva_pcd, reset_bounding_box=reset)
                    pcd_actual  = nueva_pcd
                    primera_vez = False
                    frame_count += 1

                frame = dibujar_2d(frame, coords, vis, x1, y1, x2, y2)
                break

            fps = 1.0 / (time.time() - t0 + 1e-6)
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow("2D - RealSense", frame)
            vis3d.poll_events()
            vis3d.update_renderer()

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break

    finally:
        pipeline.stop()
        vis3d.destroy_window()
        cv2.destroyAllWindows()
        print("Listo.")


if __name__ == "__main__":
    main()