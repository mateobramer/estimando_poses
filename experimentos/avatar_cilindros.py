"""
Avatar 3D con cilindros usando Open3D + RealSense D435i.
Cada segmento del esqueleto se dibuja como un cilindro 3D.
Resultado: mannequin articulado que se mueve en tiempo real.

USO:
    python avatar_cilindros.py --modelo modelo_a_fix_mejor.pth
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
import open3d as o3d
import argparse
import time
import math

KP_NAMES = [
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
    "head", "neck"
]

# colores mannequin: gris azulado uniforme por zona
C_CABEZA  = (0.9, 0.9, 1.0)
C_HOMBROS = (0.5, 0.7, 1.0)
C_BRAZOS  = (0.4, 0.6, 0.9)
C_ANTEBR  = (0.3, 0.5, 0.8)
C_TORSO   = (0.5, 0.5, 0.9)
C_PIERNAS = (0.3, 0.4, 0.8)
C_PANTORR = (0.2, 0.3, 0.7)

# segmentos: (kp_a, kp_b, radio_metros, color_rgb 0-1)
# la cabeza no va aca, se dibuja como esfera aparte
SEGMENTOS = [
    (13,  0, 0.05, C_HOMBROS),  # cuello-hombro izq
    (13,  1, 0.05, C_HOMBROS),  # cuello-hombro der
    ( 0,  2, 0.04, C_BRAZOS),   # hombro-codo izq
    ( 1,  3, 0.04, C_BRAZOS),   # hombro-codo der
    ( 2,  4, 0.03, C_ANTEBR),   # codo-muñeca izq
    ( 3,  5, 0.03, C_ANTEBR),   # codo-muñeca der
    ( 0,  6, 0.06, C_TORSO),    # hombro-cadera izq
    ( 1,  7, 0.06, C_TORSO),    # hombro-cadera der
    ( 6,  7, 0.07, C_TORSO),    # cadera-cadera
    ( 6,  8, 0.05, C_PIERNAS),  # cadera-rodilla izq
    ( 7,  9, 0.05, C_PIERNAS),  # cadera-rodilla der
    ( 8, 10, 0.04, C_PANTORR),  # rodilla-tobillo izq
    ( 9, 11, 0.04, C_PANTORR),  # rodilla-tobillo der
]

N_KP = 14
UMBRAL_VIS = 0.0


# ── One Euro Filter ───────────────────────────────────────────────
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


def cilindro_entre_puntos(pa, pb, radio, color, resolucion=10):
    """Crea un cilindro Open3D entre dos puntos 3D."""
    pa = np.array(pa)
    pb = np.array(pb)
    largo = np.linalg.norm(pb - pa)
    if largo < 0.01:
        return None

    # crear cilindro base (eje Z)
    cil = o3d.geometry.TriangleMesh.create_cylinder(
        radius=radio, height=largo, resolution=resolucion, split=1
    )
    cil.paint_uniform_color(color)
    cil.compute_vertex_normals()

    # rotar para alinear eje Z con dirección pa→pb
    eje_z   = np.array([0, 0, 1], dtype=np.float64)
    dir_ab  = (pb - pa) / largo
    eje_rot = np.cross(eje_z, dir_ab)
    norma   = np.linalg.norm(eje_rot)

    if norma > 1e-6:
        eje_rot /= norma
        angulo   = np.arccos(np.clip(np.dot(eje_z, dir_ab), -1.0, 1.0))
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(eje_rot * angulo)
        cil.rotate(R, center=[0, 0, 0])

    # trasladar al punto medio
    centro = (pa + pb) / 2
    cil.translate(centro)

    return cil


def esfera_en_punto(p, radio, color):
    """Crea una esfera Open3D en un punto 3D."""
    esfera = o3d.geometry.TriangleMesh.create_sphere(radius=radio, resolution=8)
    esfera.paint_uniform_color(color)
    esfera.compute_vertex_normals()
    esfera.translate(np.array(p))
    return esfera


def construir_avatar(pts_3d):
    """Construye el avatar 3D con cilindros, esferas y cabeza grande."""
    geometrias = []

    # cilindros del esqueleto
    for a_idx, b_idx, radio, color in SEGMENTOS:
        pa = pts_3d[a_idx]
        pb = pts_3d[b_idx]
        if pa is None or pb is None:
            continue
        cil = cilindro_entre_puntos(pa, pb, radio, color)
        if cil is not None:
            geometrias.append(cil)

    # esfera grande para la cabeza
    p_head = pts_3d[12]  # keypoint head
    p_neck = pts_3d[13]  # keypoint neck
    if p_head is not None:
        # radio de la cabeza basado en distancia head-neck si disponible
        if p_neck is not None:
            radio_cabeza = np.linalg.norm(np.array(p_head) - np.array(p_neck)) * 0.6
            radio_cabeza = np.clip(radio_cabeza, 0.08, 0.14)
        else:
            radio_cabeza = 0.11
        cabeza = esfera_en_punto(p_head, radio_cabeza, C_CABEZA)
        geometrias.append(cabeza)

        # cara: calcular direccion hacia camara (Z negativo = hacia camara)
        # los features se colocan en la parte frontal de la cabeza
        ph = np.array(p_head)
        r  = radio_cabeza

        # direccion frontal estimada: hacia la camara (Z negativo)
        front = np.array([0.0, 0.0, -1.0])
        up    = np.array([0.0, -1.0, 0.0])  # Y negativo = arriba
        right = np.cross(up, front)

        # ojo izquierdo
        pos_ojo_izq = ph + front * r * 0.85 + right * r * 0.35 + up * r * 0.2
        ojo_izq = esfera_en_punto(pos_ojo_izq, r * 0.18, (0.1, 0.1, 0.15))
        geometrias.append(ojo_izq)
        # pupila izquierda
        pupila_izq = esfera_en_punto(pos_ojo_izq + front * r * 0.05, r * 0.09, (0.05, 0.05, 0.05))
        geometrias.append(pupila_izq)

        # ojo derecho
        pos_ojo_der = ph + front * r * 0.85 - right * r * 0.35 + up * r * 0.2
        ojo_der = esfera_en_punto(pos_ojo_der, r * 0.18, (0.1, 0.1, 0.15))
        geometrias.append(ojo_der)
        # pupila derecha
        pupila_der = esfera_en_punto(pos_ojo_der + front * r * 0.05, r * 0.09, (0.05, 0.05, 0.05))
        geometrias.append(pupila_der)

        # boca (cilindro horizontal fino)
        pos_boca_izq = ph + front * r * 0.88 + right * r * 0.25 - up * r * 0.25
        pos_boca_der = ph + front * r * 0.88 - right * r * 0.25 - up * r * 0.25
        boca = cilindro_entre_puntos(pos_boca_izq, pos_boca_der, r * 0.06, (0.2, 0.05, 0.05))
        if boca is not None:
            geometrias.append(boca)

        # nariz (cono pequeño)
        pos_nariz = ph + front * r * 0.95 - up * r * 0.05
        nariz = esfera_en_punto(pos_nariz, r * 0.1, (0.85, 0.75, 0.7))
        geometrias.append(nariz)

    # cuello como cilindro corto entre head y neck
    if p_head is not None and p_neck is not None:
        cil_cuello = cilindro_entre_puntos(p_head, p_neck, 0.04, C_CABEZA)
        if cil_cuello is not None:
            geometrias.append(cil_cuello)

    # esferas pequeñas en articulaciones (excepto head, neck, muñecas y tobillos)
    kps_sin_extremidades = {"head", "neck", "left_wrist", "right_wrist",
                            "left_ankle", "right_ankle"}
    for i, p in enumerate(pts_3d):
        if p is None or KP_NAMES[i] in kps_sin_extremidades:
            continue
        esfera = esfera_en_punto(p, 0.025, (0.6, 0.7, 0.9))
        geometrias.append(esfera)

    # MANOS: esfera palma + 5 dedos
    for wrist_idx, elbow_idx in [(4, 2), (5, 3)]:
        p_wrist = pts_3d[wrist_idx]
        p_elbow = pts_3d[elbow_idx]
        if p_wrist is None:
            continue

        pw = np.array(p_wrist)

        # direccion del antebrazo (codo → muñeca)
        if p_elbow is not None:
            dir_brazo = pw - np.array(p_elbow)
            largo_brazo = np.linalg.norm(dir_brazo)
            if largo_brazo > 0.01:
                dir_brazo = dir_brazo / largo_brazo
            else:
                dir_brazo = np.array([0.0, 0.0, -1.0])
        else:
            dir_brazo = np.array([0.0, 0.0, -1.0])

        # palma: esfera achatada
        palma = esfera_en_punto(pw + dir_brazo * 0.04, 0.035, (0.85, 0.75, 0.65))
        geometrias.append(palma)

        # vectores perpendiculares para los dedos
        up_ref = np.array([0.0, -1.0, 0.0])
        lado   = np.cross(dir_brazo, up_ref)
        if np.linalg.norm(lado) < 0.01:
            lado = np.array([1.0, 0.0, 0.0])
        lado = lado / np.linalg.norm(lado)

        # 5 dedos
        offsets_lado   = [-0.04, -0.02, 0.0, 0.02, 0.04]
        largos_dedo    = [0.055, 0.07, 0.075, 0.065, 0.045]
        base_dedo      = pw + dir_brazo * 0.06

        for j in range(5):
            inicio = base_dedo + lado * offsets_lado[j]
            fin    = inicio + dir_brazo * largos_dedo[j]
            dedo   = cilindro_entre_puntos(inicio, fin, 0.008, (0.85, 0.75, 0.65))
            if dedo:
                geometrias.append(dedo)
            # punta del dedo
            punta = esfera_en_punto(fin, 0.009, (0.8, 0.7, 0.6))
            geometrias.append(punta)

    # ZAPATOS: elipsoide en tobillos apuntando segun direccion rodilla→tobillo
    for ankle_idx, knee_idx in [(10, 8), (11, 9)]:
        p_ankle = pts_3d[ankle_idx]
        p_knee  = pts_3d[knee_idx]
        if p_ankle is None:
            continue

        pa = np.array(p_ankle)

        # direccion del pie (rodilla → tobillo, proyectado al suelo)
        if p_knee is not None:
            dir_pierna = pa - np.array(p_knee)
            dir_pie = np.array([dir_pierna[0], 0.0, dir_pierna[2]])
            if np.linalg.norm(dir_pie) > 0.01:
                dir_pie = dir_pie / np.linalg.norm(dir_pie)
            else:
                dir_pie = np.array([0.0, 0.0, -1.0])
        else:
            dir_pie = np.array([0.0, 0.0, -1.0])

        # zapato: cilindro largo y bajo
        inicio_zapato = pa - dir_pie * 0.02
        fin_zapato    = pa + dir_pie * 0.09
        zapato = cilindro_entre_puntos(inicio_zapato, fin_zapato, 0.035, (0.15, 0.1, 0.1))
        if zapato:
            geometrias.append(zapato)
        # punta del zapato redondeada
        punta_zapato = esfera_en_punto(fin_zapato, 0.03, (0.15, 0.1, 0.1))
        geometrias.append(punta_zapato)
        # talon
        talon = esfera_en_punto(inicio_zapato, 0.03, (0.12, 0.08, 0.08))
        geometrias.append(talon)

    if not geometrias:
        return None

    avatar = geometrias[0]
    for g in geometrias[1:]:
        avatar += g

    return avatar


def calcular_altura(pts_3d):
    """Calcula la altura de la persona en metros usando head y tobillos."""
    p_head = pts_3d[12]  # head
    p_ankle_l = pts_3d[10]  # left_ankle
    p_ankle_r = pts_3d[11]  # right_ankle

    if p_head is None:
        return None

    # usar el tobillo disponible
    if p_ankle_l is not None and p_ankle_r is not None:
        p_ankle = [(p_ankle_l[i] + p_ankle_r[i]) / 2 for i in range(3)]
    elif p_ankle_l is not None:
        p_ankle = p_ankle_l
    elif p_ankle_r is not None:
        p_ankle = p_ankle_r
    else:
        return None

    # distancia 3D entre cabeza y tobillo
    altura = np.linalg.norm(np.array(p_head) - np.array(p_ankle))
    return round(altura, 2)


def construir_trails(historial_pts, opacidad_base=0.15):
    """Construye geometrias de trail con opacidad decreciente."""
    geometrias = []
    n = len(historial_pts)
    if n < 2:
        return geometrias

    for frame_idx, pts_3d in enumerate(historial_pts):
        if pts_3d is None:
            continue
        # opacidad: mas viejo = mas transparente
        alpha = opacidad_base + (1.0 - opacidad_base) * (frame_idx / max(n - 1, 1))
        color_trail = (alpha * 0.3, alpha * 0.6, alpha * 1.0)  # azul con alpha

        for a_idx, b_idx, radio, _ in SEGMENTOS:
            pa = pts_3d[a_idx]
            pb = pts_3d[b_idx]
            if pa is None or pb is None:
                continue
            radio_trail = radio * 0.4 * alpha  # mas fino que el avatar actual
            cil = cilindro_entre_puntos(pa, pb, max(radio_trail, 0.005), color_trail)
            if cil is not None:
                geometrias.append(cil)

    if not geometrias:
        return []

    trail = geometrias[0]
    for g in geometrias[1:]:
        trail += g
    return [trail]


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
    for a, b in [(12,13),(13,0),(13,1),(0,2),(2,4),(1,3),(3,5),(0,6),(1,7),(6,7),(6,8),(8,10),(7,9),(9,11)]:
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

    # Reintentar start() hasta que funcione
    profile = None
    print("Conectando a RealSense", end="", flush=True)
    for intento in range(15):
        try:
            profile = pipeline.start(config)
            break
        except RuntimeError:
            print(".", end="", flush=True)
            time.sleep(1)
            pipeline = rs.pipeline()   # reiniciar el pipeline en cada intento
    if profile is None:
        raise RuntimeError("No se pudo conectar a la cámara RealSense")
    print(" conectada!")

    align = rs.align(rs.stream.color)

    color_stream = profile.get_stream(rs.stream.color)
    intrinsics   = color_stream.as_video_stream_profile().get_intrinsics()
    print(f"RealSense OK: {intrinsics.width}x{intrinsics.height}")

    # visualizador Open3D
    vis = o3d.visualization.Visualizer()
    vis.create_window("Avatar 3D - RealSense", width=600, height=600)
    vis.get_render_option().background_color = np.array([0.05, 0.05, 0.1])
    vis.get_render_option().light_on = True

    avatar_actual  = None
    primera_vez    = True
    ultimo_pts_3d  = [None] * N_KP  # cache del ultimo keypoint valido

    print("Corriendo. Q en ventana 2D para salir.")

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

                coords, vis_kp = predecir(modelo, recorte, device)
                coords = pose_filter(coords)

                # keypoints 3D con cache del ultimo valido
                pts_3d = []
                w_box, h_box = x2-x1, y2-y1
                for i in range(N_KP):
                    if float(vis_kp[i]) >= UMBRAL_VIS:
                        kp_x = coords[i*2]   * w_box + x1
                        kp_y = coords[i*2+1] * h_box + y1
                        p3d  = pixel_a_3d(kp_x, kp_y, depth_frame, intrinsics)
                        if p3d is not None:
                            ultimo_pts_3d[i] = p3d  # actualizar cache
                        pts_3d.append(p3d if p3d is not None else ultimo_pts_3d[i])
                    else:
                        # usar ultimo valor valido si existe
                        pts_3d.append(ultimo_pts_3d[i])

                # construir avatar con cilindros
                nuevo_avatar = construir_avatar(pts_3d)

                if nuevo_avatar is not None:
                    if avatar_actual is not None:
                        vis.remove_geometry(avatar_actual, reset_bounding_box=False)
                    vis.add_geometry(nuevo_avatar, reset_bounding_box=primera_vez)
                    avatar_actual = nuevo_avatar

                    if primera_vez:
                        ctr = vis.get_view_control()
                        ctr.set_zoom(0.5)
                        ctr.set_front([0, 0, -1])
                        ctr.set_up([0, -1, 0])
                        primera_vez = False

                frame = dibujar_2d(frame, coords, vis_kp, x1, y1, x2, y2)

                # mostrar altura en el frame 2D y en terminal
                altura = calcular_altura(pts_3d)
                if altura is not None:
                    if not hasattr(main, "max_altura") or altura > main.max_altura:
                        main.max_altura = altura
                        print(f"Nueva altura maxima: {altura:.2f}m")
                    cv2.putText(frame, f"Altura: {altura:.2f}m", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    cv2.putText(frame, f"Max: {main.max_altura:.2f}m", (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)
                break

            fps = 1.0 / (time.time() - t0 + 1e-6)
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow("2D - RealSense", frame)
            vis.poll_events()
            vis.update_renderer()

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break

    finally:
        pipeline.stop()
        vis.destroy_window()
        cv2.destroyAllWindows()
        print("Listo.")


if __name__ == "__main__":
    main()