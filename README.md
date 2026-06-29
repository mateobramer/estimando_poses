# Estimación de Poses y Análisis de Ejercicio

**I308 - Visión Artificial · Universidad de San Andrés**  
Ana Holcman · Mateo Bramer

---

## ¿Qué hace este proyecto?

Sistema de análisis de ejercicio en tiempo real usando una red neuronal propia de pose estimation. La cámara captura el video, un modelo detecta las personas, otro modelo estima sus keypoints, y a partir de eso se calculan ángulos articulares, se aplica suavizado y se da feedback visual sobre la calidad de la sentadilla.

**Pipeline completo:**

```
Webcam → YOLO (detección de personas) → Modelo propio (keypoints) → Ángulos → One Euro Filter → Feedback en pantalla
```

Todo lo que va desde los keypoints en adelante (ángulos, filtro, análisis, visualización) fue implementado desde cero.

---

## Cómo correrlo

**Requisitos:** Python 3.11, PyTorch, timm, ultralytics, OpenCV, numpy, tkinter.

```bash
# Desde la raíz del proyecto
python app/inferencia_yolo.py
```

Esto abre la cámara, detecta personas con YOLO, estima keypoints con el modelo propio y muestra el overlay con feedback en tiempo real. Antes de arrancar aparece un menú de selección de ejercicio (`menu.py`).

**El modelo entrenado (`modelo_a_fix_mejor.pth`) debe estar en `modelos/`.** Si no está, descargarlo desde Google Drive:  
https://drive.google.com/drive/folders/1iXg-g5S0MyEKB1Qqc6zfA-tczKhg14rB?usp=drive_link

---

## Estructura del proyecto

```
estimando_poses/
├── app/               # Sistema en tiempo real — el producto final
├── entrenamiento/     # Scripts usados para entrenar en GCP
├── datos/             # Preparación del dataset CrowdPose
├── experimentos/      # Diagnósticos, comparaciones, cosas exploradas
├── modelos/           # Checkpoints entrenados (ver README interno)
├── logs/              # Salidas de entrenamiento
├── librealsense/      # SDK de Intel RealSense (integración pausada)
├── dataset_final/     # Dataset CrowdPose procesado (no va a git)
├── yolov8n.pt         # Modelo YOLO preentrenado para detección de personas
└── README.md          # Este archivo
```

---

## Carpetas en detalle

### `app/` — Sistema en tiempo real

Los módulos del sistema final. Cada uno tiene una sola responsabilidad.

| Archivo | Qué hace |
|---|---|
| `inferencia_yolo.py` | **Punto de entrada.** Loop principal: captura frames, corre YOLO, recorta personas, llama al modelo de pose, dibuja el resultado. |
| `detector_keypoints.py` | Carga el modelo entrenado y corre inferencia sobre un crop de persona. Devuelve 14 keypoints normalizados. |
| `analizador.py` | Calcula ángulos articulares a partir de keypoints, evalúa reglas de forma, cuenta repeticiones y corre DTW al terminar cada rep. |
| `angulos_referencia.py` | **Único lugar donde se definen los ejercicios.** Contiene los ángulos de referencia y reglas biomecánicas para cada ejercicio. Para agregar un ejercicio nuevo, solo hay que tocar este archivo. |
| `one_euro_filter.py` | Implementación del One Euro Filter desde cero. Suaviza los ángulos frame a frame: elimina ruido estático sin atrasar el movimiento real. |
| `visualizador.py` | Dibuja keypoints, esqueleto y feedback visual sobre el frame. |
| `grafico.py` | Renderiza el gráfico de ángulos en tiempo real en pantalla. |
| `menu.py` | Menú de selección de ejercicio con tkinter. Se muestra al iniciar. |

### `entrenamiento/` — Scripts de entrenamiento (corrieron en GCP)

Scripts usados para entrenar los modelos en la VM con GPU de Google Cloud. No se necesitan para correr el sistema, solo para re-entrenar.

| Archivo | Qué hace |
|---|---|
| `entrenar_gcp.py` | Entrena **Modelo A** — MobileNetV2 (timm) con regresión directa de coordenadas y Wing Loss. Este es el modelo que quedó. |
| `entrenar_gcp_b.py` | Entrena **Modelo B** — variante basada en heatmaps. Fue explorada y descartada. |
| `entrenar_modelob_fix.py` | Versión de Modelo B con fix de augmentación aplicado. |
| `modelo_b_timm.py` | Definición de arquitectura del Modelo B. |

**Nota importante sobre augmentación:** `torchvision.transforms` aplica transformaciones geométricas solo a la imagen, dejando las coordenadas de keypoints sin modificar. Esto genera un desalineamiento sistemático durante el entrenamiento. El fix consiste en aplicar las transformaciones conjuntamente a imagen y keypoints, y hacer swap de pares izquierda/derecha en los flips horizontales. Esta corrección está implementada en `entrenar_gcp.py` y es la razón por la que Modelo A supera a la baseline de Mateo.

### `datos/` — Preparación del dataset

Scripts y archivos usados para procesar el dataset CrowdPose.

| Archivo | Qué hace |
|---|---|
| `crowdpose_acsv.py` | Convierte las anotaciones de CrowdPose (formato JSON COCO) a CSV con coordenadas normalizadas relativas al crop de cada persona. |
| `filtrar_csv.py` | Filtra del CSV las entradas que referencian imágenes que no existen en disco (~17k imágenes faltantes en el dataset descargado). |
| `ver_14kp.py` | Script de visualización para verificar que los keypoints del CSV están correctamente alineados con las imágenes. |
| `crowdpose_train.json` / `val.json` / `test.json` / `trainval.json` | Anotaciones originales de CrowdPose. |
| `train_14kp.csv` / `val_14kp.csv` | CSVs procesados y filtrados listos para entrenar. 14 keypoints por persona: hombros, codos, muñecas, caderas, rodillas, tobillos, cabeza y cuello. |

### `experimentos/` — Exploración y diagnóstico

Scripts de diagnóstico, comparaciones y cosas que se exploraron pero no quedaron en el sistema final. Útil para entender el proceso de desarrollo.

| Archivo | Qué hace |
|---|---|
| `diagnostico_transform.py` | Script que visualiza el bug de augmentación: muestra que `torchvision` transforma la imagen pero no los keypoints. Sirvió para confirmar el problema antes de corregirlo. |
| `diagnostico_transforms.png` | Imagen de salida del diagnóstico anterior — evidencia visual del bug. |
| `comparar_augmentation.py` | Compara cuantitativamente el error del modelo con y sin el fix de augmentación. |
| `comparacion_aug.csv` | Resultados numéricos de esa comparación. |
| `augmentation_geometrica.py` | Implementación de las transformaciones geométricas correctas (imagen + keypoints juntos). |
| `analizar_modelo_b.ipynb` | Notebook de análisis del Modelo B: curvas de pérdida, predicciones, razones del abandono. |
| `inferencia_sin_yolo.py` | Versión anterior del pipeline sin detección de personas — corre el modelo de pose sobre el frame completo. |
| `correr_camara.py` | Script precursor de `inferencia_yolo.py`, usado para testear la cámara. |
| `avatar_cilindros.py` | Experimento de visualización con un avatar 3D en cilindros. No se integró al sistema final. |
| `ejemplo_uso.py` | Script de ejemplo para usar el detector de keypoints de forma standalone. |

### `modelos/` — Checkpoints

Los `.pth` están en Google Drive (no van a git por su tamaño):  
https://drive.google.com/drive/folders/1iXg-g5S0MyEKB1Qqc6zfA-tczKhg14rB?usp=drive_link

| Archivo | Descripción |
|---|---|
| `modelo_a_fix_mejor.pth` | ✅ **Modelo en uso.** Modelo A con fix de augmentación, mejor checkpoint por val loss. |
| `modelo_b_*.pth` | Checkpoints de Modelo B (heatmaps) en sus distintas variantes. Descartado. |

### `logs/` — Salidas de entrenamiento

| Archivo | Qué contiene |
|---|---|
| `log_gcp_a.csv` | Métricas por época del entrenamiento de Modelo A en GCP (loss train/val). |
| `log_train_completo.txt` | Log completo de texto del entrenamiento. |
| `log.txt` | Log auxiliar. |

---

## Flujo de desarrollo (cómo llegamos hasta acá)

1. **Dataset:** Descargamos CrowdPose, convertimos las anotaciones a CSV (`crowdpose_acsv.py`), filtramos imágenes faltantes (`filtrar_csv.py`) y verificamos visualmente (`ver_14kp.py`).

2. **Modelo A — baseline:** Entrenamos MobileNetV2 con regresión directa de coordenadas. Mateo entrenó una versión sin el fix de augmentación como comparación.

3. **Bug de augmentación:** Descubrimos que `torchvision.transforms` no transforma los keypoints al rotar/cropear la imagen. Diagnosticamos visualmente (`diagnostico_transform.py`), corregimos el pipeline y validamos la mejora (`comparar_augmentation.py`).

4. **Modelo B — exploración:** Exploramos una arquitectura basada en heatmaps. El modelo no convergió correctamente y fue descartado (`analizar_modelo_b.ipynb`).

5. **Sistema en tiempo real:** Integramos YOLO para detección multi-persona + modelo propio para pose. Implementamos ángulos, One Euro Filter, reglas biomecánicas y visualización.

6. **Intel RealSense:** Se intentó integrar la cámara de profundidad D435i pero se pausó por problemas de drivers en Apple Silicon. La carpeta `librealsense/` contiene el SDK.