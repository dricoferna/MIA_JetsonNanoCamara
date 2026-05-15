"""
Detector de latas USB webcam
Raspberry Pi 4 + Debian 13
Solución: cámara y TensorFlow en hilos separados
"""

import cv2
import json
import time
import threading
import numpy as np
import tensorflow as tf

# -----------------------------------
# Evitar sobrecarga de TensorFlow
# -----------------------------------
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

MODEL_PATH = "modelo_latas.h5"
LABELS_FILE = "labels.json"
IMG_SIZE = 224
CONF_THRESHOLD = 0.60

# Variables compartidas
latest_frame = None
prediction = None
running = True
lock = threading.Lock()

# -----------------------------------
# Cargar modelo
# -----------------------------------
print("Cargando modelo...")
model = tf.keras.models.load_model(MODEL_PATH)

with open(LABELS_FILE, "r") as f:
    labels = json.load(f)

brands = labels["brands"]
orientations = labels["orientations"]

print("Modelo cargado OK")

def save_debug_input(img_rgb):
    """
    Guarda la imagen exacta (antes del preprocess_input)
    para comparar con Windows.
    """
    cv2.imwrite("debug_model_input.jpg",
                cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

# -----------------------------------
# HILO DE CÁMARA
# (idéntico al test que sí funciona)
# -----------------------------------
def camera_loop():
    global latest_frame, running

    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(*"MJPG")
    )

    if not cap.isOpened():
        print("No se pudo abrir cámara")
        running = False
        return

    print("Cámara USB abierta OK")

    while running:
        ret, frame = cap.read()

        if not ret:
            continue

        with lock:
            latest_frame = frame.copy()

    cap.release()


# -----------------------------------
# HILO DE IA
# -----------------------------------
def inference_loop():
    global prediction, running

    while running:
        frame = None

        with lock:
            if latest_frame is not None:
                frame = latest_frame.copy()

        if frame is None:
            time.sleep(0.01)
            continue

        try:
            h, w = frame.shape[:2]

            # ROI central
            side = min(h, w)
            x0 = (w - side) // 2
            y0 = (h - side) // 2

            roi = frame[y0:y0+side, x0:x0+side]

            # Resize
            img = cv2.resize(
                roi,
                (IMG_SIZE, IMG_SIZE)
            )

            # BGR -> RGB
            img_rgb = cv2.cvtColor(
                img,
                cv2.COLOR_BGR2RGB
            )

            # Guardar una vez para inspeccionar
            save_debug_input(img_rgb)

            # ===== PRUEBA 1: sin normalizar =====
            x = img_rgb.astype(np.float32)
            x = np.expand_dims(x, axis=0)

            # Si esto no va, prueba debajo la otra variante
            # ===== PRUEBA 2: /255 =====
            # x = img_rgb.astype(np.float32) / 255.0
            # x = np.expand_dims(x, axis=0)

            # ===== PRUEBA 3: [-1,1] =====
            # x = img_rgb.astype(np.float32)
            # x = (x / 127.5) - 1.0
            # x = np.expand_dims(x, axis=0)

            outputs = model.predict(
                x,
                verbose=0
            )

            # Detectar estructura de salida
            print("Salidas:", len(outputs))

            brand_probs = outputs[0][0]
            orient_probs = outputs[1][0]

            print("Brand probs:", brand_probs)
            print("Orient probs:", orient_probs)

            b = int(np.argmax(brand_probs))
            o = int(np.argmax(orient_probs))

            bc = float(brand_probs[b])

            prediction = (
                brands[str(b)],
                orientations[str(o)],
                bc
            )

        except Exception as e:
            print("Error IA:", e)
# -----------------------------------
# MAIN
# -----------------------------------
cam_thread = threading.Thread(
    target=camera_loop,
    daemon=True
)

ai_thread = threading.Thread(
    target=inference_loop,
    daemon=True
)

cam_thread.start()
ai_thread.start()

print("Pulsa Q para salir")

prev = time.time()
fps = 0

while running:
    frame = None

    with lock:
        if latest_frame is not None:
            frame = latest_frame.copy()

    if frame is None:
        continue

    # FPS
    now = time.time()
    fps = 0.9 * fps + 0.1 * (1/(now-prev))
    prev = now

    h, w = frame.shape[:2]

    # ROI visual
    side = min(h, w)
    x0 = (w - side) // 2
    y0 = (h - side) // 2

    cv2.rectangle(
        frame,
        (x0, y0),
        (x0+side, y0+side),
        (0,255,0),
        2
    )

    # Mostrar predicción
    if prediction:
        brand, orient, conf = prediction

        cv2.putText(
            frame,
            brand,
            (20,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
        )

        cv2.putText(
            frame,
            orient,
            (20,80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,255),
            2
        )

        cv2.putText(
            frame,
            f"{conf*100:.1f}%",
            (20,120),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255,255,255),
            2
        )

    cv2.putText(
        frame,
        f"FPS {fps:.1f}",
        (20,160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255,255,255),
        2
    )

    cv2.imshow(
        "Detector USB",
        frame
    )

    if cv2.waitKey(1) == ord("q"):
        running = False
        break

cv2.destroyAllWindows()
print("Cerrado.")