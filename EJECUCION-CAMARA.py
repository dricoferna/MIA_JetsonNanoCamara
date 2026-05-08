"""
╔══════════════════════════════════════════════════════════════════╗
║        DETECCIÓN EN TIEMPO REAL - Webcam                        ║
║        Usa el modelo entrenado para detectar marca+orientación  ║
╚══════════════════════════════════════════════════════════════════╝

Uso:
    python webcam_detector.py
    python webcam_detector.py --camera 1   # si tienes múltiples cámaras
    python webcam_detector.py --threshold 0.7
"""

import cv2
import json
import numpy as np
import tensorflow as tf
import time

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
MODEL_PATH   = "modelo_latas.h5"
LABELS_FILE  = "labels.json"
IMG_SIZE     = (224, 224)
CONF_THRESHOLD = 0.6   # confianza mínima para mostrar resultado

# Colores por orientación (BGR)
ORIENT_COLORS = {
    "front": (0,   200,  50),   # verde
    "back":  (50,  50,  220),   # rojo
    "left":  (220, 150,   0),   # azul
    "right": (0,  180,  220),   # amarillo
}

# Icono de orientación
ORIENT_ICON = {
    "front": "⬛ FRONT",
    "back":  "⬛ BACK ",
    "left":  "◀  LEFT ",
    "right": "▶  RIGHT",
}

# ──────────────────────────────────────────────
# CARGAR MODELO Y ETIQUETAS
# ──────────────────────────────────────────────
def load_model_and_labels():
    print("⏳ Cargando modelo...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"✅ Modelo cargado: {MODEL_PATH}")

    with open(LABELS_FILE, "r") as f:
        labels = json.load(f)

    brands       = labels["brands"]        # {idx: nombre}
    orientations = labels["orientations"]  # {idx: nombre}

    print(f"   Marcas: {list(brands.values())}")
    print(f"   Orientaciones: {list(orientations.values())}\n")

    return model, brands, orientations


# ──────────────────────────────────────────────
# PREPROCESAR FRAME PARA EL MODELO
# ──────────────────────────────────────────────
def preprocess_frame(frame):
    """
    Recorta el ROI central del frame (cuadrado) y lo prepara para el modelo.
    """
    h, w = frame.shape[:2]
    side = min(h, w)
    y0   = (h - side) // 2
    x0   = (w - side) // 2
    roi  = frame[y0:y0+side, x0:x0+side]

    img = cv2.resize(roi, IMG_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)  # (1, 224, 224, 3)
    return img, (x0, y0, side)


# ──────────────────────────────────────────────
# OVERLAY EN EL FRAME
# ──────────────────────────────────────────────
def draw_overlay(frame, brand_name, brand_conf, orient_name, orient_conf, roi_box, fps):
    x0, y0, side = roi_box
    h, w = frame.shape[:2]

    # ── ROI box ─────────────────────────────
    color = ORIENT_COLORS.get(orient_name, (200, 200, 200))
    cv2.rectangle(frame, (x0, y0), (x0+side, y0+side), color, 2)

    # ── Panel de info (parte inferior) ───────
    panel_h = 90
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - panel_h), (w, h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    font       = cv2.FONT_HERSHEY_DUPLEX
    font_small = cv2.FONT_HERSHEY_SIMPLEX

    # Marca
    brand_text = f"MARCA:  {brand_name.upper()}"
    conf_b_txt = f"{brand_conf*100:.1f}%"
    cv2.putText(frame, brand_text, (15, h - panel_h + 28), font, 0.75, (255,255,255), 1)
    cv2.putText(frame, conf_b_txt, (w - 75, h - panel_h + 28), font, 0.65, (120,255,120), 1)

    # Orientación
    orient_text = f"ORIENT: {orient_name.upper()}"
    conf_o_txt  = f"{orient_conf*100:.1f}%"
    cv2.putText(frame, orient_text, (15, h - panel_h + 60), font, 0.75, color, 1)
    cv2.putText(frame, conf_o_txt, (w - 75, h - panel_h + 60), font, 0.65, (120,255,120), 1)

    # FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 100, 25), font_small, 0.55, (180,180,180), 1)

    # Barra de confianza marca
    bar_w = int((w - 30) * brand_conf)
    cv2.rectangle(frame, (15, h - 8), (w - 15, h - 3), (50,50,50), -1)
    cv2.rectangle(frame, (15, h - 8), (15 + bar_w, h - 3), (120,255,120), -1)

    return frame


def draw_low_conf(frame, fps):
    """Mostrar cuando la confianza es baja."""
    h, w = frame.shape[:2]
    cv2.putText(frame, "Acerca una lata...", (w//2 - 130, h//2),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, (100,100,255), 1)
    cv2.putText(frame, f"FPS: {fps:.1f}", (w-100, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180,180,180), 1)


# ──────────────────────────────────────────────
# MAIN LOOP
# ──────────────────────────────────────────────
def main(camera_id: int, threshold: float):
    model, brands, orientations = load_model_and_labels()

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"❌ No se pudo abrir la cámara {camera_id}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print("🎥 Webcam iniciada. Presiona Q para salir.\n")

    prev_time = time.time()
    fps       = 0.0

    # Suavizado (promedio móvil de las últimas N predicciones)
    SMOOTH_N   = 5
    brand_buf  = []
    orient_buf = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️  No se pudo leer frame.")
            break

        # FPS
        now       = time.time()
        fps       = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
        prev_time = now

        # Preprocesar
        img_input, roi_box = preprocess_frame(frame)

        # Predicción
        brand_probs, orient_probs = model.predict(img_input, verbose=0)
        brand_probs  = brand_probs[0]   # (N_brands,)
        orient_probs = orient_probs[0]  # (N_orients,)

        brand_idx  = int(np.argmax(brand_probs))
        orient_idx = int(np.argmax(orient_probs))
        brand_conf  = float(brand_probs[brand_idx])
        orient_conf = float(orient_probs[orient_idx])

        # Suavizado temporal
        brand_buf.append(brand_idx)
        orient_buf.append(orient_idx)
        if len(brand_buf)  > SMOOTH_N: brand_buf.pop(0)
        if len(orient_buf) > SMOOTH_N: orient_buf.pop(0)

        smooth_brand  = max(set(brand_buf),  key=brand_buf.count)
        smooth_orient = max(set(orient_buf), key=orient_buf.count)

        brand_name  = brands[str(smooth_brand)]
        orient_name = orientations[str(smooth_orient)]

        # Dibujar
        if brand_conf >= threshold:
            frame = draw_overlay(
                frame,
                brand_name, brand_conf,
                orient_name, orient_conf,
                roi_box, fps
            )
        else:
            # Dibujar ROI de todas formas
            x0, y0, side = roi_box
            cv2.rectangle(frame, (x0, y0), (x0+side, y0+side), (80,80,80), 1)
            draw_low_conf(frame, fps)

        cv2.imshow("🥤 Detector de Latas | Q para salir", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("👋 Detector cerrado.")


# ──────────────────────────────────────────────
# CONFIGURACIÓN MANUAL (cambia estos valores si necesitas)
# Compatible con Jupyter, VSCode, terminal — en cualquier entorno
# ──────────────────────────────────────────────
CAMERA_ID  =0      # 0 = cámara principal, 1 = secundaria, etc.
THRESHOLD  = CONF_THRESHOLD  # confianza mínima (0.0 - 1.0)

if __name__ == "__main__":
    main(CAMERA_ID, THRESHOLD)