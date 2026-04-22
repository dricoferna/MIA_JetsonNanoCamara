"""
Detector de presencia para Jetson Nano
Detecta si hay algo (movimiento u objeto) frente a la cámara.
Compatible con Python 3.6.9
"""

import cv2
import numpy as np
import time

# ─────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────
CAMARA_ID         = 0       # Índice de cámara (0 = primera cámara)
UMBRAL_AREA       = 1500    # Área mínima (px²) para considerar detección válida
SENSIBILIDAD      = 25      # Umbral de diferencia de píxeles (más bajo = más sensible)
MOSTRAR_VENTANA   = True    # Mostrar ventana con la imagen (requiere entorno gráfico)
INTERVALO_LOG     = 2.0     # Segundos entre mensajes de consola

# Pipeline GStreamer para cámara CSI (cámara de módulo del Jetson)
# Si usas cámara USB normal, comenta esta línea y usa: cap = cv2.VideoCapture(CAMARA_ID)
GSTREAMER_PIPELINE = (
    "nvarguscamerasrc ! "
    "video/x-raw(memory:NVMM), width=640, height=480, framerate=30/1 ! "
    "nvvidconv ! "
    "video/x-raw, format=BGRx ! "
    "videoconvert ! "
    "video/x-raw, format=BGR ! "
    "appsink"
)


def abrir_camara():
    """Intenta abrir cámara CSI primero, luego USB como fallback."""
    # Intento 1: cámara CSI (módulo de cámara Jetson)
    cap = cv2.VideoCapture(GSTREAMER_PIPELINE, cv2.CAP_GSTREAMER)
    if cap.isOpened():
        print("[INFO] Cámara CSI abierta con GStreamer.")
        return cap

    # Intento 2: cámara USB estándar
    cap = cv2.VideoCapture(CAMARA_ID)
    if cap.isOpened():
        print("[INFO] Cámara USB abierta.")
        return cap

    raise RuntimeError("No se pudo abrir ninguna cámara. Verifica la conexión.")


def preprocesar(frame):
    """Convierte el frame a escala de grises y aplica blur para reducir ruido."""
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gris = cv2.GaussianBlur(gris, (21, 21), 0)
    return gris


def detectar_objeto(frame_actual_gris, frame_anterior_gris):
    """
    Compara dos frames consecutivos para detectar cambios (movimiento/objeto).
    Devuelve (hay_deteccion: bool, contornos, frame_umbral)
    """
    diferencia = cv2.absdiff(frame_anterior_gris, frame_actual_gris)
    _, umbral = cv2.threshold(diferencia, SENSIBILIDAD, 255, cv2.THRESH_BINARY)
    umbral = cv2.dilate(umbral, None, iterations=2)

    contornos, _ = cv2.findContours(
        umbral.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Filtra contornos pequeños (ruido)
    contornos_validos = [c for c in contornos if cv2.contourArea(c) > UMBRAL_AREA]

    hay_deteccion = len(contornos_validos) > 0
    return hay_deteccion, contornos_validos, umbral


def dibujar_detecciones(frame, contornos):
    """Dibuja rectángulos alrededor de los objetos detectados."""
    for contorno in contornos:
        (x, y, w, h) = cv2.boundingRect(contorno)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return frame


def main():
    print("=" * 50)
    print("  DETECTOR DE PRESENCIA - Jetson Nano")
    print("=" * 50)

    cap = abrir_camara()

    # Leer primer frame como referencia
    ret, frame_anterior = cap.read()
    if not ret:
        raise RuntimeError("No se pudo leer el primer frame de la cámara.")

    frame_anterior_gris = preprocesar(frame_anterior)

    estado_anterior   = None   # Para no repetir el mismo mensaje
    ultimo_log        = 0.0

    print("[INFO] Iniciando detección. Pulsa Ctrl+C para salir.\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] No se pudo leer frame. Reintentando...")
                time.sleep(0.1)
                continue

            frame_gris = preprocesar(frame)
            hay_objeto, contornos, mascara = detectar_objeto(frame_gris, frame_anterior_gris)

            # ── Log por consola (con throttle para no saturar) ──────────────
            ahora = time.time()
            if hay_objeto != estado_anterior or (ahora - ultimo_log) > INTERVALO_LOG:
                estado_str = "⚠  OBJETO DETECTADO" if hay_objeto else "✓  Sin objeto"
                print(f"[{time.strftime('%H:%M:%S')}] {estado_str}")
                estado_anterior = hay_objeto
                ultimo_log      = ahora

            # ── Visualización opcional ───────────────────────────────────────
            if MOSTRAR_VENTANA:
                frame_vis = frame.copy()
                if hay_objeto:
                    frame_vis = dibujar_detecciones(frame_vis, contornos)
                    cv2.putText(frame_vis, "OBJETO DETECTADO", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    cv2.putText(frame_vis, "Sin objeto", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                cv2.imshow("Detector - frame", frame_vis)
                cv2.imshow("Detector - mascara", mascara)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[INFO] Saliendo por tecla 'q'.")
                    break

            # Actualizar frame de referencia
            frame_anterior_gris = frame_gris

    except KeyboardInterrupt:
        print("\n[INFO] Interrumpido por el usuario.")

    finally:
        cap.release()
        if MOSTRAR_VENTANA:
            cv2.destroyAllWindows()
        print("[INFO] Recursos liberados. Fin del programa.")


if __name__ == "__main__":
    main()
