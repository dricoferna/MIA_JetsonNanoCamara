"""
Detector de presencia para Jetson Nano
Detecta objetos EN MOVIMIENTO y objetos QUIETOS/ESTÁTICOS.
Compatible con Python 3.6.9

Estrategia dual:
  1. Diff frame-a-frame  → detecta movimiento (igual que antes, fiable)
  2. Diff contra frame de hace N segundos → detecta objetos que se quedaron quietos
  Si cualquiera de los dos dispara, hay objeto.
"""

import cv2
import numpy as np
import time
from collections import deque

# ─────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────
CAMARA_ID           = 0      # Índice de cámara (0 = primera cámara)
UMBRAL_AREA         = 1500   # Área mínima (px²) para considerar detección válida
SENSIBILIDAD        = 25     # Umbral de diferencia de píxeles
MOSTRAR_VENTANA     = True   # Mostrar ventana (requiere entorno gráfico)
INTERVALO_LOG       = 2.0    # Segundos entre mensajes de consola

# Detección de objeto estático:
# Se compara el frame actual con el frame de hace SEGUNDOS_REFERENCIA segundos.
# Si la escena cambió (entró un objeto y se quedó), esa diferencia persiste.
SEGUNDOS_REFERENCIA = 2.0    # Cuántos segundos atrás mirar para detectar estáticos
FPS_ESTIMADO        = 30     # FPS aproximados de tu cámara (para calcular tamaño del buffer)

# Pipeline GStreamer para cámara CSI del Jetson Nano
# Si usas cámara USB, el código hace fallback automático
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
    cap = cv2.VideoCapture(GSTREAMER_PIPELINE, cv2.CAP_GSTREAMER)
    if cap.isOpened():
        print("[INFO] Cámara CSI abierta con GStreamer.")
        return cap

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


def calcular_contornos(diff_gris):
    """Aplica umbral y devuelve contornos válidos a partir de una imagen de diferencia."""
    _, umbral = cv2.threshold(diff_gris, SENSIBILIDAD, 255, cv2.THRESH_BINARY)
    umbral = cv2.dilate(umbral, None, iterations=2)
    contornos, _ = cv2.findContours(
        umbral.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    validos = [c for c in contornos if cv2.contourArea(c) > UMBRAL_AREA]
    return validos, umbral


def detectar(frame_gris, frame_anterior_gris, frame_viejo_gris):
    """
    Detección dual:
      - diff_reciente: frame actual vs frame anterior  → movimiento
      - diff_viejo:    frame actual vs frame de hace N s → objeto estático
    Devuelve (hay_objeto, contornos, mascara_combinada)
    """
    # 1. Diferencia con frame inmediatamente anterior (movimiento)
    diff_reciente = cv2.absdiff(frame_anterior_gris, frame_gris)
    contornos_mov, mascara_mov = calcular_contornos(diff_reciente)

    # 2. Diferencia con frame antiguo (objeto estático que entró hace tiempo)
    diff_viejo = cv2.absdiff(frame_viejo_gris, frame_gris)
    contornos_est, mascara_est = calcular_contornos(diff_viejo)

    # Combinar máscaras para visualización
    mascara_combinada = cv2.bitwise_or(mascara_mov, mascara_est)

    # Hay objeto si cualquiera de los dos detecta algo
    hay_objeto = len(contornos_mov) > 0 or len(contornos_est) > 0

    # Para dibujar, usar los contornos del canal que disparó
    contornos = contornos_mov if contornos_mov else contornos_est

    return hay_objeto, contornos, mascara_combinada


def dibujar_detecciones(frame, contornos):
    """Dibuja rectángulos alrededor de los objetos detectados."""
    for contorno in contornos:
        (x, y, w, h) = cv2.boundingRect(contorno)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return frame


def main():
    print("=" * 50)
    print("  DETECTOR DE PRESENCIA - Jetson Nano")
    print("  (movimiento + objetos estáticos)")
    print("=" * 50)

    cap = abrir_camara()

    # Leer primer frame
    ret, frame_inicial = cap.read()
    if not ret:
        raise RuntimeError("No se pudo leer el primer frame de la cámara.")

    gris_inicial = preprocesar(frame_inicial)

    # Buffer circular: guarda los últimos N frames preprocesados.
    # El frame más antiguo del buffer actúa como referencia estática.
    tam_buffer = max(2, int(FPS_ESTIMADO * SEGUNDOS_REFERENCIA))
    buffer = deque([gris_inicial] * tam_buffer, maxlen=tam_buffer)

    estado_anterior = None
    ultimo_log      = 0.0

    print(f"[INFO] Buffer estático: {tam_buffer} frames (~{SEGUNDOS_REFERENCIA}s)")
    print("[INFO] Iniciando detección. Pulsa Ctrl+C (o 'q') para salir.\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] No se pudo leer frame. Reintentando...")
                time.sleep(0.1)
                continue

            frame_gris = preprocesar(frame)

            # frame anterior = último añadido al buffer
            # frame viejo    = el más antiguo del buffer (hace ~SEGUNDOS_REFERENCIA)
            frame_anterior_gris = buffer[-1]
            frame_viejo_gris    = buffer[0]

            hay_objeto, contornos, mascara = detectar(
                frame_gris, frame_anterior_gris, frame_viejo_gris
            )

            # Añadir frame actual al buffer
            buffer.append(frame_gris)

            # ── Log por consola (throttled) ───────────────────────────────────
            ahora = time.time()
            if hay_objeto != estado_anterior or (ahora - ultimo_log) > INTERVALO_LOG:
                estado_str = "⚠  OBJETO DETECTADO" if hay_objeto else "✓  Sin objeto"
                print(f"[{time.strftime('%H:%M:%S')}] {estado_str}")
                estado_anterior = hay_objeto
                ultimo_log      = ahora

            # ── Visualización opcional ────────────────────────────────────────
            if MOSTRAR_VENTANA:
                frame_vis = frame.copy()
                if hay_objeto:
                    frame_vis = dibujar_detecciones(frame_vis, contornos)
                    cv2.putText(frame_vis, "OBJETO DETECTADO", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    cv2.putText(frame_vis, "Sin objeto", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                cv2.imshow("Detector - frame",  frame_vis)
                cv2.imshow("Detector - mascara", mascara)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[INFO] Saliendo por tecla 'q'.")
                    break

    except KeyboardInterrupt:
        print("\n[INFO] Interrumpido por el usuario.")

    finally:
        cap.release()
        if MOSTRAR_VENTANA:
            cv2.destroyAllWindows()
        print("[INFO] Recursos liberados. Fin del programa.")


if __name__ == "__main__":
    main()
