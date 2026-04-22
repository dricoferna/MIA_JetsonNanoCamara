import cv2
import numpy as np
import time

# ─────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────
UMBRAL_AREA       = 1500
SENSIBILIDAD      = 25
MOSTRAR_VENTANA   = True
INTERVALO_LOG     = 2.0

FRAMES_CALIBRACION = 60
ALPHA_FONDO        = 0.002


# ─────────────────────────────────────────
# APERTURA DE CÁMARA (FIX REAL)
# ─────────────────────────────────────────
def abrir_camara():
    for i in range(3):
        print(f"[INFO] Probando cámara índice {i}...")

        cap = cv2.VideoCapture(i, cv2.CAP_V4L2)

        if cap.isOpened():
            # Forzar MJPG (CLAVE)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"[OK] Cámara funcionando en índice {i}")
                return cap

        cap.release()

    raise RuntimeError("No se pudo abrir ninguna cámara funcional")


# ─────────────────────────────────────────
# PROCESAMIENTO
# ─────────────────────────────────────────
def preprocesar(frame):
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gris = cv2.GaussianBlur(gris, (21, 21), 0)
    return gris


def detectar_objeto(frame_gris, fondo_float):
    fondo_uint8 = cv2.convertScaleAbs(fondo_float)
    diferencia  = cv2.absdiff(fondo_uint8, frame_gris)

    _, umbral = cv2.threshold(diferencia, SENSIBILIDAD, 255, cv2.THRESH_BINARY)
    umbral = cv2.dilate(umbral, None, iterations=2)

    contornos, _ = cv2.findContours(
        umbral.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    contornos_validos = [c for c in contornos if cv2.contourArea(c) > UMBRAL_AREA]

    return len(contornos_validos) > 0, contornos_validos, umbral


def actualizar_fondo(fondo_float, frame_gris, hay_objeto):
    if not hay_objeto:
        cv2.accumulateWeighted(frame_gris, fondo_float, ALPHA_FONDO)
    return fondo_float


def dibujar_detecciones(frame, contornos):
    for contorno in contornos:
        (x, y, w, h) = cv2.boundingRect(contorno)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return frame


# ─────────────────────────────────────────
# CALIBRACIÓN
# ─────────────────────────────────────────
def calibrar_fondo(cap):
    print(f"[CALIBRACIÓN] {FRAMES_CALIBRACION} frames...")

    fondo = None

    for i in range(FRAMES_CALIBRACION):
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        gris = preprocesar(frame)

        if fondo is None:
            fondo = gris.astype(np.float32)
        else:
            cv2.accumulateWeighted(gris, fondo, 0.1)

    print("[OK] Fondo calibrado\n")
    return fondo


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
def main():
    print("=== DETECTOR DE PRESENCIA (FIX USB) ===")

    cap = abrir_camara()

    fondo_float = calibrar_fondo(cap)

    estado_anterior = None
    ultimo_log = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        frame_gris = preprocesar(frame)

        hay_objeto, contornos, mascara = detectar_objeto(frame_gris, fondo_float)

        fondo_float = actualizar_fondo(fondo_float, frame_gris, hay_objeto)

        ahora = time.time()
        if hay_objeto != estado_anterior or (ahora - ultimo_log) > INTERVALO_LOG:
            print("OBJETO" if hay_objeto else "SIN OBJETO")
            estado_anterior = hay_objeto
            ultimo_log = ahora

        if MOSTRAR_VENTANA:
            frame_vis = frame.copy()

            if hay_objeto:
                frame_vis = dibujar_detecciones(frame_vis, contornos)

            cv2.imshow("Frame", frame_vis)
            cv2.imshow("Mascara", mascara)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()