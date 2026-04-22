import cv2
import numpy as np
import time

CAMARA_ID         = 0
UMBRAL_AREA       = 1500
SENSIBILIDAD      = 25
MOSTRAR_VENTANA   = True
INTERVALO_LOG     = 2.0


def abrir_camara():
    """Abre cámara USB probando varias configuraciones (compatible con Generalplus)."""

    for i in range(3):  # prueba /dev/video0,1,2
        print(f"[INFO] Probando cámara índice {i}...")

        cap = cv2.VideoCapture(i, cv2.CAP_V4L2)

        if cap.isOpened():
            # 🔥 FORZAR MJPEG (CLAVE)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

            # Opcional pero recomendable
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            # Test rápido de lectura
            ret, frame = cap.read()
            if ret:
                print(f"[INFO] Cámara USB funcionando en índice {i}")
                return cap
            else:
                print(f"[WARN] Cámara en índice {i} no devuelve frames")

        cap.release()

    raise RuntimeError("No se pudo abrir ninguna cámara funcional.")


def preprocesar(frame):
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gris = cv2.GaussianBlur(gris, (21, 21), 0)
    return gris


def detectar_objeto(frame_actual_gris, frame_anterior_gris):
    diferencia = cv2.absdiff(frame_anterior_gris, frame_actual_gris)
    _, umbral = cv2.threshold(diferencia, SENSIBILIDAD, 255, cv2.THRESH_BINARY)
    umbral = cv2.dilate(umbral, None, iterations=2)

    contornos, _ = cv2.findContours(
        umbral.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    contornos_validos = [c for c in contornos if cv2.contourArea(c) > UMBRAL_AREA]

    return len(contornos_validos) > 0, contornos_validos, umbral


def dibujar_detecciones(frame, contornos):
    for contorno in contornos:
        (x, y, w, h) = cv2.boundingRect(contorno)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return frame


def main():
    print("=" * 50)
    print("  DETECTOR DE PRESENCIA - USB FIX")
    print("=" * 50)

    cap = abrir_camara()

    ret, frame_anterior = cap.read()
    if not ret:
        raise RuntimeError("No se pudo leer el primer frame.")

    frame_anterior_gris = preprocesar(frame_anterior)

    estado_anterior = None
    ultimo_log = 0.0

    print("[INFO] Iniciando detección...\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Frame vacío, reintentando...")
                time.sleep(0.1)
                continue

            frame_gris = preprocesar(frame)
            hay_objeto, contornos, mascara = detectar_objeto(frame_gris, frame_anterior_gris)

            ahora = time.time()
            if hay_objeto != estado_anterior or (ahora - ultimo_log) > INTERVALO_LOG:
                estado_str = "⚠ OBJETO DETECTADO" if hay_objeto else "✓ Sin objeto"
                print(f"[{time.strftime('%H:%M:%S')}] {estado_str}")
                estado_anterior = hay_objeto
                ultimo_log = ahora

            if MOSTRAR_VENTANA:
                frame_vis = frame.copy()

                if hay_objeto:
                    frame_vis = dibujar_detecciones(frame_vis, contornos)
                    cv2.putText(frame_vis, "OBJETO DETECTADO", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    cv2.putText(frame_vis, "Sin objeto", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                cv2.imshow("Detector", frame_vis)
                cv2.imshow("Mascara", mascara)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_anterior_gris = frame_gris

    except KeyboardInterrupt:
        print("\n[INFO] Interrumpido")

    finally:
        cap.release()
        if MOSTRAR_VENTANA:
            cv2.destroyAllWindows()
        print("[INFO] Fin")


if __name__ == "__main__":
    main()