def abrir_camara():
    """
    Intenta abrir:
    1. Cámara CSI (Jetson)
    2. Cámara USB (probando índices y forzando MJPEG)
    """

    # ── Intento 1: CSI (por si acaso) ─────────────────────
    cap = cv2.VideoCapture(GSTREAMER_PIPELINE, cv2.CAP_GSTREAMER)
    if cap.isOpened():
        ret, _ = cap.read()
        if ret:
            print("[INFO] Cámara CSI abierta con GStreamer.")
            return cap
        cap.release()

    # ── Intento 2: USB (FIX REAL) ─────────────────────────
    for i in range(3):  # prueba /dev/video0,1,2
        print(f"[INFO] Probando cámara USB índice {i}...")

        cap = cv2.VideoCapture(i, cv2.CAP_V4L2)

        if cap.isOpened():
            # 🔥 CLAVE para cámaras problemáticas
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

            # Resolución estable
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            # Test real
            ret, frame = cap.read()
            if ret:
                print(f"[INFO] Cámara USB funcionando en índice {i}")
                return cap
            else:
                print(f"[WARN] Índice {i} abre pero no da frames")

        cap.release()

    raise RuntimeError("No se pudo abrir ninguna cámara funcional.")