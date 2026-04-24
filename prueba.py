import cv2
from ultralytics import YOLO

# Cargar modelo preentrenado
model = YOLO("yolov8n.pt")  # versión ligera

# Iniciar webcam
cap = cv2.VideoCapture(0)

# Definir zona de interés (ROI)
# (x1, y1) esquina superior izquierda, (x2, y2) inferior derecha
roi_x1, roi_y1 = 100, 100
roi_x2, roi_y2 = 400, 400

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Dibujar ROI
    cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 0), 2)

    # Detectar objetos en todo el frame
    results = model(frame)

    objeto_en_roi = False

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Centro del objeto
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Verificar si está dentro del ROI
            if roi_x1 < cx < roi_x2 and roi_y1 < cy < roi_y2:
                objeto_en_roi = True

                # Dibujar bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Mostrar resultado (respuesta del sistema)
    if objeto_en_roi:
        mensaje = "OBJETO DETECTADO EN LA ZONA"
        color = (0, 0, 255)
    else:
        mensaje = "SIN OBJETOS EN LA ZONA"
        color = (0, 255, 0)

    cv2.putText(frame, mensaje, (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Mostrar imagen
    cv2.imshow("Deteccion de Objetos", frame)

    # Salir con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
