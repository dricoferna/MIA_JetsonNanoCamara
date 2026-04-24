import cv2

def main():
    # 1. Capturar imagen desde la cámara (0 es por defecto)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: No se puede acceder a la cámara.")
        return

    print("Presiona 'q' para salir.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Invertir imagen (opcional, efecto espejo para comodidad)
        frame = cv2.flip(frame, 1)
        
        # 2. Definir una Zona de Interés (ROI)
        # Formato: [y1:y2, x1:x2]. Ajustado al centro de una imagen estándar 640x480
        roi_x, roi_y, roi_w, roi_h = 200, 150, 200, 200
        roi = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]

        # 3. Detectar si hay o no un objeto
        # Convertimos a escala de grises y aplicamos un desenfoque para reducir ruido
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray_roi, (21, 21), 0)

        # Calculamos la desviación estándar de los píxeles
        # Un área vacía suele tener valores constantes; un objeto genera variación
        _, dev_std = cv2.meanStdDev(blur)

        # Umbral de detección (ajustar según la iluminación)
        umbral = 15
        objeto_detectado = dev_std[0][0] > umbral

        # 4. Respuesta del sistema
        # Dibujar el rectángulo de la zona de interés
        color = (0, 255, 0) if objeto_detectado else (0, 0, 255) # Verde si hay, Rojo si no
        cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), color, 2)

        # Mensaje en pantalla
        mensaje = "ESTADO: OBJETO DETECTADO" if objeto_detectado else "ESTADO: VACIO"
        cv2.putText(frame, mensaje, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Mostrar la ventana
        cv2.imshow('Fase 1: Sistema de Deteccion', frame)

        # Salir con la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()