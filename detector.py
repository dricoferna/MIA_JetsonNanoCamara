import cv2

def main():
    # 1. Capturar imagen desde la cámara
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: No se puede acceder a la cámara.")
        return

    # Definimos el tamaño del cuadrado de reconocimiento aquí
    # Puedes cambiar estos valores para ajustar el tamaño exacto
    ancho_cuadrado = 300 
    alto_cuadrado = 300

    print("Presiona 'q' para salir.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1) # Efecto espejo
        
        # Obtener dimensiones del frame actual para centrar el cuadrado
        height, width, _ = frame.shape

        # Calcular coordenadas para que el cuadrado esté centrado
        roi_x = (width - ancho_cuadrado) // 2
        roi_y = (height - alto_cuadrado) // 2
        
        # 2. Definir la Zona de Interés (ROI) con el nuevo tamaño
        roi = frame[roi_y:roi_y+alto_cuadrado, roi_x:roi_x+ancho_cuadrado]

        # 3. Detección básica (Escala de grises + Desviación estándar)
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray_roi, (21, 21), 0)
        _, dev_std = cv2.meanStdDev(blur)

        # Umbral de detección (ajusta este número si es muy sensible o poco sensible)
        umbral = 15
        objeto_detectado = dev_std[0][0] > umbral

        # 4. Respuesta del sistema
        color = (0, 255, 0) if objeto_detectado else (0, 0, 255)
        
        # Dibujar el rectángulo más grande
        cv2.rectangle(frame, (roi_x, roi_y), (roi_x + ancho_cuadrado, roi_y + alto_cuadrado), color, 3)

        # Mensaje en pantalla
        mensaje = "ESTADO: OBJETO DETECTADO" if objeto_detectado else "ESTADO: VACIO"
        cv2.putText(frame, mensaje, (roi_x, roi_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Mostrar la ventana
        cv2.imshow('Fase 1: Cuadrado Grande', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()