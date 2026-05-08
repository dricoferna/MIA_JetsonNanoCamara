import cv2
import os
from datetime import datetime

# Marcas y orientaciones
marcas = ["titanium", "redbull", "monster", "cocacola", "sprite", "aquarius", "eneryeti_blanco", "eneryeti_morado"]
orientaciones = ["front", "back", "left", "right"]

# Crear carpetas
base_dir = "dataset"
os.makedirs(base_dir, exist_ok=True)

for marca in marcas:
    for orientacion in orientaciones:
        path = os.path.join(base_dir, f"{marca}-{orientacion}")
        os.makedirs(path, exist_ok=True)

# Estado actual
marca_idx = 0
orientacion_idx = 0
contador = 0

# Webcam
cap = cv2.VideoCapture(1)

print("Controles:")
print("1-8 = cambiar marca")
print("f/b/l/r = orientación")
print("t = foto | q = salir")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    marca_actual = marcas[marca_idx]
    orientacion_actual = orientaciones[orientacion_idx]

    save_path = os.path.join(base_dir, f"{marca_actual}-{orientacion_actual}")

    # Texto en pantalla
    texto = f"{marca_actual}-{orientacion_actual} | Fotos: {contador}"
    cv2.putText(frame, texto, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Captura dataset", frame)

    key = cv2.waitKey(1) & 0xFF

    # Cambiar marca
    if key in [ord(str(i)) for i in range(1, 9)]:
        marca_idx = key - ord('1')
        print(f"Marca cambiada a: {marcas[marca_idx]}")

    # Cambiar orientación
    elif key == ord('f'):
        orientacion_idx = 0
        print("Orientación: front")

    elif key == ord('b'):
        orientacion_idx = 1
        print("Orientación: back")

    elif key == ord('l'):
        orientacion_idx = 2
        print("Orientación: left")

    elif key == ord('r'):
        orientacion_idx = 3
        print("Orientación: right")

    # Tomar foto
    elif key == ord('t'):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{marca_actual}_{orientacion_actual}_{timestamp}.jpg"
        filepath = os.path.join(save_path, filename)

        cv2.imwrite(filepath, frame)
        contador += 1
        print(f"Foto guardada en {save_path}")

    # Salir
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()