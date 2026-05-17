[README.md](https://github.com/user-attachments/files/27805874/README.md)
# 🥤 Detector de Latas en Tiempo Real

Sistema de visión artificial que detecta la **marca** y **orientación** de latas de bebida usando una webcam y un modelo de deep learning entrenado con transfer learning sobre MobileNetV2.

---

## 👥 Integrantes

| Nombre |
|--------|
| *Danel Rico* |
| *Rubén Huarte* |

---

## 📌 Opción de Proyecto Elegida

**Clasificación de objetos físicos mediante visión por computador** — detección multi-etiqueta (marca + orientación) de latas de bebida en tiempo real usando una cámara web y un modelo de clasificación de imagen.

---

## 🎯 Objetivo del Sistema

Desarrollar un sistema capaz de identificar, en tiempo real y a través de una webcam:

- La **marca** de una lata de bebida (Aquarius, Coca-Cola, Monster, Red Bull, Sprite, Titanium).
- La **orientación** en que se presenta la lata (front, back, left, right).

El sistema muestra el resultado sobre el vídeo en directo con la confianza de cada predicción y suavizado temporal para mayor estabilidad.

---

## 🛠️ Materiales Utilizados

### Hardware
- Ordenador con webcam (cámara integrada o externa USB)
- Raspberry pi4

### Software y librerías
| Librería | Uso |
|----------|-----|
| TensorFlow / Keras | Entrenamiento e inferencia del modelo |
| MobileNetV2 | Backbone preentrenado (ImageNet) para transfer learning |
| OpenCV (`cv2`) | Captura de vídeo y renderizado del overlay |
| NumPy | Preprocesamiento de imágenes y manejo de arrays |
| Python 3.x | Lenguaje principal |

### Dataset
- Fotografías propias capturadas con `fotos.py` usando webcam.
- **6 marcas** × **4 orientaciones** = 24 clases combinadas.
- Marcas: `aquarius`, `cocacola`, `monster`, `redbull`, `sprite`, `titanium`.
- Orientaciones: `front`, `back`, `left`, `right`.

---

## ⚙️ Descripción General del Funcionamiento

El sistema se compone de tres módulos principales:

```
fotos.py  →  modelo_latas.h5  →  EJECUCION-CAMARA.py
(dataset)    (entrenamiento)       (inferencia RT)
```

1. **Captura del dataset** (`fotos.py`): interfaz de teclado para fotografiar latas con la webcam, organizando las imágenes en carpetas `dataset/<marca>-<orientacion>/`.

2. **Entrenamiento** (`modelo-latas-final.ipynb`): fine-tuning de MobileNetV2 con dos cabezas de clasificación independientes —una para marca y otra para orientación— exportado como `modelo_latas.h5`.

3. **Inferencia en tiempo real** (`EJECUCION-CAMARA.py`): captura frames de la webcam, recorta un ROI cuadrado central, realiza la predicción y pinta el resultado sobre el vídeo con colores por orientación, porcentajes de confianza y barra de confianza.

---

## 🟢 Fase 2 — Entrenamiento e Inferencia

### Entrenamiento (`modelo-latas-final.ipynb`)

- **Base**: MobileNetV2 preentrenado en ImageNet, capas superiores congeladas inicialmente.
- **Arquitectura multi-salida**: dos ramas `Dense` softmax, una para marca (6 clases) y otra para orientación (4 clases).
- **Fine-tuning**: segunda fase con las últimas capas del backbone descongeladas a learning rate bajo.
- **Resultados**: precisión de validación >99 % en ambas tareas (ver gráfica de entrenamiento).


### Inferencia en tiempo real (`EJECUCION-CAMARA.py`)

- Carga del modelo `.h5` y etiquetas desde `labels.json`.
- Recorte del ROI cuadrado central del frame para eliminar fondo irrelevante.
- Redimensionado a 224×224 y normalización antes de la predicción.
- **Suavizado temporal**: promedio móvil de las últimas 5 predicciones para evitar parpadeos.
- Overlay con: nombre de marca, orientación, porcentajes de confianza, FPS y barra de confianza.
- Umbral configurable (por defecto 0.6) para no mostrar predicciones inciertas.

---

## 🚀 Instrucciones de Ejecución

### Requisitos previos

```bash
pip install tensorflow opencv-python numpy
```

### 1. Capturar nuevas imágenes (opcional)

```bash
python fotos.py
```

Controles durante la captura:

| Tecla | Acción |
|-------|--------|
| `1`–`8` | Cambiar marca |
| `f` / `b` / `l` / `r` | Cambiar orientación |
| `t` | Tomar foto |
| `q` | Salir |

### 2. Entrenar el modelo (opcional)

Abrir y ejecutar `modelo-latas-final.ipynb` en Jupyter. Genera `modelo_latas.h5` y `labels.json`.

### 3. Ejecutar la detección en tiempo real

```bash
python EJECUCION-CAMARA-RASPBERRY.py
```
Se peuden observar dos archivos similares en el repositorio, el que funciona en el RaspBerry es el que he dicho el otro funciona en windows.

Parámetros configurables al inicio del fichero:

```python
CAMERA_ID  = 0     # 0 = cámara principal, 1 = secundaria, etc.
THRESHOLD  = 0.6   # confianza mínima para mostrar resultado
```

Presiona **Q** o **Esc** para cerrar la ventana.

---

## ⚠️ Problemas Encontrados

- **Muchos problemas al importar las librerias tanto al JetsonNano como a la RaspBerry
- **Necesidad de comprar latas para crear el dataset
- **Fallos constantes en la cámara al ejecutar el programa en la RaspBerry

---

## 🔮 Posibles Mejoras Futuras

- **Aumento del dataset**: utilizar una mayor cantidad de latas incluyendo bebidas alcoholicas.
- **Uso profesional**: en el caso de que se quisiera utilizar en una fábrica se podria hacer algo con los datos que extrae la cámara, por ejemplo, redirigir las latas para reciclar las del mismo tipo.

