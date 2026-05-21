import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image

# Nombre del archivo donde se congelará el cerebro del modelo
NOMBRE_MODELO = 'modelo_combinado_mnist.h5'
RUTA_CARPETA_PROPIA = 'mis_digitos'

def cargar_muestras_propias_reales():
    X_propias = []
    y_propias = []
    
    if os.path.exists(RUTA_CARPETA_PROPIA):
        print(f"\n[DETECTADO] Cargando imagenes desde '{RUTA_CARPETA_PROPIA}'...")
        for i in range(10):
            archivos = sorted([
                f for f in os.listdir(RUTA_CARPETA_PROPIA)
                if f.startswith(f"{i}_") and f.endswith(".png")
            ])
            for archivo in archivos:
                ruta = os.path.join(RUTA_CARPETA_PROPIA, archivo)
                img = Image.open(ruta).convert('L').resize((28, 28))
                img_array = np.array(img) / 255.0
                X_propias.append(img_array)
                y_propias.append(i)

    return np.array(X_propias), np.array(y_propias)

# =====================================================================
# 1. CARGA DE DATOS ORIGINALES
# =====================================================================
print("Cargando el dataset original MNIST...")
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalización estándar del profesor
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255


# =====================================================================
# 2. FUSIÓN DE DATASETS (MNIST + IMÁGENES PROPIAS)
# =====================================================================
X_mis_datos, y_mis_datos = cargar_muestras_propias_reales()

if len(X_mis_datos) > 0:
    print(f"Inyectando {len(X_mis_datos)} imágenes propias al dataset de entrenamiento...")
    # Concatenar tus imágenes con las originales 
    x_train = np.concatenate((x_train, X_mis_datos), axis=0)
    y_train = np.concatenate((y_train, y_mis_datos), axis=0)
    print("¡Fusión completada con éxito!")
else:
    print(f"\n[INFO] No se detectó la carpeta '{RUTA_CARPETA_PROPIA}' o está vacía.")
    print("El modelo funcionará únicamente con el dataset original por ahora.")


# =====================================================================
# 3. CONTROL DE ENTRENAMIENTO 
# =====================================================================
if os.path.exists(NOMBRE_MODELO) and len(X_mis_datos) == 0:
    # Si ya hay un modelo guardado y NO se agregaron imágenes nuevas, cárgalo instantáneo
    print(f"\nCargando modelo entrenado desde disco ('{NOMBRE_MODELO}')...")
    model = keras.models.load_model(NOMBRE_MODELO)
else:
    # Si no hay modelo, o si inyectaste imágenes nuevas, se ejecuta el entrenamiento
    if len(X_mis_datos) > 0:
        print("\nReentrenando el modelo para incluir tus nuevas imágenes en su memoria...")
    else:
        print("\nIniciando entrenamiento desde cero con los parámetros del profesor...")

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Entrenamiento (20 épocas, lotes de 1024)
    model.fit(x_train, y_train, epochs=50, validation_split=0.1, batch_size=128)
    
    # Guardar los cambios en el disco
    model.save(NOMBRE_MODELO)
    print(f"Modelo guardado como '{NOMBRE_MODELO}'.")


# =====================================================================
# 4. EVALUACIÓN CON DATOS REALES (AQUÍ VERÁS LA VERDADERA EFECTIVIDAD)
# =====================================================================
print("\n--- EVALUACIÓN CON IMÁGENES REALES ---")

if len(X_mis_datos) > 0:
    # Si hay fotos, evaluamos las 10 imágenes para ver si las reconoce
    print("Evaluando el rendimiento del modelo sobre tus propias imágenes:")
    predicciones = model.predict(X_mis_datos, verbose=0)
    X_evaluar, y_evaluar = X_mis_datos, y_mis_datos
else:
    # Si no hay fotos propias, evaluamos con las primeras 10 imágenes reales del dataset
    print("Evaluando el rendimiento del modelo sobre 10 imágenes reales del dataset de prueba:")
    predicciones = model.predict(x_test[:10], verbose=0)
    X_evaluar, y_evaluar = x_test[:10], y_test[:10]

# Despliegue de aciertos individuales
aciertos = 0
for idx in range(len(X_evaluar)):
    clase_predicha = np.argmax(predicciones[idx])
    clase_real = y_evaluar[idx]
    es_correcto = (clase_predicha == clase_real)
    
    if es_correcto:
        aciertos += 1
        
    status = "CORRECTO" if es_correcto else "FALLIDO"
    print(f"Muestra [{idx}]. Dígito Real: {clase_real} | Predicción de la Red: {clase_predicha} -> [{status}]")
    
print(f"\nEfectividad real observada: {aciertos}/{len(X_evaluar)} ({(aciertos/len(X_evaluar))*100:.2f}%)")