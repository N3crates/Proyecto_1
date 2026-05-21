import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from PIL import Image

NOMBRE_MODELO = 'modelo_red_convolucional.h5'
RUTA_CARPETA_INYECCION = 'mis_digitos'

def cargar_dataset_inyectado():
    """
    Escanea y aplica binarización estricta para dejar los trazos blancos sobre fondo negro.
    """
    X_inyectado = []
    y_inyectado = []
    
    if os.path.exists(RUTA_CARPETA_INYECCION):
        print(f"\n[INFO] Escaneando y purificando imágenes en: '{RUTA_CARPETA_INYECCION}'...")
        archivos = [f for f in os.listdir(RUTA_CARPETA_INYECCION) if os.path.isfile(os.path.join(RUTA_CARPETA_INYECCION, f))]
        
        for nombre_archivo in archivos:
            ruta_completa = os.path.join(RUTA_CARPETA_INYECCION, nombre_archivo)
            primer_caracter = nombre_archivo[0]
            
            if primer_caracter.isdigit():
                etiqueta = int(primer_caracter)
                if 0 <= etiqueta <= 9:
                    try:
                        img = Image.open(ruta_completa).convert('L')
                        img = img.resize((28, 28))
                        
                        # Limpieza extrema del fondo
                        img_array = np.array(img)
                        img_array = np.where(img_array > 110, 255, 0) 
                        img_array = img_array / 255.0
                        
                        X_inyectado.append(img_array)
                        y_inyectado.append(etiqueta)
                    except Exception as e:
                        print(f"[ERROR] No se pudo procesar '{nombre_archivo}': {e}")
                        
    return np.array(X_inyectado), np.array(y_inyectado)


# =====================================================================
# 1. CARGA DEL DATASET ORIGINAL
# =====================================================================
print("Cargando el dataset original (MNIST)...")
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.reshape((x_train.shape[0], 28, 28, 1)).astype('float32') / 255
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1)).astype('float32') / 255


# =====================================================================
# 2. FUSIÓN CON SOBREMUESTREO MASIVO Y BARAJADO 
# =====================================================================
X_inyectado_original, y_inyectado_original = cargar_dataset_inyectado()

if len(X_inyectado_original) > 0:
    print(f"[OK] Se detectaron {len(X_inyectado_original)} muestras inyectadas originales.")
    X_inyectado_original = X_inyectado_original.reshape((X_inyectado_original.shape[0], 28, 28, 1))
    
    # Clonamos las imágenes 100 veces para que la red no las ignore
    MULTIPLICADOR = 100
    X_masivo = np.repeat(X_inyectado_original, MULTIPLICADOR, axis=0)
    y_masivo = np.repeat(y_inyectado_original, MULTIPLICADOR, axis=0)
    
    print(f"[HACK ACTIVO] Clonando tus imágenes x{MULTIPLICADOR}. Inyectando {len(X_masivo)} trazos en total.")
    
    # Fusión masiva al final de la lista
    x_train = np.concatenate((x_train, X_masivo), axis=0)
    y_train = np.concatenate((y_train, y_masivo), axis=0)
    
    # --- EL ARREGLO CRÍTICO: BARAJAR LOS DATOS ---
    # Revolvemos los índices para esparcir tus fotos por todo el dataset
    print("[PROCESO] Barajando el dataset combinado para evitar el sesgo de validación...")
    indices = np.arange(x_train.shape[0])
    np.random.shuffle(indices)
    
    x_train = x_train[indices]
    y_train = y_train[indices]
    
    print("[OK] Fusión masiva y barajado completados.")
else:
    print(f"\n[INFO] Carpeta '{RUTA_CARPETA_INYECCION}' vacía o no detectada.")
# =====================================================================
# 3. ENTRENAMIENTO DE LA CNN 
# =====================================================================
if os.path.exists(NOMBRE_MODELO) and len(X_inyectado_original) == 0:
    print(f"\n[CACHE] Cargando modelo entrenado desde disco ('{NOMBRE_MODELO}')...")
    model = keras.models.load_model(NOMBRE_MODELO)
else:
    print("\n[PROCESO] Entrenando Red Convolucional con datos priorizados...")

    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2), 
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.fit(x_train, y_train, epochs=15, validation_split=0.1, batch_size=128)
    
    model.save(NOMBRE_MODELO)
    print(f"\n[OK] Modelo blindado guardado como '{NOMBRE_MODELO}'.")


# =====================================================================
# 4. EVALUACIÓN Y MONITOREO ESTRICTO
# =====================================================================
print("\n--- INFORME FINAL DE RENDIMIENTO ---")

if len(X_inyectado_original) > 0:
    print(f"Evaluando estrictamente sobre las {len(X_inyectado_original)} muestras originales tuyas:")
    # Usamos el arreglo original (50 fotos) para la evaluación final, NO el masivo
    predicciones = model.predict(X_inyectado_original, verbose=0)
    X_evaluar, y_evaluar = X_inyectado_original, y_inyectado_original
else:
    print("Evaluando sobre una tanda del dataset original:")
    predicciones = model.predict(x_test[:10], verbose=0)
    X_evaluar, y_evaluar = x_test[:10], y_test[:10]

aciertos = 0
for idx in range(len(X_evaluar)):
    clase_predicha = np.argmax(predicciones[idx])
    clase_real = y_evaluar[idx]
    es_correcto = (clase_predicha == clase_real)
    
    if es_correcto:
        aciertos += 1
        
    status = "CORRECTO" if es_correcto else "FALLIDO"
    print(f"Muestra [{idx:02d}]. Valor Real: {clase_real} | Predicción CNN: {clase_predicha} -> [{status}]")
    
print(f"\nPrecisión final observada: {aciertos}/{len(X_evaluar)} ({(aciertos/len(X_evaluar))*100:.2f}%)")