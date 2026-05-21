import numpy as np
import os
from PIL import Image
from tensorflow import keras


# DATA_PROCESSING.PY
# Carga las imagenes propias, las adapta al formato 28x28 del dataset
# MNIST y las combina con el dataset original para aumentar los datos.


RUTA_CARPETA_PROPIA = 'mis_digitos'


def cargar_dataset_original():
    print("Cargando dataset original MNIST...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Normalizar pixeles al rango 0-1
    x_train = x_train.astype('float32') / 255
    x_test  = x_test.astype('float32')  / 255

    print(f"  -> Entrenamiento: {len(x_train)} imagenes")
    print(f"  -> Prueba       : {len(x_test)} imagenes")
    return (x_train, y_train), (x_test, y_test)


def cargar_imagenes_propias(ruta_carpeta=RUTA_CARPETA_PROPIA):
    print(f"\nCargando imagenes propias desde '{ruta_carpeta}'...")

    X_propias = []
    y_propias = []

    for digito in range(10):
        # Cada digito es un archivo: 0.png, 1.png ... 9.png
        ruta_imagen = os.path.join(ruta_carpeta, f"{digito}.png")

        if not os.path.exists(ruta_imagen):
            print(f"  ADVERTENCIA: no se encontro '{ruta_imagen}', se omitira.")
            continue

        # Convertir a escala de grises y redimensionar a 28x28
        img = Image.open(ruta_imagen).convert('L').resize((28, 28))
        img_array = np.array(img).astype('float32')

        # Descomentar si el fondo es blanco y el digito es negro
        img_array = 255 - img_array

        # Normalizar pixeles al rango 0-1 (igual que MNIST)
        img_array = img_array / 255.0

        X_propias.append(img_array)
        y_propias.append(digito)
        print(f"  -> Digito {digito}: cargado correctamente")

    X_propias = np.array(X_propias)
    y_propias = np.array(y_propias)

    print(f"\nTotal de imagenes propias cargadas: {len(X_propias)}")
    return X_propias, y_propias


def combinar_datasets(x_train, y_train, X_propias, y_propias):
    print("\nCombinando datasets...")

    # Agregar las imagenes propias al conjunto de entrenamiento
    x_train_aumentado = np.concatenate((x_train, X_propias), axis=0)
    y_train_aumentado = np.concatenate((y_train, y_propias), axis=0)

    print(f"  -> Dataset original : {len(x_train)} muestras")
    print(f"  -> Imagenes propias : {len(X_propias)} muestras")
    print(f"  -> Dataset combinado: {len(x_train_aumentado)} muestras en total")

    return x_train_aumentado, y_train_aumentado


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = cargar_dataset_original()
    X_prop, y_prop = cargar_imagenes_propias()
    x_final, y_final = combinar_datasets(x_train, y_train, X_prop, y_prop)

    print("\n--- Verificacion Final ---")
    print(f"Forma del dataset combinado (X): {x_final.shape}")
    print(f"Forma de las etiquetas     (y): {y_final.shape}")
    print("data_processing.py ejecutado correctamente.")