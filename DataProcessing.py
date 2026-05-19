import numpy as np
import os
from PIL import Image
from sklearn.datasets import load_digits

# DATA_PROCESSING.PY
# Carga las imagenes propias, las adapta al formato 8x8 del dataset
# original y las combina para obtener un dataset aumentado.



def cargar_dataset_original():
    print("Cargando dataset original de sklearn...")
    digits = load_digits()
    print(f"  -> {len(digits.data)} imagenes cargadas, {len(set(digits.target))} clases (0-9)")
    return digits.data, digits.target


def cargar_imagenes_propias(ruta_carpeta="./mis_digitos"):
    print(f"\nCargando imagenes propias desde '{ruta_carpeta}'...")

    X_propias = []
    y_propias = []

    for digito in range(10):
        # Ruta de la subcarpeta del digito (ejemplo: ./mis_digitos/3/)
        ruta_digito = os.path.join(ruta_carpeta, str(digito))

        if not os.path.exists(ruta_digito):
            raise FileNotFoundError(
                f"No se encontro la carpeta: {ruta_digito}\n"
                f"Asegurate de tener subcarpetas del 0 al 9 dentro de '{ruta_carpeta}'"
            )

        # Solo archivos PNG dentro de la carpeta del digito
        archivos = [f for f in os.listdir(ruta_digito) if f.lower().endswith(".png")]

        if len(archivos) < 10:
            print(f"  ADVERTENCIA: digito {digito} tiene solo {len(archivos)} imagen(es). Se recomiendan al menos 10.")

        for archivo in archivos:
            ruta_imagen = os.path.join(ruta_digito, archivo)

            # Convertir a escala de grises y redimensionar a 8x8
            img = Image.open(ruta_imagen).convert('L').resize((8, 8))
            img_array = np.array(img)

            # Descomentar si el fondo es blanco y el digito es negro
            # img_array = 255 - img_array

            # Normalizar de rango 0-255 a rango 0-16 (formato sklearn)
            img_normalizada = (img_array / 255.0) * 16.0

            # Aplanar de 8x8 a vector de 64 valores
            X_propias.append(img_normalizada.flatten())
            y_propias.append(digito)

        print(f"  -> Digito {digito}: {len(archivos)} imagen(es) cargada(s)")

    X_propias = np.array(X_propias)
    y_propias = np.array(y_propias)

    print(f"\nTotal de imagenes propias: {len(X_propias)}")
    return X_propias, y_propias


def combinar_datasets(X_original, y_original, X_propias, y_propias):
    print("\nCombinando datasets...")

    # Apilar filas de ambos datasets
    X_aumentado = np.vstack((X_original, X_propias))
    y_aumentado = np.hstack((y_original, y_propias))

    print(f"  -> Dataset original : {len(X_original)} muestras")
    print(f"  -> Imagenes propias : {len(X_propias)} muestras")
    print(f"  -> Dataset combinado: {len(X_aumentado)} muestras en total")

    return X_aumentado, y_aumentado


if __name__ == "__main__":
    X_orig, y_orig = cargar_dataset_original()
    X_prop, y_prop = cargar_imagenes_propias("./mis_digitos")
    X_final, y_final = combinar_datasets(X_orig, y_orig, X_prop, y_prop)

    print("\n--- Verificacion Final ---")
    print(f"Forma del dataset combinado (X): {X_final.shape}")
    print(f"Forma de las etiquetas     (y): {y_final.shape}")
    print("data_processing.py ejecutado correctamente.")