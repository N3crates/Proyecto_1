import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split

from data_processing import cargar_dataset_original, cargar_imagenes_propias, combinar_datasets

# EVALUATION.PY
# Carga el modelo entrenado y genera metricas
# de desempeno y graficas.
# Requiere: modelo_combinado_mnist.h5 en la misma carpeta.


NOMBRE_MODELO = 'modelo_red_convolucional.h5'


def cargar_modelo():
    print(f"Cargando modelo entrenado ({NOMBRE_MODELO})...")
    modelo = keras.models.load_model(NOMBRE_MODELO)
    print("  -> Modelo cargado correctamente.")
    return modelo


def preparar_datos():
    # Cargamos MNIST y las imagenes propias y las combinamos
    (x_train, y_train), (x_test, y_test) = cargar_dataset_original()
    X_prop, y_prop = cargar_imagenes_propias()
    x_train_aug, y_train_aug = combinar_datasets(x_train, y_train, X_prop, y_prop)
    return x_train_aug, y_train_aug, x_test, y_test


def validacion_cruzada_manual(modelo, X, y, k=5):
    # Dividimos manualmente el dataset en k partes para validacion cruzada
    # ya que keras no es compatible directamente con cross_val_score de sklearn
    print(f"\n--- Validacion Cruzada ({k} Folds) ---")

    tamano_fold = len(X) // k
    scores = []

    for i in range(k):
        # Indices del fold de prueba
        inicio = i * tamano_fold
        fin    = inicio + tamano_fold

        # Separar datos de prueba y entrenamiento para este fold
        X_val   = X[inicio:fin]
        y_val   = y[inicio:fin]
        X_train = np.concatenate([X[:inicio], X[fin:]], axis=0)
        y_train = np.concatenate([y[:inicio], y[fin:]], axis=0)

        # Evaluamos el modelo (sin reentrenar, solo medimos)
        _, acc = modelo.evaluate(X_val, y_val, verbose=0)
        scores.append(acc)
        print(f"  Fold {i+1}: {acc * 100:.2f}%")

    scores = np.array(scores)
    print(f"\n  Efectividad promedio: {scores.mean() * 100:.2f}%")
    print(f"  Desviacion estandar:  {scores.std() * 100:.2f}%")
    return scores


def grafica_validacion_cruzada(cv_scores):
    fig, ax = plt.subplots(figsize=(8, 5))
    folds = [f"Fold {i+1}" for i in range(len(cv_scores))]

    barras = ax.bar(folds, cv_scores * 100, color='#4C72B0', edgecolor='black', width=0.5)

    # Linea del promedio
    promedio = cv_scores.mean() * 100
    ax.axhline(y=promedio, color='red', linestyle='--', linewidth=1.5,
               label=f'Promedio: {promedio:.2f}%')

    # Porcentaje encima de cada barra
    for barra, score in zip(barras, cv_scores):
        ax.text(barra.get_x() + barra.get_width() / 2,
                barra.get_height() + 0.1,
                f'{score * 100:.2f}%',
                ha='center', va='bottom', fontsize=10)

    ax.set_ylim(85, 100)
    ax.set_ylabel('Exactitud (%)')
    ax.set_title('Resultados de Validacion Cruzada (5 Folds)')
    ax.legend()
    plt.tight_layout()
    plt.savefig('grafica_validacion_cruzada.png', dpi=150)
    print("  -> Guardada: grafica_validacion_cruzada.png")
    plt.show()


def grafica_matriz_confusion(y_real, y_predicho):
    cm = metrics.confusion_matrix(y_real, y_predicho)
    fig, ax = plt.subplots(figsize=(9, 7))

    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)

    clases = [str(i) for i in range(10)]
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xticklabels(clases)
    ax.set_yticklabels(clases)

    # Numero dentro de cada celda con color segun el fondo
    umbral = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color_texto = "white" if cm[i, j] > umbral else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color=color_texto, fontsize=11)

    ax.set_ylabel('Digito Real')
    ax.set_xlabel('Digito Predicho')
    ax.set_title('Matriz de Confusion')
    plt.tight_layout()
    plt.savefig('grafica_matriz_confusion.png', dpi=150)
    print("  -> Guardada: grafica_matriz_confusion.png")
    plt.show()


def grafica_precision_recall(y_real, y_predicho):
    reporte = metrics.classification_report(y_real, y_predicho, output_dict=True)

    digitos   = [str(i) for i in range(10)]
    precision = [reporte[d]['precision'] for d in digitos]
    recall    = [reporte[d]['recall']    for d in digitos]

    x = np.arange(10)
    ancho = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - ancho/2, precision, ancho, label='Precision', color='#4C72B0', edgecolor='black')
    ax.bar(x + ancho/2, recall,    ancho, label='Recall',    color='#DD8452', edgecolor='black')

    ax.set_xticks(x)
    ax.set_xticklabels(digitos)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Valor (0.0 a 1.0)')
    ax.set_xlabel('Digito')
    ax.set_title('Precision y Recall por Digito')
    ax.legend()
    # Linea de referencia en 1.0
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=0.8)
    plt.tight_layout()
    plt.savefig('grafica_precision_recall.png', dpi=150)
    print("  -> Guardada: grafica_precision_recall.png")
    plt.show()


if __name__ == "__main__":
    # 1. Cargar modelo
    modelo = cargar_modelo()

    # 2. Preparar dataset aumentado
    x_train, y_train, x_test, y_test = preparar_datos()

    # 3. Validacion cruzada manual sobre el conjunto de entrenamiento
    cv_scores = validacion_cruzada_manual(modelo, x_train, y_train, k=5)
    grafica_validacion_cruzada(cv_scores)

    # 4. Predicciones sobre el conjunto de prueba de MNIST
    print("\nGenerando predicciones sobre conjunto de prueba...")
    predicciones = modelo.predict(x_test, verbose=0)

    # Convertir probabilidades a clase predicha (la de mayor probabilidad)
    y_pred = np.argmax(predicciones, axis=1)

    # 5. Metricas en consola
    exactitud = metrics.accuracy_score(y_test, y_pred)
    print('\n==================================================')
    print(f"EXACTITUD FINAL EN TEST: {exactitud * 100:.2f}%")
    print('==================================================')
    print("\nReporte de Clasificacion completo:")
    print(metrics.classification_report(y_test, y_pred))

    # 6. Graficas
    print("\nGenerando graficas...")
    grafica_matriz_confusion(y_test, y_pred)
    grafica_precision_recall(y_test, y_pred)

    print("\nevaluation.py ejecutado correctamente.")