import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn import metrics

from data_processing import cargar_dataset_original, cargar_imagenes_propias, combinar_datasets


# EVALUATION.PY
# Carga el modelo entrenado por el compañero 1, aplica validacion
# cruzada y genera graficas de metricas de desempeno.
# Requiere: modelo.pkl y scaler.pkl en la misma carpeta.



def cargar_modelo():
    print("Cargando modelo entrenado (modelo.pkl)...")
    with open("modelo.pkl", "rb") as f:
        modelo = pickle.load(f)

    print("Cargando scaler (scaler.pkl)...")
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    print("  -> Modelo y scaler cargados correctamente.")
    return modelo, scaler


def preparar_datos(scaler):
    # Cargamos y combinamos el dataset original con las imagenes propias
    X_orig, y_orig = cargar_dataset_original()
    X_prop, y_prop = cargar_imagenes_propias("./mis_digitos")
    X, y = combinar_datasets(X_orig, y_orig, X_prop, y_prop)

    # Usamos transform (no fit_transform) para respetar el scaler del entrenamiento
    X_escalado = scaler.transform(X)

    return X_escalado, y


def validacion_cruzada(modelo, X, y, k=5):
    print(f"\n--- Validacion Cruzada ({k} Folds) ---")
    cv_scores = cross_val_score(modelo, X, y, cv=k)

    for i, score in enumerate(cv_scores):
        print(f"  Fold {i+1}: {score * 100:.2f}%")

    print(f"\n  Efectividad promedio: {cv_scores.mean() * 100:.2f}%")
    print(f"  Desviacion estandar:  {cv_scores.std() * 100:.2f}%")

    return cv_scores


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

    # Etiquetas de los ejes
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

    digitos = [str(i) for i in range(10)]
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
    # Linea de referencia en 1.0 (rendimiento perfecto)
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=0.8)
    plt.tight_layout()
    plt.savefig('grafica_precision_recall.png', dpi=150)
    print("  -> Guardada: grafica_precision_recall.png")
    plt.show()


if __name__ == "__main__":
    # 1. Cargar modelo y scaler del compañero 1
    modelo, scaler = cargar_modelo()

    # 2. Preparar dataset aumentado con las imagenes propias
    X, y = preparar_datos(scaler)

    # 3. Validacion cruzada y su grafica
    cv_scores = validacion_cruzada(modelo, X, y, k=5)
    grafica_validacion_cruzada(cv_scores)

    # 4. Division para obtener predicciones y graficar metricas
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )

    # 5. Predicciones sobre el conjunto de prueba
    y_pred = modelo.predict(X_test)

    # 6. Metricas en consola
    print('\n==================================================')
    print(f"EXACTITUD FINAL EN TEST: {modelo.score(X_test, y_test) * 100:.2f}%")
    print('==================================================')
    print("\nReporte de Clasificacion completo:")
    print(metrics.classification_report(y_test, y_pred))

    # 7. Graficas
    print("\nGenerando graficas...")
    grafica_matriz_confusion(y_test, y_pred)
    grafica_precision_recall(y_test, y_pred)

    print("\nevaluation.py ejecutado correctamente.")