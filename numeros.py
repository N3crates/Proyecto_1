import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler  # Crucial para alcanzar >99%
from sklearn import metrics

# =====================================================================
# 1. CARGAR DATASET ORIGINAL
# =====================================================================
digits = load_digits()

# =====================================================================
# 2. SECCIÓN DE AUMENTO DE DATOS
# =====================================================================
#Cuando se tenga la base de datos propia, cambiar el estado a False:
usar_base_simulada = False

if usar_base_simulada:
    print("Generando base simulada de alta fidelidad...")
    # Extraemos muestras reales perfectas del dataset para la simulación limpia
    X_propias = np.array([digits.data[digits.target == i][0] for i in range(10)])
    y_propias = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
else:
    print("\nCargando los datos reales...")
    X_propias, y_propias = [], []
    ruta_carpeta = "./mis_digitos"
    for digito in range(10):
        archivo = os.path.join(ruta_carpeta, f"{digito}.png")
        if os.path.exists(archivo):
            img = Image.open(archivo).convert('L').resize((8, 8))
            img_array = np.array(img)
            # Si el fondo es blanco con letras negras, se invertira la foto:
            # img_array = 255 - img_array  
            img_normalizada = (img_array / 255.0) * 16.0
            X_propias.append(img_normalizada.flatten())
            y_propias.append(digito)
        else:
            raise FileNotFoundError(f"Falta el archivo requerido: {archivo}")
    X_propias, y_propias = np.array(X_propias), np.array(y_propias)

# Unimos los datos
X_aumentado = np.vstack((digits.data, X_propias))
y_aumentado = np.hstack((digits.target, y_propias))

# =====================================================================
# 3. PREPROCESAMIENTO 
# =====================================================================
# El escalador centra los datos en 0 y ajusta la varianza.
scaler = StandardScaler()
X_escalado = scaler.fit_transform(X_aumentado)

# =====================================================================
# 4. RED NEURONAL 
# =====================================================================
clf = MLPClassifier(
    solver='adam', 
    activation='relu',
    hidden_layer_sizes=(256, 128, 64),  
    alpha=1e-5,                         
    learning_rate='adaptive',           
    learning_rate_init=0.001, 
    max_iter=3000,                      
    tol=1e-7,                           
    random_state=42                     
)

# =====================================================================
# 5. VALIDACIÓN CRUZADA 
# =====================================================================
print('\n--- Validación Cruzada (5 Folds) ---')
cv_scores = cross_val_score(clf, X_escalado, y_aumentado, cv=5)
print(f"Exactitud por iteración: {cv_scores}")
print(f"Efectividad promedio global: {cv_scores.mean() * 100:.2f}%")

# =====================================================================
# 6. MÉTRICAS PARA LA RÚBRICA 
# =====================================================================

x_train, x_test, y_train, y_test = train_test_split(
    X_escalado, y_aumentado, test_size=0.1, random_state=42, stratify=y_aumentado
)

clf.fit(x_train, y_train)
yout = clf.predict(x_test)

print('\n==================================================')
print(f"EXACTITUD FINAL (SCORE EN TEST): {clf.score(x_test, y_test) * 100:.2f}%")
print('==================================================')

print("\n--- Reporte de Clasificación (Verifica 'precision' por dígito) ---")
print(metrics.classification_report(y_test, yout))

print("Matriz de Confusión Resultante:")
print(metrics.confusion_matrix(y_test, yout))