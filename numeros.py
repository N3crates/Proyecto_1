<<<<<<< HEAD
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
usar_base_simulada = True

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
=======
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

# =====================================================================
# 7. INTEGRACIÓN: GUI + MODELO (Integrante 4)
# =====================================================================
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageOps, ImageDraw
import traceback


def preprocesar_imagen(img_pil):
    img = img_pil.convert('L')
    img = ImageOps.invert(img)
    img = img.resize((8, 8), Image.LANCZOS)
    arr = (np.array(img, dtype=np.float32) / 255.0) * 16.0
    return arr.flatten().reshape(1, -1)


def predecir_digito(img_pil):
    vector = preprocesar_imagen(img_pil)
    vector_escalado = scaler.transform(vector)
    prediccion = clf.predict(vector_escalado)[0]
    confianza = clf.predict_proba(vector_escalado)[0][prediccion] * 100
    return int(prediccion), float(confianza)


class AppDigitos:
    def __init__(self, root):
        self.root = root
        self.root.title("Reconocimiento de Dígitos")
        self.root.resizable(False, False)
        self.tam = 280

        self.canvas = tk.Canvas(root, width=self.tam, height=self.tam,
                                bg='white', cursor='pencil')
        self.canvas.grid(row=0, column=0, columnspan=3, padx=10, pady=10)
        self.canvas.bind("<B1-Motion>", self.dibujar)

        self.imagen = Image.new("L", (self.tam, self.tam), color=255)
        self.draw = ImageDraw.Draw(self.imagen)

        tk.Button(root, text="Predecir", width=12,
                  command=self.on_predecir).grid(row=1, column=0, pady=5)
        tk.Button(root, text="Limpiar", width=12,
                  command=self.on_limpiar).grid(row=1, column=1, pady=5)
        tk.Button(root, text="Salir", width=12,
                  command=root.destroy).grid(row=1, column=2, pady=5)

        self.resultado = tk.Label(root, text="Dibuja un dígito (0-9)",
                                  font=("Arial", 16))
        self.resultado.grid(row=2, column=0, columnspan=3, pady=10)
        self.hay_trazo = False

    def dibujar(self, evento):
        r = 10
        x, y = evento.x, evento.y
        self.canvas.create_oval(x - r, y - r, x + r, y + r,
                                fill='black', outline='black')
        self.draw.ellipse([x - r, y - r, x + r, y + r], fill=0)
        self.hay_trazo = True

    def on_limpiar(self):
        self.canvas.delete("all")
        self.imagen.paste(255, [0, 0, self.tam, self.tam])
        self.resultado.config(text="Dibuja un dígito (0-9)")
        self.hay_trazo = False

    def on_predecir(self):
        try:
            if not self.hay_trazo:
                messagebox.showwarning("Entrada vacía",
                                       "Primero dibuja un dígito.")
                return
            digito, confianza = predecir_digito(self.imagen)
            self.resultado.config(
                text=f"Predicción: {digito}   (confianza: {confianza:.1f}%)"
            )
        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Error en la predicción",
                                 f"Ocurrió un problema:\n\n{e}")


def prueba_integracion():
    idx = np.where(digits.target == 3)[0][0]
    matriz = digits.images[idx]
    img_simulada = Image.fromarray(
        (255 - (matriz / 16.0) * 255).astype(np.uint8)
    ).resize((280, 280))
    pred, conf = predecir_digito(img_simulada)
    print(f"\n[Prueba end-to-end] Esperado: 3 | Predicho: {pred} | Confianza: {conf:.1f}%")


if __name__ == "__main__":
    prueba_integracion()
    root = tk.Tk()
    app = AppDigitos(root)
    root.mainloop()
>>>>>>> c5b09f3c05040a902c59d04f193f65609bd63c13
