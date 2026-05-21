============================================================
PROYECTO: RECONOCIMIENTO DE DÍGITOS MANUSCRITOS
============================================================

Descripción:
Este proyecto implementa un sistema de reconocimiento de
dígitos manuscritos utilizando redes neuronales y el dataset
MNIST. El sistema permite dibujar 5 dígitos utilizando el
mouse y predice automáticamente el número escrito.

El proyecto fue desarrollado en Python utilizando:
- TensorFlow / Keras
- Tkinter
- NumPy
- Pillow
- Matplotlib
- Scikit-learn

============================================================
REQUISITOS
============================================================

Se recomienda utilizar Python 3.11.

Instalar las librerías necesarias ejecutando el siguiente
comando en la terminal:

py -3.11 -m pip install tensorflow numpy pillow matplotlib scikit-learn

============================================================
ESTRUCTURA DEL PROYECTO
============================================================

Proyecto_1/
│
├── main.py
├── gui.py
├── data_processing.py
├── evaluation.py
├── entrenamiento.py
├── modelo_combinado_mnist.h5
├── mis_digitos/
└── README.txt

============================================================
DESCRIPCIÓN DE ARCHIVOS
============================================================

main.py
Archivo principal del proyecto. Ejecuta la interfaz gráfica.

gui.py
Contiene la interfaz gráfica del sistema.
Permite:
- Dibujar dígitos
- Limpiar los canvas
- Realizar predicciones

data_processing.py
Carga el dataset MNIST y las imágenes propias del usuario.
También combina ambos datasets para aumentar los datos de
entrenamiento.

evaluation.py
Genera:
- Validación cruzada
- Métricas de desempeño
- Matriz de confusión
- Gráficas de precisión y recall

entrenamiento.py
Entrena la red neuronal y guarda el modelo entrenado.

modelo_combinado_mnist.h5
Modelo de red neuronal entrenado.

mis_digitos/
Carpeta que contiene imágenes propias de los dígitos
(0_1.png hasta 9_5.png).

============================================================
INSTRUCCIONES DE EJECUCIÓN
============================================================

1. Abrir una terminal dentro de la carpeta del proyecto.

2. (Opcional) Entrenar nuevamente el modelo:

py -3.11 entrenamiento.py

Esto generará el archivo:
modelo_combinado_mnist.h5

3. Ejecutar la aplicación principal:

py -3.11 main.py

============================================================
USO DEL SISTEMA
============================================================

1. Dibujar un dígito en cada uno de los 5 espacios.
2. Presionar el botón "Predecir".
3. El sistema mostrará el número reconocido.
4. Presionar "Limpiar" para borrar los dibujos y comenzar
   nuevamente.

============================================================
FUNCIONAMIENTO GENERAL
============================================================

El sistema:
1. Captura el dibujo realizado por el usuario.
2. Convierte la imagen a escala de grises.
3. Redimensiona la imagen a 28x28 píxeles.
4. Normaliza los valores de los píxeles.
5. Envía la imagen al modelo entrenado.
6. Obtiene la predicción del dígito.
7. Muestra el resultado final en pantalla.

============================================================
DATASET UTILIZADO
============================================================

Se utilizó el dataset MNIST, el cual contiene miles de
imágenes de dígitos manuscritos del 0 al 9.

Además, se agregaron imágenes propias para mejorar el
rendimiento del sistema.

============================================================
AUTORES
============================================================
Nestor Isaac Hernandez Duran
Juan Carlos Zuñiga Alfaro
Carlos Aviles
Miguel Crespo Flores

Proyecto desarrollado para la materia de Inteligencia Artificial.

============================================================
