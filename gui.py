#----------
# librerias
#----------
# tiker se usa para la creacion de la interfaz grafica
import tkinter as tk
# PIL se usa para trabajar con imagenes
from PIL import Image, ImageDraw
# Numpy se usa para manejar arreglos de numeros
import numpy as np
# pickle se usa para cargar el modelo entrenado
import pickle

#--------------------------
# Cargar el modelo y scaler
#--------------------------
# Cargar el modelo entrenado

with open("modelo.pkl", "rb") as f:
    modelo = pickle.load(f)

# Cargar el scaler utilizado durante el entrenamiento
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

#-------------------------------
# Clase principal de la interfaz
#-------------------------------
class DigitRecognizerGUI:
    #----------------------
    # Constructor principal
    #----------------------
    def __init__(self, root):
        # Guardamos la ventana principal
        self.root = root

        # Titulo de la ventana
        self.root.title("--Reconocimiento de Digitos Manuscritos--")

        # Tamaño de la ventana
        self.root.geometry("1200x400")

        # Color de fondo
        self.root.configure(bg="lightgray")

        # -----------------------------------------------
        # Listas para almacenar los canvas y sus imagenes
        # -----------------------------------------------
        # Lista donde gardamos los canvas
        self.canvases = []

        # Lista donde guardamos las imagenes PIL
        self.images = []

        # Lista para dibujar sobre imagenes PIL
        self.draws = []

        # ---------------------------------------------
        # Frame principal para los 5 espacios de dibujo
        # ---------------------------------------------
        # Contador principal
        self.canvas_frame = tk.Frame(self.root, bg="lightgray")

        # Mostrar el frame en la ventana
        self.canvas_frame.pack(pady=20)

        #-------------------------
        # Creacion de los 5 canvas
        #-------------------------
        # Ciclo para crear 5 espacios de dibujo
        for i in range(5):
            #--------------------
            # Crear canvas visual
            #--------------------
            canvas = tk.Canvas(self.canvas_frame, width=150, height=150, bg="white", bd=2, relief="solid")

            # Posicionar canvas horizontalmente
            canvas.grid(row=0, column=i, padx=10)

            #------------------------------------
            # Crear imagen PIL asociada al canvas
            #------------------------------------
            # Imagen negra donde se dibujara internamente 
            image = Image.new("L", (150, 150), color=0)

            # Objeto para dibujar sobre la imagen
            draw = ImageDraw.Draw(image)

            #--------------------
            # Guardar referencias
            #--------------------
            self.canvases.append(canvas)
            self.images.append(image)
            self.draws.append(draw)

            #-------------------------------
            # Eventos del mause para dibujar
            #-------------------------------
            canvas.bind("<B1-Motion>", lambda event, index=i: self.draw(event, index))
    
        #-----------------
        # Frame de botones
        #-----------------
        self.button_frame = tk.Frame(self.root, bg="lightgray")
        self.button_frame.pack(pady=20)

        #--------------
        # Boton limpiar
        #--------------
        self.clear_button = tk.Button(
            self.button_frame,
            text="Limpiar",
            font=("Arial", 14),
            width=15,
            command=self.clear_canvases
        )
        self.clear_button.grid(row=0, column=0, padx=20)

    #----------------------------------
    # Fucnion para dibujar en el canvas
    #----------------------------------
    def draw(self, event, index):
        # Obtener canvas actual
        canvas = self.canvases[index]

        # Obtener objeto draw actual
        draw = self.draws[index]

        # Coordenadas del mause
        x = event.x
        y = event.y

        # Tamaño del pincel
        radio = 8

        # Dibujar visulamente en el canvas
        canvas.create_oval(x-radio, y-radio, x+radio, y+radio, fill="black", outline="black")

        # Dibujar tambien en la imagen PIL
        draw.ellipse((x-radio, y-radio, x+radio, y+radio), fill=255)

    #--------------------------------
    # Funcion para limpiar los canvas
    #--------------------------------
    def clear_canvases(self):
        # Recorrer los 5 canvas
        for i in range(5):
            # Limpiar el canvas visual
            self.canvases[i].delete("all")
            
            # Limpiar visualmente el canvas
            self.images[i] = Image.new("L", (150, 150), color=0)

            # Crear nuevo objeto draw
            self.draws[i] = ImageDraw.Draw(self.images[i])

        # Boton predecir
        self.predict_button = tk.Button(self.button_frame, text="Predecir", font=("Arial", 14), width=15, command=self.predict_number)
        self.predict_button.grid(row=0, column=1, padx=20)

        # Label para mostrar resultado
        self.result_label = tk.Label(self.root, text="Numero reconocido:", font=("Arial", 20), bg="lightgray")
        self.result_label.pack(pady=20)

    #----------------------------------
    # Funcion para predecir los digitos
    #----------------------------------
    def predict_number(self):
        #Lista donde se guardaran los digitos reconocidos
        predicted_digits = []

        # Recorrer los 5 canvas
        for image in self.images:
            # Redimensionar imagen a 8x8
            resized_image = image.resize((8, 8))

            # Convertir imagen a arreglo numpy
            img_array = np.array(resized_image)

            # Invertir colores (si se ocupa)
            img_array = 255 - img_array

            # Normalizar valores de 0 - 16
            img_normalizada = (img_array / 255.0) * 16.0

            # Aplanar imagen
            img_flat = img_normalizada.flatten()

            # Aplicar scaler
            img_scaled = scaler.transform([img_flat])

            # Realizar prediccion
            prediction = modelo.predict(img_scaled)[0]

            # Guardar resultado
            predicted_digits.append(str(prediction))
        
        # Unir los 5 digitos
        final_number = "".join(predicted_digits)

        # Mostrar resultado
        self.result_label.config(text=f"Numero reconocido: {final_number}")
