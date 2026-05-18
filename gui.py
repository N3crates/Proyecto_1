#----------
# librerias
#----------
# tiker se usa para la creacion de la interfaz grafica
import tkinter as tk
# PIL se usa para trabajar con imagenes
import Image, ImageDraw from PIL
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
