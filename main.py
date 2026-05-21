# Archivo principal del programa
#-------------------------------

# Importar tkinter para crear la ventana principal
import tkinter as tk
# Importar la interfaz grafica
from gui import DigitRecognizerGUI

#------------------
# Funcion Principal
#------------------
def main():
    # Crear ventana principal
    root = tk.Tk()

    # Crear aplicacion GUI
    app = DigitRecognizerGUI(root)

    # Ejecutar Ventana
    root.mainloop()

#--------------------
# Ejecucion Principal
#--------------------
if __name__ == "__main__":
    main()