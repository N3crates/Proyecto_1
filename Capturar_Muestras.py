import tkinter as tk
from PIL import Image, ImageDraw
import os

# CAPTURAR_MUESTRAS.PY
# Herramienta para dibujar y guardar 5 muestras de cada digito (0-9)
# Guarda las imagenes en la carpeta mis_digitos/ junto a este archivo

# Carpeta de salida junto al archivo actual
RUTA_SALIDA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mis_digitos')
TAMANO = 280

os.makedirs(RUTA_SALIDA, exist_ok=True)


class CapturarDigitos:

    def __init__(self, root):
        self.root = root

        # Control de digitos y muestras
        self.digito_actual = 0
        self.muestra_actual = 1
        self.total_muestras = 5

        self.root.title("Captura de Digitos")
        self.root.configure(bg='#2b2b2b')
        self.root.resizable(False, False)

        # Texto principal
        self.label_instruccion = tk.Label(
            self.root,
            text=f"Dibuja el digito: {self.digito_actual} ({self.muestra_actual}/{self.total_muestras})",
            font=("Arial", 20, "bold"),
            bg='#2b2b2b', fg='white'
        )
        self.label_instruccion.pack(pady=10)

        # Canvas negro para dibujar
        self.canvas = tk.Canvas(
            self.root, width=TAMANO, height=TAMANO,
            bg='black', cursor='crosshair'
        )
        self.canvas.pack(padx=20)

        # Imagen PIL donde se guarda el trazo real
        self.imagen = Image.new("L", (TAMANO, TAMANO), color=0)
        self.draw = ImageDraw.Draw(self.imagen)

        # Eventos del mouse
        self.canvas.bind("<B1-Motion>", self.dibujar)

        # Teclas rapidas
        self.root.bind("<Return>", lambda e: self.guardar_y_siguiente())
        self.root.bind("c", lambda e: self.limpiar())

        # Frame botones
        frame_botones = tk.Frame(self.root, bg='#2b2b2b')
        frame_botones.pack(pady=10)

        # Boton limpiar
        tk.Button(
            frame_botones, text="Limpiar",
            font=("Arial", 13), width=12,
            bg='#555', fg='white',
            command=self.limpiar
        ).grid(row=0, column=0, padx=10)

        # Boton guardar
        self.btn_guardar = tk.Button(
            frame_botones, text="Guardar y Siguiente",
            font=("Arial", 13), width=18,
            bg='#4C72B0', fg='white',
            command=self.guardar_y_siguiente
        )
        self.btn_guardar.grid(row=0, column=1, padx=10)

        # Progreso
        self.label_progreso = tk.Label(
            self.root, text="Progreso: 0/50",
            font=("Arial", 12),
            bg='#2b2b2b', fg='#aaaaaa'
        )
        self.label_progreso.pack(pady=5)

    def dibujar(self, event):
        radio = 12
        x, y = event.x, event.y

        # Dibujar en el canvas visual
        self.canvas.create_oval(x-radio, y-radio, x+radio, y+radio, fill='white', outline='white')

        # Dibujar en la imagen PIL
        self.draw.ellipse((x-radio, y-radio, x+radio, y+radio), fill=255)

    def limpiar(self):
        self.canvas.delete("all")
        self.imagen = Image.new("L", (TAMANO, TAMANO), color=0)
        self.draw = ImageDraw.Draw(self.imagen)

    def guardar_y_siguiente(self):
        # Guardar imagen con formato digito_muestra.png (ejemplo: 0_1.png)
        ruta = os.path.join(RUTA_SALIDA, f"{self.digito_actual}_{self.muestra_actual}.png")
        self.imagen.save(ruta)
        print(f"  -> Guardado: {ruta}")

        # Avanzar a la siguiente muestra
        self.muestra_actual += 1

        # Si termino las 5 muestras del digito actual
        if self.muestra_actual > self.total_muestras:
            self.muestra_actual = 1
            self.digito_actual += 1

        # Calcular progreso total
        progreso = (self.digito_actual * self.total_muestras) + (self.muestra_actual - 1)
        self.label_progreso.config(text=f"Progreso: {progreso}/50")

        if self.digito_actual >= 10:
            # Todos los digitos capturados
            self.label_instruccion.config(text="¡Listo! Todos los digitos guardados.")
            self.btn_guardar.config(state='disabled')
            self.canvas.config(bg='#1a3a1a')
            print("\n50 imagenes guardadas en 'mis_digitos/'.")
        else:
            self.label_instruccion.config(
                text=f"Dibuja el digito: {self.digito_actual} ({self.muestra_actual}/{self.total_muestras})"
            )
            self.limpiar()


if __name__ == "__main__":
    root = tk.Tk()
    app = CapturarDigitos(root)
    root.mainloop()