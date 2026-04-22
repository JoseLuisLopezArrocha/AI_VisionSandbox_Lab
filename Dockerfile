# Usar una imagen base de Python con soporte para OpenCV
FROM python:3.11-slim

# Instalar dependencias del sistema para OpenCV y Tkinter
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    python3-tk \
    && rm -rf /var/lib/apt/lists/*

# Directorio de trabajo
WORKDIR /app

# Copiar requerimientos e instalar
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto de la aplicación
COPY . .

# Comando por defecto (Lanzar la app principal)
# Nota: Requiere configuración de X11 para mostrar la GUI fuera del contenedor
CMD ["python", "main.py"]
