"""
Módulo de Utilidades y Configuración Global.

Contiene las constantes de diseño (colores, rutas), funciones de persistencia
de datos (JSON) y utilidades para la gestión de datasets y archivos.
"""
import os
import json
import cv2

# --- Constantes Globales ---
CONFIG_FILE = os.path.join(os.path.dirname(__file__), "zones.json")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

# Colores BGR para las zonas de conteo (estética premium)
ZONE_COLORS = [
    (113, 204, 46),   # Verde esmeralda
    (219, 152, 52),   # Azul dodger
    (60, 76, 231),    # Rojo alizarín
    (15, 196, 241),   # Amarillo sol
    (182, 89, 155),   # Amatista
    (34, 126, 230),   # Naranja zanahoria
]

def ensure_dataset_structure(class_name):
    """Crea la estructura de carpetas necesaria para un nuevo dataset en formato YOLO."""
    base = os.path.join(os.path.dirname(__file__), "datasets", class_name)
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        os.makedirs(os.path.join(base, sub), exist_ok=True)
    
    yaml_path = os.path.join(base, "data.yaml")
    if not os.path.exists(yaml_path):
        with open(yaml_path, "w", encoding="utf-8") as f:
            content = (
                f"train: images/train\n"
                f"val: images/val\n"
                f"nc: 1\n"
                f"names: ['{class_name}']\n"
            )
            f.write(content)
    return base

def get_next_capture_filename(class_name, dataset_dir):
    """Genera el siguiente nombre secuencial para una captura (ej: taxi_001.jpg)."""
    img_dir = os.path.join(dataset_dir, "images", "train")
    existing = [f for f in os.listdir(img_dir) if f.startswith(class_name) and f.endswith(".jpg")]
    idx = len(existing) + 1
    return f"{class_name}_{idx:03d}"

def save_app_config(source_url, zones, target_classes):
    """Guarda la configuración de zonas y filtros en el archivo JSON persistente."""
    try:
        data = {}
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)

        data[source_url] = {
            "zones": zones,
            "target_classes": target_classes
        }

        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Error al guardar configuración: {e}")
        return False

def load_app_config(source_url):
    """Carga la configuración guardada (zonas, filtros) para una fuente específica."""
    try:
        if not os.path.exists(CONFIG_FILE):
            return None

        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        return data.get(source_url)
    except Exception as e:
        print(f"Error al cargar configuración: {e}")
        return None
