"""
Módulo de Utilidades y Configuración Global.
Gestiona rutas, colores y persistencia de configuración del sistema.
"""

import os
import json
import cv2
from .error_handler import log_error

# --- Determinación de Rutas Base ---
# Root es 3 niveles arriba de app/utils/helpers.py
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_DIR = os.path.join(ROOT_DIR, "config")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
DATASETS_DIR = os.path.join(ROOT_DIR, "datasets")
LOGS_DIR = os.path.join(ROOT_DIR, "telemetry_logs")

# Archivos específicos
ZONES_CONFIG = os.path.join(CONFIG_DIR, "zones.json")
EVENTS_CONFIG = os.path.join(CONFIG_DIR, "events_config.json")

# Asegurar existencia de carpetas críticas
os.makedirs(CONFIG_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATASETS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# --- Constantes de Diseño ---
ZONE_COLORS = [
    (113, 204, 46),   # Verde esmeralda
    (219, 152, 52),   # Azul dodger
    (60, 76, 231),    # Rojo alizarín
    (15, 196, 241),   # Amarillo sol
    (182, 89, 155),   # Amatista
    (34, 126, 230),   # Naranja zanahoria
]

def ensure_dataset_structure(class_name):
    """Crea la estructura de carpetas estándar YOLO para un dataset de entrenamiento."""
    try:
        base = os.path.join(DATASETS_DIR, class_name)
        for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
            os.makedirs(os.path.join(base, sub), exist_ok=True)
        
        # Formato estándar Ultralytics YOLO — compatible con `yolo train data=data.yaml`
        yaml_path = os.path.join(base, "data.yaml")
        if not os.path.exists(yaml_path):
            abs_path = os.path.abspath(base).replace("\\", "/")
            with open(yaml_path, "w", encoding="utf-8") as f:
                content = (
                    f"# Dataset: {class_name}\n"
                    f"# Generado por VisionSandbox Lab\n"
                    f"path: {abs_path}\n"
                    f"train: images/train\n"
                    f"val: images/val\n"
                    f"\n"
                    f"# Clases\n"
                    f"nc: 1\n"
                    f"names:\n"
                    f"  0: {class_name}\n"
                )
                f.write(content)
        return base
    except Exception as e:
        log_error("EXE-UTL-HELP-02", f"Error creando estructura dataset: {e}")
        return None

def get_next_capture_filename(class_name, dataset_dir):
    """Genera el siguiente nombre secuencial para una captura."""
    try:
        img_dir = os.path.join(dataset_dir, "images", "train")
        existing = [f for f in os.listdir(img_dir) if f.startswith(class_name) and f.endswith(".jpg")]
        idx = len(existing) + 1
        return f"{class_name}_{idx:03d}"
    except Exception as e:
        log_error("EXE-UTL-HELP-02", f"Error generando nombre captura: {e}")
        return f"{class_name}_error"

def save_app_config(source_url, zones, target_classes):
    """Guarda la configuración de zonas y filtros."""
    try:
        data = {}
        if os.path.exists(ZONES_CONFIG):
            with open(ZONES_CONFIG, "r", encoding="utf-8") as f:
                data = json.load(f)

        data[source_url] = {
            "zones": zones,
            "target_classes": target_classes
        }

        with open(ZONES_CONFIG, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        log_error("EXE-UTL-HELP-02", f"Error al guardar config de zonas: {e}")
        return False

def load_app_config(source_url):
    """Carga la configuración guardada para una fuente específica."""
    try:
        if not os.path.exists(ZONES_CONFIG):
            return None

        with open(ZONES_CONFIG, "r", encoding="utf-8") as f:
            data = json.load(f)

        return data.get(source_url)
    except Exception as e:
        log_error("EXE-UTL-HELP-02", f"Error al cargar config de zonas: {e}")
        return None
