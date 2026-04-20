"""
Script de entrenamiento para modelos YOLO personalizados.

Uso:
    python train_custom.py

Requisitos:
    1. Tener imágenes anotadas en formato YOLO dentro de datasets/taxi/
    2. El archivo datasets/taxi/data.yaml debe apuntar a las carpetas correctas.

El modelo resultante se guarda automáticamente en models/custom/ para que
la aplicación principal lo cargue junto al modelo COCO.
"""

import os
import shutil
from ultralytics import YOLO

# Rutas del proyecto
BASE_DIR = os.path.dirname(__file__)
DATASET_DIR = os.path.join(BASE_DIR, "datasets", "taxi")
DATA_YAML = os.path.join(DATASET_DIR, "data.yaml")
CUSTOM_MODELS_DIR = os.path.join(BASE_DIR, "models", "custom")
OUTPUT_NAME = "taxi_detector"


def check_dataset():
    """Verifica que el dataset tiene imágenes anotadas."""
    train_imgs = os.path.join(DATASET_DIR, "images", "train")
    train_labels = os.path.join(DATASET_DIR, "labels", "train")

    if not os.path.exists(train_imgs):
        print("ERROR: No se encuentra la carpeta images/train")
        return False

    images = [f for f in os.listdir(train_imgs) if f.endswith((".jpg", ".png", ".jpeg"))]
    labels = [f for f in os.listdir(train_labels) if f.endswith(".txt")] if os.path.exists(train_labels) else []

    print(f"Imágenes de entrenamiento: {len(images)}")
    print(f"Archivos de anotación:     {len(labels)}")

    if len(images) == 0:
        print("ERROR: No hay imágenes de entrenamiento.")
        print(f"       Coloca imágenes anotadas en: {train_imgs}")
        return False

    if len(labels) == 0:
        print("ERROR: No hay archivos de anotación (.txt).")
        print(f"       Coloca las anotaciones YOLO en: {train_labels}")
        return False

    return True


def train():
    """Entrena un YOLOv11 Nano personalizado para la clase 'taxi'."""
    if not check_dataset():
        return

    print("\n" + "=" * 50)
    print("  ENTRENAMIENTO DE MODELO PERSONALIZADO")
    print("=" * 50)

    # Partir de un modelo preentrenado COCO (nano) para aprovechar el transfer learning
    model = YOLO("yolo11n.pt")

    # Entrenar solo con el dataset de taxi
    results = model.train(
        data=DATA_YAML,
        epochs=50,
        imgsz=640,
        batch=16,
        name=OUTPUT_NAME,
        patience=10,          # Early stopping si no mejora
        freeze=10,            # Congelar las 10 primeras capas del backbone
        project=os.path.join(BASE_DIR, "runs"),
    )

    # Copiar el mejor modelo a models/custom/
    best_model = os.path.join(BASE_DIR, "runs", OUTPUT_NAME, "weights", "best.pt")
    if os.path.exists(best_model):
        dest = os.path.join(CUSTOM_MODELS_DIR, "taxi.pt")
        shutil.copy2(best_model, dest)
        print(f"\nModelo guardado en: {dest}")
        print("Reinicia la aplicación para cargar el nuevo modelo.")
    else:
        print("ERROR: No se generó el modelo. Revisa los logs de entrenamiento.")


if __name__ == "__main__":
    train()
