import os
import sys

# Añadir el directorio raíz al path para importar el detector
sys.path.append(os.getcwd())

import numpy as np
from detector import ObjectDetector

def test_model_warmup():
    print("=== TEST DE WARM-UP DEL DETECTOR ===")
    
    # 1. Inicializar el detector (esto dispara scan_models y _load_custom_models)
    try:
        detector = ObjectDetector()
        print("\n[OK] Detector inicializado.")
        
        # 2. Verificamos la familia y modelo cargado por defecto
        print(f"Arquitectura actual: {detector.current_family}")
        print(f"Modelo activo: {detector.active_name}")
        
        # 3. Obtener nombres de clases
        classes = detector.get_class_names()
        
        # 4. Verificar si tenemos nombres reales o genéricos
        if not classes:
            print("[ERROR] No se obtuvieron clases.")
        else:
            print(f"\nSe han detectado {len(classes)} clases.")
            # Mostrar las primeras 5 para no saturar
            top_classes = sorted(classes.items())[:5]
            for cid, cname in top_classes:
                print(f"  ID {cid}: {cname}")
            
            # Comprobación de éxito
            if any("Clase " in str(v) for v in classes.values()):
                if len(classes) == 80 and classes[0] == "Clase 0":
                    print("\n[FALLO] Se detectaron nombres genéricos (Fallback COCO). El warm-up podría no haber funcionado.")
                else:
                    print("\n[AVISO] Algunos nombres son genéricos, pero otros podrían ser válidos.")
            else:
                print("\n[ÉXITO] Los nombres de las clases parecen ser reales y no genéricos.")

    except Exception as e:
        print(f"\n[ERROR FATAL] Durante la prueba: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_model_warmup()
