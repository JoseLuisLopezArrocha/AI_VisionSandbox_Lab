import os
import sys
import customtkinter as ctk

# Añadir el directorio raíz al path para importar los módulos locales
sys.path.append(os.getcwd())

from detector import ObjectDetector
from ui_components import ClassFilterWindow

def test_popup_initialization():
    print("=== INICIANDO PRUEBA DE DEPURACIÓN DEL POPUP ===")
    
    # 1. Preparar el entorno de CTK (sin ventana principal visible para el test)
    root = ctk.CTk()
    root.withdraw()
    
    # 2. Inicializar detector
    print("\n[STEP 1] Inicializando detector...")
    detector = ObjectDetector() 
    # El detector ahora tiene el warm-up que pusimos antes
    
    # 3. Simular la apertura del popup
    print("\n[STEP 2] Simulando apertura de ClassFilterWindow...")
    
    def on_apply(targets):
        print(f"Filtro aplicado: {targets}")
    
    # Esto disparará el [DEBUG POPUP] que acabamos de añadir
    popup = ClassFilterWindow(root, detector, None, on_apply)
    
    print("\n[STEP 3] Verificando estado interno del popup...")
    print(f"Popup all_classes size: {len(popup.all_classes)}")
    
    # Limpieza
    popup.destroy()
    root.destroy()
    print("\n=== PRUEBA FINALIZADA ===")

if __name__ == "__main__":
    test_popup_initialization()
