"""
Punto de Entrada Principal - Visión AI Engine.
Inicia la aplicación desde el paquete profesional 'app'.
"""

import sys
import traceback

def check_python_version():
    """Verifica que la versión de Python sea compatible."""
    if sys.version_info < (3, 10):
        print("ERROR: Se requiere Python 3.10 o superior.")
        print(f"Versión detectada: {sys.version}")
        sys.exit(1)

def main():
    """Inicializa y arranca la aplicación principal."""
    try:
        # Importación tardía para capturar errores de módulo o dependencias en el arranque
        from app.ui.main_window import VisionApp
        from app.utils.error_handler import log_error

        # Configurar ambiente y lanzar app
        app = VisionApp()
        app.protocol("WM_DELETE_WINDOW", app.on_closing)
        app.mainloop()
    except Exception as e:
        # Registro de errores fatales en el arranque usando el manejador estandar
        error_info = traceback.format_exc()
        try:
            from app.utils.error_handler import log_error
            log_error("EXE-SYS-BOOT-02", f"Crash crítico en arranque: {e}")
        except:
            print("="*60)
            print("CRASH CRÍTICO EN EL ARRANQUE:")
            print(error_info)
            print("="*60)
        
        # Intentar persistir error
        try:
            with open("startup_error.log", "w", encoding="utf-8") as f:
                f.write(error_info)
        except:
            pass
            
        # Intentar mostrar mensaje visual si tkinter está disponible
        try:
            import tkinter.messagebox as mb
            mb.showerror("Error de Inicio - Visión AI", f"Fallo crítico al iniciar la aplicación:\n\n{str(e)}")
        except:
            pass
        
        sys.exit(1)

if __name__ == "__main__":
    check_python_version()
    main()
