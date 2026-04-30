"""
Módulo de Gestión de Errores Centralizado.
Define los códigos de error estándar y proporciona utilidades para el registro y depuración.
"""

from typing import Optional, Dict

ERROR_CODES: Dict[str, str] = {
    # --- SISTEMA / BOOTSTRAP (SYS) ---
    "EXE-SYS-INIT-01": "Fallo crítico en la inicialización de la aplicación.",
    "EXE-SYS-BOOT-02": "Error al cargar los módulos de arranque.",
    "EXE-SYS-ENV-03": "Archivo de secretos (.env) no encontrado o corrupto.",

    # --- CORE / MOTORES (COR) ---
    "EXE-COR-CONN-01": "Error de conexión con la fuente de vídeo (Stream/Archivo).",
    "EXE-COR-DET-02": "Fallo en el motor de inferencia (Detección).",
    "EXE-COR-LOAD-03": "Error al cargar pesos del modelo .pt.",
    "EXE-COR-HW-04": "Error en el diagnóstico de hardware/GPU.",
    "EXE-COR-EVT-05": "Fallo en el motor de eventos o evaluación de reglas.",

    # --- INTERFAZ / GUI (GUI) ---
    "EXE-GUI-MAIN-01": "Error al construir el dashboard principal.",
    "EXE-GUI-COMP-02": "Fallo en un componente visual o ventana modal.",
    "EXE-GUI-DRAW-03": "Error de renderizado en el canvas de vídeo.",

    # --- UTILIDADES (UTL) ---
    "EXE-UTL-LOG-01": "Error al escribir en el log de telemetría (CSV).",
    "EXE-UTL-HELP-02": "Fallo en funciones de utilidad o persistencia JSON.",
}

def get_error_msg(code: str, details: Optional[str] = None) -> str:
    """Devuelve un mensaje formateado basado en el código de error."""
    base_msg = ERROR_CODES.get(code, "Error desconocido")
    full_msg = f"[{code}] {base_msg}"
    if details:
        full_msg += f" | Detalle: {details}"
    return full_msg

def log_error(code: str, details: Optional[str] = None) -> None:
    """Imprime el error en consola (extensible a logs de archivo)."""
    print(f"[ERROR] {get_error_msg(code, details)}")
