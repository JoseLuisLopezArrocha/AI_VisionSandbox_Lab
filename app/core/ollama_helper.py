import requests
import json
import os
import threading

# Ruta de caché persistente
CACHE_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "utils", "ollama_cache.json")

def get_ollama_models_with_vision(url):
    """
    Obtiene la lista de modelos de Ollama y verifica cuáles tienen capacidades de visión
    filtrando estrictamente por palabras clave especializadas en visión en su nombre.
    """
    try:
        # 1. Obtener lista básica de modelos
        resp = requests.get(f"{url.rstrip('/')}/api/tags", timeout=3)
        if resp.status_code != 200:
            return None, None, f"Error HTTP {resp.status_code}"
            
        data = resp.json()
        models_data = data.get("models", [])
        raw_names = [m.get("name", "") for m in models_data]
        
        # 2. Identificar modelos de visión mediante filtrado estricto por nombre
        # Esto previene el listado de modelos conversacionales pesados sin soporte de entrada visual optimizada o detección (ej: gemma3/gemma4/qwen3.6)
        vision_keywords = ["llava", "moondream", "vision", "bakllava", "qwen-vl", "paligemma", "minicpm"]
        
        vision_models = []
        other_models = []
        
        for name in raw_names:
            is_vision = any(k in name.lower() for k in vision_keywords)
            if is_vision:
                vision_models.append(name)
            else:
                other_models.append(name)
        
        # Retornar SOLO modelos de visión si existen, si no, fallback a todos
        final_list = vision_models if vision_models else other_models
        print(f"[Ollama] Filtrado estricto completado: {len(vision_models)} modelos de visión de detección identificados.")
        
        return final_list, len(vision_models), None
        
    except Exception as e:
        return None, None, str(e)
