import requests
import json
import os
import threading

# Ruta de caché persistente
CACHE_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "utils", "ollama_cache.json")

def get_ollama_models_with_vision(url):
    """
    Obtiene la lista de modelos de Ollama y verifica cuáles tienen capacidades de visión
    haciendo peticiones asíncronas a /api/show y usando una caché local.
    """
    try:
        # 1. Obtener lista básica de modelos
        resp = requests.get(f"{url.rstrip('/')}/api/tags", timeout=5)
        if resp.status_code != 200:
            return None, None, f"Error HTTP {resp.status_code}"
            
        data = resp.json()
        models_data = data.get("models", [])
        raw_names = [m.get("name", "") for m in models_data]
        
        # 2. Cargar caché
        cache = {}
        if os.path.exists(CACHE_FILE):
            try:
                with open(CACHE_FILE, "r") as f:
                    cache = json.load(f)
            except:
                pass
            
        # 3. Identificar modelos de visión
        vision_models = []
        other_models = []
        new_cache_entries = False
        
        print(f"[Ollama] Analizando capacidades de {len(raw_names)} modelos...")
        
        for name in raw_names:
            cache_key = f"{url}_{name}"
            
            if cache_key in cache:
                is_vision = cache[cache_key]
            else:
                # Comprobación por nombre
                is_vision = any(k in name.lower() for k in ["llava", "moondream", "vision", "bakllava", "qwen-vl"])
                
                try:
                    show_resp = requests.post(f"{url.rstrip('/')}/api/show", json={"model": name}, timeout=2)
                    if show_resp.status_code == 200:
                        show_data = show_resp.json()
                        capabilities = show_data.get("capabilities", [])
                        # Marcamos como visión si Ollama lo confirma
                        is_vision = "vision" in capabilities or is_vision
                except Exception as e:
                    print(f"[Ollama] Error consultando {name}: {e}")
                
                cache[cache_key] = is_vision
                new_cache_entries = True
            
            if is_vision:
                vision_models.append(name)
            else:
                other_models.append(name)
        
        # 4. Guardar caché si ha cambiado
        if new_cache_entries:
            try:
                with open(CACHE_FILE, "w") as f:
                    json.dump(cache, f)
            except:
                pass
                
        # Retornar SOLO modelos de visión si existen, si no, fallback a todos
        final_list = vision_models if vision_models else other_models
        print(f"[Ollama] Filtrado completado: {len(vision_models)} modelos de visión detectados.")
        
        return final_list, len(vision_models), None
        
    except Exception as e:
        return None, None, str(e)
