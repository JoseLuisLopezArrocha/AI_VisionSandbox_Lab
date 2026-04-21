import os
import cv2
import numpy as np
import json
import shutil
from ultralytics import YOLO, RTDETR
from hardware_manager import HardwareManager

# Directorios de modelos
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
CUSTOM_MODELS_DIR = os.path.join(MODELS_DIR, "custom")

# Escalas predefinidas para hints si se detectan en el nombre
SCALE_MAP = {
    "n": "Nano — Ultrarrápido",
    "s": "Small — Equilibrado",
    "m": "Medium — Avanzado",
    "l": "Large — Alta precisión",
    "x": "eXtra Large — Máxima precisión"
}


class ObjectDetector:
    """
    Detector de objetos dinámico con soporte para múltiples familias de modelos (YOLO, RT-DETR).
    
    Gestiona la carga en caliente de modelos, la detección acumulativa (modelo base + modelos custom)
    y el filtrado de resultados mediante zonas poligonales y umbrales de confianza.
    """

    def __init__(self, initial_family=None, initial_alias=None):
        """
        Inicializa el detector escaneando el directorio de modelos y cargando la configuración inicial.
        
        Args:
            initial_family: Nombre de la carpeta de la familia (ej: 'YOLOv11').
            initial_alias: Alias del modelo específico (ej: 'YOL 01').
        """
        self.current_hash = None
        self.model = None
        self.active_name = None
        self.custom_models = []
        self.hardware_diag = HardwareManager.get_diagnostics()
        print(f"[Detector] Hardware detectado: {self.hardware_diag['gpu_name']} | Backend: {self.hardware_diag['best_backend']}")
        
        # Estructura: { "YOLOv11": { "aliases": { "YOL 01": "path/yolo11n.pt", ... }, "metadata": {...} }, ... }
        self.architectures = {}
        self.scan_models()
        
        # Nota: Ya no cargamos un modelo automáticamente aquí para evitar picos de CPU.
        # Es la aplicación principal (main.py) quien decide qué cargar al inicio asíncronamente.
        self._load_custom_models()

    def scan_models(self):
        """Escanea MODELS_DIR buscando subcarpetas (arquitecturas) y modelos .pt."""
        self.architectures = {}
        if not os.path.exists(MODELS_DIR):
            return

        for entry in sorted(os.listdir(MODELS_DIR)):
            dir_path = os.path.join(MODELS_DIR, entry)
            if os.path.isdir(dir_path) and entry != "custom":
                # Buscar metadata si existe
                metadata = {"is_coco": True, "classes": None}
                meta_path = os.path.join(dir_path, "metadata.json")
                if os.path.exists(meta_path):
                    try:
                        with open(meta_path, "r", encoding="utf-8") as f:
                            metadata = json.load(f)
                    except Exception as e:
                        print(f"Error cargando metadata para {entry}: {e}")

                # Es una familia de modelos (ej: YOLOv11)
                family_models = []
                for f in os.listdir(dir_path):
                    if f.endswith(".pt"):
                        f_path = os.path.join(dir_path, f)
                        family_models.append({
                            "name": f,
                            "path": f_path,
                            "size": os.path.getsize(f_path)
                        })
                
                if family_models:
                    # Ordenar por tamaño (más ligero primero)
                    family_models.sort(key=lambda x: x["size"])
                    
                    # Generar aliases: 3 primeras letras + 01, 02...
                    prefix = entry[:3].upper()
                    aliases = {}
                    for i, m in enumerate(family_models, 1):
                        alias = f"{prefix} {i:02d}"
                        
                        # PRIORIDAD: Buscar versión optimizada OpenVINO si estamos en Intel
                        # Ultralytics exporta a una carpeta [nombre]_openvino_model
                        base_name = os.path.splitext(m["name"])[0]
                        ov_path = os.path.join(dir_path, f"{base_name}_openvino_model")
                        
                        if os.path.exists(ov_path) and "openvino" in self.hardware_diag["best_backend"]:
                            aliases[alias] = ov_path
                            print(f"[Detector] Usando versión OpenVINO optimizada para {alias}")
                        else:
                            aliases[alias] = m["path"]
                    
                    self.architectures[entry] = {
                        "aliases": aliases,
                        "metadata": metadata
                    }

    def _load_custom_models(self):
        """Escanea models/custom/ y carga todos los .pt que encuentre."""
        self.custom_models = []
        if not os.path.exists(CUSTOM_MODELS_DIR):
            return
        for f in sorted(os.listdir(CUSTOM_MODELS_DIR)):
            if f.endswith(".pt"):
                path = os.path.join(CUSTOM_MODELS_DIR, f)
                try:
                    m = YOLO(path)
                    # Warm-up para modelos custom
                    dummy = np.zeros((64, 64, 3), dtype=np.uint8)
                    m(dummy, verbose=False)
                    
                    self.custom_models.append((m, f))
                    print(f"Modelo custom cargado y listo (warm-up OK): {f}") 
                except Exception as e:
                    print(f"Error cargando modelo custom {f}: {e}")

    def reload_custom_models(self):
        """Recarga los modelos custom desde disco."""
        self.scan_models() # Refrescar arquitecturas también
        self._load_custom_models()
        return len(self.custom_models)

    def get_class_names(self):
        """Devuelve un dict {id: nombre} con todas las clases, respetando metadata."""
        names = {}
        
        # 1. Intentar obtener de la arquitectura/familia actual (Metadata)
        if self.current_family and self.current_family in self.architectures:
            meta = self.architectures[self.current_family].get("metadata")
            if meta and not meta.get("is_coco", True) and meta.get("classes"):
                try:
                    names = {int(i): str(name) for i, name in enumerate(meta["classes"])}
                except: pass
        
        # 2. Si no hay names, intentar extraer del modelo cargado
        if not names and self.model:
            raw_names = getattr(self.model, 'names', None)
            if raw_names:
                try:
                    if isinstance(raw_names, dict):
                        names = {int(k): str(v) for k, v in raw_names.items()}
                    elif isinstance(raw_names, (list, tuple)):
                        names = {int(i): str(v) for i, v in enumerate(raw_names)}
                except:
                    pass
        
        # 3. Fallback desesperado: Si sigue vacío, usar COCO (habitual en estos modelos)
        if not names:
            # Fallback a 80 clases COCO comunes si no detectamos nada
            print(f"[DEBUG] get_class_names: Lista vacía, aplicando fallback COCO.")
            names = {i: f"Clase {i}" for i in range(80)}

        # 4. Añadir modelos custom (offset 1000+)
        offset = 1000
        for cm, _ in self.custom_models:
            c_names = getattr(cm, 'names', None)
            if c_names:
                try:
                    items = c_names.items() if isinstance(c_names, dict) else enumerate(c_names)
                    for cid, cname in items:
                        names[int(offset + cid)] = f"{str(cname)} (custom)"
                except: pass
            offset += 100
        return names

    def get_hint_for_model(self, family, alias):
        """Intenta adivinar la escala para el hint basado en el nombre del archivo."""
        path = self.architectures.get(family, {}).get("aliases", {}).get(alias)
        if not path:
            return "Modelo personalizado"
        
        filename = os.path.basename(path).lower()
        # Buscar patrones comunes como 'yolo11n', 'rtdetr-l', etc.
        for char, hint in SCALE_MAP.items():
            # Buscamos el carácter de escala precedido de nada o guion, y seguido de .pt o nada
            # Ejemplos: n.pt, -n.pt, yolo11n, etc.
            if f"{char}.pt" in filename or f"-{char}" in filename or (char in filename and len(filename) < 15):
                # Caso especial para evitar falsos positivos si el nombre es largo
                # pero YOLO suele tener n,s,m,l,x al final del nombre base
                name_no_ext = os.path.splitext(filename)[0]
                if name_no_ext.endswith(char):
                    return hint
        
        return "Escala no identificada"

    def change_model(self, family, alias):
        """Cambia el modelo activo usando la familia y el alias generado."""
        if family not in self.architectures or alias not in self.architectures[family]["aliases"]:
            return None

        model_path = self.architectures[family]["aliases"][alias]
        new_hash = f"{family}_{alias}_{model_path}"
        
        if new_hash == self.current_hash:
            return None

        print(f"Cargando modelo dinámico: {model_path}")
        target_name = os.path.basename(model_path)

        try:
            # Determinar tipo de cargador por familia o nombre de archivo
            is_openvino = os.path.isdir(model_path)
            
            if "rtdetr" in family.lower() or "rtdetr" in target_name.lower():
                self.model = RTDETR(model_path)
            else:
                self.model = YOLO(model_path, task='detect') if is_openvino else YOLO(model_path)

            # Warm-up y Selección de Dispositivo
            try:
                # Si es OpenVINO, intentamos forzar el uso de la GPU Intel si está disponible
                target_device = "CPU"
                if is_openvino and self.hardware_diag["gpu_vendor"] == "Intel":
                    target_device = "GPU" # OpenVINO GPU target
                
                print(f"[Detector] Iniciando warm-up en {target_device}...")
                dummy = np.zeros((64, 64, 3), dtype=np.uint8)
                
                # Intentar inferencia inicial (warm-up)
                self.model(dummy, verbose=False, device=target_device if is_openvino else None)
                self.active_device = target_device
            except Exception as e:
                print(f"[Detector] AVISO: Falló carga en {target_device}, intentando fallback a CPU: {e}")
                try:
                    dummy = np.zeros((64, 64, 3), dtype=np.uint8)
                    self.model(dummy, verbose=False, device="CPU" if is_openvino else None)
                    self.active_device = "CPU"
                except Exception as e2:
                    print(f"[Detector] ERROR FATAL: No se puede cargar el modelo ni en CPU: {e2}")
                    self.active_device = "UNK"

            self.current_hash = new_hash
            self.active_name = target_name
            self.current_family = family
            self.is_openvino_active = is_openvino
            
            # Detectar si es un modelo World para habilitar prompts dinámicos
            self.is_world_model = "world" in family.lower() or "world" in target_name.lower()
            
            print(f"Modelo {target_name} cargado. Backend: {'OpenVINO ('+self.active_device+')' if is_openvino else 'Standard'}")
            return target_name
        except Exception as e:
            print(f"Error al cargar modelo {target_name}: {e}")
            return None

    def detect(self, frame, target_classes=None, zones=None, conf_threshold=0.35):
        """
        Ejecuta la inferencia utilizando el modelo principal y los modelos secundarios (custom).
        Implementa lógica de filtrado por clases y asignación de zonas espaciales.

        Lógica de ejecución:
        1. Prepara argumentos dinámicos (confianza, clases filtradas).
        2. Ejecuta modelo principal (limitado a clases COCO si aplica).
        3. Ejecuta modelos custom secuencialmente (clases offset +1000).
        4. Cruza cada detección con las zonas poligonales definidas.

        Returns:
            (frame_anotado, lista_de_detecciones): Imagen con cajas dibujadas y lista de diccionarios con data.
        """
        if self.model is None:
            return frame, []

        h_frame, w_frame = frame.shape[:2]

        # --- 1. Modelo principal ---
        kwargs = {"persist": True, "stream": False, "conf": conf_threshold, "verbose": False}
        
        # Si es OpenVINO, usamos el dispositivo que funcionó en el warm-up
        if getattr(self, 'is_openvino_active', False):
            kwargs["device"] = getattr(self, 'active_device', 'CPU')
            
        if target_classes is not None:
            # ... (filtrado previo)
            coco_classes = []
            for c in target_classes:
                try:
                    ic = int(c)
                    if ic < 1000: coco_classes.append(ic)
                except ValueError: continue
            
            if not coco_classes:
                kwargs["classes"] = [9999]
            else:
                kwargs["classes"] = coco_classes

        # Usamos .track() para mantener consistencia de IDs entre frames
        results = self.model.track(frame, **kwargs)

        detections = []
        # Iniciamos con una copia limpia. annotated se actualizará con plot() o dibujos manuales.
        annotated = frame.copy()

        # Consolidar nombres de clases para evitar accesos repetidos
        model_names = self.model.names

        for result in results:
            # result.plot() devuelve una copia, acumulamos sobre 'annotated'
            plot_res = result.plot()
            # Fusionar el plot del resultado con el acumulado (solo las cajas)
            # En Ultralytics es más limpio usar el frame original y dibujar nosotros o 
            # usar una sola llamada a result.plot() si solo hay un result (lo habitual)
            if len(results) == 1:
                annotated = plot_res
            
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                # El track_id es crucial para los conteos acumulados
                track_id = int(box.id[0]) if box.id is not None else None
                
                label = model_names.get(cls_id, f"Clase {cls_id}")
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

                # Asignación multizona (Modelo Principal)
                zone_indices = self._get_zones_for_point(cx, cy, w_frame, h_frame, zones)

                detections.append({
                    "label": label, "confidence": conf,
                    "class_id": cls_id, "zone_indices": zone_indices, # Antes zone_idx
                    "track_id": track_id,
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                })

        # --- 2. Modelos custom (taxi, etc.) ---
        for cm, cm_name in self.custom_models:
            try:
                cm_results = cm.track(frame, persist=True, stream=False, conf=conf_threshold, verbose=False)
                for result in cm_results:
                    for box in result.boxes:
                        cls_id = int(box.cls[0])
                        global_cls_id = 1000 + cls_id # ID único para este modelo custom
                        
                        # FILTRO: Solo procesar si no hay filtro o si la clase está en el target
                        if target_classes is not None and global_cls_id not in target_classes:
                            continue

                        conf = float(box.conf[0])
                        label = cm.names[cls_id]
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

                        # Asignación multizona
                        zone_indices = self._get_zones_for_point(cx, cy, w_frame, h_frame, zones)
                        track_id = int(box.id[0]) if box.id is not None else None

                        detections.append({
                            "label": label, "confidence": conf,
                            "class_id": global_cls_id, "zone_indices": zone_indices, 
                            "track_id": track_id,
                            "bbox": (int(x1), int(y1), int(x2), int(y2)),
                        })

                        # Dibujar manualmente la caja custom (color cyan) solo si pasó el filtro
                        cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 2)
                        tag = f"{label} {conf:.2f}"
                        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        cv2.rectangle(annotated, (int(x1), int(y1) - th - 8),
                                      (int(x1) + tw + 4, int(y1)), (255, 255, 0), -1)
                        cv2.putText(annotated, tag, (int(x1) + 2, int(y1) - 4),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            except Exception as e:
                print(f"Error en modelo custom {cm_name}: {e}")

        return annotated, detections

    def _get_zones_for_point(self, cx, cy, w_frame, h_frame, zones):
        """Devuelve una lista de todos los índices de zona donde cae el punto."""
        found_zones = []
        if not zones:
            return found_zones
        for i, zone_pts in enumerate(zones):
            poly = np.array(
                [(int(px * w_frame), int(py * h_frame)) for px, py in zone_pts],
                dtype=np.int32
            )
            if cv2.pointPolygonTest(poly, (cx, cy), False) >= 0:
                found_zones.append(i)
        return found_zones
    def set_world_prompt(self, prompt):
        """Reconfigura las clases del modelo YOLO-World en tiempo real."""
        if not self.is_world_model or self.model is None:
            return False
            
        try:
            # Limpiar y separar por comas
            if not prompt:
                classes = ["person"] # Default
            else:
                classes = [c.strip() for c in prompt.split(",") if c.strip()]
            
            print(f"Reconfigurando YOLO-World con: {classes}")
            self.model.set_classes(classes)
            return True
        except Exception as e:
            print(f"Error al reconfigurar clases World: {e}")
            return False

    def export_current_to_openvino(self):
        """Exporta el modelo actual al formato OpenVINO para aceleración Intel."""
        if self.model is None or self.is_openvino_active:
            return False
            
        try:
            print(f"[Detector] Iniciando optimización OpenVINO para {self.active_name}...")
            # Exportar (Esto crea una carpeta al lado del .pt)
            export_path = self.model.export(format='openvino')
            print(f"[Detector] Modelo optimizado con éxito en: {export_path}")
            
            # Refrescar arquitecturas para que reconozca el nuevo modelo
            self.scan_models()
            return True
        except Exception as e:
            print(f"[Detector] Error durante la exportación OpenVINO: {e}")
            return False
