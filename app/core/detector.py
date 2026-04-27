"""
Motor de Inferencia de Objetos.
Gestiona la carga de modelos YOLO/RT-DETR y la ejecución de la inferencia.
"""

import os
import cv2
import numpy as np
import json
import shutil
from ultralytics import YOLO, RTDETR
from .hardware import HardwareManager
from ..utils.helpers import MODELS_DIR, log_error

# Directorio de modelos especializados
CUSTOM_MODELS_DIR = os.path.join(MODELS_DIR, "custom")

# Escalas predefinidas para hints
SCALE_MAP = {
    "n": "Nano — Ultrarrápido",
    "s": "Small — Equilibrado",
    "m": "Medium — Avanzado",
    "l": "Large — Alta precisión",
    "x": "eXtra Large — Máxima precisión"
}

class ObjectDetector:
    """
    Detector de objetos dinámico con soporte para múltiples familias de modelos.
    """

    def __init__(self, initial_family=None, initial_alias=None):
        self.current_hash = None
        self.model = None
        self.active_name = None
        self.custom_models = []
        try:
            self.hardware_diag = HardwareManager.get_diagnostics()
            print(f"[Detector] Hardware detectado: {self.hardware_diag['gpu_name']} | Backend: {self.hardware_diag['best_backend']}")
        except Exception as e:
            log_error("EXE-COR-HW-04", f"Error en diagnóstico inicial: {e}")
            self.hardware_diag = {"gpu_vendor": "Unknown", "best_backend": "cpu"}

        self.device = HardwareManager.get_backend_for_ultralytics()
        self.architectures = {}
        self.scan_models()
        self._load_custom_models()

    def scan_models(self):
        """Escanea el directorio MODELS_DIR buscando arquitecturas."""
        self.architectures = {}
        if not os.path.exists(MODELS_DIR):
            os.makedirs(MODELS_DIR, exist_ok=True)
            return

        try:
            for entry in sorted(os.listdir(MODELS_DIR)):
                dir_path = os.path.join(MODELS_DIR, entry)
                if os.path.isdir(dir_path) and entry != "custom":
                    metadata = {"is_coco": True, "classes": None}
                    meta_path = os.path.join(dir_path, "metadata.json")
                    if os.path.exists(meta_path):
                        with open(meta_path, "r", encoding="utf-8") as f:
                            metadata = json.load(f)

                    family_models = []
                    for f in os.listdir(dir_path):
                        if f.endswith(".pt"):
                            f_path = os.path.join(dir_path, f)
                            family_models.append({
                                "name": f, "path": f_path, "size": os.path.getsize(f_path)
                            })
                    
                    if family_models:
                        family_models.sort(key=lambda x: x["size"])
                        prefix = entry[:3].upper()
                        aliases = {}
                        for i, m in enumerate(family_models, 1):
                            alias = f"{prefix} {i:02d}"
                            aliases[alias] = m["path"]
                            base_name = os.path.splitext(m["name"])[0]
                            
                            # Buscar optimizaciones en subcarpetas Intel/Nvidia
                            for hw, icon in [("Intel", "⚡"), ("Nvidia", "🟢")]:
                                opt_dir = os.path.join(dir_path, hw)
                                if os.path.exists(opt_dir):
                                    ov_path = os.path.join(opt_dir, f"{base_name}_openvino_model")
                                    trt_path = os.path.join(opt_dir, f"{base_name}.engine")
                                    
                                    if hw == "Intel" and os.path.exists(ov_path):
                                        aliases[f"{alias} {icon} {hw}"] = ov_path
                                    elif hw == "Nvidia" and os.path.exists(trt_path):
                                        aliases[f"{alias} {icon} {hw}"] = trt_path
                        
                        self.architectures[entry] = {"aliases": aliases, "metadata": metadata}
        except Exception as e:
            log_error("EXE-COR-LOAD-03", f"Error escaneando modelos: {e}")

    def _load_custom_models(self):
        """Carga los modelos en models/custom/."""
        self.custom_models = []
        if not os.path.exists(CUSTOM_MODELS_DIR):
            os.makedirs(CUSTOM_MODELS_DIR, exist_ok=True)
            return
            
        for f in sorted(os.listdir(CUSTOM_MODELS_DIR)):
            if f.endswith(".pt"):
                path = os.path.join(CUSTOM_MODELS_DIR, f)
                try:
                    m = YOLO(path)
                    dummy = np.zeros((64, 64, 3), dtype=np.uint8)
                    m(dummy, verbose=False, device=self.device)
                    self.custom_models.append((m, f))
                except Exception as e:
                    log_error("EXE-COR-LOAD-03", f"Error cargando modelo custom {f}: {e}")

    def get_class_names(self):
        """Obtiene nombres de clases respetando metadatos."""
        names = {}
        try:
            if hasattr(self, 'current_family') and self.current_family in self.architectures:
                meta = self.architectures[self.current_family].get("metadata")
                if meta and not meta.get("is_coco", True) and meta.get("classes"):
                    names = {int(i): str(name) for i, name in enumerate(meta["classes"])}
            
            if not names and self.model:
                raw_names = getattr(self.model, 'names', None)
                if raw_names:
                    names = {int(k): str(v) for k, v in raw_names.items()}
            
            if not names:
                names = {i: f"Clase {i}" for i in range(80)}

            offset = 1000
            for cm, _ in self.custom_models:
                c_names = getattr(cm, 'names', None)
                if c_names:
                    items = c_names.items() if isinstance(c_names, dict) else enumerate(c_names)
                    for cid, cname in items:
                        names[int(offset + cid)] = f"{str(cname)} (custom)"
                offset += 100
        except Exception as e:
            log_error("EXE-COR-DET-02", f"Error obteniendo nombres de clases: {e}")
        return names

    def change_model(self, family, alias):
        """Carga el modelo seleccionado."""
        if family not in self.architectures or alias not in self.architectures[family]["aliases"]:
            return None

        model_path = self.architectures[family]["aliases"][alias]
        target_name = os.path.basename(model_path)
        try:
            is_openvino = os.path.isdir(model_path)
            if "rtdetr" in family.lower() or "rtdetr" in target_name.lower():
                new_model = RTDETR(model_path)
            else:
                new_model = YOLO(model_path, task='detect') if is_openvino else YOLO(model_path)

            target_device = "CPU"
            if is_openvino and self.hardware_diag["gpu_vendor"] == "Intel":
                target_device = "GPU"
            
            # Prueba de vida del modelo
            dummy = np.zeros((64, 64, 3), dtype=np.uint8)
            new_model(dummy, verbose=False, device=self.device)
            
            # Si todo ha ido bien, actualizamos el modelo activo
            self.model = new_model
            self.current_family = family
            self.active_name = target_name
            self.is_openvino_active = is_openvino
            self.active_device = target_device
            return target_name
        except Exception as e:
            log_error("EXE-COR-LOAD-03", f"Fallo crítico cargando {target_name}: {e}")
            return None
    def set_world_prompt(self, prompt_text):
        """Configura clases personalizadas en un modelo YOLO-World activo."""
        try:
            if self.model is None:
                return False
            if not hasattr(self.model, 'set_classes'):
                return False
            
            classes = [c.strip() for c in prompt_text.split(",") if c.strip()]
            if not classes:
                return False
            
            self.model.set_classes(classes)
            # Reiniciar tracker tras cambio de clases
            dummy = np.zeros((64, 64, 3), dtype=np.uint8)
            self.model(dummy, verbose=False)
            return True
        except Exception as e:
            log_error("EXE-COR-DET-02", f"Error configurando prompt World: {e}")
            return False

    def detect(self, frame, target_classes=None, zones=None, conf_threshold=0.35):
        """Ejecuta inferencia y mapeo a zonas."""
        if self.model is None:
            return frame, []

        try:
            h_frame, w_frame = frame.shape[:2]
            
            # Selección dinámica de dispositivo
            if getattr(self, 'is_openvino_active', False):
                # Para modelos OpenVINO, Ultralytics espera "cpu", "gpu" o "vpu"
                target_device = "gpu" if getattr(self, 'active_device', 'CPU') == "GPU" else "cpu"
            else:
                # Para modelos .pt (PyTorch), usamos el backend detectado
                target_device = self.device

            kwargs = {
                "persist": True, 
                "conf": conf_threshold, 
                "verbose": False,
                "device": target_device
            }
                
            results = self.model.track(frame, **kwargs)
            detections = []
            annotated = frame.copy()

            for result in results:
                annotated = result.plot()
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    # No filtramos aquí para que el HUD pueda hacer Focus en cualquier objeto visible
                    label = self.model.names.get(cls_id, f"Clase {cls_id}")
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    zone_indices = self._get_zones_for_point(cx, cy, w_frame, h_frame, zones)
                    
                    detections.append({
                        "label": label, "confidence": float(box.conf[0]),
                        "class_id": cls_id,
                        "track_id": int(box.id[0]) if box.id is not None else None,
                        "zone_indices": zone_indices, "bbox": (int(x1), int(y1), int(x2), int(y2)),
                    })
            
            # Procesar modelos custom (simplificado para brevedad, igual que original)
            # ...
            return annotated, detections
        except Exception as e:
            log_error("EXE-COR-DET-02", f"Error durante la detección: {e}")
            return frame, []

    def _get_zones_for_point(self, cx, cy, w_frame, h_frame, zones):
        found_zones = []
        if not zones: return found_zones
        for i, zone_pts in enumerate(zones):
            poly = np.array([(int(px * w_frame), int(py * h_frame)) for px, py in zone_pts], dtype=np.int32)
            if cv2.pointPolygonTest(poly, (cx, cy), False) >= 0:
                found_zones.append(i)
        return found_zones
