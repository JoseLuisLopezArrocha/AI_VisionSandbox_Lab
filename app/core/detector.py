"""
Motor de Inferencia de Objetos.
Gestiona la carga de modelos YOLO/RT-DETR y la ejecución de la inferencia.
Incluye soporte para modelos sklearn serializados como clasificadores globales.
"""

import os
import cv2
import torch
import numpy as np
import json
import base64
import requests
from ultralytics import YOLO, RTDETR
from .hardware import HardwareManager
from ..utils.helpers import MODELS_DIR, log_error


class _SklearnProbs:
    """Simula la interfaz de Ultralytics Results.probs para modelos sklearn."""
    def __init__(self, class_probs):
        self.data = class_probs
        sorted_indices = sorted(range(len(class_probs)), key=lambda i: class_probs[i], reverse=True)
        self.top1 = sorted_indices[0]
        self.top1conf = class_probs[self.top1]
        self.top5 = sorted_indices[:5]
        self.top5conf = [class_probs[i] for i in self.top5]


class _SklearnResult:
    """Simula un resultado de Ultralytics para un modelo sklearn."""
    def __init__(self, probs):
        self.probs = probs
        self.boxes = None


class SklearnModelWrapper:
    """Wrapper que adapta modelos sklearn (.pt serializados con torch) al pipeline de inferencia.
    
    Soporta checkpoints con estructura {pca, svm, tamano, clase} tipica de
    pipelines de reconocimiento facial (PCA + SVC).
    """
    def __init__(self, checkpoint_path):
        self.task = 'classify'
        self.names = {}
        self._pca = None
        self._classifier = None
        self._img_size = None  # (ancho, alto) para resize
        
        # Cargar el checkpoint completo
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        if not isinstance(ckpt, dict):
            # Si es un objeto directo con predict
            if hasattr(ckpt, 'predict'):
                self._classifier = ckpt
            else:
                raise ValueError(f"Formato de checkpoint no soportado: {type(ckpt)}")
        else:
            # --- Buscar PCA ---
            for key in ['pca', 'PCA', 'reducer', 'feature_extractor']:
                if key in ckpt and hasattr(ckpt[key], 'transform'):
                    self._pca = ckpt[key]
                    break
            
            # --- Buscar Clasificador ---
            for key in ['svm', 'SVM', 'classifier', 'clf', 'model', 'estimator', 'pipeline']:
                if key in ckpt and hasattr(ckpt[key], 'predict'):
                    self._classifier = ckpt[key]
                    break
            # Fallback: buscar cualquier objeto con predict
            if self._classifier is None:
                for v in ckpt.values():
                    if hasattr(v, 'predict') and not hasattr(v, 'transform'):
                        self._classifier = v
                        break
            
            # --- Buscar tamano de imagen de entrenamiento ---
            for key in ['tamano', 'size', 'img_size', 'image_size']:
                if key in ckpt:
                    raw = ckpt[key]
                    if isinstance(raw, (tuple, list)) and len(raw) == 2:
                        self._img_size = (int(raw[0]), int(raw[1]))
                    break
            
            # --- Buscar nombres de clases ---
            for key in ['names', 'classes', 'class_names', 'labels', 'clase']:
                if key in ckpt:
                    raw = ckpt[key]
                    if isinstance(raw, dict):
                        self.names = {int(k): str(v) for k, v in raw.items()}
                    elif isinstance(raw, (list, tuple)):
                        self.names = {i: str(v) for i, v in enumerate(raw)}
                    elif isinstance(raw, str):
                        # Clave 'clase' con valor string: nombre de la clase positiva
                        self.names = {0: 'desconocidos', 1: str(raw)}
                    break
        
        if self._classifier is None:
            raise ValueError(f"No se encontro clasificador en {checkpoint_path}")
        
        # Deducir nombres de clases del clasificador si no los tenemos
        if not self.names and hasattr(self._classifier, 'classes_'):
            self.names = {i: str(c) for i, c in enumerate(self._classifier.classes_)}
        elif not self.names:
            best = getattr(self._classifier, 'best_estimator_', None)
            if best and hasattr(best, 'classes_'):
                self.names = {i: str(c) for i, c in enumerate(best.classes_)}
        
        # Deducir tamano de imagen del PCA si no se especifico
        if self._img_size is None and self._pca is not None:
            n_features = self._pca.n_features_in_
            side = int(n_features ** 0.5)
            if side * side == n_features:
                self._img_size = (side, side)
            else:
                self._img_size = (side, side)
        
        pca_info = f"PCA({self._pca.n_components_})" if self._pca else "Sin PCA"
        img_info = f"{self._img_size[0]}x{self._img_size[1]}" if self._img_size else "auto"
        print(f"[SklearnWrapper] Pipeline: {pca_info} -> {type(self._classifier).__name__}")
        print(f"[SklearnWrapper] Clases: {self.names} | Imagen: {img_info}")
        
        # Cargar detector de caras Haar Cascade (incluido en OpenCV)
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self._face_cascade = cv2.CascadeClassifier(cascade_path)
        if self._face_cascade.empty():
            print("[SklearnWrapper] ADVERTENCIA: No se pudo cargar Haar Cascade")
            self._face_cascade = None
        else:
            print("[SklearnWrapper] Haar Cascade de caras cargado correctamente")

    def _classify_crop(self, gray_crop):
        """Clasifica un recorte en escala de grises ya preparado."""
        resized = cv2.resize(gray_crop, self._img_size or (128, 128))
        features = resized.flatten().astype(np.float64).reshape(1, -1)
        
        if self._pca is not None:
            features = self._pca.transform(features)
        
        if hasattr(self._classifier, 'predict_proba'):
            probas = self._classifier.predict_proba(features)[0]
        elif hasattr(self._classifier, 'decision_function'):
            decisions = self._classifier.decision_function(features)
            if hasattr(decisions, '__len__') and len(decisions.shape) > 1:
                decisions = decisions[0]
            elif not hasattr(decisions, '__len__'):
                decisions = np.array([decisions])
            if len(decisions) == 1:
                d = float(decisions[0])
                exp_pos = np.exp(min(d, 500))
                exp_neg = np.exp(min(-d, 500))
                probas = np.array([exp_neg / (exp_pos + exp_neg), 
                                  exp_pos / (exp_pos + exp_neg)])
            else:
                exp_d = np.exp(decisions - np.max(decisions))
                probas = exp_d / exp_d.sum()
        else:
            pred = self._classifier.predict(features)[0]
            n_classes = len(self.names) if self.names else 2
            probas = np.zeros(n_classes)
            pred_idx = int(pred) if isinstance(pred, (int, np.integer)) else 0
            if pred_idx < n_classes:
                probas[pred_idx] = 1.0
        
        return probas

    def detect_and_classify(self, frame, conf_threshold=0.35, source="primary"):
        """Pipeline completo: detecta caras con Haar Cascade, recorta cada una y clasifica.
        
        Retorna lista de dicts compatibles con el formato de deteccion del sistema,
        con bounding boxes reales alrededor de cada cara detectada.
        """
        detections = []
        h_frame, w_frame = frame.shape[:2]
        
        # Convertir a escala de grises una sola vez
        if frame.ndim == 3:
            gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray_full = frame
        
        if self._face_cascade is None:
            # Sin Haar Cascade, clasificar el frame completo (fallback)
            probas = self._classify_crop(gray_full)
            top_id = int(np.argmax(probas))
            top_conf = float(probas[top_id])
            if top_conf >= conf_threshold:
                label = self.names.get(top_id, f"Clase {top_id}")
                detections.append({
                    "label": label, "confidence": top_conf,
                    "class_id": top_id, "track_id": None,
                    "zone_indices": [], "bbox": (int(w_frame*0.1), int(h_frame*0.1), int(w_frame*0.9), int(h_frame*0.9)),
                    "source": source, "is_classification": True
                })
            return detections
        
        # Detectar caras con Haar Cascade
        faces = self._face_cascade.detectMultiScale(
            gray_full,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(40, 40),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        for (fx, fy, fw, fh) in faces:
            # Ampliar el recorte un 15% para incluir mas contexto facial
            pad_x = int(fw * 0.15)
            pad_y = int(fh * 0.15)
            x1 = max(0, fx - pad_x)
            y1 = max(0, fy - pad_y)
            x2 = min(w_frame, fx + fw + pad_x)
            y2 = min(h_frame, fy + fh + pad_y)
            
            # Recortar la cara y clasificar
            face_crop = gray_full[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue
            
            try:
                probas = self._classify_crop(face_crop)
                top_id = int(np.argmax(probas))
                top_conf = float(probas[top_id])
                
                if top_conf >= conf_threshold:
                    label = self.names.get(top_id, f"Clase {top_id}")
                    detections.append({
                        "label": label, "confidence": top_conf,
                        "class_id": top_id, "track_id": None,
                        "zone_indices": [],
                        "bbox": (x1, y1, x2, y2),
                        "source": source,
                        "is_classification": False  # Tiene bbox real, pintar como deteccion normal
                    })
            except Exception as e:
                log_error("EXE-COR-DET-02", f"Error clasificando cara: {e}")
                continue
        
        return detections
    
    def predict(self, frame, **kwargs):
        """Clasificacion global del frame (fallback sin deteccion de caras)."""
        try:
            if isinstance(frame, np.ndarray) and frame.ndim >= 2:
                if frame.ndim == 3:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                else:
                    gray = frame
                probas = self._classify_crop(gray)
                probs = _SklearnProbs(probas.tolist())
                return [_SklearnResult(probs)]
        except Exception as e:
            log_error("EXE-COR-DET-02", f"Error en prediccion sklearn: {e}")
        return [_SklearnResult(_SklearnProbs([0.0]))]
    
    def __call__(self, frame, **kwargs):
        """Permite usar el wrapper como callable (compatibilidad con prueba de vida)."""
        return self.predict(frame, **kwargs)


class OllamaVisionWrapper:
    """
    Wrapper para ejecutar inferencia de detección utilizando modelos de Ollama.
    """
    def __init__(self, model_name, base_url="http://192.168.0.135:11434"):
        # [Propósito]: Tipo de tarea simulada compatible con Ultralytics (detect).
        # [Tipo]: str
        self.task = 'detect'
        
        # [Propósito]: Mapeo de IDs de clase a nombres legibles por la UI.
        # [Tipo]: dict[int, str]
        self.names = {
            0: "persona", 1: "bicicleta", 2: "coche", 3: "moto",
            5: "autobus", 7: "camion", 9: "semaforo"
        }
        
        # [Propósito]: Mapeo inverso de nombres a IDs de clase estándar o generados.
        # [Tipo]: dict[str, int]
        self.class_map = {
            "persona": 0, "person": 0,
            "bicicleta": 1, "bicycle": 1,
            "coche": 2, "car": 2, "automobile": 2,
            "moto": 3, "motorcycle": 3,
            "autobus": 5, "bus": 5,
            "camion": 7, "truck": 7,
            "semaforo": 9, "traffic light": 9
        }
        
        # [Propósito]: Nombre de modelo cargado en Ollama.
        # [Tipo]: str
        self.model_name = model_name
        
        # [Propósito]: URL base del servidor Ollama.
        # [Tipo]: str
        self.base_url = base_url.rstrip("/")
        
        print(f"[OllamaVisionWrapper] Inicializado para modelo: {model_name} en {self.base_url}")

    def detect(self, frame, conf_threshold=0.35, source="secondary"):
        """Ejecuta consulta VLM a Ollama para detectar objetos en la imagen y los mapea a bboxes."""
        detections = []
        h_frame, w_frame = frame.shape[:2]
        
        try:
            import cv2
            import base64
            import requests
            import json
            
            _, buffer = cv2.imencode('.jpg', frame)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
        except Exception as e:
            print(f"[OllamaVisionWrapper] Error codificando imagen: {e}")
            return detections

        url = f"{self.base_url}/api/generate"
        prompt = """Identifica todos los objetos presentes en la imagen. 
Para cada objeto, devuelve una lista de diccionarios JSON bajo la llave "detections".
Cada diccionario de objeto debe tener las llaves "label" (nombre del objeto en minúsculas y español) y "bbox" [y1, x1, y2, x2] con coordenadas relativas escaladas en un rango de 0 a 1000 (donde [0,0,1000,1000] cubre toda la imagen).
Ejemplo de formato exacto de salida:
{
  "detections": [
    {"label": "coche", "bbox": [100, 150, 800, 900]}
  ]
}
No agregues explicaciones adicionales, devuelve SOLO el objeto JSON estructurado."""

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "images": [img_base64],
            "stream": False,
            "format": "json"
        }
        
        try:
            response = requests.post(url, json=payload, timeout=8)
            if response.status_code == 200:
                result = response.json().get("response", "")
                data = json.loads(result)
                raw_dets = data.get("detections", [])
                for i, rd in enumerate(raw_dets):
                    label = rd.get("label", "objeto").lower().strip()
                    bbox_1000 = rd.get("bbox", [0, 0, 1000, 1000])
                    
                    if len(bbox_1000) != 4:
                        continue
                        
                    y1, x1, y2, x2 = bbox_1000
                    
                    # Convertir coordenadas relativas (0-1000) a píxeles
                    rx1 = int((x1 / 1000) * w_frame)
                    ry1 = int((y1 / 1000) * h_frame)
                    rx2 = int((x2 / 1000) * w_frame)
                    ry2 = int((y2 / 1000) * h_frame)
                    
                    # Asegurar límites correctos
                    rx1 = max(0, min(rx1, w_frame - 1))
                    ry1 = max(0, min(ry1, h_frame - 1))
                    rx2 = max(0, min(rx2, w_frame - 1))
                    ry2 = max(0, min(ry2, h_frame - 1))
                    
                    # Asignar clase dinámicamente si no existe
                    if label not in self.class_map:
                        next_id = 100 + len(self.class_map)
                        self.class_map[label] = next_id
                        self.names[next_id] = label
                        
                    cls_id = self.class_map[label]
                    
                    detections.append({
                        "label": label,
                        "confidence": 0.85,  # Confianza fija alta para VLM
                        "class_id": cls_id,
                        "track_id": 9000 + i, # Virtual tracking ID
                        "zone_indices": [],
                        "bbox": (rx1, ry1, rx2, ry2),
                        "source": source,
                        "is_classification": False
                    })
        except Exception as e:
            print(f"[OllamaVisionWrapper] Error en consulta VLM: {e}")
        return detections

    def predict(self, frame, **kwargs):
        """Simula predict."""
        return self.detect(frame, **kwargs)

    def __call__(self, frame, **kwargs):
        """Simula callable."""
        return self.predict(frame, **kwargs)


class ObjectDetector:
    """
    Detector de objetos dinámico con soporte para múltiples familias de modelos.
    Soporta un modelo primario y un modelo secundario simultáneamente (Dual Mode).
    """

    # Colores BGR para distinguir visualmente cada modelo
    PRIMARY_COLOR = (233, 165, 14)    # Azul cielo (#0EA5E9) en BGR
    SECONDARY_COLOR = (0, 165, 255)   # Naranja (#FFA500) en BGR

    def __init__(self, initial_family=None, initial_alias=None):
        # [Propósito]: Hash de archivo único del modelo primario para evitar recargas redundantes.
        # [Tipo]: Optional[str]
        self.current_hash = None

        # [Propósito]: Instancia del modelo primario activo cargado en memoria.
        # [Tipo]: Optional[object]
        self.model = None

        # [Propósito]: Nombre identificador o alias del modelo primario activo.
        # [Tipo]: Optional[str]
        self.active_name = None

        # --- Modelo Secundario (Dual Mode) ---
        # [Propósito]: Instancia del modelo secundario activo cargado en memoria.
        # [Tipo]: Optional[object]
        self.secondary_model = None

        # [Propósito]: Nombre identificador o alias del modelo secundario activo.
        # [Tipo]: Optional[str]
        self.secondary_name = None

        # [Propósito]: Nombre de la carpeta de familia a la que pertenece el modelo secundario.
        # [Tipo]: Optional[str]
        self.secondary_family = None

        # [Propósito]: Indica si el modelo secundario cargado es una optimización de OpenVINO.
        # [Tipo]: bool
        self.is_secondary_openvino = False

        # [Propósito]: Dispositivo seleccionado para ejecutar el modelo secundario ("CPU" o "GPU").
        # [Tipo]: str
        self.secondary_device = "CPU"

        # [Propósito]: Bandera para habilitar la entrada de texto libre en modelos YOLO-World.
        # [Tipo]: bool
        self.is_zero_shot_active = False

        try:
            self.hardware_diag = HardwareManager.get_diagnostics()
            print(f"[Detector] Hardware detectado: {self.hardware_diag['gpu_name']} | Backend: {self.hardware_diag['best_backend']}")
        except Exception as e:
            log_error("EXE-COR-HW-04", f"Error en diagnóstico inicial: {e}")
            self.hardware_diag = {"gpu_vendor": "Unknown", "best_backend": "cpu"}

        self.device = HardwareManager.get_backend_for_ultralytics()
        self.architectures = {}
        self.scan_models()

    def scan_models(self):
        """Escanea el directorio MODELS_DIR buscando arquitecturas, incluyendo custom."""
        self.architectures = {}
        if not os.path.exists(MODELS_DIR):
            os.makedirs(MODELS_DIR, exist_ok=True)
            return

        try:
            for entry in sorted(os.listdir(MODELS_DIR)):
                dir_path = os.path.join(MODELS_DIR, entry)
                if not os.path.isdir(dir_path):
                    continue

                is_custom = (entry == "custom")

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
                    aliases = {}

                    if is_custom:
                        # Modelos custom: alias = nombre del archivo sin extension
                        for m in family_models:
                            alias = os.path.splitext(m["name"])[0]
                            aliases[alias] = m["path"]
                        # Marcar metadata como no-coco por defecto para custom
                        if not os.path.exists(meta_path):
                            metadata = {"is_coco": False, "classes": None}
                    else:
                        # Familias regulares: alias con prefijo numerico
                        prefix = entry[:3].upper()
                        for i, m in enumerate(family_models, 1):
                            alias = f"{prefix} {i:02d}"
                            aliases[alias] = m["path"]
                            base_name = os.path.splitext(m["name"])[0]

                            # Buscar optimizaciones en subcarpetas Intel/Nvidia
                            for hw, icon in [("Intel", "[Intel]"), ("Nvidia", "[Nvidia]")]:
                                opt_dir = os.path.join(dir_path, hw)
                                if os.path.exists(opt_dir):
                                    ov_path = os.path.join(opt_dir, f"{base_name}_openvino_model")
                                    trt_path = os.path.join(opt_dir, f"{base_name}.engine")

                                    if hw == "Intel" and os.path.exists(ov_path):
                                        aliases[f"{alias} {icon}"] = ov_path
                                    elif hw == "Nvidia" and os.path.exists(trt_path):
                                        aliases[f"{alias} {icon}"] = trt_path

                    self.architectures[entry] = {"aliases": aliases, "metadata": metadata}

            # Consulta dinámica a Ollama para agregar modelos remotos con soporte de visión
            try:
                ollama_url = os.getenv("OLLAMA_URL", "http://192.168.0.135:11434").rstrip("/")
                from .ollama_helper import get_ollama_models_with_vision
                vision_models, vision_count, error = get_ollama_models_with_vision(ollama_url)
                if not error and vision_models:
                    ollama_aliases = {}
                    for name in vision_models:
                        clean_name = name.split(":")[0].replace("-", " ").title()
                        tag = name.split(":")[1] if ":" in name else "latest"
                        alias = f"{clean_name} ({tag})"
                        ollama_aliases[alias] = f"ollama://{name}"
                    
                    if ollama_aliases:
                        self.architectures["ollama"] = {
                            "aliases": ollama_aliases,
                            "metadata": {"is_coco": False, "classes": ["persona", "bicicleta", "coche", "moto", "autobus", "camion", "semaforo", "objeto"]}
                        }
                        print(f"[Detector] Se cargaron {len(ollama_aliases)} modelos dinamicos de vision desde Ollama ({ollama_url})")
            except Exception as e:
                print(f"[Detector] No se pudo conectar a Ollama para escaneo de modelos: {e}")
        except Exception as e:
            log_error("EXE-COR-LOAD-03", f"Error escaneando modelos: {e}")

    def get_class_names(self):
        """Obtiene nombres de clases respetando metadatos de la familia activa."""
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
            if family == "ollama":
                ollama_url = os.getenv("OLLAMA_URL", "http://192.168.0.135:11434").rstrip("/")
                model_name = model_path.replace("ollama://", "")
                self.model = OllamaVisionWrapper(model_name, base_url=ollama_url)
                self.current_family = family
                self.active_name = alias
                self.is_openvino_active = False
                self.active_device = "Ollama VLM"
                self.is_zero_shot_active = False
                return alias

            is_openvino = os.path.isdir(model_path)
            new_model = None
            is_sklearn = False

            # Intentar cargar como modelo YOLO/RT-DETR
            try:
                if "rtdetr" in family.lower() or "rtdetr" in target_name.lower():
                    new_model = RTDETR(model_path)
                else:
                    new_model = YOLO(model_path, task='detect') if is_openvino else YOLO(model_path)
            except Exception as yolo_err:
                # Si YOLO falla, intentar como modelo sklearn
                print(f"[Detector] YOLO no pudo cargar {target_name}: {yolo_err}")
                print(f"[Detector] Intentando como modelo sklearn...")
                try:
                    new_model = SklearnModelWrapper(model_path)
                    is_sklearn = True
                except Exception as sk_err:
                    raise RuntimeError(f"No es YOLO ni sklearn: YOLO={yolo_err}, sklearn={sk_err}")

            target_device = "cpu"
            if is_openvino and self.hardware_diag["gpu_vendor"] == "Intel":
                target_device = "cpu"
            
            # Prueba de vida del modelo con el dispositivo correcto
            dummy = np.zeros((64, 64, 3), dtype=np.uint8)
            if is_sklearn:
                try:
                    new_model(dummy, verbose=False)
                except Exception as te:
                    print(f"[Detector] Advertencia en prueba sklearn: {te}")
            else:
                test_device = target_device.lower() if is_openvino else self.device
                try:
                    new_model(dummy, verbose=False, device=test_device)
                except Exception as te:
                    print(f"[Detector] Advertencia en prueba de vida: {te}")
            
            # Si todo ha ido bien, actualizamos el modelo activo
            self.model = new_model
            self.current_family = family
            self.active_name = target_name
            self.is_openvino_active = is_openvino
            self.active_device = target_device
            
            # Detectar si el modelo soporta Zero-Shot / World Prompting
            self.is_zero_shot_active = hasattr(self.model, 'set_classes') or "world" in family.lower() or "world" in target_name.lower()
            
            return target_name
        except Exception as e:
            log_error("EXE-COR-LOAD-03", f"Fallo critico cargando {target_name}: {e}")
            return None

    def change_secondary_model(self, family, alias):
        """Carga un modelo secundario sin interferir con el primario."""
        if family not in self.architectures or alias not in self.architectures[family]["aliases"]:
            return None

        model_path = self.architectures[family]["aliases"][alias]
        target_name = os.path.basename(model_path)
        try:
            if family == "ollama":
                ollama_url = os.getenv("OLLAMA_URL", "http://192.168.0.135:11434").rstrip("/")
                model_name = model_path.replace("ollama://", "")
                self.secondary_model = OllamaVisionWrapper(model_name, base_url=ollama_url)
                self.secondary_family = family
                self.secondary_name = alias
                self.is_secondary_openvino = False
                self.secondary_device = "Ollama VLM"
                return alias

            is_openvino = os.path.isdir(model_path)
            new_model = None
            is_sklearn = False

            try:
                if "rtdetr" in family.lower() or "rtdetr" in target_name.lower():
                    new_model = RTDETR(model_path)
                else:
                    new_model = YOLO(model_path, task='detect') if is_openvino else YOLO(model_path)
            except Exception as yolo_err:
                print(f"[Detector] YOLO no pudo cargar M2 {target_name}: {yolo_err}")
                try:
                    new_model = SklearnModelWrapper(model_path)
                    is_sklearn = True
                except Exception as sk_err:
                    raise RuntimeError(f"No es YOLO ni sklearn: YOLO={yolo_err}, sklearn={sk_err}")

            target_device = "cpu"
            if is_openvino and self.hardware_diag["gpu_vendor"] == "Intel":
                target_device = "cpu"
            
            dummy = np.zeros((64, 64, 3), dtype=np.uint8)
            if is_sklearn:
                try:
                    new_model(dummy, verbose=False)
                except Exception as te:
                    print(f"[Detector] Advertencia en prueba sklearn M2: {te}")
            else:
                test_device = target_device.lower() if is_openvino else self.device
                try:
                    new_model(dummy, verbose=False, device=test_device)
                except Exception as te:
                    print(f"[Detector] Advertencia en prueba de vida M2: {te}")
            
            self.secondary_model = new_model
            self.secondary_family = family
            self.secondary_name = target_name
            self.is_secondary_openvino = is_openvino
            self.secondary_device = target_device
            return target_name
        except Exception as e:
            log_error("EXE-COR-LOAD-03", f"Fallo critico cargando modelo secundario {target_name}: {e}")
            return None

    def clear_secondary_model(self):
        """Elimina el modelo secundario, volviendo al modo single."""
        self.secondary_model = None
        self.secondary_name = None
        self.secondary_family = None
        self.is_secondary_openvino = False
        self.secondary_device = "CPU"

    def is_dual_mode(self):
        """Retorna True si hay un modelo secundario activo."""
        return self.secondary_model is not None

    @staticmethod
    def _compute_iou(box_a, box_b):
        """Calcula Intersection over Union entre dos bboxes (x1,y1,x2,y2)."""
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0

    @staticmethod
    def _merge_detections(primary_dets, secondary_dets, iou_threshold=0.5):
        """Fusiona detecciones de dos modelos eliminando duplicados por IoU.
        
        Se conserva la detección del modelo primario cuando hay solapamiento.
        """
        merged = list(primary_dets)
        for s_det in secondary_dets:
            is_duplicate = False
            for p_det in primary_dets:
                iou = ObjectDetector._compute_iou(s_det["bbox"], p_det["bbox"])
                if iou >= iou_threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                merged.append(s_det)
        return merged

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

    def detect(self, frame, target_classes=None, target_classes_secondary=None, zones=None, conf_threshold=0.35):
        """Ejecuta inferencia (single o dual) y mapeo a zonas con filtros independientes."""
        if self.model is None:
            return frame, []

        try:
            h_frame, w_frame = frame.shape[:2]
            
            # --- Inferencia del Modelo Primario ---
            if getattr(self, 'is_openvino_active', False):
                target_device = "gpu" if getattr(self, 'active_device', 'CPU') == "GPU" else "cpu"
            else:
                target_device = self.device

            kwargs = {
                "persist": True, 
                "conf": conf_threshold, 
                "verbose": False,
                "device": target_device
            }
            
            # Determinar si el modelo primario es de clasificación
            is_classify = getattr(self.model, 'task', 'detect') == 'classify'

            primary_detections = []
            if target_classes is not None and len(target_classes) == 0:
                # Si el filtro primario está vacío, no detectamos nada del primario
                pass
            elif isinstance(self.model, OllamaVisionWrapper):
                # Inferencia mediante VLM en Ollama para el modelo primario
                primary_detections = self.model.detect(
                    frame, conf_threshold=conf_threshold, source="primary"
                )
                # Asignar zonas a cada detección
                for det in primary_detections:
                    cx = (det["bbox"][0] + det["bbox"][2]) / 2
                    cy = (det["bbox"][1] + det["bbox"][3]) / 2
                    det["zone_indices"] = self._get_zones_for_point(cx, cy, w_frame, h_frame, zones)
            elif isinstance(self.model, SklearnModelWrapper):
                # Pipeline especial: Haar Cascade (detectar caras) + sklearn (clasificar cada cara)
                primary_detections = self.model.detect_and_classify(
                    frame, conf_threshold=conf_threshold, source="primary"
                )
                # Asignar zonas a cada deteccion
                for det in primary_detections:
                    cx = (det["bbox"][0] + det["bbox"][2]) / 2
                    cy = (det["bbox"][1] + det["bbox"][3]) / 2
                    det["zone_indices"] = self._get_zones_for_point(cx, cy, w_frame, h_frame, zones)
            else:
                if target_classes is not None and not is_classify:
                    kwargs["classes"] = target_classes

                # Ejecutar inferencia primaria
                try:
                    if is_classify:
                        cls_kwargs = {k: v for k, v in kwargs.items() if k not in ("persist", "classes")}
                        results = self.model.predict(frame, **cls_kwargs)
                    else:
                        results = self.model.track(frame, **kwargs)
                except Exception:
                    # Fallback a predict normal si track falla o da error de dispositivo
                    cls_kwargs = {k: v for k, v in kwargs.items() if k not in ("persist",)}
                    results = self.model.predict(frame, **cls_kwargs)

                # Parsear resultados primarios
                for result in results:
                    # Caso A: Modelo de Clasificación (probs)
                    if hasattr(result, 'probs') and result.probs is not None and (not hasattr(result, 'boxes') or result.boxes is None or len(result.boxes) == 0):
                        probs = result.probs
                        top1_id = int(probs.top1)
                        top1_conf = float(probs.top1conf)
                        if top1_conf >= conf_threshold:
                            names = getattr(self.model, 'names', {})
                            label = names.get(top1_id, f"Clase {top1_id}")
                            # Caja virtual central que cubre el 80% de la pantalla
                            x1, y1 = int(w_frame * 0.1), int(h_frame * 0.1)
                            x2, y2 = int(w_frame * 0.9), int(h_frame * 0.9)
                            cx, cy = w_frame / 2, h_frame / 2
                            zone_indices = self._get_zones_for_point(cx, cy, w_frame, h_frame, zones)
                            primary_detections.append({
                                "label": label, "confidence": top1_conf,
                                "class_id": top1_id,
                                "track_id": None,
                                "zone_indices": zone_indices, "bbox": (x1, y1, x2, y2),
                                "source": "primary",
                                "is_classification": True
                            })
                    # Caso B: Modelo de Detección (boxes)
                    elif hasattr(result, 'boxes') and result.boxes is not None:
                        for box in result.boxes:
                            cls_id = int(box.cls[0])
                            names = getattr(self.model, 'names', {})
                            label = names.get(cls_id, f"Clase {cls_id}")
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                            zone_indices = self._get_zones_for_point(cx, cy, w_frame, h_frame, zones)
                            primary_detections.append({
                                "label": label, "confidence": float(box.conf[0]),
                                "class_id": cls_id,
                                "track_id": int(box.id[0]) if box.id is not None else None,
                                "zone_indices": zone_indices, "bbox": (int(x1), int(y1), int(x2), int(y2)),
                                "source": "primary",
                                "is_classification": False
                            })

            # --- Inferencia del Modelo Secundario ---
            secondary_detections = []
            if self.secondary_model is not None:
                try:
                    if isinstance(self.secondary_model, OllamaVisionWrapper):
                        # Inferencia mediante VLM en Ollama para el modelo secundario
                        secondary_detections = self.secondary_model.detect(
                            frame, conf_threshold=conf_threshold, source="secondary"
                        )
                        # Asignar zonas a cada detección
                        for det in secondary_detections:
                            cx = (det["bbox"][0] + det["bbox"][2]) / 2
                            cy = (det["bbox"][1] + det["bbox"][3]) / 2
                            det["zone_indices"] = self._get_zones_for_point(cx, cy, w_frame, h_frame, zones)
                    elif isinstance(self.secondary_model, SklearnModelWrapper):
                        # Pipeline sklearn para modelo secundario
                        if target_classes_secondary is not None and len(target_classes_secondary) == 0:
                            pass  # Filtro vacio, no detectar nada
                        else:
                            secondary_detections = self.secondary_model.detect_and_classify(
                                frame, conf_threshold=conf_threshold, source="secondary"
                            )
                            for det in secondary_detections:
                                cx = (det["bbox"][0] + det["bbox"][2]) / 2
                                cy = (det["bbox"][1] + det["bbox"][3]) / 2
                                det["zone_indices"] = self._get_zones_for_point(cx, cy, w_frame, h_frame, zones)
                    else:
                        is_sec_classify = getattr(self.secondary_model, 'task', 'detect') == 'classify'
                        if self.is_secondary_openvino:
                            sec_device = "gpu" if self.secondary_device == "GPU" else "cpu"
                        else:
                            sec_device = self.device
                        
                        sec_kwargs = {
                            "persist": True,
                            "conf": conf_threshold,
                            "verbose": False,
                            "device": sec_device
                        }
                        
                        if target_classes_secondary is not None and len(target_classes_secondary) == 0:
                            sec_results = []
                        else:
                            if target_classes_secondary is not None and not is_sec_classify:
                                sec_kwargs["classes"] = target_classes_secondary
                            
                            if is_sec_classify:
                                cls_sec_kwargs = {k: v for k, v in sec_kwargs.items() if k not in ("persist",)}
                                sec_results = self.secondary_model.predict(frame, **cls_sec_kwargs)
                            else:
                                try:
                                    sec_results = self.secondary_model.track(frame, **sec_kwargs)
                                except Exception:
                                    try:
                                        sec_results = self.secondary_model.track(frame, conf=conf_threshold, verbose=False)
                                    except Exception:
                                        sec_results = self.secondary_model.predict(frame, conf=conf_threshold, verbose=False)

                        # Parsear resultados secundarios
                        for result in sec_results:
                            # Caso A: Clasificación
                            if hasattr(result, 'probs') and result.probs is not None and (not hasattr(result, 'boxes') or result.boxes is None or len(result.boxes) == 0):
                                probs = result.probs
                                top1_id = int(probs.top1)
                                top1_conf = float(probs.top1conf)
                                if top1_conf >= conf_threshold:
                                    names = getattr(self.secondary_model, 'names', {})
                                    label = names.get(top1_id, f"Clase {top1_id}")
                                    x1, y1 = int(w_frame * 0.15), int(h_frame * 0.15)
                                    x2, y2 = int(w_frame * 0.85), int(h_frame * 0.85)
                                    cx, cy = w_frame / 2, h_frame / 2
                                    zone_indices = self._get_zones_for_point(cx, cy, w_frame, h_frame, zones)
                                    secondary_detections.append({
                                        "label": label, "confidence": top1_conf,
                                        "class_id": top1_id,
                                        "track_id": None,
                                        "zone_indices": zone_indices, "bbox": (x1, y1, x2, y2),
                                        "source": "secondary",
                                        "is_classification": True
                                    })
                            # Caso B: Detección
                            elif hasattr(result, 'boxes') and result.boxes is not None:
                                for box in result.boxes:
                                    cls_id = int(box.cls[0])
                                    names = getattr(self.secondary_model, 'names', {})
                                    label = names.get(cls_id, f"Clase {cls_id}")
                                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                                    zone_indices = self._get_zones_for_point(cx, cy, w_frame, h_frame, zones)
                                    secondary_detections.append({
                                        "label": label, "confidence": float(box.conf[0]),
                                        "class_id": cls_id,
                                        "track_id": int(box.id[0]) if box.id is not None else None,
                                        "zone_indices": zone_indices, "bbox": (int(x1), int(y1), int(x2), int(y2)),
                                        "source": "secondary",
                                        "is_classification": False
                                    })
                except Exception as e:
                    log_error("EXE-COR-DET-02", f"Error en modelo secundario: {e}")

            # --- Fusión de detecciones ---
            if secondary_detections:
                all_detections = self._merge_detections(primary_detections, secondary_detections)
            else:
                all_detections = primary_detections

            # Construir frame anotado con colores diferenciados (delegado al painter)
            annotated = frame.copy()
            
            return annotated, all_detections
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

