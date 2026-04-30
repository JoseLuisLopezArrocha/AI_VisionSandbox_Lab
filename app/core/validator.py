"""
Modulo de Verificacion Avanzada.
Proporciona validacion secundaria mediante segmentacion local, YOLO-World,
Ollama (modelos multimodales locales) y HuggingFace (Inference API).
"""

import cv2
import base64
import requests
import json
import threading
import time
import os
import numpy as np
from ultralytics import YOLO
from ..utils.error_handler import log_error

class SecondaryValidator:
    """
    Modulo de Verificacion Avanzada.
    Soporta multiples proveedores de validacion de imagenes.
    """
    
    _seg_model = None
    _world_model = None

    @staticmethod
    def _get_seg_model():
        if SecondaryValidator._seg_model is None:
            try:
                SecondaryValidator._seg_model = YOLO("yolo11n-seg.pt")
            except Exception as e:
                log_error("EXE-COR-LOAD-03", f"Error cargando modelo segmentacion: {e}")
        return SecondaryValidator._seg_model

    @staticmethod
    def _get_world_model():
        if SecondaryValidator._world_model is None:
            try:
                SecondaryValidator._world_model = YOLO("yolov8s-worldv2.pt")
            except Exception as e:
                log_error("EXE-COR-LOAD-03", f"Error cargando modelo World: {e}")
        return SecondaryValidator._world_model

    @staticmethod
    def validate_async(frame, config, rule_name, log_callback, evidence_callback=None):
        """Ejecuta validacion en hilo secundario."""
        threading.Thread(
            target=SecondaryValidator._execute_validation,
            args=(frame, config, rule_name, log_callback, evidence_callback),
            daemon=True
        ).start()

    @staticmethod
    def _execute_validation(frame, config, rule_name, log_callback, evidence_callback):
        provider = config.get("provider", "None")
        try:
            result_ok, result_msg, evidence_img = False, "", None

            if provider == "universal":
                prompt = config.get("prompt", "object")
                result_ok, result_msg, evidence_img = SecondaryValidator._validate_universal(frame, prompt)
            elif provider == "local_seg":
                target_class = config.get("prompt", "person") or config.get("target_class", "person")
                seg_config = {**config, "target_class": target_class}
                result_ok, result_msg, evidence_img = SecondaryValidator._validate_local_seg(frame, seg_config)
            elif provider == "ollama":
                result_ok, result_msg = SecondaryValidator._validate_ollama(frame, config)
            elif provider == "huggingface":
                result_ok, result_msg = SecondaryValidator._validate_hf(frame, config)

            status = "OK" if result_ok else "NO"
            if log_callback:
                log_callback(f"[VAL:{status}] [{rule_name}] {result_msg}")
            if evidence_callback and evidence_img is not None:
                evidence_callback(evidence_img, result_msg, result_ok)
        except Exception as e:
            log_error("EXE-COR-EVT-05", f"Error en validacion ({rule_name}): {e}")

    @staticmethod
    def _validate_universal(frame, prompt):
        model = SecondaryValidator._get_world_model()
        if not model: return False, "Modelo World no disponible", None
        model.set_classes([prompt])
        results = model(frame, verbose=False)[0]
        if len(results.boxes) > 0 and float(results.boxes[0].conf[0]) > 0.25:
            return True, f"IA Universal: '{prompt}' detectado", results.plot()
        return False, f"No se encontro '{prompt}'", None

    @staticmethod
    def _validate_local_seg(frame, config):
        model = SecondaryValidator._get_seg_model()
        if not model: return False, "Modelo segmentacion no disponible", None
        target = config.get("target_class", "person")
        results = model(frame, verbose=False)[0]
        if results.masks is not None:
            for box in results.boxes:
                label = results.names[int(box.cls[0])]
                if (target == "Cualquiera" or label.lower() == target.lower()) and float(box.conf[0]) > 0.35:
                    return True, f"Evidencia confirmada ({label})", results.plot()
        return False, "No detectado por segmentacion", None

    @staticmethod
    def _validate_ollama(frame, config):
        """
        Validacion mediante Ollama (modelos multimodales locales).
        Envia la imagen al endpoint /api/generate con un prompt de confirmacion.
        """
        url = config.get("ollama_url", "").rstrip("/")
        model = config.get("ollama_model", "llava")
        prompt = config.get("prompt", "object")
        
        if not url:
            return False, "Ollama: URL no configurada (ver Ajustes)"
        if not model:
            return False, "Ollama: Modelo no configurado (ver Ajustes)"
        
        try:
            # Codificar imagen a base64
            _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            img_base64 = base64.b64encode(buffer).decode("utf-8")
            
            # Construir prompt de validacion
            question = (
                f"Analyze this image. Is there a '{prompt}' present? "
                f"Answer with exactly 'Confirmado' if yes, or 'No detectado' if no. "
                f"Then briefly explain why."
            )
            
            payload = {
                "model": model,
                "prompt": question,
                "images": [img_base64],
                "stream": False,
            }
            
            resp = requests.post(f"{url}/api/generate", json=payload, timeout=30)
            
            if resp.status_code == 200:
                data = resp.json()
                response_text = data.get("response", "").strip()
                is_confirmed = "confirmado" in response_text.lower()
                status = "Confirmado" if is_confirmed else "No detectado"
                # Truncar respuesta para el log
                short_response = response_text[:100] + ("..." if len(response_text) > 100 else "")
                return is_confirmed, f"Ollama ({model}): {status} -- {short_response}"
            else:
                return False, f"Ollama: Error HTTP {resp.status_code}"
                
        except requests.exceptions.ConnectionError:
            return False, f"Ollama: Sin conexion a {url}"
        except requests.exceptions.Timeout:
            return False, "Ollama: Timeout (>30s)"
        except Exception as e:
            return False, f"Ollama: Error -- {str(e)[:60]}"

    @staticmethod
    def _validate_hf(frame, config):
        """
        Validacion mediante HuggingFace Inference API.
        Envia la imagen al modelo configurado para clasificacion/VQA.
        """
        api_key = config.get("huggingface_api_key", "")
        model_id = config.get("huggingface_model", "Salesforce/blip-vqa-base")
        prompt = config.get("prompt", "object")
        
        if not api_key:
            return False, "HuggingFace: API Key no configurada (ver Ajustes)"
        if not model_id:
            return False, "HuggingFace: Modelo no configurado (ver Ajustes)"
        
        try:
            # Codificar imagen a bytes
            _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            img_bytes = buffer.tobytes()
            
            api_url = f"https://api-inference.huggingface.co/models/{model_id}"
            headers = {"Authorization": f"Bearer {api_key}"}
            
            # Determinar si es un modelo VQA o de clasificacion
            if "vqa" in model_id.lower() or "blip" in model_id.lower():
                # Modelo VQA: enviar pregunta + imagen
                question = f"Is there a {prompt} in this image? Answer yes or no."
                
                # Para VQA multipart
                payload = {
                    "inputs": {
                        "question": question,
                        "image": base64.b64encode(img_bytes).decode("utf-8"),
                    }
                }
                resp = requests.post(api_url, headers=headers, json=payload, timeout=30)
            else:
                # Modelo de clasificacion de imagenes: enviar solo la imagen
                resp = requests.post(api_url, headers=headers, data=img_bytes, timeout=30)
            
            if resp.status_code == 200:
                data = resp.json()
                
                if isinstance(data, list) and len(data) > 0:
                    # Respuesta de clasificacion: [{label, score}, ...]
                    if isinstance(data[0], dict) and "label" in data[0]:
                        top = data[0]
                        label_found = top.get("label", "unknown")
                        score = top.get("score", 0)
                        is_match = prompt.lower() in label_found.lower() or score > 0.5
                        status = "Confirmado" if is_match else "No detectado"
                        return is_match, f"HF ({model_id.split('/')[-1]}): {status} -- {label_found} ({score:.2f})"
                    # Respuesta VQA: [{answer, score}, ...]
                    elif isinstance(data[0], dict) and "answer" in data[0]:
                        answer = data[0].get("answer", "")
                        is_confirmed = "yes" in answer.lower() or "si" in answer.lower()
                        status = "Confirmado" if is_confirmed else "No detectado"
                        return is_confirmed, f"HF ({model_id.split('/')[-1]}): {status} -- {answer}"
                
                return False, f"HF: Respuesta inesperada del modelo"
                
            elif resp.status_code == 401:
                return False, "HF: API Key invalida (401)"
            elif resp.status_code == 503:
                return False, "HF: Modelo cargando, reintenta en unos segundos"
            else:
                return False, f"HF: Error HTTP {resp.status_code}"
                
        except requests.exceptions.Timeout:
            return False, "HF: Timeout (>30s)"
        except Exception as e:
            return False, f"HF: Error -- {str(e)[:60]}"
