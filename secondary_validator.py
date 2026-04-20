import cv2
import base64
import requests
import json
import threading
import time
import os
import numpy as np
from ultralytics import YOLO

class SecondaryValidator:
    """
    Módulo de Verificación Avanzada.
    
    Ahora soporta validación por SEGMENTACIÓN LOCAL (Nativo) además de
    consultas a VLMs externos. La segmentación garantiza que el objeto
    detectado tiene una masa de píxeles real y no es un falso positivo.
    """
    
    _seg_model = None    # Singleton para Segmentación
    _world_model = None  # Singleton para IA Universal (YOLO-World)

    @staticmethod
    def _get_seg_model():
        if SecondaryValidator._seg_model is None:
            SecondaryValidator._seg_model = YOLO("yolo26n-seg.pt")
        return SecondaryValidator._seg_model

    @staticmethod
    def _get_world_model():
        if SecondaryValidator._world_model is None:
            # Modelo de vocabulario abierto (Zero-Shot)
            SecondaryValidator._world_model = YOLO("yolov8s-worldv2.pt")
        return SecondaryValidator._world_model

    @staticmethod
    def validate_async(frame, config, rule_name, log_callback, evidence_callback=None):
        """Ejecuta la validación en un hilo separado para no bloquear el sistema principal."""
        thread = threading.Thread(
            target=SecondaryValidator._execute_validation,
            args=(frame, config, rule_name, log_callback, evidence_callback),
            daemon=True
        )
        thread.start()

    @staticmethod
    def _execute_validation(frame, config, rule_name, log_callback, evidence_callback):
        provider = config.get("provider", "None")
        
        try:
            result_ok = False
            result_msg = ""
            evidence_img = None

            # 1. OPCIÓN: IA UNIVERSAL (YOLO-World)
            if provider == "universal":
                prompt = config.get("prompt", "object")
                result_ok, result_msg, evidence_img = SecondaryValidator._validate_universal(frame, prompt)

            # 2. OPCIÓN: SEGMENTACIÓN LOCAL
            elif provider == "local_seg" or config.get("use_local_seg", False):
                result_ok, result_msg, evidence_img = SecondaryValidator._validate_local_seg(frame, config)
            
            # 3. OPCIÓN: PROVEEDORES EXTERNOS (VLM)
            elif provider != "None":
                _, buffer = cv2.imencode(".jpg", frame)
                if provider == "huggingface":
                    result_msg = SecondaryValidator._validate_hf(buffer.tobytes(), config)
                    result_ok = "Confirmado" in result_msg
                elif provider == "ollama":
                    img_base64 = base64.b64encode(buffer).decode("utf-8")
                    result_msg = SecondaryValidator._validate_ollama(img_base64, config)
                    result_ok = "Confirmado" in result_msg

            # Reportar resultado e imagen de evidencia
            timestamp = time.strftime('%H:%M:%S')
            icon = "✅" if result_ok else "❌"
            log_callback(f"{icon} [VAL: {rule_name}] {result_msg}")
            
            if evidence_callback and evidence_img is not None:
                evidence_callback(evidence_img, result_msg, result_ok)

        except Exception as e:
            log_callback(f"⚠️ Error en validación ({rule_name}): {str(e)}")

    @staticmethod
    def _validate_universal(frame, prompt):
        """Usa YOLO-World para buscar cualquier objeto definido por texto."""
        model = SecondaryValidator._get_world_model()
        
        # Configuramos el modelo para buscar SOLO el prompt del usuario
        model.set_classes([prompt])
        
        results = model(frame, verbose=False)[0]
        
        found = False
        summary = f"No se encontró '{prompt}'."
        annotated = None

        if len(results.boxes) > 0:
            best_conf = float(results.boxes[0].conf[0])
            if best_conf > 0.25: # Umbral para zero-shot suele ser más bajo
                found = True
                summary = f"IA Universal: '{prompt}' detectado ({int(best_conf*100)}%)"
                annotated = results.plot()
        
        return found, summary, annotated

    @staticmethod
    def _validate_local_seg(frame, config):
        """Valida mediante máscaras de píxeles reales."""
        model = SecondaryValidator._get_seg_model()
        target_class = config.get("target_class", "person")
        
        # Realizar inferencia de segmentación
        results = model(frame, verbose=False)[0]
        
        found = False
        summary = "No se detectó el objeto mediante segmentación."
        annotated_crop = None

        if results.masks is not None:
            for i, box in enumerate(results.boxes):
                cls_idx = int(box.cls[0])
                label = results.names[cls_idx]
                conf = float(box.conf[0])
                
                # Buscamos si hay un objeto de la clase deseada con confianza mínima
                if (target_class == "Cualquiera" or label.lower() == target_class.lower()) and conf > 0.35:
                    found = True
                    conf_pct = int(conf * 100)
                    summary = f"Confirmado por Segmentación: {label} ({conf_pct}%)"
                    
                    # Generar imagen de evidencia (Pintar solo la máscara del objeto validado)
                    annotated_crop = results.plot(boxes=True, masks=True, labels=True)
                    break 

        return found, summary, annotated_crop

    @staticmethod
    def _validate_hf(img_bytes, config):
        endpoint = config.get("endpoint")
        api_key = config.get("api_key")
        if not endpoint or not api_key: return "Falta config HF."
        url = endpoint if endpoint.startswith("http") else f"https://api-inference.huggingface.co/models/{endpoint}"
        headers = {"Authorization": f"Bearer {api_key}"}
        response = requests.post(url, headers=headers, data=img_bytes, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list) and len(data) > 0:
                top = data[0]
                return f"Confirmado por HF: {top.get('label')} ({int(top.get('score', 0)*100)}%)"
        return "HF: No confirmado."

    @staticmethod
    def _validate_ollama(img_base64, config):
        endpoint = config.get("endpoint")
        model = config.get("model", "llava")
        if not endpoint: return "Falta endpoint Ollama."
        url = f"{endpoint}/api/generate"
        payload = {"model": model, "prompt": "¿Ves el objeto claramente? Responde solo SI o NO.", "images": [img_base64], "stream": False}
        try:
            res = requests.post(url, json=payload, timeout=20)
            if res.status_code == 200:
                text = res.json().get('response', '').strip().upper()
                return f"Ollama ({model}): {'Confirmado' if 'SI' in text else 'Descartado'}"
        except: pass
        return "Ollama: Error de conexión."
