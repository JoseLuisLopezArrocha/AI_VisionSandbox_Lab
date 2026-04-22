"""
Módulo de Verificación Avanzada.
Proporciona validación secundaria mediante segmentación local, YOLO-World o VLMs externos.
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
    Módulo de Verificación Avanzada.
    """
    
    _seg_model = None
    _world_model = None

    @staticmethod
    def _get_seg_model():
        if SecondaryValidator._seg_model is None:
            try:
                SecondaryValidator._seg_model = YOLO("yolo11n-seg.pt")
            except Exception as e:
                log_error("EXE-COR-LOAD-03", f"Error cargando modelo segmentación: {e}")
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
        """Ejecuta validación en hilo secundario."""
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
                result_ok, result_msg, evidence_img = SecondaryValidator._validate_universal(frame, config.get("prompt", "object"))
            elif provider == "local_seg" or config.get("use_local_seg", False):
                result_ok, result_msg, evidence_img = SecondaryValidator._validate_local_seg(frame, config)
            elif provider != "None":
                _, buffer = cv2.imencode(".jpg", frame)
                if provider == "huggingface":
                    result_msg = SecondaryValidator._validate_hf(buffer.tobytes(), config)
                    result_ok = "Confirmado" in result_msg
                elif provider == "ollama":
                    img_base64 = base64.b64encode(buffer).decode("utf-8")
                    result_msg = SecondaryValidator._validate_ollama(img_base64, config)
                    result_ok = "Confirmado" in result_msg

            log_callback(f"{'✅' if result_ok else '❌'} [VAL: {rule_name}] {result_msg}")
            if evidence_callback and evidence_img is not None:
                evidence_callback(evidence_img, result_msg, result_ok)
        except Exception as e:
            log_error("EXE-COR-EVT-05", f"Error en validación ({rule_name}): {e}")

    @staticmethod
    def _validate_universal(frame, prompt):
        model = SecondaryValidator._get_world_model()
        if not model: return False, "Modelo World no disponible", None
        model.set_classes([prompt])
        results = model(frame, verbose=False)[0]
        if len(results.boxes) > 0 and float(results.boxes[0].conf[0]) > 0.25:
            return True, f"IA Universal: '{prompt}' detectado", results.plot()
        return False, f"No se encontró '{prompt}'", None

    @staticmethod
    def _validate_local_seg(frame, config):
        model = SecondaryValidator._get_seg_model()
        if not model: return False, "Modelo segmentación no disponible", None
        target = config.get("target_class", "person")
        results = model(frame, verbose=False)[0]
        if results.masks is not None:
            for box in results.boxes:
                label = results.names[int(box.cls[0])]
                if (target == "Cualquiera" or label.lower() == target.lower()) and float(box.conf[0]) > 0.35:
                    return True, f"Evidencia confirmada ({label})", results.plot()
        return False, "No detectado por segmentación", None

    @staticmethod
    def _validate_hf(img_bytes, config):
        # Implementación simplificada (igual que original)
        return "HF: Validación no disponible"

    @staticmethod
    def _validate_ollama(img_base64, config):
        # Implementación simplificada (igual que original)
        return "Ollama: Validación no disponible"
