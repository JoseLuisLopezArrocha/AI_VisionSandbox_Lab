"""
Módulo de Gestión de Hardware y Aceleración.
Detecta GPUs y backends disponibles para optimizar la inferencia.
"""

import platform
import subprocess
import torch
from typing import Dict, Any
from ..utils.error_handler import log_error

class HardwareManager:
    """
    Gestor de Diagnóstico de Hardware y Aceleración.
    
    Identifica la presencia de GPUs NVIDIA, AMD o Intel e informa sobre
    la disponibilidad de las librerías de aceleración correspondientes.
    """
    
    @staticmethod
    def get_diagnostics() -> Dict[str, Any]:
        """Realiza un escaneo completo del hardware y software de aceleración."""
        diag = {
            "os": platform.system(),
            "cpu": platform.processor(),
            "gpu_vendor": "Unknown",
            "gpu_name": "Generic Display Adapter",
            "backends": {
                "cuda": {"available": False, "installed": False},
                "openvino": {"available": False, "installed": False},
                "directml": {"available": False, "installed": False}
            },
            "best_backend": "cpu"
        }
        
        # 1. Detección de Fabricante de GPU (Windows)
        try:
            cmd = "wmic path win32_VideoController get name"
            output = subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL).decode().lower()
            if "nvidia" in output:
                diag["gpu_vendor"] = "NVIDIA"
                diag["gpu_name"] = "NVIDIA GeForce / Quadro"
            elif "amd" in output or "radeon" in output:
                diag["gpu_vendor"] = "AMD"
                diag["gpu_name"] = "AMD Radeon"
            elif "intel" in output:
                diag["gpu_vendor"] = "Intel"
                diag["gpu_name"] = "Intel UHD / Iris Xe / Arc"
        except (subprocess.CalledProcessError, FileNotFoundError, Exception) as e:
            log_error("EXE-COR-HW-04", f"Fallo WMIC (fallback a métodos nativos): {e}")

        # 2. Comprobación de Backends
        try:
            # CUDA (NVIDIA)
            diag["backends"]["cuda"]["available"] = torch.cuda.is_available()
            try:
                import pynvml
                diag["backends"]["cuda"]["installed"] = True
            except ImportError:
                diag["backends"]["cuda"]["installed"] = diag["backends"]["cuda"]["available"]

            # OpenVINO (Intel)
            try:
                import openvino
                diag["backends"]["openvino"]["installed"] = True
                diag["backends"]["openvino"]["available"] = (diag["gpu_vendor"] == "Intel")
            except ImportError:
                diag["backends"]["openvino"]["installed"] = False

            # DirectML (AMD)
            try:
                import torch_directml
                diag["backends"]["directml"]["installed"] = True
                diag["backends"]["directml"]["available"] = (diag["gpu_vendor"] == "AMD")
            except ImportError:
                diag["backends"]["directml"]["installed"] = False

            # 3. Determinar el mejor Backend
            if diag["backends"]["cuda"]["available"]:
                diag["best_backend"] = "cuda"
            elif diag["backends"]["openvino"]["installed"] and diag["gpu_vendor"] == "Intel":
                diag["best_backend"] = "openvino"
            elif diag["backends"]["directml"]["installed"] and diag["gpu_vendor"] == "AMD":
                diag["best_backend"] = "directml"
            elif diag["backends"]["openvino"]["installed"]:
                diag["best_backend"] = "openvino_cpu"
        except Exception as e:
            log_error("EXE-COR-HW-04", f"Error comprobando backends: {e}")
        
        return diag

    @staticmethod
    def get_backend_for_ultralytics() -> Any:
        """
        Devuelve el identificador de dispositivo óptimo para Ultralytics (modelos .pt).
        Retorna un int (0), torch.device o "cpu".
        """
        diag = HardwareManager.get_diagnostics()
        
        if diag["best_backend"] == "cuda":
            return 0  # GPU NVIDIA
            
        elif diag["best_backend"] == "directml":
            try:
                import torch_directml
                return torch_directml.device()
            except ImportError:
                return "cpu"
                
        return "cpu"
