import os
import sys
import platform
import subprocess
import torch

class HardwareManager:
    """
    Gestor de Diagnóstico de Hardware y Aceleración.
    
    Identifica la presencia de GPUs NVIDIA, AMD o Intel e informa sobre
    la disponibilidad de las librerías de aceleración correspondientes.
    """
    
    @staticmethod
    def get_diagnostics():
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
            output = subprocess.check_output(cmd, shell=True).decode().lower()
            if "nvidia" in output:
                diag["gpu_vendor"] = "NVIDIA"
                diag["gpu_name"] = "NVIDIA GeForce / Quadro"
            elif "amd" in output or "radeon" in output:
                diag["gpu_vendor"] = "AMD"
                diag["gpu_name"] = "AMD Radeon"
            elif "intel" in output:
                diag["gpu_vendor"] = "Intel"
                diag["gpu_name"] = "Intel UHD / Iris Xe / Arc"
        except:
            pass

        # 2. Comprobación de Backends
        
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
            # OpenVINO es 'disponible' si hay hardware Intel o si la librería está instalada (funciona en CPU también)
            diag["backends"]["openvino"]["available"] = (diag["gpu_vendor"] == "Intel")
        except ImportError:
            diag["backends"]["openvino"]["installed"] = False

        # DirectML (AMD / Genérico Windows)
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
            diag["best_backend"] = "openvino_cpu" # OpenVINO en CPU es más rápido que Torch CPU
        
        return diag

    @staticmethod
    def get_backend_for_ultralytics():
        """Devuelve el string crudo que Ultralytics espera o el formato de exportación."""
        diag = HardwareManager.get_diagnostics()
        if diag["best_backend"] == "cuda":
            return "0" # Device ID
        elif "openvino" in diag["best_backend"]:
            return "openvino"
        elif diag["best_backend"] == "directml":
            return "directml"
        return "cpu"

if __name__ == "__main__":
    # Test rápido
    d = HardwareManager.get_diagnostics()
    print(f"Mejor opción detectada: {d['best_backend']}")
