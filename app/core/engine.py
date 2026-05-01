import cv2
import os
import yt_dlp
import numpy as np
import threading
import time
from typing import Optional, Any, Tuple
from vidgear.gears import CamGear
from ..utils.error_handler import log_error

class VisionEngine:
    """
    Motor de Adquisición de Vídeo Híbrido con soporte de hilos.
    """

    def __init__(self, source: str, resolution: str = "720p") -> None:
        self.source: str = source
        self.resolution: str = resolution
        self.stream: Optional[CamGear] = None
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_stream: bool = False
        self.is_live: bool = False
        self.is_camera: bool = False
        
        # Hilo de lectura para fuentes no CamGear
        self._frame: Optional[np.ndarray] = None
        self._running: bool = False
        self._read_lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        
        self._connect(source, self.resolution)

    def _connect(self, source: str, resolution: str = "720p") -> None:
        """Establece conexión con la fuente de vídeo."""
        try:
            stream_protocols = ("rtsp://", "rtmp://", "http://", "https://", "udp://")
            is_url = any(source.lower().startswith(p) for p in stream_protocols)
            self.is_stream = is_url and not os.path.exists(source)
            self.is_live = False
            target_source = source

            if source.isdigit():
                self.is_live = True
                self.is_stream = False
                self.is_camera = True
                cam_idx = int(source)
                
                # MSMF es el estándar moderno en Windows y suele ser más robusto para permisos
                backends = [cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY] if os.name == 'nt' else [cv2.CAP_ANY]
                
                for backend in backends:
                    self.cap = cv2.VideoCapture(cam_idx, backend)
                    if self.cap.isOpened():
                        # Eliminamos la resolución forzada porque causaba error MF_E_INVALIDMEDIATYPE (-1072875772)
                        
                        # Prueba de lectura crítica
                        for _ in range(3):
                            ret, frame = self.cap.read()
                            if ret: 
                                self._frame = frame
                                break
                            time.sleep(0.05)
                        if ret: break
                        else: self.cap.release()
                
                if not self.cap or not self.cap.isOpened():
                    log_error("EXE-COR-CONN-02", f"Fallo total al abrir cámara {source}. Revisa permisos de Windows.")
            elif self.is_stream:
                if "youtube.com" in source or "youtu.be" in source:
                    try:
                        # Para YouTube siempre usamos CamGear ya que gestiona mucho mejor el buffering que VideoCapture
                        options = {"STREAM_RESOLUTION": resolution, "THREADED_QUEUE_MODE": True}
                        self.stream = CamGear(source=source, stream_mode=True, logging=False, **options).start()
                        self.is_live = True # Marcamos como live para usar el lector de CamGear directamente
                    except Exception as e:
                        log_error("EXE-COR-CONN-01", f"Error con CamGear en YouTube: {e}")
                        self.cap = cv2.VideoCapture(target_source)
                else:
                    self.is_live = any(p in source.lower() for p in ("rtsp://", "rtmp://", "udp://"))

                    if self.is_live:
                        options = {"STREAM_RESOLUTION": resolution, "THREADED_QUEUE_MODE": True}
                        self.stream = CamGear(source=source, stream_mode=True, logging=False, **options).start()
                    else:
                        self.cap = cv2.VideoCapture(target_source)
            else:
                if os.path.exists(source):
                    self.cap = cv2.VideoCapture(source)
                else:
                    log_error("EXE-COR-CONN-01", f"Archivo no encontrado: {source}")
            
            # Iniciar hilo de lectura si usamos VideoCapture (solo para archivos locales o fuentes no-CamGear)
            if self.cap:
                self._running = True
                self._thread = threading.Thread(target=self._reader, daemon=True)
                self._thread.start()

        except Exception as e:
            log_error("EXE-COR-CONN-01", f"Error general conexión: {e}")

    def _reader(self):
        """Hilo de lectura continua para VideoCapture."""
        while self._running:
            if self.cap:
                ret, frame = self.cap.read()
                if ret:
                    with self._read_lock:
                        self._frame = frame
                else:
                    log_error("EXE-COR-CONN-01", "Fallo de lectura en hilo _reader. Reintentando...")
                    time.sleep(0.1) # Mayor espera si falla lectura
            else:
                break
            time.sleep(0.001)

    def get_frame(self) -> Optional[np.ndarray]:
        if self.is_live and self.stream:
            return self.stream.read()
        elif self.cap:
            with self._read_lock:
                return self._frame
        return None

    def get_fps(self) -> float:
        if self.is_live and self.stream: return 30.0
        if self.cap:
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            return fps if fps > 0 else 30.0
        return 30.0

    def seek_back(self, seconds: int = 5) -> None:
        if self.is_live or not self.cap: return
        fps = self.get_fps()
        current = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        with self._read_lock:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, current - (seconds * fps)))

    def seek_forward(self, seconds: int = 5) -> None:
        if self.is_live or not self.cap: return
        fps = self.get_fps()
        current = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        with self._read_lock:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, current + (seconds * fps))

    def reconnect(self, new_source: str, resolution: Optional[str] = None) -> None:
        self.release()
        self.source = new_source
        if resolution: self.resolution = resolution
        self._connect(new_source, self.resolution)

    def release(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        
        if self.stream:
            try: self.stream.stop()
            except: pass
            self.stream = None
        if self.cap:
            self.cap.release()
            self.cap = None
        self._frame = None
