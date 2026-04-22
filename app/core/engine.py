"""
Motor de Adquisición de Vídeo.
Gestiona la conexión con streams y archivos locales.
"""

import cv2
import os
import yt_dlp
import numpy as np
from typing import Optional, Any, Tuple
from vidgear.gears import CamGear
from ..utils.error_handler import log_error

class VisionEngine:
    """
    Motor de Adquisición de Vídeo Híbrido.
    """

    def __init__(self, source: str, resolution: str = "720p") -> None:
        self.source: str = source
        self.resolution: str = resolution
        self.stream: Optional[CamGear] = None
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_stream: bool = False
        self.is_live: bool = False
        self._connect(source, self.resolution)

    def _connect(self, source: str, resolution: str = "720p") -> None:
        """Establece conexión con la fuente de vídeo."""
        try:
            stream_protocols = ("rtsp://", "rtmp://", "http://", "https://", "udp://")
            is_url = any(source.lower().startswith(p) for p in stream_protocols)
            self.is_stream = is_url and not os.path.exists(source)
            self.is_live = False
            target_source = source

            if self.is_stream:
                if "youtube.com" in source or "youtu.be" in source:
                    try:
                        ydl_opts = {'quiet': True, 'no_warnings': True}
                        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                            info = ydl.extract_info(source, download=False)
                            self.is_live = info.get('is_live', False)
                            if not self.is_live:
                                formats = info.get('formats', [])
                                res_val = int(resolution.replace("p", "")) if "p" in resolution else 720
                                best_url = None
                                for f in formats:
                                    if f.get('vcodec') != 'none' and f.get('acodec') != 'none':
                                        best_url = f.get('url')
                                        if f.get('height') and f.get('height') >= res_val:
                                            break
                                if best_url: target_source = best_url
                    except Exception as e:
                        log_error("EXE-COR-CONN-01", f"Error yt-dlp: {e}")
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
        except Exception as e:
            log_error("EXE-COR-CONN-01", f"Error general conexión: {e}")

    def get_frame(self) -> Optional[np.ndarray]:
        if self.is_stream and self.stream:
            return self.stream.read()
        elif self.cap:
            ret, frame = self.cap.read()
            return frame if ret else None
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
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, current - (seconds * fps)))

    def seek_forward(self, seconds: int = 5) -> None:
        if self.is_live or not self.cap: return
        fps = self.get_fps()
        current = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, current + (seconds * fps))

    def reconnect(self, new_source: str, resolution: Optional[str] = None) -> None:
        self.release()
        self.source = new_source
        if resolution: self.resolution = resolution
        self._connect(new_source, self.resolution)

    def release(self) -> None:
        if self.stream:
            try: self.stream.stop()
            except: pass
            self.stream = None
        if self.cap:
            self.cap.release()
            self.cap = None
