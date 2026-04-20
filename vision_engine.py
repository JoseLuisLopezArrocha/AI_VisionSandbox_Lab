import cv2
import os
from vidgear.gears import CamGear


class VisionEngine:
    """
    Motor de Adquisición de Vídeo Híbrido.
    
    Gestiona la conexión con streams remotos (YouTube, RTSP, etc.) vía CamGear
    y archivos de vídeo locales vía OpenCV. Implementa buffering asíncrono 
    y controles de navegación (seek) de manera transparente.
    """

    def __init__(self, source, resolution="720p"):
        self.source = source
        self.resolution = resolution
        self.stream = None
        self.cap = None
        self.is_stream = False
        self._connect(source, self.resolution)

    def _connect(self, source, resolution="720p"):
        """Establece conexión según el tipo de fuente."""
        # Detectar protocolos de stream o URLs
        stream_protocols = ("rtsp://", "rtmp://", "http://", "https://", "udp://")
        is_url = any(source.lower().startswith(p) for p in stream_protocols)
        
        # Si es una URL y no es un archivo local que existe, lo tratamos como stream
        self.is_stream = is_url and not os.path.exists(source)
        
        if self.is_stream:
            # Modo YouTube Stream
            options = {
                "STREAM_RESOLUTION": resolution,
                "THREADED_QUEUE_MODE": True
            }
            try:
                self.stream = CamGear(
                    source=source,
                    stream_mode=True,
                    logging=False,
                    **options
                ).start()
                print(f"Stream conectado: {source}")
            except Exception as e:
                print(f"Error en CamGear: {e}")
                self.stream = None
        else:
            # Modo Archivo Local
            if os.path.exists(source):
                self.cap = cv2.VideoCapture(source)
                if not self.cap.isOpened():
                    print(f"Error al abrir archivo: {source}")
                    self.cap = None
                else:
                    print(f"Archivo local cargado: {source}")
            else:
                print(f"Ruta no válida: {source}")

    def get_frame(self):
        """Lee el siguiente frame de la fuente activa."""
        if self.is_stream and self.stream:
            return self.stream.read()
        elif self.cap:
            ret, frame = self.cap.read()
            return frame if ret else None
        return None

    def get_fps(self):
        """Devuelve los FPS de la fuente."""
        if self.is_stream:
            return 30 # Valor genérico para streams
        if self.cap:
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            return fps if fps > 0 else 30
        return 30

    def seek_back(self, seconds=5):
        """Retrocede la reproducción n segundos (solo para archivos)."""
        if self.is_stream or not self.cap:
            return
        
        fps = self.get_fps()
        current_frame = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        target_frame = max(0, current_frame - (seconds * fps))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        print(f"Rebobinado a frame: {target_frame}")

    def reconnect(self, new_source, resolution=None):
        """Cambia la fuente liberando los recursos previos."""
        self.release()
        self.source = new_source
        if resolution:
            self.resolution = resolution
        self._connect(new_source, self.resolution)

    def is_active(self):
        return (self.stream is not None) or (self.cap is not None and self.cap.isOpened())

    def release(self):
        if self.stream:
            try: self.stream.stop()
            except: pass
            self.stream = None
        if self.cap:
            self.cap.release()
            self.cap = None
