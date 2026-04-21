import cv2
import os
import yt_dlp
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
        self.is_live = False  # Indica si es un directo (donde no se puede navegar)
        self._connect(source, self.resolution)

    def _connect(self, source, resolution="720p"):
        """Establece conexión según el tipo de fuente con detección inteligente de directos."""
        stream_protocols = ("rtsp://", "rtmp://", "http://", "https://", "udp://")
        is_url = any(source.lower().startswith(p) for p in stream_protocols)
        self.is_stream = is_url and not os.path.exists(source)
        
        # Reset de estados
        self.is_live = False
        target_source = source

        if self.is_stream:
            if "youtube.com" in source or "youtu.be" in source:
                # Análisis profundo con yt-dlp
                try:
                    ydl_opts = {'quiet': True, 'no_warnings': True}
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        info = ydl.extract_info(source, download=False)
                        self.is_live = info.get('is_live', False)
                        
                        if not self.is_live:
                            # Es un vídeo grabado: Extraer la mejor URL directa de vídeo para VideoCapture
                            formats = info.get('formats', [])
                            # Filtrar formatos que tengan video y cuya resolución sea cercana a la pedida
                            res_val = int(resolution.replace("p", "")) if "p" in resolution else 720
                            best_url = None
                            for f in formats:
                                if f.get('vcodec') != 'none' and f.get('acodec') != 'none': # Formatos con audio+video son más estables para cv2
                                    best_url = f.get('url')
                                    if f.get('height') and f.get('height') >= res_val:
                                        break
                            if best_url:
                                target_source = best_url
                                print(f"[Engine] YouTube VOD detectado. Usando VideoCapture para permitir navegación.")
                except Exception as e:
                    print(f"[Engine] Error analizando URL con yt-dlp: {e}")
            else:
                # Otros protocolos (RTSP, etc.) se consideran directos si no son archivos locales
                self.is_live = any(p in source.lower() for p in ("rtsp://", "rtmp://", "udp://")) or ("/live/" in source.lower())

            if self.is_live:
                # Modo Stream (CamGear)
                options = {"STREAM_RESOLUTION": resolution, "THREADED_QUEUE_MODE": True}
                try:
                    self.stream = CamGear(source=source, stream_mode=True, logging=False, **options).start()
                    print(f"[Engine] Stream LIVE conectado: {source}")
                except Exception as e:
                    print(f"Error en CamGear: {e}")
                    self.stream = None
            else:
                # Modo VOD Remoto (VideoCapture)
                self.cap = cv2.VideoCapture(target_source)
                if not self.cap.isOpened():
                    print(f"[Engine] Error al abrir VOD remoto: {source}")
                    self.cap = None
        else:
            # Modo Archivo Local
            if os.path.exists(source):
                self.cap = cv2.VideoCapture(source)
                print(f"[Engine] Archivo local cargado: {source}")
            else:
                print(f"[Engine] Ruta no válida o archivo inexistente: {source}")

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
        if self.is_live and self.stream:
            return 30 # Valor genérico para streams en directo
        if self.cap:
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            return fps if fps > 0 else 30
        return 30

    def seek_back(self, seconds=5):
        """Retrocede la reproducción n segundos (solo si NO es directo)."""
        if self.is_live or not self.cap:
            return
        
        fps = self.get_fps()
        current_frame = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        target_frame = max(0, current_frame - (seconds * fps))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        print(f"[Engine] Rebobinado a frame: {target_frame}")

    def seek_forward(self, seconds=5):
        """Avanza la reproducción n segundos (solo si NO es directo)."""
        if self.is_live or not self.cap:
            return
        
        fps = self.get_fps()
        current_frame = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        total_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        
        # Si es un stream VOD (remoto), CAP_PROP_FRAME_COUNT puede no ser fiable,
        # pero para vídeos de YouTube extraídos suele funcionar.
        target_frame = current_frame + (seconds * fps)
        if total_frames > 0:
            target_frame = min(total_frames - 1, target_frame)
            
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        print(f"[Engine] Avance a frame: {target_frame}")

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
