import customtkinter as ctk
import tkinter as tk
import cv2
import numpy as np
import os
import threading
import time
import webbrowser
from PIL import Image, ImageTk
from collections import Counter

# Importaciones locales (Modularización)
from ..core.engine import VisionEngine
from ..core.detector import ObjectDetector
from ..utils.helpers import (
    ZONE_COLORS, 
    ensure_dataset_structure, get_next_capture_filename,
    save_app_config, load_app_config
)
from .components import (
    AnnotationWindow, AddModelPopup, ClassFilterWindow, InfoWindow,
    ModelExplorerWindow
)
from ..utils.painter import VisualPainter
from ..core.events import EventEngine
from .events_window import EventsWindow as EventConfigWindow
from .settings_window import SettingsWindow
from ..utils.logger import DataLogger

class _SplashScreen(ctk.CTkFrame):
    """Pantalla de carga integrada en la ventana principal."""
    def __init__(self, parent):
        super().__init__(parent, fg_color="#1a1c1e")
        self.place(relx=0, rely=0, relwidth=1, relheight=1)

        
        container = ctk.CTkFrame(self, fg_color="transparent")
        container.place(relx=0.5, rely=0.5, anchor="center")
        
        ctk.CTkLabel(container, text="AI VISIONSANDBOX LAB", font=ctk.CTkFont(size=28, weight="bold"), text_color="#0ea5e9").pack(pady=(0, 10))
        
        self.progress = ctk.CTkProgressBar(container, width=300)
        self.progress.pack(pady=10)
        self.progress.set(0)
        
        self.status_label = ctk.CTkLabel(container, text="Cargando sistema...", font=ctk.CTkFont(size=12), text_color="#94a3b8")
        self.status_label.pack(pady=10)
        self.update()

    def set_status(self, text, progress_val=0.0):
        self.status_label.configure(text=text)
        self.progress.set(progress_val)
        self.update()

class VisionApp(ctk.CTk):
    """
    Aplicación Principal: Dashboard de Control de Visión Artificial.
    
    Gestiona el ciclo de vida de la aplicación, el hilo de procesamiento de vídeo,
    la persistencia de configuraciones y la orquestación entre el motor de captura,
    el detector y la capa visual.
    
    Arquitectura:
    - Threading: El procesamiento de vídeo ocurre en un hilo separado para mantener la UI fluida.
    - Persistencia: Carga configuraciones automáticas basadas en la URL del stream.
    - Dashboard: Integra telemetría en tiempo real y gráficos dinámicos.
    """

    def __init__(self):
        super().__init__()
        
        self.title("AI VisionSandbox Lab | Dashboard de Inteligencia")
        self.minsize(960, 640)
        
        # 1. Configuración de Ventana de Carga (Splash visible)
        width, height = 600, 400
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f"{width}x{height}+{x}+{y}")
        
        self.splash = _SplashScreen(self)
        self.splash.set_status("Iniciando motores core...")
        
        # 2. ESTADOS CORE
        self._render_lock = threading.Lock()
        self.raw_frame = None
        self.annotated_frame = None
        self.is_inferencing = False
        self.is_loading_model = False
        self.last_detections = []
        self.is_paused = False
        self.zones = []
        self.current_zone = []
        self.is_drawing_zone = False
        self.target_classes = None
        self.heatmap_enabled = False
        self.heatmap_acc = None
        self.conf_threshold = 0.35
        self.infer_interval = 0.10
        self.last_infer_time = 0
        self.last_infer_timestamp = 0
        self.flash_alpha = 0.0
        self.url = "https://www.youtube.com/watch?v=dfVK7ld38Ys"
        self.locked_track_id = None
        self.focus_lost_cnt = 0

        # 3. MOTORES
        self.data_logger = DataLogger()
        self.event_engine = EventEngine()
        self.detector = ObjectDetector()
        self.engine = VisionEngine(self.url)

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)

        self.splash.set_status("Construyendo interfaz...", 0.4)
        self._build_sidebar()
        self._build_main_area()
        self._build_bottom_dashboard()
        self._load_config()

        self.splash.set_status("Cargando modelo inteligente...", 0.7)
        self._load_lightest_model()
        
        self.splash.set_status("¡Listo!", 1.0)
        self.add_log("Sistema core iniciado.")

        # Transición a Dashboard completo
        self.after(500, self._show_and_start)

    def _show_and_start(self):
        """Finaliza la carga, destruye el splash y expande al dashboard completo."""
        try:
            self.splash.destroy()
            # Maximizar directamente
            self.state('zoomed')
            # Iniciar bucle de video tras breve pausa para estabilización de layout
            self.after(200, self.update_video)
        except Exception:
            self.update_video()

    def _load_lightest_model(self):
        """Busca y carga la arquitectura/alias más ligera disponible."""
        try:
            if not self.detector.architectures:
                return

            # Prioridad: Buscar YOLO11 o YOLO11N, si no el primero que haya
            families = sorted(self.detector.architectures.keys())
            target_family = families[0]
            for f in families:
                if "yolo11" in f.lower():
                    target_family = f
                    break
            
            aliases = self.detector.architectures[target_family]["aliases"]
            target_alias = sorted(aliases.keys())[0] # El 01 suele ser el más ligero
            
            threading.Thread(target=self.change_model, args=(target_family, target_alias), daemon=True).start()
        except Exception as e:
            print(f"[Main] Error seleccionando modelo ligero: {e}")

    def _build_sidebar(self):
        """Construye el panel lateral compacto y sin scroll."""
        self.sidebar = ctk.CTkFrame(self, width=280, corner_radius=0)
        self.sidebar.grid(row=0, column=0, rowspan=2, sticky="nsew")

        # Cabecera
        header_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        header_frame.pack(pady=(20, 15), padx=20, fill="x")
        ctk.CTkLabel(header_frame, text="VISIONSANDBOX LAB", font=ctk.CTkFont(size=20, weight="bold")).pack(side="left")
        
        # Botón Info estilizado como píldora elegante
        self.info_btn = ctk.CTkButton(header_frame, text="ℹ", width=28, height=28, corner_radius=14, 
                                      fg_color="#1e293b", hover_color="#334155", text_color="#38bdf8",
                                      font=ctk.CTkFont(size=14, weight="bold"),
                                      command=lambda: InfoWindow(self))
        self.info_btn.pack(side="right", padx=2)
        
        # Botón Settings (Tuerca)
        self.settings_btn = ctk.CTkButton(header_frame, text="⚙️", width=28, height=28, corner_radius=14, 
                                          fg_color="#1e293b", hover_color="#334155", text_color="#94a3b8",
                                          font=ctk.CTkFont(size=14),
                                          command=self.open_settings)
        self.settings_btn.pack(side="right", padx=2)

        # FUENTE
        self._section("FUENTE DE VÍDEO")
        
        url_row = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        url_row.pack(pady=(0, 5), padx=20, fill="x")
        
        self.url_entry = ctk.CTkEntry(url_row, placeholder_text="URL o Ruta", height=28)
        self.url_entry.insert(0, self.url)
        self.url_entry.pack(side="left", fill="x", expand=True, padx=(0, 5))
        
        # UX: Seleccionar todo al hacer clic en lugar de borrar
        def _select_all(_):
            self.url_entry.focus_set()
            self.url_entry.after(10, lambda: self.url_entry.select_range(0, 'end'))
        
        self.url_entry.bind("<Button-1>", _select_all)
        self.url_entry.bind("<Return>", lambda _: self.change_stream())

        ctk.CTkButton(url_row, text="⚡", width=28, height=28, command=self.change_stream, 
                      fg_color="#1e293b", hover_color="#334155", text_color="#38bdf8",
                      font=ctk.CTkFont(size=14, weight="bold")).pack(side="left", padx=2)
        
        ctk.CTkButton(url_row, text="📁", width=28, height=28, command=self.browse_file,
                      fg_color="#1e293b", hover_color="#334155", text_color="#38bdf8",
                      font=ctk.CTkFont(size=14, weight="bold")).pack(side="left", padx=2)
        
        self.stream_quality_var = ctk.StringVar(value="720p")
        self.stream_quality = ctk.CTkOptionMenu(self.sidebar, values=["360p", "480p", "720p", "1080p"], 
                                                variable=self.stream_quality_var, height=28, command=lambda _: self.change_stream())
        self.stream_quality.pack(pady=(0, 10), padx=20, fill="x")
        
        # MODELO
        self._section("MODELO")
        families = list(self.detector.architectures.keys())
        self.model_selector = ctk.CTkSegmentedButton(self.sidebar, values=families, command=self._on_family_change, height=28)
        self.model_selector.pack(pady=(0, 5), padx=20, fill="x")
        
        self.scale_selector = ctk.CTkSegmentedButton(self.sidebar, command=self._on_config_change, height=28)
        self.scale_selector.pack(pady=(0, 10), padx=20, fill="x")
        # BLOQUE DINÁMICO: PROMPT UNIVERSAL (Solo para YOLO-World)
        self.world_prompt_frame = ctk.CTkFrame(self.sidebar, fg_color="#1e293b", border_width=1, border_color="#38bdf8")
        # No se empaqueta inicialmente (se oculta)
        
        ctk.CTkLabel(self.world_prompt_frame, text="🔍 BÚSQUEDA UNIVERSAL AI", 
                     font=ctk.CTkFont(size=11, weight="bold"), text_color="#38bdf8").pack(pady=(5, 2))
        
        self.world_entry = ctk.CTkEntry(self.world_prompt_frame, placeholder_text="Ej: casco, persona, gato...", height=28)
        self.world_entry.pack(pady=5, padx=10, fill="x")
        self.world_entry.bind("<Return>", lambda _: self.apply_world_prompt())
        
        btn_apply = ctk.CTkButton(self.world_prompt_frame, text="Aplicar Cambios", height=24, 
                                  fg_color="#38bdf8", hover_color="#0ea5e9", text_color="#000",
                                  font=ctk.CTkFont(size=11, weight="bold"),
                                  command=self.apply_world_prompt)
        btn_apply.pack(pady=(0, 10), padx=10, fill="x")

        if families:
            self.model_selector.set(families[0])
            self._on_family_change(families[0])

        # ANÁLISIS (Compacto)
        self._section("ANÁLISIS")
        self.conf_slider = ctk.CTkSlider(self.sidebar, from_=0.01, to=0.99, number_of_steps=98, command=self._on_conf_change)
        self.conf_slider.set(0.35)
        self.conf_slider.pack(pady=(0, 5), padx=20, fill="x")
        self.conf_label = ctk.CTkLabel(self.sidebar, text="Confianza: 35%", font=ctk.CTkFont(size=11))
        self.conf_label.pack(pady=(0, 5), padx=20, anchor="w")

        self.interval_slider = ctk.CTkSlider(self.sidebar, from_=0.0, to=5.0, number_of_steps=50, command=self._on_interval_change)
        self.interval_slider.set(self.infer_interval)
        self.interval_slider.pack(pady=(5, 5), padx=20, fill="x")
        self.interval_label = ctk.CTkLabel(self.sidebar, text=f"Muestreo: {self.infer_interval:.1f}s", font=ctk.CTkFont(size=11))
        self.interval_label.pack(pady=(0, 5), padx=20, anchor="w")

        f_row = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        f_row.pack(fill="x", padx=20, pady=5)
        self.heatmap_switch = ctk.CTkSwitch(f_row, text="Heat", command=self._toggle_heatmap, width=60)
        self.heatmap_switch.pack(side="left")
        ctk.CTkButton(f_row, text="Filtro", command=self.open_class_filter, height=24).pack(side="right", fill="x", expand=True, padx=(10, 0))

        ctk.CTkButton(self.sidebar, text="🔔 Eventos e Hitos", command=self.open_events_config, 
                      fg_color="#ea580c", hover_color="##c2410c", height=32).pack(pady=(5, 5), padx=20, fill="x")

        # CAPTURA DATASET (Inline — sin popup)
        self.capture_frame = ctk.CTkFrame(self.sidebar, fg_color="#1e293b", border_width=1, border_color="#16a34a")
        self.capture_frame.pack(pady=(0, 10), padx=20, fill="x")
        
        ctk.CTkLabel(self.capture_frame, text="📸 CAPTURA DATASET", 
                     font=ctk.CTkFont(size=10, weight="bold"), text_color="#16a34a").pack(pady=(5, 2))
        
        self.capture_entry = ctk.CTkEntry(self.capture_frame, placeholder_text="Nombre: taxi, bache...", height=26)
        self.capture_entry.pack(pady=2, padx=8, fill="x")
        self.capture_entry.bind("<Return>", lambda _: self.take_capture())
        
        cap_btns = ctk.CTkFrame(self.capture_frame, fg_color="transparent")
        cap_btns.pack(pady=(2, 5), padx=8, fill="x")
        
        ctk.CTkButton(cap_btns, text="📸 Capturar", command=self.take_capture,
                      fg_color="#16a34a", hover_color="#15803d", height=26,
                      font=ctk.CTkFont(size=11)).pack(side="left", fill="x", expand=True, padx=(0, 3))
        
        ctk.CTkButton(cap_btns, text="📦 ZIP", command=self.export_dataset_zip,
                      fg_color="#6366f1", hover_color="#4f46e5", height=26, width=55,
                      font=ctk.CTkFont(size=11)).pack(side="right")

        # ZONAS
        self._section("ZONAS")
        z_btns = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        z_btns.pack(pady=(0, 10), padx=20, fill="x")
        self.draw_btn = ctk.CTkButton(z_btns, text="Pintar", command=self.toggle_zone_drawing, width=100, height=28)
        self.draw_btn.pack(side="left", padx=(0, 5))
        ctk.CTkButton(z_btns, text="Borrar", command=self.clear_zones, fg_color="#dc2626", height=28).pack(side="left", fill="x", expand=True)

        # APARIENCIA (Botones de icono ☀️/🌙)
        self._section("APARIENCIA")
        app_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        app_frame.pack(pady=(0, 20), padx=20, fill="x")
        
        self.btn_dark = ctk.CTkButton(app_frame, text="🌙", width=40, height=32, fg_color="#1e293b", hover_color="#334155",
                                      command=lambda: ctk.set_appearance_mode("Dark"))
        self.btn_dark.pack(side="left", expand=True, padx=2)
        
        self.btn_light = ctk.CTkButton(app_frame, text="☀️", width=40, height=32, fg_color="#e2e8f0", hover_color="#cbd5e1", text_color="black",
                                       command=lambda: ctk.set_appearance_mode("Light"))
        self.btn_light.pack(side="left", expand=True, padx=2)

        # SELLO ANTIGRAVITY (Al final de la sidebar, interactivo e inferior izquierda)
        self.sidebar.grid_rowconfigure(20, weight=1) # Espaciador para empujar al fondo
        # UI: Hardware status label (e.g. GPU: Intel, Backend: OpenVINO)
        self.hw_label = ctk.CTkLabel(self.sidebar, text=f"HARDWARE: {self.detector.hardware_diag['gpu_name'][:15]}", font=ctk.CTkFont(size=10), text_color="#10b981")
        self.hw_label.pack(side="bottom", pady=10)

        spacer = ctk.CTkLabel(self.sidebar, text="")
        spacer.pack(expand=True, fill="both")
        
        self.antigravity_btn = ctk.CTkLabel(self.sidebar, text="✨ Creado con Antigravity", 
                                            font=ctk.CTkFont(size=11, slant="italic"), 
                                            text_color="#38bdf8", cursor="hand2")
        self.antigravity_btn.pack(pady=15, padx=20, anchor="sw")
        self.antigravity_btn.bind("<Button-1>", lambda e: webbrowser.open("https://antigravity.google"))

    def _section(self, text):
        ctk.CTkLabel(self.sidebar, text=text, font=ctk.CTkFont(size=11, weight="bold"), text_color="#666").pack(pady=(12, 5), padx=20, anchor="w")

    def _build_bottom_dashboard(self):
        """Dashboard horizontal para analítica y logs."""
        self.dash = ctk.CTkFrame(self, height=200, corner_radius=0, fg_color="#111")
        self.dash.grid(row=1, column=1, sticky="nsew")
        self.dash.grid_columnconfigure((1, 2), weight=2)
        self.dash.grid_columnconfigure(3, weight=3) # Log más ancho
        self.dash.grid_rowconfigure(0, weight=1)

        # 1. Métricas Rápidas
        self.m_frame = ctk.CTkFrame(self.dash, fg_color="transparent")
        self.m_frame.grid(row=0, column=0, padx=20, pady=10, sticky="n")
        
        self.count_label = ctk.CTkLabel(self.m_frame, text="OBJETOS: 0", font=ctk.CTkFont(size=16, weight="bold"), text_color="#0ea5e9")
        self.count_label.pack(pady=(10, 2))
        self.zone_counts_label = ctk.CTkLabel(self.m_frame, text="Global", font=ctk.CTkFont(size=11), text_color="#aaa")
        self.zone_counts_label.pack(pady=2)
        self.infer_label = ctk.CTkLabel(self.m_frame, text="CPU: -- ms", font=ctk.CTkFont(size=10), text_color="#666")
        self.infer_label.pack(pady=2)
        
        ctk.CTkButton(self.m_frame, text="📊 CSV", command=self.export_telemetry, width=80, height=24, font=ctk.CTkFont(size=10)).pack(pady=10)

        # 1. Galería de Evidencias (Nueva Columna 0)
        self.evidence_frame = ctk.CTkFrame(self.dash, width=280, fg_color="#1a1c1e")
        self.evidence_frame.grid(row=0, column=0, padx=10, pady=15, sticky="nsew")
        self.evidence_frame.pack_propagate(False)
        
        ctk.CTkLabel(self.evidence_frame, text="📸 GALERÍA DE EVIDENCIAS", 
                     font=ctk.CTkFont(size=10, weight="bold"), text_color="#38bdf8").pack(pady=5)
        
        self.evidence_scroll = ctk.CTkScrollableFrame(self.evidence_frame, orientation="horizontal", fg_color="transparent")
        self.evidence_scroll.pack(fill="both", expand=True, padx=5, pady=5)
        self.evidence_items = []

        # 2. Gráfico Distribución (Columna 1)
        self.bar_canvas = tk.Canvas(self.dash, height=140, bg="#111", highlightthickness=0)
        self.bar_canvas.grid(row=0, column=1, padx=10, pady=15, sticky="nsew")

        # 3. Gráfico Evolución
        self.line_canvas = tk.Canvas(self.dash, height=140, bg="#111", highlightthickness=0)
        self.line_canvas.grid(row=0, column=2, padx=10, pady=15, sticky="nsew")

        # 4. Logs
        l_frame = ctk.CTkFrame(self.dash, fg_color="transparent")
        l_frame.grid(row=0, column=3, padx=20, pady=15, sticky="nsew")
        ctk.CTkLabel(l_frame, text="EVENTOS DEL SISTEMA", font=ctk.CTkFont(size=10, weight="bold"), text_color="#444").pack(anchor="w")
        self.log_textbox = ctk.CTkTextbox(l_frame, font=ctk.CTkFont(family="Consolas", size=10), height=140, border_color="#222", border_width=1)
        self.log_textbox.pack(fill="both", expand=True)

    def _build_main_area(self):
        """Área central de visualización."""
        self.main_frame = ctk.CTkFrame(self, corner_radius=15, border_width=1, border_color="#333")
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=15, pady=15)
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)

        inner = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        inner.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)
        inner.grid_rowconfigure(0, weight=1)
        inner.grid_columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(inner, bg="#1a1a2e", highlightthickness=0, cursor="crosshair")
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Button-1>", self._on_video_click)
        self.canvas.bind("<Button-3>", self._on_video_right_click)

        self.controls = ctk.CTkFrame(inner, fg_color="transparent", height=50)
        self.controls.grid(row=1, column=0, sticky="ew", padx=20, pady=(10, 20))
        
        self.rewind_btn = ctk.CTkButton(self.controls, text="⏪ -5s", command=lambda: self.engine.seek_back(5), width=70)
        self.rewind_btn.pack(side="left", padx=(0, 5))
        
        self.play_btn = ctk.CTkButton(self.controls, text="⏸ Pausa", command=self.toggle_pause, width=100)
        self.play_btn.pack(side="left", padx=5)

        self.forward_btn = ctk.CTkButton(self.controls, text="+5s ⏩", command=lambda: self.engine.seek_forward(5), width=70)
        self.forward_btn.pack(side="left", padx=(5, 10))

        # Indicador "EN DIRECTO" (Oculto por defecto)
        self.live_indicator = ctk.CTkFrame(self.controls, fg_color="transparent")
        self.live_dot = ctk.CTkLabel(self.live_indicator, text="●", text_color="#ef4444", font=ctk.CTkFont(size=18))
        self.live_dot.pack(side="left", padx=(0, 5))
        self.live_text = ctk.CTkLabel(self.live_indicator, text="EN DIRECTO", font=ctk.CTkFont(size=13, weight="bold"), text_color="#ef4444")
        self.live_text.pack(side="left")
        
        self.live_url_label = ctk.CTkLabel(self.live_indicator, text="", font=ctk.CTkFont(size=11), text_color="#64748b")
        self.live_url_label.pack(side="left", padx=(10, 0))
        
        # Sincronizar UI inicial
        self.after(500, self._update_media_controls)
        self._blink_live_indicator()

    def update_video(self):
        """Bucle de renderizado core con actualización de métricas."""
        t0 = time.time()
        
        # 1. Captura de frame
        if not self.is_paused:
            frame = self.engine.get_frame()
            if frame is not None: 
                self.raw_frame = frame
            elif not self.engine.is_stream:
                self.is_paused = True
                self.play_btn.configure(text="▶ Reproducir")

        # 2. Inferencia (Controlada por frecuencia configurada)
        now = time.time()
        should_infer = (now - self.last_infer_time) >= self.infer_interval
        
        if not self.is_inferencing and not self.is_loading_model and self.raw_frame is not None and should_infer:
            self.last_infer_time = now
            threading.Thread(target=self.run_inference, args=(self.raw_frame.copy(),), daemon=True).start()

        # 3. Renderizado (Sincronización Limpia)
        with self._render_lock:
            display = self.annotated_frame.copy() if self.annotated_frame is not None else (self.raw_frame.copy() if self.raw_frame is not None else None)
        
        if display is not None:
            
            # Capa de Zonas (Polígonos y conteos)
            display = self._draw_zones_overlay(display)
            
            if self.heatmap_enabled: 
                display = self._render_heatmap(display)
            
            self._render_canvas(display)

        # 4. Actualizar Métricas en Sidebar
        self._update_metrics(t0)
        
        # 5. Registrar en Telemetría (CSV) — Solo si hay detecciones activas
        if not self.is_paused and self.raw_frame is not None and self.last_detections:
            zone_data = getattr(self, '_last_zone_counts', [])
            self.data_logger.log(self.last_detections, zone_data)

        # 6. Sincronización
        delay = 10
        if not self.engine.is_stream and not self.is_paused:
            fps = self.engine.get_fps()
            delay = max(1, int(1000 / fps) - int((time.time() - t0) * 1000))
        self.after(delay, self.update_video)

    def _update_metrics(self, t0):
        """Delega la actualización de métricas al pintor visual."""
        self._last_zone_counts = VisualPainter.update_sidebar_metrics(self, t0, self.last_detections, self.zones)

    def add_evidence(self, img_bgr, title, is_ok):
        """Añade una miniatura de evidencia a la galería de la UI."""
        try:
            # 1. Procesar imagen para miniatura
            h, w = img_bgr.shape[:2]
            target_h = 100
            target_w = int(w * (target_h / h))
            small = cv2.resize(img_bgr, (target_w, target_h))
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(rgb)
            img_tk = ImageTk.PhotoImage(img_pil)

            # 2. Crear contenedor de la tarjeta
            border_color = "#16a34a" if is_ok else "#ef4444"
            card = ctk.CTkFrame(self.evidence_scroll, fg_color="#0f172a", border_width=2, border_color=border_color)
            card.pack(side="left", padx=5, pady=2)

            lbl_img = ctk.CTkLabel(card, image=img_tk, text="")
            lbl_img.image = img_tk  # Keep reference
            lbl_img.pack(padx=2, pady=2)
            
            ctk.CTkLabel(card, text=title[:15], font=("", 10)).pack()
            
            # Limitar a las últimas 8 evidencias
            self.evidence_items.insert(0, card)
            if len(self.evidence_items) > 8:
                old = self.evidence_items.pop()
                old.destroy()
        except Exception as e:
            print(f"Error añadiendo evidencia UI: {e}")

    def _draw_zones_overlay(self, frame):
        """Delega el dibujado de zonas al pintor visual."""
        frame = VisualPainter.draw_zones(frame, self.zones, self.last_detections)
        return VisualPainter.draw_live_zone(frame, self.current_zone)

    def _render_heatmap(self, frame):
        """Delega el mapa de calor al pintor visual."""
        result, self.heatmap_acc = VisualPainter.draw_heatmap(frame, self.last_detections, self.heatmap_acc)
        return result

    def _render_canvas(self, frame):
        """Maneja el redimensionamiento y dibujado en el canvas de Tkinter."""
        w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
        if w < 50 or h < 50: return
        fh, fw = frame.shape[:2]
        aspect = fw / fh
        nw, nh = (w, int(w / aspect)) if w / h < aspect else (int(h * aspect), h)
        
        self._display_w, self._display_h = nw, nh
        self._img_offset_x, self._img_offset_y = (w - nw) // 2, (h - nh) // 2

        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        img_tk = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)))
        self.canvas.delete("all")
        self.canvas.create_image(w // 2, h // 2, image=img_tk, anchor="center")
        self.canvas.image = img_tk

    def run_inference(self, frame):
        """Gestiona el hilo de inferencia y actualiza el estado de detección."""
        t_infer = time.time()
        self.is_inferencing = True
        try:
            # 1. Inferencia base (Obtenemos TODOS los objetos para que el clic funcione siempre)
            ann, all_detections = self.detector.detect(frame, zones=self.zones, conf_threshold=self.conf_threshold)
            
            # Guardamos la lista completa para la lógica de clic (Focus Mode)
            self.last_detections = all_detections
            
            # --- LÓGICA MODO FOCUS (FILTRADO) ---
            if self.locked_track_id is not None:
                # Filtrar solo el ID bloqueado
                detections = [d for d in all_detections if d.get("track_id") == self.locked_track_id]
                
                if not detections:
                    # Si no está en este frame, incrementamos contador de pérdida
                    self.focus_lost_cnt += 1
                    if self.focus_lost_cnt > 30: # ~3 segundos a 10fps
                        self.locked_track_id = None
                        self.add_log("Focus Mode deshabilitado (Objetivo perdido).")
                else:
                    self.focus_lost_cnt = 0
                    # RE-PINTADO: Ignoramos el frame anotado por el detector y dibujamos nosotros el foco
                    ann = VisualPainter.draw_detections(frame.copy(), detections, is_focus=True)
            else:
                # Si NO hay foco, filtramos por las clases seleccionadas por el usuario para Eventos/Zonas
                if self.target_classes:
                    detections = [d for d in all_detections if d.get("class_id") in self.target_classes]
                else:
                    detections = all_detections
            
            with self._render_lock:
                self.annotated_frame = ann
            
            # --- EVALUACIÓN DE HITOS / EVENTOS ---
            self.event_engine.update_cumulative_stats(detections)
            def on_evidence(img, msg, ok):
                self.after(0, lambda: self.add_evidence(img, msg, ok))

            self.event_engine.evaluate(detections, frame=frame, 
                                       app_log_callback=self.add_log,
                                       evidence_callback=on_evidence)

            ms = int((time.time() - t_infer) * 1000)
            self.after(0, lambda: self.infer_label.configure(text=f"INFERENCIA: {ms} ms"))
        finally: 
            self.is_inferencing = False

    # Métodos delegados y de gestión
    def toggle_pause(self):
        self.is_paused = not self.is_paused
        self.play_btn.configure(text="▶ Reproducir" if self.is_paused else "⏸ Pausa")

    def change_stream(self):
        src = self.url_entry.get().strip()
        
        # Inteligencia de Fallback: Si el campo está vacío, usar la URL activa
        if not src:
            src = self.url
            self.add_log("Campo vacío. Reintentando con la fuente activa...")
        else:
            self.url = src # Actualizar fuente de verdad

        # Sincronizar el campo de texto (especialmente útil si hubo fallback o cambio de calidad)
        self.url_entry.delete(0, 'end')
        self.url_entry.insert(0, self.url)

        resolution = self.stream_quality_var.get()
        self.add_log(f"Configurando fuente: {os.path.basename(src) if os.path.exists(src) else src[:40]+'...'} ({resolution})")
        
        # Iniciar reconexión en hilo para no bloquear UI
        def _reconnect_and_update():
            self.engine.reconnect(src, resolution=resolution)
            self.after(0, self._update_media_controls)

        threading.Thread(target=_reconnect_and_update, daemon=True).start()

    def _update_media_controls(self):
        """Actualiza la visibilidad de los controles según si es directo o vídeo grabado."""
        is_stream = self.engine.is_stream
        is_live = getattr(self.engine, 'is_live', False)
        
        # Limpiar para reordenar
        self.rewind_btn.pack_forget()
        self.play_btn.pack_forget()
        self.forward_btn.pack_forget()
        self.live_indicator.pack_forget()

        # Botones de navegación (Solo si NO es directo)
        if not is_live:
            self.rewind_btn.pack(side="left", padx=(0, 5))
        
        self.play_btn.pack(side="left", padx=5)
        
        if not is_live:
            self.forward_btn.pack(side="left", padx=(5, 10))

        # Indicador de directo (Solo si es LIVE y es STREAM)
        if is_stream and is_live:
            self.live_indicator.pack(side="left", padx=10)
            
            # Mostrar URL filtrada
            url_text = self.url_entry.get().strip()
            if len(url_text) > 45:
                url_text = url_text[:42] + "..."
            self.live_url_label.configure(text=f"|  {url_text}")
        
        mode_desc = "Streaming LIVE" if is_live else ("YouTube VOD" if is_stream else "Video Local")
        self.add_log(f"Modo {mode_desc} detectado. UI actualizada.")

    def _blink_live_indicator(self):
        """Efecto de parpadeo para el punto rojo del indicador."""
        if hasattr(self, 'live_dot') and self.live_dot.winfo_exists():
            current_color = self.live_dot.cget("text_color")
            new_color = "#ef4444" if current_color != "#ef4444" else "#1a1c1e"
            self.live_dot.configure(text_color=new_color)
        self.after(800, self._blink_live_indicator)

    def browse_file(self):
        path = tk.filedialog.askopenfilename()
        if path:
            self.url = path
            self.url_entry.delete(0, "end")
            self.url_entry.insert(0, path)
            self.change_stream()

    def add_log(self, msg):
        self.log_textbox.insert("end", f"[{time.strftime('%H:%M:%S')}] {msg}\n")
        self.log_textbox.see("end")

    def export_telemetry(self):
        """Copia el archivo de log actual a una ubicación elegida por el usuario."""
        import shutil
        source = self.data_logger.get_log_path()
        if not os.path.exists(source):
            return self.add_log("No hay datos para exportar todavía.")
        
        dest = tk.filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("Archivo CSV", "*.csv")], initialfile=os.path.basename(source))
        if dest:
            shutil.copy2(source, dest)
            self.add_log(f"Datos exportados a: {os.path.basename(dest)}")

    def _section(self, text):
        ctk.CTkLabel(self.sidebar, text=text, font=ctk.CTkFont(size=11, weight="bold"), text_color="#666").pack(pady=(10, 5), padx=20, anchor="w")

    def on_closing(self): self.engine.release(); self.destroy()

    def _save_config(self): save_app_config(self.url, self.zones, self.target_classes)
    def _load_config(self):
        cfg = load_app_config(self.url)
        if cfg: self.zones, self.target_classes = cfg.get("zones", []), cfg.get("target_classes")

    def _on_family_change(self, family):
        if not family: return
        
        # Mostrar u ocultar el bloque de búsqueda universal
        if "world" in family.lower():
            self.world_prompt_frame.pack(pady=10, padx=20, fill="x", after=self.scale_selector)
        else:
            self.world_prompt_frame.pack_forget()

        aliases = list(self.detector.architectures.get(family, {}).get("aliases", {}).keys())
        self.scale_selector.configure(values=aliases)
        if aliases: self.scale_selector.set(aliases[0]); self._on_config_change()

    def apply_world_prompt(self):
        """Aplica el texto del prompt al modelo YOLO-World de forma asíncrona."""
        prompt = self.world_entry.get().strip()
        self.add_log(f"Configurando búsqueda AI: '{prompt}'...")
        
        def task():
            if self.detector.set_world_prompt(prompt):
                self.after(0, lambda: self.add_log("✅ Modelo reconfigurado con éxito."))
            else:
                self.after(0, lambda: self.add_log("❌ Error al aplicar prompt."))
        
        threading.Thread(target=task, daemon=True).start()

    def _on_config_change(self, _=None):
        def load():
            self.is_loading_model = True
            self.detector.change_model(self.model_selector.get(), self.scale_selector.get())
            # Update Hardware Label based on the selected model's backend
            backend_info = "GPU" if getattr(self.detector, 'active_device', 'CPU') == "GPU" else "CPU"
            self.hw_label.configure(text=f"PROCESAMIENTO: {backend_info} ({self.detector.hardware_diag['gpu_vendor']})")
            self.is_loading_model = False
        threading.Thread(target=load, daemon=True).start()

    def _on_model_added(self):
        self.model_selector.configure(values=list(self.detector.architectures.keys()))

    def open_class_filter(self):
        if not self.detector or not self.detector.model:
            self.add_log("No hay un modelo cargado todavía.")
            return
        
        classes = self.detector.get_class_names()
        self.add_log(f"Abriendo filtro: {len(classes)} clases detectadas en el modelo.")
        ClassFilterWindow(self, self.detector, self.target_classes, self._on_filter_applied)

    def _on_filter_applied(self, new_targets):
        self.target_classes = new_targets
        
        # INVALIDACIÓN INMEDIATA: Limpiar frame anotado para forzar refresco visual
        # Esto hace que el video se vea 'limpio' instantáneamente hasta la próxima inferencia.
        self.annotated_frame = None
        
        if self.target_classes is None:
            self.add_log("Filtro retirado: detectando todas las clases.")
        else:
            self.add_log(f"Filtro aplicado: {len(self.target_classes)} clase(s) seleccionada(s).")
            
        # Guardar en config para la próxima vez
        self._save_config()

    def open_events_config(self):
        """Abre la ventana de configuración del motor de Hitos/Eventos."""
        classes = []
        if self.detector and self.detector.model and hasattr(self.detector.model, 'names'):
            classes = list(self.detector.model.names.values())
        
        EventConfigWindow(self, self.event_engine, classes, len(self.zones))
        
    def open_settings(self):
        """Abre la ventana de ajustes generales de la aplicación."""
        SettingsWindow(self, self.event_engine, self.detector)
        
    def toggle_zone_drawing(self):
        self.is_drawing_zone = not self.is_drawing_zone
        self.draw_btn.configure(text="Listo" if self.is_drawing_zone else "Dibujar")

    def clear_zones(self): self.zones = []; self._save_config()
    def _on_video_click(self, e): 
        if self.is_drawing_zone: 
            self.current_zone.append(((e.x-self._img_offset_x)/self._display_w, (e.y-self._img_offset_y)/self._display_h))
            return

        # Si no estamos dibujando zonas, intentamos "Lock-on" a un objeto
        if not self.last_detections:
            self.locked_track_id = None
            return

        # Convertir clic a coordenadas de frame original
        try:
            h_frame, w_frame = self.raw_frame.shape[:2]
            nx = (e.x - self._img_offset_x) / self._display_w
            ny = (e.y - self._img_offset_y) / self._display_h
            fx, fy = nx * w_frame, ny * h_frame

            # Si ya hay uno bloqueado, cualquier clic fuera o en otro sitio libera el bloqueo
            # a menos que cliquemos específicamente en otro objeto.
            found_new = None
            for d in self.last_detections:
                x1, y1, x2, y2 = d["bbox"]
                # Añadimos un pequeño margen de 10px para facilitar el clic en objetos pequeños
                if (x1 - 10) <= fx <= (x2 + 10) and (y1 - 10) <= fy <= (y2 + 10):
                    if d.get("track_id") is not None:
                        found_new = d["track_id"]
                        break
            
            if found_new is not None:
                if getattr(self, 'locked_track_id', None) == found_new:
                    self.locked_track_id = None 
                    self.add_log("Focus Mode: Objetivo liberado.")
                else:
                    self.locked_track_id = found_new
                    self.focus_lost_cnt = 0
                    self.add_log(f"Focus Mode: Fijado objetivo ID {self.locked_track_id}")
            else:
                if getattr(self, 'locked_track_id', None) is not None:
                    self.locked_track_id = None
                    self.add_log("Focus Mode: Deshabilitado (clic fuera).")
        except Exception as _e:
            print(f"Error detectando objeto por clic: {_e}")

    def _on_video_right_click(self, e):
        if self.is_drawing_zone and len(self.current_zone) >= 3:
            self.zones.append(self.current_zone); self.current_zone = []; self.toggle_zone_drawing(); self._save_config()

    def _on_conf_change(self, value):
        self.conf_threshold = value
        self.conf_label.configure(text=f"Confianza: {int(value * 100)}%")

    def _on_interval_change(self, value):
        self.infer_interval = value
        if value == 0:
            self.interval_label.configure(text="Muestreo: Cada frame")
        else:
            self.interval_label.configure(text=f"Muestreo: {value:.1f}s")

    def _toggle_heatmap(self): self.heatmap_enabled = self.heatmap_switch.get()

    # --- LÓGICA DE CAPTURA PARA DATASETS ---

    def take_capture(self):
        """Captura instantánea: pausa, congela el frame actual, y abre el anotador."""
        if self.raw_frame is None:
            self.add_log("No hay video para capturar.")
            return

        class_name = self.capture_entry.get().strip().lower().replace(" ", "_")
        if not class_name:
            self.add_log("⚠️ Escribe un nombre de clase antes de capturar.")
            self.capture_entry.focus_set()
            return

        # Congelar frame ANTES de pausar (captura instantánea del momento exacto)
        captured_frame = self.raw_frame.copy()
        
        # Pausar si no lo está
        self._was_paused_before_capture = self.is_paused
        if not self.is_paused:
            self.toggle_pause()

        try:
            dataset_dir = ensure_dataset_structure(class_name)
            base_name = get_next_capture_filename(class_name, dataset_dir)
            
            # Abrir ventana de anotación con el frame congelado
            AnnotationWindow(self, captured_frame, class_name, base_name, dataset_dir, 
                           self._on_capture_saved, self._on_capture_cancelled)
        except Exception as e:
            self.add_log(f"Error preparando captura: {e}")
            if not self._was_paused_before_capture:
                self.toggle_pause()

    def _on_capture_cancelled(self):
        """Callback si el usuario cancela la anotación."""
        if not self._was_paused_before_capture and self.is_paused:
            self.toggle_pause()

    def _on_capture_saved(self, name, boxes_count):
        """Callback cuando el usuario guarda la anotación."""
        self.add_log(f"✅ Captura guardada: {name} ({boxes_count} bboxes).")
        # Reanudar si estaba reproduciéndose antes
        if not self._was_paused_before_capture and self.is_paused:
            self.toggle_pause()

    def export_dataset_zip(self):
        """Exporta todos los datasets en un archivo ZIP listo para entrenar."""
        import zipfile
        from ..utils.helpers import DATASETS_DIR
        
        if not os.path.exists(DATASETS_DIR) or not os.listdir(DATASETS_DIR):
            self.add_log("No hay datasets para exportar.")
            return
        
        dest = tk.filedialog.asksaveasfilename(
            defaultextension=".zip",
            filetypes=[("Archivo ZIP", "*.zip")],
            initialfile="dataset_export.zip"
        )
        if not dest:
            return
        
        try:
            with zipfile.ZipFile(dest, 'w', zipfile.ZIP_DEFLATED) as zf:
                for root, dirs, files in os.walk(DATASETS_DIR):
                    for file in files:
                        full_path = os.path.join(root, file)
                        arcname = os.path.relpath(full_path, DATASETS_DIR)
                        zf.write(full_path, arcname)
            
            self.add_log(f"📦 Dataset exportado: {os.path.basename(dest)}")
        except Exception as e:
            self.add_log(f"Error exportando dataset: {e}")


