import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox
import cv2
import numpy as np
import os
import threading
import time
import webbrowser
import platform
import shutil
from PIL import Image, ImageTk
from collections import Counter

# Importaciones locales (Modularización)
from ..core.engine import VisionEngine
from ..core.detector import ObjectDetector
from ..utils.helpers import (
    ZONE_COLORS, DATASETS_DIR,
    ensure_dataset_structure, get_next_capture_filename,
    save_app_config, load_app_config
)
from .components import (
    AnnotationWindow, AddModelPopup, ClassFilterWindow, InfoWindow,
    ModelExplorerWindow, SourceSelectorWindow, FavoritesWindow
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
    """Aplicación Principal: Dashboard de Control de Visión Artificial."""

    def __init__(self):
        super().__init__()
        self.NONE_MODEL = "Sin Modelo"
        
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
        self._checked_datasets = set() # Rastreo de datasets validados en esta sesión
        self.is_auto_capturing = False # Estado de autocaptura periódica
        
        # Telemetría de Sesión
        self.total_detections_ever = 0
        self.session_start_time = time.time()
        self.session_seen_ids = set() # IDs únicos vistos en esta sesión
        self.session_class_counts = Counter() # Conteo por clase de IDs únicos
        self.bar_chart_mode = "General" # Modo de la gráfica: General, Z1, Z2...

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
        
        self.splash.set_status("Listo.", 1.0)
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
        """Inicializa el selector en 'Sin Modelo' para un arranque instantáneo."""
        try:
            self._on_no_model_click()
        except Exception as e:
            print(f"[Main] Error en inicialización de modelo: {e}")

    def _build_sidebar(self):
        """Construye el panel lateral compacto y sin scroll."""
        self.sidebar = ctk.CTkFrame(self, width=280, corner_radius=0)
        self.sidebar.grid(row=0, column=0, rowspan=2, sticky="nsew")

        # Cabecera
        header_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        header_frame.pack(pady=(12, 8), padx=20, fill="x")
        ctk.CTkLabel(header_frame, text="VISIONSANDBOX LAB", font=ctk.CTkFont(size=18, weight="bold")).pack(side="left")
        
        # Botón Info estilizado como píldora elegante
        self.info_btn = ctk.CTkButton(header_frame, text="\uE946", width=28, height=28, corner_radius=14, 
                                      fg_color="#1e293b", hover_color="#334155", text_color="#38bdf8",
                                      font=ctk.CTkFont(family="Segoe MDL2 Assets", size=14, weight="bold"),
                                      command=lambda: InfoWindow(self))
        self.info_btn.pack(side="right", padx=2)
        
        # Botón Settings (Tuerca)
        self.settings_btn = ctk.CTkButton(header_frame, text="\uE713", width=28, height=28, corner_radius=14, 
                                          fg_color="#1e293b", hover_color="#334155", text_color="#94a3b8",
                                          font=ctk.CTkFont(family="Segoe MDL2 Assets", size=14),
                                          command=self.open_settings)
        self.settings_btn.pack(side="right", padx=2)

        # FUENTE
        self._section("FUENTE DE VÍDEO")
        
        src_row = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        src_row.pack(fill="x", padx=20, pady=(0, 6))
        
        self.source_btn = ctk.CTkButton(src_row, text="Seleccionar Fuente...", height=35, 
                                        fg_color="#1e293b", hover_color="#334155", text_color="#38bdf8",
                                        font=ctk.CTkFont(size=13, weight="bold"),
                                        command=self.open_source_selector)
        self.source_btn.pack(side="left", fill="x", expand=True, padx=(0, 5))

        self.fav_btn = ctk.CTkButton(src_row, text="\uE735", width=35, height=35, 
                                     fg_color="#1e293b", hover_color="#334155", text_color="#facc15",
                                     font=ctk.CTkFont(family="Segoe MDL2 Assets", size=16),
                                     command=lambda: FavoritesWindow(self, self.change_stream))
        self.fav_btn.pack(side="right")
        
        self.stream_quality_var = ctk.StringVar(value="720p")
        self.stream_quality = ctk.CTkOptionMenu(self.sidebar, values=["360p", "480p", "720p", "1080p"], 
                                                variable=self.stream_quality_var, height=28, command=lambda _: self.change_stream())
        self.stream_quality.pack(pady=(0, 6), padx=20, fill="x")
        
        # MODELO
        self._section("MODELO")
        
        # Fila 1: Botón Maestro de Desactivación
        self.no_model_selector = ctk.CTkSegmentedButton(self.sidebar, values=["DESACTIVAR PROCESAMIENTO IA"], 
                                                       command=lambda _: self._on_no_model_click(), height=30)
        self.no_model_selector.pack(pady=(0, 6), padx=20, fill="x")
        
        # Fila 2: Familias
        families = list(self.detector.architectures.keys())
        self.model_selector = ctk.CTkSegmentedButton(self.sidebar, values=families, command=self._on_family_change, height=26)
        self.model_selector.pack(pady=(0, 4), padx=20, fill="x")
        
        # Fila 3: Escalas
        self.scale_selector = ctk.CTkSegmentedButton(self.sidebar, command=self._on_config_change, height=26)
        self.scale_selector.pack(pady=(0, 6), padx=20, fill="x")
        # BLOQUE DINÁMICO: PROMPT UNIVERSAL (Solo para YOLO-World)
        self.world_prompt_frame = ctk.CTkFrame(self.sidebar, fg_color="#1e293b", border_width=1, border_color="#38bdf8")
        
        ctk.CTkLabel(self.world_prompt_frame, text="BUSQUEDA UNIVERSAL AI", 
                     font=ctk.CTkFont(size=11, weight="bold"), text_color="#38bdf8").pack(pady=(5, 2))
        
        self.world_entry = ctk.CTkEntry(self.world_prompt_frame, placeholder_text="Ej: casco, persona, gato...", height=28)
        self.world_entry.pack(pady=5, padx=10, fill="x")
        self.world_entry.bind("<Return>", lambda _: self.apply_world_prompt())
        
        btn_apply = ctk.CTkButton(self.world_prompt_frame, text="Aplicar Cambios", height=24, 
                                  fg_color="#38bdf8", hover_color="#0ea5e9", text_color="#000",
                                  font=ctk.CTkFont(size=11, weight="bold"),
                                  command=self.apply_world_prompt)
        btn_apply.pack(pady=(0, 10), padx=10, fill="x")

        # ANÁLISIS (Compacto)
        self._section("ANÁLISIS")
        self.conf_slider = ctk.CTkSlider(self.sidebar, from_=0.01, to=0.99, number_of_steps=98, command=self._on_conf_change)
        self.conf_slider.set(0.35)
        self.conf_slider.pack(pady=(0, 2), padx=20, fill="x")
        self.conf_label = ctk.CTkLabel(self.sidebar, text="Confianza: 35%", font=ctk.CTkFont(size=11))
        self.conf_label.pack(pady=(0, 2), padx=20, anchor="w")

        self.interval_slider = ctk.CTkSlider(self.sidebar, from_=0.0, to=5.0, number_of_steps=50, command=self._on_interval_change)
        self.interval_slider.set(self.infer_interval)
        self.interval_slider.pack(pady=(2, 2), padx=20, fill="x")
        self.interval_label = ctk.CTkLabel(self.sidebar, text=f"Muestreo: {self.infer_interval:.1f}s", font=ctk.CTkFont(size=11))
        self.interval_label.pack(pady=(0, 4), padx=20, anchor="w")

        f_row = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        f_row.pack(fill="x", padx=20, pady=5)
        self.heatmap_switch = ctk.CTkSwitch(f_row, text="Heat", command=self._toggle_heatmap, width=60)
        self.heatmap_switch.pack(side="left")
        ctk.CTkButton(f_row, text="Filtro", command=self.open_class_filter, height=24).pack(side="right", fill="x", expand=True, padx=(10, 0))

        ctk.CTkButton(self.sidebar, text="Eventos e Hitos", command=self.open_events_config, 
                      fg_color="#ea580c", hover_color="#c2410c", height=28).pack(pady=(2, 4), padx=20, fill="x")

        # CAPTURA DATASET (Compacto: entrada + botones en una sola fila)
        cap_row = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        cap_row.pack(fill="x", padx=20, pady=(0, 4))
        
        self.capture_entry = ctk.CTkEntry(cap_row, placeholder_text="Dataset: Acuario, Trafico...", height=26)
        self.capture_entry.pack(side="left", fill="x", expand=True, padx=(0, 4))
        self.capture_entry.bind("<Return>", lambda _: self.take_capture())
        
        ctk.CTkButton(cap_row, text="CAP", command=self.take_capture,
                      fg_color="#16a34a", hover_color="#15803d", height=26, width=32).pack(side="left", padx=(0, 2))
        
        ctk.CTkButton(cap_row, text="RES", command=self.resume_labeling,
                      fg_color="#0ea5e9", hover_color="#0284c7", height=26, width=32).pack(side="left", padx=(0, 2))

        ctk.CTkButton(cap_row, text="ZIP", command=self.import_zip_dataset,
                      fg_color="#f59e0b", hover_color="#d97706", height=26, width=32).pack(side="left", padx=(0, 2))
        
        ctk.CTkButton(cap_row, text="EXP", command=self.export_dataset_zip,
                      fg_color="#6366f1", hover_color="#4f46e5", height=26, width=32).pack(side="left")

        # AUTOCAPTURA
        self._section("AUTOCAPTURA")
        auto_cap_row = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        auto_cap_row.pack(fill="x", padx=20, pady=(0, 4))
        
        self.auto_capture_interval = ctk.CTkEntry(auto_cap_row, placeholder_text="Seg (Ej: 5)", height=26, width=60)
        self.auto_capture_interval.pack(side="left", padx=(0, 5))
        self.auto_capture_interval.insert(0, "5")
        
        self.auto_capture_btn = ctk.CTkButton(auto_cap_row, text="Auto", command=self.toggle_auto_capture,
                                             fg_color="#10b981", hover_color="#059669", height=26)
        self.auto_capture_btn.pack(side="left", fill="x", expand=True)

        # ZONAS
        self._section("ZONAS")
        z_btns = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        z_btns.pack(pady=(0, 4), padx=20, fill="x")
        self.draw_btn = ctk.CTkButton(z_btns, text="Pintar", command=self.toggle_zone_drawing, width=100, height=28)
        self.draw_btn.pack(side="left", padx=(0, 5))
        ctk.CTkButton(z_btns, text="Borrar", command=self.clear_zones, fg_color="#dc2626", height=28).pack(side="left", fill="x", expand=True)

        # --- BLOQUE INFERIOR FIJO (Tema + Hardware) ---
        bottom = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        bottom.pack(side="bottom", fill="x", padx=20, pady=(2, 6))
        
        # Hardware
        self.hw_label = ctk.CTkLabel(bottom, 
                                     text=f"{self.detector.hardware_diag['gpu_name'][:20]} | {self.detector.hardware_diag['best_backend'].upper()}", 
                                     font=ctk.CTkFont(size=10), text_color="#10b981")
        self.hw_label.pack(fill="x", pady=(0, 6))
        
        # Tema + Créditos en una fila
        theme_row = ctk.CTkFrame(bottom, fg_color="transparent")
        theme_row.pack(fill="x")
        
        ctk.CTkButton(theme_row, text="Dark", width=40, height=26, fg_color="#1e293b", hover_color="#334155",
                      command=lambda: ctk.set_appearance_mode("Dark")).pack(side="left", padx=(0, 3))
        ctk.CTkButton(theme_row, text="Light", width=40, height=26, fg_color="#e2e8f0", hover_color="#cbd5e1", text_color="black",
                      command=lambda: ctk.set_appearance_mode("Light")).pack(side="left", padx=(0, 8))
        
        self.antigravity_btn = ctk.CTkLabel(theme_row, text="Antigravity", 
                                            font=ctk.CTkFont(size=10, slant="italic"), 
                                            text_color="#38bdf8", cursor="hand2")
        self.antigravity_btn.pack(side="right")
        self.antigravity_btn.bind("<Button-1>", lambda e: webbrowser.open("https://antigravity.google"))

    def _section(self, text):
        ctk.CTkLabel(self.sidebar, text=text, font=ctk.CTkFont(size=11, weight="bold"), text_color="#666").pack(pady=(8, 3), padx=20, anchor="w")

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
        
        ctk.CTkButton(self.m_frame, text="CSV", command=self.export_telemetry, width=80, height=24, font=ctk.CTkFont(size=10)).pack(pady=10)

        # 1. Galería de Evidencias (Nueva Columna 0)
        self.evidence_frame = ctk.CTkFrame(self.dash, width=280, fg_color="#1a1c1e")
        self.evidence_frame.grid(row=0, column=0, padx=10, pady=15, sticky="nsew")
        self.evidence_frame.pack_propagate(False)
        
        ctk.CTkLabel(self.evidence_frame, text="GALERIA DE EVIDENCIAS", 
                     font=ctk.CTkFont(size=10, weight="bold"), text_color="#38bdf8").pack(pady=5)
        
        self.evidence_scroll = ctk.CTkScrollableFrame(self.evidence_frame, orientation="horizontal", fg_color="transparent")
        self.evidence_scroll.pack(fill="both", expand=True, padx=5, pady=5)
        self.evidence_items = []

        # 2. Gráfico Distribución (Columna 1)
        self.bar_frame = ctk.CTkFrame(self.dash, fg_color="transparent")
        self.bar_frame.grid(row=0, column=1, padx=10, pady=15, sticky="nsew")
        
        # Selector de Modo de Gráfica
        self.bar_modes_frame = ctk.CTkFrame(self.bar_frame, fg_color="transparent", height=25)
        self.bar_modes_frame.pack(fill="x", pady=(0, 5))
        self._update_bar_mode_buttons()

        self.bar_canvas = tk.Canvas(self.bar_frame, height=120, bg="#111", highlightthickness=0)
        self.bar_canvas.pack(fill="both", expand=True)

        # 3. Telemetría de Sesión (Columna 2)
        self.telemetry_frame = ctk.CTkFrame(self.dash, height=140, fg_color="#111")
        self.telemetry_frame.grid(row=0, column=2, padx=10, pady=15, sticky="nsew")
        
        ctk.CTkLabel(self.telemetry_frame, text="RESUMEN DE SESION", 
                     font=ctk.CTkFont(size=9, weight="bold"), text_color="#444").pack(pady=(10, 5))
        
        self.total_ever_label = ctk.CTkLabel(self.telemetry_frame, text="0", 
                                            font=ctk.CTkFont(size=32, weight="bold"), text_color="#10b981")
        self.total_ever_label.pack()
        ctk.CTkLabel(self.telemetry_frame, text="TOTAL DETECTADOS", 
                     font=ctk.CTkFont(size=9, weight="bold"), text_color="#64748b").pack(pady=(0, 10))
        
        self.uptime_label = ctk.CTkLabel(self.telemetry_frame, text="00:00:00", 
                                        font=ctk.CTkFont(size=18, weight="bold"), text_color="#94a3b8")
        self.uptime_label.pack()
        ctk.CTkLabel(self.telemetry_frame, text="TIEMPO ACTIVO", 
                     font=ctk.CTkFont(size=9, weight="bold"), text_color="#444").pack(pady=(0, 5))

        self.breakdown_label = ctk.CTkLabel(self.telemetry_frame, text="", 
                                           font=ctk.CTkFont(size=10), text_color="#64748b", justify="left")
        self.breakdown_label.pack(pady=5)

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
        
        self.rewind_btn = ctk.CTkButton(self.controls, text="\uE892", width=40, height=32, 
                                        fg_color="#1e293b", hover_color="#334155", text_color="#e2e8f0",
                                        font=ctk.CTkFont(family="Segoe MDL2 Assets", size=14),
                                        command=lambda: self.engine.seek_back(5))
        self.rewind_btn.pack(side="left", padx=(0, 5))
        
        self.play_btn = ctk.CTkButton(self.controls, text="\uE769", width=40, height=32, 
                                      fg_color="#38bdf8", hover_color="#0ea5e9", text_color="#0f172a",
                                      font=ctk.CTkFont(family="Segoe MDL2 Assets", size=14, weight="bold"),
                                      command=self.toggle_pause)
        self.play_btn.pack(side="left", padx=5)

        self.forward_btn = ctk.CTkButton(self.controls, text="\uE893", width=40, height=32, 
                                         fg_color="#1e293b", hover_color="#334155", text_color="#e2e8f0",
                                         font=ctk.CTkFont(family="Segoe MDL2 Assets", size=14),
                                         command=lambda: self.engine.seek_forward(5))
        self.forward_btn.pack(side="left", padx=(5, 10))

        # Indicador "EN DIRECTO" (Oculto por defecto)
        self.live_indicator = ctk.CTkFrame(self.controls, fg_color="transparent")
        self.live_dot = ctk.CTkLabel(self.live_indicator, text="*", text_color="#ef4444", font=ctk.CTkFont(size=18))
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
        if hasattr(self, 'is_labeling_mode') and self.is_labeling_mode:
            # En modo etiquetado no actualizamos vídeo para ahorrar 100% recursos
            self.after(500, self.update_video)
            return

        if not self.is_paused:
            frame = self.engine.get_frame()
            if frame is not None: 
                self.raw_frame = frame
            elif not self.engine.is_stream:
                self.is_paused = True
                self.play_btn.configure(text="Reproducir")

        if self.raw_frame is None:
            self.after(30, self.update_video)
            return

        # 2. Inferencia (Controlada por frecuencia configurada)
        now = time.time()
        should_infer = (now - self.last_infer_time) >= self.infer_interval
        
        if self.detector.model and not self.is_inferencing and not self.is_loading_model and should_infer:
            self.last_infer_time = now
            threading.Thread(target=self.run_inference, args=(self.raw_frame.copy(),), daemon=True).start()

        # 3. Renderizado (Sincronización Limpia)
        # Si no hay modelo, usamos el frame raw para ahorrar CPU, 
        # pero copiamos solo si hay capas adicionales (Zonas/Heatmap) que lo modifiquen.
        has_layers = bool(self.zones or self.heatmap_enabled)
        
        if self.detector.model is None:
            display = self.raw_frame.copy() if has_layers else self.raw_frame
            self.annotated_frame = None 
            self.last_detections = []    
        else:
            with self._render_lock:
                if self.annotated_frame is not None:
                    display = self.annotated_frame.copy()
                else:
                    display = self.raw_frame.copy()
        
        if display is not None:
            # Capas adicionales solo sobre una copia o si es el frame raw y se va a refrescar
            if has_layers:
                display = self._draw_zones_overlay(display)
                if self.heatmap_enabled: 
                    display = self._render_heatmap(display)
            
            self._render_canvas(display)

        # 4. Actualizar Métricas en Sidebar (Ahora limitado internamente a 5 FPS)
        self._update_metrics(t0)
        
        # 5. Registrar en Telemetría (CSV) — Solo si hay detecciones activas
        if not self.is_paused and self.raw_frame is not None and self.last_detections:
            zone_data = getattr(self, '_last_zone_counts', [])
            self.data_logger.log(self.last_detections, zone_data)

        # 6. Sincronización Inteligente
        # Calculamos el tiempo real que ha tomado el frame para ajustar el próximo
        elapsed_ms = (time.time() - t0) * 1000
        target_delay = 33 # Objetivo ~30 FPS para suavidad UI
        
        if not self.engine.is_stream and not self.is_paused:
            fps = self.engine.get_fps()
            target_delay = 1000 / fps
            
        next_delay = max(1, int(target_delay - elapsed_ms))
        self.after(next_delay, self.update_video)

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

    def _update_bar_mode_buttons(self):
        """Actualiza los botones de modo de la gráfica según las zonas disponibles."""
        for w in self.bar_modes_frame.winfo_children():
            w.destroy()
            
        modes = ["General"] + [f"Z{i+1}" for i in range(len(self.zones))]
        for m in modes:
            is_active = (self.bar_chart_mode == m)
            btn = ctk.CTkButton(self.bar_modes_frame, text=m, width=50, height=20,
                                font=ctk.CTkFont(size=9, weight="bold"),
                                fg_color="#38bdf8" if is_active else "#1e293b",
                                text_color="#0f172a" if is_active else "#94a3b8",
                                command=lambda mode=m: self._set_bar_mode(mode))
            btn.pack(side="left", padx=2)

    def _set_bar_mode(self, mode):
        self.bar_chart_mode = mode
        self._update_bar_mode_buttons()

    def _draw_zones_overlay(self, frame):
        """Delega el dibujado de zonas al pintor visual."""
        frame = VisualPainter.draw_zones(frame, self.zones, self.last_detections)
        return VisualPainter.draw_live_zone(frame, self.current_zone)

    def _render_heatmap(self, frame):
        """Delega el mapa de calor al pintor visual."""
        result, self.heatmap_acc = VisualPainter.draw_heatmap(frame, self.last_detections, self.heatmap_acc)
        return result

    def _render_canvas(self, frame):
        """Maneja el redimensionamiento y dibujado en el canvas de Tkinter de forma optimizada."""
        w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
        if w < 50 or h < 50: return
        
        fh, fw = frame.shape[:2]
        aspect = fw / fh
        nw, nh = (w, int(w / aspect)) if w / h < aspect else (int(h * aspect), h)
        
        self._display_w, self._display_h = nw, nh
        self._img_offset_x, self._img_offset_y = (w - nw) // 2, (h - nh) // 2

        # Optimización: Solo redimensionar si es necesario y usar INTER_NEAREST para velocidad
        if (nw, nh) != (fw, fh):
            resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_NEAREST)
        else:
            resized = frame
            
        img_tk = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)))
        
        # Optimización: Usar itemconfig en lugar de borrar todo el canvas
        if not hasattr(self, '_canvas_img_id'):
            self._canvas_img_id = self.canvas.create_image(w // 2, h // 2, image=img_tk, anchor="center")
        else:
            self.canvas.itemconfig(self._canvas_img_id, image=img_tk)
            self.canvas.coords(self._canvas_img_id, w // 2, h // 2)
            
        self.canvas.image = img_tk

    def run_inference(self, frame):
        """Gestiona el hilo de inferencia y actualiza el estado de detección."""
        t_infer = time.time()
        self.is_inferencing = True
        try:
            # 1. Inferencia base (Si hay foco, detectamos todo para no perder el rastro, si no, usamos el filtro)
            t_classes = None if self.locked_track_id is not None else self.target_classes
            ann, all_detections = self.detector.detect(frame, target_classes=t_classes, zones=self.zones, conf_threshold=self.conf_threshold)
            
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
                if self.target_classes is not None:
                    detections = [d for d in all_detections if d.get("class_id") in self.target_classes]
                else:
                    detections = all_detections
            
            with self._render_lock:
                self.annotated_frame = ann
            
            # --- EVALUACIÓN DE HITOS / EVENTOS ---
            for d in detections:
                tid = d.get("track_id")
                if tid is not None and tid not in self.session_seen_ids:
                    self.session_seen_ids.add(tid)
                    self.session_class_counts[d['label']] += 1
                    self.total_detections_ever = len(self.session_seen_ids)
            
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
        self.play_btn.configure(text="Reproducir" if self.is_paused else "Pausa")

    def change_stream(self, new_url=None):
        if new_url:
            self.url = new_url
            
        # Resetear telemetría de sesión
        self.total_detections_ever = 0
        self.session_start_time = time.time()
        self.session_seen_ids = set()
        self.session_class_counts = Counter()
            
        resolution = self.stream_quality_var.get()
        self.add_log(f"Configurando fuente: {os.path.basename(self.url) if os.path.exists(self.url) else self.url[:40]+'...'} ({resolution})")
        
        # Iniciar reconexión en hilo para no bloquear UI
        def _reconnect_and_update():
            self.engine.reconnect(self.url, resolution=resolution)
            self.after(0, self._update_media_controls)

        threading.Thread(target=_reconnect_and_update, daemon=True).start()

    def open_source_selector(self):
        """Abre la ventana de selección de fuente de vídeo."""
        SourceSelectorWindow(self, self.url, self.change_stream)

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

        # Indicador de fuente y URL
        self.live_indicator.pack(side="left", padx=10)
        
        if is_live:
            self.live_dot.pack(side="left", padx=(0, 5))
            self.live_dot.configure(font=ctk.CTkFont(family="Segoe MDL2 Assets", size=10))
            self.live_text.configure(text="EN DIRECTO", text_color="#ef4444")
            self.live_text.pack(side="left")
        else:
            self.live_dot.pack_forget()
            self.live_text.configure(text="VOD", text_color="#64748b")
            self.live_text.pack(side="left")
            
        # Mostrar URL filtrada
        url_text = self.url
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

    def add_log(self, msg):
        """Añade un mensaje al log de forma segura desde cualquier hilo."""
        def _task():
            try:
                if hasattr(self, 'log_textbox') and self.log_textbox.winfo_exists():
                    self.log_textbox.insert("end", f"[{time.strftime('%H:%M:%S')}] {msg}\n")
                    self.log_textbox.see("end")
            except: pass
        self.after(0, _task)

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

    def on_closing(self): self.engine.release(); self.destroy()

    def _save_config(self): save_app_config(self.url, self.zones, self.target_classes)
    def _load_config(self):
        cfg = load_app_config(self.url)
        if cfg: self.zones, self.target_classes = cfg.get("zones", []), cfg.get("target_classes")

    def _update_world_prompt_visibility(self, visible):
        if visible:
            self.world_prompt_frame.pack(pady=10, padx=20, fill="x", after=self.scale_selector)
        else:
            self.world_prompt_frame.pack_forget()

    def _on_no_model_click(self):
        """Desactiva la IA y limpia selecciones de modelos."""
        self.no_model_selector.set("DESACTIVAR PROCESAMIENTO IA")
        self.model_selector.set("")
        self.scale_selector.configure(values=[])
        self.scale_selector.set("")
        self.detector.model = None
        self.detector.active_name = None
        self.add_log("IA Desactivada. Renderizado en crudo.")
        self._update_world_prompt_visibility(False)

    def _on_family_change(self, family):
        """Actualiza el selector de escalas según la familia."""
        if not family: return
        
        # Desactivar el botón de "Sin Modelo" si se elige una familia
        self.no_model_selector.set("")

        # Mostrar u ocultar el bloque de búsqueda universal
        self._update_world_prompt_visibility("world" in family.lower())

        scales = sorted(self.detector.architectures.get(family, {}).get("aliases", {}).keys())
        self.scale_selector.configure(values=scales)
        if scales: 
            self.scale_selector.set(scales[0])
            self._on_config_change()

    def apply_world_prompt(self):
        """Aplica el texto del prompt al modelo YOLO-World de forma asíncrona."""
        prompt = self.world_entry.get().strip()
        self.add_log(f"Configurando búsqueda AI: '{prompt}'...")
        
        def task():
            if self.detector.set_world_prompt(prompt):
                self.after(0, lambda: self.add_log("Modelo reconfigurado con exito."))
            else:
                self.after(0, lambda: self.add_log("Error al aplicar prompt."))
        
        threading.Thread(target=task, daemon=True).start()

    def _on_config_change(self, _=None):
        """Recarga el modelo con la nueva configuración de familia/escala."""
        family = self.model_selector.get()
        alias = self.scale_selector.get()
        
        if not family or not alias:
            return
            
        # Asegurar que el botón "Sin Modelo" esté desactivado
        self.no_model_selector.set("")
            
        def load():
            self.is_loading_model = True
            try:
                self.add_log(f"Cargando modelo: {family} ({alias})...")
                
                success = self.detector.change_model(family, alias)
                
                if success:
                    # Actualizar Etiqueta de Hardware de forma segura
                    backend = getattr(self.detector, 'active_device', 'CPU')
                    vendor = self.detector.hardware_diag.get('gpu_vendor', 'Unknown')
                    def _update_ui():
                        self.hw_label.configure(text=f"PROCESAMIENTO: {backend} ({vendor})", text_color="#10b981")
                        self.add_log(f"Modelo {success} cargado correctamente.")
                    self.after(0, _update_ui)
                else:
                    def _update_err():
                        self.hw_label.configure(text="ERROR AL CARGAR MODELO", text_color="#ef4444")
                        self.add_log("Error critico: No se pudo cargar el modelo.")
                    self.after(0, _update_err)
            except Exception as e:
                self.add_log(f"Error en hilo de carga: {e}")
            finally:
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

    def clear_zones(self): 
        self.zones = []
        self._save_config()
        self._update_bar_mode_buttons()
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
            
            # Ajuste de coordenadas relativo al área de la imagen en el canvas
            click_x = e.x - self._img_offset_x
            click_y = e.y - self._img_offset_y
            
            # Normalización (0.0 a 1.0)
            nx = click_x / self._display_w
            ny = click_y / self._display_h
            
            # Mapeo a coordenadas del frame original
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
            self.zones.append(self.current_zone)
            self.current_zone = []
            self.toggle_zone_drawing()
            self._save_config()
            self._update_bar_mode_buttons()

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
        """Captura instantánea: pausa, congela el frame actual, y abre el anotador multi-clase."""
        try:
            if self.raw_frame is None:
                self.add_log("Error: No hay senal de video para capturar.")
                return

            ds_name = self.capture_entry.get().strip()
            if not ds_name:
                ds_name = f"dataset_{time.strftime('%d%m%Y_%H%M%S')}"
            ds_name = ds_name.lower().replace(" ", "_")
            
            # Asegurar que DATASETS_DIR existe
            if not os.path.exists(DATASETS_DIR):
                os.makedirs(DATASETS_DIR, exist_ok=True)
                
            dataset_dir = os.path.join(DATASETS_DIR, ds_name)
            self.add_log(f"Iniciando captura para: '{ds_name}'")

            # 1. Dialogo
            if ds_name not in self._checked_datasets and os.path.exists(dataset_dir):
                resp = messagebox.askyesnocancel("Dataset Existente", 
                    f"El dataset '{ds_name}' ya existe.\n\n¿Borrar y empezar de nuevo?", parent=self)
                if resp is None: 
                    self.add_log("Captura cancelada por el usuario.")
                    return 
                if resp:
                    shutil.rmtree(dataset_dir)
                    self.add_log("Dataset borrado.")
            
            self._checked_datasets.add(ds_name)

            captured_frame = self.raw_frame.copy()
            
            self._was_paused_before_capture = self.is_paused
            if not self.is_paused:
                self.toggle_pause()

            actual_dir = ensure_dataset_structure(ds_name)
            base_name = get_next_capture_filename(ds_name, actual_dir)
            
            self._enter_labeling_mode()
            AnnotationWindow(self, captured_frame, ds_name, base_name, actual_dir, 
                           self._on_capture_saved, self._exit_labeling_mode)
        except Exception as e:
            self.add_log(f"FALLO GLOBAL CAPTURA: {e}")
            import traceback
            print(traceback.format_exc())
            was_paused = getattr(self, '_was_paused_before_capture', True)
            if not was_paused and self.is_paused:
                self.toggle_pause()

    def _on_capture_cancelled(self):
        """Callback si el usuario cancela la anotación."""
        self._exit_labeling_mode()
        was_paused = getattr(self, '_was_paused_before_capture', True)
        if not was_paused and self.is_paused:
            self.toggle_pause()

    def _on_capture_saved(self, name, boxes_count):
        """Callback cuando el usuario guarda la anotación."""
        self.add_log(f"Captura guardada: {name} ({boxes_count} bboxes).")
        was_paused = getattr(self, '_was_paused_before_capture', True)
        if not was_paused and self.is_paused:
            self.toggle_pause()

    def export_dataset_zip(self):
        """Exporta solo las imágenes etiquetadas y su metadata en un archivo ZIP."""
        import zipfile
        from ..utils.helpers import DATASETS_DIR
        
        if not os.path.exists(DATASETS_DIR) or not os.listdir(DATASETS_DIR):
            self.add_log("No hay datasets para exportar.")
            return
        
        dest = tk.filedialog.asksaveasfilename(
            defaultextension=".zip",
            filetypes=[("Archivo ZIP", "*.zip")],
            initialfile="dataset_export_selective.zip"
        )
        if not dest:
            return
        
        try:
            total_files = 0
            with zipfile.ZipFile(dest, 'w', zipfile.ZIP_DEFLATED) as zf:
                # Iterar por cada dataset (carpeta de primer nivel en DATASETS_DIR)
                for ds_name in os.listdir(DATASETS_DIR):
                    ds_path = os.path.join(DATASETS_DIR, ds_name)
                    if not os.path.isdir(ds_path): continue

                    # 1. Archivos maestros del dataset
                    for master in ["data.yaml", "classes.txt"]:
                        m_path = os.path.join(ds_path, master)
                        if os.path.exists(m_path):
                            zf.write(m_path, os.path.join(ds_name, master))
                            total_files += 1

                    # 2. Etiquetas e Imágenes sincronizadas
                    labels_dir = os.path.join(ds_path, "labels", "train")
                    images_dir = os.path.join(ds_path, "images", "train")
                    
                    if os.path.exists(labels_dir):
                        for lab_file in os.listdir(labels_dir):
                            if lab_file.endswith(".txt"):
                                base = os.path.splitext(lab_file)[0]
                                lab_full = os.path.join(labels_dir, lab_file)
                                
                                # Buscar imagen correspondiente
                                img_found = None
                                for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
                                    test_img = os.path.join(images_dir, base + ext)
                                    if os.path.exists(test_img):
                                        img_found = test_img
                                        break
                                
                                if img_found:
                                    # Añadir etiqueta
                                    zf.write(lab_full, os.path.join(ds_name, "labels", "train", lab_file))
                                    # Añadir imagen
                                    zf.write(img_found, os.path.join(ds_name, "images", "train", os.path.basename(img_found)))
                                    total_files += 2

            if total_files > 0:
                self.add_log(f"Exportacion selectiva completa: {total_files} archivos en '{os.path.basename(dest)}'.")
            else:
                self.add_log("No se encontraron imagenes con etiquetas para exportar.")
                if os.path.exists(dest): os.remove(dest)
        except Exception as e:
            self.add_log(f"Error exportando dataset: {e}")

    def import_zip_dataset(self):
        """Importa un ZIP de imágenes, las extrae y abre el anotador secuencial."""
        import zipfile
        from ..utils.helpers import DATASETS_DIR, ensure_dataset_structure
        
        file_path = tk.filedialog.askopenfilename(
            title="Seleccionar ZIP de imágenes para etiquetar",
            filetypes=[("Archivo ZIP", "*.zip")]
        )
        if not file_path:
            return
            
        ds_name = self.capture_entry.get().strip()
        if not ds_name:
            ds_name = f"import_{time.strftime('%H%M%S')}"
        ds_name = ds_name.lower().replace(" ", "_")
        
        dataset_dir = ensure_dataset_structure(ds_name)
        if not dataset_dir:
            self.add_log("Error creando estructura de dataset.")
            return

        self.add_log(f"Importando ZIP: {os.path.basename(file_path)}...")
        
        try:
            # Extraer imágenes a la carpeta 'images/train' del dataset
            target_img_dir = os.path.join(dataset_dir, "images", "train")
            
            image_paths = []
            with zipfile.ZipFile(file_path, 'r') as zf:
                for info in zf.infolist():
                    if info.is_dir(): continue
                    ext = os.path.splitext(info.filename)[1].lower()
                    if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                        # Extraer aplanado (sin carpetas internas del zip para simplificar)
                        filename = os.path.basename(info.filename)
                        dest_path = os.path.join(target_img_dir, filename)
                        
                        with zf.open(info) as source, open(dest_path, "wb") as target:
                            shutil.copyfileobj(source, target)
                        
                        image_paths.append(dest_path)
            
            if not image_paths:
                self.add_log("El ZIP no contiene imagenes validas.")
                return
                
            self.add_log(f"{len(image_paths)} imagenes importadas en '{ds_name}'.")
            
            # Abrir AnnotationWindow en modo múltiple
            self._enter_labeling_mode()
            AnnotationWindow(self, 
                             ds_name=ds_name, 
                             dataset_dir=dataset_dir, 
                             on_save=self._on_capture_saved, 
                             on_close=self._exit_labeling_mode,
                             image_files=image_paths)
            
        except Exception as e:
            self.add_log(f"Error al importar ZIP: {e}")

    def resume_labeling(self):
        """Busca datasets existentes y abre el anotador para continuar el trabajo."""
        if not os.path.exists(DATASETS_DIR):
            self.add_log("No existen datasets todavía.")
            return
            
        datasets = [d for d in os.listdir(DATASETS_DIR) if os.path.isdir(os.path.join(DATASETS_DIR, d))]
        if not datasets:
            self.add_log("No se encontraron carpetas de dataset.")
            return
            
        # Popup simple para seleccionar dataset
        popup = ctk.CTkToplevel(self)
        popup.title("Reanudar Etiquetado")
        popup.geometry("300x400")
        popup.grab_set()
        
        ctk.CTkLabel(popup, text="SELECCIONA DATASET", font=ctk.CTkFont(weight="bold")).pack(pady=10)
        
        scroll = ctk.CTkScrollableFrame(popup)
        scroll.pack(fill="both", expand=True, padx=10, pady=10)
        
        def _start_resume(ds_name):
            popup.destroy()
            ds_path = os.path.join(DATASETS_DIR, ds_name)
            img_dir = os.path.join(ds_path, "images", "train")
            
            if not os.path.exists(img_dir):
                self.add_log(f"El dataset '{ds_name}' no tiene carpeta de imagenes.")
                return
                
            image_paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            if not image_paths:
                self.add_log(f"No se encontraron imagenes en '{ds_name}'.")
                return
            
            self.add_log(f"Reanudando sesion: {ds_name} ({len(image_paths)} imagenes)")
            self._enter_labeling_mode()
            AnnotationWindow(self, 
                             ds_name=ds_name, 
                             dataset_dir=ds_path, 
                             on_save=self._on_capture_saved, 
                             on_close=self._exit_labeling_mode,
                             image_files=image_paths)

        for ds in sorted(datasets):
            ctk.CTkButton(scroll, text=ds, command=lambda d=ds: _start_resume(d)).pack(fill="x", pady=2)

    def _enter_labeling_mode(self):
        """Suspende el motor de vídeo y entra en modo ahorro de recursos para etiquetar."""
        self._was_paused_before_labeling = self.is_paused
        self.is_labeling_mode = True
        self.add_log("Modo Etiquetado: Suspendiendo motor de video para ahorrar recursos...")
        self.engine.release()
        self.hw_label.configure(text="SISTEMA SUSPENDIDO (ETIQUETANDO)", text_color="#94a3b8")

    def _exit_labeling_mode(self):
        """Sale del modo etiquetado y reactiva el motor de vídeo."""
        self.is_labeling_mode = False
        self.add_log("Modo Etiquetado finalizado: Reactivando motor...")
        
        # Restaurar estado de pausa previo
        was_paused = getattr(self, '_was_paused_before_labeling', False)
        if self.is_paused != was_paused:
            self.toggle_pause()
            
        self.change_stream() # Esto dispara la reconexión asíncrona

    # --- LÓGICA DE AUTOCAPTURA PERIÓDICA ---

    def toggle_auto_capture(self):
        """Alterna el estado de la autocaptura periódica."""
        if self.is_auto_capturing:
            self.stop_auto_capture()
        else:
            self.start_auto_capture()

    def start_auto_capture(self):
        """Inicia el proceso de captura automática."""
        try:
            val = self.auto_capture_interval.get().strip()
            interval = float(val) if val else 5.0
            if interval <= 0: raise ValueError
        except ValueError:
            self.add_log("Intervalo de autocaptura no valido (usa un numero > 0).")
            return

        # Determinar nombre del dataset (usar el de la entrada si existe)
        ds_name = self.capture_entry.get().strip()
        if not ds_name:
            ds_name = f"auto_{time.strftime('%d%m%Y_%H%M')}"
        ds_name = ds_name.lower().replace(" ", "_")
        
        # Preparar carpeta
        from ..utils.helpers import ensure_dataset_structure
        self.auto_dataset_dir = ensure_dataset_structure(ds_name)
        if not self.auto_dataset_dir:
            self.add_log("Error creando carpeta para autocaptura.")
            return

        self.auto_ds_name = ds_name
        self.is_auto_capturing = True
        self.auto_capture_btn.configure(text="Parar", fg_color="#ef4444", hover_color="#dc2626")
        self.add_log(f"Autocaptura iniciada: cada {interval}s en '{ds_name}'")
        
        # Iniciar bucle
        self._run_auto_capture_loop()

    def stop_auto_capture(self):
        """Detiene la autocaptura y abre la carpeta de resultados."""
        self.is_auto_capturing = False
        self.auto_capture_btn.configure(text="Auto", fg_color="#10b981", hover_color="#059669")
        self.add_log("Autocaptura detenida.")
        
        # Abrir carpeta (Solo en Windows)
        if hasattr(self, 'auto_dataset_dir') and os.path.exists(self.auto_dataset_dir):
            img_path = os.path.abspath(os.path.join(self.auto_dataset_dir, "images", "train"))
            if os.path.exists(img_path):
                self.add_log(f"Abriendo carpeta: {img_path}")
                try:
                    os.startfile(img_path)
                except Exception as e:
                    self.add_log(f"No se pudo abrir la carpeta: {e}")

    def _run_auto_capture_loop(self):
        """Bucle interno de guardado de frames."""
        if not self.is_auto_capturing:
            return
        
        if self.raw_frame is not None:
            from ..utils.helpers import get_next_capture_filename
            base_name = get_next_capture_filename(self.auto_ds_name, self.auto_dataset_dir)
            full_path = os.path.join(self.auto_dataset_dir, "images", "train", f"{base_name}.jpg")
            
            # Guardar frame actual de forma segura
            cv2.imwrite(full_path, self.raw_frame)
            self.add_log(f"Frame guardado: {base_name}")
        else:
            self.add_log("Autocaptura: Esperando senal de video...")
            
        # Programar siguiente
        try:
            interval_ms = int(float(self.auto_capture_interval.get()) * 1000)
        except:
            interval_ms = 5000
            
        self.after(interval_ms, self._run_auto_capture_loop)


