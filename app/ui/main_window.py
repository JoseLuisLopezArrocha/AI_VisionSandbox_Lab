import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox
import cv2
import numpy as np
import os
import threading
import time
import webbrowser
import shutil
from PIL import Image, ImageTk
from collections import Counter
from ..core.engine import VisionEngine
from ..core.detector import ObjectDetector
from ..utils.helpers import ZONE_COLORS, DATASETS_DIR, ensure_dataset_structure, get_next_capture_filename, save_app_config, load_app_config
from .components import AnnotationWindow, AddModelPopup, ClassFilterWindow, InfoWindow, ModelExplorerWindow, SourceSelectorWindow, FavoritesWindow
from ..utils.painter import VisualPainter
from ..core.events import EventEngine
from .events_window import EventsWindow as EventConfigWindow
from .settings_window import SettingsWindow
from ..utils.logger import DataLogger

class _SplashScreen(ctk.CTkFrame):
    """Pantalla de carga integrada en la ventana principal."""

    def __init__(self, parent):
        super().__init__(parent, fg_color='#1a1c1e')
        self.place(relx=0, rely=0, relwidth=1, relheight=1)
        container = ctk.CTkFrame(self, fg_color='transparent', corner_radius=0)
        container.place(relx=0.5, rely=0.5, anchor='center')
        ctk.CTkLabel(container, text='AI VISIONSANDBOX LAB', font=ctk.CTkFont(size=28, weight='bold'), text_color='#047857', corner_radius=0).pack(pady=(0, 10))
        self.progress = ctk.CTkProgressBar(container, width=300, corner_radius=0)
        self.progress.pack(pady=10)
        self.progress.set(0)
        self.status_label = ctk.CTkLabel(container, text='Cargando sistema...', font=ctk.CTkFont(size=12), text_color='#94a3b8', corner_radius=0)
        self.status_label.pack(pady=10)
        self.update()

    def set_status(self, text, progress_val=0.0):
        self.status_label.configure(text=text)
        self.progress.set(progress_val)
        self.update()

class FullscreenImageWindow(ctk.CTkToplevel):
    """Ventana emergente para visualizar evidencias a tamaño completo."""
    def __init__(self, parent, img_bgr, title, raw_bgr=None, zoom_bgr=None):
        super().__init__(parent)
        self.title(f"Evidencia: {title}")
        self.geometry("1000x700")
        self.after(100, lambda: self.focus_force())
        
        # Convertir BGR a RGB para PIL
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(rgb)
        
        # Ajustar imagen al tamaño de la ventana
        self.bind("<Configure>", lambda e: self._resize_img())
        
        self.img_label = ctk.CTkLabel(self, text="", corner_radius=0)
        self.img_label.pack(fill="both", expand=True)
        
        # Guardar frames para toggle
        self.img_ann = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        self.img_raw = None
        if raw_bgr is not None:
            self.img_raw = Image.fromarray(cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB))
        self.img_zoom = None
        if zoom_bgr is not None:
            self.img_zoom = Image.fromarray(cv2.cvtColor(zoom_bgr, cv2.COLOR_BGR2RGB))
        
        self.current_img = self.img_ann
        
        # Botones de control
        self.controls = ctk.CTkFrame(self, fg_color='transparent', height=40, corner_radius=0)
        self.controls.pack(fill='x', side='bottom', pady=10)
        
        self.btn_group = ctk.CTkFrame(self.controls, fg_color='transparent', corner_radius=0)
        self.btn_group.pack(expand=True)
        
        # Botón de modo
        self.btn_toggle = ctk.CTkButton(self.btn_group, text="VER ORIGINAL (RAW)", command=self._toggle_mode, fg_color='#334155', width=150, corner_radius=0)
        self.btn_toggle.pack(side='left', padx=5)
        
        if self.img_raw is None:
            self.btn_toggle.configure(state="disabled", text="RAW NO DISPONIBLE")
        
        if self.img_zoom:
            self.btn_zoom = ctk.CTkButton(self.btn_group, text="VER SMART ZOOM ⚡", command=self._toggle_zoom, fg_color='#1e293b', width=150, corner_radius=0)
            self.btn_zoom.pack(side='left', padx=5)
        
        self._resize_img()

    def _toggle_zoom(self):
        if self.img_zoom is None: return
        if self.current_img == self.img_zoom:
            self.current_img = self.img_ann
            self.btn_zoom.configure(text="VER SMART ZOOM ⚡", fg_color='#1e293b')
        else:
            self.current_img = self.img_zoom
            self.btn_zoom.configure(text="VOLVER A GENERAL", fg_color='#d97706')
            if self.img_raw is not None:
                self.btn_toggle.configure(text="VER ORIGINAL (RAW)", fg_color='#334155')
        self._resize_img()

    def _toggle_mode(self):
        if self.img_raw is None: return
        if self.current_img == self.img_raw:
            self.current_img = self.img_ann
            self.btn_toggle.configure(text="VER ORIGINAL (RAW)", fg_color='#334155')
        else:
            self.current_img = self.img_raw
            self.btn_toggle.configure(text="VER ANOTADA (AI)", fg_color='#059669')
            if self.img_zoom: 
                self.btn_zoom.configure(text="VER SMART ZOOM ⚡", fg_color='#1e293b')
        self._resize_img()

    def _resize_img(self):
        if self.current_img is None: return
        w, h = self.winfo_width(), self.winfo_height() - 60 # Espacio para controles
        if w < 100 or h < 100: return
        
        # Mantener ratio
        iw, ih = self.current_img.size
        ratio = min(w/iw, h/ih)
        new_size = (int(iw*ratio), int(ih*ratio))
        
        img_ctk = ctk.CTkImage(light_image=self.current_img, dark_image=self.current_img, size=new_size)
        self.img_label.configure(image=img_ctk)
        self.img_label.image = img_ctk

class VisionApp(ctk.CTk):
    """Aplicación Principal: Dashboard de Control de Visión Artificial."""

    def __init__(self):
        super().__init__()
        self.NONE_MODEL = 'Sin Modelo'
        self.title('AI VisionSandbox Lab | Dashboard de Inteligencia')
        self.minsize(960, 640)
        width, height = (600, 400)
        x = self.winfo_screenwidth() // 2 - width // 2
        y = self.winfo_screenheight() // 2 - height // 2
        self.geometry(f'{width}x{height}+{x}+{y}')
        self.splash = _SplashScreen(self)
        self.splash.set_status('Iniciando motores core...')
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
        self.target_classes = None  # Lista de IDs (int) a detectar (M1)
        self.target_classes_m2 = None # Lista de IDs (int) a detectar (M2)
        self.heatmap_enabled = False
        self.heatmap_acc = None
        self.conf_threshold = 0.35
        self.infer_interval = 0.1
        self.last_infer_time = 0
        self.last_infer_timestamp = 0
        self.flash_alpha = 0.0
        self.url = 'https://www.youtube.com/watch?v=dfVK7ld38Ys'
        self.locked_track_id = None
        self.focus_lost_cnt = 0
        self._checked_datasets = set()
        self.is_auto_capturing = False
        self.total_detections_ever = 0
        self.session_start_time = time.time()
        self.session_seen_ids = set()
        self.session_class_counts = Counter()
        self.session_zone_data = {} # {idx: {"ids": set(), "counts": Counter()}}
        self.bar_chart_mode = 'General'
        self.dashboard_config = {
            "show_top_5": True,
            "pinned_classes": [],
            "chart_type": "vbar", # 'vbar', 'hbar', 'line'
            "axis_x": "class",    # 'class', 'zone', 'time'
            "axis_y": "count",    # 'count', 'cumulative', 'conf'
            "chart_mode": "live",
            "metric_primary": "total_unique"
        }
        self.chart_history = [] # Buffer para series temporales: [(timestamp, detections, zone_counts), ...]
        self.data_logger = DataLogger()
        self.event_engine = EventEngine()
        self.detector = ObjectDetector()
        self.engine = VisionEngine(self.url)
        self._load_icons()
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=0)
        self.splash.set_status('Construyendo interfaz...', 0.4)
        self._build_top_bar()
        self._build_sidebar()
        self._build_main_area()
        self._build_bottom_dashboard()
        self._load_config()
        self.splash.set_status('Cargando modelo inteligente...', 0.7)
        self._load_lightest_model()
        self.splash.set_status('Listo.', 1.0)
        self.add_log('Sistema core iniciado.')
        self.after(500, self._show_and_start)

    def _show_and_start(self):
        """Finaliza la carga, destruye el splash y expande al dashboard completo."""
        try:
            self.splash.destroy()
            self.state('zoomed')
            self.after(200, self.update_video)
        except Exception:
            self.update_video()

    def _load_icons(self):
        """Carga los iconos locales en memoria."""
        self.icons = {}
        icon_names = ['source', 'favs', 'alerts', 'settings', 'night', 'day', 'info', 'play', 'pause', 'back', 'forward', 'models']
        for name in icon_names:
            p_light = os.path.join('assets', 'icons', f'{name}_dark.png')
            p_dark = os.path.join('assets', 'icons', f'{name}.png')
            if os.path.exists(p_dark):
                self.icons[name] = ctk.CTkImage(light_image=Image.open(p_light), dark_image=Image.open(p_dark), size=(16, 16))

    def _load_lightest_model(self):
        """Inicializa el selector en 'Sin Modelo' para un arranque instantáneo."""
        try:
            self._on_no_model_click()
        except Exception as e:
            print(f'[Main] Error en inicialización de modelo: {e}')

    def _build_top_bar(self):
        """Barra superior de herramientas y estado del stream."""
        self.top_bar = ctk.CTkFrame(self, height=45, fg_color='#0f172a', corner_radius=0)
        self.top_bar.grid(row=0, column=0, columnspan=2, sticky='ew')
        left_menu = ctk.CTkFrame(self.top_bar, fg_color='#1e293b', corner_radius=0)
        left_menu.pack(side='left', padx=10, pady=5)
        for col in range(3):
            left_menu.grid_columnconfigure(col, weight=1, uniform='topbar')
        top_btn_style = {'height': 28, 'corner_radius': 0, 'font': ctk.CTkFont(size=11, weight='bold'), 'text_color': '#94a3b8', 'fg_color': 'transparent', 'hover_color': '#334155'}
        ctk.CTkButton(left_menu, text=' FUENTE', image=self.icons.get('source'), command=self.open_source_selector, **top_btn_style).grid(row=0, column=0, sticky='ew', padx=0, pady=0)
        ctk.CTkButton(left_menu, text=' FAVS', image=self.icons.get('favs'), command=lambda: FavoritesWindow(self, self.change_stream), **top_btn_style).grid(row=0, column=1, sticky='ew', padx=0, pady=0)
        self.stream_quality_var = ctk.StringVar(value='720p')
        self.stream_quality = ctk.CTkOptionMenu(left_menu, values=['360p', '480p', '720p', '1080p'], variable=self.stream_quality_var, height=28, fg_color='#1e293b', button_color='#3b82f6', button_hover_color='#2563eb', corner_radius=0, font=ctk.CTkFont(size=11, weight='bold'), text_color='#94a3b8')
        self.stream_quality.grid(row=0, column=2, sticky='ew', padx=0, pady=0)
        center_frame = ctk.CTkFrame(self.top_bar, fg_color='transparent', corner_radius=0)
        center_frame.pack(side='left', expand=True, fill='both')
        self.live_indicator = ctk.CTkFrame(center_frame, fg_color='transparent', corner_radius=0)
        self.live_indicator.pack(expand=True)
        self.live_dot = ctk.CTkLabel(self.live_indicator, text='*', text_color='#7f1d1d', font=ctk.CTkFont(size=20, weight='bold'), corner_radius=0)
        self.live_dot.pack(side='left', padx=(0, 5))
        self.live_text = ctk.CTkLabel(self.live_indicator, text='EN DIRECTO', font=ctk.CTkFont(size=14, weight='bold'), text_color='#7f1d1d', corner_radius=0)
        self.live_text.pack(side='left')
        self.live_url_label = ctk.CTkLabel(self.live_indicator, text='', font=ctk.CTkFont(size=12), text_color='#94a3b8', corner_radius=0)
        self.live_url_label.pack(side='left', padx=(10, 0))
        right_menu = ctk.CTkFrame(self.top_bar, fg_color='transparent', corner_radius=0)
        right_menu.pack(side='right', padx=10, fill='y', pady=5)
        theme_frame = ctk.CTkFrame(right_menu, fg_color='#1e293b', corner_radius=0)
        theme_frame.pack(side='left', padx=5)
        ctk.CTkButton(theme_frame, image=self.icons.get('night'), text='', width=30, height=26, fg_color='transparent', hover_color='#334155', command=lambda: ctk.set_appearance_mode('Dark'), corner_radius=0).pack(side='left', padx=1, pady=2)
        ctk.CTkButton(theme_frame, image=self.icons.get('day'), text='', width=30, height=26, fg_color='transparent', hover_color='#334155', command=lambda: ctk.set_appearance_mode('Light'), corner_radius=0).pack(side='left', padx=1, pady=2)
        ctk.CTkButton(right_menu, image=self.icons.get('info'), text='', width=35, height=26, fg_color='#1e293b', hover_color='#334155', command=lambda: InfoWindow(self), corner_radius=0).pack(side='left', padx=2)
        ctk.CTkButton(right_menu, image=self.icons.get('settings'), text='', width=35, height=26, fg_color='#1e293b', hover_color='#334155', command=self.open_settings, corner_radius=0).pack(side='left', padx=5)

    def _build_sidebar(self):
        """Construye el panel lateral con scroll mejorado y diseño compacto."""
        self.grid_columnconfigure(0, weight=0, minsize=340) # Aumentado a 340 para evitar cortes
        self.grid_columnconfigure(1, weight=1)
        self.sidebar = ctk.CTkFrame(self, width=340, corner_radius=0)
        self.sidebar.grid(row=1, column=0, rowspan=2, sticky='nsew')
        self.sidebar.grid_propagate(False)
        # Footer fijo abajo (empaquetado primero)
        bottom = ctk.CTkFrame(self.sidebar, fg_color='transparent', corner_radius=0)
        bottom.pack(side='bottom', fill='x', padx=20, pady=(4, 10))
        self.hw_label = ctk.CTkLabel(bottom, text=f"{self.detector.hardware_diag['gpu_name'][:20]} | {self.detector.hardware_diag['best_backend'].upper()}", font=ctk.CTkFont(size=10), text_color='#10b981', corner_radius=0)
        self.hw_label.pack(fill='x', pady=(0, 2))
        self.antigravity_btn = ctk.CTkLabel(bottom, text='Antigravity System', font=ctk.CTkFont(size=10, slant='italic'), text_color='#059669', cursor='hand2', corner_radius=0)
        self.antigravity_btn.pack(fill='x')
        self.antigravity_btn.bind('<Button-1>', lambda e: webbrowser.open('https://antigravity.google'))
        # Cuerpo scrollable (Ajustado para evitar cortes laterales)
        self.sidebar_body = ctk.CTkScrollableFrame(self.sidebar, fg_color='transparent', corner_radius=0, scrollbar_button_color='#1e293b', scrollbar_button_hover_color='#334155')
        self.sidebar_body.pack(fill='both', expand=True, padx=2)
        sb = self.sidebar_body
        header_frame = ctk.CTkFrame(sb, fg_color='transparent', height=10, corner_radius=0)
        header_frame.pack(pady=(5, 5), padx=10, fill='x')
        self._section('MODELO IA')
        ctk.CTkButton(self.sidebar_body, text=' NUEVO MODELO', image=self.icons.get('models'), command=lambda: ModelExplorerWindow(self, self.detector), height=34, fg_color='#3b82f6', hover_color='#2563eb', text_color='#ffffff', font=ctk.CTkFont(size=11, weight='bold'), corner_radius=0).pack(pady=(0, 10), padx=15, fill='x')
        self.no_model_selector = ctk.CTkButton(self.sidebar_body, text='DESACTIVAR IA', command=self._on_no_model_click, height=30, fg_color='#1e293b', hover_color='#7f1d1d', text_color='#ef4444', font=ctk.CTkFont(size=11, weight='bold'), corner_radius=0)
        self.no_model_selector.pack(pady=(0, 10), padx=15, fill='x')
        
        families = list(self.detector.architectures.keys())
        ctk.CTkLabel(self.sidebar_body, text='ARQUITECTURA:', font=ctk.CTkFont(size=9, weight='bold'), text_color='#64748b', corner_radius=0).pack(padx=15, anchor='w')
        self.model_selector = ctk.CTkComboBox(self.sidebar_body, values=families, command=self._on_family_change, height=28, fg_color='#0f172a', border_color='#334155', button_color='#3b82f6', corner_radius=0)
        self.model_selector.pack(pady=(2, 6), padx=15, fill='x')
        self.model_selector.set('Seleccionar...')
        
        ctk.CTkLabel(self.sidebar_body, text='ESCALA / OPTIMIZACIÓN:', font=ctk.CTkFont(size=9, weight='bold'), text_color='#64748b', corner_radius=0).pack(padx=15, anchor='w')
        self.scale_selector = ctk.CTkComboBox(self.sidebar_body, values=[], command=self._on_config_change, height=28, fg_color='#0f172a', border_color='#334155', button_color='#3b82f6', corner_radius=0)
        self.scale_selector.pack(pady=(2, 10), padx=15, fill='x')
        self.scale_selector.set('')
        self.world_prompt_frame = ctk.CTkFrame(self.sidebar_body, fg_color='#1e293b', border_width=1, border_color='#059669', corner_radius=0)
        ctk.CTkLabel(self.world_prompt_frame, text='BÚSQUEDA UNIVERSAL AI', font=ctk.CTkFont(size=10, weight='bold'), text_color='#059669', corner_radius=0).pack(pady=(2, 0))
        w_row = ctk.CTkFrame(self.world_prompt_frame, fg_color='transparent', corner_radius=0)
        w_row.pack(fill='x', padx=5, pady=5)
        self.world_entry = ctk.CTkEntry(w_row, placeholder_text='Prompt...', height=26, font=ctk.CTkFont(size=11), corner_radius=0)
        self.world_entry.pack(side='left', fill='x', expand=True, padx=(0, 2))
        self.world_entry.bind('<Return>', lambda _: self.apply_world_prompt())
        ctk.CTkButton(w_row, text='OK', width=40, height=26, fg_color='#059669', hover_color='#047857', text_color='#000', font=ctk.CTkFont(size=10, weight='bold'), command=self.apply_world_prompt, corner_radius=0).pack(side='right')
        # --- MODELO SECUNDARIO (M2) --- Sección colapsable
        self.m2_section_frame = ctk.CTkFrame(self.sidebar_body, fg_color='transparent', corner_radius=0)
        self.m2_section_frame.pack(pady=(5, 0), padx=15, fill='x')
        ctk.CTkLabel(self.m2_section_frame, text='MODELO SECUNDARIO (M2)', font=ctk.CTkFont(size=10, weight='bold'), text_color='#f59e0b', corner_radius=0).pack(anchor='w')
        self.m2_toggle_btn = ctk.CTkButton(self.m2_section_frame, text='ACTIVAR M2', command=self._on_m2_toggle, height=26, fg_color='#1e293b', hover_color='#334155', border_width=1, border_color='#f59e0b', text_color='#f59e0b', font=ctk.CTkFont(size=10, weight='bold'), corner_radius=0)
        self.m2_toggle_btn.pack(pady=(4, 4), fill='x')
        self.m2_controls_frame = ctk.CTkFrame(self.m2_section_frame, fg_color='#1e293b', border_width=1, border_color='#f59e0b', corner_radius=0)
        # Selectores M2 (ocultos inicialmente)
        families_m2 = list(self.detector.architectures.keys())
        self.m2_family_selector = ctk.CTkComboBox(self.m2_controls_frame, values=families_m2, command=self._on_m2_family_change, height=26, fg_color='#0f172a', border_color='#f59e0b', button_color='#f59e0b', corner_radius=0)
        self.m2_family_selector.pack(pady=(6, 4), padx=8, fill='x')
        self.m2_scale_selector = ctk.CTkComboBox(self.m2_controls_frame, command=self._on_m2_config_change, height=26, fg_color='#0f172a', border_color='#f59e0b', button_color='#f59e0b', corner_radius=0)
        self.m2_scale_selector.pack(pady=(0, 4), padx=8, fill='x')
        self.m2_status_label = ctk.CTkLabel(self.m2_controls_frame, text='Sin modelo M2', font=ctk.CTkFont(size=9), text_color='#94a3b8', corner_radius=0)
        self.m2_status_label.pack(pady=(0, 2))
        self.m2_filter_btn = ctk.CTkButton(self.m2_controls_frame, text='Filtro Clases M2', command=lambda: self.open_class_filter(is_secondary=True), height=22, fg_color='#0f172a', hover_color='#1e293b', border_width=1, border_color='#f59e0b', text_color='#f59e0b', font=ctk.CTkFont(size=9, weight='bold'), corner_radius=0)
        self.m2_filter_btn.pack(pady=(0, 6), padx=8, fill='x')
        self.m2_active = False  # Estado del toggle M2
        self._section('ANÁLISIS TÁCTICO')
        
        # Confianza (Umbral)
        conf_frame = ctk.CTkFrame(self.sidebar_body, fg_color='transparent', corner_radius=0)
        conf_frame.pack(pady=(2, 8), padx=15, fill='x')
        
        ctk.CTkLabel(conf_frame, text='🔍 UMBRAL DE DETECCIÓN (Confianza)', font=ctk.CTkFont(size=10, weight='bold'), text_color='#3b82f6', corner_radius=0).pack(anchor='w')
        self.conf_slider = ctk.CTkSlider(conf_frame, from_=0.01, to=0.99, number_of_steps=98, command=self._on_conf_change, height=18, fg_color='#1e293b', progress_color='#3b82f6', button_color='#60a5fa', button_hover_color='#93c5fd', corner_radius=0)
        self.conf_slider.set(0.35)
        self.conf_slider.pack(pady=(4, 2), fill='x')
        
        conf_labels = ctk.CTkFrame(conf_frame, fg_color='transparent', corner_radius=0)
        conf_labels.pack(fill='x')
        ctk.CTkLabel(conf_labels, text='MIN (1%)', font=ctk.CTkFont(size=8), text_color='#475569', corner_radius=0).pack(side='left')
        self.conf_label = ctk.CTkLabel(conf_labels, text='35%', font=ctk.CTkFont(size=11, weight='bold'), text_color='#3b82f6', corner_radius=0)
        self.conf_label.pack(side='right')
        
        # Intervalo (Muestreo)
        time_frame = ctk.CTkFrame(self.sidebar_body, fg_color='transparent', corner_radius=0)
        time_frame.pack(pady=(4, 8), padx=15, fill='x')
        
        ctk.CTkLabel(time_frame, text='⏱ FRECUENCIA DE MUESTREO', font=ctk.CTkFont(size=10, weight='bold'), text_color='#10b981', corner_radius=0).pack(anchor='w')
        self.interval_slider = ctk.CTkSlider(time_frame, from_=0.0, to=5.0, number_of_steps=50, command=self._on_interval_change, height=18, fg_color='#1e293b', progress_color='#10b981', button_color='#34d399', button_hover_color='#6ee7b7', corner_radius=0)
        self.interval_slider.set(self.infer_interval)
        self.interval_slider.pack(pady=(4, 2), fill='x')
        
        time_labels = ctk.CTkFrame(time_frame, fg_color='transparent', corner_radius=0)
        time_labels.pack(fill='x')
        ctk.CTkLabel(time_labels, text='MÁX (FPS)', font=ctk.CTkFont(size=8), text_color='#475569', corner_radius=0).pack(side='left')
        self.interval_label = ctk.CTkLabel(time_labels, text=f'{self.infer_interval:.1f}s', font=ctk.CTkFont(size=11, weight='bold'), text_color='#10b981', corner_radius=0)
        self.interval_label.pack(side='right')
        f_row = ctk.CTkFrame(self.sidebar_body, fg_color='transparent', corner_radius=0)
        f_row.pack(fill='x', padx=15, pady=(5, 10))
        self.heatmap_switch = ctk.CTkSwitch(f_row, text='Mapa de Calor', command=self._toggle_heatmap, progress_color='#10b981', corner_radius=0, font=ctk.CTkFont(size=11))
        self.heatmap_switch.pack(side='left')
        ctk.CTkButton(f_row, text='Filtro Clases', fg_color='#3b82f6', hover_color='#2563eb', command=self.open_class_filter, height=26, width=100, font=ctk.CTkFont(size=10, weight='bold'), corner_radius=0).pack(side='right')
        z_btns = ctk.CTkFrame(self.sidebar_body, fg_color='transparent', corner_radius=0)
        z_btns.pack(pady=(5, 5), padx=15, fill='x')
        self.draw_btn = ctk.CTkButton(z_btns, text='DELIMITAR ZONAS', command=self.toggle_zone_drawing, width=110, height=32, fg_color='#1e293b', border_width=1, border_color='#334155', font=ctk.CTkFont(size=11, weight='bold'), corner_radius=0)
        self.draw_btn.pack(side='left', padx=0, fill='x', expand=True)
        ctk.CTkButton(z_btns, text='BORRAR ZONAS', command=self.clear_zones, fg_color='#450a0a', hover_color='#7f1d1d', height=32, font=ctk.CTkFont(size=11, weight='bold'), corner_radius=0).pack(side='left', fill='x', expand=True, padx=0)
        ctk.CTkButton(self.sidebar_body, text=' GESTOR DE EVENTOS / ALERTAS', image=self.icons.get('alerts'), command=self.open_events_config, height=34, fg_color='#f59e0b', hover_color='#d97706', text_color='#0f172a', font=ctk.CTkFont(size=11, weight='bold'), corner_radius=0).pack(pady=(5, 10), padx=15, fill='x')
        self._section('CONTROL DE CAPTURA')
        cap_frame = ctk.CTkFrame(self.sidebar_body, fg_color='#1e293b', border_width=1, border_color='#334155', corner_radius=0)
        cap_frame.pack(pady=(0, 10), padx=15, fill='x')
        ds_row = ctk.CTkFrame(cap_frame, fg_color='transparent', corner_radius=0)
        ds_row.pack(fill='x', padx=10, pady=(10, 5))
        self.capture_entry = ctk.CTkEntry(ds_row, placeholder_text='Dataset', height=28, border_width=1, border_color='#334155', corner_radius=0)
        self.capture_entry.pack(side='left', fill='x', expand=True, padx=(0, 5))
        btn_style_sidebar = {'width': 35, 'height': 28, 'corner_radius': 0, 'font': ctk.CTkFont(size=10, weight='bold')}
        ctk.CTkButton(ds_row, text='CAP', fg_color='#10b981', hover_color='#059669', command=self.take_capture, **btn_style_sidebar).pack(side='left', padx=0)
        ctk.CTkButton(ds_row, text='RES', fg_color='#3b82f6', hover_color='#2563eb', command=self.resume_labeling, **btn_style_sidebar).pack(side='left', padx=0)
        tools_row = ctk.CTkFrame(cap_frame, fg_color='transparent', corner_radius=0)
        tools_row.pack(fill='x', padx=10, pady=(0, 10))
        ctk.CTkButton(tools_row, text='IMPORTAR ZIP', fg_color='#f59e0b', hover_color='#d97706', command=self.import_zip_dataset, height=28, font=ctk.CTkFont(size=10, weight='bold'), corner_radius=0).pack(side='left', fill='x', expand=True, padx=(0, 2))
        ctk.CTkButton(tools_row, text='EXPORTAR ZIP', fg_color='#6366f1', hover_color='#4f46e5', command=self.export_dataset_zip, height=28, font=ctk.CTkFont(size=10, weight='bold'), corner_radius=0).pack(side='left', fill='x', expand=True, padx=(2, 0))
        auto_frame = ctk.CTkFrame(cap_frame, fg_color='#0f172a', corner_radius=0)
        auto_frame.pack(fill='x', padx=10, pady=(0, 10))
        ctk.CTkLabel(auto_frame, text='AUTO-CAP:', font=ctk.CTkFont(size=10, weight='bold'), text_color='#94a3b8', corner_radius=0).pack(side='left', padx=5)
        self.auto_capture_interval = ctk.CTkEntry(auto_frame, placeholder_text='5', height=24, width=35, border_width=0, fg_color='transparent', corner_radius=0)
        self.auto_capture_interval.pack(side='left', padx=2)
        self.auto_capture_interval.insert(0, '5')
        ctk.CTkLabel(auto_frame, text='s', font=ctk.CTkFont(size=10), text_color='#64748b', corner_radius=0).pack(side='left')
        self.auto_capture_btn = ctk.CTkButton(auto_frame, text='INICIAR', width=60, height=24, fg_color='#10b981', hover_color='#059669', font=ctk.CTkFont(size=10, weight='bold'), command=self.toggle_auto_capture, corner_radius=0)
        self.auto_capture_btn.pack(side='right', padx=5)
        self.sidebar_body._parent_canvas.configure(highlightthickness=0) # Eliminar borde de canvas interno
        self.sidebar_body._scrollbar.grid_configure(padx=(2, 0)) # Ajustar scrollbar


    def _section(self, text):
        target = self.sidebar_body if hasattr(self, 'sidebar_body') else self.sidebar
        ctk.CTkLabel(target, text=text, font=ctk.CTkFont(size=12, weight='bold'), text_color='#94a3b8', corner_radius=0).pack(pady=(12, 6), padx=15, anchor='w')

    def _build_bottom_dashboard(self):
        """Dashboard horizontal para analítica y logs."""
        self.dash = ctk.CTkFrame(self, height=200, fg_color='#111', corner_radius=0)
        self.dash.grid(row=2, column=1, sticky='nsew')
        self.dash.grid_columnconfigure((1, 2), weight=2)
        self.dash.grid_columnconfigure(3, weight=3)
        self.dash.grid_rowconfigure(0, weight=1)
        self.m_frame = ctk.CTkFrame(self.dash, fg_color='transparent', corner_radius=0)
        self.m_frame.grid(row=0, column=0, padx=20, pady=10, sticky='n')
        self.count_label = ctk.CTkLabel(self.m_frame, text='OBJETOS: 0', font=ctk.CTkFont(size=16, weight='bold'), text_color='#047857', corner_radius=0)
        self.count_label.pack(pady=(10, 2))
        self.zone_counts_label = ctk.CTkLabel(self.m_frame, text='Global', font=ctk.CTkFont(size=11), text_color='#aaa', corner_radius=0)
        self.zone_counts_label.pack(pady=2)
        self.infer_label = ctk.CTkLabel(self.m_frame, text='CPU: -- ms', font=ctk.CTkFont(size=10), text_color='#666', corner_radius=0)
        self.infer_label.pack(pady=2)
        ctk.CTkButton(self.m_frame, text='CSV', command=self.export_telemetry, width=80, height=24, font=ctk.CTkFont(size=10), corner_radius=0).pack(pady=10)
        self.evidence_frame = ctk.CTkFrame(self.dash, width=280, fg_color='#1a1c1e', corner_radius=0)
        self.evidence_frame.grid(row=0, column=0, padx=10, pady=15, sticky='nsew')
        self.evidence_frame.pack_propagate(False)
        ctk.CTkLabel(self.evidence_frame, text='GALERIA DE EVIDENCIAS', font=ctk.CTkFont(size=10, weight='bold'), text_color='#059669', corner_radius=0).pack(pady=5)
        self.evidence_scroll = ctk.CTkScrollableFrame(self.evidence_frame, orientation='horizontal', fg_color='transparent', corner_radius=0)
        self.evidence_scroll.pack(fill='both', expand=True, padx=5, pady=5)
        self.evidence_items = []
        self.bar_frame = ctk.CTkFrame(self.dash, fg_color='transparent', corner_radius=0)
        self.bar_frame.grid(row=0, column=1, padx=10, pady=15, sticky='nsew')
        self.bar_modes_frame = ctk.CTkFrame(self.bar_frame, fg_color='transparent', height=25, corner_radius=0)
        self.bar_modes_frame.pack(fill='x', pady=(0, 5))
        self._update_bar_mode_buttons()
        self.bar_canvas = tk.Canvas(self.bar_frame, height=120, bg='#111', highlightthickness=0)
        self.bar_canvas.pack(fill='both', expand=True)
        self.telemetry_frame = ctk.CTkFrame(self.dash, height=140, fg_color='#111', corner_radius=0)
        self.telemetry_frame.grid(row=0, column=2, padx=10, pady=15, sticky='nsew')
        
        # Cabecera de Telemetría con botón de ajuste
        tel_header = ctk.CTkFrame(self.telemetry_frame, fg_color='transparent', corner_radius=0)
        tel_header.pack(fill='x', pady=(10, 5))
        ctk.CTkLabel(tel_header, text='RESUMEN DE SESION', font=ctk.CTkFont(size=9, weight='bold'), text_color='#444', corner_radius=0).pack(side='left', padx=10)
        self.btn_dash_cfg = ctk.CTkButton(tel_header, text='CONFIG', width=45, height=18, font=ctk.CTkFont(size=8, weight='bold'), fg_color='#1e293b', hover_color='#334155', command=self._open_dashboard_settings, corner_radius=0)
        self.btn_dash_cfg.pack(side='right', padx=10)
        self.total_ever_label = ctk.CTkLabel(self.telemetry_frame, text='0', font=ctk.CTkFont(size=32, weight='bold'), text_color='#10b981', corner_radius=0)
        self.total_ever_label.pack()
        ctk.CTkLabel(self.telemetry_frame, text='TOTAL DETECTADOS', font=ctk.CTkFont(size=9, weight='bold'), text_color='#64748b', corner_radius=0).pack(pady=(0, 10))
        self.uptime_label = ctk.CTkLabel(self.telemetry_frame, text='00:00:00', font=ctk.CTkFont(size=18, weight='bold'), text_color='#94a3b8', corner_radius=0)
        self.uptime_label.pack()
        ctk.CTkLabel(self.telemetry_frame, text='TIEMPO ACTIVO', font=ctk.CTkFont(size=9, weight='bold'), text_color='#444', corner_radius=0).pack(pady=(0, 5))
        self.breakdown_label = ctk.CTkLabel(self.telemetry_frame, text='', font=ctk.CTkFont(size=10), text_color='#64748b', justify='left', corner_radius=0)
        self.breakdown_label.pack(pady=5)
        l_frame = ctk.CTkFrame(self.dash, fg_color='transparent', corner_radius=0)
        l_frame.grid(row=0, column=3, padx=20, pady=15, sticky='nsew')
        self.log_tabs = ctk.CTkTabview(l_frame, height=140, corner_radius=0, fg_color='transparent', segmented_button_selected_color='#334155', segmented_button_unselected_color='#0f172a')
        self.log_tabs.pack(fill='both', expand=True)
        
        tab_sys = self.log_tabs.add('SISTEMA')
        tab_hits = self.log_tabs.add('HITOS')
        
        self.log_textbox = ctk.CTkTextbox(tab_sys, font=ctk.CTkFont(family='Consolas', size=10), border_color='#222', border_width=1, corner_radius=0)
        self.log_textbox.pack(fill='both', expand=True)
        
        self.log_textbox_hitos = ctk.CTkTextbox(tab_hits, font=ctk.CTkFont(family='Consolas', size=10), border_color='#222', border_width=1, corner_radius=0, text_color='#10b981')
        self.log_textbox_hitos.pack(fill='both', expand=True)
        
        # Botón ABRIR HISTORIAL ahora en el header de l_frame si cabe, o lo dejamos fuera
        log_header = ctk.CTkFrame(l_frame, fg_color='transparent', corner_radius=0)
        log_header.pack(fill='x', before=self.log_tabs)
        ctk.CTkLabel(log_header, text='LOGS DE ACTIVIDAD', font=ctk.CTkFont(size=10, weight='bold'), text_color='#444', corner_radius=0).pack(side='left', anchor='w')
        ctk.CTkButton(log_header, text='ABRIR CARPETA', width=100, height=18, font=ctk.CTkFont(size=9, weight='bold'), fg_color='#1e293b', hover_color='#334155', text_color='#94a3b8', command=self.open_history_folder, corner_radius=0).pack(side='right')

    def _build_main_area(self):
        """Área central de visualización."""
        self.main_frame = ctk.CTkFrame(self, border_width=1, border_color='#333', corner_radius=0)
        self.main_frame.grid(row=1, column=1, sticky='nsew', padx=15, pady=15)
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)
        inner = ctk.CTkFrame(self.main_frame, fg_color='transparent', corner_radius=0)
        inner.grid(row=0, column=0, sticky='nsew', padx=2, pady=2)
        inner.grid_rowconfigure(0, weight=1)
        inner.grid_columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(inner, bg='#1a1a2e', highlightthickness=0, cursor='crosshair')
        self.canvas.grid(row=0, column=0, sticky='nsew')
        self.canvas.bind('<Button-1>', self._on_video_click)
        self.canvas.bind('<Button-3>', self._on_video_right_click)
        self.canvas.bind('<Double-Button-1>', self._on_video_double_click)
        self.fullscreen_window = None  # Referencia a la ventana fullscreen
        self.controls = ctk.CTkFrame(inner, fg_color='transparent', height=50, corner_radius=0)
        self.controls.grid(row=1, column=0, sticky='ew', padx=20, pady=(10, 20))
        self.rewind_btn = ctk.CTkButton(self.controls, image=self.icons.get('back'), text='', width=40, height=32, fg_color='#1e293b', hover_color='#334155', command=lambda: self.engine.seek_back(5), corner_radius=0)
        self.rewind_btn.pack(side='left', padx=(0, 5))
        self.play_btn = ctk.CTkButton(self.controls, image=self.icons.get('play'), text='', width=40, height=32, fg_color='#059669', hover_color='#047857', command=self.toggle_pause, corner_radius=0)
        self.play_btn.pack(side='left', padx=5)
        self.forward_btn = ctk.CTkButton(self.controls, image=self.icons.get('forward'), text='', width=40, height=32, fg_color='#1e293b', hover_color='#334155', command=lambda: self.engine.seek_forward(5), corner_radius=0)
        self.forward_btn.pack(side='left', padx=(5, 10))
        self.after(500, self._update_media_controls)
        self._blink_live_indicator()

    def update_video(self):
        """Bucle de renderizado core con actualización de métricas."""
        t0 = time.time()
        if hasattr(self, 'is_labeling_mode') and self.is_labeling_mode:
            self.after(500, self.update_video)
            return
        if not self.is_paused:
            frame = self.engine.get_frame()
            if frame is not None:
                self.raw_frame = frame
            elif not getattr(self.engine, 'is_live', False) and (not self.engine.is_stream):
                self.is_paused = True
                self.play_btn.configure(text='Reproducir')
        if self.raw_frame is None:
            loading_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(loading_frame, 'CONECTANDO CON FUENTE...', (400, 360), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
            cv2.putText(loading_frame, f'Fuente: {self.url[:40]}', (400, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 1)
            self._render_canvas(loading_frame)
            self.after(100, self.update_video)
            return
        now = time.time()
        should_infer = now - self.last_infer_time >= self.infer_interval
        if self.detector.model and (not self.is_inferencing) and (not self.is_loading_model) and should_infer:
            self.last_infer_time = now
            threading.Thread(target=self.run_inference, args=(self.raw_frame.copy(),), daemon=True).start()
        has_layers = bool(self.zones or self.heatmap_enabled or self.is_drawing_zone)
        is_dual = self.detector.is_dual_mode()
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
            # Dibujar leyenda Dual Mode sobre el frame
            if is_dual:
                display = VisualPainter.draw_model_legend(
                    display,
                    self.detector.active_name,
                    self.detector.secondary_name
                )
            if has_layers:
                display = self._draw_zones_overlay(display)
                if self.heatmap_enabled:
                    display = self._render_heatmap(display)
            self._render_canvas(display)
        self._update_metrics(t0)
        if not self.is_paused and self.raw_frame is not None and self.last_detections:
            zone_data = getattr(self, '_last_zone_counts', [])
            self.data_logger.log(self.last_detections, zone_data)
        elapsed_ms = (time.time() - t0) * 1000
        target_delay = 33
        if not self.engine.is_stream and (not self.is_paused):
            fps = self.engine.get_fps()
            target_delay = 1000 / fps
        next_delay = max(1, int(target_delay - elapsed_ms))
        self.after(next_delay, self.update_video)

    def _update_metrics(self, t0):
        """Delega la actualización de métricas al pintor visual."""
        self._last_zone_counts = VisualPainter.update_sidebar_metrics(self, t0, self.last_detections, self.zones)

    def add_evidence(self, img_bgr, title, is_ok, raw_frame=None, zoom_frame=None):
        """Añade una miniatura de evidencia a la galería de la UI."""
        try:
            h, w = img_bgr.shape[:2]
            target_h = 100
            target_w = int(w * (target_h / h))
            small = cv2.resize(img_bgr, (target_w, target_h))
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(rgb)
            img_ctk = ctk.CTkImage(light_image=img_pil, dark_image=img_pil, size=(target_w, target_h))
            
            border_color = '#16a34a' if is_ok else '#7f1d1d'
            card = ctk.CTkFrame(self.evidence_scroll, fg_color='#0f172a', border_width=2, border_color=border_color, corner_radius=0)
            card.pack(side='left', padx=5, pady=2)
            lbl_img = ctk.CTkLabel(card, image=img_ctk, text='', cursor='hand2', corner_radius=0)
            lbl_img.image = img_ctk
            lbl_img.pack(padx=2, pady=2)
            
            # Click para ampliar (Pasa los 3 frames)
            def _on_click(event, bgr=img_bgr.copy(), t=title, raw=raw_frame.copy() if raw_frame is not None else None, zoom=zoom_frame.copy() if zoom_frame is not None else None):
                FullscreenImageWindow(self, bgr, t, raw_bgr=raw, zoom_bgr=zoom)
            
            lbl_img.bind("<Button-1>", _on_click)
            
            ctk.CTkLabel(card, text=title[:15], font=('', 10), corner_radius=0).pack()
            self.evidence_items.insert(0, card)
            if len(self.evidence_items) > 8:
                old = self.evidence_items.pop()
                old.destroy()
        except Exception as e:
            self.add_log(f'Error añadiendo evidencia UI: {e}')
            print(f'Error añadiendo evidencia UI: {e}')
        self.evidence_scroll.update_idletasks()

    def _update_bar_mode_buttons(self):
        """Actualiza los botones de modo de la gráfica según las zonas disponibles."""
        for w in self.bar_modes_frame.winfo_children():
            w.destroy()
        modes = ['General'] + [f'Z{i + 1}' for i in range(len(self.zones))]
        for m in modes:
            is_active = self.bar_chart_mode == m
            btn = ctk.CTkButton(self.bar_modes_frame, text=m, width=50, height=20, font=ctk.CTkFont(size=9, weight='bold'), fg_color='#059669' if is_active else '#1e293b', text_color='#ffffff' if is_active else '#94a3b8', command=lambda mode=m: self._set_bar_mode(mode), corner_radius=0)
            btn.pack(side='left', padx=2)

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
        w, h = (self.canvas.winfo_width(), self.canvas.winfo_height())
        if w < 50 or h < 50:
            return
        fh, fw = frame.shape[:2]
        aspect = fw / fh
        nw, nh = (w, int(w / aspect)) if w / h < aspect else (int(h * aspect), h)
        self._display_w, self._display_h = (nw, nh)
        self._img_offset_x, self._img_offset_y = ((w - nw) // 2, (h - nh) // 2)
        if (nw, nh) != (fw, fh):
            resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_NEAREST)
        else:
            resized = frame
        img_tk = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)))
        if not hasattr(self, '_canvas_img_id'):
            self._canvas_img_id = self.canvas.create_image(w // 2, h // 2, image=img_tk, anchor='center')
        else:
            self.canvas.itemconfig(self._canvas_img_id, image=img_tk)
            self.canvas.coords(self._canvas_img_id, w // 2, h // 2)
        self.canvas.image = img_tk
        # Si hay ventana fullscreen abierta, actualizarla también
        if self.fullscreen_window and self.fullscreen_window.winfo_exists():
            self._update_fullscreen_frame(frame)

    def _on_video_double_click(self, event):
        """Abre/cierra la ventana de video en pantalla completa."""
        if self.is_drawing_zone:
            return  # No activar fullscreen mientras se dibujan zonas
        if self.fullscreen_window and self.fullscreen_window.winfo_exists():
            self.fullscreen_window.destroy()
            self.fullscreen_window = None
            return
        self.fullscreen_window = tk.Toplevel(self)
        self.fullscreen_window.attributes('-fullscreen', True)
        self.fullscreen_window.overrideredirect(True) # Quitar bordes de ventana totalmente
        self.fullscreen_window.configure(bg='black')
        self.fullscreen_window.focus_force()
        self.fullscreen_window.bind('<Escape>', lambda e: self._close_fullscreen())
        self.fullscreen_window.bind('<Double-Button-1>', lambda e: self._close_fullscreen())
        self.fs_canvas = tk.Canvas(self.fullscreen_window, bg='black', highlightthickness=0)
        self.fs_canvas.pack(fill='both', expand=True)
        self.fullscreen_window.update() # Forzar dibujado
        self._fs_img_id = None
        self.add_log('Pantalla completa real activada. ESC para salir.')

    def _close_fullscreen(self):
        """Cierra la ventana de pantalla completa."""
        if self.fullscreen_window and self.fullscreen_window.winfo_exists():
            self.fullscreen_window.destroy()
            self.fullscreen_window = None
            self._fs_img_id = None

    def _update_fullscreen_frame(self, frame):
        """Actualiza el frame en la ventana fullscreen."""
        try:
            if not self.fullscreen_window or not self.fullscreen_window.winfo_exists():
                return
            w = self.fs_canvas.winfo_width()
            h = self.fs_canvas.winfo_height()
            if w < 50 or h < 50:
                return
            fh, fw = frame.shape[:2]
            aspect = fw / fh
            nw, nh = (w, int(w / aspect)) if w / h < aspect else (int(h * aspect), h)
            resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
            img_tk = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)))
            if not hasattr(self, '_fs_img_id') or self._fs_img_id is None:
                self._fs_img_id = self.fs_canvas.create_image(w // 2, h // 2, image=img_tk, anchor='center')
            else:
                self.fs_canvas.itemconfig(self._fs_img_id, image=img_tk)
                self.fs_canvas.coords(self._fs_img_id, w // 2, h // 2)
            self.fs_canvas.image = img_tk
        except Exception:
            pass

    def run_inference(self, frame):
        """Gestiona el hilo de inferencia y actualiza el estado de detección."""
        t_infer = time.time()
        self.is_inferencing = True
        try:
            t_classes = None if self.locked_track_id is not None else self.target_classes
            t_classes_m2 = self.target_classes_m2
            ann, all_detections = self.detector.detect(frame, target_classes=t_classes, target_classes_secondary=t_classes_m2, zones=self.zones, conf_threshold=self.conf_threshold)
            self.last_detections = all_detections
            is_dual = self.detector.is_dual_mode()
            if self.locked_track_id is not None:
                detections = [d for d in all_detections if d.get('track_id') == self.locked_track_id]
                if not detections:
                    self.focus_lost_cnt += 1
                    if self.focus_lost_cnt > 30:
                        self.locked_track_id = None
                        self.add_log('Focus Mode deshabilitado (Objetivo perdido).')
                else:
                    self.focus_lost_cnt = 0
                    ann = VisualPainter.draw_detections(frame.copy(), detections, is_focus=True, dual_mode=is_dual)
            elif self.target_classes is not None or self.target_classes_m2 is not None:
                detections = all_detections
                # Anotar frame con detecciones ya filtradas por el detector
                ann = VisualPainter.draw_detections(ann, all_detections, dual_mode=is_dual, show_trails=False)
            else:
                detections = all_detections
                # Anotar frame con todas las detecciones coloreadas por modelo (sin trails)
                ann = VisualPainter.draw_detections(ann, all_detections, dual_mode=is_dual, show_trails=False)
            with self._render_lock:
                self.annotated_frame = ann
            for d in detections:
                tid = d.get('track_id')
                if tid is not None:
                    # Registro Global
                    if tid not in self.session_seen_ids:
                        self.session_seen_ids.add(tid)
                        self.session_class_counts[d['label']] += 1
                    
                    # Registro por Zonas
                    for zi in d.get('zone_indices', []):
                        if zi >= 0:
                            if zi not in self.session_zone_data:
                                self.session_zone_data[zi] = {"ids": set(), "counts": Counter()}
                            if tid not in self.session_zone_data[zi]["ids"]:
                                self.session_zone_data[zi]["ids"].add(tid)
                                self.session_zone_data[zi]["counts"][d['label']] += 1
            
            self.total_detections_ever = len(self.session_seen_ids)
            self.event_engine.update_cumulative_stats(detections)

            def on_evidence(img, msg, ok, raw_frame=None, zoom_frame=None):
                self.after(0, lambda i=img, m=msg, o=ok, rf=raw_frame, zf=zoom_frame: self.add_evidence(i, m, o, raw_frame=rf, zoom_frame=zf))
            self.event_engine.evaluate(detections, frame=frame, source=self.url, app_log_callback=self.add_log, evidence_callback=on_evidence)
            ms = int((time.time() - t_infer) * 1000)
            infer_text = f'INFERENCIA: {ms} ms'
            if is_dual:
                infer_text = f'DUAL INFERENCIA: {ms} ms'
            self.after(0, lambda t=infer_text: self.infer_label.configure(text=t))
        finally:
            self.is_inferencing = False

    def toggle_pause(self):
        self.is_paused = not self.is_paused
        self.play_btn.configure(text='Reproducir' if self.is_paused else 'Pausa')

    def change_stream(self, new_url=None):
        if new_url:
            self.url = new_url
        self.raw_frame = None
        self.annotated_frame = None
        self.total_detections_ever = 0
        self.session_start_time = time.time()
        self.session_seen_ids = set()
        self.session_class_counts = Counter()
        self.session_zone_data = {}
        resolution = self.stream_quality_var.get()
        self.add_log(f"Configurando fuente: {(os.path.basename(self.url) if os.path.exists(self.url) else self.url[:40] + '...')} ({resolution})")

        def _reconnect_and_update():
            self.engine.reconnect(self.url, resolution=resolution)
            self.after(0, self._update_media_controls)
        threading.Thread(target=_reconnect_and_update, daemon=True).start()

    def open_source_selector(self):
        """Abre la ventana de selección de fuente de vídeo."""
        SourceSelectorWindow(self, self.url, self.change_stream)

    def _update_media_controls(self):
        """Actualiza el estado de los controles de la Top Bar y los botones de navegación."""
        is_live = getattr(self.engine, 'is_live', False)
        if is_live:
            self.rewind_btn.pack_forget()
            self.forward_btn.pack_forget()
        else:
            self.rewind_btn.pack(side='left', padx=(0, 5))
            self.play_btn.pack(side='left', padx=5)
            self.forward_btn.pack(side='left', padx=(5, 10))
        if is_live:
            self.live_dot.pack(side='left', padx=(0, 5))
            self.live_text.configure(text='EN DIRECTO', text_color='#7f1d1d')
        else:
            self.live_dot.pack_forget()
            self.live_text.configure(text='VOD / ARCHIVO', text_color='#64748b')
        url_text = self.url
        if len(url_text) > 45:
            url_text = url_text[:42] + '...'
        self.live_url_label.configure(text=f'|  {url_text}')
        is_camera = getattr(self.engine, 'is_camera', False)
        is_stream = getattr(self.engine, 'is_stream', False)
        if is_camera:
            mode_desc = 'Cámara Local'
        else:
            mode_desc = 'Streaming LIVE' if is_live else 'YouTube VOD' if is_stream else 'Video Local'
        self.add_log(f'Fuente: {mode_desc} detectada. UI actualizada.')

    def _blink_live_indicator(self):
        """Efecto de parpadeo para el punto rojo del indicador."""
        if hasattr(self, 'live_dot') and self.live_dot.winfo_exists():
            current_color = self.live_dot.cget('text_color')
            new_color = '#7f1d1d' if current_color != '#7f1d1d' else '#1a1c1e'
            self.live_dot.configure(text_color=new_color)
        self.after(800, self._blink_live_indicator)

    def add_log(self, msg, is_event=False):
        """Añade un mensaje al log (Sistema o Hitos) de forma segura desde cualquier hilo."""

        def _task():
            try:
                target = self.log_textbox_hitos if is_event else self.log_textbox
                if hasattr(self, 'log_tabs') and target.winfo_exists():
                    target.insert('end', f"[{time.strftime('%H:%M:%S')}] {msg}\n")
                    target.see('end')
                    # Si es un hito, saltar visualmente a la pestaña de hitos para avisar al usuario
                    if is_event:
                        self.log_tabs.set('HITOS')
            except:
                pass
        self.after(0, _task)

    def open_history_folder(self):
        """Abre la carpeta de evidencias y logs en el explorador de archivos."""
        from ..utils.helpers import LOGS_DIR
        if os.path.exists(LOGS_DIR):
            os.startfile(LOGS_DIR)
        else:
            self.add_log('La carpeta de historial aún no ha sido creada.')

    def export_telemetry(self):
        """Copia el archivo de log actual a una ubicación elegida por el usuario."""
        import shutil
        source = self.data_logger.get_log_path()
        if not os.path.exists(source):
            return self.add_log('No hay datos para exportar todavía.')
        dest = tk.filedialog.asksaveasfilename(defaultextension='.csv', filetypes=[('Archivo CSV', '*.csv')], initialfile=os.path.basename(source))
        if dest:
            shutil.copy2(source, dest)
            self.add_log(f'Datos exportados a: {os.path.basename(dest)}')

    def on_closing(self):
        self.engine.release()
        self.destroy()

    def _save_config(self):
        save_app_config(self.url, self.zones, self.target_classes, self.dashboard_config)

    def _load_config(self):
        cfg = load_app_config(self.url)
        if cfg:
            self.zones, self.target_classes = (cfg.get('zones', []), cfg.get('target_classes'))
            saved_dash = cfg.get('dashboard_config')
            if saved_dash:
                self.dashboard_config.update(saved_dash)
            self._update_bar_mode_buttons()

    def _update_world_prompt_visibility(self, visible):
        if visible:
            self.world_prompt_frame.pack(pady=5, padx=20, fill='x', after=self.scale_selector)
        else:
            self.world_prompt_frame.pack_forget()

    def _on_no_model_click(self):
        """Desactiva la IA y limpia selecciones de modelos (primario y secundario)."""
        self.model_selector.set('Seleccionar...')
        self.scale_selector.configure(values=[])
        self.scale_selector.set('')
        self.detector.model = None
        self.detector.active_name = None
        # Limpiar también el modelo secundario
        self.detector.clear_secondary_model()
        self._deactivate_m2_ui()
        self.add_log('IA Desactivada. Renderizado en crudo.')
        self._update_world_prompt_visibility(False)

    def _on_family_change(self, family):
        """Actualiza el selector de escalas según la familia."""
        if not family or family == 'Seleccionar...':
            return
        scales = sorted(self.detector.architectures.get(family, {}).get('aliases', {}).keys())
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
                self.after(0, lambda: self.add_log('Modelo reconfigurado con exito.'))
            else:
                self.after(0, lambda: self.add_log('Error al aplicar prompt.'))
        threading.Thread(target=task, daemon=True).start()

    def _on_config_change(self, _=None):
        """Recarga el modelo con la nueva configuración de familia/escala."""
        family = self.model_selector.get()
        alias = self.scale_selector.get()
        if not family or not alias or family == 'Seleccionar...':
            return

        def load():
            self.is_loading_model = True
            try:
                self.add_log(f'Cargando modelo: {family} ({alias})...')
                success = self.detector.change_model(family, alias)
                if success:
                    backend = getattr(self.detector, 'active_device', 'CPU')
                    vendor = self.detector.hardware_diag.get('gpu_vendor', 'Unknown')

                    def _update_ui():
                        self.hw_label.configure(text=f'PROCESAMIENTO: {backend} ({vendor})', text_color='#10b981')
                        self.add_log(f'Modelo {success} cargado correctamente.')
                        
                        # Actualizar visibilidad del prompt de búsqueda textual (Zero-Shot)
                        self._update_world_prompt_visibility(self.detector.is_zero_shot_active)
                        
                    self.after(0, _update_ui)
                else:

                    def _update_err():
                        self.hw_label.configure(text='ERROR AL CARGAR MODELO', text_color='#7f1d1d')
                        self.add_log('Error critico: No se pudo cargar el modelo.')
                    self.after(0, _update_err)
            except Exception as e:
                self.add_log(f'Error en hilo de carga: {e}')
            finally:
                self.is_loading_model = False
        threading.Thread(target=load, daemon=True).start()

    # --- Modelo Secundario (M2) ---

    def _on_m2_toggle(self):
        """Alterna la activación del modelo secundario."""
        if self.m2_active:
            self._deactivate_m2_ui()
            self.detector.clear_secondary_model()
            self.add_log('[M2] Modelo secundario desactivado.')
            self.hw_label.configure(text=f"{self.detector.hardware_diag['gpu_name'][:20]} | {self.detector.hardware_diag['best_backend'].upper()}", text_color='#10b981')
        else:
            self.m2_active = True
            self.m2_toggle_btn.configure(text='DESACTIVAR M2', fg_color='#f59e0b', text_color='#0f172a', border_color='#f59e0b')
            self.m2_controls_frame.pack(pady=(2, 6), fill='x')
            self.m2_family_selector.set('Seleccionar...')
            self.m2_scale_selector.configure(values=[])
            self.m2_scale_selector.set('')
            self.m2_status_label.configure(text='Selecciona familia y escala', text_color='#f59e0b')
            self.add_log('[M2] Modo dual activado. Selecciona un modelo secundario.')

    def _deactivate_m2_ui(self):
        """Resetea la UI del M2 sin tocar el detector."""
        self.m2_active = False
        self.m2_toggle_btn.configure(text='ACTIVAR M2', fg_color='#1e293b', text_color='#f59e0b', border_color='#f59e0b')
        self.m2_controls_frame.pack_forget()

    def _on_m2_family_change(self, family):
        """Actualiza escalas del M2 según la familia seleccionada."""
        if not family:
            return
        scales = sorted(self.detector.architectures.get(family, {}).get('aliases', {}).keys())
        self.m2_scale_selector.configure(values=scales)
        if scales:
            self.m2_scale_selector.set(scales[0])
            self._on_m2_config_change()

    def _on_m2_config_change(self, _=None):
        """Carga el modelo secundario seleccionado."""
        family = self.m2_family_selector.get()
        alias = self.m2_scale_selector.get()
        if not family or not alias or family == 'Seleccionar...':
            return

        def load_m2():
            self.is_loading_model = True
            try:
                self.add_log(f'[M2] Cargando: {family} ({alias})...')
                success = self.detector.change_secondary_model(family, alias)
                if success:
                    def _update_ui():
                        self.m2_status_label.configure(text=f'✓ {success}', text_color='#10b981')
                        self.hw_label.configure(text='DUAL MODE ACTIVO', text_color='#f59e0b')
                        self.add_log(f'[M2] Modelo {success} cargado correctamente.')
                    self.after(0, _update_ui)
                else:
                    def _update_err():
                        self.m2_status_label.configure(text='Error cargando M2', text_color='#7f1d1d')
                        self.add_log('[M2] Error: No se pudo cargar el modelo secundario.')
                    self.after(0, _update_err)
            except Exception as e:
                self.add_log(f'[M2] Error en hilo de carga: {e}')
            finally:
                self.is_loading_model = False
        threading.Thread(target=load_m2, daemon=True).start()

    def _on_model_added(self):
        self.model_selector.configure(values=list(self.detector.architectures.keys()))

    def open_class_filter(self, is_secondary=False):
        detector_model = self.detector.secondary_model if is_secondary else self.detector.model
        if not self.detector or not detector_model:
            self.add_log(f"No hay un modelo {'secundario ' if is_secondary else ''}cargado todavía.")
            return
        
        # Obtener nombres de clases del modelo específico
        names_dict = getattr(detector_model, 'names', {})
        if not names_dict:
            classes = {i: f"Clase {i}" for i in range(80)}
        else:
            classes = {int(k): str(v) for k, v in names_dict.items()}
            
        current_targets = self.target_classes_m2 if is_secondary else self.target_classes
        label_prefix = "[M2] " if is_secondary else ""
        
        def on_applied(new_targets):
            if is_secondary:
                self.target_classes_m2 = new_targets
                self.add_log(f"[M2] Filtro {'retirado' if new_targets is None else 'aplicado'}.")
            else:
                self._on_filter_applied(new_targets)
        
        self.add_log(f"Abriendo filtro {label_prefix}: {len(classes)} clases detectadas.")
        ClassFilterWindow(self, self.detector, current_targets, on_applied, custom_classes=classes)

    def _on_filter_applied(self, new_targets):
        self.target_classes = new_targets
        self.annotated_frame = None
        if self.target_classes is None:
            self.add_log('Filtro M1 retirado: detectando todas las clases.')
        else:
            self.add_log(f'Filtro M1 aplicado: {len(self.target_classes)} clase(s).')
        self._save_config()

    def _open_dashboard_settings(self):
        from .components import DashboardSettingsWindow
        # Obtener clases disponibles de los modelos cargados
        all_classes = set()
        if self.detector.model:
            names = getattr(self.detector.model, 'names', {})
            all_classes.update(names.values())
        if self.detector.secondary_model:
            names = getattr(self.detector.secondary_model, 'names', {})
            all_classes.update(names.values())
        
        DashboardSettingsWindow(self, self.dashboard_config, sorted(list(all_classes)))

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
        """Alterna el modo de dibujo de zonas y gestiona el guardado al finalizar."""
        if self.is_drawing_zone:
            # Finalizar dibujo: Intentar guardar si hay suficientes puntos
            if len(self.current_zone) >= 3:
                self.zones.append(self.current_zone)
                self.add_log(f"Zona {len(self.zones)} delimitada con éxito.")
                self._save_config()
                self._update_bar_mode_buttons()
            elif self.current_zone:
                self.add_log("Dibujo cancelado: se requieren al menos 3 puntos.")
            
            self.current_zone = []
            self.is_drawing_zone = False
            self.draw_btn.configure(text='DELIMITAR ZONAS', fg_color='#1e293b')
        else:
            # Iniciar dibujo
            self.is_drawing_zone = True
            self.draw_btn.configure(text='LISTO (GUARDAR)', fg_color='#059669')
            self.add_log("Modo dibujo: Haz clic en el video para marcar puntos. Clic derecho o botón 'Listo' para terminar.")

    def clear_zones(self):
        self.zones = []
        self._save_config()
        self._update_bar_mode_buttons()

    def _on_video_click(self, e):
        if self.is_drawing_zone:
            self.current_zone.append(((e.x - self._img_offset_x) / self._display_w, (e.y - self._img_offset_y) / self._display_h))
            return
        if not self.last_detections:
            self.locked_track_id = None
            return
        try:
            h_frame, w_frame = self.raw_frame.shape[:2]
            click_x = e.x - self._img_offset_x
            click_y = e.y - self._img_offset_y
            nx = click_x / self._display_w
            ny = click_y / self._display_h
            fx, fy = (nx * w_frame, ny * h_frame)
            found_new = None
            for d in self.last_detections:
                x1, y1, x2, y2 = d['bbox']
                if x1 - 10 <= fx <= x2 + 10 and y1 - 10 <= fy <= y2 + 10:
                    if d.get('track_id') is not None:
                        found_new = d['track_id']
                        break
            if found_new is not None:
                if getattr(self, 'locked_track_id', None) == found_new:
                    self.locked_track_id = None
                    self.add_log('Focus Mode: Objetivo liberado.')
                else:
                    self.locked_track_id = found_new
                    self.focus_lost_cnt = 0
                    self.add_log(f'Focus Mode: Fijado objetivo ID {self.locked_track_id}')
            elif getattr(self, 'locked_track_id', None) is not None:
                self.locked_track_id = None
                self.add_log('Focus Mode: Deshabilitado (clic fuera).')
        except Exception as _e:
            print(f'Error detectando objeto por clic: {_e}')

    def _on_video_right_click(self, e):
        if self.is_drawing_zone:
            # Delegar la lógica de guardado y cambio de UI al método centralizado
            self.toggle_zone_drawing()

    def _on_conf_change(self, value):
        self.conf_threshold = value
        self.conf_label.configure(text=f'{int(value * 100)}%')

    def _on_interval_change(self, value):
        self.infer_interval = value
        if value == 0:
            self.interval_label.configure(text='MAX FPS')
        else:
            self.interval_label.configure(text=f'{value:.1f}s')

    def _toggle_heatmap(self):
        self.heatmap_enabled = self.heatmap_switch.get()

    def take_capture(self):
        """Captura instantánea: pausa, congela el frame actual, y abre el anotador multi-clase."""
        try:
            if self.raw_frame is None:
                self.add_log('Error: No hay senal de video para capturar.')
                return
            ds_name = self.capture_entry.get().strip()
            if not ds_name:
                ds_name = f"dataset_{time.strftime('%d%m%Y_%H%M%S')}"
            ds_name = ds_name.lower().replace(' ', '_')
            if not os.path.exists(DATASETS_DIR):
                os.makedirs(DATASETS_DIR, exist_ok=True)
            dataset_dir = os.path.join(DATASETS_DIR, ds_name)
            self.add_log(f"Iniciando captura para: '{ds_name}'")
            if ds_name not in self._checked_datasets and os.path.exists(dataset_dir):
                resp = messagebox.askyesnocancel('Dataset Existente', f"El dataset '{ds_name}' ya existe.\n\n¿Borrar y empezar de nuevo?", parent=self)
                if resp is None:
                    self.add_log('Captura cancelada por el usuario.')
                    return
                if resp:
                    shutil.rmtree(dataset_dir)
                    self.add_log('Dataset borrado.')
            self._checked_datasets.add(ds_name)
            captured_frame = self.raw_frame.copy()
            self._was_paused_before_capture = self.is_paused
            if not self.is_paused:
                self.toggle_pause()
            actual_dir = ensure_dataset_structure(ds_name)
            base_name = get_next_capture_filename(ds_name, actual_dir)
            self._enter_labeling_mode()
            AnnotationWindow(self, captured_frame, ds_name, base_name, actual_dir, self._on_capture_saved, self._exit_labeling_mode)
        except Exception as e:
            self.add_log(f'FALLO GLOBAL CAPTURA: {e}')
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
        self.add_log(f'Captura guardada: {name} ({boxes_count} bboxes).')
        was_paused = getattr(self, '_was_paused_before_capture', True)
        if not was_paused and self.is_paused:
            self.toggle_pause()

    def export_dataset_zip(self):
        """Exporta solo las imágenes etiquetadas y su metadata en un archivo ZIP."""
        import zipfile
        from ..utils.helpers import DATASETS_DIR
        if not os.path.exists(DATASETS_DIR) or not os.listdir(DATASETS_DIR):
            self.add_log('No hay datasets para exportar.')
            return
        dest = tk.filedialog.asksaveasfilename(defaultextension='.zip', filetypes=[('Archivo ZIP', '*.zip')], initialfile='dataset_export_selective.zip')
        if not dest:
            return
        try:
            total_files = 0
            with zipfile.ZipFile(dest, 'w', zipfile.ZIP_DEFLATED) as zf:
                for ds_name in os.listdir(DATASETS_DIR):
                    ds_path = os.path.join(DATASETS_DIR, ds_name)
                    if not os.path.isdir(ds_path):
                        continue
                    for master in ['data.yaml', 'classes.txt']:
                        m_path = os.path.join(ds_path, master)
                        if os.path.exists(m_path):
                            zf.write(m_path, os.path.join(ds_name, master))
                            total_files += 1
                    labels_dir = os.path.join(ds_path, 'labels', 'train')
                    images_dir = os.path.join(ds_path, 'images', 'train')
                    if os.path.exists(labels_dir):
                        for lab_file in os.listdir(labels_dir):
                            if lab_file.endswith('.txt'):
                                base = os.path.splitext(lab_file)[0]
                                lab_full = os.path.join(labels_dir, lab_file)
                                img_found = None
                                for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                                    test_img = os.path.join(images_dir, base + ext)
                                    if os.path.exists(test_img):
                                        img_found = test_img
                                        break
                                if img_found:
                                    zf.write(lab_full, os.path.join(ds_name, 'labels', 'train', lab_file))
                                    zf.write(img_found, os.path.join(ds_name, 'images', 'train', os.path.basename(img_found)))
                                    total_files += 2
            if total_files > 0:
                self.add_log(f"Exportacion selectiva completa: {total_files} archivos en '{os.path.basename(dest)}'.")
            else:
                self.add_log('No se encontraron imagenes con etiquetas para exportar.')
                if os.path.exists(dest):
                    os.remove(dest)
        except Exception as e:
            self.add_log(f'Error exportando dataset: {e}')

    def import_zip_dataset(self):
        """Importa un ZIP de imágenes, las extrae y abre el anotador secuencial."""
        import zipfile
        from ..utils.helpers import DATASETS_DIR, ensure_dataset_structure
        file_path = tk.filedialog.askopenfilename(title='Seleccionar ZIP de imágenes para etiquetar', filetypes=[('Archivo ZIP', '*.zip')])
        if not file_path:
            return
        ds_name = self.capture_entry.get().strip()
        if not ds_name:
            ds_name = f"import_{time.strftime('%H%M%S')}"
        ds_name = ds_name.lower().replace(' ', '_')
        dataset_dir = ensure_dataset_structure(ds_name)
        if not dataset_dir:
            self.add_log('Error creando estructura de dataset.')
            return
        self.add_log(f'Importando ZIP: {os.path.basename(file_path)}...')
        try:
            target_img_dir = os.path.join(dataset_dir, 'images', 'train')
            image_paths = []
            with zipfile.ZipFile(file_path, 'r') as zf:
                for info in zf.infolist():
                    if info.is_dir():
                        continue
                    ext = os.path.splitext(info.filename)[1].lower()
                    if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                        filename = os.path.basename(info.filename)
                        dest_path = os.path.join(target_img_dir, filename)
                        with zf.open(info) as source, open(dest_path, 'wb') as target:
                            shutil.copyfileobj(source, target)
                        image_paths.append(dest_path)
            if not image_paths:
                self.add_log('El ZIP no contiene imagenes validas.')
                return
            self.add_log(f"{len(image_paths)} imagenes importadas en '{ds_name}'.")
            self._enter_labeling_mode()
            AnnotationWindow(self, ds_name=ds_name, dataset_dir=dataset_dir, on_save=self._on_capture_saved, on_close=self._exit_labeling_mode, image_files=image_paths)
        except Exception as e:
            self.add_log(f'Error al importar ZIP: {e}')

    def resume_labeling(self):
        """Busca datasets existentes y abre el anotador para continuar el trabajo."""
        if not os.path.exists(DATASETS_DIR):
            self.add_log('No existen datasets todavía.')
            return
        datasets = [d for d in os.listdir(DATASETS_DIR) if os.path.isdir(os.path.join(DATASETS_DIR, d))]
        if not datasets:
            self.add_log('No se encontraron carpetas de dataset.')
            return
        popup = ctk.CTkToplevel(self, corner_radius=0)
        popup.title('Reanudar Etiquetado')
        popup.geometry('300x400')
        popup.grab_set()
        ctk.CTkLabel(popup, text='SELECCIONA DATASET', font=ctk.CTkFont(weight='bold'), corner_radius=0).pack(pady=10)
        scroll = ctk.CTkScrollableFrame(popup, corner_radius=0)
        scroll.pack(fill='both', expand=True, padx=10, pady=10)

        def _start_resume(ds_name):
            popup.destroy()
            ds_path = os.path.join(DATASETS_DIR, ds_name)
            img_dir = os.path.join(ds_path, 'images', 'train')
            if not os.path.exists(img_dir):
                self.add_log(f"El dataset '{ds_name}' no tiene carpeta de imagenes.")
                return
            image_paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            if not image_paths:
                self.add_log(f"No se encontraron imagenes en '{ds_name}'.")
                return
            self.add_log(f'Reanudando sesion: {ds_name} ({len(image_paths)} imagenes)')
            self._enter_labeling_mode()
            AnnotationWindow(self, ds_name=ds_name, dataset_dir=ds_path, on_save=self._on_capture_saved, on_close=self._exit_labeling_mode, image_files=image_paths)
        for ds in sorted(datasets):
            ctk.CTkButton(scroll, text=ds, command=lambda d=ds: _start_resume(d), corner_radius=0).pack(fill='x', pady=2)

    def _enter_labeling_mode(self):
        """Suspende el motor de vídeo y entra en modo ahorro de recursos para etiquetar."""
        self._was_paused_before_labeling = self.is_paused
        self.is_labeling_mode = True
        self.add_log('Modo Etiquetado: Suspendiendo motor de video para ahorrar recursos...')
        self.engine.release()
        self.hw_label.configure(text='SISTEMA SUSPENDIDO (ETIQUETANDO)', text_color='#94a3b8')

    def _exit_labeling_mode(self):
        """Sale del modo etiquetado y reactiva el motor de vídeo."""
        self.is_labeling_mode = False
        self.add_log('Modo Etiquetado finalizado: Reactivando motor...')
        was_paused = getattr(self, '_was_paused_before_labeling', False)
        if self.is_paused != was_paused:
            self.toggle_pause()
        self.change_stream()

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
            if interval <= 0:
                raise ValueError
        except ValueError:
            self.add_log('Intervalo de autocaptura no valido (usa un numero > 0).')
            return
        ds_name = self.capture_entry.get().strip()
        if not ds_name:
            ds_name = f"auto_{time.strftime('%d%m%Y_%H%M')}"
        ds_name = ds_name.lower().replace(' ', '_')
        from ..utils.helpers import ensure_dataset_structure
        self.auto_dataset_dir = ensure_dataset_structure(ds_name)
        if not self.auto_dataset_dir:
            self.add_log('Error creando carpeta para autocaptura.')
            return
        self.auto_ds_name = ds_name
        self.is_auto_capturing = True
        self.auto_capture_btn.configure(text='PARAR', fg_color='#7f1d1d', hover_color='#450a0a')
        self.add_log(f"Autocaptura iniciada: cada {interval}s en '{ds_name}'")
        self._run_auto_capture_loop()

    def stop_auto_capture(self):
        """Detiene la autocaptura y abre la carpeta de resultados."""
        self.is_auto_capturing = False
        self.auto_capture_btn.configure(text='AUTO', fg_color='#10b981', hover_color='#059669')
        self.add_log('Autocaptura detenida.')
        if hasattr(self, 'auto_dataset_dir') and os.path.exists(self.auto_dataset_dir):
            img_path = os.path.abspath(os.path.join(self.auto_dataset_dir, 'images', 'train'))
            if os.path.exists(img_path):
                self.add_log(f'Abriendo carpeta: {img_path}')
                try:
                    os.startfile(img_path)
                except Exception as e:
                    self.add_log(f'No se pudo abrir la carpeta: {e}')

    def _run_auto_capture_loop(self):
        """Bucle interno de guardado de frames."""
        if not self.is_auto_capturing:
            return
        if self.raw_frame is not None:
            from ..utils.helpers import get_next_capture_filename
            base_name = get_next_capture_filename(self.auto_ds_name, self.auto_dataset_dir)
            full_path = os.path.join(self.auto_dataset_dir, 'images', 'train', f'{base_name}.jpg')
            cv2.imwrite(full_path, self.raw_frame)
            self.add_log(f'Frame guardado: {base_name}')
        else:
            self.add_log('Autocaptura: Esperando senal de video...')
        try:
            interval_ms = int(float(self.auto_capture_interval.get()) * 1000)
        except:
            interval_ms = 5000
        self.after(interval_ms, self._run_auto_capture_loop)