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
        self.target_classes = None
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
        self.bar_chart_mode = 'General'
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
        icon_names = ['source', 'favs', 'alerts', 'settings', 'night', 'day', 'info', 'play', 'pause', 'back', 'forward']
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
        """Construye el panel lateral compacto y estático (Sin Scroll)."""
        self.grid_columnconfigure(0, weight=0, minsize=280)
        self.grid_columnconfigure(1, weight=1)
        self.sidebar = ctk.CTkFrame(self, width=280, corner_radius=0)
        self.sidebar.grid(row=1, column=0, rowspan=2, sticky='nsew')
        self.sidebar.grid_propagate(False)
        header_frame = ctk.CTkFrame(self.sidebar, fg_color='transparent', height=10, corner_radius=0)
        header_frame.pack(pady=(5, 5), padx=20, fill='x')
        self._section('MODELO IA')
        self.no_model_selector = ctk.CTkSegmentedButton(self.sidebar, values=['DESACTIVAR PROCESAMIENTO IA'], command=lambda _: self._on_no_model_click(), height=30, fg_color='#1e293b', selected_color='#10b981', corner_radius=0)
        self.no_model_selector.pack(pady=(0, 8), padx=20, fill='x')
        families = list(self.detector.architectures.keys())
        self.model_selector = ctk.CTkSegmentedButton(self.sidebar, values=families, command=self._on_family_change, height=28, fg_color='#1e293b', selected_color='#3b82f6', corner_radius=0)
        self.model_selector.pack(pady=(0, 6), padx=20, fill='x')
        self.scale_selector = ctk.CTkSegmentedButton(self.sidebar, command=self._on_config_change, height=28, fg_color='#1e293b', selected_color='#3b82f6', corner_radius=0)
        self.scale_selector.pack(pady=(0, 8), padx=20, fill='x')
        self.world_prompt_frame = ctk.CTkFrame(self.sidebar, fg_color='#1e293b', border_width=1, border_color='#059669', corner_radius=0)
        ctk.CTkLabel(self.world_prompt_frame, text='BÚSQUEDA UNIVERSAL AI', font=ctk.CTkFont(size=10, weight='bold'), text_color='#059669', corner_radius=0).pack(pady=(2, 0))
        w_row = ctk.CTkFrame(self.world_prompt_frame, fg_color='transparent', corner_radius=0)
        w_row.pack(fill='x', padx=5, pady=5)
        self.world_entry = ctk.CTkEntry(w_row, placeholder_text='Prompt...', height=26, font=ctk.CTkFont(size=11), corner_radius=0)
        self.world_entry.pack(side='left', fill='x', expand=True, padx=(0, 2))
        self.world_entry.bind('<Return>', lambda _: self.apply_world_prompt())
        ctk.CTkButton(w_row, text='OK', width=40, height=26, fg_color='#059669', hover_color='#047857', text_color='#000', font=ctk.CTkFont(size=10, weight='bold'), command=self.apply_world_prompt, corner_radius=0).pack(side='right')
        self._section('ANÁLISIS TÁCTICO')
        self.conf_slider = ctk.CTkSlider(self.sidebar, from_=0.01, to=0.99, number_of_steps=98, command=self._on_conf_change, corner_radius=0)
        self.conf_slider.set(0.35)
        self.conf_slider.pack(pady=(0, 2), padx=20, fill='x')
        self.conf_label = ctk.CTkLabel(self.sidebar, text='Confianza: 35%', font=ctk.CTkFont(size=11), corner_radius=0)
        self.conf_label.pack(pady=(0, 6), padx=20, anchor='w')
        self.interval_slider = ctk.CTkSlider(self.sidebar, from_=0.0, to=5.0, number_of_steps=50, command=self._on_interval_change, corner_radius=0)
        self.interval_slider.set(self.infer_interval)
        self.interval_slider.pack(pady=(2, 2), padx=20, fill='x')
        self.interval_label = ctk.CTkLabel(self.sidebar, text=f'Muestreo: {self.infer_interval:.1f}s', font=ctk.CTkFont(size=11), corner_radius=0)
        self.interval_label.pack(pady=(0, 6), padx=20, anchor='w')
        f_row = ctk.CTkFrame(self.sidebar, fg_color='transparent', corner_radius=0)
        f_row.pack(fill='x', padx=20, pady=(5, 10))
        self.heatmap_switch = ctk.CTkSwitch(f_row, text='Mapa de Calor', command=self._toggle_heatmap, progress_color='#10b981', corner_radius=0)
        self.heatmap_switch.pack(side='left')
        ctk.CTkButton(f_row, text='Filtro de Clases', fg_color='#3b82f6', hover_color='#2563eb', command=self.open_class_filter, height=28, corner_radius=0).pack(side='right', fill='x', expand=True, padx=(10, 0))
        z_btns = ctk.CTkFrame(self.sidebar, fg_color='transparent', corner_radius=0)
        z_btns.pack(pady=(5, 5), padx=20, fill='x')
        self.draw_btn = ctk.CTkButton(z_btns, text='DELIMITAR ZONAS', command=self.toggle_zone_drawing, width=110, height=32, fg_color='#1e293b', border_width=1, border_color='#334155', font=ctk.CTkFont(size=11, weight='bold'), corner_radius=0)
        self.draw_btn.pack(side='left', padx=0, fill='x', expand=True)
        ctk.CTkButton(z_btns, text='BORRAR ZONAS', command=self.clear_zones, fg_color='#450a0a', hover_color='#7f1d1d', height=32, font=ctk.CTkFont(size=11, weight='bold'), corner_radius=0).pack(side='left', fill='x', expand=True, padx=0)
        ctk.CTkButton(self.sidebar, text=' GESTOR DE EVENTOS / ALERTAS', image=self.icons.get('alerts'), command=self.open_events_config, height=34, fg_color='#f59e0b', hover_color='#d97706', text_color='#0f172a', font=ctk.CTkFont(size=11, weight='bold'), corner_radius=0).pack(pady=(5, 10), padx=20, fill='x')
        self._section('CONTROL DE CAPTURA')
        cap_frame = ctk.CTkFrame(self.sidebar, fg_color='#1e293b', border_width=1, border_color='#334155', corner_radius=0)
        cap_frame.pack(pady=(0, 10), padx=20, fill='x')
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
        bottom = ctk.CTkFrame(self.sidebar, fg_color='transparent', corner_radius=0)
        bottom.pack(side='bottom', fill='x', padx=20, pady=(4, 15))
        self.hw_label = ctk.CTkLabel(bottom, text=f"{self.detector.hardware_diag['gpu_name'][:20]} | {self.detector.hardware_diag['best_backend'].upper()}", font=ctk.CTkFont(size=10), text_color='#10b981', corner_radius=0)
        self.hw_label.pack(fill='x', pady=(0, 2))
        self.antigravity_btn = ctk.CTkLabel(bottom, text='Antigravity System', font=ctk.CTkFont(size=10, slant='italic'), text_color='#059669', cursor='hand2', corner_radius=0)
        self.antigravity_btn.pack(fill='x')
        self.antigravity_btn.bind('<Button-1>', lambda e: webbrowser.open('https://antigravity.google'))

    def _section(self, text):
        ctk.CTkLabel(self.sidebar, text=text, font=ctk.CTkFont(size=12, weight='bold'), text_color='#94a3b8', corner_radius=0).pack(pady=(15, 6), padx=20, anchor='w')

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
        ctk.CTkLabel(self.telemetry_frame, text='RESUMEN DE SESION', font=ctk.CTkFont(size=9, weight='bold'), text_color='#444', corner_radius=0).pack(pady=(10, 5))
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
        ctk.CTkLabel(l_frame, text='EVENTOS DEL SISTEMA', font=ctk.CTkFont(size=10, weight='bold'), text_color='#444', corner_radius=0).pack(anchor='w')
        self.log_textbox = ctk.CTkTextbox(l_frame, font=ctk.CTkFont(family='Consolas', size=10), height=140, border_color='#222', border_width=1, corner_radius=0)
        self.log_textbox.pack(fill='both', expand=True)

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

    def add_evidence(self, img_bgr, title, is_ok):
        """Añade una miniatura de evidencia a la galería de la UI."""
        try:
            h, w = img_bgr.shape[:2]
            target_h = 100
            target_w = int(w * (target_h / h))
            small = cv2.resize(img_bgr, (target_w, target_h))
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(rgb)
            img_tk = ImageTk.PhotoImage(img_pil)
            border_color = '#16a34a' if is_ok else '#7f1d1d'
            card = ctk.CTkFrame(self.evidence_scroll, fg_color='#0f172a', border_width=2, border_color=border_color, corner_radius=0)
            card.pack(side='left', padx=5, pady=2)
            lbl_img = ctk.CTkLabel(card, image=img_tk, text='', corner_radius=0)
            lbl_img.image = img_tk
            lbl_img.pack(padx=2, pady=2)
            ctk.CTkLabel(card, text=title[:15], font=('', 10), corner_radius=0).pack()
            self.evidence_items.insert(0, card)
            if len(self.evidence_items) > 8:
                old = self.evidence_items.pop()
                old.destroy()
        except Exception as e:
            print(f'Error añadiendo evidencia UI: {e}')

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

    def run_inference(self, frame):
        """Gestiona el hilo de inferencia y actualiza el estado de detección."""
        t_infer = time.time()
        self.is_inferencing = True
        try:
            t_classes = None if self.locked_track_id is not None else self.target_classes
            ann, all_detections = self.detector.detect(frame, target_classes=t_classes, zones=self.zones, conf_threshold=self.conf_threshold)
            self.last_detections = all_detections
            if self.locked_track_id is not None:
                detections = [d for d in all_detections if d.get('track_id') == self.locked_track_id]
                if not detections:
                    self.focus_lost_cnt += 1
                    if self.focus_lost_cnt > 30:
                        self.locked_track_id = None
                        self.add_log('Focus Mode deshabilitado (Objetivo perdido).')
                else:
                    self.focus_lost_cnt = 0
                    ann = VisualPainter.draw_detections(frame.copy(), detections, is_focus=True)
            elif self.target_classes is not None:
                detections = [d for d in all_detections if d.get('class_id') in self.target_classes]
            else:
                detections = all_detections
            with self._render_lock:
                self.annotated_frame = ann
            for d in detections:
                tid = d.get('track_id')
                if tid is not None and tid not in self.session_seen_ids:
                    self.session_seen_ids.add(tid)
                    self.session_class_counts[d['label']] += 1
                    self.total_detections_ever = len(self.session_seen_ids)
            self.event_engine.update_cumulative_stats(detections)

            def on_evidence(img, msg, ok):
                self.after(0, lambda: self.add_evidence(img, msg, ok))
            self.event_engine.evaluate(detections, frame=frame, app_log_callback=self.add_log, evidence_callback=on_evidence)
            ms = int((time.time() - t_infer) * 1000)
            self.after(0, lambda: self.infer_label.configure(text=f'INFERENCIA: {ms} ms'))
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
        if is_camera:
            mode_desc = 'Cámara Local'
        else:
            mode_desc = 'Streaming LIVE' if is_live else 'YouTube VOD' if is_stream else 'Video Local'
        self.add_log(f'Modo {mode_desc} detectado. UI actualizada.')

    def _blink_live_indicator(self):
        """Efecto de parpadeo para el punto rojo del indicador."""
        if hasattr(self, 'live_dot') and self.live_dot.winfo_exists():
            current_color = self.live_dot.cget('text_color')
            new_color = '#7f1d1d' if current_color != '#7f1d1d' else '#1a1c1e'
            self.live_dot.configure(text_color=new_color)
        self.after(800, self._blink_live_indicator)

    def add_log(self, msg):
        """Añade un mensaje al log de forma segura desde cualquier hilo."""

        def _task():
            try:
                if hasattr(self, 'log_textbox') and self.log_textbox.winfo_exists():
                    self.log_textbox.insert('end', f"[{time.strftime('%H:%M:%S')}] {msg}\n")
                    self.log_textbox.see('end')
            except:
                pass
        self.after(0, _task)

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
        save_app_config(self.url, self.zones, self.target_classes)

    def _load_config(self):
        cfg = load_app_config(self.url)
        if cfg:
            self.zones, self.target_classes = (cfg.get('zones', []), cfg.get('target_classes'))

    def _update_world_prompt_visibility(self, visible):
        if visible:
            self.world_prompt_frame.pack(pady=5, padx=20, fill='x', after=self.scale_selector)
        else:
            self.world_prompt_frame.pack_forget()

    def _on_no_model_click(self):
        """Desactiva la IA y limpia selecciones de modelos."""
        self.no_model_selector.set('DESACTIVAR PROCESAMIENTO IA')
        self.model_selector.set('')
        self.scale_selector.configure(values=[])
        self.scale_selector.set('')
        self.detector.model = None
        self.detector.active_name = None
        self.add_log('IA Desactivada. Renderizado en crudo.')
        self._update_world_prompt_visibility(False)

    def _on_family_change(self, family):
        """Actualiza el selector de escalas según la familia."""
        if not family:
            return
        self.no_model_selector.set('')
        self._update_world_prompt_visibility('world' in family.lower())
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
        if not family or not alias:
            return
        self.no_model_selector.set('')

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

    def _on_model_added(self):
        self.model_selector.configure(values=list(self.detector.architectures.keys()))

    def open_class_filter(self):
        if not self.detector or not self.detector.model:
            self.add_log('No hay un modelo cargado todavía.')
            return
        classes = self.detector.get_class_names()
        self.add_log(f'Abriendo filtro: {len(classes)} clases detectadas en el modelo.')
        ClassFilterWindow(self, self.detector, self.target_classes, self._on_filter_applied)

    def _on_filter_applied(self, new_targets):
        self.target_classes = new_targets
        self.annotated_frame = None
        if self.target_classes is None:
            self.add_log('Filtro retirado: detectando todas las clases.')
        else:
            self.add_log(f'Filtro aplicado: {len(self.target_classes)} clase(s) seleccionada(s).')
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
        self.draw_btn.configure(text='Listo' if self.is_drawing_zone else 'Dibujar')

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
        if self.is_drawing_zone and len(self.current_zone) >= 3:
            self.zones.append(self.current_zone)
            self.current_zone = []
            self.toggle_zone_drawing()
            self._save_config()
            self._update_bar_mode_buttons()

    def _on_conf_change(self, value):
        self.conf_threshold = value
        self.conf_label.configure(text=f'Confianza: {int(value * 100)}%')

    def _on_interval_change(self, value):
        self.infer_interval = value
        if value == 0:
            self.interval_label.configure(text='Muestreo: Cada frame')
        else:
            self.interval_label.configure(text=f'Muestreo: {value:.1f}s')

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