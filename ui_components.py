import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import os
import json
import shutil
import threading
import time
import webbrowser
import platform
import importlib.metadata
from PIL import Image, ImageTk
from vision_engine import VisionEngine
from detector import ObjectDetector
from vision_utils import MODELS_DIR, ensure_dataset_structure, get_next_capture_filename

# ===================================================================
#  COMPONENTES DE INTERFAZ (UI)
# ===================================================================

class AnnotationWindow(ctk.CTkToplevel):
    """Ventana de anotación: permite dibujar bounding boxes sobre un frame capturado."""

    def __init__(self, parent, frame, class_name, base_name, dataset_dir, on_save, on_close=None):
        super().__init__(parent)
        self.title(f"Anotar: {base_name}")
        self.frame_bgr = frame
        self.class_name = class_name
        self.base_name = base_name
        self.dataset_dir = dataset_dir
        self.on_save_callback = on_save
        self.on_close_callback = on_close
        self.boxes = []
        self.drawing = False
        self.start_x = 0
        self.start_y = 0
        self.current_rect = None

        # Escalamiento para el monitor
        h, w = frame.shape[:2]
        max_w, max_h = 960, 680
        self.scale = min(max_w / w, max_h / h, 1.0)
        self.disp_w = int(w * self.scale)
        self.disp_h = int(h * self.scale)

        win_w, win_h = self.disp_w + 20, self.disp_h + 60
        x = (self.winfo_screenwidth() // 2) - (win_w // 2)
        y = (self.winfo_screenheight() // 2) - (win_h // 2)
        self.geometry(f"{win_w}x{win_h}+{x}+{y}")
        self.resizable(False, False)
        self.grab_set()
        self.focus()

        self.ann_canvas = tk.Canvas(self, width=self.disp_w, height=self.disp_h, bg="black", highlightthickness=0, cursor="crosshair")
        self.ann_canvas.pack(padx=10, pady=(10, 5))

        resized = cv2.resize(frame, (self.disp_w, self.disp_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        self.img_pil = Image.fromarray(rgb)
        self.img_tk = ImageTk.PhotoImage(self.img_pil)
        self.ann_canvas.create_image(0, 0, image=self.img_tk, anchor="nw")

        self.ann_canvas.bind("<ButtonPress-1>", self._on_press)
        self.ann_canvas.bind("<B1-Motion>", self._on_drag)
        self.ann_canvas.bind("<ButtonRelease-1>", self._on_release)
        self.ann_canvas.bind("<Button-3>", self._on_undo)

        bottom = ctk.CTkFrame(self, fg_color="transparent")
        bottom.pack(fill="x", padx=10, pady=(0, 10))

        self.info_label = ctk.CTkLabel(bottom, text="Clic izq. = dibujar bbox | Clic der. = deshacer", font=ctk.CTkFont(size=11))
        self.info_label.pack(side="left")

        ctk.CTkButton(bottom, text="💾 Guardar", width=100, command=self._save, fg_color="#16a34a", hover_color="#15803d").pack(side="right", padx=(5, 0))
        ctk.CTkButton(bottom, text="Cancelar", width=80, command=self._cancel, fg_color="#6b7280", hover_color="#4b5563").pack(side="right")

        # Gestionar cierre por aspa (X)
        self.protocol("WM_DELETE_WINDOW", self._cancel)

    def _cancel(self):
        """Callback al cerrar sin guardar."""
        if self.on_close_callback:
            self.on_close_callback()
        self.destroy()

    def _on_press(self, event):
        self.drawing = True
        self.start_x = event.x
        self.start_y = event.y
        self.current_rect = self.ann_canvas.create_rectangle(event.x, event.y, event.x, event.y, outline="#00ff88", width=2, dash=(4, 2))

    def _on_drag(self, event):
        if self.drawing and self.current_rect:
            self.ann_canvas.coords(self.current_rect, self.start_x, self.start_y, event.x, event.y)

    def _on_release(self, event):
        if not self.drawing: return
        self.drawing = False
        x1, y1, x2, y2 = min(self.start_x, event.x), min(self.start_y, event.y), max(self.start_x, event.x), max(self.start_y, event.y)
        if (x2 - x1) < 8 or (y2 - y1) < 8:
            if self.current_rect: self.ann_canvas.delete(self.current_rect)
            return
        self.boxes.append((x1, y1, x2, y2))
        self.current_rect = None
        self._redraw()

    def _on_undo(self, event):
        if self.boxes:
            self.boxes.pop()
            self._redraw()

    def _redraw(self):
        self.ann_canvas.delete("all")
        self.ann_canvas.create_image(0, 0, image=self.img_tk, anchor="nw")
        for i, (x1, y1, x2, y2) in enumerate(self.boxes):
            self.ann_canvas.create_rectangle(x1, y1, x2, y2, outline="#00ff88", width=2)
            self.ann_canvas.create_text(x1 + 4, y1 + 4, text=f"{self.class_name} #{i + 1}", anchor="nw", fill="#00ff88", font=("Consolas", 10, "bold"))
        self.info_label.configure(text=f"{len(self.boxes)} bounding box(es) dibujada(s)")

    def _save(self):
        if not self.boxes: return
        img_path = os.path.join(self.dataset_dir, "images", "train", f"{self.base_name}.jpg")
        cv2.imwrite(img_path, self.frame_bgr)
        h_orig, w_orig = self.frame_bgr.shape[:2]
        lines = []
        for (x1, y1, x2, y2) in self.boxes:
            ox1, oy1, ox2, oy2 = x1/self.scale, y1/self.scale, x2/self.scale, y2/self.scale
            cx, cy = ((ox1 + ox2) / 2) / w_orig, ((oy1 + oy2) / 2) / h_orig
            bw, bh = (ox2 - ox1) / w_orig, (oy2 - oy1) / h_orig
            lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        label_path = os.path.join(self.dataset_dir, "labels", "train", f"{self.base_name}.txt")
        with open(label_path, "w") as f: f.write("\n".join(lines))
        self.on_save_callback(self.base_name, len(self.boxes))
        self.destroy()

class AddModelPopup(ctk.CTkToplevel):
    """Ventana modal para añadir nuevas arquitecturas y modelos, extrayendo clases automáticamente."""

    def __init__(self, parent, detector, on_success):
        super().__init__(parent)
        self.detector = detector
        self.on_success = on_success
        self.source_file = None
        self.extracted_classes = None 
        self.title("Añadir Nueva Arquitectura")
        win_w, win_h = 450, 600
        x = (self.winfo_screenwidth() // 2) - (win_w // 2)
        y = (self.winfo_screenheight() // 2) - (win_h // 2)
        self.geometry(f"{win_w}x{win_h}+{x}+{y}")
        self.resizable(False, False)
        self.grab_set()

        ctk.CTkLabel(self, text="GESTIÓN DE MODELOS", font=ctk.CTkFont(size=18, weight="bold")).pack(pady=(20, 15))
        self._label("1. Nombre de la Carpeta (Familia):")
        self.name_entry = ctk.CTkEntry(self, placeholder_text="Ej: YOLOv12, Trafico_Extremo")
        self.name_entry.pack(pady=(0, 15), padx=30, fill="x")

        self._label("2. Seleccionar archivo de pesos (.pt):")
        self.file_btn = ctk.CTkButton(self, text="📁 Buscar archivo .pt", command=self._select_file, fg_color="#334155")
        self.file_btn.pack(pady=(0, 5), padx=30, fill="x")
        self.file_label = ctk.CTkLabel(self, text="Ningún archivo seleccionado", font=ctk.CTkFont(size=10), text_color="#888")
        self.file_label.pack(pady=(0, 15))

        self._label("3. Clases incrustadas en el modelo:")
        self.classes_label = ctk.CTkLabel(self, text="Selecciona un .pt para previsualizar sus clases", font=ctk.CTkFont(size=11), text_color="#aaa")
        self.classes_label.pack(pady=(0, 5), padx=30, anchor="w")
        
        self.classes_text = ctk.CTkTextbox(self, height=120)
        self.classes_text.pack(pady=(0, 20), padx=30, fill="x")
        self.classes_text.configure(state="disabled", fg_color="#1a1c1e")

        self.save_btn = ctk.CTkButton(self, text="🚀 Instalar Arquitectura", command=self._save_model, height=40, font=ctk.CTkFont(weight="bold"), fg_color="#16a34a", hover_color="#15803d")
        self.save_btn.pack(pady=20, padx=30, fill="x")
        self.save_btn.configure(state="disabled")

    def _label(self, text):
        ctk.CTkLabel(self, text=text, font=ctk.CTkFont(size=12, weight="bold"), text_color="#38bdf8").pack(pady=(5, 5), padx=30, anchor="w")

    def _select_file(self):
        file = filedialog.askopenfilename(title="Seleccionar modelo PyTorch", filetypes=[("Modelos PyTorch", "*.pt")])
        if file:
            self.source_file = file
            self.file_label.configure(text=f"Analizando: {os.path.basename(file)}...", text_color="#facc15")
            
            self.classes_text.configure(state="normal")
            self.classes_text.delete("1.0", "end")
            self.classes_text.insert("end", "Iniciando motor Ultralytics para extraer clases incrustadas...\nPor favor, espera.")
            self.classes_text.configure(state="disabled")
            self.save_btn.configure(state="disabled")
            
            def extraer():
                try:
                    from ultralytics import YOLO
                    m = YOLO(file)
                    classes = list(m.names.values())
                    self.after(0, lambda: self._on_extracted(file, classes))
                except Exception as e:
                    self.after(0, lambda: self._on_extract_error(file, str(e)))
            
            threading.Thread(target=extraer, daemon=True).start()

    def _on_extracted(self, file, classes):
        self.extracted_classes = classes
        self.file_label.configure(text=os.path.basename(file), text_color="#4ade80")
        
        self.classes_text.configure(state="normal")
        self.classes_text.delete("1.0", "end")
        
        is_coco = len(classes) == 80 and "person" in classes and "car" in classes
        texto = f"✅ {len(classes)} clases encontradas {'(COCO estándar)' if is_coco else '(Personalizadas)'}:\n\n"
        texto += "\n".join([f"{i}. {c}" for i, c in enumerate(classes)])
        
        self.classes_text.insert("end", texto)
        self.classes_text.configure(state="disabled")
        self.classes_label.configure(text="Extracción exitosa. El sistema manejará esto por ti.")
        self.save_btn.configure(state="normal")

    def _on_extract_error(self, file, error):
        self.file_label.configure(text="Error leyendo archivo pt", text_color="#ef4444")
        self.classes_text.configure(state="normal")
        self.classes_text.delete("1.0", "end")
        self.classes_text.insert("end", f"Fallo al extraer clases:\n{error}\n\n(Puedes forzar la instalación asumiendo clases de COCO).")
        self.classes_text.configure(state="disabled")
        self.save_btn.configure(state="normal")
        self.extracted_classes = None

    def _save_model(self):
        name = self.name_entry.get().strip()
        if not name or not self.source_file:
            messagebox.showwarning("Aviso", "Asegúrate de poner un nombre y seleccionar el archivo.")
            return

        target_dir = os.path.join(MODELS_DIR, name)
        os.makedirs(target_dir, exist_ok=True)
        shutil.copy2(self.source_file, os.path.join(target_dir, os.path.basename(self.source_file)))
        
        # Generar metadata sin molestar al usuario
        if self.extracted_classes:
            is_coco = len(self.extracted_classes) == 80 and "person" in self.extracted_classes and "car" in self.extracted_classes
            meta = {
                "is_coco": is_coco,
                "classes": None if is_coco else self.extracted_classes
            }
        else:
            meta = {"is_coco": True, "classes": None}
            
        with open(os.path.join(target_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=4, ensure_ascii=False)

        self.detector.scan_models()
        self.on_success()
        self.destroy()



class ClassFilterWindow(ctk.CTkToplevel):
    """Ventana modal para seleccionar las clases a detectar."""

    def __init__(self, parent, detector, current_targets, on_apply):
        super().__init__(parent)
        self.title("Filtro de Clases")
        win_w, win_h = 420, 550
        x = (self.winfo_screenwidth() // 2) - (win_w // 2)
        y = (self.winfo_screenheight() // 2) - (win_h // 2)
        self.geometry(f"{win_w}x{win_h}+{x}+{y}")
        self.minsize(win_w, win_h)
        self.grab_set()

        self.detector = detector
        self.on_apply = on_apply
        
        # Obtener nombres de la clase de manera segura y directa
        try:
            self.all_classes = self.detector.get_class_names() if self.detector else {}
        except Exception:
            self.all_classes = {}
            
        self.checkbox_widgets = []
        self.checkboxes = {}
        self.selected_ids = set()

        # Normalizar targets actuales a int para comparación segura
        if current_targets is not None:
            for tid in current_targets:
                try: self.selected_ids.add(int(tid))
                except: pass
        else:
            self.selected_ids = set(self.all_classes.keys())

        # Header
        ctk.CTkLabel(self, text="SELECCIÓN DE CLASES", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=(15, 5))
        ctk.CTkLabel(self, text="Desmarca las clases que NO quieres detectar", text_color="#aaa", font=("", 11)).pack(pady=(0, 10))

        # Botones Rápidos
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(fill="x", padx=20, pady=5)
        ctk.CTkButton(btn_frame, text="Marcar Todas", command=self._check_all, width=120).pack(side="left", padx=5)
        ctk.CTkButton(btn_frame, text="Desmarcar Todas", command=self._uncheck_all, width=120, fg_color="#4b5563").pack(side="right", padx=5)

        # Contenedor Scrolleable
        self.scroll = ctk.CTkScrollableFrame(self)
        self.scroll.pack(fill="both", expand=True, padx=20, pady=10)

        # Buscar barra de búsqueda (opcional, mejora UX)
        self.search_var = ctk.StringVar()
        self.search_var.trace("w", self._on_search)
        self.search_entry = ctk.CTkEntry(self, placeholder_text="🔍 Buscar clase...", textvariable=self.search_var)
        self.search_entry.pack(fill="x", padx=20, pady=(0, 10))

        # Checkboxes
        self._build_list()

        # Botón Guardar
        ctk.CTkButton(self, text="✅ Aplicar Filtro", command=self._apply, fg_color="#16a34a", hover_color="#15803d", font=("", 14, "bold"), height=40).pack(fill="x", padx=20, pady=(0, 15))

    def _build_list(self, filter_text=""):
        # Limpiar widgets previos de forma segura (sin borrar internos de CTkScrollableFrame)
        for widget in self.checkbox_widgets:
            widget.destroy()
        self.checkbox_widgets.clear()
        self.checkboxes.clear()

        # Llenar lista
        if not self.all_classes:
            lbl = ctk.CTkLabel(self.scroll, text="No se encontraron clases.", text_color="#666")
            lbl.pack(pady=20)
            self.checkbox_widgets.append(lbl)
            return

        for class_id, class_name in sorted(self.all_classes.items(), key=lambda x: x[1]):
            if filter_text.lower() not in class_name.lower():
                continue
            
            var = ctk.BooleanVar(value=class_id in self.selected_ids)
            self.checkboxes[class_id] = var
            
            cb = ctk.CTkCheckBox(self.scroll, text=f"{class_name.capitalize()} (ID: {class_id})", variable=var)
            cb.pack(fill="x", pady=4, padx=5)
            self.checkbox_widgets.append(cb)
        
        # Forzar actualización de layout para que el scroll se recalcule
        self.update_idletasks()

    def _on_search(self, *args):
        self._build_list(self.search_var.get())

    def _check_all(self):
        for var in self.checkboxes.values():
            var.set(True)

    def _uncheck_all(self):
        for var in self.checkboxes.values():
            var.set(False)

    def _apply(self):
        # Actualizar set con estado actual de checkboxes (solo visibles, pero manejamos todos)
        for cls_id, var in self.checkboxes.items():
            if var.get():
                self.selected_ids.add(cls_id)
            else:
                self.selected_ids.discard(cls_id)

        # Si están todas seleccionadas, devolvemos None (es más eficiente)
        if len(self.selected_ids) == len(self.all_classes):
            final_targets = None
        else:
            final_targets = list(self.selected_ids)

        self.on_apply(final_targets)
        self.destroy()

class InfoWindow(ctk.CTkToplevel):
    """Ventana interactiva de créditos, librerías y dependencias."""

    def __init__(self, parent):
        super().__init__(parent)
        self.title("Información del Proyecto")
        win_w, win_h = 480, 600
        x = (self.winfo_screenwidth() // 2) - (win_w // 2)
        y = (self.winfo_screenheight() // 2) - (win_h // 2)
        self.geometry(f"{win_w}x{win_h}+{x}+{y}")
        self.resizable(False, False)
        self.grab_set()

        ctk.CTkLabel(self, text="ℹ️ ACERCA DE VISIÓN AI", font=ctk.CTkFont(size=18, weight="bold")).pack(pady=(20, 5))
        py_ver = platform.python_version()
        ctk.CTkLabel(self, text=f"Desarrollado en Python {py_ver}", text_color="#aaa").pack(pady=(0, 20))

        # Librerías Directas
        self._section("Librerías Directas y Apoyo")

        def lnk(text, url, pkg):
            try:
                v = importlib.metadata.version(pkg)
            except Exception:
                v = "N/A"
            btn_text = f"{text} (v{v})"
            btn = ctk.CTkButton(self, text=btn_text, fg_color="transparent", text_color="#38bdf8", hover_color="#1e293b", anchor="w", command=lambda url=url: webbrowser.open(url))
            btn.pack(fill="x", padx=40, pady=2)

        lnk("🖼️ customtkinter", "https://customtkinter.tomschimansky.com/", "customtkinter")
        lnk("👁️ opencv-python", "https://opencv.org/", "opencv-python")
        lnk("🖌️ Pillow", "https://python-pillow.org/", "Pillow")
        lnk("🧠 ultralytics", "https://ultralytics.com/", "ultralytics")
        lnk("🎥 vidgear", "https://abhitronix.github.io/vidgear/", "vidgear")

        # Librerías Transitivas
        self._section("Librerías Transitivas (Sub-dependencias)")
        
        self.dep_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.dep_frame.pack(fill="x", padx=30, pady=5)
        
        self.show_dep_btn = ctk.CTkButton(self.dep_frame, text="Ver Sub-librerías", command=self._toggle_deps, fg_color="#334155")
        self.show_dep_btn.pack(pady=10)

        self.dep_textbox = ctk.CTkTextbox(self.dep_frame, height=120, fg_color="#1a1c1e")
        def get_v(pkg):
            try: return importlib.metadata.version(pkg)
            except Exception: return "N/A"

        deps = [
            f"• torch & torchvision (v{get_v('torch')})",
            f"• numpy (v{get_v('numpy')})",
            f"• yt-dlp (v{get_v('yt-dlp')})",
            f"• scipy (v{get_v('scipy')})",
            f"• requests (v{get_v('requests')})"
        ]
        self.dep_textbox.insert("0.0", "\n".join(deps))
        self.dep_textbox.configure(state="disabled")

        ctk.CTkButton(self, text="Cerrar", command=self.destroy, fg_color="#6b7280", hover_color="#4b5563").pack(pady=(20, 20))

    def _section(self, title):
        ctk.CTkLabel(self, text=title, font=ctk.CTkFont(weight="bold", size=13)).pack(padx=30, pady=(15, 5), anchor="w")

    def _toggle_deps(self):
        if self.dep_textbox.winfo_ismapped():
            self.dep_textbox.pack_forget()
            self.show_dep_btn.configure(text="Ver Sub-librerías")
        else:
            self.dep_textbox.pack(fill="x", pady=5)
            self.show_dep_btn.configure(text="Ocultar Sub-librerías")

class CaptureNamePopup(ctk.CTkToplevel):
    """Ventana modal simple para pedir el nombre de la clase antes de capturar."""
    def __init__(self, parent, on_accept, on_cancel=None):
        super().__init__(parent)
        self.on_accept = on_accept
        self.on_cancel = on_cancel
        self.title("Nueva Captura de Dataset")
        win_w, win_h = 350, 200
        x = (self.winfo_screenwidth() // 2) - (win_w // 2)
        y = (self.winfo_screenheight() // 2) - (win_h // 2)
        self.geometry(f"{win_w}x{win_h}+{x}+{y}")
        self.resizable(False, False)
        self.grab_set()

        ctk.CTkLabel(self, text="NOMBRE DEL OBJETO / ETIQUETA", font=ctk.CTkFont(size=12, weight="bold")).pack(pady=(20, 10))
        self.entry = ctk.CTkEntry(self, placeholder_text="ej: taxi, persona, bache", width=250)
        self.entry.pack(pady=10)
        self.entry.bind("<Return>", lambda _: self._accept())
        self.entry.focus()

        btn_row = ctk.CTkFrame(self, fg_color="transparent")
        btn_row.pack(pady=10)
        ctk.CTkButton(btn_row, text="Continuar", width=100, command=self._accept, fg_color="#16a34a", hover_color="#15803d").pack(side="left", padx=5)
        ctk.CTkButton(btn_row, text="Cancelar", width=80, command=self._cancel, fg_color="#6b7280", hover_color="#4b5563").pack(side="left", padx=5)
        
        # Gestionar cierre por aspa (X)
        self.protocol("WM_DELETE_WINDOW", self._cancel)

    def _cancel(self):
        if self.on_cancel:
            self.on_cancel()
        self.destroy()

    def _accept(self):
        name = self.entry.get().strip().lower().replace(" ", "_")
        if name:
            self.on_accept(name)
            self.destroy()

class ModelExplorerWindow(ctk.CTkToplevel):
    """Ventana avanzada para gestionar las familias y archivos de modelos."""
    def __init__(self, parent, detector):
        super().__init__(parent)
        self.detector = detector
        self.title("Explorador de Inteligencia")
        win_w, win_h = 500, 600
        x = (self.winfo_screenwidth() // 2) - (win_w // 2)
        y = (self.winfo_screenheight() // 2) - (win_h // 2)
        self.geometry(f"{win_w}x{win_h}+{x}+{y}")
        self.grab_set()

        ctk.CTkLabel(self, text="GESTIÓN DE FAMILIAS Y MODELOS", font=ctk.CTkFont(size=18, weight="bold"), text_color="#38bdf8").pack(pady=(20, 10))
        
        # Botón superior: Añadir Nueva
        ctk.CTkButton(self, text="➕ Instalar Nueva Arquitectura (.pt)", command=self._add_new, fg_color="#16a34a", hover_color="#15803d", height=35).pack(pady=10, padx=20, fill="x")

        self.scroll = ctk.CTkScrollableFrame(self, fg_color="transparent")
        self.scroll.pack(fill="both", expand=True, padx=20, pady=10)

        # SECCIÓN DE ACELERACIÓN (NUEVA)
        if "openvino" in self.detector.hardware_diag["best_backend"]:
            accel_frame = ctk.CTkFrame(self, fg_color="#1a1c1e", border_width=1, border_color="#38bdf8")
            accel_frame.pack(fill="x", padx=20, pady=(0, 10))
            
            ctk.CTkLabel(accel_frame, text="🚀 ACELERACIÓN INTEL DETECTADA", 
                         font=ctk.CTkFont(size=11, weight="bold"), text_color="#38bdf8").pack(pady=(10, 5))
            
            self.btn_opt = ctk.CTkButton(accel_frame, text="Acelerar Modelo Actual (OpenVINO)", 
                                         command=self._optimize_active,
                                         fg_color="#0369a1", hover_color="#075985")
            self.btn_opt.pack(pady=(0, 10), padx=20, fill="x")

        # Footer informativo
        footer = ctk.CTkFrame(self, fg_color="transparent")
        footer.pack(fill="x", side="bottom", pady=15, padx=20)
        ctk.CTkLabel(footer, text="💡 Recomendación: Mantén una estructura de nombres clara para tus pesos\n(ej: 'yolo11n.pt' para Nano vs 'yolo11l.pt' para Large) para diferenciarlos.",
                     font=("", 10), text_color="#64748b", justify="left").pack(anchor="w")

        self._refresh()

    def _refresh(self):
        for w in self.scroll.winfo_children(): w.destroy()
        
        families = [d for d in os.listdir(MODELS_DIR) if os.path.isdir(os.path.join(MODELS_DIR, d))]
        
        if not families:
            ctk.CTkLabel(self.scroll, text="No hay modelos instalados.", text_color="#666").pack(pady=40)
            return

        for fam in sorted(families):
            f_path = os.path.join(MODELS_DIR, fam)
            models = [f for f in os.listdir(f_path) if f.endswith(".pt")]
            
            card = ctk.CTkFrame(self.scroll, fg_color="#1e293b", border_width=1, border_color="#334155")
            card.pack(fill="x", pady=5, padx=2)
            
            lbl_title = ctk.CTkLabel(card, text=fam.upper(), font=ctk.CTkFont(weight="bold"), text_color="#38bdf8")
            lbl_title.pack(side="left", padx=15, pady=12)
            
            lbl_count = ctk.CTkLabel(card, text=f"{len(models)} pesos (.pt)", font=("", 10), text_color="#94a3b8")
            lbl_count.pack(side="left", padx=5)

            ctk.CTkButton(card, text="🗑️", width=30, height=30, fg_color="#7f1d1d", hover_color="#991b1b", 
                          command=lambda f=fam: self._delete_family(f)).pack(side="right", padx=10)

            ctk.CTkButton(card, text="➕ Pesos", width=80, height=30, fg_color="#334155", hover_color="#475569", 
                          command=lambda f=fam: self._add_weights(f)).pack(side="right", padx=5)

    def _add_weights(self, fam_name):
        """Permite seleccionar un archivo .pt y copiarlo a la carpeta de la familia."""
        file = filedialog.askopenfilename(title=f"Añadir pesos a {fam_name}", filetypes=[("Pesos PyTorch", "*.pt")])
        if file:
            try:
                dest = os.path.join(MODELS_DIR, fam_name, os.path.basename(file))
                if os.path.exists(dest):
                    if not messagebox.askyesno("Confirmar", f"El archivo '{os.path.basename(file)}' ya existe. ¿Sobrescribir?"):
                        return
                
                import shutil
                shutil.copy2(file, dest)
                self.detector.scan_models()
                self._refresh()
                messagebox.showinfo("Éxito", f"Pesos añadidos correctamente a la familia {fam_name}.")
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo copiar el archivo: {e}")

    def _add_new(self):
        # Abrir el popup existente de añadir modelo
        AddModelPopup(self, self.detector, self._refresh)

    def _delete_family(self, name):
        if messagebox.askyesno("Confirmar Borrado", f"¿Estás seguro de eliminar la familia '{name}'?\nEsto borrará todos los modelos internos definitivamente."):
            try:
                shutil.rmtree(os.path.join(MODELS_DIR, name))
                self.detector.scan_models()
                self._refresh()
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo borrar: {e}")

    def _optimize_active(self):
        """Dispara el proceso de optimización del modelo cargado en el detector."""
        if not self.detector.model:
            messagebox.showwarning("Aviso", "No hay ningún modelo cargado actualmente para optimizar.")
            return
            
        if self.detector.is_openvino_active:
            messagebox.showinfo("Aviso", "El modelo actual ya está optimizado y funcionando con OpenVINO.")
            return

        self.btn_opt.configure(state="disabled", text="⏳ Optimizando... (Esto puede tardar)")
        self.update()

        def run_opt():
            success = self.detector.export_current_to_openvino()
            def done():
                self.btn_opt.configure(state="normal", text="Acelerar Modelo Actual (OpenVINO)")
                if success:
                    messagebox.showinfo("Éxito", f"Modelo '{self.detector.active_name}' optimizado para Intel.\nLa próxima vez que lo selecciones, usará la aceleración automática.")
                    self._refresh()
                else:
                    messagebox.showerror("Error", "No se pudo realizar la optimización. Verifica los logs.")
            self.after(0, done)

        threading.Thread(target=run_opt, daemon=True).start()
