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
from ..core.engine import VisionEngine
from ..core.detector import ObjectDetector
from ..utils.helpers import (
    MODELS_DIR, ensure_dataset_structure, get_next_capture_filename, log_error,
    load_favorites, save_favorites
)

# ===================================================================
#  COMPONENTES DE INTERFAZ (UI)
# ===================================================================

class AnnotationWindow(ctk.CTkToplevel):
    """Ventana de anotación: permite dibujar bounding boxes sobre un frame capturado."""

    def __init__(self, parent, frame=None, ds_name="", base_name="", dataset_dir="", on_save=None, on_close=None, image_files=None):
        super().__init__(parent)

        self.ds_name = ds_name
        self.dataset_dir = dataset_dir
        self.on_save_callback = on_save
        self.on_close_callback = on_close
        
        # Estado de múltiples imágenes
        self.image_files = image_files or [] # Lista de paths completos
        self.current_index = 0
        self.frame_bgr = frame
        self.base_name = base_name

        if self.image_files:
            self._load_image_data(0)
        
        self.title(f"Anotar: {self.base_name}")
        
        self.boxes = [] # Lista de {bbox: (x1,y1,x2,y2), label: "clase"}
        self.classes_found = self._load_existing_classes()
        self.active_class_name = self.classes_found[0] if self.classes_found else None
        
        self.drawing = False
        self.start_x = 0
        self.start_y = 0
        self.current_rect = None

        # Escalamiento (se recalcula en _load_image_data si es necesario, pero aquí para layout inicial)
        if self.frame_bgr is not None:
            self._calculate_scale(self.frame_bgr)
        else:
            self.disp_w, self.disp_h = 800, 600 # Fallback
            self.scale = 1.0

        win_w, win_h = self.disp_w + 220, self.disp_h + 120
        self.geometry(f"{win_w}x{win_h}")
        self.grab_set()

        # Layout principal
        self.main_container = ctk.CTkFrame(self, fg_color="transparent")
        self.main_container.pack(fill="both", expand=True, padx=10, pady=10)

        # Panel Izquierdo: Clases
        self.sidebar = ctk.CTkFrame(self.main_container, width=180, fg_color="#1e293b", border_width=1, border_color="#334155")
        self.sidebar.pack(side="left", fill="y", padx=(0, 10))
        
        ctk.CTkLabel(self.sidebar, text="NUEVA CLASE", font=ctk.CTkFont(size=10, weight="bold"), text_color="#94a3b8").pack(pady=(10, 0))
        self.new_class_entry = ctk.CTkEntry(self.sidebar, height=28, font=ctk.CTkFont(size=11), placeholder_text="Nombre...")
        self.new_class_entry.pack(fill="x", padx=10, pady=5)
        self.new_class_entry.bind("<Return>", lambda e: self._add_new_class_from_sidebar())
        
        ctk.CTkButton(self.sidebar, text="Añadir", height=24, font=ctk.CTkFont(size=10), 
                      fg_color="#334155", command=self._add_new_class_from_sidebar).pack(pady=(0, 10), padx=10)

        self.class_btn_frame = ctk.CTkScrollableFrame(self.sidebar, fg_color="transparent")
        self.class_btn_frame.pack(fill="both", expand=True, padx=5, pady=5)

        # Resumen del Dataset (Carga dinámica)
        ctk.CTkLabel(self.sidebar, text="CONTEO TOTAL DATASET", font=ctk.CTkFont(size=10, weight="bold"), text_color="#38bdf8").pack(pady=(10, 0))
        self.summary_label = ctk.CTkLabel(self.sidebar, text="Calculando...", font=ctk.CTkFont(size=10), text_color="#94a3b8", justify="left")
        self.summary_label.pack(fill="x", padx=10, pady=(5, 10))

        # Panel Derecho: Canvas
        self.canvas_container = ctk.CTkFrame(self.main_container, fg_color="black")
        self.canvas_container.pack(side="left", fill="both", expand=True)

        self.ann_canvas = tk.Canvas(self.canvas_container, width=self.disp_w, height=self.disp_h, bg="black", highlightthickness=0, cursor="crosshair")
        self.ann_canvas.pack(fill="both", expand=True)

        if self.frame_bgr is not None:
            resized = cv2.resize(self.frame_bgr, (self.disp_w, self.disp_h))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            self.img_tk = ImageTk.PhotoImage(Image.fromarray(rgb))
            self.ann_canvas.create_image(0, 0, image=self.img_tk, anchor="nw")

        self.ann_canvas.bind("<ButtonPress-1>", self._on_press)
        self.ann_canvas.bind("<B1-Motion>", self._on_drag)
        self.ann_canvas.bind("<ButtonRelease-1>", self._on_release)
        self.ann_canvas.bind("<Button-3>", self._on_undo)

        # Bottom Bar
        bottom = ctk.CTkFrame(self, fg_color="transparent")
        bottom.pack(fill="x", padx=10, pady=(0, 10))

        self.info_label = ctk.CTkLabel(bottom, text="Cargando...", font=ctk.CTkFont(size=11))
        self.info_label.pack(side="left")

        # Botones de navegación (Solo si hay múltiples imágenes)
        if self.image_files:
            nav_frame = ctk.CTkFrame(bottom, fg_color="transparent")
            nav_frame.pack(side="right", padx=10)
            
            ctk.CTkButton(nav_frame, text="⬅️", width=40, command=self._prev_image, fg_color="#334155").pack(side="left", padx=2)
            self.index_label = ctk.CTkLabel(nav_frame, text=f"1 / {len(self.image_files)}", font=ctk.CTkFont(size=11, weight="bold"))
            self.index_label.pack(side="left", padx=5)
            ctk.CTkButton(nav_frame, text="➡️", width=40, command=self._next_image, fg_color="#334155").pack(side="left", padx=2)

        ctk.CTkButton(bottom, text="💾 Guardar", width=100, command=self._save_and_stay, fg_color="#16a34a", hover_color="#15803d").pack(side="right", padx=(5, 0))
        if not self.image_files:
            ctk.CTkButton(bottom, text="Cancelar", width=80, command=self._cancel, fg_color="#6b7280", hover_color="#4b5563").pack(side="right")
        else:
            ctk.CTkButton(bottom, text="Finalizar", width=80, command=self._finish, fg_color="#6b7280", hover_color="#4b5563").pack(side="right")

        if self.image_files:
            self._load_current_image()
        else:
            self._refresh_class_list()
        
        self._update_dataset_summary()
        
        # Gestión de foco: Si ya hay clases, poner el foco en la ventana para atajos
        if self.classes_found:
            self.focus_set()
        else:
            self.after(500, lambda: self.new_class_entry.focus_set())

        # Bindings de teclado para atajos rápidos
        self.bind("<KeyPress>", self._on_key_press)
        self.new_class_entry.bind("<Escape>", lambda e: self.focus_set())

    def _refresh_class_list(self):
        """Actualiza la barra lateral de clases."""
        for w in self.class_btn_frame.winfo_children():
            w.destroy()
        
        self.class_buttons = {}
        for i, cls in enumerate(self.classes_found):
            is_active = (cls == self.active_class_name)
            shortcut = (i + 1) if i < 9 else (0 if i == 9 else None)
            display_text = f"[{shortcut}] {cls}" if shortcut is not None else cls
            
            btn = ctk.CTkButton(self.class_btn_frame, text=display_text, height=28,
                                fg_color="#38bdf8" if is_active else "#334155",
                                text_color="#0f172a" if is_active else "#e2e8f0",
                                hover_color="#0ea5e9",
                                command=lambda c=cls: self._set_active_class(c))
            btn.pack(fill="x", pady=2)
            self.class_buttons[cls] = btn
        
        self._redraw()

    def _set_active_class(self, cls_name):
        self.active_class_name = cls_name
        self._refresh_class_list()

    def _on_key_press(self, event):
        """Maneja atajos de teclado para selección de clase y navegación."""
        if self.focus_get() == self.new_class_entry:
            return
            
        if event.char.isdigit():
            val = int(event.char)
            idx = (val - 1) if val > 0 else 9
            if idx < len(self.classes_found):
                self._set_active_class(self.classes_found[idx])
        elif event.keysym == "Right" or event.char == "d":
            if self.image_files: self._next_image()
        elif event.keysym == "Left" or event.char == "a":
            if self.image_files: self._prev_image()

    def _add_new_class_from_sidebar(self):
        val = self.new_class_entry.get().strip()
        if not val: return
        
        new_classes = [v.strip().lower() for v in val.split(",") if v.strip()]
        for nc in new_classes:
            if nc not in self.classes_found:
                self.classes_found.append(nc)
        
        if not self.active_class_name and self.classes_found:
            self.active_class_name = self.classes_found[0]
            
        self.new_class_entry.delete(0, "end")
        self._refresh_class_list()
        self.focus_set()
        if hasattr(self.master, "add_log"):
            self.master.add_log(f"🏷️ Clases actualizadas: {', '.join(new_classes)}")

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
        
        # Usar la clase activa automáticamente
        if not self.active_class_name:
            if self.current_rect: self.ann_canvas.delete(self.current_rect)
            self._add_new_class_dialog()
            return

        self.boxes.append({"bbox": (x1, y1, x2, y2), "label": self.active_class_name})
        self.current_rect = None
        self._redraw()

    def _load_existing_classes(self):
        """Carga las clases ya registradas en el dataset para mantener consistencia."""
        classes = []
        try:
            txt_path = os.path.join(self.dataset_dir, "classes.txt")
            if os.path.exists(txt_path):
                with open(txt_path, "r", encoding="utf-8") as f:
                    classes = [l.strip() for l in f.readlines() if l.strip()]
        except: pass
        return classes

    def _on_undo(self, event):
        """Elimina una caja específica si se hace clic derecho sobre ella, o la última si no."""
        if not self.boxes: return
        found = False
        for i in range(len(self.boxes) - 1, -1, -1):
            x1, y1, x2, y2 = self.boxes[i]["bbox"]
            if (x1 - 2) <= event.x <= (x2 + 2) and (y1 - 2) <= event.y <= (y2 + 2):
                self.boxes.pop(i)
                found = True
                break
        if not found:
            self.boxes.pop()
        self._redraw()

    def _calculate_scale(self, frame):
        if frame is None:
            self.disp_w, self.disp_h = 800, 600
            self.scale = 1.0
            return
        h, w = frame.shape[:2]
        max_w, max_h = 1000, 700
        self.scale = min(max_w / w, max_h / h, 1.0)
        self.disp_w = int(w * self.scale)
        self.disp_h = int(h * self.scale)

    def _load_image_data(self, index):
        """Carga el frame y metadatos de una imagen de la lista."""
        path = self.image_files[index]
        self.frame_bgr = cv2.imread(path)
        if self.frame_bgr is None:
            return False
        self.base_name = os.path.splitext(os.path.basename(path))[0]
        self._calculate_scale(self.frame_bgr)
        self.title(f"Anotar: {self.base_name} ({index+1}/{len(self.image_files)})")
        return True

    def _load_current_image(self):
        """Actualiza el canvas con la imagen actual y carga sus etiquetas si existen."""
        if self.frame_bgr is None: return
        
        resized = cv2.resize(self.frame_bgr, (self.disp_w, self.disp_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        self.img_tk = ImageTk.PhotoImage(Image.fromarray(rgb))
        
        # Limpiar y redibujar
        self.ann_canvas.delete("all")
        self.ann_canvas.create_image(0, 0, image=self.img_tk, anchor="nw")
        
        # Intentar cargar etiquetas existentes en formato YOLO
        self.boxes = self._load_yolo_labels()
        self._refresh_class_list()
        if hasattr(self, 'index_label'):
            self.index_label.configure(text=f"{self.current_index + 1} / {len(self.image_files)}")

    def _load_yolo_labels(self):
        """Carga etiquetas de un archivo .txt si existe."""
        boxes = []
        label_path = os.path.join(self.dataset_dir, "labels", "train", f"{self.base_name}.txt")
        if os.path.exists(label_path):
            try:
                h_orig, w_orig = self.frame_bgr.shape[:2]
                with open(label_path, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            cls_id, cx, cy, bw, bh = map(float, parts)
                            # Convertir de YOLO a coordenadas de canvas
                            class_name = self.classes_found[int(cls_id)] if int(cls_id) < len(self.classes_found) else f"class_{cls_id}"
                            
                            ox1 = (cx - bw/2) * w_orig * self.scale
                            oy1 = (cy - bh/2) * h_orig * self.scale
                            ox2 = (cx + bw/2) * w_orig * self.scale
                            oy2 = (cy + bh/2) * h_orig * self.scale
                            
                            boxes.append({"bbox": (ox1, oy1, ox2, oy2), "label": class_name})
            except Exception as e:
                print(f"Error cargando etiquetas YOLO: {e}")
        return boxes

    def _next_image(self):
        self._save_internal() # Guardar actual antes de pasar
        if self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self._load_image_data(self.current_index)
            self._load_current_image()

    def _prev_image(self):
        self._save_internal()
        if self.current_index > 0:
            self.current_index -= 1
            self._load_image_data(self.current_index)
            self._load_current_image()

    def _redraw(self):
        self.ann_canvas.delete("all")
        self.ann_canvas.create_image(0, 0, image=self.img_tk, anchor="nw")
        for i, box_data in enumerate(self.boxes):
            x1, y1, x2, y2 = box_data["bbox"]
            label = box_data["label"]
            self.ann_canvas.create_rectangle(x1, y1, x2, y2, outline="#00ff88", width=2)
            self.ann_canvas.create_text(x1 + 4, y1 + 4, text=f"{label}", anchor="nw", fill="#00ff88", font=("Consolas", 10, "bold"))
        self.info_label.configure(text=f"{len(self.boxes)} objeto(s) etiquetado(s) | Clases: {', '.join(set(b['label'] for b in self.boxes))}")

    def _save_and_stay(self):
        """Guarda la imagen actual y notifica."""
        self._save_internal()
        if not self.image_files:
            self.destroy()

    def _save(self):
        self._save_internal()
        self.destroy()

    def _finish(self):
        """Guarda la imagen actual y cierra abriendo la carpeta de resultados."""
        self._save_internal()
        try:
            target_path = os.path.abspath(self.dataset_dir)
            if os.path.exists(target_path):
                os.startfile(target_path)
        except Exception:
            pass
        self.destroy()

    def _save_internal(self):
        """Lógica interna de guardado para una imagen individual."""
        if self.frame_bgr is None: return
        
        try:
            # 1. Guardar Imagen (Si es una captura nueva, si ya existe en el dataset_dir se sobreescribe)
            img_path = os.path.join(self.dataset_dir, "images", "train", f"{self.base_name}.jpg")
            # Si la imagen ya está ahí (importada), no necesitamos re-escribirla a menos que queramos asegurar formato
            cv2.imwrite(img_path, self.frame_bgr)
            
            # 2. Actualizar classes.txt
            txt_path = os.path.join(self.dataset_dir, "classes.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("\n".join(self.classes_found))
            
            # 3. Actualizar data.yaml
            self._update_yaml()
    
            # 4. Generar Labels (YOLO format)
            h_orig, w_orig = self.frame_bgr.shape[:2]
            lines = []
            for box_data in self.boxes:
                x1, y1, x2, y2 = box_data["bbox"]
                label = box_data["label"]
                if label not in self.classes_found:
                    self.classes_found.append(label)
                class_id = self.classes_found.index(label)
                
                ox1, oy1, ox2, oy2 = x1/self.scale, y1/self.scale, x2/self.scale, y2/self.scale
                cx, cy = ((ox1 + ox2) / 2) / w_orig, ((oy1 + oy2) / 2) / h_orig
                bw, bh = (ox2 - ox1) / w_orig, (oy2 - oy1) / h_orig
                lines.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                
            label_path = os.path.join(self.dataset_dir, "labels", "train", f"{self.base_name}.txt")
            os.makedirs(os.path.dirname(label_path), exist_ok=True)
            with open(label_path, "w") as f: f.write("\n".join(lines))
            
            if self.on_save_callback:
                self.on_save_callback(self.base_name, len(self.boxes))
            self._update_dataset_summary()
                
        except Exception as e:
            print(f"Error guardando: {e}")
            messagebox.showerror("Error", f"Fallo al guardar {self.base_name}: {e}")

    def _update_dataset_summary(self):
        """Calcula el total de etiquetas por clase en todo el dataset."""
        from collections import Counter
        counts = Counter()
        label_dir = os.path.join(self.dataset_dir, "labels", "train")
        if os.path.exists(label_dir):
            for file in os.listdir(label_dir):
                if file.endswith(".txt"):
                    try:
                        with open(os.path.join(label_dir, file), "r") as f:
                            for line in f:
                                parts = line.strip().split()
                                if parts:
                                    try: counts[int(parts[0])] += 1
                                    except: pass
                    except: pass
        summary_lines = []
        for i, cls in enumerate(self.classes_found):
            summary_lines.append(f"\u2022 {cls.upper()}: {counts[i]}")
        txt = "\n".join(summary_lines) if summary_lines else "Sin etiquetas todav\u00eda"
        self.summary_label.configure(text=txt)

    def _update_yaml(self):
        """Actualiza el archivo data.yaml con la lista completa de clases."""
        yaml_path = os.path.join(self.dataset_dir, "data.yaml")
        abs_path = os.path.abspath(self.dataset_dir).replace("\\", "/")
        
        names_dict = {i: name for i, name in enumerate(self.classes_found)}
        
        content = [
            f"# Dataset: {self.ds_name}",
            " # Generado por AI VisionSandbox Lab - Modo Multi-clase",
            f"path: {abs_path}",
            "train: images/train",
            "val: images/val",
            "",
            "# Clases",
            f"nc: {len(self.classes_found)}",
            "names:"
        ]
        for i, name in names_dict.items():
            content.append(f"  {i}: {name}")
            
        with open(yaml_path, "w", encoding="utf-8") as f:
            f.write("\n".join(content))

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
    """Ventana interactiva de creditos, librerias y dependencias."""

    def __init__(self, parent):
        super().__init__(parent)
        self.title("Informacion del Proyecto")
        win_w, win_h = 500, 680
        x = (self.winfo_screenwidth() // 2) - (win_w // 2)
        y = (self.winfo_screenheight() // 2) - (win_h // 2)
        self.geometry(f"{win_w}x{win_h}+{x}+{y}")
        self.resizable(False, False)
        self.grab_set()

        ctk.CTkLabel(self, text="AI VISIONSANDBOX LAB", font=ctk.CTkFont(size=18, weight="bold"), text_color="#38bdf8").pack(pady=(20, 2))
        ctk.CTkLabel(self, text="v4.0 - Motor de Vision Artificial en Tiempo Real", font=ctk.CTkFont(size=11), text_color="#94a3b8").pack(pady=(0, 2))
        py_ver = platform.python_version()
        ctk.CTkLabel(self, text=f"Python {py_ver} | {platform.system()} {platform.machine()}", text_color="#666", font=ctk.CTkFont(size=10)).pack(pady=(0, 15))

        self._section("Librerias Principales")

        def lnk(text, url, pkg):
            try:
                v = importlib.metadata.version(pkg)
            except Exception:
                v = "N/A"
            btn_text = f"{text} (v{v})"
            btn = ctk.CTkButton(self, text=btn_text, fg_color="transparent", text_color="#38bdf8", hover_color="#1e293b", anchor="w", command=lambda u=url: webbrowser.open(u))
            btn.pack(fill="x", padx=40, pady=1)

        lnk("ultralytics", "https://ultralytics.com/", "ultralytics")
        lnk("customtkinter", "https://customtkinter.tomschimansky.com/", "customtkinter")
        lnk("opencv-python", "https://opencv.org/", "opencv-python")
        lnk("vidgear", "https://abhitronix.github.io/vidgear/", "vidgear")
        lnk("Pillow", "https://python-pillow.org/", "Pillow")
        lnk("pyttsx3", "https://pypi.org/project/pyttsx3/", "pyttsx3")
        lnk("python-dotenv", "https://pypi.org/project/python-dotenv/", "python-dotenv")

        self._section("Librerias de Soporte")
        
        self.dep_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.dep_frame.pack(fill="x", padx=30, pady=5)
        
        self.show_dep_btn = ctk.CTkButton(self.dep_frame, text="Ver Sub-librerias", command=self._toggle_deps, fg_color="#334155")
        self.show_dep_btn.pack(pady=5)

        self.dep_textbox = ctk.CTkTextbox(self.dep_frame, height=150, fg_color="#1a1c1e")
        def get_v(pkg):
            try: return importlib.metadata.version(pkg)
            except Exception: return "N/A"

        deps = [
            f"PyTorch (v{get_v('torch')})",
            f"NumPy (v{get_v('numpy')})",
            f"yt-dlp (v{get_v('yt-dlp')})",
            f"requests (v{get_v('requests')})",
            f"LapX - Tracking ByteTrack (v{get_v('lapx')})",
            f"OpenVINO - Aceleracion Intel (v{get_v('openvino')})",
            f"scipy (v{get_v('scipy')})",
            f"SQLite3 (incluido en Python)",
        ]
        self.dep_textbox.insert("0.0", "\n".join(deps))
        self.dep_textbox.configure(state="disabled")

        self._section("Funcionalidades Clave")
        features = (
            "Deteccion YOLO/RT-DETR + Zero-Shot (YOLO-World)\n"
            "Tracking de objetos con ByteTrack + Focus Mode\n"
            "Zonas poligonales con motor de hitos y alertas\n"
            "Telegram, Webhooks, TTS y evidencias automaticas\n"
            "Dashboard interactivo con graficas por zonas\n"
            "Anotador visual con atajos de teclado\n"
            "Autocaptura, importacion ZIP y exportacion de datasets\n"
            "Aceleracion GPU: CUDA / OpenVINO / DirectML"
        )
        ctk.CTkLabel(self, text=features, font=ctk.CTkFont(size=10), text_color="#94a3b8", justify="left").pack(padx=40, anchor="w")

        ctk.CTkButton(self, text="Cerrar", command=self.destroy, fg_color="#6b7280", hover_color="#4b5563").pack(pady=(15, 20))

    def _section(self, title):
        ctk.CTkLabel(self, text=title, font=ctk.CTkFont(weight="bold", size=13)).pack(padx=30, pady=(12, 3), anchor="w")

    def _toggle_deps(self):
        if self.dep_textbox.winfo_ismapped():
            self.dep_textbox.pack_forget()
            self.show_dep_btn.configure(text="Ver Sub-librerias")
        else:
            self.dep_textbox.pack(fill="x", pady=5)
            self.show_dep_btn.configure(text="Ocultar Sub-librerias")

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

class SourceSelectorWindow(ctk.CTkToplevel):
    """Ventana para seleccionar la fuente de vídeo: Directo, Rediferido o Local."""
    def __init__(self, parent, current_url, on_change):
        super().__init__(parent)
        self.parent = parent
        self.on_change = on_change
        self.title("Seleccionar Fuente de Vídeo")
        win_w, win_h = 500, 450
        x = (self.winfo_screenwidth() // 2) - (win_w // 2)
        y = (self.winfo_screenheight() // 2) - (win_h // 2)
        self.geometry(f"{win_w}x{win_h}+{x}+{y}")
        self.resizable(False, False)
        self.grab_set()

        ctk.CTkLabel(self, text="CONFIGURAR FUENTE DE ENTRADA", font=ctk.CTkFont(size=18, weight="bold"), text_color="#38bdf8").pack(pady=(25, 20))

        # --- OPCIÓN 1: STREAM EN DIRECTO ---
        self._section_title("🌐 STREAM EN DIRECTO (RTSP, RTMP, HLS)")
        live_frame = ctk.CTkFrame(self, fg_color="transparent")
        live_frame.pack(fill="x", padx=40, pady=(0, 15))
        
        self.live_entry = ctk.CTkEntry(live_frame, placeholder_text="rtsp://... o URL de directo", height=32)
        self.live_entry.pack(side="left", fill="x", expand=True, padx=(0, 8))
        
        ctk.CTkButton(live_frame, text="Conectar", width=90, height=32, fg_color="#1e293b", hover_color="#334155", text_color="#38bdf8",
                      command=lambda: self._apply(self.live_entry.get())).pack(side="right")

        # --- OPCIÓN 2: VÍDEO REDIFERIDO / VOD ---
        self._section_title("📺 VÍDEO REDIFERIDO (YouTube, Vimeo, MP4 URL)")
        vod_frame = ctk.CTkFrame(self, fg_color="transparent")
        vod_frame.pack(fill="x", padx=40, pady=(0, 15))
        
        self.vod_entry = ctk.CTkEntry(vod_frame, placeholder_text="https://www.youtube.com/watch?v=...", height=32)
        self.vod_entry.pack(side="left", fill="x", expand=True, padx=(0, 8))
        
        ctk.CTkButton(vod_frame, text="Cargar", width=90, height=32, fg_color="#1e293b", hover_color="#334155", text_color="#38bdf8",
                      command=lambda: self._apply(self.vod_entry.get())).pack(side="right")

        # --- OPCIÓN 3: ARCHIVO LOCAL ---
        self._section_title("📂 ARCHIVO LOCAL")
        ctk.CTkButton(self, text="Seleccionar Archivo de Vídeo", height=35, fg_color="#334155", hover_color="#475569",
                      command=self._browse_local).pack(fill="x", padx=40, pady=(0, 20))

        # Pre-rellenar si es URL
        if current_url.startswith(("http", "rtsp", "rtmp")):
            if "youtube" in current_url or "youtu.be" in current_url:
                self.vod_entry.insert(0, current_url)
            else:
                self.live_entry.insert(0, current_url)

        # Footer
        ctk.CTkButton(self, text="Cerrar", command=self.destroy, fg_color="#6b7280", hover_color="#4b5563").pack(pady=10)

    def _section_title(self, text):
        ctk.CTkLabel(self, text=text, font=ctk.CTkFont(size=12, weight="bold"), text_color="#94a3b8").pack(padx=40, pady=(5, 5), anchor="w")

    def _apply(self, url):
        if url.strip():
            # Preguntar si quiere guardar como favorito antes de aplicar
            if messagebox.askyesno("Favoritos", "¿Deseas guardar esta fuente en tus favoritos?"):
                def on_name(name):
                    favs = load_favorites()
                    favs.append({"name": name, "url": url.strip()})
                    save_favorites(favs)
                    if hasattr(self.parent, "add_log"):
                        self.parent.add_log(f"⭐ Fuente guardada en favoritos: {name}")
                    self.on_change(url.strip())
                    self.destroy()
                CaptureNamePopup(self, on_name, title="Guardar Favorito", prompt="Nombre para esta fuente:")
            else:
                self.on_change(url.strip())
                self.destroy()

    def _browse_local(self):
        path = filedialog.askopenfilename(filetypes=[("Vídeos", "*.mp4 *.avi *.mkv *.mov"), ("Todos", "*.*")])
        if path:
            self._apply(path) # Usar _apply para permitir guardarlo en favoritos

class FavoritesWindow(ctk.CTkToplevel):
    """Ventana para gestionar y seleccionar fuentes favoritas."""
    def __init__(self, parent, on_select):
        super().__init__(parent)
        self.on_select = on_select
        self.title("Mis Fuentes Favoritas")
        self.geometry("450x500")
        self.grab_set()
        
        ctk.CTkLabel(self, text="⭐ FUENTES FAVORITAS", font=ctk.CTkFont(size=16, weight="bold"), text_color="#38bdf8").pack(pady=20)
        
        self.scroll = ctk.CTkScrollableFrame(self, fg_color="transparent")
        self.scroll.pack(fill="both", expand=True, padx=20, pady=10)
        
        self._refresh()

    def _refresh(self):
        for w in self.scroll.winfo_children(): w.destroy()
        favs = load_favorites()
        
        if not favs:
            ctk.CTkLabel(self.scroll, text="No tienes fuentes guardadas.", text_color="#666").pack(pady=40)
            return

        for i, f in enumerate(favs):
            card = ctk.CTkFrame(self.scroll, fg_color="#1e293b", border_width=1, border_color="#334155")
            card.pack(fill="x", pady=6, padx=5)
            
            # Contenedor de texto (Izquierda)
            text_frame = ctk.CTkFrame(card, fg_color="transparent")
            text_frame.pack(side="left", fill="both", expand=True, padx=15, pady=10)
            
            name_lbl = ctk.CTkLabel(text_frame, text=f['name'].upper(), font=ctk.CTkFont(size=13, weight="bold"), 
                                    text_color="#38bdf8", anchor="w")
            name_lbl.pack(fill="x")
            
            url_display = f['url']
            if len(url_display) > 55:
                url_display = url_display[:52] + "..."
                
            url_lbl = ctk.CTkLabel(text_frame, text=url_display, font=("", 10), 
                                   text_color="#94a3b8", anchor="w")
            url_lbl.pack(fill="x")

            # Contenedor de botones (Derecha)
            btn_frame = ctk.CTkFrame(card, fg_color="transparent")
            btn_frame.pack(side="right", padx=10)

            ctk.CTkButton(btn_frame, text="Cargar", width=75, height=30, fg_color="#38bdf8", hover_color="#0ea5e9", text_color="#000",
                          font=ctk.CTkFont(size=12, weight="bold"),
                          command=lambda u=f['url']: self._apply(u)).pack(side="left", padx=5)

            ctk.CTkButton(btn_frame, text="\uE74D", width=35, height=30, fg_color="#450a0a", hover_color="#7f1d1d",
                          font=ctk.CTkFont(family="Segoe MDL2 Assets", size=14),
                          command=lambda idx=i: self._delete(idx)).pack(side="left")

    def _delete(self, idx):
        favs = load_favorites()
        if idx < len(favs):
            name = favs[idx].get("name", "Fuente")
            favs.pop(idx)
            save_favorites(favs)
            if hasattr(self.master, "add_log"):
                self.master.add_log(f"🗑️ Favorito eliminado: {name}")
            self._refresh()

    def _apply(self, url):
        if hasattr(self.master, "add_log"):
            self.master.add_log(f"📂 Cargando favorito: {url[:50]}...")
        self.on_select(url)
        self.destroy()

class CaptureNamePopup(ctk.CTkToplevel):
    """Ventana modal simple para pedir un nombre (usada para etiquetas multi-clase)."""
    def __init__(self, parent, on_accept, on_cancel=None, title="Etiquetar Objeto", prompt="ETIQUETA DEL OBJETO"):
        super().__init__(parent)
        self.on_accept = on_accept
        self.on_cancel = on_cancel
        self.title(title)
        self.geometry("380x200")
        self.grab_set()
        self.resizable(False, False)

        # Centrar
        x = parent.winfo_x() + (parent.winfo_width() // 2) - 190
        y = parent.winfo_y() + (parent.winfo_height() // 2) - 100
        self.geometry(f"+{x}+{y}")

        ctk.CTkLabel(self, text=prompt, font=ctk.CTkFont(size=12, weight="bold"), text_color="#38bdf8", justify="center").pack(pady=(20, 10))
        
        self.entry = ctk.CTkEntry(self, placeholder_text="ej: pez, burbuja, coral...", width=280)
        self.entry.pack(pady=10)
        self.entry.focus_set()
        self.entry.bind("<Return>", lambda e: self._accept())

        btn_row = ctk.CTkFrame(self, fg_color="transparent")
        btn_row.pack(pady=15)
        
        ctk.CTkButton(btn_row, text="Aceptar", width=100, command=self._accept, fg_color="#16a34a", hover_color="#15803d").pack(side="left", padx=5)
        ctk.CTkButton(btn_row, text="Cancelar", width=100, command=self._cancel, fg_color="#334155").pack(side="left", padx=5)

        self.protocol("WM_DELETE_WINDOW", self._cancel)

    def _accept(self):
        label = self.entry.get().strip().lower()
        if label:
            self.on_accept(label)
            self.destroy()

    def _cancel(self):
        if self.on_cancel: self.on_cancel()
        self.destroy()
