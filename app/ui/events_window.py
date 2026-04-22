"""
Ventana de Configuración de Eventos.
Permite definir reglas lógicas y acciones automáticas.
"""

import customtkinter as ctk
from tkinter import messagebox
from ..utils.error_handler import log_error

class EventsWindow(ctk.CTkToplevel):
    """
    Interfaz Gráfica para configurar Hitos y Eventos.
    """

    def __init__(self, parent, event_engine, available_classes, zones_count):
        super().__init__(parent)
        self.engine = event_engine
        self.available_classes = ["Cualquiera"] + (available_classes or [])
        self.zones_count = zones_count
        self.rule_widgets = []

        self.title("⚙️ Gestor de Hitos y Eventos")
        self.geometry("650x600")
        self.grab_set()

        ctk.CTkLabel(self, text="HITOS Y AVISOS", font=ctk.CTkFont(size=20, weight="bold")).pack(pady=15)
        
        self._build_creator_frame()
        self.list_frame = ctk.CTkScrollableFrame(self)
        self.list_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        self._refresh_list()

    def _build_creator_frame(self):
        """Construye el formulario de creación de reglas."""
        frame = ctk.CTkFrame(self)
        frame.pack(fill="x", padx=20, pady=10)
        
        self.entry_name = ctk.CTkEntry(frame, placeholder_text="Nombre del Hito")
        self.entry_name.pack(pady=5, padx=10, fill="x")
        
        self.sel_class = ctk.CTkOptionMenu(frame, values=self.available_classes)
        self.sel_class.pack(pady=5, padx=10, side="left")
        
        self.entry_val = ctk.CTkEntry(frame, width=60, placeholder_text="Valor")
        self.entry_val.pack(pady=5, padx=10, side="left")

        ctk.CTkButton(frame, text="Añadir Regla", command=self._add_rule).pack(pady=5, padx=10, side="right")

    def _refresh_list(self):
        """Actualiza la lista de reglas activas."""
        for w in self.rule_widgets: w.destroy()
        self.rule_widgets.clear()
        
        for rule in self.engine.rules:
            row = ctk.CTkFrame(self.list_frame)
            row.pack(fill="x", pady=2)
            self.rule_widgets.append(row)
            
            lbl = ctk.CTkLabel(row, text=f"{rule['name']} | {rule['class_target']} | {rule['action']}")
            lbl.pack(side="left", padx=10)
            
            btn = ctk.CTkButton(row, text="🗑", width=30, fg_color="#dc2626", 
                                command=lambda rid=rule['id']: self._remove_rule(rid))
            btn.pack(side="right", padx=5)

    def _add_rule(self):
        try:
            name = self.entry_name.get().strip()
            val = int(self.entry_val.get().strip())
            if not name: raise ValueError("Nombre vacío")
            
            self.engine.add_rule(
                name=name,
                class_target=self.sel_class.get(),
                zone_target=-1,
                condition_op=">",
                condition_val=val,
                action_type="log",
                cooldown=10
            )
            self._refresh_list()
        except Exception as e:
            log_error("EXE-GUI-COMP-02", f"Error al añadir regla: {e}")
            messagebox.showerror("Error", "Datos de hito no válidos.")

    def _remove_rule(self, rule_id):
        self.engine.remove_rule(rule_id)
        self._refresh_list()
