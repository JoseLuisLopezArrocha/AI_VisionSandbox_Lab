"""
Ventana de Configuración de Eventos y Hitos.
Permite definir reglas lógicas flexibles con condiciones, zonas, acciones y cooldowns configurables.
"""

import customtkinter as ctk
from tkinter import messagebox
from ..utils.error_handler import log_error

# Constantes de configuración
CONDITION_OPS = [">", "<", "==", ">=", "<=", "Total >"]
ACTION_TYPES = [
    ("📋 Solo Log", "log"),
    ("📨 Telegram", "telegram"),
    ("🌐 Webhook", "webhook"),
    ("🔊 Voz (TTS)", "tts"),
    ("📋+📨+🔊 Todo", "all"),
]
COOLDOWN_PRESETS = [
    ("5s", 5), ("10s", 10), ("30s", 30), 
    ("1 min", 60), ("5 min", 300), ("Sin límite", 0),
]

class EventsWindow(ctk.CTkToplevel):
    """
    Interfaz completa para gestionar Hitos y Eventos con configuración flexible.
    """

    def __init__(self, parent, event_engine, available_classes, zones_count):
        super().__init__(parent)
        self.engine = event_engine
        self.available_classes = ["Cualquiera"] + (available_classes or [])
        self.zones_count = zones_count
        self.rule_widgets = []

        self.title("🔔 Gestor de Hitos y Eventos")
        self.geometry("720x700")
        self.grab_set()
        
        # Header
        header = ctk.CTkFrame(self, fg_color="#0f172a", corner_radius=0)
        header.pack(fill="x")
        ctk.CTkLabel(header, text="HITOS Y AVISOS", 
                     font=ctk.CTkFont(size=20, weight="bold"), 
                     text_color="#38bdf8").pack(pady=12)
        ctk.CTkLabel(header, text="Configura reglas inteligentes que se disparan en tiempo real", 
                     font=ctk.CTkFont(size=11), 
                     text_color="#64748b").pack(pady=(0, 10))

        # Formulario de Creación
        self._build_creator_frame()

        # Separador
        ctk.CTkLabel(self, text="REGLAS ACTIVAS", font=ctk.CTkFont(size=12, weight="bold"),
                     text_color="#94a3b8").pack(pady=(10, 2), padx=20, anchor="w")
        
        # Lista de reglas
        self.list_frame = ctk.CTkScrollableFrame(self, fg_color="transparent")
        self.list_frame.pack(fill="both", expand=True, padx=20, pady=(0, 15))
        
        self._refresh_list()

    def _build_creator_frame(self):
        """Construye el formulario completo de creación de reglas."""
        frame = ctk.CTkFrame(self, fg_color="#1e293b", border_width=1, border_color="#334155")
        frame.pack(fill="x", padx=20, pady=10)

        # Fila 1: Nombre del hito
        ctk.CTkLabel(frame, text="Nombre del Hito:", font=ctk.CTkFont(size=11, weight="bold"),
                     text_color="#38bdf8").pack(pady=(10, 2), padx=12, anchor="w")
        self.entry_name = ctk.CTkEntry(frame, placeholder_text="Ej: Alerta multitud, Zona vacía...")
        self.entry_name.pack(pady=(0, 8), padx=12, fill="x")

        # Fila 2: Condición (Clase + Operador + Valor)
        cond_frame = ctk.CTkFrame(frame, fg_color="transparent")
        cond_frame.pack(fill="x", padx=12, pady=2)
        
        # Clase objetivo
        cls_col = ctk.CTkFrame(cond_frame, fg_color="transparent")
        cls_col.pack(side="left", fill="x", expand=True, padx=(0, 5))
        ctk.CTkLabel(cls_col, text="Clase:", font=ctk.CTkFont(size=10), text_color="#94a3b8").pack(anchor="w")
        self.sel_class = ctk.CTkOptionMenu(cls_col, values=self.available_classes, width=140)
        self.sel_class.pack(fill="x")

        # Operador
        op_col = ctk.CTkFrame(cond_frame, fg_color="transparent")
        op_col.pack(side="left", padx=5)
        ctk.CTkLabel(op_col, text="Condición:", font=ctk.CTkFont(size=10), text_color="#94a3b8").pack(anchor="w")
        self.sel_op = ctk.CTkOptionMenu(op_col, values=CONDITION_OPS, width=80)
        self.sel_op.set(">")
        self.sel_op.pack()

        # Valor
        val_col = ctk.CTkFrame(cond_frame, fg_color="transparent")
        val_col.pack(side="left", padx=5)
        ctk.CTkLabel(val_col, text="Valor:", font=ctk.CTkFont(size=10), text_color="#94a3b8").pack(anchor="w")
        self.entry_val = ctk.CTkEntry(val_col, width=60, placeholder_text="5")
        self.entry_val.pack()

        # Fila 3: Zona + Acción + Cooldown
        opt_frame = ctk.CTkFrame(frame, fg_color="transparent")
        opt_frame.pack(fill="x", padx=12, pady=(8, 2))

        # Zona objetivo
        zone_col = ctk.CTkFrame(opt_frame, fg_color="transparent")
        zone_col.pack(side="left", fill="x", expand=True, padx=(0, 5))
        ctk.CTkLabel(zone_col, text="Zona:", font=ctk.CTkFont(size=10), text_color="#94a3b8").pack(anchor="w")
        zone_values = ["Global"] + [f"Zona {i+1}" for i in range(self.zones_count)]
        self.sel_zone = ctk.CTkOptionMenu(zone_col, values=zone_values, width=100)
        self.sel_zone.pack(fill="x")

        # Acción
        act_col = ctk.CTkFrame(opt_frame, fg_color="transparent")
        act_col.pack(side="left", fill="x", expand=True, padx=5)
        ctk.CTkLabel(act_col, text="Acción:", font=ctk.CTkFont(size=10), text_color="#94a3b8").pack(anchor="w")
        self.sel_action = ctk.CTkOptionMenu(act_col, values=[a[0] for a in ACTION_TYPES], width=140)
        self.sel_action.pack(fill="x")

        # Cooldown
        cd_col = ctk.CTkFrame(opt_frame, fg_color="transparent")
        cd_col.pack(side="left", padx=(5, 0))
        ctk.CTkLabel(cd_col, text="Cooldown:", font=ctk.CTkFont(size=10), text_color="#94a3b8").pack(anchor="w")
        self.sel_cooldown = ctk.CTkOptionMenu(cd_col, values=[c[0] for c in COOLDOWN_PRESETS], width=90)
        self.sel_cooldown.set("10s")
        self.sel_cooldown.pack()

        # Botón Añadir
        ctk.CTkButton(frame, text="➕ Crear Regla", command=self._add_rule,
                      fg_color="#16a34a", hover_color="#15803d", height=34,
                      font=ctk.CTkFont(size=12, weight="bold")).pack(pady=10, padx=12, fill="x")

    def _refresh_list(self):
        """Actualiza la lista de reglas activas con formato visual."""
        for w in self.rule_widgets:
            w.destroy()
        self.rule_widgets.clear()

        if not self.engine.rules:
            empty = ctk.CTkLabel(self.list_frame, text="No hay reglas configuradas.\nCrea tu primer hito arriba ☝️",
                                 font=ctk.CTkFont(size=12), text_color="#475569")
            empty.pack(pady=30)
            self.rule_widgets.append(empty)
            return

        for rule in self.engine.rules:
            row = ctk.CTkFrame(self.list_frame, fg_color="#1e293b", border_width=1, border_color="#334155")
            row.pack(fill="x", pady=3)
            self.rule_widgets.append(row)

            # Info principal
            info = ctk.CTkFrame(row, fg_color="transparent")
            info.pack(side="left", fill="x", expand=True, padx=10, pady=6)

            # Nombre con icono
            action_icon = self._get_action_icon(rule.get('action', 'log'))
            ctk.CTkLabel(info, text=f"{action_icon} {rule['name']}",
                        font=ctk.CTkFont(size=12, weight="bold"),
                        text_color="#e2e8f0").pack(anchor="w")
            
            # Detalles
            zone_txt = "Global" if rule.get('zone_target', -1) == -1 else f"Zona {rule['zone_target'] + 1}"
            cooldown_txt = f"{rule.get('cooldown', 10)}s"
            detail = f"{rule['class_target']} {rule['condition_op']} {rule['condition_val']} • {zone_txt} • ⏱ {cooldown_txt}"
            ctk.CTkLabel(info, text=detail,
                        font=ctk.CTkFont(size=10), text_color="#64748b").pack(anchor="w")

            # Botón eliminar
            ctk.CTkButton(row, text="🗑", width=32, height=32,
                         fg_color="#dc2626", hover_color="#b91c1c",
                         command=lambda rid=rule['id']: self._remove_rule(rid)).pack(side="right", padx=8, pady=6)

    def _get_action_icon(self, action):
        icons = {"log": "📋", "telegram": "📨", "webhook": "🌐", "tts": "🔊", "all": "🚨"}
        return icons.get(action, "📋")

    def _add_rule(self):
        try:
            name = self.entry_name.get().strip()
            val_text = self.entry_val.get().strip()
            if not name:
                messagebox.showwarning("Aviso", "Escribe un nombre para el hito.")
                return
            if not val_text:
                messagebox.showwarning("Aviso", "Introduce un valor numérico para la condición.")
                return
            
            val = int(val_text)
            
            # Resolver zona (-1 = Global, 0..N = índice de zona)
            zone_str = self.sel_zone.get()
            zone_target = -1
            if zone_str != "Global":
                zone_target = int(zone_str.replace("Zona ", "")) - 1

            # Resolver acción (mapear display → valor interno)
            action_display = self.sel_action.get()
            action_type = "log"
            for display, internal in ACTION_TYPES:
                if display == action_display:
                    action_type = internal
                    break

            # Resolver cooldown
            cooldown_display = self.sel_cooldown.get()
            cooldown = 10
            for display, val_cd in COOLDOWN_PRESETS:
                if display == cooldown_display:
                    cooldown = val_cd
                    break

            self.engine.add_rule(
                name=name,
                class_target=self.sel_class.get(),
                zone_target=zone_target,
                condition_op=self.sel_op.get(),
                condition_val=val,
                action_type=action_type,
                cooldown=cooldown
            )

            # Limpiar formulario
            self.entry_name.delete(0, "end")
            self.entry_val.delete(0, "end")
            self._refresh_list()
        except ValueError:
            messagebox.showerror("Error", "El valor debe ser un número entero.")
        except Exception as e:
            log_error("EXE-GUI-COMP-02", f"Error al añadir regla: {e}")
            messagebox.showerror("Error", f"No se pudo crear la regla: {e}")

    def _remove_rule(self, rule_id):
        if messagebox.askyesno("Confirmar", "¿Eliminar esta regla?"):
            self.engine.remove_rule(rule_id)
            self._refresh_list()
