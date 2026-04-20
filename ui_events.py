import customtkinter as ctk
from tkinter import messagebox

class EventConfigWindow(ctk.CTkToplevel):
    """Interfaz Gráfica para configurar Hitos y Eventos. Se integra con EventEngine."""

    def __init__(self, parent, event_engine, available_classes, zones_count):
        super().__init__(parent)
        self.engine = event_engine
        self.available_classes = ["Cualquiera"] + (available_classes or [])
        self.zones_count = zones_count
        self.rule_widgets = []

        self.title("⚙️ Gestor de Hitos y Eventos")
        win_w, win_h = 650, 600
        x = (self.winfo_screenwidth() // 2) - (win_w // 2)
        y = (self.winfo_screenheight() // 2) - (win_h // 2)
        self.geometry(f"{win_w}x{win_h}+{x}+{y}")
        self.resizable(False, False)
        self.grab_set()

        # Título
        ctk.CTkLabel(self, text="HITOS Y AVISOS", font=ctk.CTkFont(size=20, weight="bold"), text_color="#0ea5e9").pack(pady=(15, 5))
        ctk.CTkLabel(self, text="Configura reglas para disparar acciones automáticas", font=ctk.CTkFont(size=12), text_color="#aaa").pack(pady=(0, 15))

        # Split: Creador (Arriba) / Lista (Abajo)
        self._build_creator_frame()
        
        self.list_frame = ctk.CTkScrollableFrame(self)
        self.list_frame.pack(padx=20, pady=20, fill="both", expand=True)
        
        self._refresh_list()

    def _build_creator_frame(self):
        frame = ctk.CTkFrame(self, fg_color="#1a1c1e")
        frame.pack(padx=20, pady=10, fill="x")

        # Fila 1: Nombre
        r1 = ctk.CTkFrame(frame, fg_color="transparent")
        r1.pack(fill="x", padx=10, pady=(10, 5))
        ctk.CTkLabel(r1, text="Nombre del Hito:", width=120, anchor="w").pack(side="left")
        self.entry_name = ctk.CTkEntry(r1, placeholder_text="Ej: Demasiadas personas")
        self.entry_name.pack(side="left", fill="x", expand=True)

        # Fila 2: Objetivo (Clase y Zona)
        r2 = ctk.CTkFrame(frame, fg_color="transparent")
        r2.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(r2, text="Objeto:", width=70, anchor="w").pack(side="left")
        self.sel_class = ctk.CTkOptionMenu(r2, values=self.available_classes, width=130)
        self.sel_class.pack(side="left", padx=(0, 15))
        
        ctk.CTkLabel(r2, text="Zona:", width=50, anchor="w").pack(side="left")
        zone_vals = ["Global"] + [f"Zona {i+1}" for i in range(self.zones_count)]
        self.sel_zone = ctk.CTkOptionMenu(r2, values=zone_vals, width=120)
        self.sel_zone.pack(side="left")

        # Fila 3: Lógica
        r3 = ctk.CTkFrame(frame, fg_color="transparent")
        r3.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(r3, text="Condición:", width=80, anchor="w").pack(side="left")
        self.sel_op = ctk.CTkOptionMenu(r3, values=[">", "<", "==", "Total >"], width=80)
        self.sel_op.pack(side="left", padx=(0, 5))
        self.entry_val = ctk.CTkEntry(r3, width=60, placeholder_text="Num")
        self.entry_val.pack(side="left", padx=(0, 15))

        # Fila 4: Acción y Cooldown
        r4 = ctk.CTkFrame(frame, fg_color="transparent")
        r4.pack(fill="x", padx=10, pady=(5, 10))
        
        ctk.CTkLabel(r4, text="Acción:", width=60, anchor="w").pack(side="left")
        self.sel_action = ctk.CTkOptionMenu(r4, values=["Guardar en Log", "Enviar a Telegram", "Google Apps Script"], width=160)
        self.sel_action.pack(side="left", padx=(0, 15))
        
        ctk.CTkLabel(r4, text="Espera (s):", width=70, anchor="w").pack(side="left")
        self.entry_cd = ctk.CTkEntry(r4, width=60, placeholder_text="10")
        self.entry_cd.insert(0, "10")
        self.entry_cd.pack(side="left")

        # Fila 5: Mensaje Personalizado (Variables soportadas)
        r5 = ctk.CTkFrame(frame, fg_color="transparent")
        r5.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(r5, text="Msj Personalizado:", width=110, anchor="w").pack(side="left")
        self.entry_msg = ctk.CTkEntry(r5, placeholder_text="Variables: {conteo}, {clase}, {zona}, {nombre}", font=("", 11))
        self.entry_msg.pack(side="left", fill="x", expand=True)

        # Fila 6: Validación Secundaria
        self._section_validator(frame)

        # Botón Guardar
        ctk.CTkButton(frame, text="✅ Añadir Regla", command=self._add_rule, fg_color="#16a34a", hover_color="#15803d").pack(pady=10)

    def _section_validator(self, parent):
        sep = ctk.CTkFrame(parent, height=1, fg_color="#333")
        sep.pack(fill="x", padx=10, pady=5)
        
        self.val_enabled = ctk.BooleanVar(value=False)
        self.val_switch = ctk.CTkSwitch(parent, text="Validación Secundaria (Doble Factor)", variable=self.val_enabled, command=self._toggle_validator_ui, font=ctk.CTkFont(size=12, weight="bold"))
        self.val_switch.pack(pady=5, padx=10, anchor="w")
        
        self.val_frame = ctk.CTkFrame(parent, fg_color="transparent")
        
        r1 = ctk.CTkFrame(self.val_frame, fg_color="transparent")
        r1.pack(fill="x", pady=2)
        ctk.CTkLabel(r1, text="Proveedor:", width=80, anchor="w").pack(side="left")
        self.val_provider = ctk.CTkOptionMenu(r1, values=["IA Universal (YOLO-World)", "Segmentación Local", "Hugging Face", "Ollama / Local IP"], 
                                              width=220, command=self._on_provider_change)
        self.val_provider.pack(side="left")
        
        self.prompt_row = ctk.CTkFrame(self.val_frame, fg_color="transparent")
        ctk.CTkLabel(self.prompt_row, text="¿Qué buscar?:", width=80, anchor="w", font=("", 11, "bold"), text_color="#38bdf8").pack(side="left")
        self.val_prompt = ctk.CTkEntry(self.prompt_row, placeholder_text="Ej: casco rojo, mochila, extintor...")
        self.val_prompt.pack(side="left", fill="x", expand=True)
        
        r2 = ctk.CTkFrame(self.val_frame, fg_color="transparent")
        r2.pack(fill="x", pady=2)
        ctk.CTkLabel(r2, text="IP / URL:", width=80, anchor="w").pack(side="left")
        self.val_endpoint = ctk.CTkEntry(r2, placeholder_text="Ej: http://192.168.1.50:11434")
        self.val_endpoint.pack(side="left", fill="x", expand=True)
        
        r_model = ctk.CTkFrame(self.val_frame, fg_color="transparent")
        r_model.pack(fill="x", pady=2)
        ctk.CTkLabel(r_model, text="Modelo:", width=80, anchor="w").pack(side="left")
        self.val_model = ctk.CTkEntry(r_model, placeholder_text="Ej: llava, moondream, etc.")
        self.val_model.pack(side="left", fill="x", expand=True)

    def _on_provider_change(self, val):
        if "Universal" in val:
            self.prompt_row.pack(fill="x", pady=5)
        else:
            self.prompt_row.pack_forget()

    def _toggle_validator_ui(self):
        if self.val_enabled.get():
            self.val_frame.pack(fill="x", padx=20, pady=(0, 10))
            self.geometry("650x780")
            self.val_endpoint.focus()
        else:
            self.val_frame.pack_forget()
            self.geometry("650x600")

    def _refresh_list(self):
        for widget in self.rule_widgets:
            widget.destroy()
        self.rule_widgets.clear()
            
        for rule in self.engine.rules:
            row = ctk.CTkFrame(self.list_frame, fg_color="#262b30")
            row.pack(fill="x", pady=2, padx=2)
            self.rule_widgets.append(row)
            
            z_txt = "Global" if rule['zone_target'] == -1 else f"Z{rule['zone_target']+1}"
            desc = f"{rule['name']} | {rule['class_target']} {rule['condition_op']} {rule['condition_val']} en {z_txt} ➡️ {rule['action']}"
            
            ctk.CTkLabel(row, text=desc, font=ctk.CTkFont(size=11), anchor="w").pack(side="left", padx=10, pady=5, expand=True, fill="x")
            
            btn = ctk.CTkButton(row, text="🗑", width=30, fg_color="#dc2626", hover_color="#991b1b", 
                                command=lambda r_id=rule['id']: self._remove_rule(r_id))
            btn.pack(side="right", padx=5)

            if rule.get("validator", {}).get("provider") != "None":
                ctk.CTkLabel(row, text="🛡️ Validador Activo", font=ctk.CTkFont(size=9, weight="bold"), text_color="#38bdf8").pack(side="right", padx=10)

    def _add_rule(self):
        name = self.entry_name.get().strip()
        val_str = self.entry_val.get().strip()
        cd_str = self.entry_cd.get().strip()

        if not name or not val_str or not cd_str:
            messagebox.showerror("Error", "Rellena todos los campos (Nombre, Condición y Espera).")
            return
            
        try: val = int(val_str)
        except: return messagebox.showerror("Error", "La condición debe ser un número entero.")
        
        try: cd = int(cd_str)
        except: return messagebox.showerror("Error", "El tiempo de espera debe ser un número entero.")
        
        zona_val = self.sel_zone.get()
        z_idx = -1 if zona_val == "Global" else int(zona_val.replace("Zona ", "")) - 1
        
        action_val = self.sel_action.get()
        act_code = "log"
        if "Telegram" in action_val: act_code = "telegram"
        elif "Script" in action_val: act_code = "webhook"

        validator_config = {"provider": "None"}
        if self.val_enabled.get():
            p_val = self.val_provider.get()
            provider_id = "None"
            if "Universal" in p_val: provider_id = "universal"
            elif "Segmentación" in p_val: provider_id = "local_seg"
            elif "Hugging" in p_val: provider_id = "huggingface"
            elif "Ollama" in p_val: provider_id = "ollama"
            
            validator_config = {
                "provider": provider_id,
                "endpoint": self.val_endpoint.get().strip(),
                "model": self.val_model.get().strip(),
                "prompt": self.val_prompt.get().strip()
            }

        self.engine.add_rule(
            name=name,
            class_target=self.sel_class.get(),
            zone_target=z_idx,
            condition_op=self.sel_op.get(),
            condition_val=val,
            action_type=act_code,
            cooldown=cd,
            validator_config=validator_config,
            custom_msg=self.entry_msg.get().strip()
        )
        self.entry_name.delete(0, 'end')
        self.entry_val.delete(0, 'end')
        self.entry_msg.delete(0, 'end')
        self._refresh_list()

    def _remove_rule(self, rule_id):
        self.engine.remove_rule(rule_id)
        self._refresh_list()
