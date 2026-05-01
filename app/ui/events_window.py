"""
Ventana de Configuracion de Eventos y Hitos.
Permite definir reglas logicas flexibles con condiciones, zonas, acciones,
cooldowns y proveedores de validacion secundaria configurables.
"""
import customtkinter as ctk
from tkinter import messagebox
from ..utils.error_handler import log_error
CONDITION_OPS = ['>', '<', '==', '>=', '<=', 'Total >']
ZONE_OPERATORS = [('DENTRO (O)', 'OR'), ('DENTRO (Y)', 'AND'), ('FUERA (Exclusion)', 'NOT')]
SEVERITY_LEVELS = [('Info', '#3b82f6'), ('Alerta', '#eab308'), ('Critico', '#ef4444')]
PERSISTENCE_PRESETS = [('Instantaneo', 0), ('2s', 2), ('5s', 5), ('10s', 10), ('30s', 30)]
COOLDOWN_PRESETS = [('5s', 5), ('10s', 10), ('30s', 30), ('1 min', 60), ('5 min', 300), ('Sin limite', 0)]
VALIDATOR_PROVIDERS = [('Sin validacion', 'None'), ('YOLO-World (Universal)', 'universal'), ('Segmentacion local', 'local_seg'), ('Ollama (VLM)', 'ollama'), ('HuggingFace (API)', 'huggingface')]

class EventsHelpWindow(ctk.CTkToplevel):
    """Ventana informativa sobre el funcionamiento de los Hitos y Eventos."""

    def __init__(self, parent):
        super().__init__(parent)
        self.title('Guia de Hitos y Eventos')
        self.geometry('500x550')
        self.grab_set()
        self.resizable(False, False)
        ctk.CTkLabel(self, text='GUÍA DE PROTOCOLOS INTELIGENTES', font=ctk.CTkFont(size=16, weight='bold'), text_color='#38bdf8', corner_radius=0).pack(pady=(20, 15))
        scroll = ctk.CTkScrollableFrame(self, fg_color='transparent', corner_radius=0)
        scroll.pack(fill='both', expand=True, padx=20, pady=10)
        self._add_topic(scroll, 'Nombre del Hito', 'Es el identificador de la alerta. Aparecera en los logs y en los mensajes de Telegram/Webhook.')
        self._add_topic(scroll, 'Clase', "El tipo de objeto que activara la regla. 'Cualquiera' sirve para detectar cualquier movimiento.")
        self._add_topic(scroll, 'Condicion y Valor', "La logica matematica (>, <, ==). 'Total >' permite disparar alertas basadas en el conteo historico acumulado.")
        self._add_topic(scroll, 'Zona', 'Define si la regla aplica a todo el video (Global) o a una de las zonas poligonales que hayas dibujado.')
        self._add_topic(scroll, 'Accion', 'Puedes combinar multiples acciones: Log Visual, enviar a Telegram, Webhook o reproducir una alerta por Voz (TTS). Tambien puedes guardar una Foto de evidencia.')
        self._add_topic(scroll, 'Persistencia y Severidad', 'La Persistencia evita falsas alarmas exigiendo que la deteccion se mantenga X segundos. La Severidad clasifica el hito como Info, Alerta o Critico.')
        self._add_topic(scroll, 'Cooldown', 'Tiempo de espera obligatorio tras un disparo para evitar saturar el sistema con alertas repetidas.')
        self._add_topic(scroll, 'Validacion Secundaria', 'Permite usar un segundo modelo de IA para confirmar la deteccion antes de disparar la alerta.\n\n- YOLO-World: Busqueda universal por texto.\n- Segmentacion: Verificacion con modelo local.\n- Ollama: Modelo multimodal local (requiere endpoint).\n- HuggingFace: API de inferencia remota (requiere API Key).\n\nConfigura los endpoints y claves en el panel de Ajustes.')
        self._add_topic(scroll, 'Donde se guarda todo?', "1. Reglas: En 'config/events_config.json'.\n2. Evidencias: En 'telemetry_logs/evidences/'.\n3. Historial: En la base de datos SQLite 'telemetry_logs/vision_events.db'.")
        ctk.CTkButton(self, text='CONFIRMAR LECTURA', command=self.destroy, fg_color='#1e40af', hover_color='#1e3a8a', font=ctk.CTkFont(size=11, weight='bold'), corner_radius=0).pack(pady=20, padx=40, fill='x')

    def _add_topic(self, parent, title, text):
        f = ctk.CTkFrame(parent, fg_color='transparent', corner_radius=0)
        f.pack(fill='x', pady=8)
        ctk.CTkLabel(f, text=title, font=ctk.CTkFont(size=12, weight='bold'), text_color='#38bdf8', anchor='w', corner_radius=0).pack(fill='x')
        ctk.CTkLabel(f, text=text, font=ctk.CTkFont(size=11), text_color='#94a3b8', justify='left', wraplength=420, corner_radius=0).pack(fill='x', pady=(2, 0))

class EventsWindow(ctk.CTkToplevel):
    """
    Interfaz completa para gestionar Hitos y Eventos con configuracion flexible.
    """

    def __init__(self, parent, event_engine, available_classes, zones_count):
        super().__init__(parent)
        self.engine = event_engine
        self.available_classes = ['Cualquiera'] + (available_classes or [])
        self.zones_count = zones_count
        self.rule_widgets = []
        self.title('Gestor de Hitos y Eventos')
        self.geometry('750x780')
        self.grab_set()
        header = ctk.CTkFrame(self, fg_color='#0f172a', corner_radius=0)
        header.pack(fill='x')
        title_row = ctk.CTkFrame(header, fg_color='transparent', corner_radius=0)
        title_row.pack(pady=(12, 0), padx=20, fill='x')
        ctk.CTkLabel(title_row, text='GESTOR DE EVENTOS TÁCTICOS', font=ctk.CTkFont(size=18, weight='bold'), text_color='#38bdf8', corner_radius=0).pack(side='left', expand=True, padx=(40, 0))
        ctk.CTkButton(title_row, text='?', width=30, height=30, fg_color='#1e293b', hover_color='#334155', text_color='#38bdf8', font=ctk.CTkFont(size=14, weight='bold'), command=lambda: EventsHelpWindow(self), corner_radius=0).pack(side='right')
        ctk.CTkLabel(header, text='Configura reglas inteligentes que se disparan en tiempo real', font=ctk.CTkFont(size=11), text_color='#64748b', corner_radius=0).pack(pady=(0, 10))
        self._build_creator_frame()
        ctk.CTkLabel(self, text='REGLAS ACTIVAS', font=ctk.CTkFont(size=12, weight='bold'), text_color='#94a3b8', corner_radius=0).pack(pady=(10, 2), padx=20, anchor='w')
        self.list_frame = ctk.CTkScrollableFrame(self, fg_color='transparent', corner_radius=0)
        self.list_frame.pack(fill='both', expand=True, padx=20, pady=(0, 15))
        self._refresh_list()

    def _build_creator_frame(self):
        """Construye el formulario completo de creacion de reglas."""
        frame = ctk.CTkFrame(self, fg_color='#1e293b', border_width=1, border_color='#334155', corner_radius=0)
        frame.pack(fill='x', padx=20, pady=10)
        ctk.CTkLabel(frame, text='Nombre del Hito:', font=ctk.CTkFont(size=11, weight='bold'), text_color='#38bdf8', corner_radius=0).pack(pady=(10, 2), padx=12, anchor='w')
        self.entry_name = ctk.CTkEntry(frame, placeholder_text='Ej: Alerta multitud, Zona vacia...', corner_radius=0)
        self.entry_name.pack(pady=(0, 8), padx=12, fill='x')
        cond_frame = ctk.CTkFrame(frame, fg_color='transparent', corner_radius=0)
        cond_frame.pack(fill='x', padx=12, pady=2)
        cls_col = ctk.CTkFrame(cond_frame, fg_color='transparent', corner_radius=0)
        cls_col.pack(side='left', fill='x', expand=True, padx=(0, 5))
        ctk.CTkLabel(cls_col, text='Clase:', font=ctk.CTkFont(size=10), text_color='#94a3b8', corner_radius=0).pack(anchor='w')
        self.sel_class = ctk.CTkOptionMenu(cls_col, values=self.available_classes, width=140, corner_radius=0)
        self.sel_class.pack(fill='x')
        op_col = ctk.CTkFrame(cond_frame, fg_color='transparent', corner_radius=0)
        op_col.pack(side='left', padx=5)
        ctk.CTkLabel(op_col, text='Condicion:', font=ctk.CTkFont(size=10), text_color='#94a3b8', corner_radius=0).pack(anchor='w')
        self.sel_op = ctk.CTkOptionMenu(op_col, values=CONDITION_OPS, width=80, corner_radius=0)
        self.sel_op.set('>')
        self.sel_op.pack()
        val_col = ctk.CTkFrame(cond_frame, fg_color='transparent', corner_radius=0)
        val_col.pack(side='left', padx=5)
        ctk.CTkLabel(val_col, text='Valor:', font=ctk.CTkFont(size=10), text_color='#94a3b8', corner_radius=0).pack(anchor='w')
        self.entry_val = ctk.CTkEntry(val_col, width=60, placeholder_text='5', corner_radius=0)
        self.entry_val.pack()
        opt_frame = ctk.CTkFrame(frame, fg_color='transparent', corner_radius=0)
        opt_frame.pack(fill='x', padx=12, pady=(8, 2))
        zone_col = ctk.CTkFrame(opt_frame, fg_color='transparent', corner_radius=0)
        zone_col.pack(side='left', fill='x', expand=True, padx=(0, 5))
        ctk.CTkLabel(zone_col, text='Zonas:', font=ctk.CTkFont(size=10), text_color='#94a3b8', corner_radius=0).pack(anchor='w')
        self.selected_zones = [-1]
        self.btn_zone = ctk.CTkButton(zone_col, text='Global', width=80, command=self._open_zone_selector, fg_color='#334155', hover_color='#475569', corner_radius=0)
        self.btn_zone.pack(fill='x')
        zop_col = ctk.CTkFrame(opt_frame, fg_color='transparent', corner_radius=0)
        zop_col.pack(side='left', fill='x', expand=True, padx=5)
        ctk.CTkLabel(zop_col, text='Lugar:', font=ctk.CTkFont(size=10), text_color='#94a3b8', corner_radius=0).pack(anchor='w')
        self.sel_zop = ctk.CTkOptionMenu(zop_col, values=[z[0] for z in ZONE_OPERATORS], width=110, corner_radius=0)
        self.sel_zop.pack(fill='x')
        sev_col = ctk.CTkFrame(opt_frame, fg_color='transparent', corner_radius=0)
        sev_col.pack(side='left', fill='x', expand=True, padx=5)
        ctk.CTkLabel(sev_col, text='Severidad:', font=ctk.CTkFont(size=10), text_color='#94a3b8', corner_radius=0).pack(anchor='w')
        self.sel_severity = ctk.CTkOptionMenu(sev_col, values=[s[0] for s in SEVERITY_LEVELS], width=100, corner_radius=0)
        self.sel_severity.pack(fill='x')
        act_frame = ctk.CTkFrame(frame, fg_color='transparent', corner_radius=0)
        act_frame.pack(fill='x', padx=12, pady=(8, 2))
        ctk.CTkLabel(act_frame, text='Acciones a disparar:', font=ctk.CTkFont(size=10), text_color='#94a3b8', corner_radius=0).pack(anchor='w')
        checks_row = ctk.CTkFrame(act_frame, fg_color='transparent', corner_radius=0)
        checks_row.pack(fill='x')
        self.chk_log = ctk.CTkCheckBox(checks_row, text='Log', width=50, corner_radius=0)
        self.chk_log.pack(side='left', padx=(0, 5))
        self.chk_log.select()
        self.chk_telegram = ctk.CTkCheckBox(checks_row, text='Telegram', width=70, corner_radius=0)
        self.chk_telegram.pack(side='left', padx=5)
        self.chk_webhook = ctk.CTkCheckBox(checks_row, text='Webhook', width=70, corner_radius=0)
        self.chk_webhook.pack(side='left', padx=5)
        self.chk_tts = ctk.CTkCheckBox(checks_row, text='Voz', width=50, corner_radius=0)
        self.chk_tts.pack(side='left', padx=5)
        self.chk_photo = ctk.CTkCheckBox(checks_row, text='Foto', width=50, corner_radius=0)
        self.chk_photo.pack(side='left', padx=5)
        time_frame = ctk.CTkFrame(frame, fg_color='transparent', corner_radius=0)
        time_frame.pack(fill='x', padx=12, pady=(8, 2))
        cd_col = ctk.CTkFrame(time_frame, fg_color='transparent', corner_radius=0)
        cd_col.pack(side='left', fill='x', expand=True, padx=(0, 5))
        ctk.CTkLabel(cd_col, text='Cooldown:', font=ctk.CTkFont(size=10), text_color='#94a3b8', corner_radius=0).pack(anchor='w')
        self.sel_cooldown = ctk.CTkOptionMenu(cd_col, values=[c[0] for c in COOLDOWN_PRESETS], width=90, corner_radius=0)
        self.sel_cooldown.set('10s')
        self.sel_cooldown.pack(fill='x')
        pers_col = ctk.CTkFrame(time_frame, fg_color='transparent', corner_radius=0)
        pers_col.pack(side='left', fill='x', expand=True, padx=(5, 0))
        ctk.CTkLabel(pers_col, text='Persistencia:', font=ctk.CTkFont(size=10), text_color='#94a3b8', corner_radius=0).pack(anchor='w')
        self.sel_persistence = ctk.CTkOptionMenu(pers_col, values=[p[0] for p in PERSISTENCE_PRESETS], width=90, corner_radius=0)
        self.sel_persistence.set('Instantaneo')
        self.sel_persistence.pack(fill='x')
        val_frame = ctk.CTkFrame(frame, fg_color='transparent', corner_radius=0)
        val_frame.pack(fill='x', padx=12, pady=(8, 2))
        prov_col = ctk.CTkFrame(val_frame, fg_color='transparent', corner_radius=0)
        prov_col.pack(side='left', fill='x', expand=True, padx=(0, 5))
        ctk.CTkLabel(prov_col, text='Validacion IA:', font=ctk.CTkFont(size=10), text_color='#a78bfa', corner_radius=0).pack(anchor='w')
        self.sel_validator = ctk.CTkOptionMenu(prov_col, values=[v[0] for v in VALIDATOR_PROVIDERS], width=180, command=self._on_validator_change, corner_radius=0)
        self.sel_validator.pack(fill='x')
        prompt_col = ctk.CTkFrame(val_frame, fg_color='transparent', corner_radius=0)
        prompt_col.pack(side='left', fill='x', expand=True, padx=(5, 0))
        ctk.CTkLabel(prompt_col, text='Prompt / Clase:', font=ctk.CTkFont(size=10), text_color='#a78bfa', corner_radius=0).pack(anchor='w')
        self.entry_val_prompt = ctk.CTkEntry(prompt_col, placeholder_text='Ej: persona, casco, fuego...', corner_radius=0)
        self.entry_val_prompt.pack(fill='x')
        ctk.CTkButton(frame, text='REGISTRAR NUEVA REGLA', command=self._add_rule, fg_color='#10b981', hover_color='#059669', height=34, font=ctk.CTkFont(size=12, weight='bold'), corner_radius=0).pack(pady=10, padx=12, fill='x')

    def _on_validator_change(self, value):
        """Muestra/oculta el campo de prompt segun el proveedor seleccionado."""
        needs_prompt = value != 'Sin validacion'
        state = 'normal' if needs_prompt else 'disabled'
        self.entry_val_prompt.configure(state=state)
        if not needs_prompt:
            self.entry_val_prompt.delete(0, 'end')

    def _open_zone_selector(self):
        top = ctk.CTkToplevel(self, corner_radius=0)
        top.title('Zonas')
        top.geometry('250x300')
        top.grab_set()
        ctk.CTkLabel(top, text='Selecciona Zonas Objetivo', font=ctk.CTkFont(weight='bold'), corner_radius=0).pack(pady=10)
        scroll = ctk.CTkScrollableFrame(top, corner_radius=0)
        scroll.pack(fill='both', expand=True, padx=10, pady=10)
        vars_dict = {}
        var_g = ctk.BooleanVar(value=-1 in self.selected_zones)
        chk_g = ctk.CTkCheckBox(scroll, text='Global', variable=var_g, corner_radius=0)
        chk_g.pack(anchor='w', pady=5)
        vars_dict[-1] = var_g
        for i in range(self.zones_count):
            var_i = ctk.BooleanVar(value=i in self.selected_zones)
            chk_i = ctk.CTkCheckBox(scroll, text=f'Zona {i + 1}', variable=var_i, corner_radius=0)
            chk_i.pack(anchor='w', pady=5)
            vars_dict[i] = var_i

        def apply():
            self.selected_zones = [k for k, v in vars_dict.items() if v.get()]
            if not self.selected_zones or -1 in self.selected_zones:
                self.selected_zones = [-1]
                self.btn_zone.configure(text='Global')
            else:
                txt = ','.join([str(z + 1) for z in self.selected_zones])
                self.btn_zone.configure(text=f'Zonas: {txt}')
            top.destroy()
        ctk.CTkButton(top, text='Aceptar', command=apply, corner_radius=0).pack(pady=10)

    def _refresh_list(self):
        """Actualiza la lista de reglas activas con formato visual."""
        for w in self.rule_widgets:
            w.destroy()
        self.rule_widgets.clear()
        if not self.engine.rules:
            empty = ctk.CTkLabel(self.list_frame, text='No hay reglas configuradas.\nCrea tu primer hito arriba.', font=ctk.CTkFont(size=12), text_color='#475569', corner_radius=0)
            empty.pack(pady=30)
            self.rule_widgets.append(empty)
            return
        for rule in self.engine.rules:
            row = ctk.CTkFrame(self.list_frame, fg_color='#1e293b', border_width=1, border_color='#334155', corner_radius=0)
            row.pack(fill='x', pady=3)
            self.rule_widgets.append(row)
            info = ctk.CTkFrame(row, fg_color='transparent', corner_radius=0)
            info.pack(side='left', fill='x', expand=True, padx=10, pady=6)
            sev = rule.get('severity', 'Info')
            color_sev = '#3b82f6'
            for s in SEVERITY_LEVELS:
                if s[0] == sev:
                    color_sev = s[1]
                    break
            title_frame = ctk.CTkFrame(info, fg_color='transparent', corner_radius=0)
            title_frame.pack(fill='x')
            ctk.CTkLabel(title_frame, text=f' {sev.upper()} ', font=ctk.CTkFont(size=10, weight='bold'), fg_color=color_sev, text_color='#0f172a', corner_radius=0).pack(side='left')
            ctk.CTkLabel(title_frame, text=f" {rule['name']}", font=ctk.CTkFont(size=12, weight='bold'), text_color='#e2e8f0', corner_radius=0).pack(side='left', padx=(5, 0))
            actions = rule.get('actions', [rule.get('action', 'log')])
            actions_txt = ','.join([self._get_action_tag(a) for a in actions])
            if rule.get('save_evidence'):
                actions_txt += ',FOTO'
            ctk.CTkLabel(title_frame, text=f'[{actions_txt}]', font=ctk.CTkFont(size=10), text_color='#94a3b8', corner_radius=0).pack(side='left', padx=(5, 0))
            targets = rule.get('zone_targets', [-1])
            if not targets or -1 in targets:
                zone_txt = 'Global'
            else:
                op_txt = rule.get('zone_operator', 'OR')
                readable_targets = [str(t + 1) for t in targets]
                zone_txt = f"Zonas [{','.join(readable_targets)}] ({op_txt})"
            cooldown_txt = f"{rule.get('cooldown', 10)}s"
            pers_txt = f" | Duracion: {rule.get('persistence', 0)}s" if rule.get('persistence', 0) > 0 else ''
            validator_txt = rule.get('validator', {}).get('provider', 'None')
            if validator_txt != 'None':
                validator_txt = f' | Val: {validator_txt}'
            else:
                validator_txt = ''
            detail = f"{rule['class_target']} {rule['condition_op']} {rule['condition_val']} | {zone_txt} | CD: {cooldown_txt}{pers_txt}{validator_txt}"
            ctk.CTkLabel(info, text=detail, font=ctk.CTkFont(size=10), text_color='#64748b', corner_radius=0).pack(anchor='w')
            ctk.CTkButton(row, text='X', width=32, height=32, fg_color='#dc2626', hover_color='#b91c1c', font=ctk.CTkFont(size=12, weight='bold'), command=lambda rid=rule['id']: self._remove_rule(rid), corner_radius=0).pack(side='right', padx=8, pady=6)

    def _get_action_tag(self, action):
        tags = {'log': 'LOG', 'telegram': 'TG', 'webhook': 'WH', 'tts': 'VOZ', 'all': 'ALL'}
        return tags.get(action, 'LOG')

    def _add_rule(self):
        try:
            name = self.entry_name.get().strip()
            val_text = self.entry_val.get().strip()
            if not name:
                messagebox.showwarning('Aviso', 'Escribe un nombre para el hito.')
                return
            if not val_text:
                messagebox.showwarning('Aviso', 'Introduce un valor numerico para la condicion.')
                return
            val = int(val_text)
            zone_targets = self.selected_zones
            zop_display = self.sel_zop.get()
            zone_operator = 'OR'
            for display, internal in ZONE_OPERATORS:
                if display == zop_display:
                    zone_operator = internal
                    break
            actions = []
            if self.chk_log.get():
                actions.append('log')
            if self.chk_telegram.get():
                actions.append('telegram')
            if self.chk_webhook.get():
                actions.append('webhook')
            if self.chk_tts.get():
                actions.append('tts')
            save_evidence = bool(self.chk_photo.get())
            if not actions:
                actions = ['log']
            severity = self.sel_severity.get()
            pers_display = self.sel_persistence.get()
            persistence = 0
            for display, val_p in PERSISTENCE_PRESETS:
                if display == pers_display:
                    persistence = val_p
                    break
            cooldown_display = self.sel_cooldown.get()
            cooldown = 10
            for display, val_cd in COOLDOWN_PRESETS:
                if display == cooldown_display:
                    cooldown = val_cd
                    break
            validator_display = self.sel_validator.get()
            validator_provider = 'None'
            for display, internal in VALIDATOR_PROVIDERS:
                if display == validator_display:
                    validator_provider = internal
                    break
            validator_prompt = self.entry_val_prompt.get().strip()
            self.engine.add_rule(name=name, class_target=self.sel_class.get(), zone_targets=zone_targets, zone_operator=zone_operator, condition_op=self.sel_op.get(), condition_val=val, actions=actions, cooldown=cooldown, persistence=persistence, severity=severity, save_evidence=save_evidence, validator_provider=validator_provider, validator_prompt=validator_prompt)
            self.entry_name.delete(0, 'end')
            self.entry_val.delete(0, 'end')
            self.entry_val_prompt.delete(0, 'end')
            self.sel_validator.set('Sin validacion')
            self._refresh_list()
        except ValueError:
            messagebox.showerror('Error', 'El valor debe ser un numero entero.')
        except Exception as e:
            log_error('EXE-GUI-COMP-02', f'Error al aniadir regla: {e}')
            messagebox.showerror('Error', f'No se pudo crear la regla: {e}')

    def _remove_rule(self, rule_id):
        if messagebox.askyesno('Confirmar', 'Eliminar esta regla?'):
            self.engine.remove_rule(rule_id)
            self._refresh_list()