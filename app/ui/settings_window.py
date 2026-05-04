"""
Ventana de Configuracion Global.
Gestiona secretos de notificaciones (Telegram/Webhooks), proveedores de IA
(Ollama/HuggingFace) y diagnostico de hardware.
"""
import customtkinter as ctk
import threading
from tkinter import messagebox
from ..utils.error_handler import log_error

class SettingsWindow(ctk.CTkToplevel):
    """
    Ventana de Configuracion Global para la aplicacion.
    Incluye secciones para notificaciones, proveedores de IA y hardware.
    """

    def __init__(self, parent, event_engine, detector):
        super().__init__(parent)
        self.parent = parent
        self.engine = event_engine
        self.detector = detector
        self.title('Ajustes de Sistema')
        self.geometry('500x780')
        self.grab_set()
        ctk.CTkLabel(self, text='CONTROL DE SISTEMA CORE', font=ctk.CTkFont(size=18, weight='bold'), text_color='#0ea5e9', corner_radius=0).pack(pady=(15, 5))
        self.scroll = ctk.CTkScrollableFrame(self, fg_color='transparent', corner_radius=0)
        self.scroll.pack(fill='both', expand=True, padx=15, pady=5)
        self._section_header(self.scroll, 'WEBHOOK')
        wh_frame = ctk.CTkFrame(self.scroll, corner_radius=0)
        wh_frame.pack(fill='x', pady=(0, 10))
        ctk.CTkLabel(wh_frame, text='URL del Webhook:', corner_radius=0).pack(pady=2, padx=10, anchor='w')
        self.entry_wh = ctk.CTkEntry(wh_frame, placeholder_text='https://hooks.example.com/...', corner_radius=0)
        self.entry_wh.insert(0, self.engine.config.get('webhook_url', ''))
        self.entry_wh.pack(pady=(2, 10), padx=10, fill='x')
        
        wh_test_row = ctk.CTkFrame(wh_frame, fg_color='transparent', corner_radius=0)
        wh_test_row.pack(fill='x', padx=10, pady=(0, 10))
        self.wh_sev_var = ctk.StringVar(value="Info")
        self.wh_sev_menu = ctk.CTkOptionMenu(wh_test_row, values=["Info", "Alert", "Critical"], variable=self.wh_sev_var, width=100, height=28, corner_radius=0)
        self.wh_sev_menu.pack(side='left', padx=(0, 10))
        ctk.CTkButton(wh_test_row, text='PROBAR WEBHOOK', command=self._test_webhook, fg_color='#334155', hover_color='#475569', font=ctk.CTkFont(size=10, weight='bold'), height=28, corner_radius=0).pack(side='left', fill='x', expand=True)
        self._section_header(self.scroll, 'TELEGRAM')
        tg_frame = ctk.CTkFrame(self.scroll, corner_radius=0)
        tg_frame.pack(fill='x', pady=(0, 10))
        ctk.CTkLabel(tg_frame, text='Bot Token:', corner_radius=0).pack(pady=2, padx=10, anchor='w')
        self.entry_tg = ctk.CTkEntry(tg_frame, show='*', corner_radius=0)
        self.entry_tg.insert(0, self.engine.config.get('telegram_token', ''))
        self.entry_tg.pack(pady=2, padx=10, fill='x')
        self.entry_id = ctk.CTkEntry(tg_frame, corner_radius=0)
        self.entry_id.insert(0, self.engine.config.get('telegram_chat_id', ''))
        self.entry_id.pack(pady=2, padx=10, fill='x')
        ctk.CTkButton(tg_frame, text='PROBAR TELEGRAM', command=self._test_telegram, fg_color='#334155', hover_color='#475569', font=ctk.CTkFont(size=10, weight='bold'), height=28, corner_radius=0).pack(pady=(2, 10), padx=10, fill='x')
        self._section_header(self.scroll, 'PROVEEDORES DE IA (Validacion Secundaria)')
        test_vlm_frame = ctk.CTkFrame(self.scroll, fg_color='#1e293b', corner_radius=0)
        test_vlm_frame.pack(fill='x', pady=(0, 10))
        ctk.CTkLabel(test_vlm_frame, text='Clase para Test de Visión:', font=ctk.CTkFont(size=10), corner_radius=0).pack(side='left', padx=10, pady=5)
        self.entry_vlm_test = ctk.CTkEntry(test_vlm_frame, placeholder_text='ej: person', width=120, height=24, corner_radius=0)
        self.entry_vlm_test.insert(0, 'person')
        self.entry_vlm_test.pack(side='left', padx=5, pady=5)
        ollama_frame = ctk.CTkFrame(self.scroll, corner_radius=0)
        ollama_frame.pack(fill='x', pady=(0, 5))
        ctk.CTkLabel(ollama_frame, text='OLLAMA', font=ctk.CTkFont(size=11, weight='bold'), text_color='#a78bfa', corner_radius=0).pack(pady=(10, 2), padx=10, anchor='w')
        ctk.CTkLabel(ollama_frame, text='URL del Endpoint:', corner_radius=0).pack(pady=2, padx=10, anchor='w')
        self.entry_ollama_url = ctk.CTkEntry(ollama_frame, placeholder_text='http://localhost:11434', corner_radius=0)
        self.entry_ollama_url.insert(0, self.engine.config.get('ollama_url', ''))
        self.entry_ollama_url.pack(pady=2, padx=10, fill='x')
        self.entry_ollama_url.bind('<FocusOut>', lambda _: self._test_ollama())
        self.entry_ollama_url.bind('<Return>', lambda _: self._test_ollama())
        ctk.CTkLabel(ollama_frame, text='Modelo (Vision):', corner_radius=0).pack(pady=2, padx=10, anchor='w')
        self.entry_ollama_model = ctk.CTkComboBox(ollama_frame, values=['llava', 'moondream', 'bakllava'], corner_radius=0)
        self.entry_ollama_model.set(self.engine.config.get('ollama_model', 'llava'))
        self.entry_ollama_model.pack(pady=2, padx=10, fill='x')
        self.ollama_status = ctk.CTkLabel(ollama_frame, text='Pulsa Enter o sal de la URL para cargar modelos.', font=ctk.CTkFont(size=10), text_color='#64748b', corner_radius=0)
        self.ollama_status.pack(pady=2, padx=10, anchor='w')
        ollama_btns = ctk.CTkFrame(ollama_frame, fg_color='transparent', corner_radius=0)
        ollama_btns.pack(fill='x', padx=10, pady=(2, 10))
        ctk.CTkButton(ollama_btns, text='\ue72c MODELOS', command=self._test_ollama, fg_color='#334155', hover_color='#475569', font=ctk.CTkFont(size=9, weight='bold'), height=28, corner_radius=0).pack(side='left', fill='x', expand=True, padx=(0, 5))
        ctk.CTkButton(ollama_btns, text='\ue7b3 TEST VISIÓN', command=lambda: self._test_vlm('ollama'), fg_color='#334155', hover_color='#475569', font=ctk.CTkFont(size=9, weight='bold'), height=28, corner_radius=0).pack(side='left', fill='x', expand=True)
        hf_frame = ctk.CTkFrame(self.scroll, corner_radius=0)
        hf_frame.pack(fill='x', pady=(0, 10))
        ctk.CTkLabel(hf_frame, text='HUGGINGFACE', font=ctk.CTkFont(size=11, weight='bold'), text_color='#fbbf24', corner_radius=0).pack(pady=(10, 2), padx=10, anchor='w')
        ctk.CTkLabel(hf_frame, text='API Key:', corner_radius=0).pack(pady=2, padx=10, anchor='w')
        self.entry_hf_key = ctk.CTkEntry(hf_frame, show='*', placeholder_text='hf_xxxxxxxxxxxx', corner_radius=0)
        self.entry_hf_key.insert(0, self.engine.config.get('huggingface_api_key', ''))
        self.entry_hf_key.pack(pady=2, padx=10, fill='x')
        ctk.CTkLabel(hf_frame, text='Modelo:', corner_radius=0).pack(pady=2, padx=10, anchor='w')
        self.entry_hf_model = ctk.CTkEntry(hf_frame, placeholder_text='Salesforce/blip-vqa-base', corner_radius=0)
        self.entry_hf_model.insert(0, self.engine.config.get('huggingface_model', ''))
        self.entry_hf_model.pack(pady=2, padx=10, fill='x')
        self.hf_status = ctk.CTkLabel(hf_frame, text='', font=ctk.CTkFont(size=10), text_color='#94a3b8', corner_radius=0)
        self.hf_status.pack(pady=2, padx=10, anchor='w')
        hf_btns = ctk.CTkFrame(hf_frame, fg_color='transparent', corner_radius=0)
        hf_btns.pack(fill='x', padx=10, pady=(2, 10))
        ctk.CTkButton(hf_btns, text='CONEXIÓN', command=self._test_huggingface, fg_color='#334155', hover_color='#475569', font=ctk.CTkFont(size=9, weight='bold'), height=28, corner_radius=0).pack(side='left', fill='x', expand=True, padx=(0, 5))
        ctk.CTkButton(hf_btns, text='TEST VISIÓN', command=lambda: self._test_vlm('huggingface'), fg_color='#334155', hover_color='#475569', font=ctk.CTkFont(size=9, weight='bold'), height=28, corner_radius=0).pack(side='left', fill='x', expand=True)
        btn_frame = ctk.CTkFrame(self, fg_color='transparent', corner_radius=0)
        btn_frame.pack(fill='x', padx=15, pady=10)
        ctk.CTkButton(btn_frame, text='GUARDAR CONFIGURACIÓN', command=self._save, fg_color='#10b981', hover_color='#059669', font=ctk.CTkFont(size=12, weight='bold'), corner_radius=0).pack(fill='x', pady=0)
        ctk.CTkButton(btn_frame, text='PURGAR CREDENCIALES', command=self._clear, fg_color='#450a0a', hover_color='#7f1d1d', font=ctk.CTkFont(size=11, weight='bold'), corner_radius=0).pack(fill='x', pady=0)
        hw_frame = ctk.CTkFrame(self, fg_color='#1e293b', corner_radius=0)
        hw_frame.pack(fill='x', padx=15, pady=(5, 15))
        diag = detector.hardware_diag
        ctk.CTkLabel(hw_frame, text=f"GPU: {diag.get('gpu_name', 'N/A')}", font=ctk.CTkFont(size=11), text_color='#94a3b8', corner_radius=0).pack(pady=5, padx=10, anchor='w')
        ctk.CTkLabel(hw_frame, text=f"Backend: {diag.get('best_backend', 'cpu')}", font=ctk.CTkFont(size=11), text_color='#94a3b8', corner_radius=0).pack(pady=(0, 5), padx=10, anchor='w')
        if self.entry_ollama_url.get():
            self.after(500, self._test_ollama)

    def _section_header(self, parent, text):
        """Crea un encabezado de seccion consistente."""
        ctk.CTkLabel(parent, text=text, font=ctk.CTkFont(size=12, weight='bold'), text_color='#0ea5e9', corner_radius=0).pack(pady=(12, 4), anchor='w')

    def _save(self):
        try:
            wh_url = self.entry_wh.get().strip()
            tg_token = self.entry_tg.get().strip()
            tg_id = self.entry_id.get().strip()
            ollama_url = self.entry_ollama_url.get().strip()
            ollama_model = self.entry_ollama_model.get().strip()
            hf_key = self.entry_hf_key.get().strip()
            hf_model = self.entry_hf_model.get().strip()
            self.engine.update_config(webhook_url=wh_url, telegram_token=tg_token, telegram_chat_id=tg_id, ollama_url=ollama_url, ollama_model=ollama_model, huggingface_api_key=hf_key, huggingface_model=hf_model)
            if hasattr(self.parent, 'add_log'):
                self.parent.add_log('Ajustes: Configuración guardada correctamente.')
            self.destroy()
        except Exception as e:
            log_error('EXE-GUI-COMP-02', f'Error guardando settings: {e}')

    def _clear(self):
        """Limpia todos los campos de credenciales."""
        if messagebox.askyesno('Confirmar', 'Borrar todas las credenciales guardadas?'):
            self.entry_wh.delete(0, 'end')
            self.entry_tg.delete(0, 'end')
            self.entry_id.delete(0, 'end')
            self.entry_ollama_url.delete(0, 'end')
            self.entry_ollama_model.delete(0, 'end')
            self.entry_hf_key.delete(0, 'end')
            self.entry_hf_model.delete(0, 'end')
            self.engine.update_config('', '', '', '', '', '', '')
            if hasattr(self.parent, 'add_log'):
                self.parent.add_log('Ajustes: Credenciales eliminadas por el usuario.')

    def _test_ollama(self):
        """Realiza un health check al endpoint de Ollama y descarga la lista de modelos filtrada."""
        from ..core.ollama_helper import get_ollama_models_with_vision
        url = self.entry_ollama_url.get().strip()
        if not url:
            self.ollama_status.configure(text='Introduce una URL primero.', text_color='#ef4444')
            return
        self.ollama_status.configure(text='Consultando capacidades del servidor...', text_color='#fbbf24')

        def check():
            models, vision_count, error = get_ollama_models_with_vision(url)

            def _update_ui():
                if error:
                    self.ollama_status.configure(text=f'Error: {error[:40]}...', text_color='#ef4444')
                elif models:
                    self.entry_ollama_model.configure(values=models)
                    current = self.entry_ollama_model.get()
                    if current not in models and vision_count > 0:
                        self.entry_ollama_model.set(models[0])
                    self.ollama_status.configure(text=f'Conectado. {vision_count} modelos de vision confirmados.', text_color='#22c55e')
                else:
                    self.ollama_status.configure(text='Conectado, pero no hay modelos instalados.', text_color='#fbbf24')
            self.after(0, _update_ui)
        threading.Thread(target=check, daemon=True).start()

    def _test_webhook(self):
        """Envía una señal de prueba al webhook configurado y notifica vía log."""
        url = self.entry_wh.get().strip()
        sev = self.wh_sev_var.get()
        if not url:
            if hasattr(self.parent, 'add_log'):
                self.parent.add_log('Webhooks: Error - No hay URL configurada.')
            return
        
        if hasattr(self.parent, 'add_log'):
            self.parent.add_log(f'Webhooks: Enviando señal de prueba ({sev})...')
        
        def run_test():
            success, detail = self.engine.test_webhook(url, sev)
            if hasattr(self.parent, 'add_log'):
                if success:
                    self.parent.add_log(f'Webhooks: TEST EXITOSO (HTTP 200). Gravedad: {sev}')
                else:
                    self.parent.add_log(f'Webhooks: ERROR en el envío -> {detail[:60]}...')
        
        threading.Thread(target=run_test, daemon=True).start()

    def _test_telegram(self):
        """Envía un mensaje de prueba a Telegram."""
        token = self.entry_tg.get().strip()
        chat_id = self.entry_id.get().strip()
        if not token or not chat_id:
            if hasattr(self.parent, 'add_log'): self.parent.add_log('Telegram: Error - Faltan credenciales.')
            return
        if hasattr(self.parent, 'add_log'): self.parent.add_log('Telegram: Enviando mensaje de prueba...')
        def run():
            ok, detail = self.engine.test_telegram(token, chat_id)
            if hasattr(self.parent, 'add_log'):
                if ok: self.parent.add_log('Telegram: TEST EXITOSO (Mensaje enviado).')
                else: self.parent.add_log(f'Telegram: ERROR -> {detail[:60]}...')
        threading.Thread(target=run, daemon=True).start()

    def _test_vlm(self, provider):
        """Lanza una validación asíncrona usando los valores actuales de la UI."""
        class_name = self.entry_vlm_test.get().strip()
        frame = getattr(self.parent, 'raw_frame', None)
        
        # Capturar valores actuales de la UI para probar sin necesidad de guardar
        config_override = {}
        if provider == 'ollama':
            config_override = {
                "ollama_url": self.entry_ollama_url.get().strip(),
                "ollama_model": self.entry_ollama_model.get().strip()
            }
        elif provider == 'huggingface':
            config_override = {
                "huggingface_api_key": self.entry_hf_key.get().strip(),
                "huggingface_model": self.entry_hf_model.get().strip()
            }

        if hasattr(self.parent, 'add_log'):
            ok, msg = self.engine.test_vlm(provider, frame, class_name, self.parent.add_log, config_override)
            if not ok: self.parent.add_log(f'IA: Error de inicio -> {msg}')

    def _test_huggingface(self):
        """Verifica la API Key de HuggingFace con un whoami."""
        key = self.entry_hf_key.get().strip()
        if not key:
            self.hf_status.configure(text='Introduce una API Key primero.', text_color='#ef4444')
            return
        self.hf_status.configure(text='Verificando...', text_color='#fbbf24')

        def check():
            try:
                import requests
                resp = requests.get('https://huggingface.co/api/whoami-v2', headers={'Authorization': f'Bearer {key}'}, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    username = data.get('name', 'desconocido')
                    self.after(0, lambda: self.hf_status.configure(text=f'OK -- Usuario: {username}', text_color='#22c55e'))
                elif resp.status_code == 401:
                    self.after(0, lambda: self.hf_status.configure(text='API Key invalida (401 Unauthorized)', text_color='#ef4444'))
                else:
                    self.after(0, lambda: self.hf_status.configure(text=f'Error HTTP {resp.status_code}', text_color='#ef4444'))
            except Exception as e:
                self.after(0, lambda: self.hf_status.configure(text=f'Error de red: {str(e)[:50]}', text_color='#ef4444'))
        threading.Thread(target=check, daemon=True).start()