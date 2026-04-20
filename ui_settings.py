import customtkinter as ctk
from tkinter import messagebox
from ui_components import ModelExplorerWindow

class SettingsWindow(ctk.CTkToplevel):
    """Ventana de Configuración Global para la aplicación."""

    def __init__(self, parent, event_engine, detector):
        super().__init__(parent)
        self.engine = event_engine
        self.detector = detector
        self.title("⚙️ Ajustes de Sistema")
        
        win_w, win_h = 450, 480
        x = (self.winfo_screenwidth() // 2) - (win_w // 2)
        y = (self.winfo_screenheight() // 2) - (win_h // 2)
        self.geometry(f"{win_w}x{win_h}+{x}+{y}")
        self.resizable(False, False)
        self.grab_set()

        # Título
        ctk.CTkLabel(self, text="AJUSTES GLOBALES", font=ctk.CTkFont(size=18, weight="bold"), text_color="#38bdf8").pack(pady=(20, 10))
        
        # --- SECCIÓN: INTELIGENCIA ---
        self._section_label("GESTIÓN DE INTELIGENCIA")
        frame_intel = ctk.CTkFrame(self, fg_color="#1a1c1e")
        frame_intel.pack(padx=20, pady=10, fill="x")
        
        ctk.CTkButton(frame_intel, text="📁 Abrir Gestor de Modelos", 
                       fg_color="#1e293b", hover_color="#334155", text_color="#38bdf8",
                       command=lambda: ModelExplorerWindow(self, self.detector)).pack(pady=15, padx=20, fill="x")

        # --- SECCIÓN: NOTIFICACIONES ---
        self._section_label("CANALES DE NOTIFICACIÓN")
        
        frame_notif = ctk.CTkFrame(self, fg_color="#1a1c1e")
        frame_notif.pack(padx=20, pady=10, fill="x")
        
        # Webhook
        ctk.CTkLabel(frame_notif, text="Google Apps Script URL:", font=("", 11, "bold")).pack(pady=(10, 0), padx=10, anchor="w")
        self.entry_webhook = ctk.CTkEntry(frame_notif, placeholder_text="https://script.google.com/...", height=28)
        self.entry_webhook.insert(0, self.engine.config.get("webhook_url", ""))
        self.entry_webhook.pack(pady=(2, 10), padx=10, fill="x")
        
        # Telegram Row
        tg_row = ctk.CTkFrame(frame_notif, fg_color="transparent")
        tg_row.pack(fill="x", padx=5, pady=(0, 10))
        
        f1 = ctk.CTkFrame(tg_row, fg_color="transparent")
        f1.pack(side="left", fill="both", expand=True)
        ctk.CTkLabel(f1, text="Telegram Token:", font=("", 11, "bold")).pack(anchor="w", padx=5)
        self.entry_tg_token = ctk.CTkEntry(f1, placeholder_text="Token del Bot", show="*", height=28)
        self.entry_tg_token.insert(0, self.engine.config.get("telegram_token", ""))
        self.entry_tg_token.pack(pady=2, padx=5, fill="x")
        
        f2 = ctk.CTkFrame(tg_row, fg_color="transparent")
        f2.pack(side="left", fill="both")
        ctk.CTkLabel(f2, text="Chat ID:", font=("", 11, "bold")).pack(anchor="w", padx=5)
        
        f2_sub = ctk.CTkFrame(f2, fg_color="transparent")
        f2_sub.pack(fill="x")
        
        self.entry_tg_id = ctk.CTkEntry(f2_sub, placeholder_text="ID", width=120, height=28)
        self.entry_tg_id.insert(0, self.engine.config.get("telegram_chat_id", ""))
        self.entry_tg_id.pack(side="left", pady=2, padx=5)
        
        self.btn_test_tg = ctk.CTkButton(f2_sub, text="⚡ Prueba", width=60, height=28, 
                                         fg_color="#1e293b", hover_color="#334155", text_color="#38bdf8",
                                         command=self._test_telegram)
        self.btn_test_tg.pack(side="left", padx=2)
        
        lbl_help.pack(pady=(0, 5), padx=10, anchor="w")
        
        self.lbl_status = ctk.CTkLabel(frame_notif, text="Estado: Listo", font=("", 9), text_color="#94a3b8")
        self.lbl_status.pack(padx=10, anchor="w")
        
        # Mostrar último error si existe
        if self.engine.last_error:
            self.lbl_status.configure(text=f"Último error: {self.engine.last_error}", text_color="#f87171")

        # --- SECCIÓN: APARIENCIA ---
        self._section_label("APARIENCIA")
        frame_app = ctk.CTkFrame(self, fg_color="#1a1c1e")
        frame_app.pack(padx=20, pady=10, fill="x")
        
        mode_row = ctk.CTkFrame(frame_app, fg_color="transparent")
        mode_row.pack(fill="x", padx=10, pady=10)
        ctk.CTkLabel(mode_row, text="Modo de Color:", font=("", 11)).pack(side="left")
        
        self.appearance_mode = ctk.CTkSegmentedButton(mode_row, values=["System", "Dark", "Light"], 
                                                      command=lambda m: ctk.set_appearance_mode(m))
        self.appearance_mode.set(ctk.get_appearance_mode())
        self.appearance_mode.pack(side="right")

        # Botón Guardar
        ctk.CTkButton(self, text="💾 Guardar Cambios", font=ctk.CTkFont(weight="bold"), 
                       fg_color="#38bdf8", hover_color="#0ea5e9", text_color="#000",
                       command=self._save).pack(pady=20, padx=20, fill="x")

    def _section_label(self, text):
        lbl = ctk.CTkLabel(self, text=text, font=ctk.CTkFont(size=10, weight="bold"), text_color="#64748b")
        lbl.pack(padx=20, anchor="w")

    def _save(self):
        webhook = self.entry_webhook.get().strip()
        tg_token = self.entry_tg_token.get().strip()
        tg_id = self.entry_tg_id.get().strip()
        
        self.engine.update_config(webhook, tg_token, tg_id)
        messagebox.showinfo("Éxito", "La configuración se ha guardado correctamente.")
        self.destroy()

    def _test_telegram(self):
        """Envía un mensaje de prueba al bot usando los valores actuales."""
        token = self.entry_tg_token.get().strip()
        chat_id = self.entry_tg_id.get().strip()
        
        if not token or not chat_id:
            messagebox.showwarning("Aviso", "Introduce el Token y el Chat ID para probar.")
            return
            
        self.lbl_status.configure(text="Enviando prueba...", text_color="#38bdf8")
        self.update() # Forzar refresco UI

        # Actualizar temporalmente en el motor para la prueba
        self.engine.config["telegram_token"] = token
        self.engine.config["telegram_chat_id"] = chat_id
        
        import threading
        def run_test():
            try:
                self.engine._send_to_telegram("<b>Prueba de Conexión</b>\n¡Hola! Tu conexión desde Visión AI se ha establecido correctamente usando el modo HTML. 🚀")
                
                # Actualizar UI desde el hilo principal
                if self.engine.last_error:
                    self.after(0, lambda: self.lbl_status.configure(text=f"Error: {self.engine.last_error}", text_color="#f87171"))
                else:
                    self.after(0, lambda: self.lbl_status.configure(text="✅ Mensaje enviado con éxito", text_color="#4ade80"))
            except Exception as e:
                self.after(0, lambda: self.lbl_status.configure(text=f"Error Fatal: {str(e)}", text_color="#f87171"))

        threading.Thread(target=run_test, daemon=True).start()
