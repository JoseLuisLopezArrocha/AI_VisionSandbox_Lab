"""
Ventana de Configuración Global.
Gestiona secretos de notificaciones (Telegram/Webhooks) y ajustes visuales.
"""

import customtkinter as ctk
from tkinter import messagebox
from ..utils.error_handler import log_error

class SettingsWindow(ctk.CTkToplevel):
    """
    Ventana de Configuración Global para la aplicación.
    """

    def __init__(self, parent, event_engine, detector):
        super().__init__(parent)
        self.engine = event_engine
        self.detector = detector
        self.title("⚙️ Ajustes de Sistema")
        self.geometry("450x600")
        self.grab_set()

        ctk.CTkLabel(self, text="AJUSTES GLOBALES", font=ctk.CTkFont(size=18, weight="bold")).pack(pady=20)

        # --- Sección Webhook ---
        wh_frame = ctk.CTkFrame(self)
        wh_frame.pack(fill="x", padx=20, pady=(0, 10))
        ctk.CTkLabel(wh_frame, text="WEBHOOK", font=ctk.CTkFont(size=12, weight="bold"), text_color="#0ea5e9").pack(pady=(10, 2), padx=10, anchor="w")

        ctk.CTkLabel(wh_frame, text="URL del Webhook:").pack(pady=2, padx=10, anchor="w")
        self.entry_wh = ctk.CTkEntry(wh_frame, placeholder_text="https://hooks.example.com/...")
        self.entry_wh.insert(0, self.engine.config.get("webhook_url", ""))
        self.entry_wh.pack(pady=(2, 10), padx=10, fill="x")

        # --- Sección Telegram ---
        tg_frame = ctk.CTkFrame(self)
        tg_frame.pack(fill="x", padx=20, pady=10)
        ctk.CTkLabel(tg_frame, text="TELEGRAM", font=ctk.CTkFont(size=12, weight="bold"), text_color="#0ea5e9").pack(pady=(10, 2), padx=10, anchor="w")

        ctk.CTkLabel(tg_frame, text="Bot Token:").pack(pady=2, padx=10, anchor="w")
        self.entry_tg = ctk.CTkEntry(tg_frame, show="*")
        self.entry_tg.insert(0, self.engine.config.get("telegram_token", ""))
        self.entry_tg.pack(pady=2, padx=10, fill="x")
        
        ctk.CTkLabel(tg_frame, text="Chat ID:").pack(pady=2, padx=10, anchor="w")
        self.entry_id = ctk.CTkEntry(tg_frame)
        self.entry_id.insert(0, self.engine.config.get("telegram_chat_id", ""))
        self.entry_id.pack(pady=(2, 10), padx=10, fill="x")

        # --- Botones de Acción ---
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(fill="x", padx=20, pady=20)

        ctk.CTkButton(btn_frame, text="💾 Guardar Ajustes", command=self._save,
                       fg_color="#16a34a", hover_color="#15803d").pack(fill="x", pady=(0, 8))
        
        ctk.CTkButton(btn_frame, text="🗑 Limpiar Credenciales", command=self._clear,
                       fg_color="#dc2626", hover_color="#b91c1c").pack(fill="x")

        # --- Info Hardware ---
        hw_frame = ctk.CTkFrame(self, fg_color="#1e293b")
        hw_frame.pack(fill="x", padx=20, pady=(10, 20))
        diag = detector.hardware_diag
        ctk.CTkLabel(hw_frame, text=f"GPU: {diag.get('gpu_name', 'N/A')}", font=ctk.CTkFont(size=11), text_color="#94a3b8").pack(pady=5, padx=10, anchor="w")
        ctk.CTkLabel(hw_frame, text=f"Backend: {diag.get('best_backend', 'cpu')}", font=ctk.CTkFont(size=11), text_color="#94a3b8").pack(pady=(0, 5), padx=10, anchor="w")

    def _save(self):
        try:
            wh_url = self.entry_wh.get().strip()
            tg_token = self.entry_tg.get().strip()
            tg_id = self.entry_id.get().strip()
            
            self.engine.update_config(wh_url, tg_token, tg_id)
            messagebox.showinfo("Éxito", "Configuración guardada correctamente.")
            self.destroy()
        except Exception as e:
            log_error("EXE-GUI-COMP-02", f"Error guardando settings: {e}")

    def _clear(self):
        """Limpia todos los campos de credenciales."""
        if messagebox.askyesno("Confirmar", "¿Borrar todas las credenciales guardadas?"):
            self.entry_wh.delete(0, "end")
            self.entry_tg.delete(0, "end")
            self.entry_id.delete(0, "end")
            self.engine.update_config("", "", "")
            messagebox.showinfo("Éxito", "Credenciales eliminadas.")
