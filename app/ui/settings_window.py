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
        self.engine = event_engine
        self.detector = detector
        self.title("Ajustes de Sistema")
        self.geometry("500x780")
        self.grab_set()

        ctk.CTkLabel(self, text="AJUSTES GLOBALES", font=ctk.CTkFont(size=18, weight="bold")).pack(pady=(15, 5))

        # Contenedor scrollable para todo el contenido
        self.scroll = ctk.CTkScrollableFrame(self, fg_color="transparent")
        self.scroll.pack(fill="both", expand=True, padx=15, pady=5)

        # --- Seccion Webhook ---
        self._section_header(self.scroll, "WEBHOOK")
        wh_frame = ctk.CTkFrame(self.scroll)
        wh_frame.pack(fill="x", pady=(0, 10))

        ctk.CTkLabel(wh_frame, text="URL del Webhook:").pack(pady=2, padx=10, anchor="w")
        self.entry_wh = ctk.CTkEntry(wh_frame, placeholder_text="https://hooks.example.com/...")
        self.entry_wh.insert(0, self.engine.config.get("webhook_url", ""))
        self.entry_wh.pack(pady=(2, 10), padx=10, fill="x")

        # --- Seccion Telegram ---
        self._section_header(self.scroll, "TELEGRAM")
        tg_frame = ctk.CTkFrame(self.scroll)
        tg_frame.pack(fill="x", pady=(0, 10))

        ctk.CTkLabel(tg_frame, text="Bot Token:").pack(pady=2, padx=10, anchor="w")
        self.entry_tg = ctk.CTkEntry(tg_frame, show="*")
        self.entry_tg.insert(0, self.engine.config.get("telegram_token", ""))
        self.entry_tg.pack(pady=2, padx=10, fill="x")
        
        ctk.CTkLabel(tg_frame, text="Chat ID:").pack(pady=2, padx=10, anchor="w")
        self.entry_id = ctk.CTkEntry(tg_frame)
        self.entry_id.insert(0, self.engine.config.get("telegram_chat_id", ""))
        self.entry_id.pack(pady=(2, 10), padx=10, fill="x")

        # --- Seccion Proveedores de IA ---
        self._section_header(self.scroll, "PROVEEDORES DE IA (Validacion Secundaria)")

        # Ollama
        ollama_frame = ctk.CTkFrame(self.scroll)
        ollama_frame.pack(fill="x", pady=(0, 5))
        ctk.CTkLabel(ollama_frame, text="OLLAMA", font=ctk.CTkFont(size=11, weight="bold"), 
                     text_color="#a78bfa").pack(pady=(10, 2), padx=10, anchor="w")

        ctk.CTkLabel(ollama_frame, text="URL del Endpoint:").pack(pady=2, padx=10, anchor="w")
        self.entry_ollama_url = ctk.CTkEntry(ollama_frame, placeholder_text="http://localhost:11434")
        self.entry_ollama_url.insert(0, self.engine.config.get("ollama_url", ""))
        self.entry_ollama_url.pack(pady=2, padx=10, fill="x")

        ctk.CTkLabel(ollama_frame, text="Modelo:").pack(pady=2, padx=10, anchor="w")
        self.entry_ollama_model = ctk.CTkEntry(ollama_frame, placeholder_text="llava, moondream, bakllava...")
        self.entry_ollama_model.insert(0, self.engine.config.get("ollama_model", ""))
        self.entry_ollama_model.pack(pady=2, padx=10, fill="x")

        self.ollama_status = ctk.CTkLabel(ollama_frame, text="", font=ctk.CTkFont(size=10), text_color="#94a3b8")
        self.ollama_status.pack(pady=2, padx=10, anchor="w")

        ctk.CTkButton(ollama_frame, text="Test de Conexion", command=self._test_ollama,
                      fg_color="#334155", hover_color="#475569", height=28).pack(pady=(2, 10), padx=10, fill="x")

        # HuggingFace
        hf_frame = ctk.CTkFrame(self.scroll)
        hf_frame.pack(fill="x", pady=(0, 10))
        ctk.CTkLabel(hf_frame, text="HUGGINGFACE", font=ctk.CTkFont(size=11, weight="bold"), 
                     text_color="#fbbf24").pack(pady=(10, 2), padx=10, anchor="w")

        ctk.CTkLabel(hf_frame, text="API Key:").pack(pady=2, padx=10, anchor="w")
        self.entry_hf_key = ctk.CTkEntry(hf_frame, show="*", placeholder_text="hf_xxxxxxxxxxxx")
        self.entry_hf_key.insert(0, self.engine.config.get("huggingface_api_key", ""))
        self.entry_hf_key.pack(pady=2, padx=10, fill="x")

        ctk.CTkLabel(hf_frame, text="Modelo:").pack(pady=2, padx=10, anchor="w")
        self.entry_hf_model = ctk.CTkEntry(hf_frame, placeholder_text="Salesforce/blip-vqa-base")
        self.entry_hf_model.insert(0, self.engine.config.get("huggingface_model", ""))
        self.entry_hf_model.pack(pady=2, padx=10, fill="x")

        self.hf_status = ctk.CTkLabel(hf_frame, text="", font=ctk.CTkFont(size=10), text_color="#94a3b8")
        self.hf_status.pack(pady=2, padx=10, anchor="w")

        ctk.CTkButton(hf_frame, text="Test de Conexion", command=self._test_huggingface,
                      fg_color="#334155", hover_color="#475569", height=28).pack(pady=(2, 10), padx=10, fill="x")

        # --- Botones de Accion ---
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(fill="x", padx=15, pady=10)

        ctk.CTkButton(btn_frame, text="Guardar Ajustes", command=self._save,
                       fg_color="#16a34a", hover_color="#15803d").pack(fill="x", pady=(0, 8))
        
        ctk.CTkButton(btn_frame, text="Limpiar Credenciales", command=self._clear,
                       fg_color="#dc2626", hover_color="#b91c1c").pack(fill="x")

        # --- Info Hardware ---
        hw_frame = ctk.CTkFrame(self, fg_color="#1e293b")
        hw_frame.pack(fill="x", padx=15, pady=(5, 15))
        diag = detector.hardware_diag
        ctk.CTkLabel(hw_frame, text=f"GPU: {diag.get('gpu_name', 'N/A')}", font=ctk.CTkFont(size=11), text_color="#94a3b8").pack(pady=5, padx=10, anchor="w")
        ctk.CTkLabel(hw_frame, text=f"Backend: {diag.get('best_backend', 'cpu')}", font=ctk.CTkFont(size=11), text_color="#94a3b8").pack(pady=(0, 5), padx=10, anchor="w")

    def _section_header(self, parent, text):
        """Crea un encabezado de seccion consistente."""
        ctk.CTkLabel(parent, text=text, font=ctk.CTkFont(size=12, weight="bold"), 
                     text_color="#0ea5e9").pack(pady=(12, 4), anchor="w")

    def _save(self):
        try:
            wh_url = self.entry_wh.get().strip()
            tg_token = self.entry_tg.get().strip()
            tg_id = self.entry_id.get().strip()
            ollama_url = self.entry_ollama_url.get().strip()
            ollama_model = self.entry_ollama_model.get().strip()
            hf_key = self.entry_hf_key.get().strip()
            hf_model = self.entry_hf_model.get().strip()
            
            self.engine.update_config(
                webhook_url=wh_url,
                telegram_token=tg_token,
                telegram_chat_id=tg_id,
                ollama_url=ollama_url,
                ollama_model=ollama_model,
                huggingface_api_key=hf_key,
                huggingface_model=hf_model,
            )
            messagebox.showinfo("Correcto", "Configuracion guardada correctamente.")
            self.destroy()
        except Exception as e:
            log_error("EXE-GUI-COMP-02", f"Error guardando settings: {e}")

    def _clear(self):
        """Limpia todos los campos de credenciales."""
        if messagebox.askyesno("Confirmar", "Borrar todas las credenciales guardadas?"):
            self.entry_wh.delete(0, "end")
            self.entry_tg.delete(0, "end")
            self.entry_id.delete(0, "end")
            self.entry_ollama_url.delete(0, "end")
            self.entry_ollama_model.delete(0, "end")
            self.entry_hf_key.delete(0, "end")
            self.entry_hf_model.delete(0, "end")
            self.engine.update_config("", "", "", "", "", "", "")
            messagebox.showinfo("Correcto", "Credenciales eliminadas.")

    def _test_ollama(self):
        """Realiza un health check al endpoint de Ollama."""
        url = self.entry_ollama_url.get().strip()
        if not url:
            self.ollama_status.configure(text="Introduce una URL primero.", text_color="#ef4444")
            return

        self.ollama_status.configure(text="Conectando...", text_color="#fbbf24")

        def check():
            try:
                import requests
                resp = requests.get(f"{url.rstrip('/')}/api/tags", timeout=5)
                if resp.status_code == 200:
                    data = resp.json()
                    models = [m.get("name", "?") for m in data.get("models", [])]
                    count = len(models)
                    preview = ", ".join(models[:3])
                    if count > 3:
                        preview += f" (+{count - 3} mas)"
                    self.after(0, lambda: self.ollama_status.configure(
                        text=f"OK -- {count} modelo(s): {preview}", text_color="#22c55e"))
                else:
                    self.after(0, lambda: self.ollama_status.configure(
                        text=f"Error HTTP {resp.status_code}", text_color="#ef4444"))
            except Exception as e:
                self.after(0, lambda: self.ollama_status.configure(
                    text=f"Sin conexion: {str(e)[:50]}", text_color="#ef4444"))

        threading.Thread(target=check, daemon=True).start()

    def _test_huggingface(self):
        """Verifica la API Key de HuggingFace con un whoami."""
        key = self.entry_hf_key.get().strip()
        if not key:
            self.hf_status.configure(text="Introduce una API Key primero.", text_color="#ef4444")
            return

        self.hf_status.configure(text="Verificando...", text_color="#fbbf24")

        def check():
            try:
                import requests
                resp = requests.get("https://huggingface.co/api/whoami-v2",
                                    headers={"Authorization": f"Bearer {key}"}, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    username = data.get("name", "desconocido")
                    self.after(0, lambda: self.hf_status.configure(
                        text=f"OK -- Usuario: {username}", text_color="#22c55e"))
                elif resp.status_code == 401:
                    self.after(0, lambda: self.hf_status.configure(
                        text="API Key invalida (401 Unauthorized)", text_color="#ef4444"))
                else:
                    self.after(0, lambda: self.hf_status.configure(
                        text=f"Error HTTP {resp.status_code}", text_color="#ef4444"))
            except Exception as e:
                self.after(0, lambda: self.hf_status.configure(
                    text=f"Error de red: {str(e)[:50]}", text_color="#ef4444"))

        threading.Thread(target=check, daemon=True).start()
