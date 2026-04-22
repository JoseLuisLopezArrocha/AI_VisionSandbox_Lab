"""
Motor de Inteligencia de Eventos.
Evalúa reglas, gestiona notificaciones y carga secretos desde .env.
"""

import json
import os
import time
import threading
import requests
import cv2
from .validator import SecondaryValidator
from ..utils.helpers import EVENTS_CONFIG, LOGS_DIR, log_error
from ..utils.db_manager import DBManager

# Intentar cargar variables de entorno
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

STATS_FILE = os.path.join(os.path.dirname(EVENTS_CONFIG), "cumulative_stats.json")

class EventEngine:
    """
    Motor de Inteligencia de Eventos.
    """
    def __init__(self):
        self.rules = []
        # Cargar configuración desde entorno (.env) con fallback a vacío
        self.config = {
            "webhook_url": os.getenv("WEBHOOK_URL", ""),
            "telegram_token": os.getenv("TELEGRAM_TOKEN", ""),
            "telegram_chat_id": os.getenv("TELEGRAM_CHAT_ID", "")
        }
        self.last_triggered = {}
        self.cumulative_data = {}
        self.cumulative_counts = {}
        self.last_error = ""
        
        self.load_rules()
        self.load_stats()
        self.db = DBManager()
        self.evidence_dir = os.path.join(LOGS_DIR, "evidences")
        os.makedirs(self.evidence_dir, exist_ok=True)
        
    def load_rules(self):
        """Carga solo las reglas del JSON."""
        if os.path.exists(EVENTS_CONFIG):
            try:
                with open(EVENTS_CONFIG, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.rules = data.get("rules", [])
                    # Solo actualizamos config si NO están en env
                    file_config = data.get("config", {})
                    for k, v in file_config.items():
                        if not self.config.get(k):
                            self.config[k] = v
            except Exception as e:
                log_error("EXE-COR-EVT-05", f"Error cargando reglas: {e}")

    def save_rules(self):
        """Persiste las reglas."""
        try:
            data = {"rules": self.rules, "config": self.config}
            with open(EVENTS_CONFIG, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
        except Exception as e:
            log_error("EXE-COR-EVT-05", f"Error guardando reglas: {e}")

    def evaluate(self, detections, frame=None, app_log_callback=None, evidence_callback=None):
        """Evalúa reglas en tiempo real."""
        if not self.rules: return
        now = time.time()
        for rule in self.rules:
            if now - self.last_triggered.get(rule['id'], 0) < rule['cooldown']:
                continue
                
            count = sum(1 for d in detections if (rule['class_target'] == "Cualquiera" or d['label'] == rule['class_target']) and (rule['zone_target'] == -1 or rule['zone_target'] in d.get('zone_indices', [])))
            
            triggered = False
            op, val = rule['condition_op'], rule['condition_val']
            if op == '>' and count > val: triggered = True
            elif op == '<' and count < val: triggered = True
            elif op == '==' and count == val: triggered = True
            elif op == 'Total >' and self.cumulative_counts.get(rule['class_target'], 0) > val: triggered = True
            
            if triggered:
                self._trigger_action(rule, count, frame, app_log_callback, evidence_callback)
                self.last_triggered[rule['id']] = now
            
        # Registro continuo en DB para analítica (cada frame con detecciones)
        if detections:
            self.db.log_detections(detections)

    def _trigger_action(self, rule, current_count, frame, app_log_callback, evidence_callback):
        """Dispara acciones y validación."""
        zona_text = "Global" if rule['zone_target'] == -1 else f"Zona {rule['zone_target'] + 1}"
        msg = f"Hito: '{rule['name']}' -> {current_count} {rule['class_target']} en {zona_text}."
        
        if app_log_callback: app_log_callback(f"🔔 ACTIVADO: {rule['name']}")
        
        val_config = rule.get("validator", {})
        if val_config.get("provider") == "None":
            self._send_external_alerts(rule, msg, frame)
        elif frame is not None:
             SecondaryValidator.validate_async(frame, val_config, rule["name"], app_log_callback, 
                                            lambda img, m, ok: self._send_external_alerts(rule, m, img) if ok else None)

    def _send_external_alerts(self, rule, msg, frame=None):
        # 1. Guardar Evidencia Local
        evidence_path = ""
        if frame is not None:
            fname = f"EV_{int(time.time())}_{rule['name'].replace(' ', '_')}.jpg"
            evidence_path = os.path.join(self.evidence_dir, fname)
            cv2.imwrite(evidence_path, frame)
        
        # 2. Registrar en SQLite
        self.db.log_event(rule['name'], msg, evidence_path)
        
        # 3. TTS (Síntesis de Voz)
        self._speak(msg)

        # 4. Notificaciones Externas
        if rule['action'] in ["webhook", "all"] and self.config["webhook_url"]:
            threading.Thread(target=lambda: requests.post(self.config["webhook_url"], json={"msg": msg}), daemon=True).start()
        if rule['action'] in ["telegram", "all"] and self.config["telegram_token"]:
            threading.Thread(target=self._send_to_telegram, args=(msg, evidence_path), daemon=True).start()

    def _speak(self, text):
        """Síntesis de voz ligera (opcional)."""
        def run():
            try:
                import pyttsx3
                engine = pyttsx3.init()
                engine.say(text)
                engine.runAndWait()
            except: pass
        threading.Thread(target=run, daemon=True).start()

    def _send_to_telegram(self, text, photo_path=""):
        try:
            token, chat_id = self.config["telegram_token"], self.config["telegram_chat_id"]
            if not token or not chat_id: return
            
            if photo_path and os.path.exists(photo_path):
                url = f"https://api.telegram.org/bot{token}/sendPhoto"
                with open(photo_path, 'rb') as f:
                    requests.post(url, data={"chat_id": chat_id, "caption": text}, files={"photo": f}, timeout=15)
            else:
                url = f"https://api.telegram.org/bot{token}/sendMessage"
                requests.post(url, data={"chat_id": chat_id, "text": text}, timeout=10)
        except Exception as e:
            log_error("EXE-COR-EVT-05", f"Error Telegram: {e}")

    # --- API para EventsWindow / SettingsWindow ---

    def add_rule(self, name, class_target, zone_target, condition_op, condition_val, action_type, cooldown):
        """Crea una nueva regla de evento y persiste."""
        rule = {
            "id": f"rule_{int(time.time() * 1000)}",
            "name": name,
            "class_target": class_target,
            "zone_target": zone_target,
            "condition_op": condition_op,
            "condition_val": condition_val,
            "action": action_type,
            "cooldown": cooldown,
            "validator": {"provider": "None"}
        }
        self.rules.append(rule)
        self.save_rules()
        return rule

    def remove_rule(self, rule_id):
        """Elimina una regla por su ID y persiste."""
        self.rules = [r for r in self.rules if r["id"] != rule_id]
        self.save_rules()

    def update_config(self, webhook_url, telegram_token, telegram_chat_id):
        """Actualiza las credenciales de notificación y persiste."""
        self.config["webhook_url"] = webhook_url
        self.config["telegram_token"] = telegram_token
        self.config["telegram_chat_id"] = telegram_chat_id
        self.save_rules()

    # --- Estadísticas Acumulativas ---

    def update_cumulative_stats(self, detections):
        """Actualiza conteos históricos basados en IDs de tracking únicos."""
        changed = False
        for d in detections:
            label = d['label']
            t_id = d.get('track_id')
            if t_id is not None:
                if label not in self.cumulative_data:
                    self.cumulative_data[label] = set()
                    self.cumulative_counts[label] = self.cumulative_counts.get(label, 0)
                
                if t_id not in self.cumulative_data[label]:
                    self.cumulative_data[label].add(t_id)
                    self.cumulative_counts[label] += 1
                    changed = True
        if changed:
            # Debounce: Solo escribir a disco cada 5 segundos
            now = time.time()
            if now - getattr(self, '_last_stats_save', 0) >= 5.0:
                self.save_stats()
                self._last_stats_save = now

    def load_stats(self):
        if os.path.exists(STATS_FILE):
            try:
                with open(STATS_FILE, 'r', encoding='utf-8') as f:
                    self.cumulative_counts = json.load(f).get("counts", {})
            except Exception:
                pass

    def save_stats(self):
        try:
            with open(STATS_FILE, 'w', encoding='utf-8') as f:
                json.dump({"counts": self.cumulative_counts}, f, indent=4)
        except Exception:
            pass
