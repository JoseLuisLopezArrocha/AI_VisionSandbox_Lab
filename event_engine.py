import json
import os
import time
import threading
import requests
import cv2
from secondary_validator import SecondaryValidator

EVENTS_FILE = os.path.join(os.path.dirname(__file__), "events_config.json")
LOGS_FILE = os.path.join(os.path.dirname(__file__), "event_logs.txt")
STATS_FILE = os.path.join(os.path.dirname(__file__), "cumulative_stats.json")

class EventEngine:
    """
    Motor de Inteligencia de Eventos.
    Analiza en tiempo real las detecciones procesadas, evalúa reglas lógicas
    y dispara acciones automáticas (Telegram, Google Apps Script, Logs).
    """
    def __init__(self):
        self.rules = []
        self.config = {
            "webhook_url": "",
            "telegram_token": "",
            "telegram_chat_id": ""
        }
        self.last_triggered = {} # rule.id -> timestamp
        self.cumulative_data = {} # { class_name: set(track_ids) }
        self.cumulative_counts = {} # { class_name: total_unique_detected }
        self.last_error = "" # Almacena el último error de Telegram/Webhook
        
        self.load_rules()
        self.load_stats()
        
    def load_rules(self):
        """Carga las reglas y configuración global desde el archivo JSON."""
        if os.path.exists(EVENTS_FILE):
             try:
                 with open(EVENTS_FILE, 'r', encoding='utf-8') as f:
                     data = json.load(f)
                     if isinstance(data, dict):
                         self.rules = data.get("rules", [])
                         self.config.update(data.get("config", {}))
                     else:
                         self.rules = data # Fallback para versiones antiguas
                     
                     # MIGRACIÓN: Convertir 'email' a 'telegram' automáticamente
                     for rule in self.rules:
                         if rule.get("action") == "email":
                             rule["action"] = "telegram"
             except Exception as e:
                 print(f"[EventEngine] Error al cargar reglas: {e}")
                 self.rules = []
             
    def save_rules(self):
        """Persiste las reglas y la configuración en el JSON."""
        data = {
            "rules": self.rules,
            "config": self.config
        }
        with open(EVENTS_FILE, 'w', encoding='utf-8') as f:
             json.dump(data, f, indent=4, ensure_ascii=False)
             
    def update_config(self, webhook_url, telegram_token, telegram_chat_id):
        """Actualiza la configuración global de notificaciones."""
        self.config["webhook_url"] = webhook_url
        self.config["telegram_token"] = telegram_token
        self.config["telegram_chat_id"] = telegram_chat_id
        self.save_rules()

    def load_stats(self):
        """Carga los conteos históricos acumulados."""
        if os.path.exists(STATS_FILE):
            try:
                with open(STATS_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.cumulative_counts = data.get("counts", {})
                    # Reconstruir sets vacíos para la sesión actual
                    self.cumulative_data = {k: set() for k in self.cumulative_counts.keys()}
            except:
                self.cumulative_counts = {}

    def save_stats(self):
        """Persiste los conteos históricos en JSON."""
        try:
            with open(STATS_FILE, 'w', encoding='utf-8') as f:
                json.dump({"counts": self.cumulative_counts}, f, indent=4)
        except:
            pass

    def add_rule(self, name, class_target, zone_target, condition_op, condition_val, action_type, cooldown, validator_config=None, custom_msg=""):
        rule = {
            "id": str(int(time.time() * 1000)),
            "name": name,
            "class_target": class_target,
            "zone_target": zone_target,
            "condition_op": condition_op,
            "condition_val": int(condition_val),
            "action": action_type, # "log", "telegram", "webhook", "all"
            "cooldown": int(cooldown),
            "custom_msg": custom_msg,
            "validator": validator_config or {"provider": "None"}
        }
        self.rules.append(rule)
        self.save_rules()
        return rule

    def remove_rule(self, rule_id):
        self.rules = [r for r in self.rules if r["id"] != rule_id]
        if rule_id in self.last_triggered:
            del self.last_triggered[rule_id]
        self.save_rules()

    def evaluate(self, detections, frame=None, app_log_callback=None, evidence_callback=None):
        """Evalúa las reglas con las detecciones del frame actual."""
        if not self.rules:
            return

        now = time.time()
        for rule in self.rules:
            if now - self.last_triggered.get(rule['id'], 0) < rule['cooldown']:
                continue
                
            count = 0
            for d in detections:
                if rule['class_target'] != "Cualquiera" and d['label'] != rule['class_target']:
                    continue
                # Soporte multizona: comprobar si el objetivo está en la lista de zonas detectadas
                if rule['zone_target'] != -1 and rule['zone_target'] not in d.get('zone_indices', []):
                    continue
                count += 1
                
            triggered = False
            op = rule['condition_op']
            val = rule['condition_val']
            
            if op == '>' and count > val: triggered = True
            elif op == '<' and count < val: triggered = True
            elif op == '==' and count == val: triggered = True
            elif op == 'Total >' and self.cumulative_counts.get(rule['class_target'], 0) > val:
                triggered = True
            
            if triggered:
                self._trigger_action(rule, count, frame, app_log_callback, evidence_callback)
                self.last_triggered[rule['id']] = now

    def _trigger_action(self, rule, current_count, frame, app_log_callback, evidence_callback):
        """Ejecuta las acciones de notificación e inicia validación si aplica."""
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        zona_text = "Global" if rule['zone_target'] == -1 else f"Zona {rule['zone_target'] + 1}"
        
        # Formateo de mensaje (Estándar o Personalizado)
        custom_tpl = rule.get("custom_msg", "").strip()
        if custom_tpl:
            # Reemplazar etiquetas {etiqueta} por valores reales
            msg_text = custom_tpl.replace("{nombre}", rule["name"]) \
                                .replace("{conteo}", str(current_count)) \
                                .replace("{clase}", rule["class_target"]) \
                                .replace("{zona}", zona_text)
        else:
            # Mensaje por defecto (Fallback)
            msg_text = f"Hito: '{rule['name']}' -> {current_count} {rule['class_target']} en {zona_text}."
        
        # 1. LOG LOCAL (Inmediato)
        if app_log_callback:
            app_log_callback(f"🔔 ACTIVADO: {rule['name']}")

        # 2. ALERTA EXTERNA INMEDIATA (Solo si NO hay validación secundaria)
        val_config = rule.get("validator", {})
        if val_config.get("provider") == "None":
            self._send_external_alerts(rule, msg_text, payload={
                "timestamp": timestamp,
                "hito_nombre": rule['name'],
                "conteo": current_count,
                "clase": rule['class_target'],
                "zona": zona_text,
                "validacion": "Alerta instantánea (Sin validación IA)"
            })

        # 3. VALIDACIÓN IA SECUNDARIA (Asíncrona)
        if frame is not None and val_config.get("provider") != "None":
            val_config["target_class"] = rule["class_target"]
            
            def wrap_evidence(img, msg, ok):
                # Guardar evidencia en disco
                ev_dir = os.path.join(os.path.dirname(LOGS_FILE), "evidence")
                os.makedirs(ev_dir, exist_ok=True)
                fname = f"ev_{int(time.time())}_{rule['name'].replace(' ', '_')}.jpg"
                cv2.imwrite(os.path.join(ev_dir, fname), img)
                
                if evidence_callback:
                    icon = "✅" if ok else "❌"
                    evidence_callback(img, f"{icon} {rule['name']}", ok)
                
                if ok:
                    self._send_external_alerts(rule, msg, payload={
                        "timestamp": timestamp,
                        "hito_nombre": rule['name'],
                        "conteo": current_count,
                        "clase": rule['class_target'],
                        "zona": zona_text,
                        "validacion": msg
                    })

            SecondaryValidator.validate_async(frame, val_config, rule["name"], app_log_callback, wrap_evidence)

    def _send_external_alerts(self, rule, msg_text, payload):
        """Envía las alertas a Telegram/Google tras validación positiva."""
        if rule['action'] in ["webhook", "all"] and self.config["webhook_url"]:
            threading.Thread(target=self._send_to_google, args=(payload,), daemon=True).start()
            
        if rule['action'] in ["telegram", "all"] and self.config["telegram_token"]:
            threading.Thread(target=self._send_to_telegram, args=(msg_text,), daemon=True).start()

    def _send_to_google(self, payload):
        try:
            requests.post(self.config["webhook_url"], json=payload, timeout=10)
        except Exception as e:
            print(f"[EventEngine] Error enviando a Google: {e}")

    def _send_to_telegram(self, text):
        try:
            token = self.config.get("telegram_token", "").strip()
            chat_id = self.config.get("telegram_chat_id", "").strip()
            
            if not token or not chat_id:
                self.last_error = "Error: Falta Token o Chat ID."
                print(f"[EventEngine] {self.last_error}")
                return

            # Limpiar caracteres especiales para HTML (Telegram es estricto)
            clean_text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

            url = f"https://api.telegram.org/bot{token}/sendMessage"
            data = {
                "chat_id": chat_id,
                "text": f"<b>🚨 AVISO DE AI VISIONSANDBOX LAB 🚨</b>\n\n{clean_text}",
                "parse_mode": "HTML"
            }
            res = requests.post(url, data=data, timeout=10)
            if res.status_code != 200:
                self.last_error = f"API Telegram ({res.status_code}): {res.text}"
                print(f"[EventEngine] {self.last_error}")
            else:
                self.last_error = "" # Limpiar error si fue exitoso
                print("[EventEngine] Mensaje enviado a Telegram con éxito.")
        except Exception as e:
            self.last_error = f"Error de red: {str(e)}"
            print(f"[EventEngine] {self.last_error}")

    def update_cumulative_stats(self, detections):
        """Actualiza los conteos históricos basados en IDs de tracking únicos."""
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
            self.save_stats()
