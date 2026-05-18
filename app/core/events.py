"""
Motor de Inteligencia de Eventos.
Evalua reglas, gestiona notificaciones y carga secretos desde .env.
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
        # [Propósito]: Lista de reglas de eventos/alertas cargadas desde el archivo de configuración JSON.
        # [Tipo]: list[dict]
        self.rules = []

        # [Propósito]: Diccionario que almacena las credenciales de entorno y parámetros de APIs externas.
        # [Tipo]: dict[str, str]
        self.config = {
            "webhook_url": os.getenv("WEBHOOK_URL", ""),
            "telegram_token": os.getenv("TELEGRAM_TOKEN", ""),
            "telegram_chat_id": os.getenv("TELEGRAM_CHAT_ID", ""),
            "ollama_url": os.getenv("OLLAMA_URL", ""),
            "ollama_model": os.getenv("OLLAMA_MODEL", ""),
            "huggingface_api_key": os.getenv("HUGGINGFACE_API_KEY", ""),
            "huggingface_model": os.getenv("HUGGINGFACE_MODEL", ""),
        }

        # [Propósito]: Registro de marcas de tiempo del último disparo de cada regla para implementar control de cooldown.
        # [Tipo]: dict[str, float]
        self.last_triggered = {}

        # [Propósito]: Rastreo del momento inicial en que una condición de regla comenzó a cumplirse para el cálculo de persistencia.
        # [Tipo]: dict[str, float]
        self.active_conditions_start = {} # Para rastreo de persistencia temporal

        # [Propósito]: Estructura interna de datos acumulados sobre detecciones globales persistentes.
        # [Tipo]: dict[str, dict]
        self.cumulative_data = {}

        # [Propósito]: Mapeo de frecuencias e hitos para contadores acumulativos de eventos históricos.
        # [Tipo]: dict[str, int]
        self.cumulative_counts = {}

        # [Propósito]: Almacena el último mensaje de error ocurrido durante el envío de alertas externas o guardado de evidencias.
        # [Tipo]: str
        self.last_error = ""
        
        self.load_rules()
        self.load_stats()

        # [Propósito]: Gestor de la base de datos SQLite para registrar los hitos y detecciones de forma local.
        # [Tipo]: DBManager
        self.db = DBManager()

        # [Propósito]: Ruta absoluta del directorio local donde se guardan las capturas de imagen como evidencia.
        # [Tipo]: str
        self.evidence_dir = os.path.join(LOGS_DIR, "evidences")
        os.makedirs(self.evidence_dir, exist_ok=True)
        
    def load_rules(self):
        """Carga solo las reglas del JSON."""
        if os.path.exists(EVENTS_CONFIG):
            try:
                with open(EVENTS_CONFIG, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.rules = data.get("rules", [])
                    # Retrocompatibilidad
                    for r in self.rules:
                        if "persistence" not in r: r["persistence"] = 0
                        if "severity" not in r: r["severity"] = "Info"
                        if "actions" not in r: r["actions"] = [r.get("action", "log")]
                        if "save_evidence" not in r: r["save_evidence"] = False
                        if "zone_targets" not in r: r["zone_targets"] = [r.get("zone_target", -1)]
                        if "zone_operator" not in r: r["zone_operator"] = "OR"

                    # Solo actualizamos config si NO estan en env
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

    def _check_zone(self, detection, rule):
        targets = rule.get("zone_targets", [-1])
        if not targets or -1 in targets:
            return True
        operator = rule.get("zone_operator", "OR")
        det_zones = set(detection.get('zone_indices', []))
        target_zones = set(targets)
        
        if operator == "OR":
            return bool(det_zones.intersection(target_zones))
        elif operator == "AND":
            return target_zones.issubset(det_zones)
        elif operator == "NOT":
            return not bool(det_zones.intersection(target_zones))
        return True

    def evaluate(self, detections, frame=None, source="", app_log_callback=None, evidence_callback=None):
        """
        Evalua reglas en tiempo real sobre la rafaga de detecciones actual.
        - detections: Lista de objetos detectados en el frame actual.
        - frame: Imagen RAW para captura de evidencia.
        - app_log_callback: Funcion para registrar logs en la UI.
        """
        if not self.rules: return
        now = time.time()
        for rule in self.rules:
            # Respetar tiempo de espera entre alertas (cooldown)
            if now - self.last_triggered.get(rule['id'], 0) < rule['cooldown']:
                continue
                
            filtered = [d for d in detections if (rule['class_target'] == "Cualquiera" or d['label'] == rule['class_target']) and self._check_zone(d, rule)]
            count = len(filtered)
            
            triggered = False
            op, val = rule['condition_op'], rule['condition_val']
            if op == '>' and count > val: triggered = True
            elif op == '<' and count < val: triggered = True
            elif op == '==' and count == val: triggered = True
            elif op == '>=' and count >= val: triggered = True
            elif op == '<=' and count <= val: triggered = True
            elif op == 'Total >' and self.cumulative_counts.get(rule['class_target'], 0) > val: triggered = True
            
            if triggered:
                if rule['id'] not in self.active_conditions_start:
                    self.active_conditions_start[rule['id']] = now
                
                time_active = now - self.active_conditions_start[rule['id']]
                if time_active >= rule.get('persistence', 0):
                    self._trigger_action(rule, count, frame, source, app_log_callback, evidence_callback, filtered)
                    self.last_triggered[rule['id']] = now
            else:
                if rule['id'] in self.active_conditions_start:
                    del self.active_conditions_start[rule['id']]
            
        # Registro continuo en DB para analitica (debounced internamente por DBManager)
        if detections:
            self.db.log_detections(detections)

    def _trigger_action(self, rule, current_count, frame, source, app_log_callback, evidence_callback, detections=None):
        """Orquesta la respuesta al hito: logs, validacion IA y alertas externas."""
        targets = rule.get('zone_targets', [-1])
        if not targets or -1 in targets:
            zona_text = "Global"
        else:
            op_txt = rule.get("zone_operator", "OR")
            readable_targets = [str(t + 1) for t in targets]
            zona_text = f"Zonas [{','.join(readable_targets)}] ({op_txt})"
            
        custom_msg = rule.get("custom_message", "").strip()
        if custom_msg:
            # Reemplazar variables dinamicas si existen
            msg = custom_msg.replace("{count}", str(current_count)).replace("{class}", rule['class_target']).replace("{zone}", zona_text).replace("{name}", rule['name'])
        else:
            msg = f"Hito: '{rule['name']}' -> {current_count} {rule['class_target']} en {zona_text}."
        
        # Reportar al log del sistema (General)
        if app_log_callback:
            app_log_callback(f"[EVENTO] {rule['name']} activado.")
            # Reportar al log de HITOS (Separado)
            app_log_callback(msg, is_event=True)
        
        val_config = rule.get("validator", {})
        provider = val_config.get("provider", "None")
        
        if provider == "None":
            # Sin validacion secundaria: disparar alertas directamente
            self._send_external_alerts(rule, msg, frame, source, current_count, evidence_callback, detections)
        elif frame is not None:
            # Enriquecer val_config con los datos de conexion del engine
            enriched_config = {**val_config}
            enriched_config["ollama_url"] = self.config.get("ollama_url", "")
            enriched_config["ollama_model"] = self.config.get("ollama_model", "")
            enriched_config["huggingface_api_key"] = self.config.get("huggingface_api_key", "")
            enriched_config["huggingface_model"] = self.config.get("huggingface_model", "")
            enriched_config["class_target"] = rule.get("class_target", "Cualquiera")
            
            # Validacion via VLM (Asincrona)
            SecondaryValidator.validate_async(frame, enriched_config, rule["name"], app_log_callback, 
                                            lambda img, m, ok: self._send_external_alerts(rule, m, img, source, current_count, evidence_callback, detections) if ok else None)

    def _send_external_alerts(self, rule, msg, frame=None, source="", count=0, evidence_callback=None, detections=None):
        """Gestiona el envio de evidencias, logs a SQLite, alertas Telegram/Webhooks y TTS."""
        actions = rule.get('actions', [rule.get('action', 'log')])
        severity = rule.get('severity', 'Info')
        
        # 1. Guardar Evidencia Local
        evidence_path = ""
        raw_evidence_path = ""
        zoom_evidence_path = ""
        
        if rule.get('save_evidence', False) and frame is not None:
            ts = int(time.time())
            fname_ann = f"EV_{ts}_{severity}_{rule['name'].replace(' ', '_')}_ANN.jpg"
            fname_raw = f"EV_{ts}_{severity}_{rule['name'].replace(' ', '_')}_RAW.jpg"
            fname_zoom = f"EV_{ts}_{severity}_{rule['name'].replace(' ', '_')}_ZOOM.jpg"
            
            evidence_path = os.path.join(self.evidence_dir, fname_ann)
            raw_evidence_path = os.path.join(self.evidence_dir, fname_raw)
            zoom_evidence_path = os.path.join(self.evidence_dir, fname_zoom)
            
            try:
                # Guardar RAW
                cv2.imwrite(raw_evidence_path, frame)
                
                # Dibujar detecciones sobre la evidencia si existen
                save_frame_ann = frame.copy()
                if detections:
                    from ..utils.painter import VisualPainter
                    save_frame_ann = VisualPainter.draw_detections(save_frame_ann, detections, show_trails=False)
                    
                    # --- SMART ZOOM LOGIC ---
                    h, w = frame.shape[:2]
                    x_coords = [d['bbox'][0] for d in detections] + [d['bbox'][2] for d in detections]
                    y_coords = [d['bbox'][1] for d in detections] + [d['bbox'][3] for d in detections]
                    
                    x1, y1 = max(0, min(x_coords) - 50), max(0, min(y_coords) - 50)
                    x2, y2 = min(w, max(x_coords) + 50), min(h, max(y_coords) + 50)
                    
                    # Solo hacer zoom si el área resultante no es casi todo el frame (> 20% y < 80%)
                    area_ratio = ((x2-x1)*(y2-y1)) / (w*h)
                    if 0.01 < area_ratio < 0.7:
                        zoom_frame = frame[y1:y2, x1:x2].copy()
                        # También dibujar en el zoom para claridad
                        # Ajustar bboxes para el crop
                        zoom_dets = []
                        for d in detections:
                            zd = d.copy()
                            bx = d['bbox']
                            zd['bbox'] = (bx[0]-x1, bx[1]-y1, bx[2]-x1, bx[3]-y1)
                            zoom_dets.append(zd)
                        
                        zoom_frame = VisualPainter.draw_detections(zoom_frame, zoom_dets, show_trails=False)
                        cv2.imwrite(zoom_evidence_path, zoom_frame)
                    else:
                        zoom_evidence_path = "" # No merece la pena el zoom
                
                # Guardar ANNOTATED
                cv2.imwrite(evidence_path, save_frame_ann)
            except Exception:
                evidence_path = ""
                raw_evidence_path = ""
                zoom_evidence_path = ""
        
        # Notificar a la UI si se guardó evidencia y hay callback
        if evidence_path and evidence_callback:
            # Enviamos la anotada por defecto para la miniatura, pero permitimos acceso a ambas
            final_frame = save_frame_ann if 'save_frame_ann' in locals() else frame
            zoom_img = cv2.imread(zoom_evidence_path) if zoom_evidence_path and os.path.exists(zoom_evidence_path) else None
            evidence_callback(final_frame, msg, True, raw_frame=frame, zoom_frame=zoom_img)
        
        # 2. Registrar en SQLite (Persistencia Analitica)
        self.db.log_event(rule['name'], f"[{severity.upper()}] {msg}", evidence_path, raw_evidence_path, zoom_evidence_path)
        
        # 3. TTS (Sintesis de Voz)
        if "tts" in actions or "all" in actions:
            self._speak(msg)

        # 4. Webhook Enriquecido
        if ("webhook" in actions or "all" in actions) and self.config["webhook_url"]:
            payload = {
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                "rule_name": rule['name'],
                "severity": severity,
                "message": msg,
                "source": source,
                "class": rule['class_target'],
                "count": count,
                "zone_targets": rule.get('zone_targets', [-1])
            }
            threading.Thread(target=lambda: requests.post(self.config["webhook_url"], json=payload, timeout=10), daemon=True).start()
        
        # 5. Telegram (con foto si hay evidencia)
        if ("telegram" in actions or "all" in actions) and self.config["telegram_token"]:
            threading.Thread(target=self._send_to_telegram, args=(f"[{severity.upper()}] {msg}", evidence_path), daemon=True).start()

    def _speak(self, text):
        """Sintesis de voz ligera (opcional)."""
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

    def add_rule(self, name, class_target, zone_targets, zone_operator, condition_op, condition_val, 
                 actions, cooldown, persistence, severity, save_evidence,
                 validator_provider="None", validator_prompt="", custom_message=""):
        """Crea una nueva regla de evento y persiste."""
        rule = {
            "id": f"rule_{int(time.time() * 1000)}",
            "name": name,
            "class_target": class_target,
            "zone_targets": zone_targets,
            "zone_operator": zone_operator,
            "condition_op": condition_op,
            "condition_val": condition_val,
            "actions": actions,
            "cooldown": cooldown,
            "persistence": persistence,
            "severity": severity,
            "save_evidence": save_evidence,
            "validator": {
                "provider": validator_provider,
                "prompt": validator_prompt,
            },
            "custom_message": custom_message
        }
        self.rules.append(rule)
        self.save_rules()
        return rule

    def remove_rule(self, rule_id):
        """Elimina una regla por su ID y persiste."""
        self.rules = [r for r in self.rules if r["id"] != rule_id]
        self.save_rules()

    def update_config(self, webhook_url="", telegram_token="", telegram_chat_id="",
                      ollama_url="", ollama_model="", huggingface_api_key="", huggingface_model=""):
        """Actualiza las credenciales de notificacion y proveedores de IA, y persiste."""
        self.config["webhook_url"] = webhook_url
        self.config["telegram_token"] = telegram_token
        self.config["telegram_chat_id"] = telegram_chat_id
        self.config["ollama_url"] = ollama_url
        self.config["ollama_model"] = ollama_model
        self.config["huggingface_api_key"] = huggingface_api_key
        self.config["huggingface_model"] = huggingface_model
        self.save_rules()

    def test_webhook(self, url, severity="Info"):
        """Envía un webhook de prueba enriquecido para validación externa."""
        if not url: return False, "URL vacía"
        payload = {
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "rule_name": "TEST_SISTEMA",
            "severity": severity,
            "message": f"PRUEBA DE CONEXIÓN: Webhook configurado correctamente con gravedad '{severity}'.",
            "source": "VISION_APP_INTERNAL",
            "class": "test_signal",
            "count": 1,
            "zone_targets": [0]
        }
        try:
            resp = requests.post(url, json=payload, timeout=10)
            return resp.status_code == 200, resp.text
        except Exception as e:
            return False, str(e)

    def test_telegram(self, token, chat_id):
        """Envía un mensaje de prueba a Telegram."""
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        try:
            resp = requests.post(url, data={"chat_id": chat_id, "text": "PRUEBA DE CONEXIÓN: Tu bot de Vision App está configurado correctamente."}, timeout=10)
            return resp.status_code == 200, resp.text
        except Exception as e:
            return False, str(e)

    def test_tts(self, text="Prueba de síntesis de voz activada correctamente."):
        """Realiza una prueba de voz."""
        self._speak(text)
        return True, "Enviado a cola de voz"

    def test_vlm(self, provider, frame, class_name, log_callback, config_override=None):
        """Lanza una validación de prueba usando el proveedor VLM configurado (o override)."""
        if frame is None:
            return False, "No hay vídeo activo para capturar frame de prueba."
        
        config = self.config.copy()
        if config_override:
            config.update(config_override)
            
        config["provider"] = provider
        config["prompt"] = class_name
        
        if log_callback:
            log_callback(f"IA: Lanzando prueba de visión ({provider}) para clase '{class_name}'...")
        
        SecondaryValidator.validate_async(frame, config, "TEST_VISUAL", log_callback)
        return True, "Enviado"

    # --- Estadisticas Acumulativas ---

    def update_cumulative_stats(self, detections):
        """Actualiza conteos historicos basados en IDs de tracking unicos."""
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
