import sqlite3
import os
from datetime import datetime
from .helpers import LOGS_DIR, log_error

DB_PATH = os.path.join(LOGS_DIR, "vision_analytics.db")

class DBManager:
    """Gestiona el registro persistente en SQLite para analítica histórica."""
    
    def __init__(self):
        self._init_db()

    def _init_db(self):
        try:
            os.makedirs(LOGS_DIR, exist_ok=True)
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            # Tabla de Detecciones
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    label TEXT,
                    track_id INTEGER,
                    conf REAL,
                    zones TEXT
                )
            ''')
            
            # Tabla de Eventos (Hitos)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    rule_name TEXT,
                    message TEXT,
                    evidence_path TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
        except Exception as e:
            log_error("EXE-UTL-DB-01", f"Error inicializando DB: {e}")

    def log_detections(self, detections):
        """Registra una ráfaga de detecciones."""
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            now = datetime.now().isoformat()
            
            data = []
            for d in detections:
                zones = ",".join(map(str, d.get("zone_indices", [])))
                data.append((now, d['label'], d.get('track_id'), d['conf'], zones))
            
            cursor.executemany('INSERT INTO detections (timestamp, label, track_id, conf, zones) VALUES (?, ?, ?, ?, ?)', data)
            conn.commit()
            conn.close()
        except Exception as e:
            log_error("EXE-UTL-DB-02", f"Error logueando detecciones: {e}")

    def log_event(self, rule_name, message, evidence_path=""):
        """Registra la activación de un hito."""
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute('INSERT INTO events (rule_name, message, evidence_path) VALUES (?, ?, ?)', 
                           (rule_name, message, evidence_path))
            conn.commit()
            conn.close()
        except Exception as e:
            log_error("EXE-UTL-DB-03", f"Error logueando evento: {e}")
