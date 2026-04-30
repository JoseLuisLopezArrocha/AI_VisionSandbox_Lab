"""
Gestor de Persistencia SQLite Optimizado.
Utiliza connection pooling, WAL mode y batch inserts para rendimiento.
"""

import sqlite3
import os
import time
import threading
from datetime import datetime
from .helpers import LOGS_DIR, log_error

DB_PATH = os.path.join(LOGS_DIR, "vision_analytics.db")

class DBManager:
    """Gestiona el registro persistente en SQLite para analitica historica."""
    
    def __init__(self):
        self._conn = None
        self._lock = threading.Lock()
        self._detection_buffer = []
        self._last_flush_time = 0
        self._flush_interval = 2.0  # Segundos entre batch inserts
        self._init_db()

    def _get_conn(self):
        """Obtiene o crea la conexion persistente (connection pooling)."""
        if self._conn is None:
            try:
                os.makedirs(LOGS_DIR, exist_ok=True)
                self._conn = sqlite3.connect(DB_PATH, check_same_thread=False)
                # Optimizaciones de rendimiento
                self._conn.execute("PRAGMA journal_mode=WAL")
                self._conn.execute("PRAGMA synchronous=NORMAL")
                self._conn.execute("PRAGMA cache_size=2000")
            except Exception as e:
                log_error("EXE-UTL-DB-01", f"Error creando conexion SQLite: {e}")
        return self._conn

    def _init_db(self):
        try:
            conn = self._get_conn()
            if conn is None:
                return
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
        except Exception as e:
            log_error("EXE-UTL-DB-01", f"Error inicializando DB: {e}")

    def log_detections(self, detections):
        """Acumula detecciones en un buffer y las escribe en batch periodicamente."""
        now = time.time()
        timestamp = datetime.now().isoformat()
        
        with self._lock:
            for d in detections:
                zones = ",".join(map(str, d.get("zone_indices", [])))
                self._detection_buffer.append(
                    (timestamp, d['label'], d.get('track_id'), d.get('confidence', d.get('conf', 0)), zones)
                )
            
            # Flush si ha pasado el intervalo o el buffer es grande
            if (now - self._last_flush_time >= self._flush_interval) or len(self._detection_buffer) > 500:
                self._flush_detections()

    def _flush_detections(self):
        """Escribe el buffer de detecciones acumuladas en un solo batch."""
        if not self._detection_buffer:
            return
        
        batch = self._detection_buffer.copy()
        self._detection_buffer.clear()
        self._last_flush_time = time.time()
        
        try:
            conn = self._get_conn()
            if conn is None:
                return
            cursor = conn.cursor()
            cursor.executemany(
                'INSERT INTO detections (timestamp, label, track_id, conf, zones) VALUES (?, ?, ?, ?, ?)',
                batch
            )
            conn.commit()
        except Exception as e:
            log_error("EXE-UTL-DB-02", f"Error en batch insert de detecciones: {e}")

    def log_event(self, rule_name, message, evidence_path=""):
        """Registra la activacion de un hito (escritura inmediata por ser evento critico)."""
        try:
            conn = self._get_conn()
            if conn is None:
                return
            with self._lock:
                cursor = conn.cursor()
                cursor.execute('INSERT INTO events (rule_name, message, evidence_path) VALUES (?, ?, ?)', 
                               (rule_name, message, evidence_path))
                conn.commit()
        except Exception as e:
            log_error("EXE-UTL-DB-03", f"Error logueando evento: {e}")

    def close(self):
        """Cierra la conexion persistente y vacia el buffer."""
        with self._lock:
            self._flush_detections()
            if self._conn:
                try:
                    self._conn.close()
                except Exception:
                    pass
                self._conn = None
