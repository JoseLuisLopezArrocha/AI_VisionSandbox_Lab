"""
Motor de Telemetría y Persistencia.
Agrega y persiste detecciones en formato CSV.
"""

import csv
import os
import time
from datetime import datetime
from collections import Counter
from .helpers import LOGS_DIR, log_error

class DataLogger:
    """
    Motor de Telemetría y Persistencia.
    """

    def __init__(self):
        try:
            if not os.path.exists(LOGS_DIR):
                os.makedirs(LOGS_DIR, exist_ok=True)
            
            self.current_file = os.path.join(LOGS_DIR, f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            self.last_log_time = 0
            self.pending_data = []
            self._initialize_csv()
        except Exception as e:
            log_error("EXE-UTL-LOG-01", f"Error al inicializar logger: {e}")

    def _initialize_csv(self):
        """Crea encabezado CSV."""
        if not os.path.exists(self.current_file):
            with open(self.current_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "Objeto", "Total", "Zonas_Detalle"])

    def log(self, detections, zones_count):
        """Registra datos agregados cada segundo."""
        try:
            now = time.time()
            counts = Counter([d['label'] for d in detections])
            self.pending_data.append((counts, zones_count))
            
            if now - self.last_log_time >= 1.0:
                self._flush_pending(now)
                self.last_log_time = now
        except Exception as e:
            log_error("EXE-UTL-LOG-01", f"Error durante log: {e}")

    def _flush_pending(self, now):
        """Escribe datos acumulados en disco."""
        if not self.pending_data: return
        
        final_counts = Counter()
        final_zones = []
        for counts, z_count in self.pending_data:
            for label, count in counts.items():
                final_counts[label] = max(final_counts[label], count)
            final_zones = z_count

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(self.current_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not final_counts:
                writer.writerow([timestamp, "Ninguno", 0, ""])
            else:
                for label, count in final_counts.items():
                    z_detail = " • ".join([f"Z{i+1}:{final_zones[i]}" for i in range(len(final_zones))]) if final_zones else "Global"
                    writer.writerow([timestamp, label, count, z_detail])
        self.pending_data = []

    def get_log_path(self):
        return self.current_file
