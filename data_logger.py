import csv
import os
import time
from datetime import datetime
from collections import Counter

LOGS_DIR = "telemetry_logs"

class DataLogger:
    """
    Motor de Telemetría y Persistencia.
    
    Agrega las detecciones en intervalos de un segundo y las persiste en 
    formato CSV para su posterior análisis estadístico o auditoría.
    """

    def __init__(self):
        if not os.path.exists(LOGS_DIR):
            os.makedirs(LOGS_DIR)
        
        self.current_file = os.path.join(LOGS_DIR, f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        self.last_log_time = 0
        self.pending_data = [] # Lista de frames detectados en el último segundo
        
        self._initialize_csv()

    def _initialize_csv(self):
        """Crea el encabezado del archivo si no existe."""
        if not os.path.exists(self.current_file):
            with open(self.current_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "Objeto", "Total", "Zonas_Detalle"])

    def log(self, detections, zones_count):
        """Registra detecciones agregando datos si ha pasado un segundo."""
        now = time.time()
        
        # Recopilar datos del frame actual
        counts = Counter([d['label'] for d in detections])
        self.pending_data.append((counts, zones_count))
        
        if now - self.last_log_time >= 1.0:
            self._flush_pending(now)
            self.last_log_time = now

    def _flush_pending(self, now):
        """Procesa y guarda los datos acumulados del último segundo."""
        if not self.pending_data:
            return

        # Sacar el promedio o el máximo del segundo
        # Para simplificar y ser útil, usaremos la detección máxima de cada clase en este segundo
        final_counts = Counter()
        final_zones = []
        
        for counts, z_count in self.pending_data:
            for label, count in counts.items():
                final_counts[label] = max(final_counts[label], count)
            # Para zonas, guardamos el último estado conocido o el máximo
            final_zones = z_count 

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        with open(self.current_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # Escribir una fila por cada clase detectada en ese segundo
            if not final_counts:
                # Opcional: registrar fila vacía para mantener el ritmo temporal
                writer.writerow([timestamp, "Ninguno", 0, ""])
            else:
                for label, count in final_counts.items():
                    z_detail = " • ".join([f"Z{i+1}:{final_zones[i]}" for i in range(len(final_zones))]) if final_zones else "Global"
                    writer.writerow([timestamp, label, count, z_detail])
        
        self.pending_data = []

    def get_log_path(self):
        return self.current_file
