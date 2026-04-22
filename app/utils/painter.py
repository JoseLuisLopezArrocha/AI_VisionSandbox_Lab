import cv2
import numpy as np
import time
from collections import Counter
from .helpers import ZONE_COLORS

class VisualPainter:
    """
    Motor Gráfico y Telemétrico.
    
    Centraliza todas las operaciones de pintado sobre el frame de vídeo y la 
    actualización de métricas en los lienzos (canvas) del Dashboard.
    Aísla la lógica de visualización del motor de detección.
    """

    @staticmethod
    def draw_zones(frame, zones, detections):
        """Dibuja polígonos, contornos y etiquetas de conteo por zona."""
        h, w = frame.shape[:2]
        if not zones:
            return frame

        # 1. Rellenos semitransparentes
        overlay = frame.copy()
        for i, zone in enumerate(zones):
            color = ZONE_COLORS[i % len(ZONE_COLORS)]
            pts = np.array([(int(x * w), int(y * h)) for x, y in zone], np.int32)
            cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)

        # 2. Contornos y Etiquetas
        zone_summaries = {}
        for d in detections:
            # Soporte multizona: iterar sobre todos los índices de zona detectados
            z_indices = d.get("zone_indices", [])
            for zi in z_indices:
                if zi >= 0:
                    if zi not in zone_summaries: zone_summaries[zi] = Counter()
                    zone_summaries[zi][d["label"]] += 1

        for i, zone in enumerate(zones):
            color = ZONE_COLORS[i % len(ZONE_COLORS)]
            pts = np.array([(int(x * w), int(y * h)) for x, y in zone], np.int32)
            cv2.polylines(frame, [pts], True, color, 2)

            # Cartel de información en el centro de la zona
            cx = int(np.mean([x for x, _ in zone]) * w)
            cy = int(np.mean([y for _, y in zone]) * h)
            
            title = f"ZONA {i+1}"
            details = ", ".join([f"{k}:{v}" for k, v in zone_summaries.get(i, {}).items()])
            lines = [title]
            if details: lines.append(details)
            else: lines.append("Vacio") # Arreglado error de codificación

            # Estética del cartel
            font = cv2.FONT_HERSHEY_SIMPLEX
            f_scale = 0.5
            thick = 1
            l_height = 18
            
            max_w = 0
            for line in lines:
                (tw, _), _ = cv2.getTextSize(line, font, f_scale, thick)
                max_w = max(max_w, tw)
            
            bx_w, bx_h = max_w + 10, len(lines) * l_height + 10
            rx, ry = cx - bx_w // 2, cy - bx_h // 2
            
            cv2.rectangle(frame, (rx, ry), (rx + bx_w, ry + bx_h), (0,0,0), -1)
            cv2.rectangle(frame, (rx, ry), (rx + bx_w, ry + bx_h), color, 1)
            
            for j, line in enumerate(lines):
                cv2.putText(frame, line, (rx + 5, ry + 15 + j * l_height), font, f_scale, (255,255,255) if j==0 else color, thick)

        return frame

    @staticmethod
    def draw_heatmap(frame, detections, heatmap_acc):
        """Genera y aplica un mapa de calor acumulativo."""
        h, w = frame.shape[:2]
        if heatmap_acc is None or heatmap_acc.shape != (h, w):
            heatmap_acc = np.zeros((h, w), np.float32)
        
        heatmap_acc *= 0.95
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            cv2.circle(heatmap_acc, ((x1+x2)//2, (y1+y2)//2), 20, 1.0, -1)
        
        norm = cv2.normalize(heatmap_acc, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        heat = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        result = cv2.addWeighted(frame, 0.7, heat, 0.3, 0)
        return result, heatmap_acc

    @staticmethod
    def update_sidebar_metrics(app, t0, detections, zones):
        """Actualiza las etiquetas de la interfaz basándose en los datos actuales."""
        try:
            # Tiempo de proceso total (Render)
            elapsed = int((time.time() - t0) * 1000)
            if hasattr(app, 'infer_label'):
                app.infer_label.configure(text=f"PROCESO: {elapsed} ms")
            
            # Conteo total
            total = len(detections)
            app.count_label.configure(text=f"OBJETOS: {total}")

            # Conteos por zonas
            zone_current_counts = []
            if zones:
                zone_counts = Counter()
                for d in detections:
                    # Soporte multizona: el objeto puede sumar a varias zonas si se solapan
                    z_indices = d.get("zone_indices", [])
                    for zi in z_indices:
                        if zi >= 0:
                            zone_counts[zi] += 1
                
                for i in range(len(zones)):
                    zone_current_counts.append(zone_counts[i])

                z_text = " • ".join([f"Z{i+1}: {zone_counts[i]}" for i in range(len(zones))])
                app.zone_counts_label.configure(text=z_text)
            else:
                app.zone_counts_label.configure(text="Conteo global")

            # Actualizar historial para el gráfico (Mantener últimos 30 segundos)
            if not hasattr(app, 'history_buffer'):
                app.history_buffer = []
            
            now_ms = time.time()
            if not hasattr(app, 'last_history_update') or now_ms - app.last_history_update >= 1.0:
                app.history_buffer.append(total)
                if len(app.history_buffer) > 30:
                    app.history_buffer.pop(0)
                app.last_history_update = now_ms

            # Dibujar Gráficos en el Dashboard
            if hasattr(app, 'bar_canvas'):
                VisualPainter.draw_bar_chart(app.bar_canvas, detections)
            if hasattr(app, 'line_canvas'):
                VisualPainter.draw_line_chart(app.line_canvas, app.history_buffer)

            return zone_current_counts
        except Exception as e:
            # Fallo silencioso en métricas para no crashear el core
            print(f"Error actualizando métricas UI: {e}")

    @staticmethod
    def draw_live_zone(frame, current_zone):
        """Dibuja los puntos de la zona que se está creando actualmente."""
        if not current_zone:
            return frame
        h, w = frame.shape[:2]
        curr_pts = [(int(x * w), int(y * h)) for x, y in current_zone]
        for j, pt in enumerate(curr_pts):
            cv2.circle(frame, pt, 5, (0, 255, 0), -1)
            if j > 0: cv2.line(frame, curr_pts[j-1], pt, (0, 255, 0), 2)
        return frame

    @staticmethod
    def draw_bar_chart(canvas, detections):
        """Dibuja gráfico de barras de distribución."""
        canvas.delete("all")
        w, h = canvas.winfo_width(), canvas.winfo_height()
        if w < 50 or h < 50: return

        counts = Counter([d['label'] for d in detections])
        if not counts:
            canvas.create_text(w/2, h/2, text="Sin detecciones", fill="#444", font=("Arial", 10))
            return

        labels = list(counts.keys())[:6]
        vals = [counts[l] for l in labels]
        max_val = max(vals) if vals else 1
        
        padding = 30
        bar_area_h = h - 60
        bar_w = (w - (padding * 2)) / len(labels)
        
        for i, (label, val) in enumerate(zip(labels, vals)):
            bh = (val / max_val) * bar_area_h
            x0 = padding + i * bar_w
            canvas.create_rectangle(x0, h - 30 - bh, x0 + bar_w - 10, h - 30, fill="#0ea5e9", outline="")
            canvas.create_text(x0 + (bar_w-10)/2, h - 15, text=label, fill="#94a3b8", font=("Arial", 9))
            canvas.create_text(x0 + (bar_w-10)/2, h - 45 - bh, text=str(val), fill="#fff", font=("Arial", 9, "bold"))

    @staticmethod
    def draw_line_chart(canvas, history):
        """Dibuja gráfico de línea de tendencia temporal."""
        canvas.delete("all")
        w, h = canvas.winfo_width(), canvas.winfo_height()
        if w < 50 or h < 50 or not history: return

        max_h = max(max(history), 5)
        padding = 30
        plot_w = w - (padding * 2)
        plot_h = h - 60
        
        step_x = plot_w / 30
        pts = []
        for i, val in enumerate(history):
            pts.extend([padding + i * step_x, (h - 30) - (val / max_h) * plot_h])
        
        if len(pts) >= 4:
            canvas.create_line(pts, fill="#10b981", width=3, smooth=True)
            poly_pts = [padding, h - 30] + pts + [pts[-2], h - 30]
            canvas.create_polygon(poly_pts, fill="#10b981", stipple="gray25", outline="")

        canvas.create_text(padding, 20, text=f"MÁXIMO: {max(history)}", fill="#10b981", anchor="nw", font=("Arial", 8, "bold"))
    _track_history = {} # Historial de trayectorias {id: [(x,y), ...]}

    @staticmethod
    def draw_detections(frame, detections, is_focus=False, show_trails=True):
        """Dibuja cajas, etiquetas y opcionalmente trayectorias."""
        h, w = frame.shape[:2]
        
        # Indicador de Modo Focus
        if is_focus:
            cv2.rectangle(frame, (10, 10), (220, 45), (0, 0, 0), -1)
            cv2.rectangle(frame, (10, 10), (220, 45), (0, 255, 255), 2)
            cv2.putText(frame, "TRACKING FOCUS ACTIVE", (20, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # Limpiar IDs antiguos del historial para no saturar memoria
        current_ids = {d.get("track_id") for d in detections if d.get("track_id") is not None}
        for tid in list(VisualPainter._track_history.keys()):
            if tid not in current_ids:
                # Si el ID no aparece en 50 frames, lo borramos (podría reaparecer pero es seguro)
                if not hasattr(VisualPainter, '_cleanup_cnt'): VisualPainter._cleanup_cnt = {}
                VisualPainter._cleanup_cnt[tid] = VisualPainter._cleanup_cnt.get(tid, 0) + 1
                if VisualPainter._cleanup_cnt[tid] > 50:
                    del VisualPainter._track_history[tid]
                    del VisualPainter._cleanup_cnt[tid]

        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            label = d["label"]
            conf = d["confidence"]
            cls_id = d["class_id"]
            t_id = d.get("track_id")
            
            color = (0, 215, 255) if is_focus else ((255, 255, 0) if cls_id >= 1000 else (233, 165, 14))
            thick = 3 if is_focus else 2
            
            # Dibujar Trayectoria (Trail)
            if show_trails and t_id is not None:
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                if t_id not in VisualPainter._track_history:
                    VisualPainter._track_history[t_id] = []
                VisualPainter._track_history[t_id].append(center)
                if len(VisualPainter._track_history[t_id]) > 30:
                    VisualPainter._track_history[t_id].pop(0)
                
                # Dibujar línea de puntos
                pts = VisualPainter._track_history[t_id]
                for i in range(1, len(pts)):
                    alpha = i / len(pts)
                    c = (int(color[0]*alpha), int(color[1]*alpha), int(color[2]*alpha))
                    cv2.line(frame, pts[i-1], pts[i], c, 2)

            # Dibujar Caja
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)
            
            # Dibujar Etiqueta
            id_txt = f" ID:{t_id}" if t_id is not None else ""
            tag = f"{label.upper()}{id_txt} {conf:.2f}"
            if is_focus: tag = f"[ FOCUS ] {tag}"
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            f_scale = 0.5
            thick_txt = 2 if is_focus else 1
            (tw, th), baseline = cv2.getTextSize(tag, font, f_scale, thick_txt)
            
            cv2.rectangle(frame, (x1, y1 - th - 12), (x1 + tw + 10, y1), color, -1)
            cv2.putText(frame, tag, (x1 + 5, y1 - 8), font, f_scale, (0, 0, 0), thick_txt)
            
        return frame
