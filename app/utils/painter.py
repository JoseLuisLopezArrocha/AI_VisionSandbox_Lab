import cv2
import numpy as np
import time
from collections import Counter
from .helpers import ZONE_COLORS

class VisualPainter:
    """
    Motor Grafico y Telemetrico.
    
    Centraliza todas las operaciones de pintado sobre el frame de video y la 
    actualizacion de metricas en los lienzos (canvas) del Dashboard.
    Aisla la logica de visualizacion del motor de deteccion.
    """

    _track_history: dict = {}
    _cleanup_cnt: dict = {}

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
        """Actualiza las etiquetas de la interfaz con frecuencia controlada (5 FPS)."""
        try:
            now = time.time()
            if not hasattr(app, '_last_metrics_update'): app._last_metrics_update = 0
            
            # Solo actualizar etiquetas de texto 5 veces por segundo
            should_update_text = (now - app._last_metrics_update >= 0.2)
            
            if should_update_text:
                app._last_metrics_update = now
                
                # Tiempo de proceso total (Render)
                elapsed = int((time.time() - t0) * 1000)
                if hasattr(app, 'infer_label'):
                    app.infer_label.configure(text=f"PROCESO: {elapsed} ms")
                
                # Conteo total
                total = len(detections)
                app.count_label.configure(text=f"OBJETOS: {total}")

                # Conteos por zonas
                if zones:
                    zone_counts = Counter()
                    for d in detections:
                        z_indices = d.get("zone_indices", [])
                        for zi in z_indices:
                            if zi >= 0:
                                zone_counts[zi] += 1
                    
                    z_text = " • ".join([f"Z{i+1}: {zone_counts[i]}" for i in range(len(zones))])
                    app.zone_counts_label.configure(text=z_text)
                else:
                    app.zone_counts_label.configure(text="Conteo global")

                # Actualizar telemetría de sesión
                if hasattr(app, 'total_ever_label'):
                    app.total_ever_label.configure(text=f"{app.total_detections_ever:,}")
                
                if hasattr(app, 'uptime_label') and app.session_start_time:
                    uptime_sec = int(time.time() - app.session_start_time)
                    hrs = uptime_sec // 3600
                    mins = (uptime_sec % 3600) // 60
                    secs = uptime_sec % 60
                    app.uptime_label.configure(text=f"{hrs:02d}:{mins:02d}:{secs:02d}")

                if hasattr(app, 'breakdown_label'):
                    top = app.session_class_counts.most_common(5)
                    if top:
                        txt = "TOP 5 ÚNICOS:\n" + "\n".join([f"• {k.upper()}: {v}" for k, v in top])
                        app.breakdown_label.configure(text=txt)
                    else:
                        app.breakdown_label.configure(text="Sin datos únicos")

            # La gráfica ya tiene su propio control de tiempo interno, pero la llamamos siempre
            # para que use el historial de detecciones si es necesario.
            if now - getattr(app, '_last_bar_draw', 0) >= 0.2:
                if hasattr(app, 'bar_canvas'):
                    VisualPainter.draw_bar_chart(app, app.bar_canvas, getattr(app, 'last_detections', []))
                app._last_bar_draw = now

            # Retornar conteos de zonas para el logger (esto sí cada frame si se requiere precisión)
            zone_current_counts = []
            if zones:
                z_counts = Counter()
                for d in detections:
                    for zi in d.get("zone_indices", []):
                        if zi >= 0: z_counts[zi] += 1
                for i in range(len(zones)):
                    zone_current_counts.append(z_counts[i])
            
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
    def draw_bar_chart(app, canvas, detections):
        """Dibuja gráfico de barras interactivo con soporte de zonas."""
        canvas.delete("all")
        w, h = canvas.winfo_width(), canvas.winfo_height()
        if w < 50 or h < 50: return

        # --- FILTRADO POR MODO DE GRÁFICA (ZONAS) ---
        mode = getattr(app, 'bar_chart_mode', 'General')
        filtered_detections = detections
        if mode != "General":
            try:
                zone_idx = int(mode[1:]) - 1 # De "Z1" sacamos 0
                filtered_detections = [d for d in detections if zone_idx in d.get("zone_indices", [])]
            except:
                pass
        
        # Agrupar por class_id para que el filtrado sea preciso
        labels_map = {}
        counts = Counter()
        for d in filtered_detections:
            cid = d['class_id']
            labels_map[cid] = d['label']
            counts[cid] += 1
            
        if not counts:
            txt = "Sin detecciones" if mode == "General" else f"Sin datos en {mode}"
            canvas.create_text(w/2, h/2, text=txt, fill="#444", font=("Arial", 10))
            return

        sorted_cids = sorted(counts.keys(), key=lambda x: counts[x], reverse=True)[:6]
        max_val = max(counts.values()) if counts else 1
        
        padding = 30
        bar_area_h = h - 60
        bar_w = (w - (padding * 2)) / len(sorted_cids)
        
        def on_bar_click(cid):
            # Lógica de alternancia (Toggle)
            if app.target_classes and cid in app.target_classes and len(app.target_classes) == 1:
                app.target_classes = None
                app.add_log("Filtro de clase deshabilitado.")
            else:
                app.target_classes = [cid]
                app.add_log(f"Filtrando solo clase: {labels_map[cid].upper()}")
            
            # Notificar a la app para guardar config y limpiar frames
            if hasattr(app, '_on_filter_applied'):
                app._on_filter_applied(app.target_classes)
            
            # Forzar redibujado inmediato tras clic
            VisualPainter.draw_bar_chart(app, canvas, detections)

        # Vincular evento general una sola vez (Tkinter find_withtag es más robusto)
        canvas.tag_bind("bar_obj", "<Button-1>", lambda e: None) # Placeholder

        for i, cid in enumerate(sorted_cids):
            val = counts[cid]
            label = labels_map[cid]
            bh = (val / max_val) * bar_area_h
            x0 = padding + i * bar_w
            
            # Color especial si está filtrado
            is_filtered = app.target_classes and cid in app.target_classes
            color = "#f59e0b" if is_filtered else "#0ea5e9"
            
            tag = f"bar_{cid}"
            rect_id = canvas.create_rectangle(x0, h - 30 - bh, x0 + bar_w - 10, h - 30, 
                                            fill=color, outline="", tags=(tag, "bar_obj"))
            
            canvas.create_text(x0 + (bar_w-10)/2, h - 15, text=label, fill="#94a3b8", font=("Arial", 9), tags=(tag, "bar_obj"))
            canvas.create_text(x0 + (bar_w-10)/2, h - 45 - bh, text=str(val), fill="#fff", font=("Arial", 9, "bold"), tags=(tag, "bar_obj"))
            
            # Bind click (Uso de clausura con cid actual)
            canvas.tag_bind(tag, "<Button-1>", lambda e, c=cid: on_bar_click(c))

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
