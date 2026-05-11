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
                
                # Lógica de Cambio de Contexto (Zonas vs General)
                mode = getattr(app, 'bar_chart_mode', 'General')
                
                if mode == "General":
                    total_ever = app.total_detections_ever
                    breakdown_source = app.session_class_counts
                    total_now = len(detections)
                else:
                    try:
                        zi = int(mode[1:]) - 1
                        z_data = app.session_zone_data.get(zi, {"ids": set(), "counts": Counter()})
                        total_ever = len(z_data["ids"])
                        breakdown_source = z_data["counts"]
                        total_now = sum(1 for d in detections if zi in d.get("zone_indices", []))
                    except:
                        total_ever = 0
                        breakdown_source = Counter()
                        total_now = 0

                # Tiempo de proceso total (Render)
                elapsed = int((time.time() - t0) * 1000)
                if hasattr(app, 'infer_label'):
                    app.infer_label.configure(text=f"PROCESO: {elapsed} ms")
                
                # Conteo actual (Inmediato)
                app.count_label.configure(text=f"OBJETOS: {total_now}")

                # Conteos por zonas (El pie de página del contador)
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

                # Actualizar telemetría de sesión (Adaptable a Zona)
                if hasattr(app, 'total_ever_label'):
                    app.total_ever_label.configure(text=f"{total_ever:,}")
                
                if hasattr(app, 'uptime_label') and app.session_start_time:
                    uptime_sec = int(time.time() - app.session_start_time)
                    hrs = uptime_sec // 3600
                    mins = (uptime_sec % 3600) // 60
                    secs = uptime_sec % 60
                    app.uptime_label.configure(text=f"{hrs:02d}:{mins:02d}:{secs:02d}")

                if hasattr(app, 'breakdown_label'):
                    cfg = getattr(app, 'dashboard_config', {})
                    show_top = cfg.get("show_top_5", True)
                    pinned = cfg.get("pinned_classes", [])
                    
                    lines = []
                    
                    # 1. Mostrar Pinned Classes (Siempre visibles)
                    if pinned:
                        lines.append("MÉTRICAS FIJAS:")
                        for p_name in pinned:
                            count = breakdown_source.get(p_name, 0)
                            lines.append(f"• {p_name.upper()}: {count}")
                        lines.append("") # Separador
                    
                    # 2. Mostrar Top 5 (Automático)
                    if show_top:
                        top = breakdown_source.most_common(5)
                        if top:
                            lines.append(f"TOP 5 ÚNICOS ({mode.upper()}):")
                            for k, v in top:
                                # Evitar duplicar si ya está en pinned
                                if k not in pinned:
                                    lines.append(f"• {k.upper()}: {v}")
                    
                    if lines:
                        app.breakdown_label.configure(text="\n".join(lines))
                    else:
                        app.breakdown_label.configure(text=f"Sin datos en {mode}")

            # La gráfica ya tiene su propio control de tiempo interno, pero la llamamos siempre
            # para que use el historial de detecciones si es necesario.
            if now - getattr(app, '_last_bar_draw', 0) >= 0.2:
                if hasattr(app, 'bar_canvas'):
                    VisualPainter.draw_bar_chart(app, app.bar_canvas, getattr(app, 'last_detections', []))
                app._last_bar_draw = now

            # 3. Mantener buffer de historial (Series Temporales)
            if not hasattr(app, 'chart_history'): app.chart_history = []
            if now - getattr(app, '_last_history_snapshot', 0) >= 1.0: # Cada segundo
                app.chart_history.append((now, list(detections), zone_current_counts))
                if len(app.chart_history) > 60: # Mantener 60 segundos
                    app.chart_history.pop(0)
                app._last_history_snapshot = now

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
        """Redirigido al nuevo motor de graficos modular."""
        VisualPainter.draw_chart(app, canvas, detections)

    @staticmethod
    def draw_chart(app, canvas, detections):
        """Motor de graficos avanzado: modular y configurable."""
        canvas.delete("all")
        w, h = canvas.winfo_width(), canvas.winfo_height()
        if w < 50 or h < 50: return
        
        cfg = getattr(app, 'dashboard_config', {})
        ctype = cfg.get("chart_type", "vbar")
        ax_x = cfg.get("axis_x", "class")
        ax_y = cfg.get("axis_y", "count")
        
        # --- 1. RECOLECCION DE DATOS SEGUN CONFIGURACION ---
        data = [] # List of (label, value)
        
        if ax_x == "class":
            # X = Clases. Y = Métrica elegida.
            source = app.session_class_counts if ax_y == "cumulative" else Counter([d['label'] for d in detections])
            if ax_y == "conf":
                # Media de confianza por clase (en el frame actual)
                conf_sums = Counter()
                conf_counts = Counter()
                for d in detections:
                    conf_sums[d['label']] += d.get('confidence', 0)
                    conf_counts[d['label']] += 1
                data = [(k, conf_sums[k]/conf_counts[k]) for k in conf_sums]
            else:
                data = sorted(source.items(), key=lambda x: x[1], reverse=True)[:6]
        
        elif ax_x == "zone":
            # X = Zonas. Y = Métrica elegida.
            zones_count = len(app.zones) if hasattr(app, 'zones') else 0
            for i in range(zones_count):
                label = f"Z{i+1}"
                if ax_y == "cumulative":
                    val = len(app.session_zone_data.get(i, {"ids": set()})["ids"])
                else:
                    val = sum(1 for d in detections if i in d.get("zone_indices", []))
                data.append((label, val))
        
        elif ax_x == "time":
            # Caso especial: Serie temporal (solo util para linea)
            if not getattr(app, 'chart_history', []): 
                canvas.create_text(w/2, h/2, text="Esperando datos temporales...", fill="#444", font=("Arial", 10))
                return
            # Y = Conteo de objetos
            data = [(time.strftime("%M:%S", time.localtime(t)), len(dets)) for t, dets, zc in app.chart_history]
        
        if not data:
            canvas.create_text(w/2, h/2, text="Sin datos para esta configuracion", fill="#444", font=("Arial", 10))
            return

        # --- 2. RENDERIZADO SEGUN TIPO ---
        max_val = max([v for l, v in data]) if data else 1
        if max_val == 0: max_val = 1
        
        padding = 40
        if ctype == "vbar":
            bar_w = (w - (padding * 2)) / len(data)
            for i, (label, val) in enumerate(data):
                bar_h = (val / max_val) * (h - 60)
                x0 = padding + (i * bar_w) + 5
                y0 = h - 25 - bar_h
                x1 = x0 + bar_w - 10
                y1 = h - 25
                canvas.create_rectangle(x0, y0, x1, y1, fill="#10b981" if i==0 else "#065f46", outline="#10b981", width=1)
                canvas.create_text((x0+x1)/2, y0-10, text=f"{val:.1f}" if ax_y=="conf" else str(int(val)), fill="#94a3b8", font=("Arial", 8))
                canvas.create_text((x0+x1)/2, h-12, text=label[:8], fill="#444", font=("Arial", 8))
                
        elif ctype == "hbar":
            bar_h = (h - (padding * 2)) / len(data)
            for i, (label, val) in enumerate(data):
                bar_w = (val / max_val) * (w - 120)
                y0 = padding + (i * bar_h) + 2
                x0 = 80
                y1 = y0 + bar_h - 4
                x1 = x0 + bar_w
                canvas.create_rectangle(x0, y0, x1, y1, fill="#3b82f6", outline="#3b82f6", width=1)
                canvas.create_text(40, (y0+y1)/2, text=label[:10], fill="#94a3b8", font=("Arial", 8))
                canvas.create_text(x1+15, (y0+y1)/2, text=str(int(val)), fill="#64748b", font=("Arial", 8))

        elif ctype == "line":
            pts = []
            step_x = (w - (padding * 2)) / max(len(data)-1, 1)
            for i, (label, val) in enumerate(data):
                px = padding + (i * step_x)
                py = h - 35 - (val / max_val) * (h - 70)
                pts.append((px, py))
            
            if len(pts) > 1:
                canvas.create_line(pts, fill="#38bdf8", width=2, smooth=True)
                # Dibujar ultimo punto destacado
                lx, ly = pts[-1]
                canvas.create_oval(lx-3, ly-3, lx+3, ly+3, fill="#38bdf8")
                canvas.create_text(lx, ly-15, text=str(int(data[-1][1])), fill="#38bdf8", font=("Arial", 10, "bold"))
            
            # Ejes y etiquetas minimas
            canvas.create_line(padding, h-35, w-padding, h-35, fill="#222")
            canvas.create_text(w/2, h-12, text=f"HISTORIAL TEMPORAL (X:{ax_x.upper()} Y:{ax_y.upper()})", fill="#444", font=("Arial", 8))

    @staticmethod
    def draw_detections(frame, detections, is_focus=False, show_trails=True, dual_mode=False):
        """Dibuja cajas, etiquetas y opcionalmente trayectorias.
        
        En Dual Mode, las detecciones del modelo primario se dibujan en azul
        y las del secundario en naranja, usando el campo 'source' de cada detección.
        """
        from ..core.detector import ObjectDetector

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
                if not hasattr(VisualPainter, '_cleanup_cnt'): VisualPainter._cleanup_cnt = {}
                VisualPainter._cleanup_cnt[tid] = VisualPainter._cleanup_cnt.get(tid, 0) + 1
                if VisualPainter._cleanup_cnt[tid] > 50:
                    del VisualPainter._track_history[tid]
                    del VisualPainter._cleanup_cnt[tid]

        # Capa para cajas translucidas
        box_overlay = frame.copy()
        
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            label = d["label"]
            conf = d["confidence"]
            cls_id = d["class_id"]
            t_id = d.get("track_id")
            source = d.get("source", "primary")
            
            # Selección de color basada en fuente del modelo
            if is_focus:
                color = (0, 215, 255)
            elif dual_mode:
                color = ObjectDetector.PRIMARY_COLOR if source == "primary" else ObjectDetector.SECONDARY_COLOR
            else:
                color = (255, 255, 0) if cls_id >= 1000 else (233, 165, 14)
            
            thick = 3 if is_focus else 2
            
            # Dibujar Trayectoria (Trail) — solo para primario con tracking
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

            # Dibujar Caja (Fondo translucido)
            cv2.rectangle(box_overlay, (x1, y1), (x2, y2), color, -1)
            # Dibujar Borde
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)
            
            # Dibujar Etiqueta
            id_txt = f" id:{t_id}" if t_id is not None else ""
            source_tag = " [M2]" if dual_mode and source == "secondary" else ""
            tag = f"{label.upper()}{id_txt} {conf:.2f}{source_tag}"
            if is_focus: tag = f"[ FOCUS ] {tag}"
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            f_scale = 0.5
            thick_txt = 2 if is_focus else 1
            (tw, th), baseline = cv2.getTextSize(tag, font, f_scale, thick_txt)
            
            # Etiqueta opaca sobre la caja
            cv2.rectangle(frame, (x1, y1 - th - 12), (x1 + tw + 10, y1), color, -1)
            cv2.putText(frame, tag, (x1 + 5, y1 - 8), font, f_scale, (0, 0, 0), thick_txt)
            
        # Mezclar cajas translucidas
        cv2.addWeighted(box_overlay, 0.25, frame, 0.75, 0, frame)
        return frame

    @staticmethod
    def draw_model_legend(frame, primary_name, secondary_name):
        """Dibuja una leyenda flotante en la esquina superior derecha del frame.
        
        Muestra el nombre de cada modelo con su color correspondiente.
        Solo se dibuja en Dual Mode (cuando secondary_name no es None).
        """
        from ..core.detector import ObjectDetector

        if secondary_name is None:
            return frame

        h, w = frame.shape[:2]
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        f_scale = 0.45
        thick = 1
        
        # Preparar textos
        m1_text = f"M1: {primary_name or 'Sin Modelo'}"
        m2_text = f"M2: {secondary_name}"
        
        (tw1, th1), _ = cv2.getTextSize(m1_text, font, f_scale, thick)
        (tw2, th2), _ = cv2.getTextSize(m2_text, font, f_scale, thick)
        
        max_tw = max(tw1, tw2)
        padding = 8
        line_h = 22
        box_w = max_tw + 50 + padding * 2  # Espacio para icono de color + texto
        box_h = line_h * 2 + padding * 2 + 4
        
        # Posición: esquina superior derecha
        x_start = w - box_w - 15
        y_start = 15
        
        # Fondo semi-transparente
        overlay = frame.copy()
        cv2.rectangle(overlay, (x_start, y_start), (x_start + box_w, y_start + box_h), (15, 15, 25), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        
        # Borde
        cv2.rectangle(frame, (x_start, y_start), (x_start + box_w, y_start + box_h), (100, 100, 120), 1)
        
        # Título
        cv2.putText(frame, "DUAL MODE", (x_start + padding, y_start + 14), font, 0.4, (180, 180, 200), 1)
        
        # Línea M1 con cuadrado de color
        y_line1 = y_start + padding + 22
        cv2.rectangle(frame, (x_start + padding, y_line1 - 8), (x_start + padding + 14, y_line1 + 4), ObjectDetector.PRIMARY_COLOR, -1)
        cv2.putText(frame, m1_text, (x_start + padding + 22, y_line1 + 2), font, f_scale, (220, 220, 240), thick)
        
        # Línea M2 con cuadrado de color
        y_line2 = y_line1 + line_h
        cv2.rectangle(frame, (x_start + padding, y_line2 - 8), (x_start + padding + 14, y_line2 + 4), ObjectDetector.SECONDARY_COLOR, -1)
        cv2.putText(frame, m2_text, (x_start + padding + 22, y_line2 + 2), font, f_scale, (220, 220, 240), thick)
        
        return frame

