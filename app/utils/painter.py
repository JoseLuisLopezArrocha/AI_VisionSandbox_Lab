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

    # [Propósito]: Historial de coordenadas centrales de objetos indexados por track_id para renderizar trazas de movimiento.
    # [Tipo]: dict[int, list[tuple[int, int]]]
    _track_history: dict = {}

    # [Propósito]: Contador de desvanecimiento por ID de tracking para eliminar trazas inactivas y optimizar memoria.
    # [Tipo]: dict[int, int]
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
                # Asegurar que el texto es ASCII para evitar crashes en cv2.putText
                clean_line = "".join([c if ord(c) < 128 else "?" for c in line])
                cv2.putText(frame, clean_line, (rx + 5, ry + 15 + j * l_height), font, f_scale, (255,255,255) if j==0 else color, thick)

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
            
            zone_counts = Counter() # Inicializar siempre al principio
            
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

                # Conteos por zonas (Inmediato)
                if zones:
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
                app.chart_history.append((now, list(detections), zone_counts))
                if len(app.chart_history) > 60: # Mantener 60 segundos
                    app.chart_history.pop(0)
                app._last_history_snapshot = now

            return zone_counts
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
    @staticmethod
    def draw_chart(app, canvas, detections):
        """Motor de graficos avanzado: modular y configurable.
        
        Soporta Modo Dual (M1 + M2) mostrando dos graficos independientes en paralelo.
        Soporta filtrado por clase para desglose comparativo de General vs Zonas.
        """
        canvas.delete("all")
        w, h = canvas.winfo_width(), canvas.winfo_height()
        if w < 50 or h < 50: return
        
        cfg = getattr(app, 'dashboard_config', {})
        ctype = cfg.get("chart_type", "vbar")
        ax_x = cfg.get("axis_x", "class")
        ax_y = cfg.get("axis_y", "count")
        filter_class = cfg.get("filter_class", "Todas")
        
        is_dual = app.detector.is_dual_mode() if hasattr(app, 'detector') else False
        
        if is_dual:
            # Dividir canvas al 50%
            w_half = w // 2
            
            # Separar detecciones
            primary_dets = [d for d in detections if d.get("source", "primary") == "primary"]
            secondary_dets = [d for d in detections if d.get("source") == "secondary"]
            
            # 1. Obtener datos para M1
            data_m1 = []
            if ax_x == "class":
                source = app.session_class_counts if ax_y == "cumulative" else Counter([d['label'] for d in primary_dets])
                if ax_y == "conf":
                    conf_sums = Counter()
                    conf_counts = Counter()
                    for d in primary_dets:
                        conf_sums[d['label']] += d.get('confidence', 0)
                        conf_counts[d['label']] += 1
                    data_m1 = [(k, conf_sums[k]/conf_counts[k]) for k in conf_sums]
                else:
                    data_m1 = sorted(source.items(), key=lambda x: x[1], reverse=True)[:5]
            
            elif ax_x == "zone":
                zones_count = len(app.zones) if hasattr(app, 'zones') else 0
                if filter_class != "Todas":
                    # Barra General para la clase filtrada
                    if ax_y == "cumulative":
                        g_val = app.session_class_counts.get(filter_class, 0)
                    elif ax_y == "conf":
                        primary_target_dets = [d for d in primary_dets if d['label'] == filter_class]
                        g_val = (sum(d.get('confidence', 0) for d in primary_target_dets) / len(primary_target_dets)) if primary_target_dets else 0
                    else:
                        g_val = sum(1 for d in primary_dets if d['label'] == filter_class)
                    data_m1.append(("General", g_val))
                    
                    # Barras de zonas para la clase filtrada
                    for i in range(zones_count):
                        label = f"Z{i+1}"
                        if ax_y == "cumulative":
                            val = app.session_zone_data.get(i, {"counts": Counter()})["counts"].get(filter_class, 0)
                        elif ax_y == "conf":
                            target_dets = [d for d in primary_dets if d['label'] == filter_class and i in d.get("zone_indices", [])]
                            val = (sum(d.get('confidence', 0) for d in target_dets) / len(target_dets)) if target_dets else 0
                        else:
                            val = sum(1 for d in primary_dets if d['label'] == filter_class and i in d.get("zone_indices", []))
                        data_m1.append((label, val))
                else:
                    # Sin filtrar: contar todo en cada zona
                    for i in range(zones_count):
                        label = f"Z{i+1}"
                        if ax_y == "cumulative":
                            val = len(app.session_zone_data.get(i, {"ids": set()})["ids"])
                        else:
                            val = sum(1 for d in primary_dets if i in d.get("zone_indices", []))
                        data_m1.append((label, val))
            
            elif ax_x == "time":
                if getattr(app, 'chart_history', []):
                    data_m1 = [(time.strftime("%M:%S", time.localtime(t)), sum(1 for d in dets if d.get("source", "primary") == "primary")) for t, dets, zc in app.chart_history]
            
            # 2. Obtener datos para M2
            data_m2 = []
            if ax_x == "class":
                source_m2 = getattr(app, 'session_class_counts_m2', Counter()) if ax_y == "cumulative" else Counter([d['label'] for d in secondary_dets])
                if ax_y == "conf":
                    conf_sums = Counter()
                    conf_counts = Counter()
                    for d in secondary_dets:
                        conf_sums[d['label']] += d.get('confidence', 0)
                        conf_counts[d['label']] += 1
                    data_m2 = [(k, conf_sums[k]/conf_counts[k]) for k in conf_sums]
                else:
                    data_m2 = sorted(source_m2.items(), key=lambda x: x[1], reverse=True)[:5]
            
            elif ax_x == "zone":
                zones_count = len(app.zones) if hasattr(app, 'zones') else 0
                if filter_class != "Todas":
                    # Barra General para la clase filtrada en M2
                    if ax_y == "cumulative":
                        g_val = getattr(app, 'session_class_counts_m2', Counter()).get(filter_class, 0)
                    elif ax_y == "conf":
                        secondary_target_dets = [d for d in secondary_dets if d['label'] == filter_class]
                        g_val = (sum(d.get('confidence', 0) for d in secondary_target_dets) / len(secondary_target_dets)) if secondary_target_dets else 0
                    else:
                        g_val = sum(1 for d in secondary_dets if d['label'] == filter_class)
                    data_m2.append(("General", g_val))
                    
                    # Barras de zonas para la clase filtrada en M2
                    for i in range(zones_count):
                        label = f"Z{i+1}"
                        if ax_y == "cumulative":
                            val = getattr(app, 'session_zone_data_m2', {}).get(i, {"counts": Counter()})["counts"].get(filter_class, 0)
                        elif ax_y == "conf":
                            target_dets = [d for d in secondary_dets if d['label'] == filter_class and i in d.get("zone_indices", [])]
                            val = (sum(d.get('confidence', 0) for d in target_dets) / len(target_dets)) if target_dets else 0
                        else:
                            val = sum(1 for d in secondary_dets if d['label'] == filter_class and i in d.get("zone_indices", []))
                        data_m2.append((label, val))
                else:
                    # Sin filtrar M2
                    for i in range(zones_count):
                        label = f"Z{i+1}"
                        if ax_y == "cumulative":
                            val = 0
                            if hasattr(app, 'session_zone_data_m2') and i in app.session_zone_data_m2:
                                val = sum(app.session_zone_data_m2[i]["counts"].values())
                        else:
                            val = sum(1 for d in secondary_dets if i in d.get("zone_indices", []))
                        data_m2.append((label, val))
            
            elif ax_x == "time":
                if getattr(app, 'chart_history', []):
                    data_m2 = [(time.strftime("%M:%S", time.localtime(t)), sum(1 for d in dets if d.get("source") == "secondary")) for t, dets, zc in app.chart_history]
            
            # Dibujar divisor Cyberpunk en el medio
            canvas.create_line(w_half, 10, w_half, h - 25, fill="#334155", width=1, dash=(2, 2))
            
            # Renderizar M1 (Izquierda)
            title_m1 = f"M1: {app.detector.active_name or 'Primario'}"
            if filter_class != "Todas":
                title_m1 += f" ({filter_class.upper()})"
            VisualPainter._render_sub_chart(app, canvas, 0, w_half - 10, h, title_m1, data_m1, ctype, "primary", ax_y)
            
            # Renderizar M2 (Derecha)
            title_m2 = f"M2: {app.detector.secondary_name or 'Secundario'}"
            if filter_class != "Todas":
                title_m2 += f" ({filter_class.upper()})"
            VisualPainter._render_sub_chart(app, canvas, w_half + 10, w_half - 10, h, title_m2, data_m2, ctype, "secondary", ax_y)
            
            # Footer general de comparativa
            x_label_map = {"class": "CLASES", "zone": "ZONAS", "time": "TIEMPO"}
            y_label_map = {"count": "CANTIDAD", "cumulative": "ACUMULADO", "conf": "CONFIANZA %"}
            footer_text = f"COMPARATIVA DUAL: {x_label_map.get(ax_x, ax_x).upper()} vs {y_label_map.get(ax_y, ax_y).upper()}"
            if filter_class != "Todas":
                footer_text += f" [FILTRADO: {filter_class.upper()}]"
            canvas.create_text(w/2, h-10, text=footer_text, fill="#475569", font=("Arial", 8, "bold"))
            
        else:
            # Modo Simple: renderizar un unico grafico a pantalla completa
            data = []
            if ax_x == "class":
                source = app.session_class_counts if ax_y == "cumulative" else Counter([d['label'] for d in detections])
                if ax_y == "conf":
                    conf_sums = Counter()
                    conf_counts = Counter()
                    for d in detections:
                        conf_sums[d['label']] += d.get('confidence', 0)
                        conf_counts[d['label']] += 1
                    data = [(k, conf_sums[k]/conf_counts[k]) for k in conf_sums]
                else:
                    data = sorted(source.items(), key=lambda x: x[1], reverse=True)[:6]
            
            elif ax_x == "zone":
                zones_count = len(app.zones) if hasattr(app, 'zones') else 0
                if filter_class != "Todas":
                    # Barra General
                    if ax_y == "cumulative":
                        g_val = app.session_class_counts.get(filter_class, 0)
                    elif ax_y == "conf":
                        target_dets = [d for d in detections if d['label'] == filter_class]
                        g_val = (sum(d.get('confidence', 0) for d in target_dets) / len(target_dets)) if target_dets else 0
                    else:
                        g_val = sum(1 for d in detections if d['label'] == filter_class)
                    data.append(("General", g_val))
                    
                    # Barras de zonas
                    for i in range(zones_count):
                        label = f"Z{i+1}"
                        if ax_y == "cumulative":
                            val = app.session_zone_data.get(i, {"counts": Counter()})["counts"].get(filter_class, 0)
                        elif ax_y == "conf":
                            target_dets = [d for d in detections if d['label'] == filter_class and i in d.get("zone_indices", [])]
                            val = (sum(d.get('confidence', 0) for d in target_dets) / len(target_dets)) if target_dets else 0
                        else:
                            val = sum(1 for d in detections if d['label'] == filter_class and i in d.get("zone_indices", []))
                        data.append((label, val))
                else:
                    # Sin filtrar
                    for i in range(zones_count):
                        label = f"Z{i+1}"
                        if ax_y == "cumulative":
                            val = len(app.session_zone_data.get(i, {"ids": set()})["ids"])
                        else:
                            val = sum(1 for d in detections if i in d.get("zone_indices", []))
                        data.append((label, val))
            
            elif ax_x == "time":
                if not getattr(app, 'chart_history', []): 
                    canvas.create_text(w/2, h/2, text="Esperando datos temporales...", fill="#444", font=("Arial", 10))
                    return
                data = [(time.strftime("%M:%S", time.localtime(t)), len(dets)) for t, dets, zc in app.chart_history]
            
            title = f"MODELO: {app.detector.active_name or 'Primario'}"
            if filter_class != "Todas":
                title += f" ({filter_class.upper()})"
            VisualPainter._render_sub_chart(app, canvas, 0, w, h, title, data, ctype, "primary", ax_y)
            
            # Footer general
            x_label_map = {"class": "CLASES", "zone": "ZONAS", "time": "TIEMPO"}
            y_label_map = {"count": "CANTIDAD", "cumulative": "ACUMULADO", "conf": "CONFIANZA %"}
            footer_text = f"ANÁLISIS: {x_label_map.get(ax_x, ax_x).upper()} vs {y_label_map.get(ax_y, ax_y).upper()}"
            canvas.create_text(w/2, h-10, text=footer_text, fill="#475569", font=("Arial", 8, "bold"))

    @staticmethod
    def _render_sub_chart(app, canvas, x_offset, width, height, title, data, ctype, model_type, ax_y):
        """Renderiza un grafico especifico en una subseccion horizontal del canvas."""
        if not data:
            cx = x_offset + width / 2
            cy = height / 2
            canvas.create_text(cx, cy - 10, text=title.upper(), fill="#64748b" if model_type == "primary" else "#f59e0b", font=("Arial", 8, "bold"))
            canvas.create_text(cx, cy + 10, text="ESPERANDO DETECCIONES...", fill="#475569", font=("Arial", 7, "bold"))
            return

        max_val = max([v for l, v in data]) if data else 1
        if max_val == 0: max_val = 1
        
        padding = 35
        
        # Dibujar titulo de la sub-grafica
        title_color = "#38bdf8" if model_type == "primary" else "#fb923c"
        canvas.create_text(x_offset + width / 2, 15, text=title.upper(), fill=title_color, font=("Arial", 9, "bold"))
        
        # Eje X o Base
        y_baseline = height - 28
        canvas.create_line(x_offset + padding, y_baseline, x_offset + width - padding, y_baseline, fill="#334155", width=1)
        
        if ctype == "vbar":
            bar_w = (width - (padding * 2)) / len(data)
            for i, (label, val) in enumerate(data):
                bar_h = (val / max_val) * (height - 65)
                x0 = x_offset + padding + (i * bar_w) + 4
                y0 = height - 30 - bar_h
                x1 = x0 + bar_w - 8
                y1 = height - 30
                
                # Colores temáticos
                fill_col = "#0369a1" if model_type == "primary" else "#b45309"
                out_col = "#0ea5e9" if model_type == "primary" else "#ffa500"
                highlight_col = "#38bdf8" if model_type == "primary" else "#fb923c"
                
                # Barra con relieve
                canvas.create_rectangle(x0, y0, x1, y1, fill=fill_col, outline=out_col, width=1)
                canvas.create_rectangle(x0+1, y0+1, x1-1, y1, fill=highlight_col, outline="", width=0)
                
                val_str = f"{val:.1f}" if ax_y == "conf" else str(int(val))
                canvas.create_text((x0+x1)/2, y0-8, text=val_str, fill="#f8fafc", font=("Arial", 8, "bold"))
                canvas.create_text((x0+x1)/2, height-18, text=label[:8].upper(), fill="#64748b", font=("Arial", 7))
                
        elif ctype == "hbar":
            bar_h = (height - padding - 25) / len(data)
            for i, (label, val) in enumerate(data):
                bar_w = (val / max_val) * (width - 100)
                y0 = padding + (i * bar_h) + 2
                x0 = x_offset + 55
                y1 = y0 + bar_h - 4
                x1 = x0 + bar_w
                
                fill_col = "#0284c7" if model_type == "primary" else "#ea580c"
                canvas.create_rectangle(x0, y0, x1, y1, fill=fill_col, outline=fill_col, width=1)
                canvas.create_text(x_offset + 25, (y0+y1)/2, text=label[:6].upper(), fill="#94a3b8", font=("Arial", 7))
                val_str = f"{val:.1f}" if ax_y == "conf" else str(int(val))
                canvas.create_text(x1+12, (y0+y1)/2, text=val_str, fill="#64748b", font=("Arial", 7))

        elif ctype == "line":
            pts = []
            step_x = (width - (padding * 2)) / max(len(data)-1, 1)
            for i, (label, val) in enumerate(data):
                px = x_offset + padding + (i * step_x)
                py = height - 35 - (val / max_val) * (height - 70)
                pts.append((px, py))
            
            line_col = "#0ea5e9" if model_type == "primary" else "#ffa500"
            bg_col = "#0c4a6e" if model_type == "primary" else "#7c2d12"
            
            if len(pts) > 1:
                # Area translúcida
                poly_pts = [x_offset + padding, height-30] + [p for pt in pts for p in pt] + [x_offset + width - padding, height-30]
                canvas.create_polygon(poly_pts, fill=bg_col, outline="")
                canvas.create_line(pts, fill=line_col, width=2, smooth=True)
                
                # Ultimo punto destacado
                lx, ly = pts[-1]
                canvas.create_oval(lx-3, ly-3, lx+3, ly+3, fill=line_col, outline="#fff")
                val_str = f"{data[-1][1]:.1f}" if ax_y == "conf" else str(int(data[-1][1]))
                canvas.create_text(lx, ly-12, text=val_str, fill=line_col, font=("Arial", 8, "bold"))
            
            canvas.create_text(x_offset + padding, height-18, text=str(data[0][0])[:6], fill="#475569", font=("Arial", 7))
            canvas.create_text(x_offset + width - padding, height-18, text=str(data[-1][0])[:6], fill="#475569", font=("Arial", 7))

        elif ctype == "area":
            pts = []
            step_x = (width - (padding * 2)) / max(len(data)-1, 1)
            for i, (label, val) in enumerate(data):
                px = x_offset + padding + (i * step_x)
                py = height - 35 - (val / max_val) * (height - 70)
                pts.append((px, py))

            line_col = "#38bdf8" if model_type == "primary" else "#fb923c"
            fill_cols = ["#0c4a6e", "#0369a1", "#0284c7"] if model_type == "primary" else ["#7c2d12", "#9a3412", "#ea580c"]
            
            if len(pts) > 1:
                base_y = height - 30
                num_layers = len(fill_cols)
                for li, gc in enumerate(fill_cols):
                    offset = (num_layers - li) * 2
                    shifted_pts = [(px, min(py + offset, base_y)) for px, py in pts]
                    poly = [(x_offset + padding, base_y)]
                    poly.extend(shifted_pts)
                    poly.append((pts[-1][0], base_y))
                    flat = [coord for pt in poly for coord in pt]
                    canvas.create_polygon(flat, fill=gc, outline="")

                canvas.create_line(pts, fill=line_col, width=2, smooth=True)
                for i, (px, py) in enumerate(pts):
                    canvas.create_oval(px-2, py-2, px+2, py+2, fill=line_col, outline="")
                lx, ly = pts[-1]
                val_str = f"{data[-1][1]:.1f}" if ax_y == "conf" else str(int(data[-1][1]))
                canvas.create_text(lx, ly-12, text=val_str, fill=line_col, font=("Arial", 8, "bold"))
            
            canvas.create_text(x_offset + padding, height-18, text=str(data[0][0])[:6], fill="#475569", font=("Arial", 7))
            canvas.create_text(x_offset + width - padding, height-18, text=str(data[-1][0])[:6], fill="#475569", font=("Arial", 7))

        elif ctype == "radar":
            import math
            cx_r = x_offset + width // 2
            cy_r = height // 2 + 10
            radius = min(width // 2 - padding, height // 2 - 20) - 5
            n = len(data)
            if n < 3:
                canvas.create_text(cx_r, cy_r, text="RADAR: Min 3 clases", fill="#475569", font=("Arial", 8, "bold"))
            else:
                angle_step = 2 * math.pi / n
                for ring in [0.5, 1.0]:
                    r = radius * ring
                    ring_pts = []
                    for i in range(n):
                        angle = -math.pi / 2 + i * angle_step
                        rx = cx_r + r * math.cos(angle)
                        ry = cy_r + r * math.sin(angle)
                        ring_pts.append((rx, ry))
                    flat_ring = [coord for pt in ring_pts for coord in pt]
                    canvas.create_polygon(flat_ring, fill="", outline="#1e293b", width=1)
                
                data_pts = []
                line_col = "#0ea5e9" if model_type == "primary" else "#ffa500"
                bg_col = "#0c4a6e" if model_type == "primary" else "#7c2d12"
                
                for i, (label, val) in enumerate(data):
                    angle = -math.pi / 2 + i * angle_step
                    r = (val / max_val) * radius
                    dx = cx_r + r * math.cos(angle)
                    dy = cy_r + r * math.sin(angle)
                    data_pts.append((dx, dy))
                    
                    # Eje y etiqueta
                    ex = cx_r + radius * math.cos(angle)
                    ey = cy_r + radius * math.sin(angle)
                    canvas.create_line(cx_r, cy_r, ex, ey, fill="#334155", width=1)
                    lx = cx_r + (radius + 8) * math.cos(angle)
                    ly = cy_r + (radius + 8) * math.sin(angle)
                    canvas.create_text(lx, ly, text=label[:5].upper(), fill="#94a3b8", font=("Arial", 6))
                
                flat_data = [coord for pt in data_pts for coord in pt]
                canvas.create_polygon(flat_data, fill=bg_col, outline=line_col, width=1.5)

        elif ctype == "scatter":
            # Dispersion
            step_x = (width - padding * 2) / max(len(data), 1)
            for i, (label, val) in enumerate(data):
                bx = x_offset + padding + step_x * (i + 0.5)
                by = height - 35 - (val / max_val) * (height - 70)
                
                color = "#10b981" if model_type == "primary" else "#fb923c"
                canvas.create_oval(bx-4, by-4, bx+4, by+4, fill=color, outline="#fff", width=1)
                val_str = f"{val:.1f}" if ax_y == "conf" else str(int(val))
                canvas.create_text(bx, by-10, text=val_str, fill="#fff", font=("Arial", 7, "bold"))
                canvas.create_text(bx, height-18, text=label[:6].upper(), fill="#64748b", font=("Arial", 7))

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
        # Separar detecciones normales de las de clasificación global
        classifications = [d for d in detections if d.get("is_classification", False)]
        regular_detections = [d for d in detections if not d.get("is_classification", False)]

        # Capa para cajas translucidas de detecciones regulares
        box_overlay = frame.copy()
        
        for d in regular_detections:
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
            
            if dual_mode and source == "secondary":
                # Diseño Cyberpunk de Brackets para M2 (Secundario)
                bracket_len = max(8, min(20, int((x2 - x1) * 0.2)))
                cv2.line(frame, (x1, y1), (x1 + bracket_len, y1), color, 2)
                cv2.line(frame, (x1, y1), (x1, y1 + bracket_len), color, 2)
                cv2.line(frame, (x2, y1), (x2 - bracket_len, y1), color, 2)
                cv2.line(frame, (x2, y1), (x2, y1 + bracket_len), color, 2)
                cv2.line(frame, (x1, y2), (x1 + bracket_len, y2), color, 2)
                cv2.line(frame, (x1, y2), (x1, y2 - bracket_len), color, 2)
                cv2.line(frame, (x2, y2), (x2 - bracket_len, y2), color, 2)
                cv2.line(frame, (x2, y2), (x2, y2 - bracket_len), color, 2)
                
                # Retícula de enfoque central flotante (+)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.line(frame, (cx - 4, cy), (cx + 4, cy), color, 1)
                cv2.line(frame, (cx, cy - 4), (cx, cy + 4), color, 1)
                
                # Etiqueta minimalista para M2
                font = cv2.FONT_HERSHEY_SIMPLEX
                f_scale = 0.45
                thick_txt = 1
                tag = f"{label.upper()} [M2] {conf:.2f}"
                (tw, th), baseline = cv2.getTextSize(tag, font, f_scale, thick_txt)
                
                label_y = y1 - 8
                if label_y < 15:
                    label_y = y2 + th + 8
                    
                # Fondo negro sutil semi-transparente
                lbl_bg = frame.copy()
                cv2.rectangle(lbl_bg, (x1, label_y - th - 6), (x1 + tw + 10, label_y + 4), (15, 23, 42), -1)
                cv2.addWeighted(lbl_bg, 0.7, frame, 0.3, 0, frame)
                
                # Borde fino de color secundario y texto
                cv2.rectangle(frame, (x1, label_y - th - 6), (x1 + tw + 10, label_y + 4), color, 1)
                cv2.putText(frame, tag, (x1 + 5, label_y - 1), font, f_scale, (248, 250, 252), thick_txt)
            else:
                # --- M1 o Modo Simple (Bounding Box Completo) ---
                # Dibujar Trayectoria (Trail) — solo para primario con tracking
                if show_trails and t_id is not None:
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    if t_id not in VisualPainter._track_history:
                        VisualPainter._track_history[t_id] = []
                    VisualPainter._track_history[t_id].append(center)
                    if len(VisualPainter._track_history[t_id]) > 30:
                        VisualPainter._track_history[t_id].pop(0)
                    
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

        # --- Renderizar HUD de Clasificación Global ---
        if classifications:
            # Color verde esmeralda para el escaneo de clasificación
            color_scan = (129, 185, 16) # Emerald Green in BGR (16, 185, 129) -> wait, BGR is (129, 185, 16) or (16, 185, 129) in RGB.
            # BGR: Blue=16, Green=185, Red=129 -> (16, 185, 129)
            color_scan_bgr = (16, 185, 129)
            
            # Dibujar líneas de escaneo en los bordes para dar sensación cybernetic-hud
            cx1, cy1 = int(w * 0.1), int(h * 0.1)
            cx2, cy2 = int(w * 0.9), int(h * 0.9)
            
            # Crear efecto de marco de foco punteado
            for i in range(cx1, cx2, 30):
                cv2.line(frame, (i, cy1), (min(i+15, cx2), cy1), color_scan_bgr, 1)
                cv2.line(frame, (i, cy2), (min(i+15, cx2), cy2), color_scan_bgr, 1)
            for i in range(cy1, cy2, 30):
                cv2.line(frame, (cx1, i), (cx1, min(i+15, cy2)), color_scan_bgr, 1)
                cv2.line(frame, (cx2, i), (cx2, min(i+15, cy2)), color_scan_bgr, 1)
            
            # Tarjeta de estado flotante
            card_x1, card_y1 = 15, 60 if is_focus else 15
            card_x2, card_y2 = 360, card_y1 + 80
            
            # Fondo de la tarjeta (Slate 900 con transparencia del 85%)
            card_bg = frame.copy()
            cv2.rectangle(card_bg, (card_x1, card_y1), (card_x2, card_y2), (30, 20, 15), -1) # Slate-900 BGR
            cv2.addWeighted(card_bg, 0.8, frame, 0.2, 0, frame)
            
            # Borde de la tarjeta
            cv2.rectangle(frame, (card_x1, card_y1), (card_x2, card_y2), color_scan_bgr, 1)
            
            # Esquinas resaltadas (cyber-punk style)
            ac = 8
            for tx, ty in [(card_x1, card_y1), (card_x2, card_y1), (card_x1, card_y2), (card_x2, card_y2)]:
                dx = ac if tx == card_x1 else -ac
                dy = ac if ty == card_y1 else -ac
                cv2.line(frame, (tx, ty), (tx + dx, ty), color_scan_bgr, 3)
                cv2.line(frame, (tx, ty), (tx, ty + dy), color_scan_bgr, 3)
                
            # Textos informativos
            cv2.putText(frame, "IA CLASIFICACION GLOBAL (HUD)", (card_x1 + 15, card_y1 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_scan_bgr, 1)
            
            for idx, c_det in enumerate(classifications[:2]):
                lbl = c_det["label"].upper()
                conf = c_det["confidence"]
                source_lbl = " [M1]" if c_det.get("source") == "primary" else " [M2]"
                txt = f"{idx+1}. {lbl}{source_lbl} -> {conf * 100:.1f}%"
                cv2.putText(frame, txt, (card_x1 + 15, card_y1 + 45 + idx * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

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

