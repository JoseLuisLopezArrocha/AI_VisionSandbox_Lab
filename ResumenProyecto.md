# 🛰️ Visión AI Engine - Resumen Maestro del Proyecto v2.5

Este documento constituye la **fuente de verdad definitiva** para el proyecto "02 Proyecto Vision Streaming". Está diseñado para permitir que cualquier desarrollador o agente de IA comprenda la arquitectura, el flujo de datos y la responsabilidad de cada componente sin necesidad de auditar el código fuente completo.

---

## 📖 1. Visión General del Proyecto
**Visión AI Engine** es un Dashboard de monitorización avanzado que integra visión artificial en tiempo real sobre streams de vídeo (YouTube, RTSP, archivos locales). El sistema no solo detecta objetos, sino que permite definir zonas analíticas, configurar hitos inteligentes con validación mediante IAs externas (VLMs) y generar telemetría detallada.

---

## 🏗️ 2. Arquitectura de Software
El proyecto sigue un diseño **modular y desacoplado**, separando la lógica de adquisición, el motor de inferencia, la gestión de eventos y la capa de presentación.

### Diagrama de Flujo de Datos (Data Pipeline)
1.  **Adquisición** (`VisionEngine`): Captura de frames asíncrona.
2.  **Procesamiento** (`ObjectDetector`): Inferencia YOLO/RT-DETR + Modelos Custom.
3.  **Lógica de Negocio** (`EventEngine`): Evaluación de reglas y condiciones (hitos).
4.  **Validación** (`SecondaryValidator`): Confirmación vía Ollama/HF si el hito lo requiere.
5.  **Persistencia** (`DataLogger`): Registro ligero de telemetría en CSV (Optimizado para uso personal).
6.  **Visualización** (`VisualPainter`): Dibujado estético y actualización de Dashboard (Main) optimizado para evitar parpadeos y picos de CPU.

---

## 📂 3. Estructura de Archivos y Responsabilidades

### 📁 Directorios Raíz
*   `models/`: Almacén central de pesos (.pt). Organizado por subcarpetas (Familias de modelos).
*   `models/custom/`: Modelos especializados (ej: detección de taxis) que se ejecutan en paralelo al principal.
*   `telemetry_logs/`: Carpeta donde se guardan los archivos CSV diarios de detecciones.
*   `datasets/`: (Opcional) Estructuras para entrenamiento de nuevos modelos.
*   `venv/`: Entorno virtual Python con dependencias aisladas.

### 📄 Archivos de Código (Core)
| Archivo | Responsabilidad Principal |
| :--- | :--- |
| `main.py` | **Orquestador Central**. Levanta el Dashboard, gestiona el hilo de vídeo y coordina todos los módulos. |
| `detector.py` | **Motor de Inferencia**. Carga pesos dinámicamente, filtra clases y maneja arquitecturas YOLOv8-v11 y RT-DETR. |
| `vision_engine.py` | **Driver de Vídeo**. Maneja la conexión con CamGear (YouTube) y VideoCapture (Local). |
| `visual_painter.py` | **Capa Estética**. Dibuja cajas, polígonos de zonas, mapas de calor y gráficos del Dashboard (Canvas). |
| `event_engine.py` | **Cerebro de Eventos**. Evalúa si se cumplen condiciones (ej: >5 personas en Zona 1). |
| `secondary_validator.py`| **Validador IA**. Consulta APIs de Ollama o Hugging Face para validaciones críticas asíncronas. |
| `data_logger.py` | **Grabador de Telemetría**. Escribe en CSV el estado del sistema segundo a segundo (Uso ligero). |
| `vision_utils.py` | **Utilidades**. Constantes de color, rutas de archivos y carga/guardado de configuraciones JSON. |
| `error_handler.py`| **Manejador de Errores**. Centraliza alertas por consola para un diagnóstico rápido y directo. |

### 📄 Componentes de Interfaz (UI)
*   `ui_components.py`: Contiene ventanas modales (`ClassFilterWindow`, `AddModelPopup`, `InfoPopup`).
*   `ui_events.py`: Interfaz de usuario para el gestor de Hitos y validadores secundarios.

### 📄 Configuración y Persistencia
*   `zones.json`: Guarda polígonos de zonas y filtros de clase asociados a cada URL de vídeo.
*   `events_config.json`: Guarda las reglas de eventos configuradas por el usuario.
*   `requirements.txt`: Lista de dependencias (ultralytics, customtkinter, vidgear, opencv).

---

## 🧠 4. Sistema de Inteligencia y Modelos
El proyecto soporta **Familias de Modelos**. Una familia es una carpeta dentro de `models/` que contiene archivos `.pt`.
- El sistema escanea estas carpetas y genera alias automáticamente (ej: "YOL 01").
- **Metadatos**: Cada carpeta puede tener un `metadata.json` para definir si el modelo es COCO o tiene clases personalizadas.
- **Modelos Custom**: Cualquier modelo en `models/custom/` se cargará como "Detector Secundario", funcionando en tándem con el principal.

---

## 📊 5. Análisis y Telemetría
El Dashboard inferior muestra:
- **Gráfico de Barras**: Distribución actual de objetos (Top 6).
- **Gráfico de Línea**: Tendencia de conteo total en los últimos 30 segundos.
- **Métricas de Zona**: Conteo independiente por cada polígono dibujado.
- **Logs de Sistema**: Registro en tiempo real de eventos y acciones de validación.

---

## 🗺️ 6. Hoja de Ruta (Next Steps)
1.  **Acciones Reales**: Implementar el envío efectivo de Webhooks y Emails (actualmente solo loguean).
2.  **Seguimiento de Trayectorias**: Implementar algoritmos de Tracking (DeepSORT/ByteTrack) para medir tiempos de permanencia.
3.  **Exportación PDF**: Generar reportes diarios automáticos basados en los logs CSV.
4.  **Encriptación**: Proteger las API Keys de los validadores secundarios en el JSON.
*(Completado)* **Optimización del Core**: Añadido Type Hinting completo, optimización del Canvas y detección robusta de hardware.

---
> [!TIP]
> **Nota para Agentes**: Al realizar modificaciones, priorizar siempre el mantenimiento de la asincronía en el hilo de vídeo (`update_video` en `main.py`) para evitar congelamientos en el Dashboard.
