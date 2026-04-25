# 🛰️ Visión AI Engine — Resumen Maestro del Proyecto v4.0

Este documento constituye la **fuente de verdad definitiva** para el proyecto "02 Proyecto Vision Streaming". Refleja la arquitectura modular profesional y todas las capacidades implementadas.

---

## 🏗️ 1. Arquitectura de Software (Modular)
El proyecto se ha reestructurado como un paquete Python profesional dentro de la carpeta `app/`, eliminando la estructura plana anterior para mejorar la mantenibilidad.

### 📂 Estructura de Paquetes
*   `app/core/`: Motores lógicos (Detección, Captura de Vídeo, Eventos, Hardware, Validación).
*   `app/ui/`: Interfaz gráfica basada en CustomTkinter (Dashboard, Anotador, Ajustes, Eventos, Explorador de Modelos).
*   `app/utils/`: Utilidades transversales (Pintado visual, Base de Datos SQLite, Logger CSV, Helpers, Gestor de Errores).

---

## 🚀 2. Capacidades Avanzadas (Implementadas)

### 🧠 Motor de Detección
*   **Soporte Multi-Arquitectura**: YOLO (v8/v11), RT-DETR, YOLO-World (Zero-Shot) y modelos personalizados (.pt).
*   **Object Tracking (ByteTrack)**: Seguimiento de IDs únicos con gestión de pérdida de objetivo y reidentificación.
*   **Focus Mode**: Clic sobre un objeto para seguirlo exclusivamente, aislando su detección del resto de la escena.
*   **Aceleración Automática**: Detección de GPU (NVIDIA CUDA / Intel OpenVINO / AMD DirectML) con fallback seguro a CPU.

### 📊 Dashboard Interactivo
*   **Gráfica de Barras Interactiva**: Distribución de clases en tiempo real con filtrado por zona (General/Z1/Z2...) y selección de clase individual mediante clic.
*   **Contador Único de Sesión**: Conteo preciso basado en IDs de tracking para evitar duplicados.
*   **Top 5 Breakdown**: Las 5 clases más detectadas con su conteo de objetos únicos.
*   **Reloj de Uptime**: Tiempo activo de la sesión actualizado en directo.
*   **Galería de Evidencias**: Miniaturas en tiempo real de las capturas de eventos activados.

### 📊 Analítica Persistente
*   **SQLite (DBManager)**: Registro de cada detección y evento en `telemetry_logs/vision_analytics.db` para consultas históricas.
*   **CSV (DataLogger)**: Registro en tiempo real de conteos agregados por clase y zona.
*   **Evidencia Visual Automática**: Frames guardados en `telemetry_logs/evidences/` al activarse un hito, vinculados a la base de datos.

### 🏷️ Herramientas de Etiquetado
*   **Anotador Visual Multi-Clase**: Interfaz para dibujar bounding boxes con soporte para múltiples clases en formato YOLO estándar.
*   **Atajos de Teclado**: Teclas `1`-`9` y `0` para selección rápida de las primeras 10 clases.
*   **Navegación Rápida**: Teclas `A`/`D` o flechas izquierda/derecha para navegar entre imágenes en modo multi-imagen.
*   **Borrado Selectivo**: Clic derecho sobre una caja para eliminarla individualmente. Clic fuera deshace la última.
*   **Importación ZIP**: Importa paquetes de imágenes (.zip) para etiquetado secuencial masivo.
*   **Autocaptura Periódica**: Captura automática de frames a intervalos configurables para generar datasets sin intervención.
*   **Conteo de Dataset**: Panel lateral que muestra en tiempo real cuántas etiquetas de cada clase tiene el dataset actual.
*   **Exportación ZIP**: Genera un archivo ZIP del dataset listo para entrenamiento.
*   **Guardado Automático al Finalizar**: El botón "Finalizar" guarda la última imagen y abre la carpeta del dataset.

### 📸 Evidencia Visual Automática
Cuando un hito/evento se activa, el sistema guarda automáticamente un frame de evidencia en `telemetry_logs/evidences/`. Estas imágenes se vinculan en la base de datos y pueden ser enviadas opcionalmente vía Telegram.

### 🔊 Síntesis de Voz (TTS)
Integración de anuncios por voz mediante `pyttsx3`. El sistema puede anunciar eventos críticos por los altavoces (ej: *"Persona detectada en Zona de Peligro"*), mejorando la accesibilidad y la respuesta en tiempo real.

### 📈 Seguimiento de Trayectorias (Trails)
El motor visual (`VisualPainter`) dibuja **estelas de movimiento** para cada objeto con Tracking ID. Estas trayectorias están codificadas por colores y permiten visualizar el flujo de movimiento de los elementos en escena.

### 🔔 Hitos Dinámicos y Flexibles
El sistema de eventos ha sido rediseñado para permitir una configuración total por parte del usuario:
- **Operadores lógicos**: `>`, `<`, `==`, `>=`, `<=`, `Total >`.
- **Acciones granulares**: Registro local, Telegram (con foto), Webhooks, Voz (TTS) o todas a la vez.
- **Control de Cooldown**: Evita la saturación de alertas con tiempos de espera configurables.
- **Validación Secundaria**: Doble factor IA con YOLO-World, Segmentación, Ollama o HuggingFace.

### 🐳 Contenerización (Docker)
Incluye un `Dockerfile` optimizado con todas las dependencias de sistema (OpenCV, Tkinter, OpenVINO) para facilitar el despliegue en entornos aislados.

---

## 🎨 3. Interfaz de Usuario (UI/UX)
*   **Pantalla de Carga (Splash)**: Barra de progreso con etapas de inicialización para feedback visual inmediato.
*   **Sidebar Inteligente**: HUD compacto que agrupa fuente, modelos, análisis, datasets, autocaptura y zonas.
*   **Dashboard de Analítica**: Canvas en tiempo real con gráficas interactivas, telemetría de sesión y galería de evidencias.
*   **Controles de Reproducción**: Pausa, retroceso y avance para vídeos locales y YouTube VOD, con indicador de "EN DIRECTO".
*   **Temas**: Alternancia entre modo Oscuro y modo Claro.
*   **Diagnóstico de Hardware**: Panel inferior que muestra en tiempo real la GPU detectada y el backend de inferencia activo.

---

## 🛠️ 4. Componentes Clave

| Componente | Ruta | Función |
| :--- | :--- | :--- |
| **Orquestador** | `main.py` | Punto de entrada con manejo de errores de arranque. |
| **Detector** | `app/core/detector.py` | Inferencia YOLO/RT-DETR, prompts YOLO-World y modelos custom. |
| **Engine** | `app/core/engine.py` | Adquisición de vídeo multi-hilo (CamGear + OpenCV). |
| **Eventos** | `app/core/events.py` | Evaluación de reglas, alertas (Telegram/Webhooks/TTS) y SQLite. |
| **Hardware** | `app/core/hardware.py` | Diagnóstico de GPU y selección automática de backend. |
| **Validator** | `app/core/validator.py` | Validación secundaria vía VLMs (YOLO-World, Segmentación, Ollama). |
| **Main Window** | `app/ui/main_window.py` | Dashboard principal con renderizado optimizado. |
| **Components** | `app/ui/components.py` | Ventanas modales (Anotador, Filtros, Info, Modelos, Fuentes). |
| **Events Window** | `app/ui/events_window.py` | Configuración de reglas de hitos y eventos. |
| **Settings** | `app/ui/settings_window.py` | Ajustes de Telegram, Webhooks y credenciales. |
| **Painter** | `app/utils/painter.py` | Renderizado de cajas, zonas, heatmaps, gráficas y trayectorias. |
| **Logger** | `app/utils/logger.py` | Telemetría CSV con escritura debounced. |
| **DB Manager** | `app/utils/db_manager.py` | Persistencia SQLite para analítica histórica. |
| **Helpers** | `app/utils/helpers.py` | Rutas, constantes y configuración JSON. |
| **Error Handler** | `app/utils/error_handler.py` | Códigos de error centralizados y logging. |

---

## 🗺️ 5. Hoja de Roadmap Actualizado
1.  **Dashboard Web**: Crear una API FastAPI que consuma la base de datos SQLite para mostrar gráficas en un navegador.
2.  **Control Inbound**: Permitir que el bot de Telegram responda a comandos (ej: "/status" o "/screenshot").
3.  **Filtrado por Trayectoria**: Disparar eventos solo si un objeto cruza una línea en una dirección específica (Cross-line counting).
4.  **Exportación Selectiva** [COMPLETADO]: El sistema ahora permite exportar únicamente las imágenes que han sido efectivamente etiquetadas, ignorando los descartes y optimizando el dataset para entrenamiento.
