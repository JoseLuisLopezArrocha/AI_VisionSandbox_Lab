# 🛰️ Visión AI Engine - Resumen Maestro del Proyecto v3.5

Este documento constituye la **fuente de verdad definitiva** para el proyecto "02 Proyecto Vision Streaming". Refleja la arquitectura modular profesional y las capacidades avanzadas de analítica.

---

## 🏗️ 1. Arquitectura de Software (Modular)
El proyecto se ha reestructurado como un paquete Python profesional dentro de la carpeta `app/`, eliminando la estructura plana anterior para mejorar la mantenibilidad.

### 📂 Estructura de Paquetes
*   `app/core/`: Motores lógicos (Detección, Captura de Vídeo, Eventos, Hardware).
*   `app/ui/`: Interfaz gráfica basada en CustomTkinter (Ventana Principal, Ajustes, Eventos).
*   `app/utils/`: Utilidades transversales (Pintado visual, Base de Datos, Loggers, Helpers).

---

## 🚀 2. Capacidades Avanzadas (Implementadas)

### 📊 Analítica Persistente (SQLite)
A diferencia del CSV básico, el sistema ahora integra un **DBManager** que registra cada detección y evento en una base de datos SQLite local (`telemetry_logs/vision_analytics.db`). Permite realizar consultas históricas complejas sobre conteos, zonas y trackings.

### 📸 Evidencia Visual Automática
Cuando un hito/evento se activa, el sistema guarda automáticamente un frame de evidencia en `telemetry_logs/evidences/`. Estas imágenes se vinculan en la base de datos y pueden ser enviadas opcionalmente vía Telegram.

### 🔊 Síntesis de Voz (TTS)
Integración de anuncios por voz. El sistema puede anunciar eventos críticos por los altavoces (ej: *"Persona detectada en Zona de Peligro"*), mejorando la accesibilidad y la respuesta en tiempo real.

### 📈 Seguimiento de Trayectorias (Trails)
El motor visual (`VisualPainter`) ahora dibuja **estelas de movimiento** para cada objeto con Tracking ID. Estas trayectorias están codificadas por colores y permiten visualizar el flujo de movimiento de los elementos en escena.

### 🔔 Hitos Dinámicos y Flexibles
El sistema de eventos ha sido rediseñado para permitir una configuración total por parte del usuario:
- **Operadores lógicos**: `>`, `<`, `==`, `>=`, `<=`, `Total >`.
- **Acciones granulares**: Registro local, Telegram (con foto), Webhooks o Voz (TTS).
- **Control de Cooldown**: Evita la saturación de alertas con tiempos de espera configurables.

### 🐳 Contenerización (Docker)
Incluye un `Dockerfile` optimizado con todas las dependencias de sistema (OpenCV, Tkinter, OpenVINO) para facilitar el despliegue en entornos aislados.

---

## 🎨 3. Interfaz de Usuario (UI/UX)
*   **Sidebar Inteligente**: HUD compacto que agrupa las herramientas de captura, zonas y análisis en un diseño de una sola columna sin necesidad de scroll.
*   **Dashboard de Analítica**: Canvas en tiempo real para visualización de tendencias y distribución de clases.
*   **Diagnóstico de Hardware**: Panel inferior que muestra en tiempo real la GPU detectada y el backend de inferencia activo (ej: OpenVINO / CPU).

---

## 🛠️ 4. Componentes Clave

| Componente | Ruta | Función |
| :--- | :--- | :--- |
| **Orquestador** | `main.py` | Punto de entrada. Lanza la `VisionApp`. |
| **Detector** | `app/core/detector.py` | Inferencia YOLO/RT-DETR y prompts YOLO-World. |
| **Eventos** | `app/core/events.py` | Evaluación de reglas, disparo de alertas y **SQLite Logging**. |
| **Painter** | `app/utils/painter.py` | Renderizado de cajas, zonas, mapas de calor y **trayectorias**. |
| **DB Manager**| `app/utils/db_manager.py` | Gestión de base de datos SQLite. |
| **Validator** | `app/core/validator.py` | Validación secundaria vía VLMs (Gemma/SAM placeholders). |

---

## 🗺️ 5. Hoja de Roadmap Actualizado
1.  **Dashboard Web**: Crear una API FastAPI que consuma la base de datos SQLite para mostrar gráficas en un navegador.
2.  **Control Inbound**: Permitir que el bot de Telegram responda a comandos (ej: "/status" o "/screenshot").
3.  **Filtrado por Trayectoria**: Disparar eventos solo si un objeto cruza una línea en una dirección específica (Cross-line counting).
