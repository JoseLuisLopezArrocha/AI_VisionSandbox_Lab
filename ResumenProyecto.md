# 🛰️ Visión AI Engine - Resumen Maestro del Proyecto v3.0

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

### 🐳 Contenerización (Docker)
Incluye un `Dockerfile` optimizado con todas las dependencias de sistema (OpenCV, Tkinter, OpenVINO) para facilitar el despliegue en entornos aislados.

---

## 🛠️ 3. Componentes Clave (Nomenclatura Actualizada)

| Componente | Ruta | Función |
| :--- | :--- | :--- |
| **Orquestador** | `main.py` | Punto de entrada. Lanza la `VisionApp`. |
| **Detector** | `app/core/detector.py` | Inferencia YOLO/RT-DETR y prompts YOLO-World. |
| **Eventos** | `app/core/events.py` | Evaluación de reglas, disparo de alertas y **SQLite Logging**. |
| **Painter** | `app/utils/painter.py` | Renderizado de cajas, zonas, mapas de calor y **trayectorias**. |
| **DB Manager**| `app/utils/db_manager.py` | **(NUEVO)** Gestión de base de datos SQLite. |
| **Validator** | `app/core/validator.py` | Validación secundaria vía VLMs (Gemma/SAM placeholders). |

---

## ⚙️ 4. Configuración y Secretos
*   `.env`: Almacena `TELEGRAM_TOKEN`, `TELEGRAM_CHAT_ID` y `WEBHOOK_URL`.
*   `requirements.txt`: Incluye `pyttsx3` para voz y `python-dotenv` para secretos.

---

## 🗺️ 5. Hoja de Roadmap Actualizado
1.  **Dashboard Web**: Crear una API FastAPI que consuma la base de datos SQLite para mostrar gráficas en un navegador.
2.  **Control Inbound**: Permitir que el bot de Telegram responda a comandos (ej: "/status" o "/screenshot").
3.  **Filtrado por Trayectoria**: Disparar eventos solo si un objeto cruza una línea en una dirección específica (Cross-line counting).
