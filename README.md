# 👁️ AI VisionSandbox Lab

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLOv11-0062FF?style=for-the-badge&logo=google-cloud&logoColor=white)
![CustomTkinter](https://img.shields.io/badge/UI-CustomTkinter-blueviolet?style=for-the-badge)
![SQLite](https://img.shields.io/badge/DB-SQLite-003B57?style=for-the-badge&logo=sqlite&logoColor=white)

**AI VisionSandbox Lab** es un motor de análisis de video en tiempo real diseñado para transformar cualquier stream (YouTube, cámaras IP o Webcams) en una fuente de datos estructurada. Utiliza algoritmos de visión computacional de última generación para detectar, seguir y reaccionar ante eventos específicos mediante reglas inteligentes.

---

## 🚀 Funcionalidades Principales

> [!NOTE]
> **Filosofía del Proyecto:** Esta aplicación ha sido desarrollada para uso personal y sin ánimo de lucro. Su arquitectura está diseñada para ser extremadamente fácil de ejecutar ("Plug & Play") sin requerir configuraciones complejas de servidores o bases de datos externas.

### 🧠 Inteligencia de Detección
- **Arquitectura Modular:** Soporte dinámico para modelos YOLOv8, YOLOv11, RT-DETR y arquitecturas personalizadas (.pt).
- **YOLO-World Integration:** Detección "Zero-Shot" mediante prompts de texto (ej: detecta "un paraguas rojo" sin entrenar el modelo).
- **Object Tracking Avanzado:** Seguimiento de IDs únicos mediante **ByteTrack** para evitar conteos duplicados y mejorar la precisión.
- **Aceleración de Hardware:** Detección automática de GPU y selección inteligente de backend (CUDA / OpenVINO / DirectML / CPU).

### 📐 Control Espacial y Eventos
- **Multizona (Superposiciones):** Sistema de zonas poligonales dibujables donde un objeto puede activar múltiples reglas simultáneamente.
- **Motor de Hitos (Milestones):** Configuración de reglas lógicas complejas:
  - `Individual:` Activación si hay > X objetos en una zona.
  - `Acumulado:` Activación basada en el histórico total detectado en la sesión.
- **Validación Secundaria (Doble Factor):** Capacidad de usar un segundo modelo de IA (YOLO-World, Segmentación, Ollama o HuggingFace) para validar detecciones críticas antes de disparar una alerta.

### 📊 Dashboard Interactivo
- **Gráfica de Barras por Zonas:** Visualización en tiempo real de la distribución de clases con filtrado por zona (General / Z1 / Z2...). Las barras son interactivas: haz clic para filtrar una clase específica.
- **Contador Único de Sesión:** Conteo preciso basado en IDs de tracking, evitando contar el mismo objeto dos veces.
- **Top 5 Breakdown:** Desglose en vivo de las 5 clases más detectadas con conteo de IDs únicos.
- **Reloj de Uptime:** Tiempo activo de la sesión de monitorización actualizado en directo.
- **Galería de Evidencias:** Miniaturas en tiempo real de las capturas de eventos activados.

### 🔔 Notificaciones y Telemetría
- **Alertas de Telegram:** Notificaciones enriquecidas con HTML, incluyendo evidencia fotográfica del evento.
- **Webhooks de Google:** Integración directa con Google Apps Script para automatizar hojas de cálculo.
- **Síntesis de Voz (TTS):** Anuncios por voz de eventos críticos mediante `pyttsx3`.
- **Logging Dual:** Registro CSV en tiempo real + persistencia en base de datos SQLite para analítica histórica.
- **Evidencia Visual Automática:** Captura y almacenamiento de frames cuando se activa un hito.

### 🏷️ Herramientas de Etiquetado y Datasets
- **Anotador Visual Integrado:** Herramienta de bounding boxes con soporte multi-clase y formato YOLO estándar.
- **Atajos de Teclado:** Teclas `1-9` y `0` para selección rápida de clases. `A/D` o flechas para navegar entre imágenes.
- **Borrado Selectivo:** Clic derecho sobre una caja para eliminarla individualmente, o fuera para deshacer la última.
- **Importación de Datasets (ZIP):** Importa paquetes de imágenes para etiquetado secuencial masivo.
- **Autocaptura Periódica:** Captura automática de frames a intervalos configurables para generar datasets sin intervención.
- **Conteo de Dataset:** Panel en tiempo real que muestra cuántas etiquetas de cada clase llevas en el dataset actual.
- **Exportación ZIP:** Exporta datasets completos listos para entrenamiento.

### 🎨 Visualización Avanzada
- **Mapa de Calor (Heatmap):** Superposición de densidad de detecciones sobre el frame.
- **Trayectorias (Trails):** Estelas de movimiento codificadas por color para cada objeto con Tracking ID.
- **Focus Mode:** Clic sobre un objeto para seguirlo exclusivamente y aislar su detección.
- **Controles de Reproducción:** Pausa, retroceso y avance para vídeos locales y YouTube VOD.

---

## 📂 Estructura del Proyecto

```
02 Proyecto Vision Streaming/
├── main.py                     # Punto de entrada
├── train_custom.py             # Script de entrenamiento personalizado
├── requirements.txt            # Dependencias
├── Dockerfile                  # Contenerización
├── .env                        # Secretos (Telegram, Webhooks)
│
├── app/                        # Paquete principal
│   ├── core/                   # Motores lógicos
│   │   ├── detector.py         # Inferencia YOLO / RT-DETR
│   │   ├── engine.py           # Adquisición de vídeo (CamGear + OpenCV)
│   │   ├── events.py           # Motor de hitos, reglas y alertas
│   │   ├── hardware.py         # Diagnóstico de GPU y backends
│   │   └── validator.py        # Validación secundaria (VLMs)
│   │
│   ├── ui/                     # Interfaz gráfica
│   │   ├── main_window.py      # Dashboard principal
│   │   ├── components.py       # Ventanas modales (Anotador, Filtros, Info, Modelos)
│   │   ├── events_window.py    # Configuración de reglas de eventos
│   │   └── settings_window.py  # Ajustes generales (Telegram, Webhooks)
│   │
│   └── utils/                  # Utilidades transversales
│       ├── painter.py          # Renderizado visual (Zonas, Gráficas, Trails, Heatmap)
│       ├── helpers.py          # Configuración, rutas y persistencia JSON
│       ├── logger.py           # Telemetría CSV
│       ├── db_manager.py       # Persistencia SQLite
│       └── error_handler.py    # Códigos de error centralizados
│
├── models/                     # Pesos de modelos (.pt, OpenVINO)
├── datasets/                   # Datasets de entrenamiento (YOLO format)
├── config/                     # Configuración local (Zonas, Eventos, Favoritos)
└── telemetry_logs/             # Logs CSV, SQLite y evidencias
```

---

## 🛠️ Stack Tecnológico

### Librerías Principales
| Librería | Propósito | Enlace |
| :--- | :--- | :--- |
| **Ultralytics** | Motor de Inferencia YOLO / RT-DETR | [GitHub](https://github.com/ultralytics/ultralytics) |
| **CustomTkinter** | Interfaz de Usuario Moderna | [Docs](https://customtkinter.tomschimansky.com/) |
| **VidGear** | Procesamiento de Video de Alto Rendimiento | [GitHub](https://github.com/abhitronix/vidgear) |
| **OpenCV** | Visión Computacional Core | [Web](https://opencv.org/) |
| **Pillow** | Manipulación de imágenes para UI | [Web](https://python-pillow.org/) |
| **PyTorch** | Backend de Deep Learning (CUDA / CPU) | [Web](https://pytorch.org/) |
| **pyttsx3** | Síntesis de Voz (TTS) offline | [PyPI](https://pypi.org/project/pyttsx3/) |
| **python-dotenv** | Gestión segura de secretos (.env) | [PyPI](https://pypi.org/project/python-dotenv/) |

### Librerías de Soporte
| Librería | Propósito |
| :--- | :--- |
| **yt-dlp** | Resolución de URLs de YouTube y streams |
| **requests** | Comunicación HTTP (Telegram, Webhooks) |
| **LapX** | Motor de Asociación para Tracking (ByteTrack) |
| **NumPy** | Operaciones matriciales y procesamiento de frames |
| **OpenVINO** | Aceleración de inferencia en hardware Intel |
| **SQLite3** | Base de datos analítica local (incluido en Python) |

---

## 📦 Modelos Recomendados

Para empezar a usar la aplicación, puedes descargar los siguientes pesos oficiales de Ultralytics:

- [yolo11n.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt) (Modelo Nano — Máxima velocidad, ideal para CPUs).
- [yolo11s.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt) (Modelo Small — Balance entre precisión y velocidad).

> **Nota:** Coloca tus modelos en la carpeta `/models/<nombre_familia>/` y reinicia el Explorador de Inteligencia de la app.

---

## 🏗️ Mejoras Arquitectónicas (v4.0)

- **Paquete Modular (`app/`):** Código organizado en `core/`, `ui/` y `utils/` para máxima mantenibilidad.
- **Tipado Estricto (Type Hinting):** Todo el núcleo está tipado, mejorando la seguridad en tiempo de ejecución.
- **Renderizado Optimizado:** Uso de `itemconfig` en Canvas para evitar parpadeos y reducir carga de CPU.
- **Adquisición Multi-hilo:** Lectura de frames en hilo separado para mantener la fluidez del vídeo.
- **Detección de Hardware Robusta:** Identificación automática de aceleradores (CUDA, OpenVINO, DirectML) con fallback seguro a CPU.
- **Gestor de Errores Centralizado:** Códigos de error estandarizados para diagnóstico rápido.

---

## 🚀 Inicio Rápido (Desbloqueo de Entorno)

Para evitar errores de tipo `ModuleNotFoundError` (como `customtkinter`), asegúrate de ejecutar el proyecto utilizando el entorno virtual pre-configurado:

### Windows (PowerShell)
```powershell
.\venv\Scripts\python main.py
```

### Windows (CMD)
```cmd
venv\Scripts\python main.py
```

---

## 🤖 Créditos y Desarrollo

Este proyecto ha sido desarrollado en colaboración con **Antigravity**, un potente agente de IA diseñado por el equipo de **Advanced Agentic Coding** de **Google DeepMind**.

---

## ⚖️ Licencia

Este proyecto está bajo la **Licencia MIT**. Puedes usarlo, modificarlo y distribuirlo libremente, siempre que mantengas la atribución original.

Copyright (c) 2026 Jose L.
