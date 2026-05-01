# AI VisionSandbox Lab

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLOv11-0062FF?style=for-the-badge&logo=google-cloud&logoColor=white)
![CustomTkinter](https://img.shields.io/badge/UI-CustomTkinter-blueviolet?style=for-the-badge)
![SQLite](https://img.shields.io/badge/DB-SQLite-003B57?style=for-the-badge&logo=sqlite&logoColor=white)

**AI VisionSandbox Lab** es un motor de analisis de video en tiempo real disenado para transformar cualquier stream (YouTube, camaras IP o Webcams) en una fuente de datos estructurada. Utiliza algoritmos de vision computacional de ultima generacion para detectar, seguir y reaccionar ante eventos especificos mediante reglas inteligentes.

---

## Funcionalidades Principales

> [!NOTE]
> **Filosofia del Proyecto:** Esta aplicacion ha sido desarrollada para uso personal y sin animo de lucro. Su arquitectura esta disenada para ser extremadamente facil de ejecutar ("Plug & Play") sin requerir configuraciones complejas de servidores o bases de datos externas.

### Inteligencia de Deteccion
- **Arquitectura Modular:** Soporte dinamico para modelos YOLOv8, YOLOv11, RT-DETR y arquitecturas personalizadas (.pt).
- **YOLO-World Integration:** Deteccion "Zero-Shot" mediante prompts de texto (ej: detecta "un paraguas rojo" sin entrenar el modelo).
- **Object Tracking Avanzado:** Seguimiento de IDs unicos mediante **ByteTrack** para evitar conteos duplicados y mejorar la precision.
- **Aceleracion de Hardware:** Deteccion automatica de GPU y seleccion inteligente de backend (CUDA / OpenVINO / DirectML / CPU).

### Control Espacial y Eventos
- **Multizona (Superposiciones):** Sistema de zonas poligonales dibujables donde un objeto puede activar multiples reglas simultaneamente.
- **Motor de Hitos (Milestones):** Configuracion de reglas logicas complejas:
  - `Individual:` Activacion si hay > X objetos en una zona.
  - `Acumulado:` Activacion basada en el historico total detectado en la sesion.
- **Validacion Secundaria (Doble Factor):** Capacidad de usar un segundo modelo de IA (YOLO-World, Segmentacion, Ollama o HuggingFace) para validar detecciones criticas antes de disparar una alerta.

### Dashboard Interactivo
- **Grafica de Barras por Zonas:** Visualizacion en tiempo real de la distribucion de clases con filtrado por zona (General / Z1 / Z2...). Las barras son interactivas: haz clic para filtrar una clase especifica.
- **Contador Unico de Sesion:** Conteo preciso basado en IDs de tracking, evitando contar el mismo objeto dos veces.
- **Top 5 Breakdown:** Desglose en vivo de las 5 clases mas detectadas con conteo de IDs unicos.
- **Reloj de Uptime:** Tiempo activo de la sesion de monitorizacion actualizado en directo.
- **Galeria de Evidencias:** Miniaturas en tiempo real de las capturas de eventos activados.

### Notificaciones y Telemetria
- **Alertas de Telegram:** Notificaciones enriquecidas con HTML, incluyendo evidencia fotografica del evento.
- **Webhooks de Google:** Integracion directa con Google Apps Script para automatizar hojas de calculo.
- **Sintesis de Voz (TTS):** Anuncios por voz de eventos criticos mediante `pyttsx3`.
- **Logging Dual:** Registro CSV en tiempo real + persistencia en base de datos SQLite para analitica historica.
- **Evidencia Visual Automatica:** Captura y almacenamiento de frames cuando se activa un hito.

### Herramientas de Etiquetado y Datasets
- **Anotador Visual Integrado:** Herramienta de bounding boxes con soporte multi-clase y formato YOLO estandar.
- **Atajos de Teclado:** Teclas `1-9` y `0` para seleccion rapida de clases. `A/D` o flechas para navegar entre imagenes.
- **Borrado Selectivo:** Clic derecho sobre una caja para eliminarla individualmente, o fuera para deshacer la ultima.
- **Importacion de Datasets (ZIP):** Importa paquetes de imagenes para etiquetado secuencial masivo.
- **Autocaptura Periodica:** Captura automatica de frames a intervalos configurables para generar datasets sin intervencion.
- **Conteo de Dataset:** Panel en tiempo real que muestra cuantas etiquetas de cada clase llevas en el dataset actual.
- **Exportacion ZIP:** Exporta datasets completos listos para entrenamiento.

### Visualizacion Avanzada
- **Mapa de Calor (Heatmap):** Superposicion de densidad de detecciones sobre el frame.
- **Trayectorias (Trails):** Estelas de movimiento codificadas por color para cada objeto con Tracking ID.
- **Focus Mode:** Clic sobre un objeto para seguirlo exclusivamente y aislar su deteccion.
- **Controles de Reproduccion:** Pausa, retroceso y avance para videos locales y YouTube VOD.

---

## Requisitos Previos

- **Python 3.10 a 3.12** (se recomienda **3.12** para máximo rendimiento)
- **GPU NVIDIA** con CUDA (opcional, mejora rendimiento x10)
- **Conexion a Internet** para streams de YouTube y APIs externas

---

## Estructura del Proyecto

```
02 Proyecto Vision Streaming/
├── main.py                     # Punto de entrada
├── train_custom.py             # Script de entrenamiento personalizado
├── requirements.txt            # Dependencias
├── Dockerfile                  # Contenerizacion
├── .env                        # Secretos (Telegram, Webhooks, API Keys)
│
├── app/                        # Paquete principal
│   ├── core/                   # Motores logicos
│   │   ├── detector.py         # Inferencia YOLO / RT-DETR
│   │   ├── engine.py           # Adquisicion de video (CamGear + OpenCV)
│   │   ├── events.py           # Motor de hitos, reglas y alertas
│   │   ├── hardware.py         # Diagnostico de GPU y backends
│   │   └── validator.py        # Validacion secundaria (VLMs, Ollama, HuggingFace)
│   │
│   ├── ui/                     # Interfaz grafica
│   │   ├── main_window.py      # Dashboard principal
│   │   ├── components.py       # Ventanas modales (Anotador, Filtros, Info, Modelos)
│   │   ├── events_window.py    # Configuracion de reglas de eventos
│   │   └── settings_window.py  # Ajustes generales (Telegram, Webhooks, API Keys IA)
│   │
│   └── utils/                  # Utilidades transversales
│       ├── painter.py          # Renderizado visual (Zonas, Graficas, Trails, Heatmap)
│       ├── helpers.py          # Configuracion, rutas y persistencia JSON
│       ├── logger.py           # Telemetria CSV
│       ├── db_manager.py       # Persistencia SQLite
│       └── error_handler.py    # Codigos de error centralizados
│
├── models/                     # Pesos de modelos (.pt, OpenVINO)
├── datasets/                   # Datasets de entrenamiento (YOLO format)
├── config/                     # Configuracion local (Zonas, Eventos, Favoritos)
└── telemetry_logs/             # Logs CSV, SQLite y evidencias
```

---

## Stack Tecnologico

### Librerias Principales
| Libreria | Proposito | Enlace |
| :--- | :--- | :--- |
| **Ultralytics** | Motor de Inferencia YOLO / RT-DETR | [GitHub](https://github.com/ultralytics/ultralytics) |
| **CustomTkinter** | Interfaz de Usuario Moderna | [Docs](https://customtkinter.tomschimansky.com/) |
| **VidGear** | Procesamiento de Video de Alto Rendimiento | [GitHub](https://github.com/abhitronix/vidgear) |
| **OpenCV** | Vision Computacional Core | [Web](https://opencv.org/) |
| **Pillow** | Manipulacion de imagenes para UI | [Web](https://python-pillow.org/) |
| **PyTorch** | Backend de Deep Learning (CUDA / CPU) | [Web](https://pytorch.org/) |
| **pyttsx3** | Sintesis de Voz (TTS) offline | [PyPI](https://pypi.org/project/pyttsx3/) |
| **python-dotenv** | Gestion segura de secretos (.env) | [PyPI](https://pypi.org/project/python-dotenv/) |

### Librerias de Soporte
| Libreria | Proposito |
| :--- | :--- |
| **yt-dlp** | Resolucion de URLs de YouTube y streams |
| **requests** | Comunicacion HTTP (Telegram, Webhooks, APIs IA) |
| **LapX** | Motor de Asociacion para Tracking (ByteTrack) |
| **NumPy** | Operaciones matriciales y procesamiento de frames |
| **OpenVINO** | Aceleracion de inferencia en hardware Intel |
| **SQLite3** | Base de datos analitica local (incluido en Python) |

---

## Modelos Recomendados

Para empezar a usar la aplicacion, puedes descargar los siguientes pesos oficiales de Ultralytics:

- [yolo11n.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt) (Modelo Nano -- Maxima velocidad, ideal para CPUs).
- [yolo11s.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt) (Modelo Small -- Balance entre precision y velocidad).

> **Nota:** Coloca tus modelos en la carpeta `/models/<nombre_familia>/` y reinicia el Explorador de Inteligencia de la app.

---

## Configuracion

### Archivo .env (Secretos)

Crea un archivo `.env` en la raiz del proyecto con las siguientes variables:

```env
# Telegram Bot API
TELEGRAM_TOKEN=tu_token_aqui
TELEGRAM_CHAT_ID=tu_chat_id_aqui

# Webhooks
WEBHOOK_URL=https://hooks.example.com/...

# Proveedores de IA (Validacion Secundaria)
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=llava
HUGGINGFACE_API_KEY=hf_xxxxxxxxxxxx
HUGGINGFACE_MODEL=Salesforce/blip-vqa-base
```

### Panel de Ajustes (UI)

Todas estas credenciales tambien se pueden configurar desde el boton de **Ajustes** en la barra lateral de la aplicacion, sin necesidad de editar archivos manualmente.

---

## Mejoras Arquitectonicas (v4.0)

- **Paquete Modular (`app/`):** Codigo organizado en `core/`, `ui/` y `utils/` para maxima mantenibilidad.
- **Tipado Estricto (Type Hinting):** Todo el nucleo esta tipado, mejorando la seguridad en tiempo de ejecucion.
- **Renderizado Optimizado:** Uso de `itemconfig` en Canvas para evitar parpadeos y reducir carga de CPU.
- **Adquisicion Multi-hilo:** Lectura de frames en hilo separado para mantener la fluidez del video.
- **Deteccion de Hardware Robusta:** Identificacion automatica de aceleradores (CUDA, OpenVINO, DirectML) con fallback seguro a CPU.
- **Gestor de Errores Centralizado:** Codigos de error estandarizados para diagnostico rapido.
- **Persistencia SQLite Optimizada:** Modo WAL, connection pooling y batch inserts para rendimiento de escritura.

---

## Inicio Rapido

Para evitar errores de tipo `ModuleNotFoundError` (como `customtkinter`), asegurate de ejecutar el proyecto utilizando el entorno virtual pre-configurado:

### Windows (PowerShell)
```powershell
.\venv\Scripts\python main.py
```

### Windows (CMD)
```cmd
venv\Scripts\python main.py
```

---

## Creditos y Desarrollo

Este proyecto ha sido desarrollado en colaboracion con **Antigravity**, un potente agente de IA disenado por el equipo de **Advanced Agentic Coding** de **Google DeepMind**.

---

## Licencia

Este proyecto esta bajo la **Licencia MIT**. Puedes usarlo, modificarlo y distribuirlo libremente, siempre que mantengas la atribucion original.

Copyright (c) 2026 Jose L.
