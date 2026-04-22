# 👁️ AI VisionSandbox Lab

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLOv11-0062FF?style=for-the-badge&logo=google-cloud&logoColor=white)
![CustomTkinter](https://img.shields.io/badge/UI-CustomTkinter-blueviolet?style=for-the-badge)

**AI VisionSandbox Lab** es un motor de análisis de video en tiempo real diseñado para transformar cualquier stream (YouTube, cámaras IP o Webcams) en una fuente de datos estructurada. Utiliza algoritmos de visión computacional de última generación para detectar, seguir y reaccionar ante eventos específicos mediante reglas inteligentes.

---

## 🚀 Funcionalidades Principales

> [!NOTE]
> **Filosofía del Proyecto:** Esta aplicación ha sido desarrollada para uso personal y sin ánimo de lucro. Su arquitectura está diseñada para ser extremadamente fácil de ejecutar ("Plug & Play") sin requerir configuraciones complejas de servidores o bases de datos externas, priorizando la accesibilidad para cualquier usuario.

### 🧠 Inteligencia de Detección
- **Arquitectura Modular:** Soporte dinámico para modelos YOLOv8, YOLOv11 y arquitecturas personalizadas (.pt).
- **YOLO-World Integration:** Detección "Zero-Shot" mediante prompts de texto (ej: detecta "un paraguas rojo" sin entrenar el modelo).
- **Object Tracking Avanzado:** Seguimiento de IDs únicos mediante **ByteTrack** para evitar conteos duplicados y mejorar la precisión.

### 📐 Control Espacial y Eventos
- **Multizona (Superposiciones):** Sistema de zonas poligonales dibujables donde un objeto puede activar múltiples reglas simultáneamente.
- **Motor de Hitos (Milestones):** Configuración de reglas lógicas complejas:
  - `Individual:` Activación si hay > X objetos en una zona.
  - `Acumulado:` Activación basada en el histórico total detectado en la sesión.
- **Validación Secundaria (Doble Factor):** Capacidad de usar un segundo modelo de IA (Local o en la Nube) para validar detecciones críticas antes de disparar una alerta.

### 🔔 Notificaciones y Telemetría
- **Alertas de Telegram:** Notificaciones enriquecidas con HTML, incluyendo evidencia fotográfica del evento.
- **Webhooks de Google:** Integración directa con Google Apps Script para automatizar hojas de cálculo o bases de datos.
- **Logging en tiempo real:** Registro CSV de todas las detecciones y exportación de estadísticas mensuales.

### 🛠️ Herramientas de Entrenamiento
- **Dataset Capturer:** Herramienta integrada para capturar frames y realizar anotaciones (Bounding Boxes) manualmente, facilitando el re-entrenamiento de modelos propios.

---

## 🏗️ Mejoras Arquitectónicas (v2.5+)

Para garantizar la estabilidad y el rendimiento en sistemas de escritorio, el motor cuenta con:
- **Tipado Estricto (Type Hinting):** Todo el núcleo está tipado, mejorando la seguridad en tiempo de ejecución.
- **Optimización de Interfaz (Canvas):** Renderizado inteligente de métricas sin parpadeos, reduciendo la carga de CPU.
- **Detección de Hardware Robusta:** Identificación automática y a prueba de fallos de aceleradores (CUDA, OpenVINO, DirectML).
- **Gestor de Errores Simplificado:** Alertas claras en consola diseñadas para facilitar el uso diario sin saturar el sistema con logs complejos.

---

## 🛠️ Stack Tecnológico

| Librería | Propósito | Enlace |
| :--- | :--- | :--- |
| **Ultralytics** | Motor de Inferencia YOLOv8/11 | [GitHub](https://github.com/ultralytics/ultralytics) |
| **CustomTkinter** | Interfaz de Usuario Moderna | [Docs](https://customtkinter.tomschimansky.com/) |
| **VidGear** | Procesamiento de Video de Alto Rendimiento | [GitHub](https://github.com/abhitronix/vidgear) |
| **OpenCV** | Visión Computacional Core | [Web](https://opencv.org/) |
| **LapX** | Motor de Tracking (ByteTrack) | [PyPI](https://pypi.org/project/lapx/) |

---

## 📦 Modelos Recomendados

Para empezar a usar la aplicación, puedes descargar los siguientes pesos oficiales de Ultralytics:

- [yolo11n.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt) (Modelo Nano - Máxima velocidad, ideal para CPUs).
- [yolo11s.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt) (Modelo Small - Balance entre precisión y velocidad).

> **Nota:** Coloca tus modelos en la carpeta `/models/yolo/` y reinicia el Explorador de Inteligencia de la app.

---

## 🤖 Créditos y Desarrollo

Este proyecto ha sido desarrollado en colaboración con **Antigravity**, un potente agente de IA diseñado por el equipo de **Advanced Agentic Coding** de **Google DeepMind**.

- **Antigravity:** [Explorar en Google DeepMind](https://github.com/google-deepmind/antigravity)

---

## ⚖️ Licencia

Este proyecto está bajo la **Licencia MIT**. Puedes usarlo, modificarlo y distribuirlo libremente, siempre que mantengas la atribución original.

Copyright (c) 2026 Jose L.
