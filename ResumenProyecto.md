# Vision AI Engine -- Resumen Maestro del Proyecto v4.1

Este documento constituye la **fuente de verdad definitiva** para el proyecto "02 Proyecto Vision Streaming". Refleja la arquitectura modular profesional y todas las capacidades implementadas.

---

## 1. Arquitectura de Software (Modular)
El proyecto se ha reestructurado como un paquete Python profesional dentro de la carpeta `app/`, eliminando la estructura plana anterior para mejorar la mantenibilidad.

### Estructura de Paquetes
*   `app/core/`: Motores logicos (Deteccion, Captura de Video, Eventos, Hardware, Validacion).
*   `app/ui/`: Interfaz grafica basada en CustomTkinter (Dashboard, Anotador, Ajustes, Eventos, Explorador de Modelos).
*   `app/utils/`: Utilidades transversales (Pintado visual, Base de Datos SQLite, Logger CSV, Helpers, Gestor de Errores).

---

## 2. Capacidades Avanzadas (Implementadas)

### Motor de Deteccion
*   **Soporte Multi-Arquitectura**: YOLO (v8/v11), RT-DETR, YOLO-World (Zero-Shot) y modelos personalizados (.pt).
*   **Object Tracking (ByteTrack)**: Seguimiento de IDs unicos con gestion de perdida de objetivo y reidentificacion.
*   **Focus Mode**: Clic sobre un objeto para seguirlo exclusivamente, aislando su deteccion del resto de la escena.
*   **Aceleracion Automatica**: Deteccion de GPU (NVIDIA CUDA / Intel OpenVINO / AMD DirectML) con fallback seguro a CPU.

### Dashboard Interactivo
*   **Grafica de Barras Interactiva**: Distribucion de clases en tiempo real con filtrado por zona (General/Z1/Z2...) y seleccion de clase individual mediante clic.
*   **Contador Unico de Sesion**: Conteo preciso basado en IDs de tracking para evitar duplicados.
*   **Top 5 Breakdown**: Las 5 clases mas detectadas con su conteo de objetos unicos.
*   **Reloj de Uptime**: Tiempo activo de la sesion actualizado en directo.
*   **Galeria de Evidencias**: Miniaturas en tiempo real de las capturas de eventos activados.

### Analitica Persistente
*   **SQLite (DBManager)**: Registro de cada deteccion y evento en `telemetry_logs/vision_analytics.db` para consultas historicas. Optimizado con WAL mode y batch inserts.
*   **CSV (DataLogger)**: Registro en tiempo real de conteos agregados por clase y zona.
*   **Evidencia Visual Automatica**: Frames guardados en `telemetry_logs/evidences/` al activarse un hito, vinculados a la base de datos.

### Herramientas de Etiquetado
*   **Anotador Visual Multi-Clase**: Interfaz para dibujar bounding boxes con soporte para multiples clases en formato YOLO estandar.
*   **Atajos de Teclado**: Teclas `1`-`9` y `0` para seleccion rapida de las primeras 10 clases.
*   **Navegacion Rapida**: Teclas `A`/`D` o flechas izquierda/derecha para navegar entre imagenes en modo multi-imagen.
*   **Borrado Selectivo**: Clic derecho sobre una caja para eliminarla individualmente. Clic fuera deshace la ultima.
*   **Importacion ZIP**: Importa paquetes de imagenes (.zip) para etiquetado secuencial masivo.
*   **Autocaptura Periodica**: Captura automatica de frames a intervalos configurables para generar datasets sin intervencion.
*   **Conteo de Dataset**: Panel lateral que muestra en tiempo real cuantas etiquetas de cada clase tiene el dataset actual.
*   **Exportacion ZIP**: Genera un archivo ZIP del dataset listo para entrenamiento.
*   **Guardado Automatico al Finalizar**: El boton "Finalizar" guarda la ultima imagen y abre la carpeta del dataset.

### Evidencia Visual Automatica
Cuando un hito/evento se activa, el sistema guarda automaticamente un frame de evidencia en `telemetry_logs/evidences/`. Estas imagenes se vinculan en la base de datos y pueden ser enviadas opcionalmente via Telegram.

### Sintesis de Voz (TTS)
Integracion de anuncios por voz mediante `pyttsx3`. El sistema puede anunciar eventos criticos por los altavoces (ej: *"Persona detectada en Zona de Peligro"*), mejorando la accesibilidad y la respuesta en tiempo real.

### Seguimiento de Trayectorias (Trails)
El motor visual (`VisualPainter`) dibuja **estelas de movimiento** para cada objeto con Tracking ID. Estas trayectorias estan codificadas por colores y permiten visualizar el flujo de movimiento de los elementos en escena.

### Hitos Dinamicos y Flexibles
El sistema de eventos ha sido redisenado para permitir una configuracion total por parte del usuario:
- **Operadores logicos**: `>`, `<`, `==`, `>=`, `<=`, `Total >`.
- **Acciones granulares**: Registro local, Telegram (con foto), Webhooks, Voz (TTS) o todas a la vez.
- **Control de Cooldown**: Evita la saturacion de alertas con tiempos de espera configurables.
- **Validacion Secundaria**: Doble factor IA con YOLO-World, Segmentacion, Ollama o HuggingFace.

### Proveedores de IA para Validacion (v4.1)
El sistema permite configurar proveedores externos de IA para la comprobacion de hitos:
- **Ollama**: Endpoint local configurable con modelos multimodales (llava, moondream, etc.).
- **HuggingFace**: Inference API con API Key para modelos VQA y clasificacion de imagenes.
- **Configuracion desde UI**: Panel de Ajustes con campos para endpoints, API keys y test de conexion.

### Contenerizacion (Docker)
Incluye un `Dockerfile` optimizado con todas las dependencias de sistema (OpenCV, Tkinter, OpenVINO) para facilitar el despliegue en entornos aislados.

---

## 3. Interfaz de Usuario (UI/UX)
*   **Pantalla de Carga (Splash)**: Barra de progreso con etapas de inicializacion para feedback visual inmediato.
*   **Sidebar Inteligente**: HUD compacto que agrupa fuente, modelos, analisis, datasets, autocaptura y zonas.
*   **Dashboard de Analitica**: Canvas en tiempo real con graficas interactivas, telemetria de sesion y galeria de evidencias.
*   **Controles de Reproduccion**: Pausa, retroceso y avance para videos locales y YouTube VOD, con indicador de "EN DIRECTO".
*   **Temas**: Alternancia entre modo Oscuro y modo Claro.
*   **Diagnostico de Hardware**: Panel inferior que muestra en tiempo real la GPU detectada y el backend de inferencia activo.
*   **Panel de Ajustes**: Configuracion centralizada de credenciales (Telegram, Webhooks) y proveedores de IA (Ollama, HuggingFace).

---

## 4. Componentes Clave

| Componente | Ruta | Funcion |
| :--- | :--- | :--- |
| **Orquestador** | `main.py` | Punto de entrada con manejo de errores de arranque. |
| **Detector** | `app/core/detector.py` | Inferencia YOLO/RT-DETR, prompts YOLO-World y modelos custom. |
| **Engine** | `app/core/engine.py` | Adquisicion de video multi-hilo (CamGear + OpenCV). |
| **Eventos** | `app/core/events.py` | Evaluacion de reglas, alertas (Telegram/Webhooks/TTS) y SQLite. |
| **Hardware** | `app/core/hardware.py` | Diagnostico de GPU y seleccion automatica de backend. |
| **Validator** | `app/core/validator.py` | Validacion secundaria via VLMs (YOLO-World, Segmentacion, Ollama, HuggingFace). |
| **Main Window** | `app/ui/main_window.py` | Dashboard principal con renderizado optimizado. |
| **Components** | `app/ui/components.py` | Ventanas modales (Anotador, Filtros, Info, Modelos, Fuentes). |
| **Events Window** | `app/ui/events_window.py` | Configuracion de reglas de hitos y eventos. |
| **Settings** | `app/ui/settings_window.py` | Ajustes de Telegram, Webhooks, API Keys de IA y proveedores. |
| **Painter** | `app/utils/painter.py` | Renderizado de cajas, zonas, heatmaps, graficas y trayectorias. |
| **Logger** | `app/utils/logger.py` | Telemetria CSV con escritura debounced. |
| **DB Manager** | `app/utils/db_manager.py` | Persistencia SQLite optimizada (WAL, pooling, batch). |
| **Helpers** | `app/utils/helpers.py` | Rutas, constantes y configuracion JSON. |
| **Error Handler** | `app/utils/error_handler.py` | Codigos de error centralizados y logging. |

---

## 5. Hoja de Roadmap Actualizado
1.  **Dashboard Web**: Crear una API FastAPI que consuma la base de datos SQLite para mostrar graficas en un navegador.
2.  **Control Inbound**: Permitir que el bot de Telegram responda a comandos (ej: "/status" o "/screenshot").
3.  **Filtrado por Trayectoria**: Disparar eventos solo si un objeto cruza una linea en una direccion especifica (Cross-line counting).
4.  **Exportacion Selectiva** [COMPLETADO]: El sistema ahora permite exportar unicamente las imagenes que han sido efectivamente etiquetadas, ignorando los descartes y optimizando el dataset para entrenamiento.
