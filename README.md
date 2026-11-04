# GIA v2 (Qt Edition)

GIA v2 es un sistema asistivo de escritorio para personas con movilidad reducida orientado a control facial sobre Windows. El proyecto usa una cámara web para extraer landmarks faciales, mover el cursor mediante control heurístico de cabeza/nariz y reconocer gestos discretos con un clasificador local que combina una red recurrente **GRU en NumPy** y un adaptador de **Regresión Logística**.

La interfaz de usuario ha sido migrada por completo a **PySide6 (Qt6)** y dividida en dos aplicaciones independientes para maximizar el rendimiento y la facilidad de uso.

---

## Objetivo

Construir una base sólida para un sistema asistivo:
* `offline-first`
* orientado a `Windows desktop`
* con perfiles por usuario
* con **Entrenador independiente** para calibración y perfilado
* con **Asistente en Runtime** optimizado para uso diario
* con control híbrido: heurístico para movimiento continuo y ML (GRU + LogReg) para gestos discretos

---

## Estado Actual

La versión actual implementa:
* **Aplicación única** (`main.py`): Una sola ventana con tres pantallas que comparten la cámara sin chocar.
  * **Inicio**: perfiles, su estado y la acción recomendada para cada uno. Puede iniciar el asistente solo con el último perfil (con 5 s para cancelar).
  * **Calibrar y entrenar**: diagnóstico de cámara y audio, guía de gestos, captura guiada y entrenamiento del modelo.
  * **Asistente**: control facial en tiempo real, con el bucle de cámara en un hilo independiente para evitar bloqueos. Terminar la sesión vuelve al inicio.
* **Modo Compacto Flotante**: Widget superior sin marco (`FramelessWindow`) con feed circular recortado de cámara, indicadores básicos y log compacto.
* **Modelo Híbrido Temporal**: Extractor secuencial GRU (NumPy) y clasificación local con Regresión Logística.

---

## Estructura del Proyecto

La app está dividida en módulos con responsabilidades separadas:

```text
main.py                -> Punto de entrada de GIA
main_trainer.py        -> Acceso directo: abre GIA en Calibrar y entrenar
app/
  main_window.py       -> Ventana única: coordina las pantallas y la cámara
  home_page.py         -> Pantalla de inicio y gestión de perfiles
  trainer_app.py       -> Pantalla de calibración y entrenamiento
  runtime_app.py       -> Pantalla del asistente en uso
  assistive_controls.py -> Controlador central de la sesión (audio, mouse, logs)
  app_runtime.py       -> Instancia única y registro de errores
  runtime_app.py       -> Aplicación asistiva principal en PySide6
  qt_helpers.py        -> Hilos de cámara QThread y estilos QSS de Qt6
  config_store.py      -> Configuración y almacenamiento de perfiles
  landmark_provider.py -> Extracción de landmarks faciales (MediaPipe)
  heuristic_engine.py  -> Movimiento continuo del cursor
  gesture_ml.py        -> Clasificador híbrido GRU + LogisticRegression
  gesture_catalog.py   -> Catálogo descriptivo de gestos y ayudas de comandos
  voice_router.py      -> Comando de voz local (Whisper Router)
  session_logger.py    -> Logging en JSONL y exportación a Excel
  models.py            -> Modelos de datos
config/
  settings.json        -> Ajustes globales de cámara/FPS
  user_profiles/       -> Archivos JSON de perfiles de usuario
data/
  calibration/         -> Datasets y modelos versionados por perfil (.pkl, .json)
  logs/                -> Archivos de logs de sesión
  models/              -> Modelo face_landmarker.task de MediaPipe
tests/
  test_core.py         -> Pruebas unitarias de componentes
```

---

## Instalación

### 1. Crear entorno virtual
```powershell
python -m venv venv
```

### 2. Activar entorno
```powershell
venv\Scripts\Activate.ps1
```

### 3. Instalar dependencias
```powershell
pip install -r requirements.txt
```

---

## Ejecución

Todo se hace desde la misma aplicación:
```powershell
python main.py
```
Opciones: `--no-autostart` no inicia el asistente automáticamente; `--train [PERFIL]` abre directamente *Calibrar y entrenar*.

Solo puede haber una GIA abierta a la vez. Los errores inesperados se guardan en `data/logs/errores_gia.log`.

### 1. Fase de Calibración y Entrenamiento
* **Uso**:
  1. En *Inicio*, crea un perfil con *Nuevo perfil*, o selecciona uno y pulsa *Calibrar y entrenar*.
  2. Ajusta la sensibilidad y zona muerta en el paso *Diagnóstico*.
  3. Selecciona cada gesto y haz clic en *Capturar* para registrar las muestras (cuenta regresiva de 3s).
  4. Ve al paso *Entrenamiento*, haz clic en *Entrenar modelo* y luego en *Probar ahora*.

### 2. Fase de Uso Diario (Runtime Asistivo)
* **Uso**:
  1. En *Inicio*, selecciona el perfil calibrado y pulsa *Iniciar asistente*, o espera el inicio automático.
  2. El asistente comenzará a mover el cursor según la posición de tu nariz.
  3. Ejecuta gestos faciales (guiños, cejas arriba, sonrisa) para hacer clics o pausar el sistema.
  4. Mantén ambos ojos cerrados (900 ms) para activar los comandos de voz de Whisper.

---

## Lógica de Machine Learning (GRU + Regresión Logística)

El sistema ya no clasifica frames de manera estática. Utiliza un enfoque temporal dinámico:
* **Codificador GRU en NumPy**: Utiliza una red recurrente GRU de 32 unidades con pesos deterministas inicializados bajo el concepto de *Reservoir Computing*. Mapea la secuencia temporal de 12 frames a un embedding de 32 dimensiones.
* **Adaptación Local**: Se entrena un clasificador `LogisticRegression` sobre los embeddings de la GRU. Esto permite un entrenamiento sumamente rápido ($< 5\text{ ms}$), requiriendo muy pocas muestras por gesto y eliminando casi por completo el riesgo de sobreajuste.

---

## Pruebas

Para ejecutar las pruebas de regresión de los componentes y algoritmos del sistema:
```powershell
venv\Scripts\python.exe -m unittest tests.test_core tests.test_app_shell
```
