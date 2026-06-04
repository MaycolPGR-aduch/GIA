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
* **GIA Trainer** (`main_trainer.py`): Panel para crear, seleccionar y configurar perfiles, realizar pruebas de diagnóstico de cámara/audio, ver guías de gestos y ejecutar la captura guiada de muestras.
* **GIA Runtime Assistant** (`main.py`): Ejecución del asistente facial en tiempo real con bucles de procesamiento en un hilo independiente (`QThread`) para evitar bloqueos. Incluye estados visuales, logs de comandos y consola de voz.
* **Modo Compacto Flotante**: Widget superior sin marco (`FramelessWindow`) con feed circular recortado de cámara, indicadores básicos y log compacto.
* **Modelo Híbrido Temporal**: Extractor secuencial GRU (NumPy) y clasificación local con Regresión Logística.

---

## Estructura del Proyecto

La app está dividida en módulos con responsabilidades separadas:

```text
main.py                -> Punto de entrada para el Runtime Asistivo
main_trainer.py        -> Punto de entrada para el Calibrador/Entrenador
app/
  assistive_controls.py -> Controlador central del runtime (audio, mouse, logs)
  trainer_app.py       -> Aplicación de entrenamiento en PySide6
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

El sistema se divide en dos fases operativas:

### 1. Fase de Calibración y Entrenamiento
Para crear un nuevo perfil de usuario, probar el hardware o calibrar gestos por primera vez, ejecuta el calibrador:
```powershell
python main_trainer.py
```
* **Uso**:
  1. Crea o selecciona un perfil.
  2. Ajusta la sensibilidad y zona muerta en la pestaña *Diagnóstico*.
  3. Selecciona cada gesto y haz clic en *Capturar* para registrar las muestras (cuenta regresiva de 3s).
  4. Ve a la pestaña *Entrenamiento* y haz clic en *Entrenar Modelo*.

### 2. Fase de Uso Diario (Runtime Asistivo)
Una vez calibrado el perfil, inicia el asistente de movilidad:
```powershell
python main.py
```
* **Uso**:
  1. Selecciona el perfil calibrado en el dropdown y haz clic en *Iniciar*.
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
venv\Scripts\python.exe -m unittest tests.test_core
```
