# GIA v2

GIA v2 es un sistema asistivo de escritorio para personas con movilidad reducida orientado a control facial sobre Windows. El proyecto usa una cámara web para extraer landmarks faciales, mover el cursor mediante control heurístico de cabeza/nariz y reconocer gestos discretos con un clasificador local entrenado por perfil.

El estado actual del proyecto ya no corresponde al prototipo original descrito en `README_GIA.md`. Este `README.md` documenta la implementación vigente y debe mantenerse actualizado conforme avance el sistema.

## Objetivo

Construir una base sólida para un sistema asistivo:

- `offline-first`
- orientado a `Windows desktop`
- con perfiles por usuario
- con launcher obligatorio previo a la sesión
- con calibración antes de usar el runtime
- con control híbrido: heurístico para movimiento continuo y ML para gestos discretos

## Estado actual

La versión actual implementa:

- launcher con selección y creación de perfiles
- guía de gestos y comandos de voz
- diagnóstico básico de cámara, audio y modelo
- calibración y captura guiada de muestras faciales
- entrenamiento de un modelo por perfil
- runtime con cámara, estado del sistema y acciones asistivas
- logging estructurado por sesión con exporte secundario a Excel

La versión actual todavía está en fase de validación funcional. La arquitectura base ya existe, pero aún requiere afinado de sensibilidad, experiencia de usuario y evaluación con uso real prolongado.

## Funcionalidades implementadas

### Launcher obligatorio

Antes de entrar al runtime el usuario pasa por un launcher con:

- selección de perfil
- creación de perfil
- guía visual de gestos y comandos
- diagnóstico
- ajustes rápidos del perfil
- calibración y entrenamiento

### Control heurístico continuo

El movimiento del cursor se calcula a partir de la posición de la nariz y una referencia neutral del usuario. El pipeline incluye:

- zona muerta
- suavizado
- límite de velocidad
- recentrado
- verificación de estabilidad facial

### Gestos discretos con ML

Los gestos discretos actuales son:

- guiño izquierdo intencional
- guiño derecho intencional
- ambos ojos cerrados
- boca abierta sostenida
- sonrisa
- cejas levantadas
- gesto de confirmación

Acciones actuales:

- clic izquierdo
- clic derecho
- activar escucha de voz
- abrir guía rápida
- recentrar cursor
- pausar o reanudar

### Voz

La voz es opcional por perfil. Actualmente se usa:

- `faster-whisper` para STT offline
- router determinista local para comandos

Comandos de voz actuales:

- pausar sistema
- reanudar sistema
- centrar cursor
- abrir guía
- cerrar sistema
- volumen 25, 50, 75, 100
- abrir Gmail, Facebook, WhatsApp, YouTube

### Logging

Cada sesión registra eventos estructurados:

- inicio y fin de sesión
- cambios de estado
- gestos aceptados
- errores
- comandos de voz
- volumen

El log se guarda como `jsonl` y opcionalmente se exporta también a `xlsx`.

## Arquitectura actual

La app está dividida en módulos con responsabilidades separadas:

```text
main.py
app/
  assistive_controls.py
  launcher_gui.py
  runtime_gui.py
  config_store.py
  landmark_provider.py
  heuristic_engine.py
  gesture_ml.py
  gesture_catalog.py
  voice_router.py
  session_logger.py
  models.py
config/
  settings.json
  user_profiles/
data/
  calibration/
  logs/
  models/
tests/
  test_core.py
```

### Flujo general

```text
Launcher
  -> selección de perfil
  -> calibración
  -> entrenamiento del modelo del perfil
  -> runtime

Runtime
  -> cámara
  -> FaceLandmarker
  -> landmarks + métricas
  -> heurística continua
  -> clasificación ML por ventana temporal
  -> validación por confianza, duración y cooldown
  -> acción asistiva
  -> logging
```

## Tecnologías usadas

- Python 3.10+
- CustomTkinter
- OpenCV
- MediaPipe Tasks FaceLandmarker
- NumPy
- PyAutoGUI
- faster-whisper
- sounddevice
- scipy
- pyttsx3
- pycaw
- scikit-learn
- joblib
- pandas
- openpyxl

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

## Ejecución

```powershell
python main.py
```

## Flujo de uso recomendado

1. Ejecutar `python main.py`
2. Crear o seleccionar un perfil
3. Revisar la pestaña `Guia`
4. Ejecutar `Diagnostico`
5. Ajustar sensibilidad, zona muerta, suavizado y velocidad
6. Capturar postura neutral
7. Capturar gestos faciales
8. Entrenar modelo
9. Iniciar runtime

## Entrenamiento de gestos

El entrenamiento actual es local y por perfil.

### Cómo se captura

Cada gesto se captura con un flujo guiado:

1. cuenta regresiva de `3 s`
2. inicio de captura
3. mantener el gesto durante el tiempo sugerido
4. almacenamiento de muestras faciales frame a frame

No se guardan imágenes crudas como dataset principal. Lo que se usa son muestras faciales derivadas de landmarks.

### Qué se captura por frame

Cada frame genera un `FaceSample` con:

- landmarks faciales normalizados
- métricas derivadas
- posición de la nariz
- centro facial
- escala facial

Métricas derivadas actuales:

- `left_eye_ratio`
- `right_eye_ratio`
- `mouth_open_ratio`
- `smile_ratio`
- `brow_raise_ratio`

### Cómo se entrena

El clasificador usa ventanas temporales de `12` frames.

De cada ventana se extraen features con:

- landmarks aplanados
- métricas por frame
- medias
- desviación estándar
- delta entre primer y último frame

Modelo actual:

- `RandomForestClassifier`
- `n_estimators=300`
- `random_state=42`
- `class_weight="balanced_subsample"`

### Qué se guarda

Modelo del perfil:

- [data/calibration/<perfil>_gesture_model.pkl](D:\IEEE-Movilidad-reducidad\ProyectoGIA-main\GIA\data\calibration)

Resumen del entrenamiento:

- [data/calibration/<perfil>_samples.json](D:\IEEE-Movilidad-reducidad\ProyectoGIA-main\GIA\data\calibration)

Configuración global:

- [config/settings.json](D:\IEEE-Movilidad-reducidad\ProyectoGIA-main\GIA\config\settings.json)

Perfil del usuario:

- [config/user_profiles/](D:\IEEE-Movilidad-reducidad\ProyectoGIA-main\GIA\config\user_profiles)

Modelo facial de MediaPipe:

- [data/models/face_landmarker.task](D:\IEEE-Movilidad-reducidad\ProyectoGIA-main\GIA\data\models\face_landmarker.task)

### Formatos

- `.pkl` para el modelo serializado con `joblib`
- `.json` para el resumen de entrenamiento
- `.jsonl` para logs de sesión
- `.xlsx` para exporte secundario de métricas/eventos

## Configuración por perfil

Cada perfil guarda actualmente:

- sensibilidad del cursor
- zona muerta
- suavizado
- velocidad máxima
- umbrales de confianza por gesto
- duración mínima por gesto
- cooldown por gesto
- estado de calibración
- referencias neutrales del rostro
- banderas de voz y TTS

## Seguridad y validación de acciones

Antes de ejecutar un gesto, el runtime valida:

- que exista rostro
- que el rostro esté suficientemente estable
- que la confianza del modelo supere el umbral
- que el gesto dure el tiempo mínimo
- que no esté en cooldown
- que el estado del sistema permita esa acción

En pausa, solo se aceptan gestos seguros específicos.

## Pruebas disponibles

Actualmente existe una base pequeña de pruebas en:

- [tests/test_core.py](D:\IEEE-Movilidad-reducidad\ProyectoGIA-main\GIA\tests\test_core.py)

Ejecutar:

```powershell
venv\Scripts\python.exe -m unittest tests.test_core
```

## Limitaciones actuales

- aún no existe validación extensa con usuarios reales
- el modelo actual es por perfil y todavía no usa un modelo base global
- el dataset bruto de landmarks no se persiste completamente para reentrenamiento avanzado
- el control fino del cursor aún necesita afinado
- pueden aparecer logs internos de MediaPipe/TFLite en consola
- el sistema es Windows-first y no está preparado como multiplataforma

## Próximos pasos recomendados

- guardar también muestras crudas serializadas de landmarks
- mejorar la calibración con validación de calidad por gesto
- añadir barras de progreso y feedback visual más claro
- crear más pruebas de integración
- evaluar falsos positivos en sesiones largas
- preparar empaquetado con PyInstaller

## Archivos importantes

- [main.py](D:\IEEE-Movilidad-reducidad\ProyectoGIA-main\GIA\main.py): punto de entrada
- [app/launcher_gui.py](D:\IEEE-Movilidad-reducidad\ProyectoGIA-main\GIA\app\launcher_gui.py): launcher y calibración
- [app/runtime_gui.py](D:\IEEE-Movilidad-reducidad\ProyectoGIA-main\GIA\app\runtime_gui.py): interfaz runtime
- [app/assistive_controls.py](D:\IEEE-Movilidad-reducidad\ProyectoGIA-main\GIA\app\assistive_controls.py): controlador principal del runtime
- [app/landmark_provider.py](D:\IEEE-Movilidad-reducidad\ProyectoGIA-main\GIA\app\landmark_provider.py): extracción facial
- [app/heuristic_engine.py](D:\IEEE-Movilidad-reducidad\ProyectoGIA-main\GIA\app\heuristic_engine.py): movimiento continuo
- [app/gesture_ml.py](D:\IEEE-Movilidad-reducidad\ProyectoGIA-main\GIA\app\gesture_ml.py): clasificador y features
- [app/voice_router.py](D:\IEEE-Movilidad-reducidad\ProyectoGIA-main\GIA\app\voice_router.py): comandos de voz locales
- [app/config_store.py](D:\IEEE-Movilidad-reducidad\ProyectoGIA-main\GIA\app\config_store.py): perfiles, settings y rutas

## Nota de mantenimiento

Este documento debe actualizarse cada vez que cambie cualquiera de estos elementos:

- catálogo de gestos
- pipeline de entrenamiento
- estructura de carpetas
- flujo del launcher
- comandos de voz
- formato de persistencia
- dependencias del proyecto
