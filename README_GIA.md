# GIA – Gestural Interaction Assistant

**GIA** (*Gestural Interaction Assistant*) es un sistema de interacción asistiva orientado a personas con movilidad reducida. El proyecto permite controlar funciones básicas de una computadora mediante visión computacional, gestos faciales y comandos de voz, reduciendo la dependencia del mouse y teclado físicos.

El sistema utiliza una cámara web para detectar puntos faciales, interpretar movimientos de la nariz como desplazamiento del cursor, reconocer guiños como acciones de clic y ejecutar comandos por voz para facilitar tareas digitales cotidianas.

---

## 1. Objetivo del proyecto

Desarrollar un sistema multimodal de bajo costo que permita a personas con movilidad reducida interactuar con una computadora mediante:

- Movimiento facial para controlar el cursor.
- Guiños para ejecutar clic izquierdo y clic derecho.
- Comandos de voz para abrir aplicaciones, ejecutar acciones y apoyar la interacción.
- Retroalimentación visual y auditiva.
- Registro de eventos para evaluar rendimiento, precisión y usabilidad.

---

## 2. Público objetivo

El sistema está dirigido principalmente a:

- Personas con movilidad reducida en extremidades superiores.
- Usuarios que presentan dificultad para manipular mouse y teclado convencionales.
- Entornos educativos, universitarios o domésticos donde se requieran soluciones de accesibilidad de bajo costo.
- Investigadores o estudiantes interesados en tecnologías asistivas, visión computacional e interacción humano-computadora.

---

## 3. Enfoque académico del proyecto

Este proyecto se plantea como un sistema de **interacción humano-computadora asistiva** basado en:

- Tecnología asistiva.
- Visión computacional.
- Reconocimiento de puntos faciales.
- Interacción multimodal.
- Accesibilidad digital.
- Evaluación experimental de usabilidad y rendimiento.

Nombre académico sugerido:

> **GIA: A Multimodal Assistive Human-Computer Interaction System for People with Reduced Mobility**

---

## 4. Funcionalidades principales

### 4.1 Control facial del cursor

El sistema detecta la posición de puntos faciales, especialmente la zona de la nariz, para convertir pequeños movimientos de la cabeza o rostro en desplazamientos del cursor.

Funciones asociadas:

- Detección facial en tiempo real.
- Seguimiento de nariz o punto facial de referencia.
- Movimiento del cursor en pantalla.
- Aplicación de zona muerta para evitar movimientos involuntarios.
- Suavizado del movimiento para mejorar estabilidad.

### 4.2 Clic mediante guiños

El sistema interpreta determinados gestos oculares como acciones del mouse.

Funciones asociadas:

- Guiño izquierdo para clic izquierdo.
- Guiño derecho para clic derecho.
- Cierre de ambos ojos para activar comandos especiales o modo de voz.
- Tiempo mínimo de activación para evitar falsos positivos.
- Cooldown entre clics para evitar acciones repetidas no deseadas.

### 4.3 Comandos de voz

El sistema permite complementar el control facial con comandos hablados.

Funciones asociadas:

- Reconocimiento de voz.
- Ejecución de acciones como abrir navegador, YouTube, Gmail, WhatsApp u otras aplicaciones.
- Control de volumen.
- Posible dictado de texto.
- Retroalimentación mediante síntesis de voz.

### 4.4 Interfaz gráfica

El sistema debe contar con una interfaz clara y accesible.

Elementos recomendados:

- Vista de cámara.
- Estado de detección facial.
- Estado del sistema: activo, pausado, calibrando, sin rostro detectado.
- Botones grandes y claros.
- Panel de comandos disponibles.
- Indicador de FPS.
- Indicador de última acción ejecutada.
- Botón de pausa o emergencia.

### 4.5 Registro de eventos

El sistema debe registrar eventos para permitir evaluación experimental.

Eventos sugeridos:

- Inicio y fin de sesión.
- Rostro detectado o perdido.
- Movimiento de cursor.
- Clic izquierdo.
- Clic derecho.
- Comando de voz reconocido.
- Comando de voz fallido.
- Recalibración.
- Activación de pausa.
- Errores del sistema.
- FPS promedio.

---

## 5. Arquitectura propuesta

Estructura general recomendada:

```text
Camera Input
    ↓
Facial Landmark Detection
    ↓
Calibration and Signal Normalization
    ↓
Interaction Decision Layer
    ↓
Cursor Control / Click Actions / Voice Commands
    ↓
Assistive User Interface
    ↓
Event Logging and Evaluation
```

---

## 6. Estructura recomendada del proyecto

```text
gia/
│
├── main.py
├── README.md
├── MEJORAS_Y_CAMBIOS.md
├── requirements.txt
│
├── config/
│   ├── settings.json
│   └── user_profiles.json
│
├── vision/
│   ├── face_tracker.py
│   ├── blink_detector.py
│   ├── nose_controller.py
│   └── calibration.py
│
├── interaction/
│   ├── cursor_controller.py
│   ├── voice_commands.py
│   ├── safety_manager.py
│   └── command_router.py
│
├── ui/
│   ├── app_window.py
│   ├── calibration_screen.py
│   ├── metrics_panel.py
│   └── accessibility_controls.py
│
├── logging_system/
│   ├── event_logger.py
│   └── metrics_exporter.py
│
├── reports/
│   └── experimental_results/
│
├── assets/
│   ├── icons/
│   └── sounds/
│
└── tests/
    ├── test_blink_detector.py
    ├── test_cursor_controller.py
    └── test_voice_commands.py
```

---

## 7. Tecnologías sugeridas

### Lenguaje principal

- Python 3.10 o superior.

### Librerías principales

- OpenCV: procesamiento de video.
- MediaPipe: detección de puntos faciales.
- PyAutoGUI: control del cursor y acciones del sistema.
- SpeechRecognition o Whisper: reconocimiento de voz.
- pyttsx3: síntesis de voz local.
- CustomTkinter o PySide6: interfaz gráfica.
- Pandas/OpenPyXL: exportación de métricas.
- NumPy: cálculo numérico.
- PyInstaller: generación de ejecutable.

---

## 8. Instalación

### 8.1 Clonar el repositorio

```bash
git clone <url-del-repositorio>
cd gia
```

### 8.2 Crear entorno virtual

```bash
python -m venv venv
```

### 8.3 Activar entorno virtual

En Windows:

```bash
venv\\Scripts\\activate
```

En Linux/macOS:

```bash
source venv/bin/activate
```

### 8.4 Instalar dependencias

```bash
pip install -r requirements.txt
```

---

## 9. Ejecución del sistema

```bash
python main.py
```

Parámetros sugeridos para futuras versiones:

```bash
python main.py --camera 0 --profile default --debug
```

---

## 10. Flujo de uso recomendado

1. Abrir el sistema.
2. Seleccionar o crear perfil de usuario.
3. Iniciar calibración.
4. Mantener el rostro en posición neutral.
5. Calibrar movimiento facial.
6. Calibrar guiño izquierdo.
7. Calibrar guiño derecho.
8. Probar movimiento del cursor.
9. Activar modo de interacción.
10. Usar comandos faciales y de voz.
11. Exportar métricas de sesión.

---

## 11. Evaluación experimental sugerida

Para validar el sistema, se recomienda realizar pruebas con usuarios usando tareas controladas.

### Tareas sugeridas

| Código | Tarea |
|---|---|
| T1 | Mover el cursor hacia un objetivo |
| T2 | Ejecutar clic izquierdo |
| T3 | Ejecutar clic derecho |
| T4 | Abrir una aplicación mediante voz |
| T5 | Buscar una frase en el navegador |
| T6 | Escribir una frase mediante dictado |
| T7 | Activar y desactivar pausa |
| T8 | Recalibrar el sistema |

### Métricas sugeridas

| Métrica | Descripción |
|---|---|
| Tiempo por tarea | Tiempo necesario para completar cada tarea |
| Tasa de éxito | Porcentaje de tareas completadas correctamente |
| Falsos clics | Número de clics involuntarios |
| Precisión de comandos de voz | Porcentaje de comandos reconocidos correctamente |
| FPS promedio | Rendimiento en tiempo real |
| Recalibraciones | Número de recalibraciones por sesión |
| Pérdidas de rostro | Veces que el sistema perdió el seguimiento facial |
| SUS | Escala de usabilidad percibida |

---

## 12. Consideraciones de accesibilidad

El diseño del sistema debe considerar:

- Botones grandes y visibles.
- Alto contraste.
- Indicadores visuales claros.
- Retroalimentación auditiva.
- Modo pausa accesible.
- Configuración de sensibilidad.
- Reducción de acciones involuntarias.
- Uso con cámara web convencional.
- Bajo costo de implementación.

---

## 13. Limitaciones actuales esperadas

- Dependencia de iluminación adecuada.
- Posible pérdida de seguimiento facial.
- Ruido ambiental puede afectar comandos de voz.
- Dificultad para diferenciar guiños en algunos usuarios.
- Dependencia inicial de Windows si se usan librerías específicas del sistema.
- Necesidad de validación con usuarios reales con movilidad reducida.

---

## 14. Trabajo futuro

- Incorporar teclado virtual.
- Agregar perfiles personalizados.
- Mejorar la detección de guiños.
- Añadir modelos adaptativos por usuario.
- Integrar dictado avanzado.
- Crear instalador ejecutable.
- Evaluar con usuarios con diferentes grados de movilidad reducida.
- Implementar dashboard de métricas.
- Mejorar compatibilidad multiplataforma.
- Publicar resultados como artículo académico.

---

## 15. Estado del proyecto

Estado sugerido:

> Prototipo funcional en proceso de reestructuración para validación académica y presentación en congreso.

---

## 16. Licencia

Definir según los objetivos del equipo o institución.

Opciones recomendadas:

- MIT License, si se desea liberar el código.
- Licencia institucional, si pertenece a la universidad.
- Licencia privada, si será usado como prototipo cerrado.

---

## 17. Autoría y reconocimiento

Si el proyecto fue retomado a partir de un prototipo institucional previo, se recomienda declarar de forma transparente:

> Este proyecto fue retomado a partir de un prototipo institucional inconcluso y reestructurado para su mejora técnica, documentación, validación experimental y presentación académica.

