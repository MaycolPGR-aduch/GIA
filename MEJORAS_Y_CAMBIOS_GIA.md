# GIA – Mejoras y cambios necesarios para versión académica y de congreso

Este documento reúne los cambios técnicos, funcionales y metodológicos necesarios para convertir el proyecto **GIA – Gestural Interaction Assistant** en un sistema sólido, documentado y evaluable para una presentación académica o artículo de congreso.

---

## 1. Objetivo de la reestructuración

El objetivo principal es transformar el prototipo actual en un sistema asistivo con mayor calidad técnica, claridad arquitectónica y capacidad de evaluación experimental.

El proyecto debe pasar de ser una demostración funcional a una propuesta académica defendible bajo el enfoque de:

> Sistema multimodal de interacción humano-computadora para personas con movilidad reducida mediante visión computacional, gestos faciales y comandos de voz.

---

## 2. Cambios prioritarios

### 2.1 Modularizar el código

Actualmente, si gran parte de la lógica está concentrada en uno o pocos archivos, se recomienda dividir el sistema en módulos especializados.

Estructura sugerida:

```text
vision/
interaction/
ui/
logging_system/
config/
reports/
tests/
```

Beneficios:

- Mejor mantenimiento.
- Mayor claridad para nuevos desarrolladores.
- Facilidad para probar módulos por separado.
- Mejor explicación en el paper.
- Escalabilidad del sistema.

Prioridad: **Alta**

---

### 2.2 Crear un archivo de configuración

Los valores ajustables no deben quedar escritos directamente en el código.

Crear:

```text
config/settings.json
```

Debe incluir:

```json
{
  "camera_index": 0,
  "cursor_sensitivity": 1.5,
  "dead_zone": 0.03,
  "smoothing_factor": 0.6,
  "blink_threshold_left": 0.22,
  "blink_threshold_right": 0.22,
  "blink_min_duration_ms": 250,
  "click_cooldown_ms": 800,
  "voice_activation_mode": "both_eyes",
  "enable_voice_feedback": true,
  "enable_event_logging": true
}
```

Prioridad: **Alta**

---

### 2.3 Implementar perfiles de usuario

Cada usuario puede tener necesidades, movilidad y expresiones faciales diferentes.

Crear:

```text
config/user_profiles.json
```

Cada perfil debe guardar:

- Nombre del usuario.
- Sensibilidad del cursor.
- Umbral de guiño izquierdo.
- Umbral de guiño derecho.
- Zona muerta.
- Factor de suavizado.
- Preferencia de comandos.
- Fecha de última calibración.

Prioridad: **Alta**

---

## 3. Mejoras en visión computacional

### 3.1 Calibración formal del usuario

Agregar una pantalla o flujo de calibración inicial.

Pasos recomendados:

1. Detectar rostro.
2. Solicitar mirada al centro.
3. Registrar posición neutral de nariz.
4. Solicitar movimiento hacia la izquierda.
5. Solicitar movimiento hacia la derecha.
6. Solicitar movimiento hacia arriba y abajo.
7. Calibrar guiño izquierdo.
8. Calibrar guiño derecho.
9. Calibrar cierre de ambos ojos.
10. Guardar perfil.

Beneficios:

- Mayor precisión.
- Adaptación por usuario.
- Menos falsos positivos.
- Mejor sustento académico.

Prioridad: **Alta**

---

### 3.2 Suavizado del cursor

El movimiento facial suele generar vibración o pequeños saltos. Se debe implementar un filtro de suavizado.

Opciones:

- Promedio móvil.
- Filtro exponencial.
- Suavizado por interpolación.
- Límite de velocidad máxima.

Ejemplo conceptual:

```python
smoothed_x = alpha * current_x + (1 - alpha) * previous_x
smoothed_y = alpha * current_y + (1 - alpha) * previous_y
```

Prioridad: **Alta**

---

### 3.3 Zona muerta

Implementar una zona muerta para que movimientos mínimos del rostro no muevan el cursor.

Ejemplo:

```python
if abs(delta_x) < dead_zone:
    delta_x = 0

if abs(delta_y) < dead_zone:
    delta_y = 0
```

Prioridad: **Alta**

---

### 3.4 Detección de pérdida de rostro

El sistema no debe ejecutar acciones si el rostro no está detectado.

Acciones recomendadas:

- Pausar movimiento del cursor.
- Bloquear clics.
- Mostrar alerta visual.
- Registrar evento en log.
- Reanudar solo cuando el rostro esté estable.

Prioridad: **Alta**

---

## 4. Mejoras en detección de guiños

### 4.1 Tiempo mínimo de activación

No ejecutar clic apenas se detecta un ojo cerrado. Validar que el gesto dure un tiempo mínimo.

Ejemplo:

```text
Guiño válido = ojo cerrado durante al menos 250 ms
```

Prioridad: **Alta**

---

### 4.2 Cooldown entre clics

Después de un clic, esperar un tiempo antes de permitir otro clic.

Ejemplo:

```text
click_cooldown_ms = 800
```

Beneficio:

- Reduce doble clic accidental.
- Evita múltiples acciones no deseadas.

Prioridad: **Alta**

---

### 4.3 Diferenciación entre guiño y parpadeo natural

El sistema debe distinguir:

- Parpadeo normal.
- Guiño izquierdo.
- Guiño derecho.
- Cierre de ambos ojos.

Criterios sugeridos:

- Duración.
- Simetría entre ambos ojos.
- Umbral específico por usuario.
- Estado del sistema.

Prioridad: **Alta**

---

## 5. Capa de seguridad

### 5.1 Modo pausa

Agregar un modo de pausa accesible.

Formas de activarlo:

- Comando de voz: “pausar sistema”.
- Cierre prolongado de ambos ojos.
- Botón visible en pantalla.
- Tecla física de emergencia para acompañante/cuidador.

Prioridad: **Alta**

---

### 5.2 Bloqueo de acciones críticas

Algunas acciones deben requerir confirmación.

Ejemplos:

- Cerrar una aplicación.
- Enviar mensaje.
- Eliminar archivo.
- Apagar o reiniciar el equipo.

Prioridad: **Media**

---

### 5.3 Confirmación visual y auditiva

Antes de acciones importantes, mostrar y/o decir:

```text
¿Desea ejecutar esta acción?
```

Opciones:

- Confirmar con voz.
- Confirmar con guiño.
- Cancelar con cierre de ambos ojos.

Prioridad: **Media**

---

## 6. Mejoras en comandos de voz

### 6.1 Normalizar comandos

Crear un diccionario de comandos.

Ejemplo:

```json
{
  "abrir navegador": "open_browser",
  "abrir youtube": "open_youtube",
  "abrir gmail": "open_gmail",
  "subir volumen": "volume_up",
  "bajar volumen": "volume_down",
  "pausar sistema": "pause_system",
  "reanudar sistema": "resume_system"
}
```

Prioridad: **Alta**

---

### 6.2 Agregar confirmación de comandos

Cuando el comando se reconozca, el sistema debe confirmar:

```text
Comando reconocido: abrir navegador
```

Prioridad: **Media**

---

### 6.3 Registrar comandos fallidos

Registrar:

- Audio no reconocido.
- Comando no encontrado.
- Acción no ejecutada.
- Tiempo de reconocimiento.

Prioridad: **Alta**

---

### 6.4 Agregar dictado de texto

Funcionalidad sugerida:

- Activar dictado.
- Escribir texto reconocido en el campo activo.
- Comando “borrar”.
- Comando “enter”.
- Comando “copiar”.
- Comando “pegar”.

Prioridad: **Media-Alta**

---

## 7. Mejoras en interfaz gráfica

### 7.1 Panel de estado

Agregar indicadores visibles:

- Rostro detectado.
- Sistema activo.
- Sistema pausado.
- Modo voz activo.
- Última acción ejecutada.
- FPS.
- Perfil activo.
- Estado de calibración.

Prioridad: **Alta**

---

### 7.2 Botones accesibles

Recomendaciones:

- Botones grandes.
- Alto contraste.
- Texto claro.
- Íconos simples.
- Separación suficiente.
- Evitar saturación visual.

Prioridad: **Media-Alta**

---

### 7.3 Pantalla de calibración

Crear una pantalla específica para guiar al usuario.

Debe mostrar:

- Paso actual.
- Instrucción clara.
- Barra de progreso.
- Confirmación de éxito.
- Opción de repetir paso.

Prioridad: **Alta**

---

### 7.4 Panel de métricas

Agregar panel o ventana de métricas:

- FPS.
- Clics realizados.
- Falsos clics reportados.
- Comandos de voz exitosos.
- Comandos fallidos.
- Pérdidas de rostro.
- Tiempo de sesión.

Prioridad: **Alta**

---

## 8. Registro y exportación de métricas

### 8.1 Crear sistema de logging estructurado

Formato recomendado CSV o JSON.

Campos sugeridos:

```text
timestamp, session_id, user_profile, event_type, event_value, confidence, fps, notes
```

Tipos de evento:

- SESSION_START
- SESSION_END
- FACE_DETECTED
- FACE_LOST
- CURSOR_MOVE
- LEFT_CLICK
- RIGHT_CLICK
- VOICE_COMMAND_SUCCESS
- VOICE_COMMAND_FAIL
- CALIBRATION_START
- CALIBRATION_END
- PAUSE_ON
- PAUSE_OFF
- ERROR

Prioridad: **Alta**

---

### 8.2 Exportar resultados

Formatos recomendados:

- CSV.
- Excel.
- JSON.
- PDF opcional.

Prioridad: **Alta**

---

### 8.3 Generar resumen de sesión

Al finalizar una sesión, generar resumen:

- Duración total.
- FPS promedio.
- Clics totales.
- Comandos exitosos.
- Comandos fallidos.
- Eventos de pérdida de rostro.
- Recalibraciones.
- Errores.

Prioridad: **Media-Alta**

---

## 9. Evaluación experimental para paper

### 9.1 Agregar modo de prueba

Crear un modo experimental que guíe tareas específicas.

Tareas recomendadas:

1. Mover cursor a un objetivo.
2. Hacer clic izquierdo.
3. Hacer clic derecho.
4. Abrir navegador por voz.
5. Buscar una palabra.
6. Escribir frase por dictado.
7. Activar pausa.
8. Recalibrar.

Prioridad: **Alta**

---

### 9.2 Medir tiempo por tarea

Registrar:

- Hora de inicio.
- Hora de fin.
- Duración.
- Éxito o fallo.
- Número de errores.

Prioridad: **Alta**

---

### 9.3 Registrar falsos clics

Se puede registrar de dos formas:

1. Manual: el evaluador marca falso clic.
2. Automática: clic fuera de zona objetivo durante una tarea.

Prioridad: **Alta**

---

### 9.4 Aplicar encuesta SUS

Agregar formulario externo o interno para la **System Usability Scale**.

Prioridad: **Media**

---

## 10. Limpieza de dependencias

### 10.1 Revisar requirements.txt

Eliminar dependencias no usadas.

El archivo debe contener solo librerías necesarias.

Ejemplo tentativo:

```text
opencv-python
mediapipe
numpy
pyautogui
customtkinter
pillow
SpeechRecognition
pyttsx3
openpyxl
pandas
pyinstaller
```

Agregar otras solo si realmente se usan:

```text
openai-whisper
sentence-transformers
pycaw
comtypes
```

Prioridad: **Alta**

---

### 10.2 Fijar versiones

Para reproducibilidad académica, usar versiones.

Ejemplo:

```text
opencv-python==4.9.0.80
mediapipe==0.10.14
numpy==1.26.4
```

Prioridad: **Media-Alta**

---

## 11. Empaquetado como ejecutable

### 11.1 Crear ejecutable para Windows

Usar PyInstaller:

```bash
pyinstaller --noconfirm --onefile --windowed main.py
```

Prioridad: **Media**

---

### 11.2 Incluir assets

Si se usan íconos o sonidos:

```bash
pyinstaller --add-data "assets;assets" main.py
```

Prioridad: **Media**

---

## 12. Pruebas del sistema

### 12.1 Pruebas unitarias

Crear pruebas para:

- Detección de guiños.
- Cálculo de zona muerta.
- Suavizado.
- Router de comandos de voz.
- Exportación de logs.

Prioridad: **Media**

---

### 12.2 Pruebas manuales

Crear checklist:

- La cámara inicia correctamente.
- El rostro se detecta.
- El cursor se mueve.
- El clic izquierdo funciona.
- El clic derecho funciona.
- La voz se activa.
- Los comandos se ejecutan.
- La pausa funciona.
- Los logs se guardan.
- La app se cierra correctamente.

Prioridad: **Alta**

---

## 13. Cambios necesarios para artículo académico

Para que el proyecto genere un artículo sólido, se necesita implementar o documentar:

- Arquitectura modular.
- Calibración de usuario.
- Capa de seguridad.
- Suavizado del cursor.
- Registro de métricas.
- Modo experimental.
- Pruebas con usuarios.
- Tablas de resultados.
- Análisis de errores.
- Limitaciones.
- Comparación con trabajos relacionados.

Prioridad: **Alta**

---

## 14. Roadmap sugerido

### Fase 1: Ordenamiento técnico

- Limpiar estructura.
- Crear README.
- Limpiar requirements.
- Separar módulos.
- Crear configuración.

### Fase 2: Usabilidad y seguridad

- Agregar calibración.
- Agregar pausa.
- Agregar cooldown.
- Agregar suavizado.
- Mejorar interfaz.

### Fase 3: Métricas y evaluación

- Implementar logging.
- Exportar resultados.
- Crear modo experimental.
- Aplicar tareas de prueba.

### Fase 4: Preparación para congreso

- Crear figuras.
- Generar tablas.
- Redactar paper.
- Preparar demo.
- Preparar presentación oral.

---

## 15. Cambios mínimos para una versión presentable

Si el tiempo es limitado, implementar como mínimo:

1. README formal.
2. Limpieza de dependencias.
3. Calibración inicial.
4. Suavizado de cursor.
5. Cooldown de clics.
6. Pausa de emergencia.
7. Logging de eventos.
8. Exportación CSV/Excel.
9. Panel de estado.
10. Pruebas con 5 usuarios.

Con estos cambios, el sistema ya puede sostener un artículo preliminar.

---

## 16. Definición de versión

Versión sugerida:

```text
GIA v1.0 – Academic Prototype
```

Criterios para declarar v1.0:

- Control facial operativo.
- Guiños funcionales.
- Voz funcional.
- Calibración disponible.
- Seguridad básica implementada.
- Logging disponible.
- Métricas exportables.
- Documentación mínima completa.

---

## 17. Nota ética y de autoría

Si el proyecto fue retomado desde una base institucional, se recomienda mantener transparencia:

> El sistema se desarrolló a partir de un prototipo institucional inconcluso, reestructurado y ampliado con fines académicos, de accesibilidad y validación experimental.

Esto protege la autoría y permite reconocer adecuadamente aportes previos.

