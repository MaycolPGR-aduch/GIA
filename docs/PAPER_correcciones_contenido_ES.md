# Paper Intercon — contenido corregido para pegar en la plantilla IEEE

**Idioma:** español (traducir al inglés al final — requisito de camera-ready).
**Uso:** cada bloque indica si se **MANTIENE**, se **REEMPLAZA** o se **AÑADE**. Los `⬜ [POR COMPLETAR: ...]` son datos que **solo tú puedes aportar**.
**Base de datos reales:** modelo de gestos v3 del perfil "Maycol 2" + perfiles multiusuario + logs (`data/calibration/`, `data/logs/`).

---

## ⚠️ PARTE 1 — Lo que SOLO TÚ debes completar

Estos puntos no puedo resolverlos yo; requieren tu decisión o datos que no están en el repo:

1. **⬜ Autores 2 y 3.** El bloque tiene la plantilla IEEE sin llenar. Debe coincidir **exactamente** con lo registrado en EasyChair (la carta prohíbe agregar/quitar autores). Llénalos con los coautores ya registrados o elimina los bloques si la sumisión era de un solo autor.
2. **⬜ Condiciones de prueba** (piden R1): iluminación (interior/artificial/natural), nivel de ruido ambiental, modelo de cámara y micrófono, distancia usuario-cámara. Descríbelas según tu setup real.
3. **⬜ Métricas por tarea aún no instrumentadas:** tiempo de completado por tarea (T1–T7), precisión fina del puntero, falsos clic/min medidos, y SUS/usabilidad. → Se dejan marcadas como *pendientes de protocolo controlado* (honesto y ya avalado por los revisores). Si logras medir alguna antes del 20-jul, la insertas.
4. **⬜ Número de sesiones y duración analizadas:** decide qué subconjunto de logs reportas (ver nota en Resultados). El agregado real disponible es de 15 sesiones "Maycol 2" (~3997 s); el número "3680,95 s / 2 sesiones" del borrador no se reproduce.
5. **⬜ Gestión administrativa:** registro de ≥1 autor y validación en **IEEE PDF eXpress** antes del 20-jul-2026.
6. **⬜ Traducción final al inglés** de todo el manuscrito + pasada de estilo académico (frases más cortas, sin repeticiones — observación de R1).
7. **⬜ Insertar la figura de matriz de confusión:** `data/calibration/Maycol 2/Maycol 2_confusion_matrix_v3.png` (ya generada).

---

## PARTE 2 — Correcciones puntuales (buscar → reemplazar)

### Título, Abstract, Keywords — MANTENER con un ajuste
- Abstract: donde dice *"obtiene landmarks faciales para estimar postura, movimiento del cursor, gestos y estados faciales"*, cambiar a:
  > "obtiene landmarks faciales para estimar postura y estados faciales, a partir de los cuales se deriva el movimiento del cursor y se reconocen gestos discretos"

### Introducción y Revisión de literatura — MANTENER (sin cambios de fondo)
Solo aplicar los cambios globales de la Parte 4 (formato/numeración).

---

## PARTE 3 — Secciones reescritas (texto listo para pegar)

### 🔧 MATERIALES Y MÉTODOS — Percepción facial (REEMPLAZAR el párrafo de MediaPipe)

> Ligada a la capa de captura se encuentra la capa de percepción facial, implementada mediante MediaPipe Face Landmarker. Esta tecnología, basada en aprendizaje automático, detecta un conjunto denso de landmarks faciales tridimensionales y, opcionalmente, coeficientes de expresión (blendshapes) y matrices de transformación facial [9]. **GIA utiliza únicamente un subconjunto de 20 landmarks de los 478 disponibles, a partir de los cuales calcula cinco métricas faciales por fotograma: la apertura relativa de cada ojo (eye aspect ratio), la apertura de la boca, un índice de sonrisa y la elevación de cejas. El sistema no emplea blendshapes.** Con estas señales, GIA deriva variables de interacción como el desplazamiento relativo de la nariz respecto a la postura neutra, la apertura ocular, la estabilidad del rostro y la presencia probable de gestos faciales intencionales.

*(Motivo: el código no consume blendshapes; usa 20 landmarks → 5 métricas. Corrige la observación técnica y refuerza la reproducibilidad que pide R2.)*

### 🔧 MATERIALES Y MÉTODOS — Modalidad de voz (REEMPLAZAR el párrafo de Vosk/Whisper)

> La modalidad de voz se implementa mediante reconocimiento automático de habla **offline con Whisper Small**, ejecutado localmente sobre CPU con cuantización int8, en español y con filtrado por actividad de voz (VAD). Aunque existen otros motores ASR locales de baja huella, como Vosk [12], GIA adopta Whisper Small por su robustez en comandos cortos sin conexión. En el sistema, la voz cumple un rol complementario al control facial: permite activar comandos globales, confirmar o cancelar acciones, pausar el sistema, iniciar la recalibración o ejecutar tareas que resultarían más lentas mediante movimiento facial.

*(Motivo: elimina la inconsistencia Vosk-vs-Whisper que señaló R1; Whisper queda como el motor real y Vosk solo como referencia relacionada.)*

### 🔧 MATERIALES Y MÉTODOS — Clasificador de gestos (REEMPLAZAR el párrafo del modelo híbrido)

> Para acciones discretas (clic, selección, pausa o confirmación), GIA combina reglas temporales con un clasificador entrenado por perfil. El esquema es híbrido de dos etapas. La primera es un **codificador temporal GRU con pesos deterministas fijados por semilla, que no se ajustan por entrenamiento**: actúa como un proyector reproducible que transforma una ventana de 12 fotogramas de las cinco métricas faciales en un vector de 32 dimensiones. La segunda es un clasificador de **Regresión Logística que sí se entrena localmente por usuario** sobre esos vectores, elegido por ser ligero, rápido (reentrenamiento en ~1 s) e interpretable. Esta separación permite reconocer la dinámica temporal de un gesto —un guiño intencional se distingue de un parpadeo por su evolución en varios fotogramas— manteniendo un entrenamiento local de muy bajo costo. En todos los casos, las predicciones pasan por umbrales de confianza, duración mínima y periodos de enfriamiento antes de ejecutarse.

*(Motivo: aclara que la GRU es determinista/no entrenada —el aporte novedoso— y responde a "¿cómo entrenan una GRU con pocos frames?". En versiones iniciales se evaluó Random Forest [17] como línea base, lo cual puedes mantener.)*

### ➕ MATERIALES Y MÉTODOS — NUEVA subsección "Conjunto de datos y entrenamiento" (AÑADIR, responde a R2)

> **Conjunto de datos y entrenamiento.** El conjunto de datos es propio y se construye por usuario durante una fase de calibración guiada; no se emplean repositorios públicos. Para cada gesto se capturan alrededor de 300 fotogramas mientras el usuario mantiene la expresión correspondiente. Sobre cada secuencia se aplica una ventana deslizante de 12 fotogramas (paso 1), y cada ventana se codifica en un vector de 32 dimensiones mediante el codificador GRU. La partición entrenamiento/validación es **temporal (75 %/25 %), sin barajado**, para evitar fuga de información entre fotogramas contiguos de una misma secuencia. A modo de referencia, el modelo del perfil instrumentado agrupó **1112 ventanas (833 de entrenamiento y 279 de validación)** distribuidas en siete clases (postura neutral y seis gestos), con el siguiente número de ventanas por clase: neutral 250, guiño izquierdo 287, sonrisa 199, boca en O 141, cierre de ambos ojos 111, guiño derecho 78 y confirmación 46.

*(Motivo: R2 exige origen del dataset, nº de muestras y % de split. Todo verificable en `model_registry.json`.)*

---

## PARTE 4 — RESULTADOS (reescritura completa)

> **Nota:** esta sección reemplaza las tablas de "Métricas técnicas" (14,8 FPS, 84 ms…) y "Rendimiento funcional por tarea" (91,2 %, 93,4 %…) del borrador, que no provenían de mediciones. La nueva versión se apoya en la validación real del clasificador y en los registros de ejecución.

### RESULTADOS — Configuración evaluada (MANTENER la tabla, con estos ajustes)
- "Modelo de gestos" → **"Codificador GRU determinista + Regresión Logística (32-dim, ventana de 12 fotogramas)"**.
- "Conjunto de interacción" → **"6 gestos faciales activos + 1 referencia neutral"** (ya correcto).
- La fila "Tasa de ejecución 15 FPS / 14,8 FPS": dejar **"15 FPS (objetivo configurado)"** y quitar el "14,8 FPS" salvo que lo midas. Ver tabla de pendientes.

### ➕ RESULTADOS — NUEVA subsección "Evaluación del clasificador de gestos" (responde a R2 y R3)

> **Evaluación del clasificador de gestos.** Se evaluó de forma aislada el módulo de clasificación facial sobre el conjunto de validación temporal (25 % reservado). El modelo del perfil instrumentado alcanzó una **exactitud de validación de 98,2 %** (macro-F1 = 0,978; F1 ponderado = 0,983) sobre 279 ventanas. La Fig. X presenta la matriz de confusión: los únicos errores correspondieron a cinco ventanas de guiños clasificadas como cierre de ambos ojos (tres del guiño izquierdo y dos del derecho), un patrón esperable por la cercanía visual entre ambos gestos oculares. La Tabla Y detalla la precisión, exhaustividad (recall) y F1 por clase.

**Fig. X.** — insertar `Maycol 2_confusion_matrix_v3.png`. Pie sugerido:
> Fig. X. Matriz de confusión del clasificador de gestos (perfil instrumentado, 279 ventanas de validación, exactitud 98,2 %).

**Tabla Y. Métricas por clase del clasificador de gestos (validación).**

| Clase | Precisión | Recall | F1 | n |
|---|---|---|---|---|
| Neutral | 1,00 | 1,00 | 1,00 | 62 |
| Guiño izquierdo | 1,00 | 0,96 | 0,98 | 72 |
| Guiño derecho | 1,00 | 0,90 | 0,95 | 20 |
| Cierre de ambos ojos | 0,85 | 1,00 | 0,92 | 28 |
| Boca en O (sostenida) | 1,00 | 1,00 | 1,00 | 35 |
| Sonrisa | 1,00 | 1,00 | 1,00 | 50 |
| Confirmación | 1,00 | 1,00 | 1,00 | 12 |
| **Macro promedio** | **0,98** | **0,98** | **0,98** | 279 |

### ➕ RESULTADOS — NUEVA subsección "Validación entre usuarios" (responde a R1 y R3)

> **Validación entre usuarios.** Para explorar la generalización más allá de un único perfil, se entrenó y validó el clasificador con perfiles independientes capturados por distintos usuarios (Tabla Z). En cuatro perfiles con calibración adecuada, la exactitud de validación se mantuvo entre 91,8 % y 100 %, con una media de **97,1 %**. Un quinto perfil, capturado con un número sustancialmente menor de ventanas, obtuvo una exactitud inferior (74,4 %), lo que refuerza la observación de que la calidad y la cantidad de muestras por perfil condicionan directamente el desempeño del clasificador.

**Tabla Z. Exactitud de validación del clasificador por perfil de usuario.**

| Perfil | Ventanas (train/val) | Exactitud val. | Macro-F1 |
|---|---|---|---|
| Usuario A | 885 (663/222) | 100,0 % | 1,000 |
| Usuario B | 737 (553/184) | 98,4 % | 0,984 |
| Usuario C (instrumentado) | 1112 (833/279) | 98,2 % | 0,978 |
| Usuario D | 677 (507/170) | 91,8 % | 0,922 |
| Usuario E (submuestreado) | 157 (118/39) | 74,4 % | 0,702 |

*⬜ [POR COMPLETAR: decide si anonimizas los nombres (Usuario A–E) o usas iniciales/consentimiento. Recomiendo anonimizar.]*

### 🔧 RESULTADOS — Comportamiento en ejecución (REEMPLAZAR el bloque funcional inventado)

> **Comportamiento en ejecución.** Los registros de ejecución del perfil instrumentado permiten caracterizar el comportamiento del prototipo en uso real. El control continuo del cursor (seguimiento de la nariz con suavizado, zona muerta y recentrado) se mantuvo estable y fue el componente más maduro. En los gestos discretos, la capa de decisión operó con una **política de aceptación conservadora**: una fracción elevada de candidatos fue rechazada antes de ejecutarse. El análisis de las causas de rechazo es informativo: la gran mayoría se debió a **baja confianza del clasificador (~95 %)** y solo una pequeña parte a rostro inestable, lo que indica que el seguimiento facial fue confiable durante las sesiones y que el margen de ajuste está en los umbrales, no en el tracking. La confianza media de los gestos aceptados (~0,71) fue claramente superior a la de los rechazados (~0,41), coherente con los umbrales calibrados. En el canal de voz, el principal factor limitante observado fue el **bajo nivel de señal de audio (RMS)**, que produjo transcripciones vacías; desde el punto de vista de seguridad, una transcripción vacía es preferible a una orden equivocada, pero reduce la eficiencia e indica que la captura/transcripción de audio —no el enrutador de comandos— es el punto de mejora prioritario.

*⬜ [POR COMPLETAR: si decides reportar cifras absolutas (nº de sesiones, duración total, nº de candidatos aceptados/rechazados, intentos de voz), fija el subconjunto exacto de logs y colócalas aquí. El agregado real disponible: 15 sesiones ≈ 3997 s; 185 gestos aceptados / 1416 rechazados; rechazos por baja confianza 1349. Verifica antes de publicar.]*

### 🔧 RESULTADOS — Tabla de estado de medición (MANTENER y ampliar)
Mantén la tabla "Tareas y estado actual de la medición". Añade explícitamente como *pendiente de protocolo controlado*: tiempo por tarea, precisión fina del puntero, falsos clic/min, tasa de recalibración y SUS. Esto es honesto y ya fue avalado por los revisores.

---

## PARTE 5 — DISCUSIÓN y CONCLUSIONES

### 🔧 DISCUSIÓN — un cambio puntual
- Donde dice *"...comprimir la evolución temporal de landmarks y blendshapes en una secuencia corta de frames..."*, reemplazar por:
  > "...comprimir la evolución temporal de las métricas faciales derivadas de los landmarks en un vector compacto..."

### 🔧 DISCUSIÓN / CONCLUSIONES — reforzar con la evidencia nueva (opcional pero recomendado)
Puedes añadir una frase que capitalice el resultado real:
> "La evaluación aislada del clasificador de gestos (98,2 % de exactitud de validación en el perfil instrumentado y una media de 97,1 % entre cuatro perfiles) aporta la evidencia empírica del módulo de aprendizaje automático, mientras que el análisis de rechazos confirma que la capa de decisión prioriza la seguridad sobre la sensibilidad."

El resto de Discusión, Conclusiones y Recomendaciones se **MANTIENE** (son sólidas y honestas).

---

## PARTE 6 — REFERENCIAS (correcciones)

1. **Añadir la cita de Whisper** (falta y es componente central). Sugerida:
   > A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey, and I. Sutskever, "Robust Speech Recognition via Large-Scale Weak Supervision," *arXiv preprint arXiv:2212.04356*, 2022.
   Citarla donde aparece "Whisper Small" en Métodos y Resultados.
2. **Arreglar la cita colgante [18] (SendInput).** El texto cita `[18]` pero la lista termina en `[17]`. Opción recomendada (coincide con el código real, que usa PyAutoGUI): reformular la capa de ejecución y citar:
   > Al Sweigart, "PyAutoGUI documentation," 2023. [En línea]. Disponible: https://pyautogui.readthedocs.io
   Y cambiar en el texto *"funciones de entrada como SendInput [18]"* → *"una biblioteca de automatización de entrada como PyAutoGUI [18]"*.
3. **Renumerar** la lista tras añadir las dos referencias y **verificar que toda cita en el texto tenga entrada** (revisar [12]–[18] después de pegar).

---

## PARTE 7 — Cambios globales de formato (observación de R1 + carta)

- **Encabezados a español:** "Introduction" → "Introducción"; "References" → "Referencias". *(Abstract— y Keywords— pueden quedar en inglés: es estándar IEEE.)*
- **Formato numérico:** unificar con coma decimal (3680,95; 42,11 %; 22,77 %); hoy conviven punto y coma.
- **Estilo de encabezados:** unificar mayúsculas/versalitas IEEE en todas las secciones.
- **Figuras:** verificar que las 9 imágenes tengan "Fig. N." correlativo y estén **referenciadas en el texto**; añadir la nueva Fig. X (matriz de confusión).
- **Extensión:** verificar 6–8 páginas tras traducir.
- **Redacción (R1):** acortar frases largas y eliminar repeticiones en la pasada de traducción al inglés.

---

## Checklist final

**Contenido (puedo aplicarlo yo si me lo pides):**
- [ ] Percepción facial: quitar blendshapes, describir 20 landmarks → 5 métricas
- [ ] Voz: Vosk→Whisper + añadir cita de Whisper
- [ ] Clasificador: GRU determinista + 32-dim
- [ ] Nueva subsección "Conjunto de datos y entrenamiento" (1112 ventanas, split 75/25)
- [ ] Resultados: insertar matriz de confusión (Fig. X) + Tabla Y por clase
- [ ] Resultados: Tabla Z multiusuario
- [ ] Resultados: reemplazar bloque funcional inventado por comportamiento real
- [ ] Discusión: quitar "blendshapes"
- [ ] Referencias: Whisper + arreglar [18] + renumerar
- [ ] Formato: encabezados español, coma decimal

**Solo tú:**
- [ ] Autores 2 y 3 (según EasyChair)
- [ ] Condiciones de prueba (luz, ruido, cámara, micrófono, distancia)
- [ ] Decidir sesiones/duración a reportar
- [ ] Medir (si hay tiempo) tiempo por tarea / SUS, o dejar como pendiente
- [ ] Registro + IEEE PDF eXpress
- [ ] Traducción final al inglés + pasada de estilo

---

## PARTE 8 — Plantillas para completar

> Llena los campos `⬜ ____`. Los textos *en cursiva* son ejemplos de formato/granularidad esperada (bórralos al completar). Cada plantilla incluye una versión en **prosa lista para pegar** y una **tabla de apoyo**.

### 8.1 Condiciones de prueba (responde a R1)

**Tabla de apoyo (para ti, no necesariamente va al paper):**

| Ítem | Tu valor | Ejemplo |
|---|---|---|
| Entorno | ⬜ ____ | *laboratorio interior / oficina / hogar* |
| Iluminación — tipo | ⬜ ____ | *artificial LED de techo / natural / mixta* |
| Iluminación — dirección | ⬜ ____ | *frontal difusa / lateral / con contraluz de ventana* |
| Iluminación — intensidad | ⬜ ____ | *~300–500 lux (si mediste) o "iluminación de oficina estándar"* |
| Ruido ambiental — nivel | ⬜ ____ | *bajo / moderado / alto; ~40 dB(A) si mediste* |
| Ruido ambiental — fuentes | ⬜ ____ | *ventilador, tráfico lejano, voces ocasionales* |
| Cámara — modelo | ⬜ ____ | *webcam integrada del portátil / Logitech C920* |
| Cámara — resolución / FPS | 1280×720 · 15 FPS (obj.) | *(ya confirmado en config)* |
| Micrófono — modelo/tipo | ⬜ ____ | *micrófono integrado / diadema USB / condensador externo* |
| Micrófono — frecuencia | 16 kHz | *(ya confirmado en config)* |
| Distancia usuario–cámara | ⬜ ____ | *≈ 45–70 cm* |
| Postura / altura de cámara | ⬜ ____ | *usuario sentado, cámara a la altura de los ojos* |
| Equipo de cómputo | ⬜ ____ | *Intel Core i5, 16 GB RAM DDR4, Windows 11* |
| Fecha(s) de sesiones | ⬜ ____ | *jun. 2026* |

**Prosa lista para pegar (en "Configuración experimental"):**

> Las sesiones se realizaron en ⬜ *[entorno]* bajo iluminación ⬜ *[tipo y dirección]* de intensidad ⬜ *[nivel]*, con un nivel de ruido ambiental ⬜ *[bajo/moderado/alto]* proveniente principalmente de ⬜ *[fuentes]*. La captura visual se hizo con ⬜ *[cámara]* configurada a 1280×720 px y una tasa objetivo de 15 FPS, y la captura de audio con ⬜ *[micrófono]* a 16 kHz. El usuario se ubicó a una distancia aproximada de ⬜ *[__ cm]* de la cámara, ⬜ *[postura]*. Las pruebas de integración se ejecutaron en ⬜ *[equipo: CPU, RAM, SO]*.

### 8.2 Métricas por tarea (responde a R1 y R3)

**Cómo medir (protocolo mínimo sugerido):**
- **Tasa de éxito** = éxitos / intentos. Define "éxito": acción solicitada completada sin asistencia manual, sin reiniciar y sin generar una acción distinta. Haz **N ≥ 10 intentos** por tarea.
- **Tiempo por tarea** = con cronómetro (o marca de tiempo en logs), desde el inicio de la tarea hasta completarla. Reporta la media (y desviación si puedes).
- **Falsos clic/min** = nº de clics no intencionados ÷ minutos de uso activo.
- **Precisión fina del puntero** = (opcional) sobre un blanco pequeño, mide sobrepaso (overshoot), tiempo de estabilización o jitter. Si no lo instrumentas, déjalo como *pendiente*.

**Tabla lista para pegar (rellena o marca "pendiente"):**

| Tarea | Intentos (N) | Éxitos | Tasa de éxito | Tiempo prom. (s) | Errores observados |
|---|---|---|---|---|---|
| T1. Cursor hacia objetivo | ⬜ __ | ⬜ __ | ⬜ __ % | ⬜ __ | *sobrepaso / deriva* |
| T2. Clic izquierdo (guiño izq.) | ⬜ __ | ⬜ __ | ⬜ __ % | ⬜ __ | *guiño no detectado / confusión con parpadeo* |
| T3. Clic derecho (guiño der.) | ⬜ __ | ⬜ __ | ⬜ __ % | ⬜ __ | *sensibilidad a asimetría* |
| T4. Pausa / reanudación | ⬜ __ | ⬜ __ | ⬜ __ % | ⬜ __ | *retardo en el cambio de estado* |
| T5. Comando de voz | ⬜ __ | ⬜ __ | ⬜ __ % | ⬜ __ | *transcripción vacía por señal baja* |
| T6. Recentrado / recalibración | ⬜ __ | ⬜ __ | ⬜ __ % | ⬜ __ | *recentrado repetido tras deriva* |
| T7. Uso continuo | ⬜ __ min | — | — | — | *fatiga, pérdidas de rostro* |
| **Falsos clic/min** | | | ⬜ __ /min | | |

> Si no alcanzas a medir alguna fila, escribe **"Pendiente de protocolo controlado"** en la celda. Es una postura honesta y ya aceptada por los revisores.

### 8.3 Usabilidad — SUS (responde a R3)

Si aplicas la **System Usability Scale** (10 ítems, escala 1–5). Puntúa cada ítem de 1 (muy en desacuerdo) a 5 (muy de acuerdo):

1. Creo que usaría este sistema con frecuencia.
2. Encontré el sistema innecesariamente complejo.
3. Pensé que el sistema era fácil de usar.
4. Necesitaría apoyo técnico para poder usar el sistema.
5. Las funciones del sistema estaban bien integradas.
6. Había demasiada inconsistencia en el sistema.
7. Imagino que la mayoría aprendería a usarlo rápidamente.
8. Encontré el sistema muy engorroso de usar.
9. Me sentí muy seguro usando el sistema.
10. Necesité aprender muchas cosas antes de poder usarlo.

**Cálculo SUS (0–100):** ítems impares → (respuesta − 1); ítems pares → (5 − respuesta); suma todo y multiplica por 2,5.
⬜ **Puntaje SUS obtenido: ____ / 100** (⬜ nº de participantes: ____)

> **Alternativa** si no aplicas SUS completo: valora de 1 a 5 estas dimensiones y repórtalas como percepción preliminar → facilidad de aprendizaje ⬜ __, comodidad ⬜ __, fatiga percibida ⬜ __, confianza en los clics ⬜ __, confianza en la voz ⬜ __, utilidad percibida ⬜ __. Indica nº de evaluadores. Si no aplicaste ningún instrumento, marca **"no medido"** (no inventes valores).
