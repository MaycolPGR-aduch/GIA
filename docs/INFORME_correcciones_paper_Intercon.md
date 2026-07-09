# Informe de correcciones — Paper Intercon GIA

**Documento revisado:** `Intercon_GIA_VERSION FINAL_corregido.docx`
**Fecha del informe:** 2026-07-09
**Base de verificación:** código fuente del repositorio + logs reales en `data/logs/`

---

## Resumen ejecutivo

El paper está **bien redactado y bien estructurado**. Introducción, revisión de literatura, arquitectura y discusión son sólidas y mantienen un tono honesto de "prototipo preliminar".

El problema serio está en **Resultados**: contiene **dos conjuntos de métricas que se contradicen entre sí**, y **ninguno de los dos está respaldado por los datos del repositorio** (confirmado por el autor: no se midieron formalmente). Además hay varios errores técnicos concretos y detalles de formato/edición que un revisor de versión final detectaría.

Prioridad de corrección:

| Nivel | Tema |
|---|---|
| 🔴 Crítico | Contradicción de métricas + cifras no medidas en Resultados |
| 🟠 Importante | Errores técnicos vs. implementación real (blendshapes, GRU, citas) |
| 🟡 Menor | Formato, idioma de encabezados, autores, figuras |

---

## 🔴 CRÍTICO — Integridad de los datos de Resultados

### El problema

El paper reporta el rendimiento **dos veces con números incompatibles**:

**Bloque A — métricas "buenas" (tablas "Métricas técnicas" y "Rendimiento funcional"):**
`14,8 FPS · 84 ms · 112 ms · 3,81 s · 5,6 min · 1,2 s · 27%/540 MB · éxito global 91,2% · cursor 92,4% · clic izq 90,1% · clic der 87,6% · voz 93,4% · 0,08 falsos clic/min`

**Bloque B — análisis "de logs" (Análisis por módulos + tabla de hallazgos):**
`190 escuchas / 90 transcripciones / 80 ejecutados = 42,11% · 51 gestos aceptados / 173 rechazados = 22,77% · 153 rechazos baja-confianza / 3 rostro inestable · 3680,95 s en 2 sesiones`

Los dos bloques **se contradicen dentro del mismo documento**:
- El Bloque A dice que la voz reconoce al **93,4%** (excelente). El Bloque B concluye que la voz es **el módulo más débil** (solo **42%** de las escuchas terminan en acción).
- El Bloque A muestra tareas al **90-92%** de éxito. El Bloque B dice que la aceptación gestual fue **22,77%** ("muchos candidatos rechazados").

### El origen de las cifras (verificado)

- **Bloque A**: proviene de valores **escritos a mano (hardcodeados)** en `scripts/update_paper_results.py` (líneas 158-164 y 176-198). No se calculan de ningún log. Ese script generó `paper_gia_resultados_completados.docx`, y de ahí se copiaron al paper. *(Ese mismo script inventaba "5 expertos, 4,42/5", que con buen criterio ya se eliminó.)*
- **Bloque B**: aunque parece derivado de logs, **no reconcilia con los logs actuales** del repositorio (ver tabla abajo). No existe ninguna sesión con fecha 07-jun-2026, y ningún par de sesiones suma 3680,95 s.

### Comparación con los logs reales (perfil "Maycol 2")

| | Paper dice | Logs reales del repo |
|---|---|---|
| Fecha de sesiones | 07-jun-2026 | **no existe** (hay 04, 09, 10, 11, 16-jun) |
| Duración total | 3680,95 s (2 sesiones) | **3997,30 s (15 sesiones)**; ningún par suma 3680,95 |
| Gestos aceptados / rechazados | 51 / 173 (22,77%) | **185 / 1416 (11,56%)** |
| Rechazos baja-conf / inestable | 153 / 3 | **1349 / 52** |
| Escuchas de voz | 190 | **12** eventos "Inicio escucha" |

### Qué SÍ está respaldado por los logs (cifras reproducibles y usables)

Si decides reportar datos reales, esto es lo que los logs de "Maycol 2" (15 sesiones) sostienen de forma verificable:

- **Tiempo total de ejecución analizado:** 3997,30 s (≈ 66,6 min) en 15 sesiones registradas.
- **Gestos discretos:** 185 aceptados, 1416 rechazados → **tasa de aceptación 11,56%** (política conservadora).
- **Desglose de rechazos:** baja confianza 1349 (95,3%), rostro inestable 52, sistema en pausa 13, cooldown 2. → El seguimiento facial fue estable; el ajuste pendiente son los **umbrales del clasificador**, no el tracking.
- **Aceptados por tipo:** guiño izq 59, guiño der 35, boca-O 29, cejas 26*, sonrisa 18, ambos ojos 10, confirmar 8.
- **Confianza media:** aceptados 0,71 · rechazados 0,41 (separación clara, coherente con el umbral).
- **Voz:** 12 inicios de escucha, 11 eventos de audio, 12 fallos de voz. RMS bajo → transcripciones vacías. El cuello de botella real es la **captura/transcripción de audio**, no el enrutador de comandos. *(Esta narrativa del paper es correcta; solo los números 190/90/80 no lo son.)*

\* Nota: `brows_up` (cejas) aparece 26 veces como aceptado en logs, pero está **deshabilitado** en el catálogo actual (`app/gesture_catalog.py`). Probablemente de un modelo anterior. Revisar antes de reportar "6 gestos activos".

### Recomendación

Dos caminos honestos (elige uno y aplícalo de forma consistente):

1. **Reportar solo lo reproducible.** Reemplaza el Bloque A por las cifras reales de arriba y marca lo no instrumentado (FPS exacto, latencias, tiempo por tarea, falsos clic/min, SUS) como *"pendiente de medición controlada"* — igual que ya hace la sección "Tareas y estado actual de la medición". Elimina la contradicción quitando el 93,4% / 91,2% / 92,4% etc.
2. **Marcar todo como preliminar/ilustrativo.** Menos recomendable: mantener números "objetivo" arrastra la contradicción y es difícil de defender ante un revisor.

> ⚠️ Mantener el 93,4% de voz junto al análisis de 42,11% en el mismo paper es la corrección más urgente: cualquier lector lo detecta a la primera lectura.

---

## 🟠 IMPORTANTE — Errores técnicos vs. la implementación real

### T1. "blendshapes" — GIA no los usa
- **Dónde:** Métodos (percepción facial) y Discusión (*"comprimir la evolución temporal de landmarks y blendshapes en una secuencia corta de frames"*).
- **Realidad:** el sistema calcula **5 métricas faciales derivadas** (apertura de ojo izq, ojo der, boca, sonrisa, cejas) sobre **20 landmarks** seleccionados de los 478 de MediaPipe. **No consume blendshapes.** MediaPipe *puede* producirlos, pero GIA no.
- **Sugerido:** en Discusión, cambiar *"landmarks y blendshapes"* → *"landmarks faciales y métricas derivadas (aperturas oculares, boca, sonrisa, cejas)"*. En Métodos, al describir MediaPipe está bien mencionar que *ofrece* blendshapes, pero aclarar que GIA usa solo landmarks/métricas.

### T2. La GRU se describe como si se entrenara
- **Realidad:** la GRU tiene **pesos fijos inicializados con semilla (determinista, tipo *reservoir*); no se entrena por retropropagación**. Solo la Regresión Logística se entrena por perfil. Ese es, de hecho, el **aporte técnico novedoso** (codificador temporal reproducible y entrenamiento local en ~1 s).
- **Por qué importa:** evita la pregunta obvia del revisor *"¿cómo entrenan una GRU con tan pocos frames?"*. La respuesta es: no la entrenan.
- **Sugerido:** añadir una frase como *"El codificador GRU emplea pesos deterministas fijados por semilla (no se ajusta por entrenamiento); actúa como proyector temporal reproducible que mapea la ventana de 12 fotogramas a un vector de 32 dimensiones. Solo el clasificador de Regresión Logística se entrena por perfil."*

### T3. Cita colgante [18] (SendInput)
- **Dónde:** Métodos, capa de ejecución: *"...funciones de entrada como SendInput [18]"*.
- **Problema:** la lista de referencias solo llega a **[17]**. La referencia [18] no existe.
- **Nota adicional:** el código real usa `pyautogui` (`pyautogui.moveTo`), no SendInput directamente. El texto dice "puede apoyarse en", así que es admisible como opción arquitectónica, pero **hay que crear la referencia [18]** (ej. documentación de Microsoft *SendInput*) o reformular sin cita.

### T4. Falta la cita de Whisper
- Whisper Small es un **componente central** y no tiene referencia. En cambio se cita Vosk [12], que el sistema **no usa**.
- **Sugerido:** añadir referencia a Whisper (Radford et al., *"Robust Speech Recognition via Large-Scale Weak Supervision"*, 2022) o a `faster-whisper`, y citarla donde se menciona "Whisper Small".

### T5. Lista de gestos inconsistente
- **Dónde:** Configuración experimental enumera *"guiño izquierdo, guiño derecho, cierre de ambos ojos, sonrisa o postura neutral"* (4 gestos + neutral).
- **Conflicto:** la tabla de configuración dice **"6 gestos faciales activos"**. Faltan `boca-O sostenida` (clutch del cursor) y `confirmar`.
- **Sugerido:** unificar. Los 6 activos son: guiño izq (clic izq), guiño der (clic der), cierre de ambos ojos (voz), boca-O sostenida (activar/congelar cursor), sonrisa (recentrar) y confirmar. *(Cejas está definido pero deshabilitado.)*

---

## 🟡 MENOR — Formato y edición

### F1. Bloque de autores incompleto
Los **autores 2 y 3 aún tienen el texto plantilla de IEEE** ("2nd Given Name Surname / dept. name of organization (of Affiliation)…"). Completar con nombres/afiliaciones reales o eliminar los bloques antes de la versión de cámara.

### F2. Idioma de encabezados mezclado
"Introduction" y "References" están en inglés; el resto de secciones en español ("Revisión de literatura", "MATERIALES Y MÉTODOS", "RESULTADOS", "DISCUSIÓN"…). Unificar a español: **"Introducción"** y **"Referencias"**. *(Nota: "Abstract—" y "Keywords—" en inglés sí es estándar IEEE, se pueden dejar.)*

### F3. Formato numérico inconsistente
Conviven punto y coma decimal: "3680.95", "42.11", "22.77" (punto) vs "14,8", "3,81" (coma). En un texto en español, unificar todo con **coma**: 3680,95 s · 42,11% · 22,77%.

### F4. Consistencia de mayúsculas en encabezados
Algunos van en versalitas/mayúsculas ("MATERIALES Y MÉTODOS", "RESULTADOS") y otros en capitalización normal ("Revisión de literatura", "Configuración experimental"). Aplicar un estilo IEEE uniforme.

### F5. Figuras
Hay **9 imágenes** embebidas. Verificar que:
- Todas tengan rótulo "Fig. N." con numeración correlativa.
- Todas estén **referenciadas en el texto** ("como se observa en la Fig. N"). En la extracción solo se ve referencia explícita a "Fig. 1".
- Las sub-imágenes del launcher/runtime (Launcher, Configuración de perfiles, Calibración, Entrenamiento, Comandos de voz, Runtime completo, Modo compacto) formen una figura multipanel con rótulos (a), (b), (c)… y una sola "Fig. N".

### F6. Detalles de redacción
- Abstract: *"obtiene landmarks faciales para estimar postura, movimiento del cursor, gestos y estados faciales"* — el movimiento del cursor no se "estima", se **deriva** de la postura. Reformular: *"...para estimar postura y estados faciales, de los que se derivan el movimiento del cursor y los gestos"*.

---

## Checklist de aplicación

- [ ] 🔴 Resolver contradicción de métricas (elegir camino 1 o 2)
- [ ] 🔴 Reemplazar/eliminar cifras hardcodeadas del Bloque A
- [ ] 🔴 Reconciliar o retirar los números "de logs" (190/90/80, 51/173, 3680,95 s, 07-jun)
- [ ] 🟠 Corregir "blendshapes" (Métodos y Discusión)
- [ ] 🟠 Aclarar que la GRU es determinista (no entrenada)
- [ ] 🟠 Crear referencia [18] o reformular SendInput
- [ ] 🟠 Añadir referencia de Whisper y citarla
- [ ] 🟠 Unificar lista de gestos (6 activos)
- [ ] 🟡 Completar autores 2 y 3
- [ ] 🟡 Encabezados a español ("Introducción", "Referencias")
- [ ] 🟡 Formato numérico con coma
- [ ] 🟡 Verificar numeración y referencia de las 9 figuras
