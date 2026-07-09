# Mapeo de correcciones — Revisores Intercon 2026 (Paper ID 54)

**Estado:** Aceptado para presentación oral · **Camera-ready: 20-jul-2026 23:59 (GMT-5)** vía EasyChair (IEEE Proceedings).
**Recomendación de los 3 revisores:** Aceptado con revisiones mayores (2) / menores (1).
**Complemento de:** `INFORME_correcciones_paper_Intercon.md` (hallazgos de integridad de datos).

---

## 0. Requisitos duros de la carta de aceptación (no negociables)

| Requisito | Detalle | Impacto |
|---|---|---|
| 🌐 **Idioma** | El manuscrito final debe estar **enteramente en inglés** | La versión `_corregido.docx` está en **español** → **traducción completa requerida** |
| 📄 **Extensión** | Recomendado **6 a 8 páginas completas** | Verificar longitud tras traducir (con 9 figuras) |
| 🧾 **IEEE PDF eXpress** | Validar/generar el PDF final con PDF eXpress antes de subir | Paso técnico obligatorio |
| 👥 **Lista de autores** | **No se aceptan cambios** (agregar/quitar autores) | ⚠️ Los autores 2 y 3 están como plantilla IEEE sin llenar. Deben coincidir **exactamente** con lo registrado en EasyChair — llenarlos con los coautores ya registrados, o quitarlos si la sumisión era de 1 autor. No se puede agregar gente nueva. |
| 💳 **Registro** | ≥1 autor debe registrarse antes del 20-jul | El presentador oral en Trujillo |
| 🎤 **Presentación oral** | Obligatoria, presencial (12-14 ago, UPAO, Trujillo) | — |

---

## 1. Resumen de puntajes

| Criterio | R1 | R2 | R3 | Lectura |
|---|:--:|:--:|:--:|---|
| Originalidad | 4 | 4 | 4 | **Fortaleza consolidada** |
| Relevancia/Contribución | 4 | 4 | 4 | **Fortaleza consolidada** |
| Metodología y Evidencia | 2 | 2 | 4 | 🔴 **Punto débil principal** |
| Claridad/Estructura/Redacción | 2 | 3 | 3 | 🟠 Requiere trabajo |
| Resultados y Conclusiones | 3 | 1 | 4 | 🔴 **Punto débil principal** |
| Calidad general | 3 | 2 | 3 | Revisiones mayores |

**Diagnóstico:** la *idea* está bien valorada (originalidad y relevancia = 4 unánime). Lo que hunde la nota es **evidencia empírica insuficiente** y **redacción**. Ambos son corregibles.

---

## 2. 🔑 Insight estratégico

**Casi todo lo que piden los revisores ya existe como dato real en el repositorio.** Reportar esos datos reales **resuelve simultáneamente**:
1. Las exigencias de los revisores (matriz de confusión, muestras, splits, distribución de errores, falsos positivos).
2. El problema de integridad detectado antes (cifras inventadas/hardcodeadas).

Una sola acción — *reemplazar las métricas inventadas por las métricas reales del modelo + logs* — cubre a los tres revisores y sanea el paper.

### Evidencia ML real disponible (perfil "Maycol 2", modelo activo **v3**)

Fuente: `data/calibration/Maycol 2/model_registry.json` + `..._dataset_summary.json`

- **Dataset:** propio (self-captured por perfil), **no público**. ~260–300 frames capturados por gesto.
- **Clases (7):** neutral + 6 gestos (guiño izq, guiño der, ambos ojos, boca-O, sonrisa, confirmar). *(brows_up se eliminó en v3 → coherente con "6 gestos activos".)*
- **Ventanas:** 1112 totales → **833 entrenamiento / 279 validación** = **split temporal 75/25**.
- **Ventanas por clase:** neutral 250, guiño izq 287, sonrisa 199, boca-O 141, ambos ojos 111, guiño der 78, confirmar 46.
- **Exactitud de validación: 98,2 %** · Macro-F1 0,978 · Weighted-F1 0,983.
- **Matriz de confusión (real, 279 ventanas):** solo **5 errores**, todos guiños confundidos con "ambos ojos cerrados" (3 guiño-izq→ambos, 2 guiño-der→ambos).
- **F1 por clase:** confirmar/boca-O/neutral/sonrisa = 1,00; guiño izq 0,979; guiño der 0,947; ambos ojos 0,918 (precisión 0,848, recall 1,0).
- `window_size=12`, `feature_dim=32`, `LogisticRegression`, umbrales por clase disponibles.

> Este bloque responde **directamente** al Revisor 2 (matriz de confusión + muestras + split + origen del dataset) y valida el módulo ML por separado.

### Evidencia adicional disponible
- **Logs de sesión reales** (`data/logs/`): distribución de errores, aceptación/rechazo gestual, desglose por causa, panorama de voz, confianza media. → responde a R1 (attempts, error distribution) y R3 (falsos positivos, sesiones largas).
- **Datos multi-usuario reales:** hay perfiles con modelos entrenados y logs de **~5 usuarios** (Alejandro, Daniel, Diego, Maycol, Maycol 2). → permite ampliar más allá de "un solo perfil" (crítica central de R1).

---

## 3. Mapeo comentario → acción → dato disponible

### Revisor 1 (Metodología 2 · Claridad 2 · Resultados 3)

| # | Comentario | Acción a realizar | ¿Dato/fuente? |
|---|---|---|---|
| R1.1 | Evaluación solo en 2 sesiones internas y 1 perfil → poca generalización | Ampliar a los **múltiples perfiles con logs reales** (≈5 usuarios) y/o enmarcar mejor la limitación | ✅ Logs de Alejandro/Daniel/Diego/Maycol/Maycol 2 |
| R1.2 | Faltan mediciones formales: tiempo de tarea, precisión, errores de activación | Medir o **declarar explícitamente como pendiente** con protocolo definido (ya hay tabla de "estado de medición") | ⚠️ Parcial: falsos positivos y aceptación sí están en logs; tiempo por tarea no |
| R1.3 | Métricas necesitan: nº total de intentos, protocolo de prueba, distribución de errores, condiciones de luz/ruido, variabilidad/IC | Añadir subsección de **protocolo** + tabla de distribución de errores (real de logs) + describir condiciones | ✅ Distribución de errores en logs; ⚠️ luz/ruido e IC hay que documentarlos/estimarlos |
| R1.4 | Redacción: repeticiones, frases largas, gramática, inglés no académico | **Reescritura + traducción profesional al inglés**, acortar frases, quitar repeticiones | 🔴 Tarea de edición mayor |
| R1.5 | Inconsistencia **Vosk (métodos) vs Whisper (resultados)** | Presentar **Whisper Small como el motor real en todo el paper**; mencionar Vosk solo como trabajo relacionado (o quitarlo). Añadir **cita de Whisper** | ✅ Código usa faster-whisper "small" |
| R1.6 | Reporta **3680.95 s y 368.95 s** como duración total | En `_corregido.docx` las 3 menciones ya dicen 3680.95 (typo resuelto). **Pero** 3680,95 s no coincide con logs → reconciliar el número de fondo | ⚠️ Ver informe de integridad |

### Revisor 2 (Resultados 1 — el más crítico con el ML)

| # | Comentario | Acción a realizar | ¿Dato/fuente? |
|---|---|---|---|
| R2.1 | Aclarar **origen del dataset** (público o propio) | Indicar que es **dataset propio**, capturado por perfil durante calibración guiada | ✅ Confirmado |
| R2.2 | Especificar **nº total de muestras** y **% split train/test** | Reportar: 1112 ventanas (833/279), split temporal **75/25**, ~300 frames/gesto, ventanas por clase | ✅ Registro del modelo |
| R2.3 | Validar **etapas de entrada individuales** (rostro Y voz), no solo el sistema completo | Añadir evaluación por módulo: ML gestual (matriz de confusión) + voz (tasa transcripción/ejecución de logs) | ✅ Ambos disponibles |
| R2.4 | **Matriz de confusión obligatoria** para clasificación facial | Insertar la matriz real (7×7) + precisión/recall/F1 por clase | ✅ **Existe** (98,2 %, 5 errores) |
| R2.5 | Lista de clases citada: "(guiño izq, guiño der, ambos ojos, sonrisa, neutral)" | Corregir a **6 gestos + neutral** (faltan boca-O y confirmar) | ✅ Clases v3 |
| R2.6 | Si la clasificación de entrada falla, el error se propaga | El argumento se refuerza mostrando la matriz (98,2 %) + capa de seguridad heurística | ✅ |

### Revisor 3 (el más benévolo — Metodología y Resultados 4)

| # | Comentario | Acción a realizar | ¿Dato/fuente? |
|---|---|---|---|
| R3.1 | Evaluación más rigurosa del componente ML de gestos | Cubierto por la matriz de confusión + métricas por clase | ✅ |
| R3.2 | Pruebas con **usuarios objetivo reales** | Declarar como trabajo futuro (no hay usuarios con discapacidad aún); ampliar a los perfiles disponibles como paso intermedio | ⚠️ Limitación honesta |
| R3.3 | Métricas de usabilidad más claras | Definir instrumento (SUS) como pendiente, o aplicarlo si hay tiempo | ⚠️ No instrumentado |
| R3.4 | Comparación con métodos de interacción alternativos | Añadir tabla comparativa (vs Auxilio, XULIA, Sanvaad) — reforzar la discusión existente | ✅ Ya hay base en literatura |
| R3.5 | Análisis más profundo de falsos positivos y fiabilidad en sesiones largas | Usar logs: rechazos por causa, 0 falsos... derivar tasa real; duración de sesiones | ✅ Logs |

---

## 4. Consolidado: incidencias que aparecen en revisores + auditoría interna

| Tema | Revisor(es) | Auditoría interna | Prioridad |
|---|---|---|---|
| Evidencia ML (matriz confusión, muestras, split) | R2, R3 | Datos reales disponibles | 🔴 Alta |
| Cifras de resultados inventadas/contradictorias | (implícito R1/R2) | **Confirmado hardcodeado** | 🔴 Alta |
| Vosk vs Whisper | R1 | Confirmado (código usa Whisper) | 🟠 Media |
| Traducción a inglés + redacción | R1 (+carta) | — | 🔴 Alta |
| Generalización (1 perfil) | R1, R3 | Hay ~5 perfiles con datos | 🟠 Media |
| "blendshapes" no usados | — | Confirmado | 🟠 Media |
| GRU determinista (no entrenada) | — | Confirmado | 🟠 Media |
| Lista de gestos (6 vs 4+neutral) | R2 | Confirmado (v3 = 6+neutral) | 🟠 Media |
| Cita [18] colgante + falta cita Whisper | — | Confirmado | 🟠 Media |
| Autores 2-3 sin completar | (carta) | Confirmado | 🟡 Bloqueante camera-ready |
| Encabezados/idioma/formato numérico | R1 | Confirmado | 🟡 Baja |

---

## 5. Plan sugerido para el camera-ready (deadline 20-jul)

1. **Resolver autores + registro + PDF eXpress** cuanto antes (bloqueantes administrativos).
2. **Reescribir Resultados con datos reales** (matriz de confusión, muestras/split, distribución de errores de logs). — cubre R2, R3 y la integridad.
3. **Corregir inconsistencias técnicas** (Vosk→Whisper + cita, blendshapes, GRU determinista, lista de gestos, cita [18]).
4. **Ampliar evaluación** a los perfiles multi-usuario disponibles (mitiga R1/R3) y **documentar protocolo + condiciones** (luz/ruido) y qué queda pendiente.
5. **Traducir todo a inglés** y hacer una pasada de estilo académico (acortar frases, quitar repeticiones) — R1 + carta.
6. **Verificar extensión 6-8 págs** y numeración/referencia de las 9 figuras + añadir figura de matriz de confusión.
7. Preparar la **carta de respuesta a revisores** (response letter) mapeando cada comentario a su cambio (esta tabla es la base).
