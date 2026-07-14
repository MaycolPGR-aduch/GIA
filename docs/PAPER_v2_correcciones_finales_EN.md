# GIA-TRADUCIDO-V2 — correcciones finales (inglés)

Estado verificado: **8 páginas** (tope del rango 6-8), 6759 palabras, 19 referencias consistentes, sin cifras fabricadas.
Todo lo de abajo está dimensionado para **no pasar de 8 páginas**: primero se recorta (§0), luego se añade.

---

## §0 — Presupuesto de páginas: qué recortar primero

Estás en 8 páginas exactas, así que **antes de añadir, recorta**. Los tres bloques siguientes son redundancias que **R1 ya señaló** ("repetitions, long sentences"), así que quitarlos mejora la nota de claridad *y* libera espacio:

| Recorte | Dónde | Por qué | Gana |
|---|---|---|---|
| **1. Párrafos de explicación de la GRU en "Experimental setup"** | *"The GRU is a recurring network that processes sequences…"* + *"This combination separates two complementary tasks…"* | **Duplican** la descripción de la GRU ya dada en Materials and Methods (*"The first is a GRU temporal encoder with deterministic weights…"*). Basta con dejarla en Métodos. | ~2/3 col. |
| **2. Párrafo de definición de éxito repetido** | *"To estimate functional performance, internal tests were carried out by task, considering as a successful attempt…"* | Repite literalmente lo ya dicho en *"Functional performance by tasks"*. | ~4 líneas |
| **3. Config de cámara repetida** | *"The camera was set to 1280 x 720 pixels, with a target rate of 15 FPS…"* | Ya está en la **Tabla I**. | ~3 líneas |

> Con esos tres recortes te sobra espacio para todo lo que sigue.

---

## §1 (a) — Declaración ética  🔴

**Dónde:** al final de la subsección *"Preliminary test with target population"*.

> **Ethical considerations.** The field session was carried out with the prior institutional authorization of the OMAPED center, the public body responsible for the care of the participants. ⬜*[Informed consent was obtained from the parents or legal guardians of all participants]*, who were present throughout the session together with staff from the University of Sciences and Humanities. Participation was voluntary and could be interrupted at any time. No clinical records or personal health data were collected, and all facial processing was performed locally on the device, with no transmission of images or biometric data to external services. ⬜*[The study protocol was reviewed and approved by the Ethics Committee of ______ under record No. ______.]*

> ⚠️ **Importante — no exageres.** Redacta **exactamente lo que ocurrió**:
> - Si el consentimiento fue **verbal** y no escrito, dilo así (*"Verbal informed consent was obtained from…"*).
> - Si **no tienes aprobación de un comité de ética**, **borra la última frase**. No la inventes: es peor un dato falso que una limitación declarada.
> - Si no la tienes, puedes añadir en *Limitations*: *"The field observation was conducted under institutional authorization and guardian consent, but without formal ethics committee review; a controlled study with full ethical approval is planned."*

---

## §2 (b) — Condiciones de prueba (pedido explícito de R1)  🟠

**Dónde:** en *"Experimental setup"*, tras la frase del equipo (Intel Core i5 / 16 GB).

> The sessions were conducted indoors under ⬜*[artificial LED ceiling lighting / mixed natural and artificial lighting]* with a predominantly frontal and diffuse incidence, and under a ⬜*[low / moderate]* ambient noise level originating mainly from ⬜*[a fan and occasional nearby conversation]*. The user remained seated at approximately ⬜*[45–70]* cm from the camera, which was positioned at ⬜*[eye]* level. Visual capture used the laptop's integrated HD webcam at 1280 × 720 px and audio capture used the integrated microphone at 16 kHz.

*(Solo tú puedes llenar los ⬜. Si no mediste lux/dB, describe cualitativamente — es válido y suficiente para lo que pidió R1.)*

---

## §3 (c) — Distribución de errores con números reales (pedido de R1)  🟠

**Dónde:** en *"Module Analysis"*, **reemplazando** la frase cualitativa actual (*"…a large number of candidates were also rejected. This behavior responds to a conservative acceptance policy…"*).

> Across the seven internal sessions logged with the evaluated model (model_version = 3), the decision layer accepted 69 gesture events and rejected 450 candidates, i.e. an acceptance rate of 13.3 %. This behavior responds to a deliberately conservative acceptance policy: blocking a doubtful gesture is preferable to executing an accidental click. The breakdown of rejections is informative: 435 (96.7 %) were blocked by low classifier confidence, only 13 (2.9 %) by unstable face tracking and 2 (0.4 %) by cooldown. This indicates that facial tracking remained reliable throughout the sessions and that the margin for adjustment lies in the classifier thresholds rather than in the tracking.

**Procedencia (verificado en `data/logs/`, filtrado a `model_version == 3`):**

| Dato | Valor |
|---|---|
| Sesiones con eventos de v3 | 7 |
| Gestos aceptados | 69 |
| Candidatos rechazados | 450 |
| Tasa de aceptación | 13,3 % |
| Rechazo por baja confianza | 435 (96,7 %) |
| Rechazo por rostro inestable | 13 (2,9 %) |
| Rechazo por cooldown | 2 (0,4 %) |

> ✅ **Por qué estos y no otros:** filtré los logs a `model_version = 3`, que es el modelo que declara el paper. El agregado de *todas* las sesiones (185 aceptados / 1416 rechazados) mezcla v2 y v3 e incluye `brows_up`, gesto que **ya no existe en v3** — reportarlo contradiría el "6 active facial gestures" de la Tabla I. Estos números son coherentes con todo lo demás del paper.

---

## §4 (d) — Correcciones de inglés académico (R1: Clarity = 2)  🟠

Buscar → Reemplazar:

| # | Actual | Corregido |
|---|---|---|
| 1 | `The GRU is a **recurring** network` | `The GRU is a **recurrent** neural network` ← **error técnico, prioritario** |
| 2 | `if **you** look at a single image` | `when observing a single frame` |
| 3 | `**It's** deliberately simple` | `It is deliberately simple` |
| 4 | `allowing **you** to update **your** custom model` | `allowing the custom model to be updated` |
| 5 | `3.0 **sec** segments` | `3.0 **s** segments` |
| 6 | `called "Navigate." **this** command is associated` | `called "Navigate". **This** command is associated` |
| 7 | `viable for a desktop **preview**` | `viable for a **preliminary desktop version**` |
| 8 | `some **retrieval** functions` | `some **recovery** functions` |
| 9 | `depends on **progressive monitoring** and not on specific decisions` | `depends on **continuous tracking** rather than on discrete decisions` |
| 10 | `assistive interaction at the **desk**` | `assistive interaction **on the desktop**` |
| 11 | `The voice **mode** is implemented` | `The voice **modality** is implemented` |
| 12 | `room for improvement in **pickup** and feedback` | `room for improvement in **audio capture** and feedback` |
| 13 | `the **Facial Perception** module` | `the **facial perception** module` |
| 14 | `false clicks,··and document` (doble espacio) | espacio simple |
| 15 | `In the alternative cursor control, an assistant controller` | `In alternative cursor control, an assistive controller` |

**Coma splice (frase del 98,2 %) — reemplazar completa:**
> ~~"…achieved a validation accuracy of 98.2 % (macro-F1 = 0.978; F1 weighted = 0.983) over 279 windows, the classifier correctly identified 274 of 279 validation samples, which shows stable performance in most of the gestural classes evaluated."~~
>
> ✅ "…achieved a validation accuracy of 98.2 % (macro-F1 = 0.978; weighted F1 = 0.983) over 279 windows, correctly identifying 274 of the 279 validation samples. This indicates stable performance across most of the gesture classes evaluated."

**FPS presentado como medición (coherencia):**
> ~~"the system showed a preliminary performance close to 15 FPS"~~
> ✅ "the system operated at the configured target rate of 15 FPS"

---

## §5 — Recordatorio bloqueante 🔴

- [ ] **Autores 2 y 3** siguen como `2nd Given Name Surname` / `3rd Given Name Surname`. Deben coincidir **exactamente** con EasyChair (la carta prohíbe agregar/quitar autores).
- [ ] **IEEE PDF eXpress** + registro de ≥1 autor — antes del **20-jul-2026 23:59 (GMT-5)**.
- [ ] Reverificar **8 páginas** tras aplicar recortes (§0) y añadidos (§1-§3).
