# Propuesta de modelo CASMI26 (2026-11-12): SR-1+

**Fecha:** 2026-11-12 (jueves) · **Merge:** 2026-12-07 · **Final:** 2026-12-14 · **Posición:** 0.480 público (rango ~890/5,100); bronce 0.483, plata 0.487, oro 0.494, primero 0.540; ~950 equipos en 0.480–0.481 con el mismo linaje. · **Origen:** revisión de 8 áreas + crítico (11-11), 5 propuestas (A–E) con 3 jueces, síntesis «SR-1+» y 6 verificaciones adversariales (rail, sesgo y evaluación, integración, estrategia, cómputo, investigación). Este documento es la versión editada tras esas verificaciones. Todo lo marcado «(estimación)» no está medido.

**Estado:** propuesta. No se ha lanzado nada (ni Picasso, ni Kaggle, ni la GPU local); todo lo que gasta cómputo o envíos espera las decisiones de §8.

---

## 0. Resumen ejecutivo

- **Qué proponemos.** Un plan en tres capas, con el modelo propio como entregable de investigación y no como palanca principal del LB:
  1. **Pick A = LS-480**, el notebook público-only reproducible (0.479), sin pesos nuevos, con imagen Docker y T4 fijadas. Se envía y **se selecciona en la UI antes del 11-18**, para que la autoselección no elija nuestros 0.480 no reproducibles (scorer retirado).
  2. **SR-1**, nuestro bi-encoder w4 reentrenado para no aprender el prior de «parecido a librería» (lift +0.274 SD en np_c2 y +0.308 en np_tims), refit con todos los datos y 3 semillas. Entra solo por un overlay *leader-safe* que no toca el rank 1, y solo en la pick B.
  3. **R1-Gate**, un retador condicional de rank 1 entre isómeros de la fórmula del líder (líder antiguo fijado en rank 2), con evidencia neutral respecto a la pertenencia (MetFrag-lite, verosimilitud forward contada, margen de SR-1). Candidato a pick B si pasa cribas de daño catastrófico.
  - Brazo de investigación: **FragWM** (world model de fragmentación a nivel de pico) y el estudio de localización del prior (E012); fuera del camino del 12-07.
- **Por qué.** El rank 1 es ~75 % del MRR en eval-c3 (0.472 de 0.625) y lo decide la pipeline pública. Nuestro modelo aprendió un prior de pertenencia a librería que las verdades ocultas de clase 2/3 nunca tienen. El hueco del segundo scorer acoplaba el rank 1 (7/430 moléculas). SR-1 corrige el sesgo y el acoplamiento; R1-Gate ataca la única fuente de error identificable (isómeros de la fórmula del líder: la verdad está en ranks 2–6 con esa fórmula en el 36.7 % de las listas de eval-c3).
- **Valor esperado honesto (private, frente a la base reproducible 0.479; estimaciones):**
  - SR-1 en el hueco: P(admitido) ≈ 0.2–0.35; E[Δ | admitido] ≈ 0 (−0.0008 a +0.0012 antes de encoger por transferencia); atribuible ≈ 0 a +0.0003. Un segundo scorer de ruido puro compra el mismo valor de opción (+0.0004 a +0.0011).
  - R1-Gate: E[Δ | admitido] +0.0005 a +0.002; con la política de mejor-de-dos, valor de opción ≈ +0.0013–0.0017 incondicional.
  - Merge con una pipeline no-fork: si ocurre (P ≈ 0.15–0.3), E[max(0,Δ)] ≈ +0.005–0.007; incondicional ≈ +0.001–0.002. Es la mayor palanca hacia bronce.
  - El oro (+0.014) está fuera de alcance; el objetivo realista es bronce (+0.003–0.004 sobre la base, dentro del bloque de forks).
- **Qué necesitamos del usuario (§8):** rota de Picasso (J1 11-16 y J2/J3 condicionales, J0 opcional), envío y selección temprana de la pick A, lectura hoy del contador de cuota GPU, verificar en la página de reglas que cuenta la mejor de las dos picks, aprobar la descarga sin GPU de los outputs de casmi26-eval-ext/-ext-b (F_ext), empezar ya la búsqueda de merge, y decidir sobre fe_D/DreaMS, ChemBERTa y extlib.

---

## 1. Revisión

### 1.1 Dataset: hechos clave y problemas

| Hecho | Valor | Fuente |
|---|---|---|
| Librería de entrenamiento `work/lib` | 1,951,652 espectros, 267,831 estructuras (filtro: aductos de test, \|ppm\| ≤ 20, ≥3 picos, desde 2,539,608) | revisión dataset |
| timsTOF y química confundidos | todo el timsTOF (822,372) es Enveda; Enveda-180 es química de screening tipo fármaco (NP-score mediano −1.50); toda la química NP es Orbitrap/QTOF salvo np-examples (250 estructuras, 1,182 espectros) | revisión dataset |
| Muestreo | uniforme por molécula: 67.9 % de los draws son e180, 8.1–9.3 % NP | revisión dataset / método |
| Condiciones | 48 % de los espectros de np-examples usan esquemas de CE ausentes en e180 y 35 % son fusionados; el 14.5 % de la librería (todo GNPS) tiene CE NaN → 30 eV | revisión dataset |
| Lo que ocultó w4base_s0 | 85,701 espectros, 92.6 % NP, incluidos los 1,182 timsTOF NP y ~181 espectros públicos por molécula de np-examples | revisión dataset |
| Ventanas de candidatos | PubChem ±10 ppm: mediana 5,266–6,063 candidatos (902–980 de la misma fórmula) | revisión dataset |
| Documentación | PubMed>0: 98 % en np_tims, 70 % en np_c2, 34 % en el universo NP realista de clase 2 | revisión dataset |
| Celda nueva posible | MassBank-harmonized: 476 estructuras no presentes en train con aductos de test (4,553 registros tras quitar 2 alias por clave oficial) + 110 referencias externas NP; 480 en rango de masa (237 NP-db, 30 MassSpecGym) | verificación de sesgo (`nit_check.py`) |

**Problemas que condicionan el modelo:**
- El modelo desplegado se entrenó sin los datos más parecidos al test, que el scorer público sí vio (mecanismo de H1). La memorización es el mayor efecto medido: +0.234 [0.209, 0.259] en modelo solo cuando se ven los espectros (np_c2).
- Los negativos duros de NP son escasos: mediana de 4 estructuras de train con la misma fórmula por ancla NP (21.5 % sin ninguna).
- Ninguna celda existente tiene verdades NP fuera de train.parquet, y los hechos del test oculto (nº de espectros ~1,500 o ~2,500, ~85 % [M+H]+, esquemas de CE) no están documentados. El test visible es un dummy de Enveda-180.
- Las consultas ocultas vienen de extractos (coaislamiento); todas las de entrenamiento y validación son estándares.

### 1.2 Evaluación: qué se sostiene, qué falló y la errata de eval-c3

**Se sostiene.**
- La réplica del métrico es fiel si se usan claves oficiales (`casmi.metric.ik14`): 29/430 verdades de eval-c3 tienen clave oficial distinta de la cruda.
- El replay del stack completo (`hybrid/replay_stack.py`) reproduce byte a byte los ficheros V2, CUSTOM y final de la ejecución eval-c3. La verificación de integración lo repitió: el fichero público crudo (camino b0repro sin bundle) reproduce los cuatro hashes (V2 3e7988ba…, CUSTOM 915a96c3…, physical 86ab25f2…, final ad86fc2f…).
- El lift de pertenencia es real y reproducible: coeficiente de efectos fijos +0.274 [0.213, 0.324] SD (np_c2, listas p25) y +0.308 [0.235, 0.380] (np_tims). Es igual o mayor para estructuras de train que el modelo no vio (np_tims, misma fórmula: vistas +0.247, ocultas +0.338), así que es un prior distribucional y no memorización.

**Falló.**
- Ningún instrumento offline tiene historial de calibración positivo frente al LB. Efectos grandes encogen 3–7× (prior de popularidad +0.50 → +0.075); efectos < 0.02 desaparecen o cambian de signo (P001: +0.010 → −0.010).
- El MDE de un probe es ≈ 0.019 (SE pareado ≈ 0.0075 para cambios del hueco; ≈ 0.0125–0.016 para movimientos de rank 1 o rankers enteros).
- No existe ningún par de semillas libre de fuga de la receta w4: contrastes de modelo solo < ~0.04 en np_c2 no se pueden decidir.
- El registro no impone el pre-registro: el hash solo cubre `rules`. En un sandbox con el borrador de E006 (brazos como conjuntos de semillas), `freeze` sobrescribió la lista declarada de brazos con resultados previos (solo la recalcula para brazos de un único peso, así que quedó vacía y post_hoc=false), y `report` falló de 8 formas distintas (§3.4).

**Errata de eval-c3 (obligatoria al leer el Anexo B y el commit 8671293).** eval-c3 **no** son «430 moléculas tipo fármaco de Enveda-180». Es una celda de **clase 3 simulada, tipo NP, con consultas Orbitrap/QTOF**:
- NP-score mediano 1.74 (mínimo 1.00); 63 % en COCONUT, 43.5 % en LOTUS; 0 espectros timsTOF; 0 verdades con espectros de Enveda-180 o np-examples; PubMed>0 45.6 %; masa mediana 431.6 Da; clase 3 en 430/430 (crítico). Comprobación propia sobre las 484 filas de `work/c3data/answers.parquet`: NP-score mediano 1.77, COCONUT 0.63, LOTUS 0.434, timsTOF 0.
- Todas sus verdades son estructuras de train/pool (el pool contiene el 100 % de las verdades y el 49.4 % de los candidatos), y el scorer público vio ~92 % de ellas.
- Consecuencias:
  - El diagnóstico de P001 en el Anexo B («la evaluación tipo fármaco no transfiere a NP») apunta al eje equivocado. Los ejes que separan eval-c3 del test son el instrumento, la mezcla de clases (sin clase 1), la pertenencia de las verdades a train/pool y la documentación.
  - El +0.0104 [+0.0027, +0.0185] del commit 8671293 se concentra en verdades poco documentadas; con doc ≥ 6 da −0.0014 [−0.013, +0.011]. eval-c3 favorece por construcción a los modelos con lift (copia con lift +0.0039 frente a copia neutral −0.0015 con α 0.25).
  - eval-c3 queda solo para mecánica (hashes, invariancia del rank 1, referencia de ruido). Nunca como predictor de despliegue.
  - Además, eval-c3 tiene CE ausente en 534/959 espectros, mientras que el test visible tiene 0/1,213: sus replays ejercitan el camino «CE missing», no el que usará el test.
- Queda registrada en el Anexo C de `docs/research_plan.md`, sin editar el Anexo B ni los commits. El texto de ayuda de `scripts/meta/build_cells.py --c3` («Enveda-180 class-3 cell») tenía el mismo error y está corregido.

### 1.3 Método: nuestro modelo y el stack

**Nuestro modelo (w4base_s0).**
- Bi-encoder de 18.5 M parámetros: Transformer de espectro d384/6 capas sobre ≤64 tokens de pico; torre de molécula = MLP sobre ChemBERTa-77M-MTR congelado (384) + Morgan r2 2,048 bits.
- Cabezas nce, fwd (copia FiLM de nce: r intra-lista 0.95–0.96), pred y fp. Score desplegado nf = (z(nce)+z(fwd))/2.
- Modelo solo: 0.36–0.38 MRR en ventanas completas NP. Errores: isómeros de la misma fórmula (91–100 % de los top-1 erróneos). Interpolador de análogos: 0.10–0.19 cuando ninguna estructura de train alcanza Tanimoto 0.55, frente a 0.48 con ≥0.85.
- Sus 32 negativos duros son siempre filas de base de datos (no train), uniformes en ±10 ppm (`hard_pool` 0). Por eso InfoNCE aprende log p_train(m)/q(m): el lift de §1.2.

**El stack 0.480.**
- La pipeline congelada 0.465 construye las listas de 25 y toma casi todas las decisiones de rank 1. Encima: scorer neural público (Saner) z-ensemblado 0.5/0.5 con un segundo scorer (retirado), RRF *leader-safe* (w_top 0.25, w_mid 0.75), el fragment check (MetFrag-lite, puede mover el rank 1) y la capa final Variant M.
- El 0.5 del segundo scorer se eligió en el LB público (las variantes 0.3/0.4/0.6 dieron 0.474). La base reproducible es el público-only (~0.479).
- El hueco no era *leader-safe*: con nuestro modelo el rank 1 final cambió en 7/430 moléculas, por el término w_top y por la ventana de contendientes del fragment check. Cada probe del hueco mezclaba ±1–2 moléculas públicas de rank 1.
- La pipeline favorece estructuras de train: 49 % de los líderes erróneos de eval-c3 lo son.
- Precedentes en el LB: todo movedor aprendido libre de rank 1 perdió (−0.019 a −0.113; P001 −0.010; nuestro ranker de una celda −0.152 en clase 1). Solo ganaron retadores con gate (fragment check +0.005; rr3 FREEZE +0.009, no reproducido por otros).

**Base y cumplimiento.** La base reproducible 0.479 contiene fe_D (fine-tune de DreaMS dentro del ranker v4b), con la pregunta NIST sin responder (volcado del 11-09). La pick A no cubre esa exposición (§9.2).

---

## 2. El modelo propuesto

### 2.1 Etapas

| Etapa | Qué es | Pick | Entrada | Salida esperada |
|---|---|---|---|---|
| 0. LS-480 | El camino b0repro existente **sin bundle**: ya reproduce los 4 hashes de eval-c3, así que no lleva código LS nuevo. Imagen Docker fijada (`dockerImage` + `dockerImagePinningType: original`) y `machineShape: NvidiaTeslaT4` | A (siempre) | ninguna | respuestas idénticas a la base 0.479 |
| 1. SR-1 | Bi-encoder neutral respecto a la pertenencia, refit con todos los datos, 3 semillas, solo en ranks 2–25 vía overlay *public-first* | B | E006 + E009 | ≈ 0 en LB; valor principal de investigación |
| 2. R1-Gate | Retador de rank 1 dentro de la fórmula del líder, líder antiguo en rank 2 | B (condicional) | E010 con cribas de daño | E[Δ\|admitido] +0.0005 a +0.002 (estimación) |
| Investigación | FragWM, E012 (localización del prior), base vectorial | ninguna antes del 12-14 | — | tesis |

### 2.2 Arquitectura de SR-1

- Familia w4 sin cambios de tamaño: 18.5 M parámetros entrenables; 16.0 M por semilla tras podar pred/fp (≈32 MB en fp16).
- Espectro: el encoder actual más un embedding de condición: multi-hot de 10 bins de CE (≤12, 13–17, 18–22, 23–32, 33–37, 38–45, 46–55, 56–65, 66–90, >90 eV), flag de espectro fusionado y flag de CE ausente (sustituye el NaN → 30 eV), con dropout de condición del 30 %. Sin token de instrumento (aplazado).
- Molécula: ChemBERTa-77M-MTR congelado + Morgan 2,048 → MLP 2432-768-256 (sin cambios).
- Score desplegado: nf = (z(nce)+z(fwd))/2 dentro de la lista, promediado sobre los espectros de la molécula y z-promediado sobre 3 semillas.
- **Paquete S (CE-set, pesos NP, jitter de m/z):** se declara *full_method* y no *one_axis*. Su efecto esperado es 0 ± 0.02, y el piloto contado predice ≈0 en el ranking por la condición de CE. Solo entra si sobra capacidad (§4.1).

### 2.3 Entrenamiento y negativos

- Datos: `work/lib` completo (sin datos externos); tablas de moléculas emb_v4 + un delta emb_v5 (~60k filas) que da negativos a las ~3.7k anclas antes excluidas.
- Poblaciones:
  - (a) Gemelos para E006: misma población que w4base_s0 (`--val work/cells`; holdout y np-examples ocultos). Comprobado: las 484 claves de `work/c3data/exclude_all_ik.txt` ya están dentro de `work/cells/exclude_all_ik.txt`, así que w4base_s0 también ocultó c3data y puede seguir siendo semilla de A0.
  - (b) Folds desplazados F1–F3 (E008): K=3 sobre las 1,476 estructuras de np_tims ∪ np_c2 ∪ c3, agrupadas por scaffold/Tanimoto ≥ 0.7. F_k oculta el fold k entero y solo el timsTOF del fold k+1 (da np_tims_c1, clase 1 entre instrumentos).
  - (c) Refit con todos los datos, semillas 0–2 (J3), solo si se mantiene el interés en SR-1 (§4.1).
- Brazos de negativos (K=32 por espectro):
  - **A0:** 32 filas de base de datos **uniformes** en ±10 ppm (la receta w4; no hay prioridad de misma fórmula: en la ventana de un ancla NP solo un 39 % mediano comparte fórmula).
  - **A1:** 16 identidades de train (misma fórmula primero; luego cascada 10 ppm → 0.05 Da → 0.5 Da; alias del ancla enmascarados) + 16 de base de datos, más el experto de sesgo b(m). Se declara como **paquete** (composición + PoE + prioridad de fórmula + softmax separada), salvo que la mitad de base de datos se haga uniforme como en A0 para tener un único eje.
  - **A2:** 32 identidades de train, sin base de datos. Elimina el prior train-frente-a-base de datos, pero **no es neutral por construcción**: la cabeza fp (sin negativos propios) muestra el mismo lift (+0.252 vistas / +0.337 ocultas), y la frecuencia de la cascada varía por estructura (SD 0.69 nats en NP).
  - **A0r:** A0 con residualización por pertenencia (λ × SD de la lista restado a las claves de train), sin entrenar. λ se recalcula sobre la definición p25 congelada (0.263 o 0.274 según la definición), en las mismas unidades en que se aplica, con λ estratificado por bit NP (0.31 NP frente a 0.16 no-NP). Su lift en np_c2 es in-sample. Baja por construcción las verdades de clase 1 (que son de train): coste a medir en np_tims_c1.
  - **A0q (nuevo, sin etiquetas):** score menos su media sobre un banco de 64 espectros de train (normalización por banco de consultas; variante Dynamic Inverted Softmax). Se calcula con la misma maquinaria que blind:nf. No usa pertenencia, así que no penaliza la clase 1 por construcción.
  - Contingencia **A1h** (solo si A1 pierde destreza): mitad de base de datos con isómeros Tanimoto-duros, incluidos gemelos PubChem no-train.
- Calendario de optimización: la receta w4 (batch 256, local_frac 0.5, 31,020 pasos, AdamW 3e-4 OneCycle, wd 0.01, bf16, clip 1, checkpoint del último paso), siempre en modo *eager* dentro de E006.

### 2.4 Debiasing: especificación del producto de expertos (PoE)

- b(m) = MLP 2432-384-1 sobre las features de molécula; opcionalmente condicionado a aducto/polaridad (la composición de la librería cambia con ellos).
- Se suma a **todos** los logits candidatos de toda softmax que contenga identidades de train o filas de base de datos; **nunca** a softmaxes solo in-batch. El término duro de fwd se separa explícitamente, con un test unitario.
- Identificabilidad: con entrenamiento conjunto solo τ·cos + b está identificado y cos puede quedarse con parte del prior (una sonda lineal ciega al espectro separa train de base de datos con 0.754 de exactitud por pares; ≈1 nat). Por eso b(m) **se pre-ajusta offline como clasificador train-frente-a-base de datos y se congela**, o se entrena con su propia loss y el gradiente PoE bloqueado.
- Monitor de lift durante el entrenamiento: auditoría sobre 100 moléculas cada 5k pasos.
- E007 (si se hace): los negativos se muestrean con los mismos pesos de molécula que las anclas, o se resta log w(m) a cada logit negativo (corrección logQ). Con pesos NP ×5 sin corregir se crea un prior NP de ≈ log 5 = 1.6 nats que la auditoría (que controla el bit NP) no ve.

### 2.5 Integración en el notebook (overlay *public-first*)

- La celda 18 queda **byte-idéntica** y siempre produce el final público.
- Celda 17b: copia la salida del fragment check a `submission_physical.csv`.
- Celda 18b: recalcula el final LS con `_pxm_main` en una carpeta auxiliar donde solo cambia el fichero neural (ensemble 0.75·z(público) + 0.25·z(SR-1)). Acepta el fichero LS solo si:
  - el ensemble tiene el esquema y el índice públicos con `neural_score` finito;
  - status «applied» y ninguna línea «Note:» en la salida capturada;
  - rank 1 igual al final público en todas las filas;
  - `moved_ids`, `leader_moves` y `lib_hit_molecules` iguales a MERGE_REPORT;
  - |Δ ranks_reordered| ≤ 5 % de las filas;
  - pasan los asserts de formato de la celda 18.
  Si algo falla, se queda el final público. Motivo: la celda 18 se traga errores de entrada (L22-23, L136-147, L155-167, L247-262) y sigue diciendo «applied». En una prueba con un fallo inyectado, el rank 1 quedó intacto en 430/430 pero el MRR cayó −0.0111 [−0.0158, −0.0066].
- Se elimina la composición CUSTOM de ranks 2–25 en la celda 14 (aporta +0.0001, una molécula); SR-1 entra solo por el rango neural (nr) de la celda 18. CUSTOM y V2 quedan públicos.
- SR-1 corre en la celda 14, **antes** de las etapas de outsiders y fragment check (corrige la afirmación previa). Por eso lleva guard de tiempo (no corre pasadas ~7.0 h), watchdog de 900 s por subproceso, check del PLM que falla cerrado (salida ≠ 0 si max|diff| > 0.05) y gate de cordura (cobertura neural_ok, scores finitos, correlación de rangos mediana con el scorer público en [0.15, 0.99]).
- `replay_stack.extract` lee las celdas del notebook construido; brazo «ls» que ejecuta 17b/18b tal cual; `tests/test_leader_safe.py` incluye el caso de fallo inyectado y exige que termine en el final público.

### 2.6 R1-Gate (retador de rank 1, condicional)

- **Elegibilidad** (todas): lib_max < 0.80; ≥2 espectros con ≥5 picos; el fragment check no movió el rank 1; contendiente con la fórmula del líder en ranks 2–6; líder no `class1_confirmed`; V definido si el líder es de librería; contendiente y líder no comparten el conjunto de masas de fragmentos de ≤2 cortes.
- **Evidencia** (neutral por construcción o por auditoría): margen MetFrag-lite (lift +0.04, n.s.), margen de verosimilitud forward contada A0 (108 tasas; lift +0.06/+0.10, n.s.; su lift a ciegas es ≈0), margen nf de SR-1 y contexto.
- **Decisión**: dos logísticas con cross-fit; disparar si P_c(1 − 1/r) − 0.5·P_l > 0.05 y el margen es positivo en ≥70 % de los espectros; tope del 8 % de disparos; máximo un movimiento por molécula.
- **Ajuste de coeficientes**: sobre las listas de fórmula del líder de np_c2 (y F_ext si sirve) o, de np_nit, solo registros CC0/CC BY. Nunca registros NC/SA. F_nit queda para evaluación.
- **Entradas que falla cerrado**: exige status «applied» en MERGE_REPORT y un estado de overlay conocido; recalcula LIB_MAX desde `neural_public.parquet` (nunca `LIB_MAX.get(mid, 0.0)`, que con un diccionario vacío haría elegible todo); lee V de `COMP/train.parquet` por clave (≈0.6 s y 1.5 GB por row group en local) con tope de 600 s (V indefinido = no dispara); no dispara si SR-1 no corrió.
- **Daño de clase 1**: la tasa de disparos erróneos se mide con referencias propias submuestreadas a densidad ordinaria (mediana 5 espectros [M+H]+, y 3× esa cifra) y con coaislamiento α 0.3, sobre la lista final *merged* y no sobre la lista principal. El impuesto a nivel LB se escribe como ≈ 0.152 × 0.5 × p_fire (anclado en la sonda solo-librería).

### 2.7 Brazo de investigación: FragWM y E012

- **FragWM**: verosimilitud p(picos | molécula, condición) sobre los fragmentos propios del candidato (≤3 cortes) × desplazamiento de H × carga, con término de ruido. GNN (GINE d192, 4 capas, ≈1.6 M parámetros), condicionamiento AdaLN/FiLM, sin negativos.
  - Arte previo: el núcleo es FraGNNet (arXiv 2404.02360). También ICEBERG, MARASON (2502.17874) y GLACIER (2606.29161). Se reutiliza o porta el código de FraGNNet (BSD-2 según la revisión, a confirmar) en vez de escribir el enumerador desde cero.
  - Lo nuevo y defendible: el condicionamiento (conjunto de CE, fusionado/ausente, timsTOF), su uso como verificador auditado frente al prior y los contrastes A0/A2/A3.
  - A2 (condiciones anuladas) se decide por la NLL por pico en held-out. El MRR de lista queda como secundario: el piloto contado da +0.0025 (n.s.) por CE y +0.009 (n.s.) por instrumento.
  - A3 se rediseña como modelo separado con el mismo tronco, pasos y datos que A1 y solo salida latente, o se declara *full_method*.
  - El gate G2 de pertenencia se calcula sobre los grupos completos de la misma fórmula (las listas recortadas solo tienen 32–44 filas de train ocultas).
- **E012 «prior-loc»**: ¿dónde vive el prior, en el marginal (eliminable por b(m) o por normalización por banco) o en interacciones espectro-molécula?
  - Ejecución: CPU, carga w4base_s0, <15 min, <4 GB.
  - Predicciones:
    - H1: lift ciego por banco de nce/fwd ≥ +0.10 SD.
    - H2: la cabeza jepa está dominada por tipicidad (Spearman ≥ 0.5).
    - H3: A0q no inferior (|lift| ≤ 0.10; MRR y mrrT no peores que A0 en más de 0.02 en np_c2 y 0.036 en np_tims).
  - Consecuencias: si H1 falla (lift ciego < 0.05 con lift real ≥ 0.2), A1 sale de J1. Si H3 pasa, A0q sustituye a A0r como fallback sin entrenamiento.
- **Afirmación de tesis acotada**: el mecanismo positivo-no-etiquetado ya está publicado. arXiv 2610.37979 («Separability Is Not Alignability», 2026-10-27; su abstract dice que un objetivo basado en candidatos «rewards a spectrum-independent prior over the benchmark's decoys, which we isolate»). Lo nuevo es:
  - el lift de efectos fijos sobre candidatos erróneos (vistos/ocultos);
  - el prior en cabezas sin negativos y la tipicidad de cabezas latentes (jepa: +0.201 → +0.023 al controlar tipicidad);
  - que la verosimilitud forward sin entrenar no tiene lift ciego (+0.022/−0.091 n.s., frente a +0.412/+0.329 del bi-encoder);
  - la comparación de correcciones sobre listas idénticas, y un test desplazado (timsTOF NP, test de extractos).

---

## 3. Evaluación y pre-registros

### 3.1 Celdas y marcos

| Celda / marco | Qué es | Papel | Limitación principal |
|---|---|---|---|
| **np_nit** (nueva) | MassBank-harmonized + referencias NP externas, ausentes de train por clave proporcionada, oficial y cruda (y por clave de esqueleto); 480 en rango; verdad en el top-25 por documentación (p25) en 283–287 | decisión de **destreza** (S1) | 48 % de las verdades p25 son de Eawag (cianopéptidos, PFAS; mediana 704 Da); masa mediana p25 507 Da frente a 354 de np-examples → tamaño efectivo 165 (75 NP) tras reponderar; solo 4.1 % de los candidatos erróneos p25 son de train (56 % de listas sin ninguno); estándares de referencia con aspecto de librería (percentil ciego mediano 0.75 en el proxy np_tims) |
| **np_c2** (801) / **np_tims** (245) | celdas NP existentes | decisión de **neutralidad** (lift FE), mrrT, docgrad, controles ciegos | verdades de librería: el MRR crudo está inflado por el lift; no es input de veredicto |
| F1–F3 OOF / **np_tims_c1** | folds desplazados | clase 2 OOF y clase 1 entre instrumentos | necesitan J2 |
| **F_ext** | outputs existentes de casmi26-eval-ext/-ext-b (Enveda-180 Zenodo timsTOF, estructuras fuera de train) | **primer** marco a nivel pipeline (coincidió en signo con el LB 3/3) | química tipo fármaco; hay que comprobar que siguen descargables y completos |
| **F_nit** | una ejecución Kaggle (~5.5 h GPU) de la pipeline pública sobre consultas de np_nit | marco NP a nivel pipeline, solo si F_ext no basta | 93–96 % de sus verdades llevan el bit de depositante de librería espectral de PubChem (frente a 3–6 % de un proxy de clase 2 NP), que la pipeline premia (pop+0.5 y regla pop ≥ 5) |
| **F_c1** | listas *merged* del panel NP «reference_in_other_libraries» con consultas np_c1x; referencias submuestreadas (np_c1x_sub) y coaislamiento a03/a06 | daño de clase 1 | panel de los estándares NP más referenciados (mediana de 91 referencias propias del mismo aducto, frente a 5 en NP ordinarias) |
| eval-c3 | ejecución en caché | solo mecánica (hashes, 0/430 cambios de rank 1, referencia de ruido) | sesgada hacia modelos con lift; 56 % de espectros sin CE |

**Especificación congelada de np_nit** (antes de E006):
- Excluir por clave proporcionada, oficial y cruda.
- Solo registros LC-ESI de alta resolución (fuera QQ/QQQ/EI/GC: 52 estructuras solo baja resolución o GC).
- Primaria reponderada por masa al histograma de np-examples (o restringida a 157–600 Da, ≈161 verdades p25); sin reponderar como sensibilidad.
- Estratos: Eawag/resto, MassSpecGym, emb_v4 «db» (42, negativos reales de entrenamiento) frente a «dist» (11, nunca entrenados), NP-db, tercil de documentación, licencia, tipicidad.
- Excluir las claves de np_nit del delta emb_v5 y de toda tabla de J3.
- Registrar n_truth_in_p25 al construir y borrar la rama de *fallback* (283–287 > 150).
- Licencias: CC BY-SA 73 %, CC BY 24 %, NC 2.7 %. Uso solo de evaluación, salvo CC0/CC BY si se usa para ajustar el gate.

### 3.2 Métricas y controles

- **lift_fe** (redefinido): coeficiente de efectos fijos (FE/FWL) sobre `is_train` entre candidatos erróneos de la lista p25 congelada. Molécula como efecto fijo; controles: decil de documentación, bit NP, decil de Tanimoto a la verdad; nf promediado sobre semillas; bootstrap por molécula con 2,000 réplicas. Se reporta además el grupo «train oculto» (estructuras nunca ancla ni negativo en ningún brazo) y el estrato NP. La definición por molécula del borrador encoge el lift a 0.175–0.179 frente a 0.27–0.31: con ella, el gate de 0.10 equivalía a ≈0.16.
- Una sola definición p25: doc = pc_lsid + pc_lpmid (ya ln(1+x)), empates por zlib.crc32.
- Métricas de modelo: p25:nf (reponderado p25w), mass:nf, mrrT:nf, blind:nf (banco de 64 espectros de masa igualada), lift por cabeza (nce, fwd, pred, fp), lift ciego por banco, Spearman de tipicidad, docgrad por molécula (Spearman intra-lista), pipe25 solo como informe.
- Controles: espectro ciego para todo brazo aprendido; A0r y A0q sin entrenamiento; 4 scorers de ruido por peso del hueco + brazo identidad; espectro equivocado de la misma fórmula para todo scorer de rank 1 (ratio de ganancia ≤ 1/3); MetFrag-lite y A0 contado sin entrenar como comparadores.
- Semillas: A0 con 3 (w4base_s0/s1/s2, primer σ_seed libre de fuga); cada brazo candidato con 2; un brazo con <2 semillas terminadas se reporta pero no puede pasar.
- Multiplicidad: «Bonferroni al 97.5 %» (no «Holm»); `boot()` debe aceptar el nivel declarado.

### 3.3 E006 (negativos neutrales): reglas de veredicto corregidas

Pregunta: ¿se puede eliminar por construcción el lift de pertenencia sin perder destreza espectral entre isómeros cuando la verdad no es de train?

| Bloque | Reglas |
|---|---|
| **N1** neutralidad | \|lift_fe\| < 0.10 y cota superior 95 % < 0.15, en np_c2 y en np_tims |
| **N2** | \|lift_fe del grupo train oculto\| < 0.10 (ventanas completas de la misma fórmula, np_c2+np_tims agrupadas) |
| **N3** | lift_fe en el estrato NP < 0.15 (np_c2) |
| **S1** destreza (decisión) | A1 − A0r en np_nit p25w: cota inferior unilateral 95 % > −0.03. Con SE ≈ 0.0135: P(pasa \| Δ=0) ≈ 0.7; P(pasa \| Δ=−0.03) = 0.05 |
| **S2** | A1 − A0 en mrrT:nf (np_c2 y np_tims): cota inferior unilateral 95 % > −0.03 (inflada para A1/A2 por la asimetría de exposición) |
| **S3** control ciego | en np_nit, (A1−A0)p25 − (A1−A0)blind_p25 ≥ 0; en np_c2/np_tims, A1 − A0 en blind:nf < +0.01 |
| **skill_lost** | cota superior 97.5 % de (A1 − A0r \| np_nit p25w) < 0, o de (A1 − A0 \| mrrT np_c2) < −0.03 |
| **lift_persists** | ci_low(lift_fe, np_c2) > 0.10 o ci_low(lift_fe oculto) > 0.10 |
| Superioridad (secundaria) | A1 − A0 en np_nit p25: ci_low > 0 → etiqueta «adopt-superior». Efecto esperado de un brazo perfectamente neutral: +0.018–0.020; potencia ≈ 0.15–0.3 |
| Mecanismo (informe) | A2 elimina el lift ciego de nce; si persisten los lifts de fp/pred en A2, la ruta de tipicidad es real; blind:nf de las verdades de np_nit frente a azar y frente a np_c2 (si está claramente por encima del azar, «skill_lost» se lee como «prior perdido» y se decide en sub-listas igualadas por pertenencia) |

- **A1_admit** = N1 ∧ N2 ∧ N3 ∧ S1 ∧ S2 ∧ ¬skill_lost; **A1_adopt** = A1_admit ∧ superioridad. A2: bloques explícitos idénticos (no un texto).
- **A0r_ship** (o A0q_ship): ningún candidato admitido ∧ cota inferior unilateral 95 % de (A0r − A0 \| np_nit p25w) > −0.01; su coste en clase 1 se comprueba en E008 (np_tims_c1).
- Consecuencia: se despliega A1 solo si A1_admit; si A1 falla S1 y A0r_ship se cumple, se envía A0r (sin entrenamiento). Si todo falla, la familia no entra en ninguna pick y el resultado negativo se publica.
- **Potencia (corregida):** SD pareada real entre modelos 0.25–0.32 por molécula → SE 0.011–0.015 con n=283; MDE80 0.035–0.045 (0.045–0.06 con tamaño efectivo 165). El contrafactual de quitar el lift vale +0.047 (np_c2) / +0.066 (np_tims), pero solo +0.018 / +0.020 con la mezcla de señuelos de np_nit.

### 3.4 Borradores de las siguientes exploraciones

- **E008** (congelar 11-17): F1–F3 de la receta candidata. Pasa si np_tims p25 OOF ≥ gemelo − 0.02. Informa np_tims_c1 − np_tims OOF (exposición de clase 1) y, sin coste extra, el AUROC del margen z OOF de SR-1 entre líderes de clase 1 (R1c) y líderes señuelo del fold k (hipótesis de «confianza memorizada»). Si np_tims_c1 muestra daño dentro de lista, las listas con lib_max ≥ 0.87 se congelan en la etapa 1.
- **E009** (congelar 11-20): admisión al hueco.
  - Refit frente al ensemble de gemelos en np_nit: p25w no inferior (cota inferior unilateral > −0.01) y subida de blind < 0.02.
  - Lift en np_nit **en ventanas completas**, agrupado con los lifts OOF de F1–F3 en np_c2/np_tims (en np_nit solo, informe: ≈280 filas de train en 124 listas dan IC ±0.15–0.2).
  - Replay LS con α 0.25 en F_ext (o F_nit): rank 1 sin cambios en todas las listas y Δ ≥ media de ruido + 0.002 y ≥ −0.001.
  - Mecánica en eval-c3: hashes, 0/430 cambios de rank 1.
  - Advertencia registrada: con esta regla, la copia neutral (−0.0015) y la de tipo oculto (−0.0028) **fallan** en eval-c3, y la tasa de paso es 0.19 / 0.36 / 0.55 / 0.78 para Δ verdadero −0.003 / −0.0015 / 0 / +0.002.
  - La evidencia del camino desplegado (CE presente) debe venir de F_ext/F_nit o de consultas con formato del test visible.
- **E010** (congelar antes de puntuar ningún candidato en el marco elegido): R1-Gate.
  - Primaria: Δ RR neto OOF por molécula con el líder antiguo fijado.
  - Cribas para la pick B (catastróficas): media del marco realista ≥ −0.003; disparos erróneos de clase 1 ≤ 3 % **a densidad de referencias ordinaria y con a03**; disparos ≤ 8 % en todo marco; falla cerrado; la ejecución oculta termina; ratio de espectro equivocado ≤ 1/3; \|lift\| ≤ 0.10.
  - «Adopt» (para ambas picks) exige además ci_low > 0 y f2·g·0.5 − 0.152·0.5·p_fire ≥ 0 a densidad ordinaria con transferencia ×1/3.
  - El rail del líder sospechoso se reporta como diagnóstico (§7), con todos sus gates y sobre la lista final *merged*.
- **E011** (solo si el piloto CPU v2 de IsoArb vence a MetFrag-lite con ci_low > 0 y \|lift\| ≤ 0.10).
- **E012** (CPU, antes de congelar E006 si hay capacidad; si no, A0q y el lift ciego por banco entran en E006 como brazo derivado y métricas de informe).
- **Registro** (requisito previo de todo congelado; probado con un *freeze+report* en sandbox sobre ficheros de evaluación ficticios):
  - brazos como conjuntos de semillas en freeze, leak y per;
  - freeze nunca sobrescribe `arms_with_results_at_freeze` y detecta resultados por semilla;
  - brazos derivados (A0r, A0q) con sus propios ficheros de evaluación;
  - entrada np_nit en CELLS con regla de fuga «no está en el manifiesto de entrenamiento» (la exposición emb «db» es un estrato, no una fuga);
  - tests a nivel de brazo y tests unilaterales nuevos; nivel de IC configurable;
  - tolerancia a columnas y brazos de informe ausentes; bloques de veredicto que no sean listas se rechazan;
  - IDs únicos (el borrador de FragWM también usa E006: se renumera);
  - `report` se niega si el fichero congelado no está en un commit ancestro.

---

## 4. Cómputo

### 4.1 Picasso (un job a la vez, ≤2 h, sin cadenas, sbatch de un fichero desde una carpeta nueva)

Capacidad (estimaciones, sin medida en A100/B200):
- Por brazo: 7.5–9.5 GB de GPU (store de 3.6 GB por proceso + 4.6 GiB de activaciones en fp32 a batch 256) y ~5 GB de RSS (≈7 GB al construir el store).
- Techo por proceso de ~25–50 it/s, limitado por el host (2,373 kernels y 9,116 despachos por paso).
- Agregado: A100 40 GB ~25–57 it/s (3–4 brazos como máximo; un banco de la tesis perdió 5 de 8 brazos por OOM en esa tarjeta); B200 ~50–170 it/s.
- Necesidades: J1 ≥ 38 it/s (6 brazos en 82 min; ≥19 para el veredicto mínimo de A1), J2 ≥ 31, J3 ≥ 19.6.

| Job | Fecha | GPU | Min | Contenido | Prerrequisitos |
|---|---|---|---|---|---|
| J0 (opcional) | 11-14/15, hora valle | b200 o dgx | ≤30 | preflight; 1 brazo 1,000 pasos; rampa 2/4(/6) brazos; un ciclo USR1→guardar→reanudar en GPU; paridad de compile en 300 pasos; featurización de 10k filas de np_nit con paridad; un dump → `bench0.json` | carpeta J0, venv dedicado verificado, datos en fscratch con manifiesto |
| J1 | 11-16 | B200 preferida (`sbatch --test-only` en b200 y dgx) | 115 | preflight ≤3 min; np_nit featurizado **solo y primero** (o ya en local); un brazo real medido a solas y rampa de brazos reales limitada por memoria GPU, RAM y núcleos (añadir solo si el agregado sube ≥10 %); brazos E006 en orden A1_s0, A0_s1, A1_s1, A2_s0, A0_s2, A2_s1, todos *eager*; parada por `--minutes` en el minuto ~95; dumps secuenciales 95–105; manifiesto de salida ~108 | E006 congelado; registro endurecido; cascada de negativos + PoE con tests; np_nit con sha en `registry/cells.json`; venv CASMI dedicado; datos (~12.5–13 GB) con manifiesto; smoke CPU del job exacto en un store diminuto. Resume **no** es prerrequisito: sin él solo se lanzan brazos que terminen antes del minuto 95 con margen 1.2 |
| J2 | 11-19 (condicional) | B200 o dgx | ≤115 (medido × 1.25) | restos de J1 (`--resume`); F1–F3 (E008); E007 solo si cabe según la escala de throughput; A1h solo si E006b se congeló | informe E006; E008 congelado; fichero de folds congelado; resume probado |
| J3 | 11-22 (condicional) | dgx si el banco lo permite | ≤100 | refit con todos los datos × 3 semillas de la receta admitida; dumps np_nit del refit; embedding de la librería (1.95 M espectros, ~1 GB fp16) solo si sobra tiempo | A1_admit o A0r/A0q_ship; E009 congelado; se mantiene el interés en SR-1 (pick B o investigación) |
| J4/J5 | 11-25 / 11-28 (condicionales) | dgx o b200 | ≤100 | IsoArb (E011) y su refit, **solo** si el piloto CPU v2 pasa; si no, búfer o devolución a la tesis | piloto v2 (11-20), E011 congelado |
| J6 | desde 12-01 (opcional) | A100 | ≤100 | piloto D1 de FragWM (fold 0, `--minutes 80`), NLL por pico frente a A0 contado | código y regla D1 en el registro; carril libre; si no, después del 12-14 |

**Escala de throughput** (congelada antes de J1; R_N = it/s agregados medidos):
- R_N ≥ 38: plan tal cual.
- 25–38: J1 hace A1_s0, A0_s1, A1_s1, A2_s0 (+A0_s2); A2_s1 pasa a J2 o se descarta.
- 17–25: núcleo mínimo (E006 = A1 frente a A0; E007 después del 12-14; E008 = F1–F3; 2 semillas de refit; J4/J5 como búfer; IsoArb fuera).
- < 17, OOM o fallo de entorno: parar, guardar estado, diagnosticar y pedir el siguiente hueco en B200.
- Nunca se cambian batch, pasos, precisión o modo *eager* para ganar velocidad.

**Reglas operativas:**
- `-t 01:55:00` con `--signal=B:USR1@300` solo como red de seguridad. El trap reenvía USR1 **solo** a los procesos de entrenamiento (un Python sin manejador muere con SIGUSR1).
- Rutas por `SLURM_SUBMIT_DIR`; `set -uo pipefail` sin `-e`; `--deadline` para que un job tardío se cancele solo.
- `umask 077`, `chmod 700` en las raíces CASMI y `rsync --chmod=D700,F600 --files-from` con lista cerrada; el preflight falla si hay permisos de grupo u otros. Los datos CASMI se borran de Picasso tras el paquete de reproducibilidad.
- Venv CASMI propio construido desde un lock del `.venv` local, con la rueda de torch adecuada al driver del nodo (local: torch 2.15.0+cu132; B200 necesita sm_100). Nunca se instala nada en el venv de la tesis. HF_HOME con el snapshot de ChemBERTa (14 MB) y HF_HUB_OFFLINE=1; cachés Triton/Inductor separadas.
- Sin token de Kaggle en el HPC; `setup_b200.sh` no se usa; los agentes nunca hacen ssh (lo ejecuta el usuario).
- Coste núcleo: J1 + (J2) + (J3) ≈ 5–5.5 GPU-h; máximo 6 jobs antes del 12-07, lo que deja a la tesis 4–8 de los ~10–14 huecos.

### 4.2 Preparación solo CPU (nice 19, `CUDA_VISIBLE_DEVICES=""`, ≤6 procesos, <8 GB cada uno; la RTX 3060 no se toca)

| Tarea | Fecha | Comprobación |
|---|---|---|
| Resume completo en `train.py` (USR1 instalado antes de cargar el store; estado atómico con modelo, optimizador, scheduler, scaler, paso, best, log, RNG CPU/CUDA, generador del GPUStore; assert de `--gpu-data`; salida 99 antes de L198 si para por tiempo) | 11-12/13 | en un store diminuto, 60 pasos seguidos = 30 + USR1 + `--resume` + 30, bit a bit; results.jsonl intacto hasta completar |
| Cascada de negativos de train + máscara `neg_is_train` | 11-12/13 | nunca devuelve el ancla, sus alias ni estructuras ocultas; ≥96 % de anclas NP llenan 16 (96.9 % a ±0.5 Da) |
| PoE pre-ajustado y congelado (o con gradiente bloqueado), pérdida dura separada, monitor de lift | 11-13 | smoke de 200 pasos; b(m) ausente del state_dict de inferencia; test unitario de la separación de fwd |
| np_nit con la especificación de §3.1 | 11-13/14 | sin claves de train (proporcionada, oficial, cruda, esqueleto); 480 en rango; n_truth_in_p25 registrado; sha en `registry/cells.json` antes del congelado |
| Registro endurecido (§3.4) | 11-13/14 | *freeze+report* en sandbox sobre ficheros ficticios sin errores |
| Venv de Picasso, transferencia de datos y manifiesto, carpeta J1 (y J0) | 11-14/15 | import + matmul bf16 + un batch en el nodo de login; `sha256sum -c` remoto; grep sin credenciales; smoke CPU del job exacto |
| Congelar E006 (y E012 si dio tiempo) | 11-15 | commit + push antes de J1 y de cualquier score de np_nit |
| Métricas y auditoría (`eval_cells.py` con `--out`, `audit_membership.py`) | 11-16/17 | w4base_s0 re-keyed reproduce mass:nf 0.375 (np_c2) y 0.356–0.359 (np_tims) ±0.01; lift_fe reproduce +0.274 / +0.308 |
| Pick A: builder con imagen fijada y T4 | 11-16/17 | build sin bundle reproduce los hashes en replay; digest de la imagen de P001/eval-c3 leído una vez (decisión del usuario) |
| Overlay *public-first* (celdas 17b/18b), replay y tests | 11-17/18 | identidad de hashes; 0/430 cambios de rank 1 con ruido; el fallo inyectado termina en el final público |
| emb_v5 delta, folds, activos de clase 1 (np_c1x, np_c1x_sub a densidad ordinaria y 3×, a03/a06), A0 contado | 11-18 | ChemBERTa CPU coseno ≥ 0.9999 en 100 filas; A0 contado reproduce 0.322 (np_c2) y 0.202 (np_tims); reconstrucción de la lista *merged* del panel ≈ 0.714 frente a 0.718 publicado |
| Piloto IsoArb v2 | 11-18/20 | pasa solo si vence a MetFrag-lite (ci_low > 0) y \|lift\| ≤ 0.10, <30 min CPU |
| R1-Gate y su etapa de replay | 11-21/23 | salida byte-idéntica con el gate apagado; nunca dispara con lib_max ≥ 0.80, class1_confirmed, V indefinido, movimiento previo del fragment check o empate; permutación; tope del 8 % |
| Paquete de reproducibilidad | 11-30 → 12-04 | un clon limpio más los manifiestos reconstruye el sha del bundle; licencias y declaración sin NIST por componente |

### 4.3 Kaggle (30 GPU-h/semana; notebooks ≤9 h en T4)

- **Hoy:** leer el contador de cuota (P001 corrió el 11-10), lo que dice si los reruns ocultos cuentan.
- **Semana del 11-16:** commit de la pick A (≈0.4 h) y envío + selección; F_ext por API (0 GPU, si el usuario lo aprueba); F_nit (≈5.5 h) solo si F_ext no sirve.
- **Semana del 11-23:** commit de SR-1 (≈0.4 h) y P003 si E009 admite. El envío de la pick A hace de control; P002 deja de ser un control aparte.
- **Semana del 11-30:** commit de R1-Gate y P004 si E010 lo deja pasar a la pick B; sin control aparte de la misma semana.
- **Semana del 12-07:** sin reruns finales: un envío se puntúa una vez en ambos splits, y las picks seleccionadas no se vuelven a ejecutar.
- En el peor caso (los reruns cuentan) ≈5.9 / 9.4 / 9.4 h por semana: el presupuesto no es la restricción.

---

## 5. Calendario con puntos de decisión

| Fecha | Trabajo | Decisión |
|---|---|---|
| 11-12 (jue) | resume; cascada; leer contador de cuota; empezar contactos de merge | — |
| 11-13 (vie) | PoE; registro; np_nit; (E012 en CPU si se aprueba) | — |
| 11-14 (sáb) | np_nit congelado; venv de Picasso y datos; carpetas J0/J1 | J0 sí/no |
| 11-15 (dom) | **congelar E006** (y E012) | **D0:** si algo no está listo, J1 se desplaza como máximo 2 días; sin resume, J1 corre igual con brazos que terminen |
| 11-16 (lun) | **J1**; builder de la pick A; métricas/auditoría | — |
| 11-17 (mar) | informe E006; congelar E008 (E006b si hace falta); commit + envío de la pick A | **D1:** receta de negativos (A1 / A2 / A0r / A0q / nada) |
| 11-18 (mié) | **seleccionar la pick A en la UI**; F_ext; emb_v5, folds, activos de clase 1, A0 contado; piloto IsoArb | — |
| 11-19 (jue) | **J2** si D1 lo justifica | — |
| 11-20 (vie) | piloto IsoArb; congelar E009 | **D2:** IsoArb sí/no. Límite: si J1 no ha corrido, fuera E007/E008 |
| 11-21 (sáb) | informe E008 | **D3:** tamaño de la exposición de clase 1; receta de despliegue |
| 11-22 (dom) | **J3** si se mantiene SR-1 | — |
| 11-23 (lun) | bundle; E009; commit de SR-1; congelar E010 | **D4:** SR-1 en la pick B sí/no |
| 11-24 → 11-26 | P003 (SR-1) si D4 lo admite | — |
| 11-26 | — | **D4b:** objetivo del merge decidido (contactos iniciados el 11-12) |
| 11-27 (vie) | — | **D5:** veto de P003 (solo bug/runtime o Δ ≤ −0.03) |
| 11-29 (dom) | informe E010 | **D6:** R1-Gate en la pick B sí/no |
| 11-30 → 12-02 | commit de R1-Gate y P004 | límite: si J3 no corrió, no se envía refit |
| 12-03 (jue) | lectura de P004 | **D7** |
| 12-01 → 12-04 | reproducibilidad; erratas; E005 archivado; contingencia fe_D si se aprueba | **D8 (vie 12-04):** merge sí/no y picks provisionales |
| 12-07 (lun) | cierre de merges; nada nuevo después | — |
| 12-08 → 12-13 | congelado; publicación de código y pesos | **12-12:** decisión de cumplimiento (fe_D) |
| 12-14 (lun) | picks finales: A = LS-480 (o variante D-neutralizada si hubo fallo de cumplimiento), B = componentes admitidos o la pick del socio | — |

Holgura: 7 días naturales tras D5, pero solo ≈2 días de margen dentro de la cadena J1→J3. Un fallo de entorno en J1 cuesta ≈3 días, y la cesión a la tesis hasta 2 días por job. Las reglas de corte (J1 antes del 11-20, J3 antes del 11-30) mantienen la fecha a costa de E007/E008.

---

## 6. Riesgos y criterios de parada

| Riesgo | Mitigación | Parada |
|---|---|---|
| Ningún brazo es neutral (el prior vive en interacciones) | E012 antes del congelado; A0r/A0q sin entrenamiento | ningún brazo pasa N1–N3 el 11-20 → SR-1 no entra en ninguna pick; resultado negativo publicado |
| El brazo neutral pierde destreza | S1 frente a A0r; A1h | skill_lost para A1, A2 y A1h → la familia no se despliega |
| np_nit poco representativo (Eawag, masa, estándares típicos) | reponderación por masa, estratos, blind:nf de verdades | blind:nf de verdades claramente sobre el azar → decisión en sub-listas igualadas por pertenencia |
| Throughput bajo u OOM en A100 | rampa medida, escala congelada, B200 por `--test-only` | R_N < 17 o OOM → parar y siguiente hueco en B200 |
| Entorno de Picasso roto | venv dedicado verificado el 11-14; J0 | J1 pierde el hueco → la cadena se desplaza; recortes por la escala |
| El overlay degrada en silencio | aceptación por 7 condiciones; test de fallo inyectado | cualquier condición falla → final público |
| Imagen sin fijar (con numpy ≥ 2.5 las moléculas elegibles para un movimiento del fragment check caen de 248 a 89 de 400) | `dockerImage` + `original` + T4 en todo push, incluida la pick A | sin digest leído → no se envía |
| R1-Gate neto negativo (impuesto de clase 1 a densidad ordinaria, coaislamiento) | lib_max < 0.80, class1_confirmed, cribas de §3.4 | cualquier criba falla o P004 con bug/runtime o Δ ≤ −0.03 → fuera de la pick B |
| La mejor de las dos picks no cuenta | verificar en la página de reglas antes del 11-23 | si no cuenta, la pick B vuelve al nivel estricto «adopt» |
| fe_D/DreaMS declarado no elegible | contingencia D-neutralizada (§9.2) | fallo antes del 12-12 → ambas picks D-neutralizadas |
| Sobrecarga (una persona, ~1,000 líneas antes de J1) | camino pre-J1 recortado a resume, negativos/PoE, núcleo de np_nit, carpeta y congelado | congelado de E006 después del 11-17 → fuera IsoArb; después del 11-19 → fuera la etapa 2 |
| Conflicto con la tesis en Picasso | CASMI cede; un job a la vez | cada cesión desplaza ≤2 días |

---

## 7. Alternativas descartadas y por qué

**Rail del líder sospechoso (propuesta E): descartado como componente del LB; queda una variante solo como investigación.** El rail promueve el primer candidato no-librería con la fórmula del líder (ranks 2–6) cuando el líder es una estructura de librería. La síntesis lo descartó con un razonamiento en parte equivocado; el veredicto se mantiene, con otros números (verificación propia de `c3_rail.parquet` y `panel_feat.parquet`):

| Marco (por molécula de clase 2) | Sin gate | LIB_MAX < 0.9 |
|---|---|---|
| eval-c3, stack real | +0.0747 (57 suben / 0 bajan) | +0.0544 (42 / 0) |
| Panel, «removed from pool», lista principal antes del merge (el marco de la síntesis) | +0.0040 | +0.0020 |
| Panel, lista final *merged* (0.714 reconstruido frente a 0.718 publicado) | +0.0450 (21 / 0) | +0.0430 (20 / 0; 20 de 21 son propuestas PubChem) |
| Misma lista, canal PubChem sin documentación | +0.0100 | +0.0080 |

- El +0.004 era un artefacto de puntuar la lista antes del merge PubChem. El +0.075 de E omitía su propio gate LIB_MAX.
- En eval-c3, las 57 verdades promovidas vienen del canal principal (residentes en el pool y memorizadas); en el panel, las ganancias son propuestas PubChem fuertes detrás de un líder de librería (equivale a bajar el listón de promoción S>6 ∧ pop≥5). El valor real depende de la documentación y la residencia en pool de las verdades ocultas (PubMed>0 ≈ 34 %), que ningún marco offline representa.
- **La razón vinculante es el impuesto de clase 1**, infraestimado ≈3×: las moléculas de np-examples tienen una mediana de 91 referencias propias del mismo aducto, frente a 5 en una estructura NP ordinaria. Con referencias submuestreadas a densidad ordinaria, el impuesto del rail con coseno (τ 0.7) pasa de −0.027 a −0.089 por molécula de clase 1, y la tasa de disparo de 5.4 % a 17.8 % (26.1 % con a03).
- Neto LB = f2·x·g2 − 0.152·0.5·p_fire. A densidad ordinaria es negativo en todo τ y toda transferencia (τ 0.7: −0.011 con ×1/3, −0.005 con ×1), con P(neto>0) ≤ 0.05.
- Solo una variante tiene señal: disparar cuando el ranker principal da al líder un score bajo (AUROC 0.885 frente a 0.772 del coseno; 1 disparo en 241 listas de clase 1). Pero el umbral se eligió después de mirar (1 de 6), descansa en 9 subidas, la señal es memorización de las moléculas del panel, y la celda 31 reescribe BASE sin esos scores. Queda solo como investigación. Se pre-registra (umbral −0.5 fijado a priori) cuando existan un hook que guarde los scores de BASE antes de la celda 31, un marco de clase 1 a nivel stack (consultas de np-examples con sus espectros ocultos, ≈4.7 h de T4) y un rerun de eval-c3 con scores exportados. Se enviaría solo si la cota inferior 80 % del neto LB > 0 a densidad ordinaria, ×1/3 y f1 0.25.

**Otras alternativas:**

| Alternativa | Motivo |
|---|---|
| XCWM (JEPA entre condiciones) antes del 12-07 | incremento sobre el coseno ≤ +0.002 aun alcanzando su objetivo (P ≈ 0.35); techo de clase 1 ≈ 0.008; la idea de entrenamiento ya está en arXiv 2610.37979; tesis después del 12-14, con control de acción identidad |
| SimCover-NP (simulación a 1.1 M estructuras) | 3–4 huecos extra de Picasso; MDE80 (≈0.046) mayor que su efecto central (+0.035); el hueco pasa ≤ +0.002 por +0.10 de modelo solo; segunda ola |
| FragWM como retador de rank 1 antes del 12-07 | precisión del retador sin entrenar en np_tims ≤ 0.33 frente a 0.55 necesaria; 6–8 días de desarrollo |
| IsoArb en Picasso sin piloto CPU | el piloto de mundo cerrado perdió con MetFrag-lite sin entrenar (0.417 frente a 0.558), con lift −0.24 en np_tims |
| Gate de despliegue en eval-c3 | sus verdades son de train: favorece al modelo con lift (+0.0039 frente a −0.0015) y dio el signo LB equivocado en componentes públicos |
| Excluir de np_nit las estructuras negativas de emb_v4 | quita ~49 de 227 NP-db y una condición realista; se estratifica |
| Token de instrumento (E005) | riesgo declarado de prior «timsTOF = tipo fármaco»; sin ganancia en la literatura (MS2DeepScore 2); aplazado |
| Subir o ajustar en el LB el peso del hueco | el ruido cuesta −0.0082 con α 0.5 frente a −0.0025 con 0.25; maldición del ganador |
| Bi-encoder más grande o init JEPA | E001 negativo; efectos dentro del ruido de semilla; la tarea de entrenamiento ya satura |
| SR-1 como retador independiente de rank 1 | precisión frente al líder fusionado con documentación 0.00–0.07 |
| Componentes nuevos en ambas picks | pierde el suelo del mejor-de-dos |
| Base sin fe_D como pick por defecto | 7/7 sustituciones de FPNet/fe_D reentrenados perdieron 0.013–0.038 en el LB; se prepara como contingencia (§9.2) |
| Seed solo-Morgan en J3 por defecto | ChemBERTa sin tarjeta de licencia en HF, pero riesgo bajo y valor de SR-1 ≈ 0; solo si SR-1 se admite y ChemBERTa se disputa |
| Reserva de emergencia en Kaggle T4×2 para el refit | gasta cuota de probes para un componente de EV ≈ 0; solo como opción del usuario y nunca para contrastes de E006 |

---

## 8. Decisiones que necesitamos del usuario

1. **Picasso:** aprobar J1 (11-16), J2 (11-19) y J3 (11-22) como condicionales a D1/D4, J4/J5 solo si pasa el piloto de IsoArb, y J6 solo con carril libre. Opcional: J0 (≤30 min, 11-14/15). Elección de GPU por `--test-only` (b200|dgx). Venv CASMI dedicado. ~12.5–13 GB de datos en fscratch con permisos solo de propietario, borrados tras el 12-14. Las órdenes ssh/rsync/sbatch las ejecuta el usuario.
2. **Kaggle hoy:** leer el contador de cuota GPU (P001).
3. **Pick A:** enviar LS-480 en cuanto pase el replay (objetivo 11-16 → 11-18) y **seleccionarla en la UI de inmediato**; autorizar leer una vez por API el digest de la imagen de P001/eval-c3 para fijarla.
4. **Reglas:** comprobar en el navegador que cuenta la mejor de las dos picks (la página de reglas no se puede leer sin JavaScript; el host dijo «I believe so»), y si se publicarán las respuestas del test o se admitirán envíos tardíos tras el 12-14.
5. **Política de la pick B:** aceptar el cambio de objetivo (§9.1) solo si se confirma el punto 4.
6. **F_ext / F_nit:** aprobar la descarga sin GPU de los outputs de casmi26-eval-ext/-ext-b. F_nit (≈5.5 GPU-h y un dataset privado con registros CC BY-NC-SA) solo si F_ext no sirve, con el bonus del bit de depositante neutralizado y pre-registrado.
7. **E012:** aprobar el chequeo CPU que carga w4base_s0 (<15 min, <4 GB, nice 19), si da tiempo antes del 11-15.
8. **Merge:** empezar ya la búsqueda (§9.1).
9. **Cumplimiento:** preparar o no la contingencia D-neutralizada (1–2 días CPU); investigar o excluir extlib; ChemBERTa sin seed solo-Morgan salvo disputa.
10. **IsoArb y FragWM:** mantener IsoArb condicionado al piloto v2 o quitarlo; FragWM solo CPU hasta el 12-14, reutilizando código de FraGNNet si la licencia lo permite.
11. **Registro:** archivar E005 (superado: se paró en el paso 18,400 sin resume). La errata de eval-c3/P001 ya está en el Anexo C del plan.

---

## 9. Estrategia más allá del modelo

### 9.1 Picks y merge

- **Pick A** fija el suelo: reproducible, imagen fijada, seleccionada pronto. Esto evita que se autoseleccionen nuestras 0.480 no reproducibles (la autoselección por defecto de Kaggle no está verificada para esta competición) y adelanta la marca de tiempo para desempates entre forks idénticos (regla estándar, no verificada).
- **Pick B, si cuenta la mejor de las dos:** se elige por E[max(0, Δ_B)] y P(Δ_B ≥ m), con m ≈ +0.002 a +0.004 (margen privado hasta bronce), y no por «pasa gates y estimación puntual ≥ 0».
  - Valores de referencia (bootstrap de 275 moléculas privadas sobre el replay LS de eval-c3, α 0.25):

    | Brazo | Δ medio | E[max(0,Δ)] | P(Δ ≥ 0.004) |
    |---|---|---|---|
    | copia neutral | −0.0015 | +0.0006 | 0.036 |
    | ruido (4 semillas) | −0.0009 a −0.0042 | +0.0004 a +0.0011 | 0.036–0.090 |
    | retador de rank 1, s ≈ 0.0085, μ −0.002 a 0 (estimación) | — | +0.0025 a +0.0034 | P(margen) ≈ 0.28–0.36 |
    | socio de merge descorrelacionado, s ≈ 0.017, μ −0.005 a 0 (estimación) | — | +0.0046 a +0.0068 | P(Δ ≥ 0.003) ≈ 0.32–0.43 |

  - Componentes admitidos por cribas catastróficas (§3.4) y agrupados en la pick B. El veto de LB de la pick B pasa a ser solo de bug o runtime, o Δ ≤ −0.03: el veto de −0.012 tiraría un R1-Gate de media cero un 17 % de las veces moviendo su media solo ≈ −0.0007.
  - Si no se confirma la regla, todo vuelve al nivel estricto.
- **Merge (la palanca mayor):**
  - Contactos del 11-12 al 11-26 con equipos de pipeline no-fork o materialmente distinta, público ≥ ~0.475 y dispuestos a compartir las dos picks; en especial equipos frenados por la cuota de GPU (el rango 38 describe así una palanca de +0.007–0.010).
  - Lo que ofrecemos (descrito, no compartido): huecos de Picasso, herramientas de replay, marcos de clase 1 y np_nit, SR-1.
  - Antes del merge no se comparte nada en privado: el código o los datos desarrollados con la competición no pueden compartirse en privado fuera del equipo (regla citada en el foro).
  - Decisión D8 el 12-04; comprobar antes el límite combinado de envíos.
- **Objetivo:** bronce/plata (0.483/0.487 públicos, dentro del bloque de forks), no oro. Junto a E[Δ] se reporta P(rango ≤ ~510).

### 9.2 Cumplimiento de la base

- LS-480 cubre el scorer retirado y no usa pesos nuevos, pero **no** cubre fe_D (fine-tune de DreaMS dentro de ranker_v4b, cargado por el ModelBank de fpnet6), que comparten ambas picks. La pregunta NIST20+MoNA sigue sin respuesta, y el host puede retirar envíos no reproducibles «regardless of ranking».
- Contingencia (decisión del usuario, 1–2 días CPU):
  - ModelBank solo A+B; las features fzD a partir de análogos A/B, o ranker_v4b reentrenado sin D con el código público de entrenamiento;
  - desconectar el dataset DreaMS de terceros (su uso en la base es nulo);
  - medir su coste en eval-c3 y F_ext.
- Regla para el 12-12: si se declara no elegible, ambas picks D-neutralizadas; si sigue sin respuesta, se mantiene LS-480 con la variante D-neutralizada puntuada y seleccionable; si se declara elegible, sin cambios. P(que importe) ≈ 0.01–0.06 (estimación).
- ChemBERTa-77M-MTR: sin tarjeta de modelo (ni licencia ni datos) en Hugging Face; el hilo del foro solo tiene título. Riesgo bajo para un equipo sin premio.
- extlib: separa el frente de forks de 0.483–0.487 de nuestra base. Nuestro chequeo da 0/600 promociones falsas con TAU ≥ 0.70 y 1/15 de recall con 0.80, pero no tiene fuente ni licencia en disco. Se incluye en la pick B solo si se establece su procedencia; si no, se registra su exclusión.
- El paquete de reproducibilidad (código, configs, logs, registro de pesos con sha256, manifiestos y licencias, scripts de Picasso sin rutas de cuenta) se publica antes del 12-14.

### 9.3 Clase 3

- La clase 3 es el 25–46 % del test (estimación por sondas), está fuera del alcance de cualquier ranker y quizá se concentre en el privado (especulación del foro).
- Evidencia: nuestro recall en la celda de clase 3 es 0.30–0.38; un generador más amplio perdió 0.024 en el LB; el rango 38 describe pasos estables de +0.003–0.004.
- Opción solo si el merge falla y queda tiempo: inserción conservadora con gate de novedad en ranks 2–5 o en la cola (añadir guesses válidos detrás nunca baja la puntuación).
- EV +0.0005 a +0.0015 (estimación); 4–6 días CPU y un envío.
