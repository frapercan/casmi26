# Plan final de investigación CASMI26: JEPA sobre un dataset meta-clasificado y una validación calibrada al LB (revisión 2)

**Equipo:** xaxipirulazo · **Fecha:** 2026-11-09 (lunes) · **Deadline final:** 2026-12-14 · **Merge:** 2026-12-07 · **Base de partida:** plan *validation-first*, con injertos de *leaderboard-pragmatic* (base reproducible en la semana 1, decisiones por P(Δ_private>0), selección de picks simulada, etiquetado [obs]/[eval], tabla de candidatos) y de *jepa-first* (Tier M/Tier P, celdas M grandes, `pairs_xcond`, JEPA-v2 detallado, tabla de MDE). Esta revisión incorpora la revisión adversarial (hechos, fuga, viabilidad, estrategia); el apéndice «Notas de revisión» recoge lo aceptado con cambios y lo rechazado.

**Principio rector.** Antes de entrenar nada, medimos qué merece la pena tocar. Ningún componente entra en una submission, tampoco los de JEPA, si no pasa los gates de adopción G0–G4 (§4.11). Los gates se evalúan sobre celdas libres de fuga y meta-clasificadas, con potencia declarada de antemano para el efecto esperado de cada componente. La calibración con el LB solo estima los **pesos efectivos de clase p_k = π_k·κ_k**, que son lo identificable; la mezcla π no lo es. La transferencia offline → LB se modela con **riesgo explícito de inversión de signo**, porque ningún Δ offline honesto positivo ha transferido todavía con el mismo signo. JEPA es la línea de investigación central. Puede falsarse por componentes con fecha (§5.9) frente a un brazo de reconstrucción con el mismo cómputo, y sus salidas solo llegan al sistema por canales que no tocan el rank 1.

---

## 0. Resumen ejecutivo

- **Posición y ruido.** Estamos en 0.480 de LB público. El oro anda por ~0.490, a ~1.25 moléculas públicas: cada molécula vale ~0.008 sobre ~125 públicas [foro 748029, 755039]. El SE del LB público es ≈ ±0.040 y el del privado (~275 moléculas) ≈ ±0.027. Alrededor de 950 equipos son forks casi idénticos que se moverán juntos en el privado. Lo que cuenta es P(Δ_private > 0) de cada cambio, no su Δ medio.
- **Lo que ya está medido.**
  - f1 ≈ 0.16–0.20 [foro 741404; nb kitopl].
  - Set honesto (Enveda-180 Zenodo, n=400): base 0.520, oráculo sobre brazos 0.624 (+0.104), 20 % de verdades fuera del top-25. La cabeza fp sola da 0.320, pero es el mejor socio de 2 brazos (+0.061) [docs/kaggle_discussion.md §4]. Ojo: el +0.061 es oráculo bruto; el nulo de un socio barajado ronda +0.014…+0.02 (§4.8).
  - Mover el rank 1 con un modelo aprendido destruye score: w_top 0/0.25/0.5/1 → 0.472/0.472/0.448/0.359 [nb busyaprime_lb-0-477]. Nuestros re-ranks dieron 0.463/0.397/0.381.
- **Hechos verificados en local que cambian el plan:**
  1. **Ningún run nuestro de referencia usa init JEPA.** En el log de wm7c3 la línea de comandos no lleva `--init-encoder` (default `''` en `scripts/train.py:88`) y no aparece ninguna línea «encoder initialised». `train_jax.py` (wm4) no tiene opción de init, pero sí `--w-pred` (`scripts/train_jax.py:109`, la cabeza «jepa» PLM-latente). Los argumentos del job encolado en TPU no están en el repo (`exp/` solo trae `reg.json`, `scale.json`, `b200_bench.json`). **No se cancela a ciegas** (V8): si es `w_pred=0`, se mantiene, porque mide si la cabeza jepa ayuda a las demás.
  2. **Bug en `src/casmi/jepa.py`.** Los tokens enmascarados son todos el mismo `mask_tok` sin posición, de modo que el predictor da la misma predicción a todos ellos (diferencia máxima ≤ 6e-7) y la loss de tokens solo aprende la media. Además, el target EMA corre con dropout y `pretrain_jepa.py --val` usa `load_exclude` y no `load_exclude_all` (no es seguro para clase 3).
  3. **Posible atajo de canonicalización.** `plm.py` embebe SMILES crudos: positivos en formato train y negativos en formato DB. Es el patrón que llevó DreaMS+ChemBERTa de 82.41 a 6.70 hit@1 en la auditoría de MassSpecGym [arXiv 2606.19624]. Encima, los negativos duros solo cubren el 19.0 % de las moléculas de train y coinciden con los distractores del holdout1500. El 0.674–0.695 interno no se puede interpretar (el modelo solo da 0.122 en LB).
  4. **Corrección del electrón ausente en el set honesto.** `hybrid/e180_cell.py` y `hybrid/ext_pre_cell.py` no la aplican; solo lo hace `scripts/make_valset.py:85`. El honest set tiene un sesgo de ~+1.6 ppm a 348 Da respecto al test oculto [foro 743395, 748882 C2]. Además, `e180_cell.py` filtra por IK14 crudo (`~b.ik14.isin(known)`), no por `skey`.
  5. **El probe «0.382» es el ranker de candidatos de una sola celda** (−0.152 en clase 1) [docs/kaggle_discussion.md §5]. No mide la pipeline sin el scorer retirado. Todos los probes históricos (0.152, 0.122, 0.463/0.397/0.381, 0.472, 0.382) están medidos contra la base antigua con el scorer retirado, no contra B0, así que no entran en el modelo de diferencias pareadas (§4.9).
  6. **Faltan artefactos locales:** `emb/mol_table.parquet`, `emb/fp.npy` y `candidates.parquet`. Cualquier reentrenamiento empieza por regenerarlos. Además, `casmi26-pubchem-popularity-prior/casmi26-pubchem-tier/pc_mass.npy` está corrupto (80,185,393 valores difieren de la copia v4g a partir de la fila 25,690,096; misma forma y dtype, carga sin error). **La pipeline de terceros localiza ese directorio con `find('pc_smiles.npy')`** (`work/inventory/g9/frozen_0465.txt:674`), y en el mirror local eso resuelve a la copia corrupta: cualquier dump local de B0 tendría el canal PubChem mal sin avisar.
  7. **B0 y su cumplimiento.** El 0.484 público de alex0484-extlib es el cambio TAU 0.85 → 0.80 sobre la base de ozertuu (0.478 en board, que ya traía el overlay): +0.006, menos de una molécula [nb alexchilton]. **B0 = alex0484 con TAU 0.85** (comportamiento de ozertuu verbatim); TAU 0.80 pasa a ser un componente más. B0 usa (a) el motor v4b, cuyas features `model_views` incluyen fe_D, un fine-tune DreaMS de terceros cuyo código se editó después del run (pregunta al host del 11-05 sin respuesta); (b) el relleno de ranks vacíos con las listas DreaMS de hengck23 (fine-tune sin verificar; pregunta NIST20+MoNA abierta); (c) extlib, sin README ni licencia. Por eso se construye **B0-min** (sin relleno hengck23, sin extlib hasta V4, sin fe_D o con fe_D reentrenado por nosotros con el código público de ahmedberatozer y logs), y al menos una pick final se basa en B0-min.
  8. **extlib no es química NP de extractos.** Es sobre todo una librería combinatoria Orbitrap de conjugados de ácidos biliares y N-acil amidas. En una muestra de 4,000 estructuras, ~47 % tienen núcleo esteroide, ~57 % amida secundaria y ~66 % éster; 2.6 % están en COCONUT y 1.1 % tienen clase NP [inv. data:pools; muestra de revisión]. Las celdas basadas en extlib se estratifican por quimiotipo y **no** sirven de ancla.
  9. **El harness local no puede usar la imagen fijada.** Todas las copias de alex0484 fijan una imagen en `gcr.io/kaggle-private-byod` (alexchilton `sha256:9a6bcfda…`, rogersjohnson `sha256:de153e6d…`), y `podman manifest inspect` devuelve 403. No hay docker en local, solo podman 5.7.0 con CDI `nvidia.com/gpu`. La imagen pública `gcr.io/kaggle-gpu-images/python:latest` (22.1 GB) usa Python 3.13, mientras que la pipeline instala wheels cp312 (p. ej. `rdkit-2025.3.6-cp312…whl.ice`) con `sys.executable`; el entorno sin fijar ya movió un LB de 0.441 a 0.427 [foro 749908]. papermill no está en `.venv`, y el mirror local de v4g conserva el layout plano antiguo.
  10. **Las celdas sacadas de train están contaminadas para los componentes aprendidos de B0.** P-C1T: Saner vio el 89 % y spec2mol el 96 %. Pool de P-C3T: 90.5 % y 95 %. P-C3D: 446 y 457 de 484. `fpnet_full1` lo vio todo (top-1 de 0.893 en fold 0 frente a 0.669–0.692 de la fold-safe). La base da VALMRR 0.587 en la celda c3 con la verdad en DB frente a 0.520 en el honest set. P-C3T es además química no-NP (98.2 % `non_np`, 0 % en COCONUT, `np_score` mediano −1.21), y 133/484 verdades de P-C3D tienen espectros en las librerías del overlay.
  11. **La calibración original no identificaba nada y los gates no tenían potencia.** π_k y κ_k solo aparecen multiplicados. El prior de contracción s ~ U(0.3, 0.6) se apoyaba en un par mal citado (el +0.019 era la validación sobre train; en honest fue −0.057). Con las mitades `report` del 60 %, G1/G2 tenían potencia 0.14–0.74 para los efectos buscados. Todo esto se rehace en §4.6–4.11.
- **Plan en una línea por semana:**
  - S1: entorno y B0/B0-min reproducibles, registro de celdas congelado antes de entrenar, P-C2E y P-C1T con dumps, J0/J1/J2, estudio de identificabilidad.
  - S2: resto de celdas Tier P, overlay e inserción fp por gates con potencia declarada, smoke JEPA (con brazo de reconstrucción) y world model sobre el encoder del smoke.
  - S3: ablación JEPA principal con endpoint agrupado, forward, clase 3 2.0 y el primer dry run completo.
  - S4: congelar, decidir el merge solo con información pública y preparar la reproducibilidad.
  - S5: seleccionar los picks por simulación del private sobre el lockbox, usado una sola vez.

---

## 1. Estado del arte (foro, notebooks públicos, literatura)

### 1.1 Reglas, datos y scorer (host)

| Hecho | Fuente |
|---|---|
| CC-BY-NC permitido. NIST y cualquier derivado (incluidos pesos) prohibidos, con descalificación. El host puede auditar la reproducibilidad. | foro 750330, 741359, 744470 (vía 748270) |
| Permitidos: MassSpecGym y modelos entrenados con él (ICEBERG/GLACIER msg_all), pesos DreaMS, CFM-ID 4, GNPS/MassBank/MoNA nuevos (CC BY-SA ok), GNPS-NIST14-MATCHES, Enveda-180 Zenodo, PubChem/COCONUT/ChEBI/LMSD, LOTUS/NPAtlas, conteos de PubChem como feature, TabPFN, checkpoints oficiales de FRIGID | foro 748270, 745604, 742991, 744338, 745841, 748065, 748882 B1–B3, 746430, 745148 |
| El host se reserva retirar del LB las submissions no reproducibles y recalcular medallas. **Tentativo:** que baste con que una de las dos picks sea reproducible («I believe so, unless kaggles rules say otherwise», D. Healey, 11-04); sigue abierto en 753502, en un seguimiento de 748065 (11-04) y en 748882 Q2 (10-30). Las dos picks se hacen reproducibles. Un DreaMS fine-tuned compartido sin código de entrenamiento no está permitido. Los probes no seleccionados que usen pesos no elegibles a sabiendas «may impact eligibility». | foro 752541, 748065, 748882 |
| Pregunta abierta: la config de DreaMS lista NIST20+MoNA como *probing set*. El fine-tune del `embedding_model` no está verificado. fe_D (v4b) sin respuesta del host (11-05). | foro 748882 (11-05), 748799 |
| Scorer en vivo = MRR first-match v13. Quita el estereo antes de canonicalizar tautómeros y del InChIKey-14. Añadir guesses válidos detrás nunca baja la puntuación. | foro 748882 C1, 746878 |
| La corrección de −0.000548 Da en m/z positivo solo se aplicó al test oculto. Train y Zenodo no la llevan. | foro 743395, 748882 C2, 746954 |
| Split público/privado 31:69. Las tres clases aparecen en ambos; el host **no** revela si el split está estratificado. Un hilo votado especula que la clase 3 se concentra en el privado. | foro 755039 (k256.dev, 9 votos), 748882 C3 |
| `collision_energy_ev` es nominal; una lista indica espectros fusionados (24.6 % del test visible). enveda-np-examples usa esquemas que no aparecen en enveda-180: 35, 80, [20,50], hasta 5 valores con 0 eV. | foro 753419 + chequeo local |
| `test.parquet` visible = 100 % filas de enveda-180 (coincidencia exacta de picos). Nuestro mapeo: 395 e180_only + 5 e180∩public. Columnas: `molecule_id, spectrum_id, ms2_mzs, ms2_normalized_intensities, base_peak_intensity, adduct, ionization_mode, instrument_type, precursor_mz, collision_energy_orig, collision_energy_ev, collision_energy_orig_units` (sin `precursor_error_ppm`). | foro 741404; `work/inventory/visible_test_to_ik14.csv`; esquema local |
| **En disputa:** si el rerun oculto reutiliza los `spectrum_id` visibles. youyouwou dice que detectar el rerun por hash de IDs dio 0.000 (2 refs); el host (comentario de inversion, 10-07) dice «no overlap in IDs … between dummy test and rerun test». Hashear el contenido es seguro en ambos casos. El hilo 756912 está vacío en el dump. | nb youyouwou_casmi26-v131diag400:288-290; foro 745715 |
| **Por confirmar:** un comentario de 748882 (WEI ZIHE12, 10-30) cita una resolución del host, R-3544964 en el hilo 752297, que permitiría «automated, unsupervised processing across test spectra». 752297 está vacío en local; hay que leerlo a mano. | foro 748882 |
| Docker sin pin: 0.441 → 0.427 | foro 749908 |
| Locks `_champion_top1` por `molecule_id` en notebooks públicos | foro 745715; nb imranarif536, nursrijan_forensics |
| Entrenar fuera de Kaggle está permitido | foro 741876 |
| Compartir código o datos de la competición en privado fuera del equipo no está permitido; el código se comparte públicamente | foro (host), `work/forum/ALL.md:1252` |

### 1.2 Mezcla de clases y ruido del LB

- **Clase 1.** Library-only 0.151–0.152 / MRR de clase 1 0.87–0.93 da f1 ≈ 0.16–0.17 [foro 741404; nb kitopl (17.3 %); nb beraterolelk].
- **Clases 2/3, en disputa:**
  - 16/45/39 según prvsiyan. Su propia ecuación da f2 ≈ 18 %, así que es inconsistente [foro 741597].
  - 28/20/52 según Bull, con un pool sin candidatos PubChem. Su «clase 3» absorbe la clase 2 solo-PubChem [foro 748524].
  - El host habla de «many» moléculas no descritas [foro 741359].
- **Ruido:**
  - ~125–130 moléculas públicas. Una molécula de rank 2 a rank 1 vale ~0.004; de fallo a rank 1, ~0.008 [nb busyaprime; foro 748029].
  - Reruns del mismo código: ±0.002 por los time guards [nb youyouwou, rogersjohnson]. Código idéntico ha dado 0.437 y 0.440 [foro 748029].
  - La semilla mueve 0.007–0.02 [foro 748524, 743254].
  - La puntuación de la página sobrestima hasta 0.010 frente al rerun (heliosli 0.478 → 0.468) [nb youyouwou_v159extlib479].

### 1.3 Notebooks públicos: familias y deltas medidos

**Familias.**
- (A) Ahmed v4n → E006 (0.354 → 0.455): engine-1 FPNet f·z con LightGBM y generador `derive.py`, engine-2 prvsiyan/megayak, canal PubChem, ICE/GL en grupos de misma fórmula, RRF entre engines y cola metric-safe.
- (B) La «frozen 0.465» de Gluzdov con overlays (0.466 → 0.484): scorer neural de Saner leader-safe, fragment check, priors de documentación/NP/masa, extlib.
- (C) Analog propagation independiente (0.24–0.34).
- 103 de los 112 notebooks con ≥0.460 comparten el bundle v4g [nb bgunshui].

| Qué mueve el LB | Δ | Fuente |
|---|---|---|
| Quitar candidatos solo-PubChem | −0.055 (0.464 → 0.409) | nb dmitriigluzdov Data Behind |
| Promoción PubChem antes de la fusión, con veto del engine-2 / forzarla después | ≥ +0.024 / −0.031 | nb seyitkaangunes_e006-n1a, guhongbin_v459 |
| Fusión RRF de dos engines heterogéneos | +0.014 a +0.016 | nb seyitkaangunes v4n-fusion |
| ICE / GLACIER solo [M+H]+ / apilarlos en la fusión final | +0.013 / +0.006 / −0.015 | nb ahmedberatozer_v7c, amanatar_v18 |
| Proxies de aducto N1A | +0.008 (3 bases) | nb mononom, guhongbin_v498 |
| Popularidad (pool / PubChem) | +0.010 / +0.004 | nb ahmedberatozer_v7c |
| PubChemLite en engine-2 | +0.004 a +0.008 (3 bases) | nb ahmedberatozer_v14 |
| E2R2: top-1 del engine-2 en rank 2 | +0.002 (2 bases) | nb antriksharya_e2rank2 |
| Neural en ranks 2-25 (w_mid 0.75, rank 1 protegido) | +0.004…+0.012 acumulado | nb busyaprime |
| Segundo scorer en z-ensemble 0.5/0.5: 0.479 → 0.480. Pesos 0.3/0.4/0.6 → 0.474 cada uno; reemplazar → 0.460. La diferencia 0.480 frente a 0.474 (0.006) está por debajo de nuestra regla de 0.008: **0.5 es un default, no un óptimo validado** | +0.001 | nb bgunshui; alexchilton/rogersjohnson |
| Overlay extlib: TAU 0.85 → 0.80 sobre la base de ozertuu, que ya traía el overlay (página 0.482, board 0.478 → 0.484) | +0.006 (<1 molécula) por el cambio de TAU; el overlay frente a una base sin él (popwenman V22, 0.481) da +0.003, dentro del ruido de rerun (~0.002) | nb alexchilton |
| guhongbin_v586: el overlay extlib V47 dio cero promociones efectivas; K≥10 es el gate del overlay open-MS/MS V50 | ≈0 | nb guhongbin_v586:12-25 |
| rr3 FREEZE (reranker de isómeros con estructuras de train congeladas) | +0.009, dataset privado | nb heliosli_casmi26-0-486 |

**Qué pierde:**
- Cualquier modelo aprendido que mueva el rank 1.
- FPNet reentrenada con +26.5 % de Tanimoto offline: perdió 7/7 swaps (0.418–0.443 frente a 0.456) [nb kozykappa].
- Scorer long-trained (0.484 → 0.718 offline): −0.019 [nb busyaprime].
- RR8 LambdaRank: OOF 0.7283 frente a MetFrag-lite 0.6905 [nb romanrozen]; en board, V8 «RR»/«BOTH» 0.460/0.461 frente a 0.481 (≈ −0.020) [nb alexchilton, rogersjohnson].
- NX de Ahmed con peso 2: 0.472 → 0.414 (−0.058) [nb busyaprime §7, punto 5].
- JESTR fusionado: −0.022. CFM-ID: −0.018 [nb dmitriigluzdov From Spectra].
- Generador de clase 3 más ancho: −0.024. Router PubChem: −0.069 [nb wangpenghua_v29].
- rescue=3 sobre la base m1 de busyaprime: −0.006 [nb romanrozen, alexchilton].

**SSL en notebooks:**
- Ningún notebook con LB usa JEPA.
- El único JEPA público (theomainetucker, JAX) no tiene LB. Sus ideas útiles: decoder de invariantes físicas, LLR Bernoulli calibrado, término bilineal de isómeros y log-sum-exp por IK14. Re-ordena el rank 1, que es justo lo prohibido [nb theomainetucker; módulos en `work/inventory/g9/`].
- DreaMS aparece como vista fe_D fine-tuned a fingerprint, sin ganancia visible (v4b 0.354 = v1) [nb ahmedberatozer_v4l].
- Como canal de análogos empata con la búsqueda clásica en clase 2 (0.58 frente a 0.57–0.63) [nb megayak].
- El «DreaMS tail» es un no-op porque falta `dreams_np_mass.npy` [nb obstacledeveloper].

### 1.4 Lecciones de validación de la comunidad

- Los holdouts sacados de train mienten porque todos los modelos públicos vieron todo train: el ranker acierta top-1 ~93 % en local [nb huseyinemreaksoy]. Las FP públicas dan 0.49 en estructuras no vistas frente a 0.76–0.82 en vistas [nb megayak].
- enveda-np-examples está contaminado: sus 250 estructuras están en otras librerías [foro 741597]; 235/250 están en MassSpecGym.
- La clase 1 real es cross-instrument: entropía ~0.90 frente a ~0.99 entre librerías públicas [foro 756504].
- Simular clase 3 por borrado sobrestima el alcance del generador ~3× [foro 743254]. Con cap de Tanimoto del análogo (sin cap / ≤0.70 / ≤0.55) el recall es 46/32/16.5 % [foro 753400]. Un generador llega a 100/130, pero solo 7/130 quedan en rank 1 [foro 750241].
- Los flags de DB curada salen positivos en holdout y negativos en LB [foro 743254]. La trampa de popularidad infla el CV ~+0.50 [nb saddamaditya].
- ICEBERG da +0.02 en clase 2, pero ≈0 fuera de MassSpecGym [foro 748524]. Una inserción PubChem con +0.01–0.02 local fue 0.295 → 0.288 en LB [foro 748524].
- El 98 % de los errores son isómeros de la misma fórmula [foro 743254, 756715].
- Con dos proxies que discrepan, se acepta solo lo que no empeora ninguno [foro 743254].
- Plantillas reutilizables:
  - Estratos de Bull con 5 draws y un split final de 170 sin tocar [foro 748524, 748068].
  - Regímenes A/B/C con split tune/report y bootstrap [nb octaviograu].
  - `purge='all'/'lib'` [nb gengsr].
  - Bench con gates declarados de antemano: N1A +0.270 [0.117, 0.443] [nb guhongbin_v498].
  - Precisión del overlay a prevalencia natural: 66.2 %, +0.00115 [easonyanyan, en nb guhongbin_v586].
  - Probes de desacuerdo `probe`/`r1probe` [nb antriksharya_e2rank2].
  - Contraste dentro de la misma consulta (verdad en DB frente a borrada) para medir el efecto de la pertenencia a DB [foro 748524].

### 1.5 Literatura: SSL y JEPA en MS/MS

- **DreaMS** [Nat Biotechnol 2025]: masked peaks + retention order, sobre GeMS 42–201 M espectros. En library matching iguala al coseno: 0.726 frente a 0.725 [arXiv 2510.26715].
- **MSBERT**: dos vistas enmascaradas, recall@1 0.787 frente a DreaMS 0.716 en GNPS no visto [Anal Chem 2024].
- **SpecBridge** [arXiv 2601.17204]: **retirado** (v3: «We have found a problem in the preprocessing/evaluation pipeline»). No se usa como evidencia.
- **MSAlign** [arXiv 2605.19752]: InfoNCE con negativos del conjunto de candidatos da 47.5 frente a 19.5 con in-batch; la late fusion suma +9.5 a +11.8.
- **Única evidencia MS-JEPA** [arXiv 2610.37979, Tabla 1 y §6]:
  - JEPA-init frente a fresh: 0.542 frente a 0.566 (MaxSim, objetivo candidate-aware) y 0.476 frente a 0.463 (MaxSim sim-to-real).
  - El objetivo candidate-aware aporta +0.07 (preentrenado) y +0.10 (fresh).
  - Los espectros simulados aportan +0.22 solo en MaxSim fresh (0.241 → 0.463); en scoring global, +0.13 a +0.14.
  - El 0.122 de la torre ChemBERTa entrenada con candidatos es Δ0 = MRR_ciego − MRR_azar (MRR ciego ≈ 0.146), no su MRR.
  - Advierte que los objetivos con candidatos premian un prior independiente del espectro: suben el MRR ciego en 0.070–0.085. Por eso el control ciego es criterio de paso duro para J9 y J11 (§5.4).
- **Recetas JEPA:**
  - I-JEPA: multi-block 54.2 frente a random 17.6; targets latentes 66.9 frente a píxel 40.7; predictor más profundo 64.0 → 66.9 [arXiv 2301.08243].
  - data2vec: targets = media de las top-K capas normalizadas, y multi-máscara [arXiv 2202.03555, 2212.07525].
  - LeJEPA/SIGReg: λ 0.05–0.1 sin EMA; la loss correlaciona con downstream (Spearman ~0.85–0.99) [arXiv 2511.08544].
  - LeWM: AdaLN-zero, ~15M parámetros, horas en una GPU [arXiv 2603.19312].
  - IWM: la equivariancia aparece solo con predictor condicionado (MRR 0 → 0.82) [arXiv 2403.00504].
  - V-JEPA 2-AC: encoder congelado + predictor condicionado [arXiv 2506.09985].
  - JEPA-WMs: AdaLN ≈ concat, profundidad 6 [arXiv 2512.24497].
  - SALT: teacher congelado [arXiv 2509.24317].
  - VL-JEPA: InfoNCE 30.3 frente a coseno 20.2 en R@1 [arXiv 2512.10942]. Explica por qué nuestra cabeza «jepa» (regresión coseno a ChemBERTa, 0.403–0.456) pierde frente a nce.
  - Polymer-JEPA: +39.8 % (R² de EA) en el régimen con menos etiquetas y +0.4 % con 8 % de etiquetas [arXiv 2506.18194]. LeJEPA molecular: el probe gana, el fine-tune es nulo [arXiv 2609.04261]. Littwin: el sesgo útil de JEPA necesita profundidad [arXiv 2407.03475].

### 1.6 Literatura: elucidación estructural, simulación, de novo

- MassSpecGym: MIST 14.64 / 9.57 hit@1 (masa/fórmula); de novo top-1 = 0 % [arXiv 2410.23326].
- La auditoría «in the Wild» encontró fallos en ≥17 de 26 papers [arXiv 2606.19624]:
  - atajo de canonicalización (82.41 → 6.70);
  - orden por defecto de PubChem sin espectro = 49.98 %;
  - bug de padding de MIST;
  - augmentation de MIST con espectros GLACIER de las 4 M moléculas de preentrenamiento de MassSpecGym v1.0: 9.57 → 40.31 sin fuga (47.01 con fuga);
  - recomienda, para afirmaciones de generalización, excluir análogos cercanos (Tanimoto ≥0.7 o MCES<10); recomienda controles de espectro constante (es una recomendación, no un experimento reportado) y BCa con 20k remuestreos.
- En la Tabla 3 de la auditoría (reimplementación interna, CE imputada), ICEBERG da 36.18 % y GLACIER 39.47 % [arXiv 2606.19624]. El paper de GLACIER reporta Recall@1 en MassSpecGym de 69.7/70.0 (masa) y 48.5/49.9 (fórmula), e ICEBERG 2.0 63.95/44.35 [arXiv 2606.29161, Tabla 1]. Los protocolos difieren.
- MVP: espectros de consenso 26.37 → 35.99 [bioRxiv 2025.11.12.688047].
- FRIGID: acc@1 42.4 % si el FP predicho tiene Tanimoto ≥0.75, 3.0 % si está en 0.25–0.5 [arXiv 2604.16648]. MolForge con FP oráculo: 46 % [arXiv 2508.04180].
- FOAM (GA con oráculo forward): top-10 4.25 → 10.28 [arXiv 2602.07709]. MADGEN con scaffold oráculo: 1.31 → 10.5 [arXiv 2501.01950].
- Gupta & Skinnider: un modelo ciego al espectro da 9.5 % top-1 frente a 14.6 % de MIST [bioRxiv 10.64898/2026.05.03.722532].

### 1.7 Literatura: validación y estadística

- CASMI 2016: reentrenar sin los compuestos del reto baja CSI:FID de 70 a 36; los reference counts solos dan 143 frente a 82 [J Cheminf 9:22].
- CASMI 2017, rico en NP: la metadata *baja* el top-5 de 59 a 53 [Metabolites 10:260].
- Myopic MCES: los scaffold splits engañan [Nat Commun 16, 2025]. Los cluster splits son más duros que los de scaffold [arXiv 2406.00873].
- Decidir con t pareado o permutación y reportar con bootstrap; el bootstrap-shift es anticonservador [Urbano SIGIR 2019].
- Priors bajo label shift: EM de Saerens / BBSE [Neural Comput 2002; ICML 2018]. Confianza calibrada COSMIC, AUC 0.82 [Nat Biotechnol 40:411].
- Ladder [Blum & Hardt 2015]: los tests pequeños son los outliers de overfitting en Kaggle [Roelofs NeurIPS 2019].

### 1.8 Hueco atacable

Nadie evalúa la identificación bajo una mezcla oculta de clases 1/2/3 calibrada contra un LB de ~125 moléculas. Tampoco hay publicado ningún JEPA de espectros timsTOF condicionado por metadata de medida (CE, aducto, polaridad, instrumento) que se use como world model verificador y como guía de un generador anclado en análogos. Nuestro orden: (1) medición consciente de la mezcla (celdas, pesos efectivos p_k, gates con potencia); (2) JEPA condicionado, falsable rápido y comparado con un SSL de reconstrucción del mismo cómputo.

---

## 2. Diagnóstico

### 2.1 Descomposición del 0.480

LB ≈ Σ f_c·m_c, con LB_lib = 0.152 ≈ f1·m1_lib. La incógnita clave es m2: lo difícil que es la clase 2 oculta (NP documentado) comparada con nuestro set honesto (drug-like Enamine).

| Escenario | m1_lib | f1 | m1_pipe | m2_pipe | m3_pipe | f2 | f3 |
|---|---|---|---|---|---|---|---|
| A: clase 1 fácil, m2 = honesto | 0.91 | 0.167 | 0.93 | 0.52 | 0.02 | 0.62 | 0.22 |
| B: clase 1 cross-instrument | 0.75 | 0.203 | 0.78 | 0.52 | 0.02 | 0.61 | 0.19 |
| C: clase 2 NP más fácil (priors) | 0.91 | 0.167 | 0.93 | 0.65 | 0.02 | 0.49 | 0.34 |
| D: clase 2 muy documentada | 0.91 | 0.167 | 0.93 | 0.80 | 0.02 | 0.40 | 0.44 |

- Tras la clase 1 queda un residuo de ≈0.32 en todos los escenarios. f3 se mueve entre 0.19 y 0.44 según m2. **Es la misma degeneración que impide identificar π con probes:** cualquier combinación (f_k, m_k) con el mismo producto reproduce el LB. Por eso la calibración trabaja con p_k = π_k·κ_k (§4.9).
- Quitar los candidatos solo-PubChem cuesta −0.055 [nb dmitriigluzdov]. Eso son ≥7 moléculas públicas resueltas por PubChem fuera de train∪COCONUT, lo que contradice el 52 % de clase 3 de Bull.
- El set honesto no es representativo químicamente:
  - Enveda-180 tiene 4 de 182,941 moléculas en la tabla NP [inv. data:pools].
  - Mediana de depositantes en PubChem: e180 3 frente a np-examples 154 [inv. data:train-meta].
- **Las dos mediciones más valiosas** son m2 en una clase 2 NP sin fuga, estratificada por documentación (P-C2NPX), y una clase 1 NP cross-instrument (P-C1NP). La cota diagonal de f3 en el oculto (P_gen) solo se gasta si el estudio de identificabilidad dice que estrecha p3 (§4.9).

### 2.2 Correcciones de hechos antes de calibrar

| # | Hecho | Acción |
|---|---|---|
| V1 | 0.382 = ranker de candidatos de una celda, no «pipeline sin scorer retirado». Los probes históricos están pareados contra la base antigua | Reconstruir `configs/lb_probes.yaml` desde el historial (`tools/kaggle_api.py`): fecha, sub id, notebook, versión, base, hipótesis. Marcar cuáles pueden entrar en el ajuste (§4.9.1) |
| V2 | El honest set no lleva la corrección del electrón | Parchear `hybrid/e180_cell.py` y `ext_pre_cell.py` y re-puntuar; el Δ mide la sensibilidad a 1.6 ppm |
| V3 | `pc_mass.npy` corrupto en la copia popularity-prior (1,052,524,936 B frente a 847,004,040), y `find('pc_smiles.npy')` de B0 resuelve a ese directorio | Re-descargar el fichero (`tools/kaggle_api.py`, por fichero) a un directorio nuevo y montarlo con bind mount sobre la ruta corrupta dentro del contenedor. `assert sha256` en nuestros loaders **y** sha256 de todos los ficheros de `/kaggle/input` que abre B0 contra los sha que vuelca el kernel de entorno en Kaggle (V10); abortar si alguno no coincide |
| V4 | extlib sin README ni licencia (los notebooks dicen «CC0»), y **B0 lo usa online** | Preguntar en el foro (público) e identificar el origen. Hasta resolverlo, la pick A usa B0-min (sin extlib, o con overlay MassBank-h/CC0). P-B0-min es un probe obligatorio |
| V5 | El rango oculto 157–1159 Da (mediana 348) y los ~1,500 espectros solo figuran en el contexto | Localizar la fuente de ambos; decidir masa neutra o precursor; la distribución de `n_spec` pasa a ser parámetro de sensibilidad (§3.10) |
| V6 | Builder de submission | Copiar el `docker_image` pin, detectar el rerun por hash de contenido (seguro tanto si se reutilizan los IDs como si no, §1.1), `grep _champion_top1 = 0` |
| V7 | Artefactos ausentes (`mol_table`, `fp.npy`, `candidates.parquet`) | Regenerar con SMILES canónicos; es código nuevo con tope de memoria (J2, §5.4) |
| V8 | El job encolado en TPU podría no ser un duplicado | Leer sus argumentos hoy. Cancelar solo si reproduce wm4 sin cambios; si es `w_pred=0`, mantenerlo |
| V9 | Cumplimiento de B0 sin confirmar: fe_D (v4b `model_views`), relleno DreaMS de hengck23, extlib | Construir B0-min (sin relleno hengck23, sin extlib, sin fe_D) y, en S2, B0-min' con fe_D reentrenado con el código público de ahmedberatozer y logs. Seguir 753502 y 748065 |
| V10 | Imagen fijada en registro privado (403), sin docker, Python 3.13 frente a cp312, papermill ausente, layout plano del mirror | Kernel de entorno en Kaggle con «Pin to original environment» que vuelque `python -V`, `pip freeze`, `nvidia-smi`, las rutas que resuelve cada `find()` de B0 y el sha256 de cada input que abre (~10 min de GPU). Con eso se monta el harness (§4.12) |
| V11 | P-C2E se filtró por IK14 crudo; el host dice que esas estructuras «likely were filtered out in the train.parquet preparation» (filtro desconocido) | Reconstruir P-C2E por `skey`, contar alias y el pool test-like real (solo 408 de 816 ausentes tienen [M+H]+ [foro 745148]); comparar el QC (prec_err, n_peaks, n_spec) con e180 de train y post-estratificar si difiere |
| V12 | `tools/kaggle_forum.py` no lee comentarios; 745758, 752297, 750684 y 754732 ocupan 2 bytes en local | Leer a mano en el navegador 752297, 753502, 748065, 745758, 750684 y 754732 y resumirlos en `work/forum/notes.md` |

### 2.3 Nuestro modelo: 0.67 offline frente a 0.122 en LB

Tres hipótesis, todas comprobables en la semana 1:
1. **Atajo de formato SMILES.** Positivos en formato train, negativos en formato DB [arXiv 2606.19624].
2. **Solapamiento de negativos y distractores.** Los negativos cubren 50,877 de 267,831 moléculas de train (19 %) y son las ventanas de masa del holdout [inv. data:our-assets].
3. **Dominio.** holdout1500 no es timsTOF y el 30.5 % está en MassSpecGym [inv. g_models].

Encaja con las tres que la fp (Morgan, invariante al formato) sea nuestro mejor socio honesto (+0.061 bruto). Datos de los runs:
- wm7c3 (d384/L6, nce+fwd+fp+sae, hard-k 32, sin init JEPA), holdout1500: nce 0.695, fwd 0.691, fp 0.472, jepa 0.444.
- v2 (JEPA init, d256/L4): fp 0.438, nce 0.320.
- v1 → v2: fp −0.002 y nce −0.007, con JEPA confundido con el centrado del PLM y la SAE, y una sola semilla.

### 2.4 Margen por clase (contribución esperada al private)

| Clase | Estado | Mecanismo | Δ private realista |
|---|---|---|---|
| 1 (f1 ~0.17–0.20) | m1 ~0.91–0.93 en P-C1T (drug-like, MSnLib Orbitrap). En `g8_extlib_fp.csv`, de 15 positivos a TAU 0.70 hay 3 promociones: 2 de otra estructura (cos 0.763 y 0.717, ambas extlib) y 1 correcta (MassBank-h, 0.83). Recall 1/15 (IC95 Clopper–Pearson 0.002–0.32); 0/600 falsas promociones en negativos (FPR ≤0.5 %), lo que da una precisión ≈0.73 a f1 = 0.17, no ≥0.9 | overlay calibrado + librerías abiertas (MassBank-h, GNPS/MassBank CC0, open MS/MS); embedding IWM como criterio silver | +0.002…+0.008 |
| 2 ranking (f2 ~0.4–0.6) | oráculo +0.104 en honesto; socio fp +0.061 bruto (≈ +0.04 neto del nulo); errores = isómeros | inserción en rank 2 del top-1 de fp/fwd ante desacuerdo con gate; fwd dentro del grupo de fórmula | +0.003…+0.010 |
| 2 cobertura | 20 % de verdades fuera del top-25 | rescate de ranks 26-60 / outsiders con consenso | +0.001…+0.005 |
| 3 (f3 ~0.2–0.45) | +0.026 en la celda de borrado; con el sesgo ×1/3 [foro 743254] ≈ +0.009 en MRR de clase 3; × f3 0.2–0.45 = +0.0018…+0.004, menos ≤0.001 en moléculas con la verdad en DB | inserción con gate de novedad + padres kNN + verificador | **+0.001…+0.004** |
| Reproducibilidad | la base 0.480 depende de un scorer retirado; B0 depende de fe_D, del relleno hengck23 y de extlib | B0 (alex0484, TAU 0.85) + B0-min + segundo scorer reentrenado con código público | evita pérdidas de ~0.1 |

### 2.5 El bloque de forks y la utilidad en el private

~950 equipos en 0.480–0.481 corren casi el mismo código. Sus scores privados estarán muy correlados, con una dispersión de ~±0.002 por los time guards, y las líneas de plata y bronce caen dentro de ese bloque. Consecuencias:
1. B0 debe estar en la frontera del bloque.
2. Cada componente se juzga por P(Δ_private > 0), calculada con el modelo de inversión de signo (§4.10). Un Δ esperado de +0.005 con SD 0.008 da P ≈ 0.73 sin inversión, y ≈ 0.61 con ρ = 0.25 de riesgo de inversión. Un Δ negativo nos baja cientos de puestos.
3. Ser reproducible es una ventaja relativa si el host retira submissions no reproducibles.

### 2.6 Cartera priorizada (EV por hora; estimaciones)

P(pasa gates) se recalcula con los gates nuevos y su potencia (§4.6, §4.11). Un componente con potencia <0.5 para su efecto esperado queda como «no verificable offline» y solo puede ir a la pick B.

| # | Acción | Coste | ΔLB esperado | P(pasa gates) |
|---|---|---|---|---|
| 1 | Entorno (V10) + B0 (TAU 0.85) + B0-min reproducibles, docker pin, health logs | 2 días, ~6.5 h Kaggle GPU | protege 0.48 | 0.8 |
| 2 | Auditoría de heads (canon, espectro ciego, cobertura de negativos) | ~4 GPU-h | habilitante | — |
| 3 | Celdas + B0-cellsafe + estudio de identificabilidad + simulación del private | 3 días CPU + ~50 h de pipeline (1 draw) | habilitante | — |
| 4 | Inserción en rank 2 del top-1 de fp **existente** (sin reentrenar) ante desacuerdo | 4 h | +0.002…+0.008 | 0.45 |
| 5 | Overlay de clase 1 calibrado (TAU 0.80 como componente; gold solo con ≥30 promociones correctas) | 1 día CPU | +0.002…+0.008 | 0.35 |
| 6 | fp con augmentation GLACIER (piloto 50k moléculas) | ~4 GPU-h | 0…+0.008 | 0.25 |
| 7 | Clase 3 2.0 (gate de novedad + padres + verificador) | ~15 h | +0.001…+0.004 | 0.25 |
| 8 | JEPA-v2 + world model condicionado | ~70 GPU-h | 0…+0.008 | 0.15–0.30 |
| 9 | Merge con un pipeline distinto (solo información pública antes del merge) | social | +0.005…+0.02 | ? |

La clase 3 baja del puesto 6 al 7 por su EV corregido (+0.001…+0.004).

---

## 3. Dataset meta-clasificado

### 3.1 Principios

1. **Clave única = `skey`**, la del scorer oficial (`src/casmi/metric.ik14`: quitar estereo → tautómero canónico → bloque 1 del InChIKey), más `raw_ik14_aliases`. El 1.58 % de las estructuras de train (4,370 de 275,810) tienen IK14 cruda distinta de la `skey` [nb kitopl_casmi-2026-data-pitfalls:378]; dentro de las librerías NP la proporción es ~4.4 % (2,904 de 66,490) [foro 742042]. Ocultar por la clave cruda deja escapar ~7 % [docs/kaggle_discussion.md §5].
2. **Una única `smiles_canon()`** (RDKit, sin estereo, fragmento mayor, neutralizado) para verdad, candidatos, negativos, ChemBERTa y Morgan. Un test de regresión compara contra el scorer v13.
3. **Cada campo lleva un nivel y una marca:**
   - [q] observable de consulta: se calcula **solo** con las 12 columnas de `test.parquet` (§1.1). Vale como feature de gate, router o ranker.
   - [cand] observable de candidato: propiedad de un candidato de la DB (fórmula, popularidad, NP). Vale como feature del candidato.
   - [eval] nivel verdad o evaluación: los mismos campos unidos a la `skey` de la verdad, y todo lo que dependa de la etiqueta. Solo para estratificar y evaluar; prohibido como feature.
   - `in_train` y `src` son *rails*, nunca features: la fuga de procedencia daba 0.94 frente a 0.55 honesto [nb beraterolelk]. El bit spectral-library de `pc_src_mask` (incluye NIST) nunca es feature.
   - `tests/test_obs_features.py` falla si alguna feature de gate, router o LambdaRank no se puede calcular desde `test.parquet` + la DB de candidatos.
4. **Toda exclusión se propaga a la vez** a librería (incluidas **todas las librerías del overlay**: extlib, MassBank-h, open MS/MS, CC0, NP external), train supervisado, corpus JEPA, pools de negativos, pares `xcond`, sintéticos y, en clase 3, a la DB de candidatos.
5. **No se redistribuye nada** derivado de la competición (CC BY-NC). Todo vive en `work/meta/` o en datasets privados del equipo.
6. **Se parte de lo existente:** `work/inventory/train_molecule_meta_v1.parquet` (275,810×84), `work/inventory/pools/meta_ik14.parquet` (1,389,632×72, `class_proxy_v2`), `work/inventory/g_models/leakfree_mask.parquet`, `work/inventory/train_meta2.parquet` (2.54 M espectros sin picos), `work/inventory/pools/keyhash.py`.
7. **El registro de celdas se congela antes de cualquier entrenamiento.** Su sha se escribe en el `meta.json` de cada run. Una celda dada de alta después solo se evalúa con modelos entrenados después de su alta.

### 3.2 Fuentes

| Fuente | Contenido | Licencia | Uso |
|---|---|---|---|
| train.parquet (11 libs) | 2,539,608 espectros / 275,810 ik14 | CC BY-NC | base, librería |
| Enveda-180 Zenodo 21346580 | 775 estructuras fuera de train (por IK14 crudo; se rehace por `skey`, V11) con algún aducto del test (3,072 espectros), más ~2,900 solo con aductos in-source. Solo 408 de 816 ausentes tienen [M+H]+ [foro 745148]; filtro de QC del host desconocido | CC BY 4.0, aprobado [foro 745148] | P-C2E, P-C2Hisf |
| casmi26-extlib | 200,724 espectros / 151,338 ik14; 144,924 fuera de train; ~17.3k en el tier PubChem; 98 % LC-ESI-Orbitrap positivo; quimiotipo de conjugados | sin documentar (V4); B0 lo usa online | P-C2X, P-C3X (estratos), overlay (solo tras V4), corpus |
| MassBank harmonized | 139,744 registros (MS2 118,402; 10,018 ik14; 285 MS2 fuera de train) | por registro | clase 1 |
| NP external_reference_spectra | 1,384 espectros / 782 claves | CC BY / CC0 | clase 1 |
| sergiomalv open MS/MS (**no local**) | 421,671 espectros / 60,377 ik14 (MSnLib, NEXUS, GNPS, MassBank, Eawag) [nb seyitkaangunes_e006-n1a] | por registro | P-C2NPX, P-C1NP, clase 1. Se descarga el lunes y se cuentan las NP fuera de train∪MSG∪extlib antes de comprometer P-C2NPX |
| senanuretin GNPS CC0 / MassBank CC0 2026.03 | ~3,051 + ~4,567 | CC0 | clase 1 |
| PubChem tier (copia **v4g**) + delta + growth | 105,875,489 filas / 89,944,372 ik14 (150–1250 Da, CHNOPS+hal) + 36,277 + 11,177 | dominio público | pertenencia, densidad |
| Popularity prior | lsid, lpmid, lpatent, lmesh, lsyn, depositors, vendors, src_mask, np bits, LOTUS refs, formula isomers | PD / CC BY / CC BY-NC | popularidad |
| COCONUT (unión de 3 copias), LOTUS, NPAtlas, ChEBI r255, LMSD, PubChemLite | 524,403 / 149,983 / 33,497 / 77,997 / 20,436 / 385,760 | CC BY 4.0, CC BY-NC, CC0 | pertenencia |
| NP knowledge table | 153,964 con NPClassifier, taxonomía, refs, `npatlas_first_year`, unidades de azúcar y acilo | LOTUS CC BY, NPAtlas CC BY-NC | clase NP |
| MassSpecGym ik14 | 28,937 (28,697 en train) | MIT | estrato de fuga ICE/GL/DreaMS |
| Folds identity/scaffold (Gluzdov) | 277,566 filas, 5 folds (fold 0 = split del FPNet fold-safe, verificado). `fold_identity` y `fold_scaffold` son independientes | Apache-2.0 / términos de competición | Tier M, B0-cellsafe |
| casmi26-fold-safe-fpnet | 5 FPNet fold-safe (`fpnet_fold0..4`) | público | B0-cellsafe |
| DreaMS | embeddings de 1,153,785 e180 + 231,104 MSG × 1024 (fine-tune sin verificar); `ssl_model.ckpt` (Zenodo 10997887, GeMS; **no local**). GeMS (MassIVE) no se puede purgar de nuestras celdas; el solape no está comprobado | MIT / CC BY 4.0 | baseline, teacher (solo SSL en submission) |
| GLACIER/ICEBERG msg_all | checkpoints locales; GLACIER 70.8 pred/s y 0.46 GB en T4, ICEBERG 24.4 pred/s y 5.23 GB (`casmi26-eval-c3/*/meta.json`) | MIT, aprobados [foro 744338] | sintéticos, verificador |

### 3.3 Tablas y tamaños (`work/meta/v2/`)

| Archivo | Unidad | Filas | Tamaño |
|---|---|---|---|
| `molecules.parquet` | `skey` | ~1.45 M (universo U = train ∪ Zenodo ∪ extlib ∪ MBH ∪ open MS/MS ∪ pools) × ~140 col | ~300 MB |
| `fp_ecfp4.npy` | `skey` | ECFP4 2048 sin quiralidad, uint64[32] | ~370 MB |
| `spectra.parquet` | espectro | ~3.4 M (train + Zenodo + extlib + MBH MS2 + open MS/MS + NP ext + CC0) × ~45 col, sin picos | ~200 MB |
| `pairs_xcond.parquet` | par de espectros de la misma `skey` | ~1.2 M | ~60 MB |
| `synth/registry.parquet` + `synth/*.parquet` | espectro sintético | según J10 (piloto 300k espectros) | ~1–3 GB |
| `cells.parquet` (registro) + `cells/<CELL>/` | consulta | ~40k (incluidas las M ampliadas) | <20 MB |
| `cand/<CELL>.parquet` (tabla C) | consulta × candidato | ~100 por consulta (celdas P) | ~250 MB |
| `exclude/{<CELL>,ALL_CELLS,ANALOGS_07,ANALOGS_055}.txt` | `skey` | — | — |

### 3.4 Campos de `molecules.parquet`

`molecules.parquet` va por `skey` y sirve a dos usos. Unido a un **candidato**, los campos marcados [cand] son features legítimas. Unido a la **verdad** de una consulta, todo es [eval]. Los observables de consulta [q] viven en `cells/<CELL>/queries.parquet` y se calculan solo con `test.parquet`.

| Grupo | Campos | Cálculo | Tag (candidato / verdad) |
|---|---|---|---|
| Identidad | `skey`, `raw_ik14_aliases`, `smiles_canon`, `formula`, `exact_mass`, `n_heavy`, `formal_charge`, `n_frag`, `elem_class` (CHO/CHNO/CHNOPS/hal/otros), `has_N/S/P/F/Cl/Br/I` | RDKit (meta_v1 ya trae la mayoría) | cand / eval |
| Procedencia | `n_spec_<lib>` (11 libs + zenodo, extlib, mbh_ms2, open_msms, np_ext, cc0), `n_libs`, `n_public_libs`, `inst_fams`, `has_timstof`, `ion_modes`, `adduct_classes`, `ce_schemes`, `n_testlike_spec` | groupby sobre `spectra.parquet` | eval / eval |
| Novedad | `nov_true` ∈ {1,2,3,3?}, `c1_src`, `c1_enveda_only`, `in_pubchem` (tier ∪ delta ∪ growth), `in_pubchem_ext` (+PubChemLite, ChEBI, LMSD, LOTUS, NPAtlas), `in_coconut`, `in_lotus`, `in_npatlas`, `in_chebi`, `in_lmsd`, `in_msg`, `tier_filter_reason` (mass_out/charge/element/multicomp/clean), `nov_conf` (high/med/low), `pc_spectral_dep_bit` (incluye NIST; solo flag) | §3.5; hash exacto de ik14 (`keyhash.py`) | pertenencia a DB = cand; resto eval / eval |
| Química NP | `np_score` (Ertl), `np_pathway/superclass/class` (tabla NP; para el resto, NPClassifier offline con los modelos de Zenodo 5068687 en CPU, solo celdas y sus candidatos; si no, transferencia kNN con Tanimoto ≥0.7 marcada `npc_src=knn`), `classyfire_superclass` (API cacheada, solo celdas, opcional), `n_glyco`, `n_acyl`, `kingdom`, `top_genus`, `npatlas_first_year` | `scripts/meta/npclass.py` (entorno aparte, §3.10) | química = cand, taxonomía y año = eval / eval |
| **Quimiotipo** | `chemotype` ∈ {steroid_conj, acyl_amide, nucleoside_conj, glycoside, other}, con flags `has_steroid_core`, `has_sec_amide`, `has_ester` | SMARTS RDKit | eval / eval |
| Estructura | `murcko`, `generic_murcko`, `butina06` (cluster ECFP4 a 0.6), `fold_identity`, `fold_scaffold`, `fold_cluster`, `scaffold_novel_f0` (Murcko ausente de los folds 1–4) | RDKit + folds de Gluzdov | eval / eval |
| Distancia a train | `nn_tan_train`, `nn_tan_e180`, `nn_tan_publib` (ECFP4, sin self ni propio fold), `nn_tan_train_f14` (contra folds 1–4 ∪ corpus de preentrenamiento), `nn_tan_lib_after_hide` (por celda), `top5_tan_mean`, `n_analog_ge07`, `n_analog_ge085`, `mces_nn` (myopic MCES T=15 **solo** en celdas de clase 3 y M4, ≈1,830 moléculas × top-50 prefiltrados por Tanimoto ≥0.4 ≈ 91k pares), `spec_analog_max_tan` (top-30 análogos espectrales, lógica de `diag_class3.py`) | `scripts/meta/struct_dist.py` (popcount numba) | cand (salvo MCES) / eval |
| Popularidad | `pc_cid`, `pc_lsid`, `pc_lpmid`, `pc_lpatent`, `pc_lmesh`, `pc_lsyn`, `pc_ndep`, `pc_nvend`, `src_*` (bits expandidos **sin** el bit spectral-library), `chebi_star`, `lotus_refs`, `pop_pct_in_formula` | arrays fila-alineados, OR/max por ik14 | cand / eval |
| Densidad de candidatos alrededor de la verdad | `n_tier_5ppm`, `n_tier_10ppm`, `n_pool_10ppm`, `n_coco_10ppm`, `n_same_formula_tier` (`pc_formula_isomers`), `n_same_formula_pool`, `np_frac_window` | `searchsorted` sobre `pc_mass` v4g (con assert de sha) | — / eval |
| Exposición a modelos | `fsfp_oof_model`, `spec2mol_unseen`, `saner_unseen`, `ours_<run>_unseen`, `seen_by_public_fpnet`, `in_dreams_e180` | `leakfree_mask.parquet` + runs nuevos | eval / eval |
| Dificultad sin pipeline | `fsfp_oof_rank` (mediana 95), `lib_best_other`, `dreams_knn_tan` | modelos locales | eval / eval |
| Dificultad con pipeline (solo celdas Tier P) | `rank_<arm>`, `lib_max`, `in_top25`, `in_top60`, `pc_rel`, `glacier_cov` | stage dumps | eval / eval |
| Rango del test | `mass_bin` (<157, 157–235, 235–348, 348–500, 500–800, 800–1159, >1159), `in_test_mass_range` | — | cand / eval |
| Celdas | `cells` (lista), `pretrain_excluded`, `train_excluded` | registro | eval / eval |

**Observables de consulta** (`queries.parquet`, [q], solo columnas de `test.parquet`): `q_n_spec`, `q_adducts`, `q_polarities`, `q_stepped_ce`, `q_ce_set`, `q_prec_mz`, `q_neutral_mass` (precursor − aducto, con la corrección del electrón del test), `q_mass_bin`, `q_n_tier_10ppm` y `q_n_same_formula_window` (alrededor de `q_neutral_mass`), `q_n_peaks_floor1`, `q_lib_max` (de la búsqueda en librería), `q_quality` (§3.6).

### 3.5 Etiquetado de novedad

- **S_spec (snapshot del 2026-10-19).** Librerías no-Enveda de train ∪ extlib (si V4 lo confirma; si no, `c1_src` lo marca) ∪ MassBank-h MS2 ∪ open MS/MS ∪ NP ext ∪ CC0. Sin NIST.
- `nov_true = 1` si la `skey` tiene espectro en S_spec. Si solo lo tiene en Enveda (e180/Zenodo), se marca `c1_enveda_only=1` aparte, porque no sabemos si el host cuenta Enveda-180 como «public».
- `nov_true = 2` si no es 1 y está en `in_pubchem_ext ∪ in_coconut` (= `class_proxy_v2`).
- `nov_true = 3` si no es 1 ni 2 y `tier_filter_reason == 'clean'` (un componente, neutra, CHNOPS+hal, 150–1250 Da). Si no es clean, `3?`: se reporta aparte y no se mezcla.
- El bit spectral-library de `pc_src_mask` **no** sirve como etiqueta (incluye NIST; recall del 22 % [inv. data:pools]).
- **Re-etiquetado obligatorio** de P-C3D y P-C3T con S_spec completo: en P-C3D, 133/484 verdades tienen espectros en extlib (57), MassBank-h MS2 (87) o NP ext (`has_public_spectra_ext` = 139) → `nov_true = 1`. Se sacan de la celda (P-C3D-clean, 351) y se reportan como estrato aparte.
- Conteos esperados en train: c1 ≈ 95k; c2 ≈ 175,850 (e180); c3-proxy e180 ≈ 4,795 (todas e180_only, 98.2 % `non_np`, 0 % COCONUT); extlib c3 ≈ 133k (quimiotipo de conjugados).
- En cada consulta de celda, **`cls_sim`** recoge lo que se oculta al sistema:
  - `c1_sim`: solo los espectros propios de la familia de instrumento de la consulta.
  - `c2_sim`: todos los espectros de la estructura, en train y en todas las librerías del overlay.
  - `c3_del`: además, la estructura fuera de las DB.
  - `c3_cap07` / `c3_cap055`: además, análogos ≥0.7 / ≥0.55 fuera de librería, DB, train, JEPA, `xcond` y sintéticos.

  Es independiente de `nov_true`.

### 3.6 Campos de `spectra.parquet`

- **Procedencia:** `suid` (fuente + fila), `skey`, `source`, `ingest_lib`, `record_licence`.
- **Ion:** `adduct`, `adduct_class` (MpH/MmH/cation/anion/multimer/in-source loss), `test_like_adduct` (los 7 visibles + [M+H−H2O]+ y relacionados [foro 741471]), `polarity`.
- **Instrumento:** `instrument_raw`, `inst_fam` (timsTOF/orbitrap/QTOF/IT/QqQ/GC/unknown; hay que corregir el mapeo de GC-APCI-QTOF y LC-ESI-Q).
- **CE:** `ce_list` (separadores `,` y `;`), `ce_n`, `ce_min/max`, `ce_ramp`, `ce_scheme` (`e180_20/40/60`, `npex_20;50`, …), `ce_scheme_npex`, `ce_unit` (eV/NCE/unknown), `ce_ev_est` (NCE→eV ≈ NCE·prec/500), `ce_missing`. Sustituye el «NaN → 30 eV» silencioso, que afecta al 13.3 %.
- **Masa:** `precursor_mz`, `prec_decimals`, `prec_err_ppm`, `prec_err_band` (relativa a la precisión de la librería; riken redondea a 2 decimales, ~14 ppm a 348 Da), `src_ppm_offset` (mediana de ppm de la fuente calculada sobre **otras** estructuras), `electron_corrected`.
- **Picos:** `n_peaks_raw`, `n_peaks_floor1` (suelo del 1 %: test 230 → 18 frente a train 27 → 14 [foro 741404]), `n_peaks_above_prec`, `prec_peak_present`, `spec_entropy`, `base_peak`.
- **Calidad (dos campos distintos):**
  - `ref_quality_tier` [eval], solo para referencias, entrenamiento y evaluación:
    - A: ppm dentro de banda, ≥5 picos tras el suelo, sin `dup_conflict` (mismo hash con otra `skey` [foro 744446]) y aducto no dímero.
    - B: cumple 3 de 4.
    - C: el resto.
  - `q_quality` [q], el único filtro permitido en inferencia y sobre consultas de celdas: ≥5 picos tras el suelo, aducto no dímero, CE parseable. Sin ppm ni `skey`. Ninguna consulta de celda se filtra por `ref_quality_tier` (§3.11.5).
- **Indexación:** `fold_*`, `cells`, `jepa_corpus` (bool), `dreams_row`.

### 3.7 `pairs_xcond.parquet` (acciones del world model)

Columnas: `row_a, row_b, skey, d_ce, d_adduct, d_mode, inst_a, inst_b, pair_type`. Tipos:

| Tipo | Origen | Volumen |
|---|---|---|
| `ce_ladder` | e180, ~6.3 espectros y 4 patrones de CE por molécula; 4 pares por molécula | ~700k |
| `adduct` | [M+H]+ ↔ [M+Na]+ / [M+NH4]+ / pérdida de agua | ~150k |
| `polarity` | 36.8 % de las e180 tienen ambas polaridades | ~200k |
| `xinst` | e180 timsTOF ↔ Orbitrap (solo las e180∩public fuera del fold 0; ver abajo), np-examples fuera del fold 0, riken∩mona (11,207), pluskal∩gnps (15,140) | ~100k |

**Regla de disyunción.** De las 2,288 e180∩public, las 414 del fold 0 de identidad forman a la vez P-C1T y M5 (solo evaluación). `xcond-train` sale de las 1,874 restantes, quitando los clusters `butina06` que tocan el fold 0 (~1,400). Ningún par `xinst` toca una estructura ni un cluster de P-C1T/M5 o de P-C1NP.

### 3.8 Registro de sintéticos (`synth/registry.parquet`)

Espectros GLACIER ([M+H]+ y [M+Na]+, CE {20,40,60}) de candidatos LOTUS/COCONUT/PubChemLite en 157–1159 Da.
- **Excluye** `ALL_CELLS` y `ANALOGS_07`. La fuga infló MIST de 40.31 a 47.01 [arXiv 2606.19624].
- Se marcan `synthetic=1` y `sim_model=glacier_msg_all`.
- Volumen: piloto de 50k moléculas × 2 aductos × 3 CE = 300k predicciones ≈ 1.2 h en T4 al throughput medido (70.8 pred/s). Se escala a ~150k moléculas (~3.5 h) solo si el piloto pasa J10 (§5.4).

### 3.9 Registro de celdas y tabla C

- **`cells.parquet`:** `cell`, `qid` (aleatorio, esquema del test), `skey`, `cls_sim`, `nov_true`, `split` ∈ {`tune_hist`, `work`, `lockbox`}, `draw` (0 en Tier P; 0–4 en Tier M), observables [q] y estratos [eval].
  - `tune_hist`: moléculas ya usadas en decisiones previas (las 400 del honest set histórico: socio fp, pesos de re-rank, router, cell ranker, inserción de clase 3). Solo sirven para afinar.
  - `lockbox`: 20 % de cada celda por hash de `skey`, tomado **solo** de moléculas nunca usadas. Se abre una única vez, en el `pick_sim` final del 12-10 (como los 170 de Bull [foro 748068]).
  - `work`: el resto; sobre él corren G1/G2 con cross-fit.
- **`cand/<CELL>.parquet` (tabla C):** unión del top-60 de cada brazo (~80–120 candidatos por consulta). Columnas:
  - rangos y scores de cada brazo (final, V0/V2/V3/CUSTOM, engine-1, engine-2, PubChem, extlib);
  - `lib_max`, `lib_sim_cand`, ICE/GL z + cobertura, FPNet fold-safe f·z;
  - nuestras cabezas canónicas (fp-LL, fwd, nce), similitud de análogo latente, surprise;
  - ppm centrado por modo, popularidad [cand], flags NP, tamaño del grupo de fórmula y z dentro del grupo;
  - cobertura por etapa del dump (ICE/GL puntuadas, profundidad PubChem) y flag de truncado por tiempo;
  - `y`.

  Sin `in_train` ni `src` como features.

### 3.10 Scripts (`scripts/meta/`, siempre con `.venv/bin/python`)

Dependencias que faltan en `.venv` (myopic_mces, pulp, tensorflow/keras para NPClassifier, dreams, papermill) van a un entorno uv aparte, `.venv-meta`, probado el lunes con 100 moléculas.

```bash
.venv/bin/python scripts/meta/canon.py --selftest            # smiles_canon(), skey(); regresión vs metric.ik14
.venv/bin/python scripts/meta/zenodo_e180.py --by-skey --out work/ext/zenodo   # port de hybrid/e180_cell.py por skey (+ in-source); cuenta alias y pool test-like
.venv/bin/python scripts/meta/build_spectra_meta.py --out work/meta/v2/spectra.parquet       # ~25 min, <8 GB RAM
.venv/bin/python scripts/meta/build_universe.py --base work/inventory/train_molecule_meta_v1.parquet \
    --pools work/inventory/pools/meta_ik14.parquet --leak work/inventory/g_models/leakfree_mask.parquet \
    --out work/meta/v2/molecules.parquet                                                      # ~15 min
.venv/bin/python scripts/meta/label_novelty.py --in work/meta/v2/molecules.parquet \
    --overlay-libs extlib,mbh,open_msms,cc0,np_ext                                            # ~10 min
.venv/bin/python scripts/meta/chemotype.py --in work/meta/v2/molecules.parquet              # SMARTS, ~10 min
.venv/bin/python scripts/meta/struct_dist.py --ref train,e180,publib,f14 --threads 12       # Tanimoto, ~1 h (ANALOGS_07)
.venv-meta/bin/python scripts/meta/struct_dist.py --mces-cells P-C3T,P-C3X,P-C3D,M4 --mces-prefilter-tan 0.4   # ~91k pares, S2
.venv/bin/python scripts/meta/cand_density.py --assert-sha e0d7b6d7 --ppm 5,10              # ~5 min
.venv-meta/bin/python scripts/meta/npclass.py --only cells,cell_candidates --threads 12     # ~2–4 h CPU, S2
.venv/bin/python scripts/meta/xcond_pairs.py --out work/meta/v2/pairs_xcond.parquet
.venv/bin/python scripts/meta/build_cells.py --cell P-C2E --by-skey --electron --draws 1 \
    --nspec-mix visible,scaled3.75 --tune-hist hybrid/ext_pre_cell.py:seed0 --lockbox 0.2
.venv/bin/python scripts/meta/build_cells.py --cell P-C1T --fold 0 --electron --draws 1 --lockbox 0.2
.venv/bin/python scripts/meta/build_cells.py --cell M1 --fold 0 --n 5000 --testlike --electron --draws 5
.venv/bin/python scripts/meta/build_cells.py --cell M2s --fold 0 --scaffold-novel --n 5000 --electron
.venv/bin/python scripts/meta/build_cells.py --cell M4 --fold 0 --np --max-nn-tan-f14 0.7 --n 3000
.venv/bin/python scripts/meta/leak_audit.py --cells work/meta/v2/cells --all-corpora \
    --overlay-libs extlib,mbh,open_msms,cc0,np_ext                                            # exit 1 si falla
.venv/bin/python scripts/meta/freeze_registry.py --out work/meta/v2/cells/REGISTRY.sha   # antes de entrenar
.venv/bin/python scripts/meta/pipeline_features.py --cell P-C2E --dumps work/runs/pipe_B0min/P-C2E
```

Cada celda escribe `queries.parquet`, `answers.parquet`, `exclude_{lib,db,train,pretrain}.txt` y `meta.json` (semilla, sha de inputs, filtros, sha del registro).

**Espectros por consulta.** `make_valset.SPEC_PER_MOL` es el histograma del test visible (1,213 espectros para 400 moléculas, 3.03 por molécula), pero el oculto tiene ~1,500 para ~400 (~3.75; fuente por documentar en V5). Con un solo espectro el honest set da 0.159, así que `n_spec` es el estrato observable más fuerte. Las consultas se sortean de una **mezcla 50/50** del histograma visible y del histograma escalado a media 3.75, de modo que se puede reponderar a cualquiera de los dos objetivos (post-estratificación por `q_n_spec` 1 / 2–3 / ≥4). El honest set histórico usó todos los espectros de cada molécula y no es comparable con la P-C2E nueva.

### 3.11 Controles de fuga (asserts en `leak_audit.py` y en el harness)

1. Se oculta por `skey` y por todos sus alias.
2. En celdas de clase 2 y 3, la búsqueda por clave exacta devuelve 0 hits **en la librería de train y en cada librería del overlay** (extlib, MassBank-h, open MS/MS, CC0, NP ext) [foro 741597]. En clase 3, además, 0 hits en ninguna DB de candidatos.
3. En clase 3 se excluyen análogos con Tanimoto ≥0.7 o MCES<10 (y ≥0.55 en `cap055`) de librería, DB, train, corpus JEPA, `xcond` y sintéticos. Se reporta por cap [foro 753400; arXiv 2606.19624].
4. `pretrain_jepa.py --val` pasa a leer `exclude_all`. El corpus JEPA excluye `ALL_CELLS ∪ ANALOGS_07`.
5. Ninguna cantidad derivada de la etiqueta selecciona consultas. Filtrar por ppm en las consultas infló el MRR a 0.958 [nb kitopl]. Ese filtro va solo en la referencia (`ref_quality_tier`).
6. Todo SMILES pasa por `smiles_canon`, y se registra el hash de la función.
7. Las métricas se cruzan con `seen_by_*` e `in_msg`. ICE/GL/DreaMS solo cuentan en moléculas no-MSG [foro 748524].
8. El informe por terciles de `pc_lsid` de la verdad es obligatorio (trampa de popularidad), como descriptivo. Ningún flag de DB curada se usa como regla de rank 1.
9. La corrección del electrón se aplica a toda consulta Enveda positiva: `make_valset.py` ya lo hace y se añade a `e180_cell.py`, `ext_pre_cell.py` y `make_class3.py`.
10. Assert de sha256 de los inputs: el build aborta si `pc_mass.npy` no es la copia v4g. En el harness, sha256 de todos los ficheros que abre B0 contra los del kernel de entorno (V3, V10).
11. Ninguna cantidad derivada de etiquetas o scores se calcula sobre el `test.parquet` visible [foro 743254]. Única excepción documentada: el histograma de `n_spec` (metadata de formato, sin etiqueta), y solo como uno de los dos objetivos de sensibilidad (§3.10).
12. `tests/test_obs_features.py`: toda feature de gate, router o ranker se calcula desde `test.parquet` + DB de candidatos.
13. El registro de celdas y `ANALOGS_07` se congelan antes de cualquier entrenamiento; cada run registra su sha.

---

## 4. Validación exhaustiva

### 4.1 Dos niveles

- **Tier M (modelos).** Nuestras cabezas y encoders se entrenan con el **fold 0 de identidad fuera** (más `ALL_CELLS ∪ ANALOGS_07`), que es el mismo fold del OOF del FPNet fold-safe y permite compararlos directamente [inv. g_models: ~55.1k estructuras por fold, ~36.6k e180]. La evaluación es por consulta contra tablas de candidatos fijas: `msg256` (≤256, primero misma fórmula, luego ±10 ppm) y `full10ppm`. Cuesta segundos o minutos por checkpoint, por eso las celdas M se amplían (§4.2). Aquí sí hay 5 draws de espectros. Comando: `tools/cells.py eval-model --ckpt … --cells M1,M2s,M3,M4,M5`.
- **Tier P (pipeline).** La pipeline congelada se ejecuta **con un solo draw por estructura** y stage dumps (top-60 de cada engine, lista PubChem, ICE/GL, evidencias, cobertura por etapa, al estilo de `DUMP_STAGES` [nb gengsr, huseyinemreaksoy]). Después, toda capa final se reproduce en CPU con `tools/cells.py replay`. La varianza de draw solo se mide en las capas reproducibles en CPU (búsqueda en librería, overlay), que sí admiten 5 draws.
  - **Chunks del tamaño del oculto.** Cada sesión procesa ≤400 moléculas / ~1,500 espectros, con los time guards de B0 intactos (`TIME_GUARD_HOURS = 7.5`, RR8 8.5 h, `GL_BUDGET = 4000`, presupuesto ICE) y `--cpus 4`, para que salten igual que en el rerun oculto. Cada dump registra qué etapas corrieron, cuántas moléculas puntuaron ICE/GL y la profundidad PubChem. Un chunk con etapas `skipped_due_to_runtime_budget` se marca y no entra en m[a,c] sin repetirlo.
  - **Base por celda.** En las celdas externas (P-C2E, P-C2Hisf, P-C2NPX, P-C2X, P-C3X) todos los componentes de B0 son libres de fuga, y la base es **B0-min**. En las celdas sacadas de train (P-C1T, P-C1NP, P-C3T, P-C3D-clean) la base es **B0-cellsafe** = B0-min con (a) `fpnet_fold0` en lugar de `fpnet_full1` en las celdas del fold 0, y en P-C3D-clean un sub-chunk por fold con su `fpnet_fold{k}`; (b) el scorer de Saner reentrenado por nosotros sin `ALL_CELLS` (el mismo reentreno del segundo scorer, en local); (c) los rankers LightGBM v3/v4b reentrenados sin `ALL_CELLS` si el código público lo permite, o desactivados; (d) el canal spec2mol desactivado (vio el 95–96 %). El LB sigue con B0/B0-min. Si (c) no se resuelve en S2, P-C1T, P-C1NP y P-C3T quedan como diagnóstico y no fijan umbrales.
  - **Transferencia.** B0-min y B0-cellsafe se comparan en 200 moléculas de P-C2E, donde ambos están limpios. Si |Δ| > 0.01, los Δ de brazos medidos sobre cellsafe llevan una advertencia de transferencia.
  - **Coste.** Medido en el linaje E006 (`casmi26-eval-c3`): 292 min para 430 moléculas / 959 espectros en Kaggle T4 (ICE+GL 4,618 s, engine-2 2,622 s, PubChem 3,275 s), es decir, 0.68 min/molécula con 2.23 espectros por molécula. Con 3.03–3.75 espectros por molécula, y si escala por espectro, son ~0.92 min/molécula. **Se re-mide en B0-min en el primer chunk de P-C2E** (por molécula y por espectro) antes de fijar la cola de S2.
  - **Replay con presupuesto.** El replay reproduce el presupuesto de tiempo por molécula del run oculto (la capa de clase 3 lo limita por edad de sesión, commit 244696b) y reporta qué fracción de moléculas queda truncada.
  - **Dónde corre.** En local bajo podman con el entorno reconstruido (§4.12) si D0b pasa; si no, Kaggle GPU con las celdas núcleo (P-C2E, P-C1T, P-C2NPX, P-C3D-clean, P-C1NP).

### 4.2 Celdas

| Celda | Universo | n | `cls_sim` | Instrumento | Libre de fuga para | Uso |
|---|---|---|---|---|---|---|
| **P-C1T** | e180∩public del fold 0 de identidad (414). Refs: 94 % pluskal_ms2 Orbitrap (1,478 estructuras solo pluskal, 795 pluskal+gnps en el universo completo); química drug-like (98.7 % `non_np`) | 414 | 1 cross-instrument | timsTOF → Orbitrap | overlay (no aprendido); con B0-cellsafe, FPNet y Saner | ancla de clase 1 (una de dos); calibración del overlay; no-daño en rank 1. Estratificada por fuente de referencia |
| **P-C1NP** (nueva) | NP (`any_np_db`) del fold 0 con espectros en ≥2 fuentes o familias de instrumento; consulta de una familia (preferente QTOF/timsTOF; sub-estrato de las 55 np-examples del fold 0), se oculta la familia de la consulta y quedan las refs de las otras [foro 756504]. Solo registros con ≥4 decimales de precursor y CE en eV; precursor desplazado por `src_ppm_offset` | ~300 (contar en S1) | 1 cross-instrument NP | mixto | ídem, con B0-cellsafe | segunda fuente de clase 1; prior de κ1 |
| P-C1NPex | np-examples (250; 235 en MSG) | 250 | 1 | timsTOF | — (contaminada, satura ~0.91–0.93) | solo sanity |
| **P-C2E** | Zenodo fuera de train **por `skey`**, aductos del test, **con electrón**; `tune_hist` = las 400 históricas, `work` + `lockbox` = las ~375 nunca usadas (estratificadas por aducto) | 775 (a recontar, V11) | 2 drug-like | timsTOF | todos | ancla principal de clase 2 |
| P-C2Hisf | Zenodo solo in-source, submuestra | 300 | 2 | timsTOF | todos | más n en clase 2, aductos raros (proxy N1A) |
| **P-C2NPX** | open MS/MS + delta MSnLib: NP (NPClassifier pathway ≠ NA o `np_score`>0) fuera de train ∪ MSG ∪ extlib. Registros con ≥4 decimales y CE en eV, precursor desplazado por `src_ppm_offset`. **Estratificada por terciles de documentación** (`pc_ndep`, `pc_nvend`, `pc_lsid`) y reponderada a la distribución de COCONUT∩PubChem fuera de train (mediana `pc_ndep` 6, `pc_nvend` 0) | 400 (si la descarga y el conteo lo permiten) | 2 NP | Orbitrap/QTOF | todos | segunda ancla de m2, **nunca sola**; κ2 por estrato de documentación |
| P-C2X | extlib ∩ PubChem fuera de train, `steroid_conj` ≤30 %, cap por scaffold/acilo | 250 | 2 | Orbitrap+ | todos | estrato de sensibilidad |
| P-C3T | e180 `nov_true=3` clean del fold 0 (~960 en el fold), re-etiquetada con S_spec. Química no-NP (98 % `non_np`, 0 % COCONUT, `np_score` mediano −1.21) | 400 | 3 timsTOF | timsTOF | con B0-cellsafe | **estrato** «instrumento del test, química no-NP»; no es ancla de κ3 ni criterio de J12/J13 |
| P-C3X | extlib fuera de PubChem∪COCONUT, estratificado por quimiotipo y `nn_tan_lib_after_hide` | 250 | 3 genuina | Orbitrap | todos | estrato; recall del generador por quimiotipo |
| **P-C3D-clean** | c3data sin las 133 verdades con espectros en librerías del overlay; cap 0.70 en corrida real (cap 0.55 solo por replay, marcado) | 351 | 3 simulada NP | Orbitrap/QTOF | con B0-cellsafe por sub-chunks de fold | **ancla de clase 3** en Tier P; re-dump con B0-cellsafe (los dumps de `casmi26-eval-c3` son de la base 0.465 + z-ensemble y solo sirven para recall del generador) |
| **M1** | e180 del fold 0, consultas test-like, electrón | 5,000 | 2 | timsTOF | nuestros + FPNet fold-safe | selección de checkpoint; endpoint secundario de J7; DreaMS sin exposición probable a GeMS |
| M2s | e180 del fold 0 cuyo Murcko **no aparece** en los folds 1–4 (≈37 % del fold 0, ≈13.6k disponibles) | 5,000 | 2 difícil | timsTOF | ídem | generalización estructural sin reentrenar |
| M3 | NP públicos (`any_np_db`) del fold 0 | 1,500 | 2 NP | QTOF/Orbitrap | ídem | química NP |
| M4 | NP del fold 0 con `nn_tan_train_f14 < 0.7` (sin análogo ≥0.7 en el entrenamiento ni en el corpus de preentrenamiento) | ≤3,000 (contar en S1) | 3 (régimen sin análogos) | mixto | ídem | régimen con pocas etiquetas; ancla de clase 3 en Tier M (J13) |
| **M5** | e180∩public del fold 0 (las 414 de P-C1T) + np-examples del fold 0 (55, estrato aparte) | 469 | 1 cross-instrument | timsTOF ↔ Orbitrap | ídem | invariancia instrumental, IWM |
| T1 | NPAtlas `first_year ≥ 2015` con espectros (353), excluida del entrenamiento | 353 | 3 temporal | mixto | modelos reentrenados | proxy prospectivo [CAFA] |

Notas:
- **Post-estratificación.** Pesos por bin de masa hacia el rango del test (V5), por `q_n_spec` hacia los dos objetivos (§3.10) y por aducto. Doble objetivo de química (perfil e180 y perfil np-examples) como sensibilidad.
- **Regla de los dos proxies.** Un cambio en clase 2 no debe empeorar ni el estrato timsTOF (P-C2E) ni el NP (P-C2NPX o, en su defecto, P-C2X + M3) [foro 743254]. En P-C2NPX, además, la ganancia no puede venir solo del tercil más documentado.
- **Celdas NP no-timsTOF y la regla de masa de B0.** B0 trae una regla de rank 1 ajustada al timsTOF corregido (`MASS_CENTER −1.5 ppm`, `MASS_LEAD_OFF 5 ppm` con librería <0.9) y un prior de exactitud de masa. Por eso, en P-C2NPX, P-C2X, P-C3X, P-C3D-clean y P-C1NP se reporta B0 junto a B0 con el knob `V8='BASE'` del notebook (regla de masa desactivada). La diferencia se trata como confusión instrumental al derivar m2 y κ2.
- **Pipeline sobre celdas (orden de prioridad, 1 draw, chunks ≤400):** P-C2E, P-C1T, P-C2NPX, P-C1NP, P-C3D-clean, P-C2Hisf, P-C3T, P-C3X, P-C2X. Son ~3,450 moléculas ≈ 53 h a 0.92 min/molécula (más 200 moléculas de B0-min para la transferencia). Si el coste real pasa de 1.1 min/molécula, se recortan primero P-C2X y P-C3X, luego P-C2Hisf.

### 4.3 Splits para entrenar nuestros componentes

- Exclusión fija = la unión de los `exclude_train.txt` de todas las celdas P y T1, más `ANALOGS_07` de las de clase 3. Se congela antes del primer entrenamiento (§3.1.7).
- Para las ablaciones (Tier M), además fold 0 fuera. M2s y M4 son subconjuntos naturales del fold 0 (scaffold ausente de los folds 1–4; sin análogos ≥0.7), así que no hacen falta entrenamientos separados. Para el modelo de producción se reentrena con todo menos las celdas.
- La selección de checkpoint se hace en **M1** (n=5,000). holdout1500 se abandona: está comprometido por el solape con los negativos.
- Las reglas y rankers aprendidos usan 10 folds de cross-fit por `butina06`, no por scaffold [arXiv 2406.00873; Nat Commun 16], sobre `work` (y `tune_hist`) de cada celda, sin tocar el `lockbox`. Los umbrales se eligen en 9 folds y se evalúan en el décimo; G1/G2 usan el Δ fuera de fold agregado sobre la celda completa. G1 se reporta además por separado sobre las moléculas nunca usadas.

### 4.4 Estratos

- **Estratos de decisión, pre-registrados (máximo 4):** clase de celda, `q_n_spec` (1 / 2–3 / ≥4), `q_lib_max` (<0.5 / 0.5–0.9 / ≥0.9) y polaridad. Solo estos pueden justificar una regla de enrutado o una adopción condicionada.
- **Estratos descriptivos** (se reportan siempre, nunca motivan una adopción): `q_mass_bin`, CE ramp, aducto, `q_n_peaks_floor1`, gap top1-top2, acuerdo entre brazos, `q_n_tier_10ppm`, tercil de `n_same_formula_tier`*, tercil de `pc_lsid`*, `nn_tan_lib_after_hide` (<0.55 / 0.55–0.7 / 0.7–0.85 / ≥0.85)*, `mces_nn` (<10 / ≥10)*, `np_pathway`*, `chemotype`*, `in_msg`*, `nov_true`*, fuente de referencia (P-C1T)*, tercil de documentación (P-C2NPX)*. Los marcados con * son de la verdad, [eval].

### 4.5 Métricas

- MRR@25 oficial (primer acierto entre las guesses no vacías; `src/casmi/metric.py`), top-1, top-5, recall@25, `pool_hit` y MRR | in25.
- Cada cambio se descompone en ganancia y coste.
- **Hit@1 en el panel de isómeros de misma fórmula** (la métrica más sensible [foro 743254]).
- Tanimoto del FP predicho en los bins de FRIGID [arXiv 2604.16648].
- Clase 3: recall@K del generador a presupuesto fijo, MCES@1 y Tanimoto@1.
- Gates y novedad: AUROC y ECE.
- World model: MRR de equivariancia frente a 256 latentes misma-molécula-otra-condición [arXiv 2403.00504].
- Ties pesimistas en todas nuestras cabezas (hoy `rank` solo cuenta los estrictamente mayores).
- Cobertura por etapa y fracción truncada por tiempo en cada dump y replay.
- Toda señal se reporta también como «% del gap del oráculo capturado», neto del oráculo nulo (§4.8).

### 4.6 Estadística, tamaños, MDE y potencia

- **Decisión:** t pareado o permutación sobre ΔRR por molécula, remuestreando moléculas y no espectros. Multiplicidad **entre componentes**: Benjamini–Hochberg con q = 0.10 sobre el registro global de G1 (`work/gates/registry.csv`), además de Holm dentro de cada familia de brazos. El winner's curse lo corrige el `lockbox`, que se abre una sola vez.
- **Informe:** IC BCa 95 % con 10k remuestreos; en el informe final 99.9 % con 20k [arXiv 2606.19624].
- **Semillas:** ≥3 por brazo en comparaciones de modelo (J7, incluido DreaMS-SALT si se corre). Excepción declarada para smoke tests: J3 (2 semillas) y J4 (1 semilla), que no deciden adopción. Se promedian por molécula y luego se testea pareado [foro 748524].
- **Draws:** 1 en Tier P; 5 en Tier M y en las capas reproducibles en CPU (§4.1).
- **IC absoluto** (σ_RR ≈ 0.45): n=400 → ±0.044; 775 → ±0.032; 2,000 → ±0.020.
- **MDE pareado** (80 % de potencia, α=0.05 bilateral) = 2.8·σ_d/√n; con α=0.10 unilateral el factor es 2.12. Las n son las de uso real: nunca usadas de P-C2E (375), P-C2E completa con cross-fit (775), clase 2 agrupada sin lockbox (~1,200 = P-C2E 620 + Hisf 240 + NPX 320), M1 (5,000) y M-shift agrupada (M2s ∪ M4 ∪ M5 ≈ 8,500):

| σ_d | n=375 | n=775 | n=1,200 | n=2,000 | n=5,000 | n=8,500 |
|---|---|---|---|---|---|---|
| 0.066 (variante de capa final; medido en honest) | 0.0095 | 0.0066 | 0.0053 | 0.0041 | 0.0026 | 0.0020 |
| 0.12 (inserción rank-safe) | 0.0174 | 0.0121 | 0.0097 | 0.0075 | 0.0048 | 0.0036 |
| 0.30 (modelo frente a modelo, sin promediar semillas) | 0.0434 | 0.0302 | 0.0242 | 0.0188 | 0.0119 | 0.0091 |

- **Potencia de G1** (unilateral α=0.10, antes de BH) para los efectos que buscamos:

| σ_d | Δ real | n=375 | n=775 | n=1,200 | n=2,000 |
|---|---|---|---|---|---|
| 0.066 | +0.005 | 0.57 | 0.80 | 0.91 | 0.98 |
| 0.066 | +0.010 | 0.95 | ≈1 | ≈1 | ≈1 |
| 0.12 | +0.005 | 0.32 | 0.45 | 0.56 | 0.72 |
| 0.12 | +0.010 | 0.63 | 0.85 | 0.95 | 0.99 |

- σ_d se mide en la semana 1: el de las capas finales con los dumps de P-C2E, y el de modelo frente a modelo con el par de semillas de J3 (promediar semillas puede bajarlo bastante de 0.30).
- **Regla de potencia.** Cada componente declara antes de evaluarse su Δ esperado, el σ_d medido, la n y la potencia resultante (`tools/cells.py gate --declare`). Con potencia <0.5 para su Δ esperado, el componente queda «no verificable offline» y solo puede ir a la pick B si además es rank-safe y pasa G2. Las tasas reales de falso rechazo y de falsa adopción del procedimiento se miden por replay inyectando Δ = 0 y Δ = +0.005 (`tools/cells.py gate-sim --inject 0,0.005`).

### 4.7 Controles obligatorios

- **Espectro ciego:** espectro constante y donante desordenado (`hybrid/blind_scores_cell.py`) en cada cabeza (nce, fwd, fp, JEPA, IWM). Un control ciego que recupera >1/3 de la ganancia es un fallo. En J9 y J11 es criterio de paso duro (§5.4), porque los objetivos con candidatos suben el MRR ciego en 0.070–0.085 [arXiv 2610.37979].
- **Etiquetas barajadas** para rankers aprendidos (`signal_ensemble_cell.py`).
- **Invariancia de batch:** batch 1 frente a 64, |Δscore| < 1e-4 (bug de padding de MIST [arXiv 2606.19624]).
- **Baseline ingenuo** solo de popularidad por celda (CAFA Naive).
- **Leave-out:** componentes reentrenados sin las celdas frente a la versión pública [CASMI 2016: 70 → 36]; en particular B0-min frente a B0-cellsafe (§4.1).
- **Health logs** en cada submission y en cada corrida de pipeline, con cobertura por etapa. Los fallos silenciosos de ICE costaron ≥0.02 [nb amanatar_v18].

### 4.8 Techos (oráculos) por celda

`in_DB`, `in_topK` (25/60/150), oráculo sobre brazos (honest: 0.624), mejor socio de 2 brazos (fp +0.061 bruto), oráculo de inserción (ganancia máxima de insertar el top-1 del brazo X en rank p ∈ {2,3,5} si fuera perfecto), techo del generador (verdad ∈ generados) y techo «FP oráculo» por bins de Tanimoto.

**Línea base nula.** Un reordenamiento aleatorio del top-25 ya da una ganancia de oráculo g(r) = (1/25)·Σ_{u<r}(1/u − 1/r): 0.020 con r = 2, 0.051 con r = 5 y 0.077 con r = 10; con la distribución de ranks del honest set, el nulo ronda +0.014…+0.02. Por eso a cada oráculo de socio se le resta el máximo entre el de un socio barajado y el de un socio ciego al espectro (solo popularidad). **Si la ganancia neta de un brazo nuevo frente a B0 es < +0.01, ese brazo no se integra** [docs/kaggle_discussion.md §3].

### 4.9 Calibración contra el LB (pesos efectivos p_k)

1. **Tabla de probes** (V1) en `configs/lb_probes.yaml`, cada probe con su sub id y su base.
   - Entran en el ajuste como probes **absolutos** solo los históricos cuyo brazo se puede reproducir en CPU sobre los dumps de las celdas: library-only (0.152; etapa de búsqueda en librería) y model-only (0.122; nuestras cabezas wm7c3 sobre la tabla C, aproximación marcada).
   - Los demás históricos (re-ranks 0.463/0.397/0.381, gated 0.472, ranker de una celda 0.382, 775 Zenodo en librería 0.480) están pareados contra la base antigua con el scorer retirado. No hay presupuesto para dumps Tier P de esa base, así que **no entran en el ajuste**; solo cuentan como pares cualitativos de transferencia de signo (§4.10).
2. **Matriz offline m[a,c]** para cada brazo y celda Tier P, por replay sobre los dumps (base B0-min o B0-cellsafe según la celda, §4.1), sin GPU.
3. **Modelo identificable.** π_k y κ_k solo aparecen multiplicados, así que se estima **p_k = π_k·κ_k**:
   - Diferencias pareadas: y_a − y_B0 = Σ_k p_k·(μ_k(a) − μ_k(B0)) + e_a. Probes absolutos: y_a = Σ_k p_k·μ_k(a) + e_a.
   - μ_k(a) = Σ_c w_kc·m[a,c] con **w_kc fijados por diseño, no ajustados**: clase 1 ← {P-C1T, P-C1NP} a partes iguales; clase 2 ← {P-C2E, P-C2NPX reponderada por documentación, P-C2Hisf}; clase 3 ← {P-C3D-clean}. Las alternativas (incluir P-C2X, P-C3T o P-C3X; otros pesos) son escenarios de sensibilidad. Así solo se ajustan 3 parámetros (p1, p2, p3).
   - π **no se identifica**. Solo se usa en `pick_sim` para la varianza multinomial, sorteado del prior con la restricción Σ_k p_k/κ_k = 1. Priors: κ1 ~ LogNormal con IC90 [0.8, 1.1] (equivale a m1_lib ∈ [0.75, 0.93], escenarios A/B de §2.1); κ2 ~ LogNormal sobre [0.7, 1.6], por estrato de documentación; κ3 sobre [0.5, 3]; π ~ Dirichlet(2,2,2).
   - Ruido de cada diferencia: σ_a ≈ √(n_cambios)·sd(ΔRR | cambio)/125, estimado offline. Los probes absolutos usan su varianza tipo Poisson (pocas unidades de RR).
   - Script: `.venv/bin/python scripts/calibrate_lb.py --param p --probes configs/lb_probes.yaml --cells work/meta/v2/cells --out work/calib/posterior.npz`.
4. **Estudio de identificabilidad antes de gastar probes.** `scripts/calibrate_lb.py --identifiability --planned configs/probes_planned.yaml --n-sim 2000`: sortea p del prior, simula el conjunto de probes planificado con el ruido del LB, ajusta e informa cuánto se estrecha el posterior de p2 y p3 y si cambia alguna decisión de G3 o de picks. Un probe se envía solo si reduce la SD posterior de p2 o p3 en ≥25 %, o cambia una decisión en ≥20 % de las simulaciones. Si p3 sigue con la anchura del prior, G3 deja de usar la ponderación por p y pasa a no-daño en el peor caso del prior, y las submissions liberadas (~10) van a confirmaciones A/B.
5. **Chequeo de ajuste (no de identificación).** Error leave-one-probe-out con |err| ≤ max(0.010, 1.5·SE_probe) en ≥75 % de los probes. Si falla, las celdas no representan el test: se amplían los priors de κ, se reporta y se decide solo con G1/G2 y picks conservadoras.
6. **Probes nuevos** (≤12, distintos entre sí; repetir el mismo probe no reduce el ruido de muestreo). Se priorizan señales de ≥3 moléculas públicas:
   - **P-B0, P-B0-min:** base reproducible (TAU 0.85) y base mínima conforme (§6). P-B0-min mide además el coste de cumplimiento.
   - **P_pc:** solo el canal PubChem fuera de train∪COCONUT. Mide p2 por el lado PubChem (Ahmed midió 0.037 ≈ 4.6 moléculas [nb ahmedberatozer_v3]).
   - **P_strat:** probes restringidos a subconjuntos observables (p. ej. solo moléculas con `q_lib_max<0.5`, el resto CCO; o `q_n_tier_10ppm` alto frente a bajo). Separan clase 1 del resto.
   - **P_fpdis:** el top-1 de nuestra fp **existente** (wm7c3) solo donde discrepa del r1 de B0-min; resto CCO. Mide en el oculto el valor de la inserción en rank 2 [nb antriksharya_e2rank2].
   - **P_gen** (una sola variante, la profunda, y solo si el estudio de identificabilidad la justifica): solo candidatos generados ausentes de toda DB; resto «CCO». Mide p3·μ3(gen), que es justo lo que necesita una decisión de inserción de clase 3; la señal es de ~2–5 unidades de RR, con SE relativo ~50 %.
   - **P_dup:** reenvío idéntico de B0-min para medir el ruido de rerun en nuestra cuenta.
   - Descartados: P_ext (~1 molécula, por debajo del ruido), segunda variante de P_gen. P_r1dis queda condicionado al estudio.
7. **Salida:** posterior de p (y sorteos de π implicados), que alimenta `pick_sim.py`.
8. **Estimación no supervisada de la mezcla en el notebook.** Si la lectura manual de 752297 confirma que se permite procesar a través de los espectros del test (R-3544964, citado en 748882), se añade como feature de gate candidata la estimación de la fracción de clase 1 del test sin etiquetas (`scripts/class_mix.py`, o EM de Saerens). Solo se usa si, en un replay sobre mezclas sintéticas de celdas, la política guiada por ella supera a la fija (G1–G3). Si no se confirma la regla, queda desactivada.

### 4.10 Simulación del private (`scripts/pick_sim.py`)

Para cada configuración candidata c:
1. Sortear p del posterior y κ de sus priors; de ahí π_pub.
2. **Escenarios de split.**
   - Aleatorio: π_priv ~ Multinomial(275, π_pub)/275 (SD ≈ 0.026–0.028 por proporción con π ≈ 0.25–0.3).
   - Desplazado (el host no dice si el split está estratificado [748882 C3]; 755039 especula con clase 3 concentrada en el privado): π3_priv = π3_pub + δ con δ ∈ {0, 0.05, 0.10, 0.20}, y π1 bajando en la misma cantidad.
3. Sortear moléculas por clase: de los Δ fuera de fold de `work` (simulaciones intermedias) o del `lockbox` (simulación final, una sola vez), con los w_kc fijos de §4.9.
4. **Transferencia offline → private con riesgo de inversión.** Pares honestos reales (Δ honest → Δ LB): (−0.057 → −0.017), (−0.113 → −0.083), (−0.185 → −0.099) para el re-rank con w = 0.5/0.75/1, ratios 0.30/0.73/0.54 con el mismo signo; y (+0.037 → −0.098) para el cell-ranker, atribución pendiente de V1. El +0.019 que se citaba era la validación sobre train, no el honest set. Hay más inversiones positivo → negativo en la comunidad: inserción PubChem +0.01–0.02 → 0.295 → 0.288 [foro 748524], kozykappa 0/7, busyaprime 0.718 offline → −0.019, RR8 OOF +0.037 → ≈ −0.020 (§1.3). Ningún Δ offline honesto positivo ha transferido aún con el mismo signo, así que:
   - Δ_off < 0: Δ_LB = s⁻·Δ_off + ε, con s⁻ ~ U(0.30, 0.75) (contracción observada en los tres pares honestos negativos).
   - Δ_off > 0: Δ_LB = s·Δ_off + ε, donde s es una mezcla. Con probabilidad ρ_tipo hay inversión (s ~ −U(0.3, 1.0)); con 1 − ρ_tipo, s ~ U(0.2, 1.0). s se considera no identificado hasta que haya ≥3 pares positivos honestos confirmados con las celdas nuevas.
   - Prior de ρ por tipo de cambio: cambios que tocan el rank 1 (overlay gold, reglas de rank 1) ρ ~ Beta(5,5) (media 0.5); cambios rank-safe (inserciones en rank ≥2, rescate de cola, clase 3 en rank ≥6) ρ ~ Beta(2,6) (media 0.25, ≥0.2). Las inversiones históricas vinieron de validaciones con fuga o de evaluar una celda y desplegar en todas (la clase 1 sin medir); ambas causas las cubren ahora las celdas de clase 1 y G2, pero no se asume que desaparezcan.
   - ρ_tipo se actualiza con cada A/B de LB **discriminante**, es decir, con |E[Δ_pub | sin inversión] − E[Δ_pub | inversión]| ≥ 2·SE_pub pareado, mediante la verosimilitud de la mezcla.
5. Ruido ε: el de rerun (±0.002) más el multinomial del paso 2.

Salidas por escenario: E[Δ_c], P(Δ_c > 0) y U_c = P(Δ_c > +0.002), donde +0.002 es la línea del bloque de forks y plata.

### 4.11 Gates de adopción (un único conjunto, `tools/cells.py gate --arm X --base B0min`)

- **G0, integridad y runtime.** Controles de §4.7 superados (ciego ≤ 1/3 de la ganancia y ≤ azar + 0.02 en cabezas solas; duro en J9/J11), `leak_audit` OK, `test_obs_features` OK, canonicalización uniforme, réplica de la métrica OK, health logs verdes. **Runtime medido en T4** sobre una celda de 400 moléculas / ~1,500 espectros (dry run en Kaggle): el conjunto de capas añadidas ≤45 min en total, ordenadas por EV y saltadas cuando se acaba el tiempo, manteniendo intacto el upstream.
- **G1, celda objetivo.** Declaración previa (Δ esperado, σ_d, n, potencia; §4.6). Δ fuera de fold agregado sobre las celdas objetivo de la clase (sin `lockbox`) con test unilateral α=0.10 tras BH. Se reporta además sobre las moléculas nunca usadas y el signo por fold (descriptivo).
- **G2, no-daño.**
  - Una única prueba de no-inferioridad sobre el agregado de las celdas no objetivo ponderado por p_k, en el peor caso de las esquinas del posterior de p: cota unilateral inferior al 95 % > −δ_NI, con δ_NI = 2.5·SE_agg (así un cambio neutro pasa con P ≈ 0.80). Se reporta δ_NI en absoluto (≈0.004–0.008).
  - Veto catastrófico por celda: Δ_c ≤ −0.02 con cota superior unilateral al 95 % < 0 en cualquier celda, con atención especial a P-C1T y P-C1NP. Así se habría detectado el −0.152 del ranker de una celda.
- **G3, robustez a la mezcla y a la transferencia.** Con `pick_sim` y el modelo de inversión: P(Δ_private > 0) ≥ 0.70 para la pick A y ≥ 0.55 para la pick B en el split aleatorio, y ΔLB predicho ≥ −0.002 en las esquinas del posterior (percentiles 5 y 95 de p2 y p3) y en todo el rango de δ del split desplazado. Si el estudio de identificabilidad deja p3 con la anchura del prior, G3 = no-daño en el peor caso del prior. La pick A solo admite cambios rank-safe.
- **G4, board (Ladder).** **Todo** componente adoptado lleva un A/B en el LB, también los de ΔLB predicho < +0.004. El LB solo veta: ΔLB < −max(0.008, 2·SE_pub pareado) rechaza el componente. Cada A/B discriminante actualiza ρ_tipo (§4.10). Nunca se promociona por LB [Blum & Hardt 2015].
- **Riesgo de no adoptar nada.** Si el 11-29 nada ha pasado G1–G4, la pick B puede llevar componentes «no verificables offline» que sean rank-safe y pasen G0, G2 y G4. La pick A se queda en B0-min (o B0 si V4/V9 se resuelven).

### 4.12 Harness

- **Entorno (V10).** El lunes se lanza en Kaggle un kernel con «Pin to original environment» sobre la copia de B0 que vuelca `python -V`, `pip freeze`, `nvidia-smi`, las rutas que resuelve cada `find()` de B0 y el sha256 de cada fichero de `/kaggle/input` que abre (~10 min de GPU).
  - Se prueba `podman pull` del digest exacto (se espera 403: `gcr.io/kaggle-private-byod`).
  - Si falla, se construye la imagen desde el tag público de `gcr.io/kaggle-gpu-images/python` que tenga Python 3.12 más `pip install -r freeze_B0.txt`. Alternativa: venv de uv con cpython-3.12.13 y el mismo freeze fuera de contenedor. papermill se instala en ese entorno.
  - **B0 por defecto = copia de alexchilton (digest `9a6bcfda…`)**, que documenta las tablas de board. La de rogersjohnson (`de153e6d…`) solo si el diff de código muestra que es la que puntuó en rerun.
- **Ejecución.** `tools/local_pipeline.py --nb B0min --cell P-C2E --chunk A --dump`: papermill dentro de `podman run --device nvidia.com/gpu=all --cpus 4 --memory 30g -v /kaggle/input:/kaggle/input:ro`. Los ficheros corregidos van con bind mount sobre su ruta (p. ej. el `pc_mass.npy` re-descargado sobre `casmi26-pubchem-popularity-prior/casmi26-pubchem-tier/pc_mass.npy`), y el layout de directorios se replica según las rutas `find()` del kernel de entorno. Antes de cada corrida se comprueba el sha256 de todo input contra el de Kaggle, y se aborta si alguno no coincide. Escribe en `work/runs/pipe_<base>/<cell>/<chunk>/stages/`, con cobertura por etapa.
- **D0 medible.** El top-25 local coincide con Kaggle en ≥95 % de las moléculas del smoke de 12 (D0) y del chunk A de P-C2E de 400 moléculas (D0b), con la misma cobertura por etapa. El byte-diff es demasiado exigente.
- **Ruta completa en Kaggle.** El commit de un notebook corre el smoke de 12 moléculas salvo que `IS_RERUN` sea verdadero. Un commit sobre una celda propia (IDs distintos de los del test visible) fuerza la ruta completa, porque la firma no coincide. Así se verifica de punta a punta y se obtienen a la vez los dumps de referencia del chunk A.
- `tools/cells.py run|replay|eval-model|report|gate|gate-sim`. Produce `work/cells/<arm>/<cell>/{submission.csv, rr.parquet}` y `work/cells/report_<arm>.md` con métricas por estrato, IC, potencia declarada, oráculos netos y gates.

---

## 5. Línea de investigación JEPA

### 5.1 Punto de partida

- **Lo que hay.** JEPA de bloques en m/z (3 × 0.4/3 del rango), EMA 0.996 → 1, smooth-L1 sobre tokens y CLS y varianza 0.1·relu(1−std), que nunca actúa porque std ≈ 1.07. El corpus son 1,895,419 espectros etiquetados. Solo se usó como init en v2; los runs de referencia (wm4, wm7c3) no lo usan.
- **Defectos:** tokens enmascarados idénticos, target con dropout, sin token de instrumento, CE NaN→30 eV, no es seguro para clase 3.
- **La cabeza «jepa» no es JEPA.** Es regresión coseno a ChemBERTa congelado y tiene peso 0 en `configs/weights_v2.json`. El componente útil más parecido a un world model es la cabeza fwd.
- **Prior de la literatura:** con ~2.5 M espectros etiquetados, el preentrenamiento aporta poco a la recuperación en distribución: en MS-JEPA el init JEPA queda por debajo del fresh con objetivo de candidatos (0.542 frente a 0.566) y apenas por encima en sim-to-real (0.476 frente a 0.463) [arXiv 2610.37979]; Polymer-JEPA pasa de +39.8 % con muy pocas etiquetas a +0.4 % con 8 % [arXiv 2506.18194]. Lo plausible está en el shift de instrumento, la detección de novedad y la clase 3.

### 5.2 Hipótesis (prior subjetivo)

- **H0 (reparación, 0.8):** con queries de máscara con posición y target en modo eval, la std de las predicciones enmascaradas supera 0.3× la de las visibles, y el predictor **sin contexto** no alcanza el 80 % de la reducción de loss del predictor completo. Ningún veredicto previo sobre JEPA es informativo.
- **H1 (representación, 0.35):** un JEPA-v2 congelado supera al init aleatorio y al brazo de reconstrucción (mismo cómputo) en probes de FP y en kNN cross-instrument, y queda a ≤0.01 de DreaMS-SSL en M1.
- **H2 (init, 0.15–0.25):** inicializar con JEPA-v2 la receta wm canónica mejora el endpoint agrupado M-shift (M2s ∪ M4 ∪ M5) en ≥ +0.010 frente a scratch y no queda por debajo de la reconstrucción, sin pérdida en M1 (≥ −0.005).
- **H3 (world model condicionado IWM, 0.30):** el predictor condicionado a Δc aprende la equivariancia (MRR ≥0.5) y su mapeo timsTOF → Orbitrap mejora el overlay de clase 1 a igual tasa de falsos positivos.
- **H4 (forward en latente, 0.35):** fwd en dos etapas con InfoNCE sobre candidatos de misma fórmula supera a fp y a MetFrag-lite en el panel de isómeros no-MSG, pasa el control ciego y cubre aductos que ICE/GL no cubren.
- **H5 (surprise, 0.40):** surprise más densidad latente predice «verdad fuera de DB» con AUROC ≥0.75 en el contraste dentro de la misma consulta (c2_sim frente a c3_del).
- **H6 (sintéticos, 0.25):** el piloto GLACIER sube el Tanimoto de fp en ≥0.03 en M3/M4 [arXiv 2606.19624].
- **H7 (redundancia, 0.5):** la cabeza cross-modal entrenada con InfoNCE converge a nce [arXiv 2512.10942]. Si ocurre, se elimina.

### 5.3 Arquitectura JEPA-v2 (`src/casmi/jepa.py`, `scripts/pretrain_jepa.py`)

1. **Queries de máscara con posición.**
   - **Variante B (primaria):** queries sobre una **rejilla fija de m/z** dentro de las ventanas enmascaradas. No revela dónde hay picos; el target de cada punto de la rejilla es la media ponderada de los tokens target en ±0.5 Da, o «vacío».
   - Variante A (secundaria, con control): `mask_tok + W·[Fourier(m/z), Fourier(prec − m/z)]` del pico oculto, sin intensidad. Un token de pico es `MLP([Fourier(m/z), Fourier(prec−m/z), intensidad])` (`src/casmi/model.py:43, :58`), así que la query de A revela todo menos la intensidad y el objetivo podría colapsar a predecir intensidad. Control obligatorio: un predictor que ve la query pero no el contexto. Si ese control alcanza ≥80 % de la reducción de loss de A, o ≥80 % de su ganancia en el probe, A es un atajo y se descarta.
   - El context encoder sigue viendo solo `mask_tok`.
2. `self.tgt.eval()` siempre.
3. **Targets data2vec:** media de las salidas de las últimas 4 capas, cada una con instance-norm.
4. **Multi-máscara M=4** por pasada del teacher [data2vec 2.0]. Junto con el predictor de profundidad 4 (hoy 2) puede multiplicar ×3–4 el coste por paso. Se mide en el benchmark del lunes; si el paso cuesta >2× el de v1, M baja a 2.
5. **Colapso, tres brazos:** EMA + varianza (control); EMA + VICReg completo (varianza + covarianza en CLS y tokens promediados) [C-JEPA]; **SIGReg sin EMA** (λ 0.05, 1024 proyecciones, BN-projector tras el LayerNorm) [LeJEPA, LeWM]. Con SIGReg, la loss de entrenamiento sirve de criterio de selección sin etiquetas.
6. **Máscaras:** bloques en m/z {0.4, 0.6} y bloques de neutral loss. Predictor de profundidad 4, anchura 192.
7. **Condicionamiento AdaLN-zero** (§5.6) en encoder, predictor y fwd.
8. **Auxiliares físicos baratos** desde la fórmula: masa neutra, mass defect, DBE, conteo de heteroátomos [nb theomainetucker; Polymer-JEPA].
9. **Brazo de reconstrucción (Recon-MPM)** con el mismo encoder, corpus, pasos y máscaras: predice en el espacio de entrada (m/z e intensidad de los picos enmascarados, al estilo DreaMS/MSBERT). Es el comparador que permite atribuir una ganancia al objetivo JEPA y no al SSL en general (I-JEPA: latente 66.9 frente a píxel 40.7).
10. **Tests unitarios** (`tests/test_jepa.py`):
    - std de las predicciones enmascaradas > 0.3 × std visible;
    - control sin contexto (Variante A y B);
    - invariancia al batch;
    - `exclude_all` aplicado y sha del registro congelado (ninguna `skey` de celdas en el corpus).

### 5.4 Experimentos

Los costes de referencia están medidos en **T4**: wm7c3 hizo 62,130 pasos en ~12,180 s a ~5.1 it/s (run completo ~3.5 h), y el JEPA d256/L4 de 30k pasos × 512 tardó 6,833 s. El lunes se corre en la RTX 3060 un benchmark de 500 pasos de la receta wm7c3 (bf16, que la 3060 elige en auto) y del paso de JEPA-v2, y se re-escala la tabla. Se aplica un margen ×1.3 a **todo**, J7 incluido.

| ID | Experimento | Métrica | Coste | Éxito | Falsa si |
|---|---|---|---|---|---|
| **J0** | Fix de §5.3, puntos 1–10, + `exclude_all` + tokens de instrumento y `ce_missing` + `smiles_canon`. Mínimo en S1: queries con posición, `tgt.eval()`, `exclude_all` | tests unitarios | 0 h (1–2 días dev) | tests pasan | — |
| **J1** | Auditoría del atajo en wm7c3, solo evaluación: candidatos con SMILES crudos frente a canónicos (formato DB = SMILES crudos de los pools v4g locales), espectro constante, donante desordenado y batch 1/64, en holdout1500, P-C2E (honest, con electrón) y M1 | MRR ciego de nce; gap de formato | 0.5 h | **Atajo confirmado** si el MRR ciego de nce ≥ 0.07 (3× azar 0.023) o el gap ≥ 0.10. En ese caso todas las líneas base se rehacen sobre la versión canónica | — |
| **J2** | Reconstrucción de datos (código nuevo; `scripts/build_candidates.py` lee dos datasets ausentes del mirror y `embed_molecules.py` solo toma ±10 ppm del holdout): `candidates.parquet` y `mol_table v3` con negativos de DB para **todas** las moléculas de train, ≤16 por molécula en ±10 ppm (primero misma fórmula), reutilizando una muestra fija del tier por fórmula, con un pool total ≤4–5 M filas (64 por molécula darían ~14 M filas y ~14.7 GB, que no caben con `--gpu-data`). Alternativa: `plm` en memoria host pinned con gather en GPU. Re-embedding ChemBERTa-77M y Morgan sobre `smiles_canon` | cobertura de negativos; VRAM | 1 día dev + 3 h GPU + 2 h CPU | cobertura ≥95 % (hoy 19 %) y el pool cabe en 12 GB junto al modelo | — |
| **J3** | Baseline wm-canon: `train.py --d 384 --layers 6 --fwd --hard-k 32 --w-fwd 2 --sae … --gpu-data --val <fold0+celdas> --seed {0,1}`, mitad de duración, **después del congelado del registro** | fp/fwd/nce en M1, M2s, M4, M5; panel de isómeros; **σ_d entre semillas** | 2 × 1.8 h | referencia; σ_d para la potencia de J7 (smoke: 2 semillas) | — |
| **J4** | Smoke JEPA-v2 d256/L4, 20k pasos × 512 (1 semilla, smoke): {B+SIGReg, B+EMA/VICReg, B+SIGReg con máscara 0.6 + NL, A+SIGReg con control sin contexto, **Recon-MPM**} | ratio de std enmascarada; control sin contexto; probe lineal FP (AUC media de bits) en M1 congelado; kNN misma-molécula-otro-espectro en M5; Spearman loss↔probe (SIGReg) | 5 × 1.3 h ≈ 6.5 h | mejor brazo JEPA: probe ≥ init aleatorio + 0.02, ≥ JEPA-v1 + 0.02, y no peor que Recon-MPM | ningún brazo JEPA ≥ aleatorio + 0.01 → H1 cae (F1a) |
| **J5** | Baselines congelados: aleatorio d384, encoder v2, wm7c3, DreaMS-SSL (`ssl_model.ckpt`, a descargar; recalculado en celdas M, ~1 h; los embeddings precalculados solo para investigación). Opcional: solape con GeMS por hash de picos en un shard | probes de J4 en M1 (timsTOF, exposición a GeMS improbable), M3 y M5 (posible exposición, marcada) | 2 h | vara de medir | — |
| ~~J6~~ | **Eliminado.** Sus umbrales (+0.01, \|Δ\|<0.005) quedaban dentro de un SE (~0.012). Su presupuesto pasa al brazo Recon de J4 y a semillas de J7 | — | — | — | — |
| **J7** | Ablación principal d384/L6, fold 0 fuera: init {scratch ×3, JEPA-v2-d384 ×3 (preentrenamiento 1 × ~4.5 h), Recon-MPM-d384 ×3 (preentrenamiento ~4.5 h; solo si Recon supera al aleatorio + 0.01 en J4)}. DreaMS-SALT ×3 (destilación CLS desde `ssl_model.ckpt`) solo si J5 lo pone ≥0.02 por delante de JEPA-v2 en M1 y queda contingencia; se reporta aparte como referencia de escala de datos | **Endpoint primario pre-registrado:** RR por molécula de la cabeza fp sobre `full10ppm`, promediado en semillas y pareado, en M-shift = M2s ∪ M4 ∪ M5 (n ≈ 8,500). Secundarios: fwd, M1, T1, panel de isómeros, Tanimoto de FP | (9 × 3.5 + 9) × 1.3 ≈ 53 h | JEPA − scratch ≥ +0.010 y JEPA − Recon ≥ 0 en el primario (p<0.05, Holm sobre los 2 contrastes), M1 ≥ −0.005. Potencia ≈0.8 con σ_d = 0.30; se recalcula con el σ_d de J3 antes de lanzar y, si no llega a 0.8, se amplían M2s/M4 hasta ~20k | se cumple F1b (§5.9) |
| **J8** | World model IWM sobre `pairs_xcond` con el encoder congelado ganador **de J4/J5** (en S2, desacoplado de J7): predictor AdaLN-zero frente a FiLM frente a concat, profundidad 4 | **MRR de equivariancia @256**; top-1 cross-instrument en M5 frente a entropía y sqrt-coseno; recall del overlay a FPR ≤0.5 % en P-C1T | 3 × 2 h = 6 h | equivariancia ≥0.5 **y** (M5 top-1 ≥ entropía + 0.03, o recall del overlay +5 pts a FPR fijo) | equivariancia < 0.2 → H3 cae (F2) |
| **J9** | fwd en dos etapas (estilo V-JEPA 2-AC): latentes en caché (2.54 M × 384 fp16 ≈ 2 GB), predictor (mol + cond) → latente, InfoNCE con negativos de misma fórmula de todo el pool [MSAlign]; puntuación bajo las condiciones reales de cada espectro de la consulta, agregada con log-sum-exp por IK14; variante con K=8 slots [DINO-WM] | hit@1 en el panel de isómeros no-MSG frente a fp, fwd conjunto, MetFrag-lite (`frag_rescore`) y GLACIER; oráculo neto de 2 brazos con B0-min en P-C2E | 1 + 4 × 1 h = 5 h | ≥ fp + 0.03 y ≥ MetFrag-lite + 0.03; oráculo neto ≥ +0.04, o complementario a fp (3 brazos ≥ +0.06 neto). **Paso duro:** el MRR ciego sube ≤1/3 de la ganancia | ≤ MetFrag-lite, o falla el control ciego → H4 cae (F3) |
| **J10** | GLACIER sintético: throughput ya medido (70.8 pred/s en T4); piloto 50k moléculas NP × 2 aductos × 3 CE = 300k pred (~1.2 h T4); reentrenar fp con mezcla real:sintético 4:1 a mitad de duración (1.8 h). No en paralelo con ICE (5.2 GB) ni con un entrenamiento >6 GB | Tanimoto de fp; M3, M4; M1 sin caída | ~4 h (+ ~5 h si se escala a 150k) | Tanimoto +0.03 y MRR +0.02 en M3 o M4, M1 ≥ −0.005 | sin ganancia → no se escala (F4) |
| **J11** | Aligner estilo MSAlign sobre latentes DreaMS-SSL o JEPA, con InfoNCE sobre el conjunto de candidatos y late fusion con fp/fwd/masa (condicional: solo si J5 o J4 dan latentes útiles; desde la contingencia) | socio-oráculo neto en P-C2E; panel de isómeros | 3 h | mejor socio neto que fp, o pasa G1/G2 como inserción. **Paso duro:** el MRR ciego sube ≤1/3 de la ganancia (los objetivos con candidatos lo suben 0.070–0.085 [arXiv 2610.37979]) | no mejora o falla el control ciego |
| **J12** | Gate de novedad (COSMIC): logística monótona sobre surprise, densidad kNN latente, gaps, `q_lib_max`, `n_cand` y acuerdo entre brazos; **contraste dentro de la misma consulta**: cada consulta de P-C2E, P-C2NPX y P-C3D-clean bajo `c2_sim` (verdad en DB) y `c3_del` (verdad borrada) [foro 748524]. En las celdas de clase 2, `c3_del` se aproxima por replay (borrado post-hoc de la verdad de los dumps, que ignora la regeneración interna); en P-C3D-clean hay corrida real | AUROC/ECE del contraste (solo mide la pertenencia a DB); P-C3T y P-C3X como estratos aparte | 2 h CPU | AUROC ≥0.75 y la política de inserción pasa G1–G3 | AUROC < 0.65 (F5) |
| **J13** | Padres y verificador de clase 3 en `src/casmi/gen3.py`: padres por kNN latente (JEPA/DreaMS) frente a coseno binned; bucle tipo FOAM (edits → fwd/fp → top-20 → edits), con 300–500 llamadas por molécula que pase el gate [arXiv 2602.07709] | recall@30 y MRR insertado en **P-C3D-clean y M4** por cap; P-C3T y P-C3X por separado | 4 h | recall +0.05 absoluto a presupuesto fijo en P-C3D-clean **y** M4 | ≤ +0.01 (F6) |

Total JEPA y afines: ≈ 85 h de GPU (J11 y DreaMS-SALT condicionales). La TPU **no** se usa para la ablación, para no mezclar frameworks. Queda para semillas JAX del modelo de producción wm-canon (`casmi26-train-v4` con `mol_table` canónico), fuera del camino crítico, y para el job encolado si resulta ser `w_pred=0` (V8).

### 5.5 Datos de preentrenamiento

1. **In-domain timsTOF:** enveda-180 (1,153,785 menos celdas y análogos ≥0.7) y Zenodo solo in-source fuera de P-C2Hisf. Zenodo con aductos del test **no**, porque es P-C2E.
2. **Cross-instrument:** librerías públicas de train (~1.39 M) menos celdas y análogos.
3. **Opcional, solo si J4 da señal:** extlib fuera de celdas (tras V4), MassBank-h MS2 no duplicado y 1–2 shards GeMS-C QTOF (9.3 GB cada uno; HF `roman-bushuiev/GeMS`).
4. **Sintético:** registro de §3.8 (J10).
5. **Sin TTA** sobre el test oculto mientras no se confirme a mano la resolución de 752297 (V12). Si se confirma, la TTA se trata como cualquier componente (G0–G4).

Preprocesado común: suelo del 1 %, top-64 picos, fuera picos > prec+1.5 y sqrt + L2. En entrenamiento, fuera aductos dímero y `ref_quality_tier` C. En inferencia y en consultas de celdas solo se aplica `q_quality` (sin ppm ni `skey`) [foro 741404].

### 5.6 Condicionamiento por metadata de medida

El actual «CE/100 + aducto» se sustituye por un vector de condición:
- aducto (los 10 tipos del test + otro);
- polaridad;
- CE multi-hot nominal {0,20,35,40,50,60,80} + flag de fusión (lista = adquisiciones fusionadas [foro 753419]);
- `ce_unit`, `ce_ev_est` y `ce_missing`;
- `inst_fam` (8 valores; hoy no se usa);
- Fourier(prec).

Se inyecta con AdaLN-zero, así que el modelo arranca incondicional [arXiv 2603.19312, 2512.24497]. En J8 se compara con FiLM (actual) y concat-to-token [DINO-WM]. En inferencia el test es 100 % timsTOF, y el token de instrumento permite «traducir» referencias Orbitrap.

### 5.7 World model, forward y surprise

- **fwd_v2** es el predictor de J9. Se puntúa por candidato bajo las condiciones reales de cada espectro de la consulta y se agrega por molécula con log-sum-exp por IK14.
- Va **gateado como GLACIER:** solo actúa dentro del grupo de fórmula del líder y nunca sobre listas con `q_lib_max ≥ 0.9`. ICE/GL bajaron la verdad en 18 de 400 hits exactos [nb huseyinemreaksoy].
- Su nicho prioritario son los aductos y modos que ICE/GL no cubren: modo negativo, [M+NH4]+ y pérdidas de agua [nb samanyu1808; N1A +0.008].
- **surprise(q)** = 1 − max_c cos(z_q, fwd(c, cond_q)) sobre los candidatos de la DB en la ventana de masa, normalizado por `q_mass_bin` y `n_cand`. Es feature del gate J12.

### 5.8 Cómo entran las salidas JEPA en el sistema (rails)

1. **Rank 2 ante desacuerdo (E2R2-like).** Si el top-1 de fwd_v2/fp ≠ r1 de B0 y el gate calibrado da P(correcto) > τ (τ elegido en 9 folds de cross-fit), ese candidato va a rank 2. El rank 1 nunca se toca. La **baseline de este canal es la fp existente de wm7c3**, sin reentrenar, para que el canal no espere a J3–J9.
2. **Ranks 2-25.** Término z en RRF con w ≈ 0.75 [nb busyaprime]. Estructuras de train congeladas (rr3 FREEZE [nb heliosli]). Peso del segundo scorer en 0.5 **como default**: la evidencia (0.480 frente a 0.474) está por debajo de nuestra regla de 0.008, así que, si se toca, se afina offline en las celdas.
3. **Rescate de cola.** Candidatos de ranks 26-60 de los engines y outsiders. Si superan al rank 20 de B0 por un margen y hay consenso con otro engine, entran con `safe_extend` + `assert_monotone` [nb obstacledeveloper; nursrijan_metric-fusion].
4. **Clase 1 cross-instrument.** TAU 0.80 se evalúa como componente frente a B0 (TAU 0.85) en P-C1T, P-C1NP y las celdas de clase 2. La similitud en latente IWM es un segundo criterio del tier «silver» (rank 2) del overlay [nb matterhorn3838]. El tier «gold» (rank 1) se reserva al coseno calibrado con `q_lib_max<0.9`, y solo se fija si la calibración observa **≥30 promociones correctas** en las celdas de clase 1 y una cota superior de FPR ≤0.15 % en ≥2,000 consultas de clase 2/3 (lo que hace falta para precisión ≥0.9 a f1 = 0.17 con recall ~0.07). Se reportan siempre los errores de estructura entre los positivos, no solo los falsos positivos en negativos. Si no se llega, todo va a silver.
5. **Clase 3.** Padres por kNN latente, verificador fwd y profundidad de inserción fijada por el gate J12: rank 10 por defecto, 6 si P(no en DB) es alta. Siempre detrás del rank 1 y del bloque superior.
6. **LambdaRank en ranks 2-25 (opcional, baja prioridad).** La evidencia pública de rerankers aprendidos es negativa (RR8, V8 «RR»/«BOTH»). Si se hace, se entrena solo con filas de celdas libres de fuga para todas las features (P-C2E, P-C2Hisf, P-C2NPX, P-C2X, P-C3X), con filas ponderadas por π del prior (o su peor caso) y cross-fit de 10 folds con los controles de §4.7. P-C1T y P-C3T solo sirven para el no-daño de G2.

### 5.9 Falsación (por componente y con fecha)

- **F1a (jue 11-19):** J4 no supera al init aleatorio + 0.01 en el probe de FP con ningún brazo JEPA, o todos los brazos JEPA quedan por debajo de Recon-MPM. El preentrenamiento JEPA se cierra como palanca, y J7 se reduce a scratch frente a Recon (pregunta «¿ayuda el SSL?»).
- **F1b (jue 11-26):** en el endpoint primario de J7, el IC95 de JEPA − scratch incluye 0 (con potencia ≥0.8 para +0.010), **o** JEPA ≤ Recon-MPM. El init JEPA sale de las submissions. Queda como resultado negativo documentado: bug + ablación limpia + comparación con reconstrucción. DreaMS-SALT no falsa el objetivo JEPA; se reporta como referencia de escala de datos, en M1.
- **F2 (mar 11-24):** equivariancia IWM < 0.2 (J8 corre en S2 sobre el encoder de J4/J5). H3 se cierra y el overlay de clase 1 sigue con coseno.
- **F3 (jue 11-26):** fwd_v2 ≤ MetFrag-lite en el panel de isómeros no-MSG, sin oráculo neto sobre B0, o falla el control ciego. El world model no entra.
- **F4 (mar 11-24):** el piloto GLACIER no mejora la fp. No se escala.
- **F5/F6 (dom 11-29):** AUROC del gate < 0.65 en el contraste dentro de consulta, o recall de padres ≤ +0.01 en P-C3D-clean y M4. La clase 3 se queda con la inserción fija actual (top-5 en rank 10).
- **F7, global (dom 11-29):** ningún componente JEPA o world model pasa G0–G3 en al menos una celda objetivo. La línea sale de las submissions y el cómputo restante va al overlay, a la inserción en rank 2 y a la robustez del notebook.

Cada falsador cierra **su** componente. No hace falta que fallen todos a la vez.

---

## 6. Calendario 2026-11-09 → 2026-12-14

### 6.1 Semanas e hitos de decisión (D0–D7)

| Semana | Trabajo | Hito |
|---|---|---|
| **S1 (11-09 → 11-15)**, recortada al camino crítico | V1, V2, V3, V6, V8, V9, V10, V11, V12; kernel de entorno; harness podman/uv; B0 (TAU 0.85) y B0-min en Kaggle con commits smoke; commit de ruta completa de B0-min sobre el chunk A de P-C2E; Saner-cellsafe en local; dataset v2 mínimo (`spectra`, `molecules` con [q]/[cand]/[eval], novedad, Tanimoto/ANALOGS_07); **registro de celdas congelado el jueves** (todas las celdas P/T como listas de `skey`, más M1, M2s, M4, M5) antes de cualquier entrenamiento; dumps de P-C2E (chunks A y B) y P-C1T; J0 mínimo, J1, J2 con tope, J3 después del congelado; benchmark 3060; estudio de identificabilidad; probes P-B0, P-B0-min, P_fpdis y P_pc (este último si el estudio lo justifica) | **D0 (jue 11-12):** B0 ≥ 0.470 en LB, B0-min puntuado y smoke-12 local ≈ Kaggle (≥95 % del top-25). Si B0 falla, todo se para hasta arreglarlo. **D0b (vie 11-13):** chunk A local ≈ Kaggle (≥95 % y la misma cobertura por etapa); si falla, Tier P pasa a Kaggle solo con las celdas núcleo. **D1 (dom 11-15):** veredicto de J1, σ_d medido (capas y semillas), potencia real, resultado del estudio de identificabilidad, coste real por molécula |
| **S2 (11-16 → 11-22)** | Dumps de P-C2NPX (si el conteo lo permite), P-C1NP, P-C3D-clean (re-dump con B0-cellsafe), P-C2Hisf, P-C3T, P-C3X y P-C2X; NPClassifier y MCES (solo clase 3 y M4) en `.venv-meta`; `pairs_xcond`; J4 (5 brazos), J5, J8 sobre el ganador de J4/J5; J10 piloto; overlay de clase 1 (TAU/K, gold/silver con el criterio de ≥30 promociones); inserción de fp existente en rank 2; calibración v1 (p_k) + `pick_sim` v1; B0-min' (fe_D reentrenado) si V9 lo pide; P_strat y P_dup | **D2 (mié 11-18):** la inserción en rank 2 y el overlay pasan G0–G2 con potencia declarada y se lanzan sus A/B (G4) → B1 = B0-min + componentes. **F1a (jue 11-19).** **D3 (dom 11-22):** decisiones de licencia de extlib (V4) y de cumplimiento de B0 (V9); si no se resuelven, overlay solo con MassBank-h + CC0 + open MS/MS y pick A sobre B0-min |
| **S3 (11-23 → 11-29)** | J7 (~53 h), J9, J12, J13 (J11 si procede); clase 3 2.0 en P-C3D-clean y M4; **primer dry run completo** en Kaggle sobre una celda de 400 moléculas / ~1,500 espectros (runtime de G0); LambdaRank solo si sobra tiempo; confirmaciones en LB (≤2/día) | **D4 (mar 11-24):** F2, F4. **D5 (jue 11-26):** F1b, F3. **D6 (dom 11-29):** F5–F7; lista cerrada de componentes candidatos |
| **S4 (11-30 → 12-06)** | Modelo de producción wm-canon (todo menos celdas; 1 semilla en la 3060, el resto en TPU); ensamblado de ≤4 configuraciones; dry runs 2 y 3 (≤9 h, guard de 8.75 h, casos límite: CE con 2–5 valores y 0 eV, aductos raros, 1159 Da); paquete de reproducibilidad (código, data cards, `docs/provenance.md` sin NIST, licencias); evaluación del merge solo con información pública | **D7 (vie 12-04):** merge sí/no. Criterio: el brazo del candidato es un notebook **público** que corremos nosotros en nuestras celdas y da ganancia de oráculo neta ≥ +0.02, es reproducible y su LB público es coherente. Antes del merge no se intercambia nada en privado. Candidatos a buscar entre autores de notebooks públicos de la familia C o de motores distintos; 750572 (−3 votos, cuerpo vacío) y 745952 (LB 0.413) son débiles, y 751356 no es una oferta de merge. Se comprueba el número combinado de submissions frente al límite de merge de Kaggle antes del 12-04. Se ejecuta antes del 12-07 |
| **S5 (12-07 → 12-14)** | Congelar features el 12-07; solo bugs y robustez; cada configuración final 2× en Kaggle (ruido de rerun); posterior final; `pick_sim` final con el **lockbox**, abierto una sola vez | Picks provisionales el jue 12-10; **selección explícita el sáb 12-12**; 12-13 de margen sin código nuevo; 12-14 cierre |

### 6.2 Presupuesto de cómputo (único)

| Recurso | Disponible | Asignación |
|---|---|---|
| RTX 3060 local (tope **165 h**; la corrida Tier P cuenta como GPU ocupada, porque ICE usa 5.2 GB y no cabe junto a un entrenamiento grande; costes en T4 a re-escalar con el benchmark) | ~35 días | **S1 ≈ 32 h**: benchmark 0.5, Saner-cellsafe 3, J1 0.5, J2 3, J3 3.6, smoke-12 0.5, réplica local del chunk A 6.1, chunk B de P-C2E 5.8, P-C1T (B0-cellsafe) 6.3, transferencia B0-min frente a cellsafe en 200 moléculas 3.1 · **S2 ≈ 53 h**: Tier P (P-C2NPX 6.1, P-C1NP 4.6, P-C3D-clean 5.4, P-C2Hisf 4.6, P-C3T 6.1, P-C3X 3.8, P-C2X 3.8) 34.4, J4 6.5, J5 2, J8 6, J10 4 · **S3 ≈ 62 h**: J7 53, J9 5, J13 4 · **S4 ≈ 11.5 h**: producción 3.5, dry runs locales 4, re-dumps 4 · **S5 1 h** · **contingencia ≈ 6 h** (J11 3 si procede). Si el coste real de Tier P pasa de 1.1 min/molécula, se recortan P-C2X, P-C3X y P-C2Hisf |
| CPU local (12 hilos, 61 GB) | continuo | dataset meta, MCES limitado (~91k pares), NPClassifier (2–4 h), calibración, estudio de identificabilidad, `pick_sim`, replays, `gate-sim` |
| Kaggle GPU (comprobar el lunes el día de reset de la cuota) | ~9.5 h esta semana + 30 h/semana (~120 h en S2–S5 si el reset es semanal) | **S1 ≈ 6.5 h**: kernel de entorno 0.2; commits smoke de B0 y B0-min (2 × ~0.45, ~26 min cada uno); commits de probe P_fpdis y P_pc (2 × 0.45); commit de ruta completa de B0-min sobre el chunk A de P-C2E (400 moléculas, ~4.5 h). Sin commit completo de 5 h. Si Saner-cellsafe no corre en local, +2.3 h (8.8 h). **S2–S5 ≈ 61 h**: ~55 commits de probe/A-B × 0.45 ≈ 25 h; 3 dry runs × ~5 h = 15 h; fallback Tier P núcleo si D0b falla (chunk B de P-C2E, P-C1T, P-C2NPX, P-C3D-clean, P-C1NP ≈ 21 h) |
| Kaggle TPU v5e-8 | 20 h/semana | Job encolado: según V8. Semillas JAX de wm-canon (producción) desde S3; fuera del camino crítico |
| Submissions | 5/día × 36 días = 180 | ≤12 de calibración y base (tras el estudio de identificabilidad), ~40 A/B de G4 (todo componente adoptado), ~12 finales (2× por configuración), ≥1/día de reserva hasta S4 (~70–80 en total) |

### 6.3 Reglas de uso del LB

Una variable por submission. Los probes de calibración se diseñan como filas distintas del modelo en p_k y solo se envían si el estudio de identificabilidad los justifica. No se persiguen deltas públicos < 0.008. El LB solo veta (G4), pero todo componente adoptado pasa por un A/B. Cada configuración final se envía dos veces (ruido ±0.002). Todo probe se construye sobre B0 o B0-min, nunca sobre base480 ni sobre el linaje del scorer retirado.

### 6.4 Estrategia de las dos picks

- **Pick A (robusta y reproducible).** B0-min, salvo que V4 y V9 se resuelvan a favor de B0, más los componentes rank-safe que pasan G0–G4 con P(Δ_private>0) ≥ 0.70 bajo el modelo de inversión: inserción en rank 2 del brazo fp/fwd, rescate de cola, overlay de clase 1 solo en silver salvo que el gold cumpla §5.8.4, y clase 3 conservadora con gate. Además, ΔLB predicho ≥ −0.002 en toda la familia de splits desplazados. Todo con código propio o público con generación documentada. Nada de datasets privados de terceros, kernels borrados, assets con NIST, extlib sin licencia ni fe_D sin código de entrenamiento.
- **Pick B (hedge de clase 3 alta / diversidad frente al bloque).** Sobre B0 o B0-min. Se distingue de A en la dimensión más incierta, y se elige maximizando la utilidad en los escenarios de δ alto del split desplazado, no en un percentil del posterior que no ve ese desplazamiento. Normalmente: inserción de clase 3 más profunda gobernada por el gate de novedad, padres JEPA y verificador fwd, más nuestro brazo en ranks 2-25. Restricción: E[Δ_B] ≥ E[Δ_A] − 0.005 en el split aleatorio. También reproducible, porque la respuesta del host sobre «una pick basta» es tentativa.
- **Selección.** Se elige el par (A, B) que maximiza E[max(U_A, U_B)] con `scripts/pick_sim.py` sobre el **lockbox**, abierto una sola vez. Ninguna pick se elige por una diferencia pública menor de 2 moléculas (0.016). La selección es explícita: no se deja la autoselección por LB público.

---

## 7. Riesgos y mitigaciones

| Riesgo | Señal | Mitigación |
|---|---|---|
| Base no reproducible o retirada del LB (scorer retirado, fe_D sin código, relleno hengck23, extlib sin licencia) | D0 falla; foro 752541, 753502, 748065, 748882 | B0 = alex0484 con TAU 0.85 + B0-min (y B0-min' con fe_D reentrenado con código público); `docs/provenance.md` desde S1; pick A sobre B0-min salvo resolución de V4/V9 |
| El harness local no reproduce B0 (imagen privada 403, Python 3.13 frente a cp312, layout del mirror, `find()` apuntando a la copia corrupta) | D0/D0b fallan; cobertura por etapa distinta | Kernel de entorno (V10), imagen pública + freeze o uv 3.12, bind mounts, sha256 de todos los inputs; si D0b falla, Tier P solo en Kaggle con celdas núcleo |
| Linaje NIST (atlas ICEBERG [foro 745134], fine-tune del `embedding_model` de DreaMS [748799], bit depositor de PubChem) | dudas en el foro | En submission solo `ssl_model.ckpt` de DreaMS y checkpoints msg_all; prohibido el atlas NIST; el bit spectral-library de `pc_src_mask` nunca es feature (los demás `src_*` sí, como [cand]); auditar si v4g usa el atlas |
| Licencia de extlib no verificable | D3 sin respuesta | Overlay solo con MassBank-h, CC0 y open MS/MS; extlib solo para validación offline; pick A sin extlib |
| extlib no representa NP ocultos | métricas dominadas por `steroid_conj` | Estratificar por quimiotipo, cap ≤30 %, P-C2X/P-C3X solo como estratos; anclas en P-C2E, P-C2NPX, P-C1NP y P-C3D-clean |
| Celdas sacadas de train infladas por memorización (Saner, spec2mol, `fpnet_full1`, LightGBM in-sample) | B0-min frente a cellsafe difiere; VALMRR 0.587 frente a 0.520 | B0-cellsafe en esas celdas; si los rankers no se pueden reentrenar, esas celdas quedan como diagnóstico |
| P-C3T química no-NP, P-C3D con verdades en librerías del overlay | AUROC/recall que no transfieren | P-C3T como estrato; P-C3D-clean (sin las 133); J12 con contraste dentro de consulta; J13 en P-C3D-clean y M4 |
| La calibración no identifica π | estudio de identificabilidad; error LOO | Parametrización en p_k; w_kc fijos; probes solo si estrechan; si p3 sigue con la anchura del prior, G3 en el peor caso del prior |
| Inversión de signo offline → LB | A/B negativos | ρ por tipo de cambio en `pick_sim`; A/B obligatorio (G4); pick A solo rank-safe |
| Gates sin potencia | potencia declarada <0.5 | Cross-fit sobre celdas completas, agrupación por clase, celdas M ampliadas, no-inferioridad agregada; componentes no verificables solo en pick B |
| Split público/privado no aleatorio | — (no observable) | Familia de splits desplazados en `pick_sim`; pick A robusta a δ ≤ 0.2; pick B elegida para δ alto |
| La pipeline no corre en la 3060 (CUDA/driver, ruedas RDKit en subproceso) o los guards saltan distinto | OOM, ICE/GL con 0 moléculas, etapas `skipped_due_to_runtime_budget` | Health logs y cobertura por etapa [nb amanatar_v18]; chunks ≤400 con `--cpus 4`; fallback Kaggle con celdas núcleo |
| Sesgo del electrón, centros de masa y regla de masa de B0 en celdas no-timsTOF | Δ al aplicar V2; B0 frente a `V8='BASE'` | Corrección en todas las celdas Enveda positivas; registros con ≥4 decimales y desplazamiento `src_ppm_offset`; ventanas ≥ ±5 ppm; centro medido en runtime [nb youyouwou_v152/v153] |
| Atajos en nuestras cabezas | J1; controles ciegos | Canonicalización única, negativos de todo el pool, ties pesimistas, control ciego en G0 (duro en J9/J11) |
| JEPA no aporta, o aporta lo mismo que la reconstrucción | F1a/F1b | Falsadores por componente con brazo Recon; el cómputo pasa a J9/J12/J13, al overlay y a la inserción en rank 2; resultado negativo publicado en la discusión |
| Sobreajuste al público y shake-up de forks | ganancias públicas < 0.01 | Gates offline multi-celda, BH entre componentes, lockbox, LB solo veta, picks por `pick_sim` |
| Fallos en el rerun oculto (timeouts, CE con 0 eV y 5 valores, reutilización de IDs en disputa) | «Notebook Threw Exception» | Guard por hash de contenido, Docker pin [foro 749908], tests con entradas tipo np-examples, guard de 8.75 h, capas añadidas ≤45 min con salto por tiempo, fallback atómico a B0-min, primer dry run completo en S3 |
| GPU local compartida o más lenta que lo previsto | benchmark del lunes; throughput < 50 % | Colas nocturnas (`tools/sweep.py`); prioridad de los dumps núcleo sobre JEPA; recortes de §6.2 |
| Ruido de semilla (0.007–0.02) | IC que cruza 0 | 3 semillas en J7; endpoint agrupado M-shift (n ≈ 8,500) con σ_d medido |
| Corrupción de datos (`pc_mass.npy`, 2 parquets SHA_DIFF) | sha no coincide | Assert de sha en cada loader y en el harness; copia v4g o re-descarga por fichero |
| Locks por `molecule_id` heredados | grep | El builder falla si aparece `_champion_top1` |
| Compartir en privado por el merge | — | Antes del merge solo información pública; el brazo del socio se evalúa corriendo su notebook público |
| Demasiados frentes | hitos D incumplidos | S1 recortada al camino crítico; orden fijo por EV/hora (§2.6); si D1 se retrasa, se recortan J5, J11, P-C2X, P-C3X y P-C2Hisf |

---

## 8. Primeros 7 días (checklist)

**Lunes 11-09**
- [ ] V8: leer los argumentos del job encolado en TPU (UI de Kaggle o `tools/kaggle_api.py`). Cancelar solo si reproduce wm4 sin cambios; si es `w_pred=0`, mantenerlo y anotarlo.
- [ ] V1: `tools/kaggle_api.py` → volcar el historial de submissions en `configs/lb_probes.yaml` (sub id, fecha, notebook, versión, base, LB, hipótesis). Confirmar la atribución del 0.382 y marcar qué probes pueden entrar en el ajuste.
- [ ] V12: leer a mano en el navegador 752297, 753502, 748065, 745758, 750684 y 754732 (`tools/kaggle_forum.py` no lee comentarios) → `work/forum/notes.md`.
- [ ] Comprobar el día de reset de la cuota de Kaggle GPU.
- [ ] V2: parchear `hybrid/e180_cell.py`, `hybrid/ext_pre_cell.py` y `scripts/make_class3.py` para aplicar `ELECTRON=0.000548` a las consultas positivas.
- [ ] `scripts/meta/canon.py` (`smiles_canon`, `skey`) con test de regresión contra `src/casmi/metric.ik14` sobre 10k pares train/pool.
- [ ] V3: `src/casmi/io_checks.py::assert_sha(path, prefix)`; re-descargar `pc_mass.npy` de popularity-prior a un directorio nuevo (`tools/kaggle_api.py`, por fichero) y verificar su sha.
- [ ] V10: `kaggle kernels pull -m` de las dos copias de alex0484 (alexchilton, rogersjohnson); diff de código; decidir el digest de B0 (por defecto alexchilton `9a6bcfda…`); `podman manifest inspect` del digest; lanzar en Kaggle el kernel de entorno con «Pin to original environment» (`python -V`, `pip freeze`, `nvidia-smi`, rutas de `find()`, sha256 de los inputs).
- [ ] Empezar la descarga de sergiomalv open MS/MS y de `ssl_model.ckpt` de DreaMS, cada uno en su directorio nuevo.
- [ ] Montar `.venv-meta` (uv) con myopic_mces, pulp, tensorflow/keras (NPClassifier) y dreams; probarlo con 100 moléculas.
- [ ] Benchmark de 500 pasos en la 3060: receta wm7c3 (bf16) y paso JEPA-v1; re-escalar la tabla de §5.4.
- [ ] Empezar J0 mínimo (queries con posición A/B, `tgt.eval()`, `exclude_all`) + `tests/test_jepa.py` con el control sin contexto.
- *Salida:* `configs/lb_probes.yaml`, `work/forum/notes.md`, parches del electrón, `canon.py` en verde, digest de B0 decidido, kernel de entorno lanzado, benchmark de la 3060.

**Martes 11-10**
- [ ] Con el volcado del kernel de entorno: construir la imagen podman (tag público de `kaggle-gpu-images` con Python 3.12 + `pip install -r freeze_B0.txt`) o el venv uv 3.12.13; instalar papermill; probar la GPU con CDI; preparar los bind mounts (incluido el `pc_mass.npy` re-descargado) y el chequeo de sha256 de todos los inputs contra los de Kaggle.
- [ ] V9: construir B0 (TAU 0.85) y B0-min (sin relleno hengck23, sin extlib, sin fe_D) con el builder (V6: pin, hash de contenido, `grep _champion_top1`). Commits smoke en Kaggle y submissions **P-B0** y **P-B0-min**.
- [ ] J1 (solo evaluación): wm7c3 con candidatos crudos frente a canónicos, espectro constante, donante desordenado y batch 1/64 en holdout1500 y en el honest set con electrón → `work/audit/j1.json`.
- [ ] J2 con tope: ≤16 negativos por molécula, pool ≤4–5 M filas (o `plm` pinned en host) → `emb/mol_table.parquet`, `emb/plm.npy`, `emb/fp.npy` con cobertura ≥95 % (V7).
- [ ] `scripts/meta/build_spectra_meta.py` → `work/meta/v2/spectra.parquet`.
- *Salida:* entorno del harness listo, P-B0 y P-B0-min enviados, veredicto preliminar de J1, `spectra.parquet`, J2 en marcha.

**Miércoles 11-11**
- [ ] `build_universe.py`, `label_novelty.py` (con las librerías del overlay), `chemotype.py`, `cand_density.py` → `molecules.parquet` v2 con [q]/[cand]/[eval]; `tests/test_obs_features.py`.
- [ ] V11: `scripts/meta/zenodo_e180.py --by-skey`: alias, pool test-like real y comparación de QC con e180 de train.
- [ ] `struct_dist.py` (Tanimoto contra train, e180, publib y folds 1–4) → `ANALOGS_07`, `nn_tan_train_f14`, `scaffold_novel_f0`.
- [ ] Contar las NP de open MS/MS fuera de train∪MSG∪extlib (P-C2NPX) y las candidatas a P-C1NP y M4.
- [ ] `build_cells.py` de todas las celdas como listas de `skey`: P-C2E (`tune_hist` / `work` / `lockbox`), P-C1T (414 del fold 0), P-C1NP, P-C2Hisf, P-C2NPX, P-C2X, P-C3T, P-C3X, P-C3D-clean, T1, M1, M2s, M4, M5.
- [ ] `leak_audit.py --overlay-libs …`; el exit debe ser 0.
- [ ] Kaggle: commit de ruta completa de B0-min sobre el chunk A de P-C2E (las 400 `tune_hist`, ~4.5 h, dataset privado del equipo).
- *Salida:* registro de celdas v1 a punto de congelar, conteos reales de P-C2E/P-C2NPX/P-C1NP/M4, chunk A corriendo en Kaggle.

**Jueves 11-12 (D0)**
- [ ] `freeze_registry.py` a mediodía: sha del registro y de `ANALOGS_07`. Desde aquí, ningún entrenamiento sin ese sha en su `meta.json`.
- [ ] Saner-cellsafe en local (reentreno sin `ALL_CELLS`, ~3 h); decidir qué se hace con los rankers LightGBM v3/v4b (reentrenar sin `ALL_CELLS` o desactivar) y con spec2mol (desactivado) en B0-cellsafe.
- [ ] D0: B0 ≥ 0.470 en LB, B0-min puntuado; smoke-12 local frente a Kaggle (≥95 % del top-25). Si B0 falla, todo se dedica a corregirlo. Si B0-min < B0 − 0.016, planificar en S2 dos submissions para localizar el coste (sin relleno / sin extlib).
- [ ] Noche: réplica local del chunk A de P-C2E (~6.1 h).
- *Salida:* registro congelado, informe D0, Saner-cellsafe.

**Viernes 11-13 (D0b)**
- [ ] D0b: chunk A local frente a Kaggle (≥95 % del top-25, misma cobertura por etapa). Si falla, Tier P pasa a Kaggle con celdas núcleo y se re-planifica §6.2.
- [ ] `tools/cells.py replay|report` sobre el chunk A: coste real por molécula y por espectro, σ_d de las capas finales, primera tabla de potencia real.
- [ ] Simulación de la inserción en rank 2 con la fp **existente** de wm7c3 en el chunk A (`tune_hist`): oráculo neto de inserción y τ por cross-fit.
- [ ] Construir y enviar el probe **P_fpdis** sobre B0-min.
- [ ] J3 semillas 0 y 1 (wm-canon, mitad de duración, ~3.6 h) después del congelado.
- [ ] Noche: chunk B de P-C2E en local (las ~375 nunca usadas, ~5.8 h).
- *Salida:* informe D0b, `report_B0min_P-C2E-A.md`, σ_d de capas, P_fpdis enviado, J3 en marcha.

**Sábado 11-14**
- [ ] `scripts/calibrate_lb.py --identifiability` con el conjunto de probes planificado; decidir qué probes se envían (P_pc, P_strat, P_gen).
- [ ] `scripts/pick_sim.py` (esqueleto: split aleatorio y desplazado, modelo de inversión, `work` y `lockbox` separados).
- [ ] Transferencia: B0-min frente a B0-cellsafe en 200 moléculas de P-C2E (~3.1 h).
- [ ] Overlay de clase 1 por replay en CPU: barrido de TAU {0.70…0.90} × K {3,4,6} sobre P-C1T (recall, errores de estructura entre positivos) y sobre P-C2E como clase 2 (falsas promociones), con 5 draws en la capa de librería. Precisión a prevalencia natural y conteo de promociones correctas frente al criterio de 30.
- [ ] Publicar en el foro la pregunta sobre la procedencia y licencia de extlib (V4).
- [ ] Noche: P-C1T con B0-cellsafe (~6.3 h).
- *Salida:* estudio de identificabilidad, esqueleto de `pick_sim`, curva TAU/K del overlay, chequeo de transferencia.

**Domingo 11-15 (D1)**
- [ ] Submission **P_pc**, solo si el estudio de identificabilidad la justifica.
- [ ] σ_d entre las semillas de J3 → potencia de J7 y tamaño de M2s/M4.
- [ ] Informe D1 en `work/reports/D1.md`:
  - veredicto de J1 (atajo sí/no, con umbrales);
  - σ_d (capas y semillas), MDE y potencia por componente declarado;
  - resultado del estudio de identificabilidad y probes elegidos;
  - estado de B0/B0-min (LB, coste de cumplimiento) y de V4/V9;
  - coste real de Tier P por molécula y cola de S2 re-planificada;
  - lista de celdas pendientes de dumps para S2.
- [ ] Planificar la cola nocturna de S2 (`tools/sweep.py`): P-C2NPX, P-C1NP, P-C3D-clean, luego J4.
- *Salida esperada al cierre de S1:* entorno y B0/B0-min reproducibles, registro de celdas congelado con auditoría de fuga, dumps de P-C2E (A y B) y P-C1T, σ_d y potencia medidos, J0 mínimo, J1 y J2 cerrados, J3 con dos semillas, 3–4 probes nuevos en `configs/lb_probes.yaml`.

---

## Notas de revisión

**Aceptado tal cual o con cambios menores.** Se corrigieron todas las citas señaladas (MS-JEPA, Polymer-JEPA, GLACIER/ICEBERG, auditoría in-the-Wild, 1.58 % frente a 4.4 %, atribuciones de RR8/NX/rescue=3, fila del overlay extlib, v586, peso del segundo scorer, reproducibilidad tentativa, IDs del rerun en disputa, resolución citada de 752297) y las fechas (D2 mié 11-18, D7 vie 12-04, F1a jue 11-19). SpecBridge sale de la evidencia. Entran el margen de clase 3 (+0.001…+0.004), la regla de semillas con excepción para smoke tests, el S1 recortado, el congelado del registro antes de entrenar, el kernel de entorno, podman, el bind mount del `pc_mass.npy`, el sha de todos los inputs, la lectura manual de hilos, el runtime en G0, el primer dry run en S3, el MCES limitado, el throughput medido de GLACIER, J2 con tope de memoria, P-C3D re-dumpeado con otra base y sin las 133, P-C3T como estrato, P-C1NP y el prior de κ1, la estratificación por documentación de P-C2NPX, la regla de masa en celdas no-timsTOF, el contraste dentro de consulta para J12, la separación [q]/[cand]/[eval], `q_quality`, el oráculo neto, el criterio de gold del overlay, `n_spec` como sensibilidad, la reconstrucción de P-C2E por `skey`, el lockbox, el split desplazado, el brazo Recon-MPM, la Variante B como primaria, el control ciego duro en J9/J11, el máximo de 4 estratos de decisión, el merge solo con información pública, B0 con TAU 0.85 y B0-min.

**Aceptado con modificaciones (y por qué):**
1. *Calibración (crítico).* Se adopta p_k = π_k·κ_k. Además de declarar π no identificado, se fijan los w_kc por diseño para que solo queden 3 parámetros ajustables, porque con w_kc libres seguiría habiendo más grados de libertad que probes informativos.
2. *Contracción e inversión (crítico).* Se modela la mezcla con P(inversión) explícita, como se pidió, pero con ρ dependiente del tipo de cambio: media 0.5 para cambios que tocan el rank 1 y 0.25 para cambios rank-safe. Un revisor propuso 0.3–0.5 para todo. Bajamos los rank-safe a 0.25 (≥0.2, el mínimo de otro revisor) porque las inversiones históricas vinieron de validaciones con fuga o de no medir la clase 1, y ambas causas las atacan ahora las celdas de clase 1 y G2. Con ese modelo, el umbral de G3 pasa de 0.80 a 0.70 para la pick A. No es una relajación: 0.80 se calculaba con un s>0 que sobrestimaba P de forma sistemática, y con la inversión explícita 0.80 sería inalcanzable para cualquier componente con ρ ≥ 0.2. La estrictez se mantiene por otra vía: la pick A solo admite cambios rank-safe y exige A/B en el LB.
3. *B0-cellsafe.* Se restringen P-C1T y P-C3T al fold 0 con `fpnet_fold0`, como se pidió. P-C3D-clean no se restringe al fold 0, porque quedaría en ~70 moléculas: se re-dumpea en sub-chunks por fold, cada uno con su `fpnet_fold{k}` OOF. spec2mol se desactiva en cellsafe porque, si memorizó, «generaría» la verdad de clase 3.
4. *M2/M4 sin reentrenar.* En vez de entrenamientos separados con el scaffold fold 0 fuera, M2s y M4 se definen como subconjuntos naturales del fold 0 de identidad (Murcko ausente de los folds 1–4; sin análogos ≥0.7 en el entrenamiento). Mide lo mismo sin duplicar J7, y da n de sobra (~13.6k disponibles para M2s).
5. *P-C1T y M5.* Comparten las 414 estructuras e180∩public del fold 0. Ambas son solo de evaluación, así que compartirlas no introduce fuga, y deja las 1,874 restantes para `xcond-train`.
6. *Draws.* Tier P con 1 draw, como se pidió. Se descarta la variante opcional de 2 draws en 200 moléculas por presupuesto; la varianza de draw se mide en las capas reproducibles en CPU.
7. *GLACIER.* Dos revisores daban cifras contrarias (7.6 h a 11 pred/s frente a 1.2 h a 70.8 pred/s). Se usa el throughput medido en nuestro log (70.8 pred/s en T4) y se mantienen los dos aductos.
8. *J6.* Se elimina en lugar de rehacer sus umbrales por IC: con su n, ningún umbral útil quedaba fuera de un SE, y sus 5 h rinden más en el brazo Recon y en semillas de J7.
9. *P_gen.* No se elimina del todo, aunque un revisor lo considera poco informativo, porque mide directamente p3·μ3(gen), que es lo que necesita una decisión de inserción de clase 3. Queda en una sola variante y condicionado al estudio de identificabilidad.

**Rechazado.** Ninguna objeción se rechaza en su totalidad.

---

## Anexo A — Resultados del 2026-11-09 (ejecución local, RTX 3060)

**J1 (listas del pipeline, celda c3, 401 moléculas con la verdad en el top-25).** A ciegas, con los espectros de otra molécula, el modelo saca ≈ azar (nce 0.139 frente a 0.154). Canonicalizar todos los candidatos solo cuesta ≈0.025. En este escenario la verdad ya viene en formato canónico en 376/401 casos. `scripts/audit_j1.py`.

**J1b (escenario de la evaluación interna: verdad en formato train frente a ~200 candidatos PubChem de la misma masa, 418 moléculas, azar 0.030).**

| MRR | fp | fwd | jepa | nce | sae |
|---|---|---|---|---|---|
| crudo | 0.395 | 0.749 | 0.678 | 0.753 | 0.709 |
| todo canónico | 0.395 | 0.667 | 0.452 | 0.658 | 0.481 |
| a ciegas, crudo | 0.063 | 0.073 | 0.162 | 0.083 | 0.163 |

Atajo de formato confirmado: las cabezas jepa y sae son en gran parte atajo, y nce/fwd tienen ~0.09 inflado. A esto se suma una segunda inflación: los negativos duros de entrenamiento eran exactamente los distractores de evaluación (solo había filas DB alrededor de las masas del holdout). `scripts/audit_j1b.py`.

**J0.** Bug de JEPA v1 confirmado: con el modelo en eval, la dispersión de las predicciones enmascaradas dentro de un espectro es 0.0. v2 (queries con m/z, contexto ciego a los picos ocultos, target en eval, control sin contexto) da 0.32. `tests/test_jepa.py`.

**J2.** `scripts/build_mol_table.py` genera `work/emb_v3`:
- 267,831 moléculas de train, 2.04 M negativos (58 % de misma fórmula) para todas ellas y 201,295 distractores de evaluación disjuntos por hash.
- Todo canónico sin estereo.
- 2,234 estructuras de validación excluidas.

**Meta v1.** `scripts/meta/build_molecules.py` genera `work/meta/v1/molecules.parquet` (267,831 × 71). Clases de novedad sobre las estructuras de train:

| Clase | Estructuras |
|---|---|
| 1 | 90,411 |
| 2 | 172,692 |
| 3 | 4,728 |

**Hallazgo de dominio.** Enveda-180, es decir, todo nuestro timsTOF de train y la base de todas las celdas de validación usadas hasta hoy, es química de screening tipo fármaco: 0 % en COCONUT, NP-score −1.5, 36 % halogenados, 2.5 % con PubMed y masas entre 235 y 499 Da. `enveda-np-examples` es NP: 99.6 % en COCONUT, NP-score +1.42, 96 % con PubMed, masas de 133 a 1109 Da con mediana 354. Esto encaja con la descripción del test oculto (157–1159 Da, mediana 348, extractos biológicos).

Para celdas parecidas al test hay 23,415 estructuras NP (COCONUT, 157–1159 Da, una sola especie neutra) con espectros públicos en train, 15,276 de ellas con PubMed. Por librería: gnps 15.3k, riken 11.1k, mona 8.5k, pluskal 7.7k, msdial 6.5k, massbank 4.9k. De ellas, 295 son timsTOF (244 de np-examples).

Consecuencia: las celdas de clase 2 y 3 del plan deben construirse sobre estas estructuras NP, ocultando todos sus espectros de la librería (c2_sim) y, en clase 3, también de las DB (c3_del), con estratificación por masa como el test. Las celdas basadas en Enveda-180 pasan a ser un estrato de control fuera de dominio.

## Anexo B — Resultados del 2026-11-09/10: celdas NP, comparación con el método público y B0-repro

**Celdas de validación NP** (`scripts/meta/build_cells.py`; candidatos = toda la ventana PubChem a 10 ppm, deduplicada por InChIKey-14, SMILES canónicos sin estéreo):

| celda | moléculas | consultas | candidatos/mol (mediana) | misma fórmula | doc mediana | qué simula |
|---|---|---|---|---|---|---|
| `np_tims` | 245 | 1,168 timsTOF | 5,266 | 902 | 13.0 | np-examples (excluidas de todo entrenamiento) |
| `np_c2` | 801 | 3,606 | 6,063 | 980 | 5.5 | NP de COCONUT/LOTUS/NPAtlas, masa igualada a np-examples, todos sus espectros ocultos |
| `np_like` (`work/cells_x`) | 801 | 3,430 | 5,409 | 601 | 5.2 | NP por NP-score > 1 sin exigir pertenencia a DB (82 % la tiene); solo para priors |

Registro de exclusión: `work/cells/exclude_all_ik.txt` (1,585 estructuras + celda c3). Tabla de entrenamiento `work/emb_v4` sin ellas.

**MRR@25 (ventana completa, honesto salvo donde se indica):**

| | np_tims | np_c2 |
|---|---|---|
| prior doc = ln(1+sustancias)+ln(1+PubMed) | 0.788 | 0.330 |
| modelo w4base, solo (nce / fwd / (nce+fwd)/2) | 0.347 / 0.318 / 0.357 | 0.354 / 0.359 / 0.375 |
| z(fwd) + 0.25·doc | 0.887 | 0.577 |
| J3 (vio los espectros de np_c2) | 0.887 | 0.750 (fuga: memorización +0.23 en modelo solo) |
| ensemble 3 checkpoints | 0.892 | — |

Semillas: holdout 0.752/0.751, np_tims 0.887/0.886. Sensibilidad a ±0.000548 Da en las consultas: ruido (±0.01 modelo solo, 0.887–0.890 fusionado).

**Comparación con el método público** (`scripts/meta/panel_h2h.py`; panel de 250 NP timsTOF, mismos candidatos y métrica que `casmi26-natural-product-panel-benchmark`): nuestro modelo solo 0.453 frente a FPNet fold-safe 0.298 (+0.136, IC95 [+0.088, +0.187]); el FPNet A+B público, que vio esas moléculas, 0.522. Con prior: 0.991 frente a 0.982 (saturado).

**Priors:** bonus por bit COCONUT/NP sobre doc: np_tims 0.788→0.808, np_like 0.246→0.301 (no circular; las moléculas sin bit bajan 0.088→0.053, compensa si >~1/3 del test tiene bit), np_c2 0.330→0.436 (circular). Fusión logit condicional aprendida (`scripts/meta/fusion_cells.py`): np_c2 0.611 aprendida en np_tims (0.666 con CV interna, circular); np_tims 0.875 frente a 0.887 de la regla manual. Duplicados IK14 en las listas del pipeline: 0 (ya deduplica).

**Calibración aproximada con las sondas LB** (rangos, no medida): solo librería 0.152 → clase 1 ≈ 17–19 %; con MRR plausibles de 0.5–0.8 en clase 2 y 0.05–0.1 en clase 3, la clase 3 sería ≈ 25–46 % del test. +0.05 en clase 2 ≈ +0.02 LB.

**Historia del notebook 0.480** (sus propias tablas): m1 (solo neural público) 0.477 → capa final midx 0.479 → segundo scorer (bgunshui, kernel ya borrado) 0.480. Sin el kernel borrado el notebook cae al neural público solo (~0.479). Los cambios de modelo neural mejores offline perdieron en el LB (−0.019 a −0.058); el puesto 1 es lo frágil.

**B0-repro** (`hybrid/build_b0repro.py`, `hybrid/ours_neural_scores.py`, `hybrid/build_ours_bundle.py`): nuestro modelo en el hueco del segundo scorer (0.5·z(público)+0.5·z(nuestro), puestos 2–25; puesto 1 leader-safe sin cambios) y en los candidatos externos; si falla, el notebook vuelve al público solo. Bundle de 85 MB (código, checkpoint, ChemBERTa original del hub, media PLM); 12 s en GPU / 58 s en CPU para 10.7k candidatos. Repetición offline del consenso sobre la ejecución real `eval-c3` (430 moléculas Enveda-180, fuera de dominio): sin neural 0.629, neural público 0.615, público + nuestro 0.630 (+0.014 frente al público, IC95 [+0.007, +0.023], 75 suben / 44 bajan; `hybrid/replay_consensus.py`). Pendiente: prueba de humo local del notebook completo y, con permiso, dataset privado + commit (~26 min GPU) + 1 envío (~3.5–4 h GPU).

---

## Anexo C — Errata del 2026-11-12: qué es eval-c3 y qué implica para P001

El Anexo B («Repetición offline del consenso sobre la ejecución real `eval-c3` (430 moléculas Enveda-180, fuera de dominio)») y el mensaje del commit 8671293 («430 drug-like Enveda-180 molecules hidden from the library») describen mal la celda. No se editan; esta nota los corrige.

**Qué es eval-c3.** Es la ejecución en Kaggle de la celda `work/c3data` (P-C3D en §4): clase 3 simulada sobre estructuras **tipo NP**, con consultas **Orbitrap/QTOF**. No es Enveda-180 ni química tipo fármaco.
- Comprobación propia sobre las 484 filas de `work/c3data/answers.parquet`:
  - NP-score mediano 1.77 (mínimo 1.00);
  - 63 % en COCONUT y 43 % en LOTUS;
  - 0 espectros timsTOF y ninguna verdad con espectros de Enveda-180 o de np-examples.
- En las 430 moléculas de la ejecución (revisión del 11-11):
  - NP-score mediano 1.74, PubMed>0 en el 45.6 % y masa mediana 431.6 Da;
  - clase 3 en 430/430;
  - todas las verdades son estructuras de train/pool, y el scorer neural público vio ~92 % de ellas;
  - 534 de sus 959 espectros no tienen CE, frente a 0 de 1,213 en el test visible.

**Consecuencias.**
- P001 dio 0.469 frente a 0.479 de la versión solo pública: −0.010, z ≈ −1.3 frente a efecto nulo. La lectura «la evaluación tipo fármaco no transfiere a NP» apunta al eje equivocado. Lo que separa eval-c3 del test es:
  - el instrumento;
  - la mezcla de clases (no hay clase 1);
  - que las verdades pertenecen a train/pool;
  - la documentación;
  - el camino de CE ausente.
- eval-c3 favorece por construcción a los modelos con lift de pertenencia. En la verificación del 11-12, con α 0.25 en el hueco, una copia con lift da +0.0039 y una copia neutral −0.0015. Por eso ni el +0.014 [+0.007, +0.023] del Anexo B ni el +0.0104 [+0.0027, +0.0185] del commit 8671293 predicen el LB.
- Desde ahora eval-c3 solo sirve para comprobar la mecánica (hashes, invariancia del rank 1 y referencia de ruido). Nunca se usa como predictor de despliegue.
- El texto de ayuda de `scripts/meta/build_cells.py --c3` tenía el mismo error y se corrigió en el mismo commit que esta nota.

El detalle, la verificación y el plan que se deriva están en `docs/model_proposal.md` (§1.2 y §3).
