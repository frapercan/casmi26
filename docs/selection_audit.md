# E013 — Auditoría de selección del linaje público 0.480 (2026-11-12)

Diseño congelado antes de calcular (`registry/drafts/E013.json`, commit 27c56a1). Script:
`scripts/meta/selection_audit.py`. Tablas por molécula en `work/e013/` (no versionadas).

**Marco.** F_ext son 400 moléculas de Enveda-180 (Zenodo) cuyas estructuras no están en train.parquet: timsTOF,
química tipo fármaco, solo clase 2. Hay dos ejecuciones en Kaggle (`casmi26-eval-ext` y `-ext-b`):
- tienen idénticas las 26 celdas de la pipeline pública;
- difieren solo en nuestra celda final de evaluación y en una fuente de datos extra que usa esa celda;
- puntuación: MRR@25 con la clave oficial.

Limitación principal: no hay clase 1 ni clase 3. Un paso que proteja aciertos de librería puede no verse aquí sin que
eso sea ruido de selección.

## Resultados

**Q1. Variantes frente a la final (media de las dos ejecuciones).**

| Variante | MRR ej. 1 | MRR ej. 2 | Δ frente a la final [IC95] |
|---|---|---|---|
| V3 | 0.5236 | 0.5268 | +0.0020 [−0.0042, +0.0075] |
| final | 0.5196 | 0.5269 | — |
| V2 | 0.5217 | 0.5243 | −0.0002 [−0.0063, +0.0052] |
| CUSTOM | 0.5183 | 0.5234 | −0.0024 [−0.0084, +0.0024] |
| V4 | 0.5155 | 0.5238 | −0.0036 [−0.0099, +0.0017] |
| public 0.466 | 0.5142 | 0.5184 | −0.0069 [−0.0146, +0.0002] |
| base 0.465 | 0.5079 | 0.5079 | −0.0153 [−0.0250, −0.0060] |

**Regla de la pick A (fijada de antemano):** ninguna variante supera a la final con cota inferior unilateral del 95 %
> +0.005 en ambas ejecuciones. **Se mantiene la final pública.**

**Q2. La misma pipeline, dos ejecuciones.**
- Las variantes sin scorer neural (base 0.465, V0, controles) coinciden molécula a molécula.
- Las que lo llevan difieren en el 9–18 % de las moléculas: final 0.5196 frente a 0.5269 (−0.0074), V4 −0.0083,
  CUSTOM −0.0051.
- El rank 1 casi no cambia (0–0.5 % de moléculas); la variación está en los puestos 2–25.
- Lo más probable es no determinismo del cálculo neural. Como la última celda difiere, no queda demostrado.
- Encaja con lo documentado en el LB: la puntuación de la página de un notebook superó en 0.010 a su re-ejecución
  (heliosli 0.478 → 0.468).

**Q3. Pasos documentados del LB frente a F_ext.**

| Paso | Δ LB público | Δ F_ext [IC95] |
|---|---|---|
| base 0.465 → public 0.466 | +0.001 | +0.0084 [+0.0047, +0.0125] |
| public 0.466 → V2 (0.468) | +0.002 | +0.0067 [+0.0025, +0.0112] |
| V2 → CUSTOM (0.472) | +0.004 | −0.0021 [−0.0052, +0.0008] |
| CUSTOM → final (0.477) | +0.005 | +0.0024 [−0.0025, +0.0085] |

- Los dos primeros pasos se sostienen, y con más fuerza que en el LB.
- Los dos últimos suman +0.009 en el LB y ≈ +0.000 aquí (razón 0.03).
- Es el patrón de la selección: los pasos grandes y tempranos son reales, y los últimos, elegidos por décimas en
  125 moléculas, no se reproducen.
- Salvedad: lo que aportan CUSTOM y el fragment check en clase 1 no se ve en este marco.

**Q4. Simulación de selección.** 4,000 repartos al azar en 125 moléculas públicas y 275 privadas; cada variante sale de
una de sus dos ejecuciones; se elige la mejor en la parte pública.

| Conjunto de candidatas | Inflación media (público − privado de la elegida) | Pérdida frente a la mejor esperada |
|---|---|---|
| 13 variantes | +0.0050 | 0.0020 |
| V0–V4 + CUSTOM | +0.0006 | 0.0005 |
| V2, CUSTOM, final | +0.0028 | ≈ 0 |
| re-enviar la final y quedarse con la mejor de sus 2 ejecuciones | +0.0006 | — |

La V3 es la elegida el 40 % de las veces y la final el 30 %. En este marco, elegir entre muchas variantes infla el
público en unas milésimas: el mismo orden de magnitud que la distancia al bronce (+0.003).

**Q5. Resolución del LB público.**
- El error estándar de una diferencia entre variantes de este linaje, en 125 moléculas, tiene una mediana de 0.0056
  (0.0027–0.0049 en los cuatro pasos documentados; máximo 0.0103).
- Diferencias de 0.001–0.005 entre variantes cercanas caben en ±1 error estándar.

## Lectura

- El núcleo de la pipeline pública es bueno de verdad: base 0.465 → final da +0.015 [+0.006, +0.025] en un marco que
  nadie ha usado para elegir.
- Los últimos retoques elegidos en el LB no se reproducen aquí (≈ 0 de +0.009).
- La misma pipeline varía entre ejecuciones lo suficiente como para mover el público más que la distancia al bronce.
- Por tanto, las diferencias de unas milésimas en el LB entre forks (0.480 frente a 0.483–0.487) no son evidencia de
  que esos equipos estén por delante en el privado.
- Consecuencias para el plan:
  - la pick A se queda en la final pública (regla de E013);
  - no se usa el LB para elegir entre variantes que difieren en menos de ~2 moléculas;
  - una pick B poco correlacionada con el bloque de forks (R1-Gate, o la pipeline de un socio de merge) gana valor;
  - para comparar con re-ejecuciones, el control debe ser la misma variante re-ejecutada, no su puntuación de página.
