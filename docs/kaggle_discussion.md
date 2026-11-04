<!--
Living public draft of our Kaggle discussion post (English, posted as-is).
Rules for this file:
  * SHARE: the evaluation protocol, the leak-free query set recipe, negative results, oracle ceilings,
    strata, anything that improves how the field measures progress.
  * KEEP PRIVATE (never in this file): model code/weights and training recipe of our world model
    (losses, negatives, conditioning), router features/thresholds that pass the gate, any configuration
    that moves the board, the withdrawn third-party scorer copy, our submission ids.
  * Every number carries its interval and its frame; no number here is a board number unless it says so.
Changelog at the bottom; bump it on every update.
-->

# Your validation is lying to you: a leak-free evaluation for CASMI26 (protocol, oracle ceiling, open to merging)

**TL;DR.** A validation set cut from `train.parquet` cannot rank the public pipelines, because their models
and priors were trained on all of `train.parquet`. We built a leak-free query set from the host-approved
Enveda-180 release, scored the public 0.480 pipeline on it with the official rule, and report every
difference with a paired bootstrap interval, per stratum, together with an oracle ceiling. Notebook:
**[link to the Code notebook]**. We are also **looking for teammates** (details at the end).

---

## 1. The problem, measured on our own account

On a validation set held out from `train.parquet`, one re-ranking change gained **+0.019 MRR@25**. On the
public board the same change scored **−0.017**, and two stronger versions of it scored −0.083 and −0.099.
Ours is not an isolated case: several threads report the same pattern (local gains that never transfer,
e.g. retrained fingerprint nets that win offline and lose on every board swap).

The reason is structural. The public pipeline's neural scorers, fingerprint nets and priors were fitted on
the whole of `train.parquet`. Any molecule you hold out from it has already been seen by them, so the
pipeline looks near-perfect on your validation (we measured ~0.78) and *anything* you add can only make it
look worse, or better for the wrong reasons. The board, meanwhile, has ~125 public molecules: with most
reciprocal ranks at 0 or 1, the standard error of a single score is about **±0.045**, and only large
differences are visible.

## 2. A query set nobody has trained on

The Enveda-180 Zenodo release (record 21346580, CC-BY-4.0; the host confirmed it may be used, thread 745148)
contains spectra of structures that are **absent from `train.parquet`**: **775 structures / 3,072 spectra**
with the adducts the competition kept for that library (our count, consistent with the ~816 reported in that
thread). They are timsTOF spectra, like the test.

The query set:
* structures absent from `train.parquet` (InChIKey first block, official canonicalisation);
* only adducts that occur in the visible test, and only molecules with at least one `[M+H]+` or `[M-H]-`
  spectrum, which gives an adduct mix close to the test (≈85% `[M+H]+`, ≈13% `[M-H]-`);
* fresh random `molecule_id` / `spectrum_id`, the library left untouched (the real `train.parquet`).

The public pipeline is then run end to end on it, unchanged except for the folder it reads.

**Caveat, stated up front.** These are drug-like Enamine compounds, not natural products, and most of them
are in PubChem. The set measures *ranking among database candidates without a library match* (the hard part
of class 2) honestly; it says nothing about natural-product class 3. Results are a within-frame comparison,
never a board estimate.

## 3. Protocol

Borrowed from how protein-function prediction is evaluated in CAFA (temporal holdouts, per-category cells,
paired intervals; see the CAFA-evaluator and our fork cafaeval-protea used in PROTEA):

1. **Official scoring per molecule**: stereo removed, tautomer canonicalised, InChIKey first block, first
   hit among the non-empty guesses, MRR@25.
2. **Paired bootstrap 95% CI** of every difference against a base arm. A mean without an interval is not a
   result.
3. **Golden rule for features**: a signal may be used only if it can be computed identically for a molecule
   nobody has seen, from data available at inference. Leakage can be *definitional* (the value depends on
   the label or the split) or *constructional* (the way rows are built encodes the split).
4. **Strata.** Observable at test time (usable for routing): precursor mass, number of spectra, peak count,
   polarity, stepped collision energy, agreement between arms. Evaluation-only: the true class.
5. **Oracle ceiling**: MRR if the best arm were chosen for every molecule. If it is close to the base, no
   routing or blending can help, and a new expert is needed.
6. **Gate.** A change counts only if it is positive in ≥ 8 of 10 disjoint molecule folds (chosen on nine,
   applied blind to the tenth) **and** its CI lower bound is above zero.

## 4. Results

*[TBD: filled from the notebook once the runs finish.]*

| arm | MRR@25 | Δ vs base | 95% CI |
|---|---|---|---|
| public 0.480 pipeline (base) | – | – | – |
| … | | | |

* Oracle ceiling over all arms: *[TBD]*
* Where the arms disagree (strata): *[TBD]*
* Gated router on observables: *[TBD: pass / fail]*

## 5. Negative results we are happy to share

* **Re-ranking the pipeline's top-25 with a separately trained model** (blend weights 0.5 / 0.75 / 1.0):
  0.463 / 0.397 / 0.381 on the board against 0.480. The heavier the weight, the larger the loss; this is
  far outside board noise.
* **Adding the 775 missing Enveda-180 structures to the spectral library**: 0.480 → 0.480 on the public
  board. They are either not in the public test or do not change its ranks.
* **Train-based validation**: +0.019 locally vs −0.017 on the board (section 1).

## 6. What we are looking for

We are a small research-minded team (protein function prediction / CAFA evaluation background, see PROTEA)
working on self-supervised latent models of the spectrum–structure relation. We are **open to merging**
with people who bring one or more of:
* mass-spectrometry or natural-product chemistry expertise;
* GPU time beyond the Kaggle quota;
* experience with structure generation for molecules outside every database (class 3).

The merger deadline is **7 December**. Reply here or message us.

---

<sub>Changelog · v0.1 (2026-11-04): first draft, results pending.</sub>
