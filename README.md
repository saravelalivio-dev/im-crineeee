# 🌸 HYGEIA — Interactive Research Dashboard

An Interpretable Deep Learning and Multi-Objective Pipeline for De Novo
Antimicrobial Peptide Design targeting *Klebsiella pneumoniae*

Saravel Bless Y. Alivio · Princess Fely T. Cuering · Lianne Brylle L. Oros
Davao City National High School — DepEd Division Science & Technology Fair 2026

---

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Files

| File | Role |
|---|---|
| `app.py` | Layout and interaction — the four tabs |
| `hygeia_data.py` | Every published table from the manuscript, in one place |
| `hygeia_features.py` | Physicochemical descriptor equations |
| `hygeia_models.py` | Gatekeeper, Shapley attribution, ACPE Pareto sort |
| `hygeia_theme.py` | Pink design system: CSS + Plotly template |
| `.streamlit/config.toml` | Pink theme for Streamlit's own widgets |
| `_smoketest.py` | Headless check — runs `app.py` without a browser |

Keeping the paper's numbers in `hygeia_data.py` rather than scattered through
the UI means the dashboard can't drift out of sync with the manuscript.

## The four tabs

1. **Generation & Gate** — ProtGPT2-style temperature sampling, the two-layer
   screening gate, live descriptor calculation, and the `p_active`
   classification card.
2. **SHAP Attribution** — global mean |φ| ranking, a local waterfall explaining
   one peptide's score, the Table 4 statistical validation, and the feature-set
   ablation.
3. **Pareto / ACPE** — 2-D and 3-D objective space with clickable points,
   threshold guides, per-candidate objective profiles, and front hypervolume.
4. **Structural Dynamics** — HPEPDOCK results per target, eigenvalue spectra,
   mode-variance partitioning, the stiffness anomaly, and deformability /
   B-factor profiles.

## What is real and what is reconstructed

The dashboard is explicit about provenance everywhere, because a research tool
that blurs measured and illustrative values is worse than no tool.

**Reproduced exactly from the manuscript.** All classifier metrics, the nine
candidate sequences and their descriptors, SHAP global importances and the
Table 4 correlations, all HPEPDOCK statistics, and every iMODS eigenvalue,
variance and stiffness figure.

**Verified against the manuscript by recomputation.** The descriptor engine
reimplements each published equation, and was checked against Table 6:

| Descriptor | Agreement |
|---|---|
| GRAVY | exact to 3 dp |
| Aliphatic index | exact to 3 dp |
| Net charge | within 0.10 |
| Isoelectric point | within 0.73 |
| Hydrophobic moment | rank order identical; see below |

The Pareto sort reproduces Table 10 exactly — seven non-dominated, Peptide_08
and Peptide_09 dominated — and the safety-first selection rule returns
Peptide_07 and Peptide_03, matching Section 6.

Two convention notes, both documented in the source:

- **Net charge** is computed with a C-terminally amidated peptide, standard in
  synthetic AMP design and the only convention that reproduces the published
  charges. **pI** keeps a free carboxyl, the universal pI convention and again
  what the manuscript's column reflects.
- **Hydrophobic moment** uses the *normalised* Eisenberg consensus scale, which
  runs ≈1.57× smaller in absolute terms than the scale behind Table 6
  (ratio 1.572 ± 0.034 across all nine candidates, so rank order is identical).
  The gate always evaluates thresholds on computed values so published and
  generated candidates are judged on one scale, and the UI quotes the
  equivalent threshold in both conventions.

**Surrogate, and labelled as such in the UI.** The production Random Forest
(1,000+ DBAASP sequences, 30 SHAP-selected features) is not bundled, so a
surrogate forest is fitted at runtime from the directional relationships the
manuscript's own TreeSHAP analysis validated. The nine published peptides
always report their manuscript `p_active`. Drop
`hygeia_rf_gatekeeper.joblib` beside `app.py` and the surrogate banner
disappears.

Sequence generation likewise stands in for ProtGPT2 (a ~3 GB transformer, past
the memory ceiling of a free hosted container) by reproducing its temperature
sampling behaviour. The function signature is unchanged, so swapping in the
real generator touches nothing downstream.

**Schematic.** The deformability / B-factor trace is reconstructed from the
manuscript's written description of each profile, so its *shape* is
discussable but its values are not measurements. The chart title and a warning
both say so. Upload the real iMODS export to replace it.

## Deployment notes

Learned from the earlier failed deploy:

- **Never put `cd-hit` in `requirements.txt`.** CD-HIT is a C/C++ command-line
  program with no PyPI distribution, so pip aborts the entire install and the
  app never starts. System packages belong in a `packages.txt` file, which
  Streamlit installs with `apt-get`. This app needs no system packages.
- `pymoo` and `modlamp` are left commented out in `requirements.txt`. Every
  descriptor and the Pareto sort are implemented natively, so neither is
  required, and `modlamp` in particular fails to build on recent Python.
- `shap` is optional. With it, attribution uses exact TreeSHAP; without it, a
  vectorised permutation Shapley estimator runs in about 0.3 s.
- Python 3.14 works but is very new. If a wheel misbehaves, pin Python 3.12 in
  the app's **Advanced settings** when deploying.
- `torch` and `fair-esm` are not dependencies here. They were the other risk in
  the original requirements file — a large ESM model will exhaust the ~1 GB
  free-tier limit.

## Limitations worth stating in a defence

- `p_active` is a screening probability, not a measured MIC.
- Toxicity and haemolysis are model predictions, not assay results.
- Docking scores rank poses; they are not binding free energies.
- NMA is a harmonic approximation around one static pose — it is a flexibility
  screen, not a substitute for molecular dynamics.
- No candidate in the published pool clears all four Research-Question-4
  thresholds at once. That conflict is the finding, and it is why the pipeline
  needs a Pareto stage rather than a weighted score.

Everything is in silico. Predicted values are computational hypotheses for
prioritising wet-lab work, not evidence of antimicrobial efficacy or safety.
