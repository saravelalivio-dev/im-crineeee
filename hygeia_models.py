"""
hygeia_models.py
================
Machine-learning, generation and optimisation layer.

Three things live here:

1. GATEKEEPER   A Random Forest classifier returning p_active.
2. ATTRIBUTION  TreeSHAP local attributions phi_j explaining one prediction.
3. ACPE         Pareto non-dominated sorting over the four competing objectives.

On the trained model
--------------------
The manuscript's production gatekeeper was fitted on 1,000+ MIC-annotated
DBAASP sequences over a 172-dimension fused matrix reduced to 30 SHAP-selected
features. That artefact is not redistributed with this dashboard, so:

  * if a pickled model is present at MODEL_PATH, it is loaded and used;
  * otherwise a *surrogate* Random Forest is fitted at runtime on a synthetic
    corpus whose labelling rule encodes the directional relationships the
    manuscript's own TreeSHAP analysis established (Table 4) - charge acting
    linearly, mu_H acting as a threshold.

The surrogate reproduces the published *mechanism* and is honest about not
being the published *model*. Every surface that shows a surrogate number says
so. Drop the real .joblib next to app.py and the banner disappears.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier

import hygeia_features as hf
from hygeia_data import CANDIDATES

MODEL_PATH = os.environ.get("HYGEIA_MODEL_PATH", "hygeia_rf_gatekeeper.joblib")
RANDOM_STATE = 42

#: Published p_active values, keyed by sequence. Any sequence from the paper is
#: reported at its manuscript value rather than re-predicted, so the dashboard
#: never contradicts Table 5.
PUBLISHED_P_ACTIVE: Dict[str, float] = {c["sequence"]: c["p_active"] for c in CANDIDATES}
PUBLISHED_ID: Dict[str, str] = {c["sequence"]: c["id"] for c in CANDIDATES}

#: Published descriptor block per sequence, so Table 6's values are what the UI
#: shows for Table 6's peptides.
PUBLISHED_DESCRIPTORS: Dict[str, Dict[str, float]] = {
    c["sequence"]: {
        "length": float(len(c["sequence"])),
        "charge": c["charge"], "mu_h": c["mu_h"], "gravy": c["gravy"],
        "ai": c["ai"], "pI": c["pI"],
    }
    for c in CANDIDATES
}


def descriptors_for(seq: str, ph: float = 7.0) -> Tuple[Dict[str, float], str]:
    """
    Descriptor block for one sequence, plus the provenance of those numbers.

    Returns ("manuscript", ...) for the nine published candidates at pH 7.0, and
    ("computed", ...) otherwise. Recomputed descriptors agree closely with the
    manuscript - GRAVY and aliphatic index to 3 decimal places, net charge
    within 0.10, pI within 0.73 - but the published figures are authoritative
    for the published peptides, and mixing the two silently would be worse than
    either. The one descriptor with a genuine convention gap is the hydrophobic
    moment: this module uses the *normalised* Eisenberg consensus scale, which
    runs about 1.57x smaller than the scale behind Table 6 (ratio 1.572 +/-
    0.034 across all nine, so rank order is identical and only the absolute
    magnitude differs).

    Away from pH 7.0 the published block no longer applies - charge and pI are
    pH-dependent - so everything is recomputed.
    """
    cleaned = hf.clean_sequence(seq)
    if abs(ph - 7.0) < 1e-9 and cleaned in PUBLISHED_DESCRIPTORS:
        block = dict(PUBLISHED_DESCRIPTORS[cleaned])
        computed = hf.describe(cleaned, ph=ph)
        # Composition fractions are not tabulated in the manuscript, so they
        # come from the recomputation either way.
        for key in hf.FEATURE_ORDER:
            block.setdefault(key, computed[key])
        return block, "manuscript"
    return hf.describe(cleaned, ph=ph), "computed"


# ===========================================================================
# 1. GATEKEEPER
# ===========================================================================

def _synthetic_corpus(n: int = 1400, seed: int = RANDOM_STATE):
    """
    Build a labelled corpus for the surrogate gatekeeper.

    Sequences are sampled from residue frequencies typical of cationic AMPs,
    then labelled by a latent activity score built from the two mechanisms the
    manuscript's TreeSHAP analysis validated statistically:

      charge term   linear and proportional        (r = 0.8130, rho = 0.8117)
      mu_H term     saturating / threshold-like    (r = 0.7036, rho = 0.9087)

    The mu_H term uses a logistic in (mu_H - 0.26) so that *low* amphipathicity
    is penalised steeply while *high* amphipathicity adds only a modest bonus -
    the asymmetry the bee-swarm plot shows, where the largest negative
    attributions anywhere in the model (phi ~= -0.20) come from peptides lacking
    facial segregation.

    Class balance is held near the manuscript's 29.9% positive prevalence so the
    surrogate inherits a comparable operating point.
    """
    rng = np.random.default_rng(seed)

    # Residue pool weighted toward the composition of curated active AMPs.
    alphabet = list(hf.CANONICAL_AA)
    weights = np.array([
        0.09,  # A
        0.02,  # C
        0.03,  # D
        0.03,  # E
        0.04,  # F
        0.09,  # G
        0.03,  # H
        0.05,  # I
        0.09,  # K
        0.07,  # L
        0.02,  # M
        0.05,  # N
        0.04,  # P
        0.03,  # Q
        0.11,  # R
        0.06,  # S
        0.05,  # T
        0.05,  # V
        0.03,  # W
        0.02,  # Y
    ], dtype=float)
    weights /= weights.sum()

    X, y = [], []
    for _ in range(n):
        length = int(rng.integers(10, 31))          # canonical design window
        seq = "".join(rng.choice(alphabet, size=length, p=weights))
        d = hf.describe(seq)

        # Latent activity score -> probability -> Bernoulli label.
        charge_term = 0.42 * d["charge"]                                  # linear
        amphipathy_term = 3.1 / (1.0 + np.exp(-18.0 * (d["mu_h"] - 0.26)))  # threshold
        gravy_term = 0.28 * d["gravy"]                                    # mild
        # Aliphatic excess is a liability, not an asset, for predicted activity.
        ai_term = -0.004 * max(0.0, d["ai"] - 80.0)
        score = charge_term + amphipathy_term + gravy_term + ai_term - 4.1

        p = 1.0 / (1.0 + np.exp(-score))
        X.append([d[k] for k in hf.FEATURE_ORDER])
        y.append(int(rng.random() < p))

    return np.asarray(X, dtype=float), np.asarray(y, dtype=int)


@dataclass
class Gatekeeper:
    """Random Forest screening layer. ``is_surrogate`` drives the UI banner."""

    model: RandomForestClassifier
    is_surrogate: bool
    feature_names: List[str] = field(default_factory=lambda: list(hf.FEATURE_ORDER))

    # -- prediction ---------------------------------------------------------
    def predict_proba(self, seq: str) -> float:
        """
        p_active for one sequence.

        A sequence published in Table 5 returns its manuscript value verbatim.
        Anything else is scored by the loaded or surrogate forest: the mean of
        the per-tree votes for the positive class.
        """
        cleaned = hf.clean_sequence(seq)
        if cleaned in PUBLISHED_P_ACTIVE:
            return PUBLISHED_P_ACTIVE[cleaned]
        x = np.asarray([hf.feature_vector(cleaned)], dtype=float)
        return float(self.model.predict_proba(x)[0, 1])

    def predict_batch(self, seqs: Sequence[str]) -> List[float]:
        return [self.predict_proba(s) for s in seqs]

    def is_published(self, seq: str) -> str | None:
        """Return the Peptide_NN id if this sequence is one of the paper's nine."""
        return PUBLISHED_ID.get(hf.clean_sequence(seq))


def load_gatekeeper() -> Gatekeeper:
    """
    Load the production forest if it is on disk, otherwise fit the surrogate.

    Wrapped in ``st.cache_resource`` by the caller so the fit happens once per
    server process, not once per rerun.
    """
    if os.path.exists(MODEL_PATH):
        try:
            import joblib
            return Gatekeeper(model=joblib.load(MODEL_PATH), is_surrogate=False)
        except Exception:
            pass  # fall through to the surrogate rather than crashing the app

    X, y = _synthetic_corpus()
    # Hyperparameters mirror the manuscript's benchmarking configuration:
    # balanced class weights for the 70/30 imbalance, bounded depth against
    # overfitting on a small feature space.
    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=12,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X, y)
    return Gatekeeper(model=rf, is_surrogate=True)


# ===========================================================================
# 2. GENERATION - SHAP-steered sampler
# ===========================================================================

#: Residue frequency table biased by the manuscript's attribution findings:
#: cationic residues up (charge is the first-ranked feature), with enough
#: hydrophobic bulk available to let a hydrophobic face form.
_GEN_WEIGHTS = {
    "R": 0.130, "K": 0.085, "G": 0.085, "A": 0.080, "N": 0.060,
    "L": 0.060, "S": 0.055, "V": 0.050, "I": 0.045, "T": 0.045,
    "W": 0.040, "Y": 0.035, "F": 0.035, "P": 0.035, "Q": 0.030,
    "M": 0.025, "H": 0.025, "E": 0.025, "D": 0.020, "C": 0.015,
}


def generate_sequences(
    n: int = 9,
    length_range: Tuple[int, int] = (18, 28),
    temperature: float = 1.0,
    template: str = "",
    seed: int | None = None,
) -> List[str]:
    """
    Stochastic de novo sampler standing in for ProtGPT2.

    The manuscript generates with ProtGPT2 steered by the classifier's own SHAP
    attributions. ProtGPT2 is a ~3 GB autoregressive transformer, far past the
    memory ceiling of a free hosted Streamlit container, so this function
    reproduces the *sampling behaviour* rather than the model:

      temperature  rescales the residue distribution as p_i^(1/T) renormalised.
                   T < 1 sharpens toward the cationic/amphipathic prior
                   (conservative, paper-like candidates); T > 1 flattens it
                   toward uniform (exploratory, more diverse, lower expected
                   p_active). This is the identical transformation applied to
                   transformer logits.
      template     any residues supplied are held fixed as a prefix, so a known
                   motif can be elongated instead of sampled from scratch.

    To wire in the real generator, replace the body with a call to
    ``transformers.pipeline("text-generation", model="nferruz/ProtGPT2")`` and
    keep the same signature - nothing downstream changes.
    """
    rng = np.random.default_rng(seed)
    residues = np.array(list(_GEN_WEIGHTS.keys()))
    base = np.array(list(_GEN_WEIGHTS.values()), dtype=float)

    # Temperature rescaling, exactly as applied to softmax logits.
    t = max(0.05, float(temperature))
    probs = base ** (1.0 / t)
    probs /= probs.sum()

    prefix = hf.clean_sequence(template)
    lo, hi = length_range
    lo, hi = max(10, min(lo, hi)), min(30, max(lo, hi))

    out: List[str] = []
    for _ in range(n):
        target = int(rng.integers(lo, hi + 1))
        remaining = max(0, target - len(prefix))
        tail = "".join(rng.choice(residues, size=remaining, p=probs))
        out.append((prefix + tail)[:30])
    return out


def apply_gate(
    seqs: Sequence[str],
    gatekeeper: Gatekeeper,
    min_len: int = 10,
    max_len: int = 30,
    min_charge: float = 2.0,
    min_mu_h: float = 0.13,
) -> List[dict]:
    """
    Two-layer screening gate.

    Layer 1  the canonical 10-30 residue design window (training-set support).
    Layer 2  mechanistic floors on net charge and amphipathicity, the two
             properties TreeSHAP identified as principal drivers. The mu_H floor
             exists because low amphipathicity produces the model's largest
             negative attributions - a peptide without facial segregation cannot
             insert into the bilayer regardless of its other properties.

    ``min_mu_h`` defaults to 0.13 on the *normalised* Eisenberg scale used by
    hygeia_features, which corresponds to roughly 0.20 on the scale behind
    Table 6. The gate is always evaluated against freshly computed descriptors,
    even for published peptides whose displayed values come from the
    manuscript - otherwise the same threshold would be applied to two different
    scales and published candidates would be judged more leniently than
    generated ones.

    Returns one record per sequence with descriptors, p_active and gate verdict,
    so the UI can show rejected candidates and the reason for rejection rather
    than silently dropping them.
    """
    records = []
    for seq in seqs:
        cleaned = hf.clean_sequence(seq)
        ok_len, len_msg = hf.validate_sequence(cleaned, min_len, max_len)
        if cleaned:
            d, source = descriptors_for(cleaned)   # for display
            gate_d = hf.describe(cleaned)          # for thresholds, one scale
        else:
            d = gate_d = {k: 0.0 for k in hf.FEATURE_ORDER}
            source = "computed"

        reasons = []
        if not ok_len:
            reasons.append(len_msg)
        if ok_len and gate_d["charge"] < min_charge:
            reasons.append(
                f"Net charge {gate_d['charge']:+.2f} below the +{min_charge:.1f} floor."
            )
        if ok_len and gate_d["mu_h"] < min_mu_h:
            reasons.append(
                f"Hydrophobic moment {gate_d['mu_h']:.3f} below the "
                f"{min_mu_h:.2f} floor (≈{min_mu_h * hf.MOMENT_SCALE_RATIO:.2f} "
                f"on the Table 6 scale)."
            )

        passed = not reasons
        records.append({
            "sequence": cleaned,
            "published_id": gatekeeper.is_published(cleaned),
            "descriptor_source": source,
            "passed": passed,
            "reasons": reasons,
            "p_active": gatekeeper.predict_proba(cleaned) if passed else None,
            "aggregation": hf.aggregation_band(d["ai"]),
            **d,
        })
    return records


# ===========================================================================
# 3. ATTRIBUTION - TreeSHAP with a Monte-Carlo Shapley fallback
# ===========================================================================

_BACKGROUND_CACHE: np.ndarray | None = None


def _background_matrix(n: int = 120) -> np.ndarray:
    """
    Reference distribution for absent features, built once per process.

    Shapley values need a notion of "feature not present", supplied here by
    sampling values from a background sample - the standard interventional
    convention.
    """
    global _BACKGROUND_CACHE
    if _BACKGROUND_CACHE is None:
        _BACKGROUND_CACHE, _ = _synthetic_corpus(n=n, seed=7)
    return _BACKGROUND_CACHE


def local_attribution(gatekeeper: Gatekeeper, seq: str,
                      n_permutations: int = 200) -> Tuple[Dict[str, float], float, str]:
    """
    Local SHAP values phi_j for one peptide: why did *this* sequence get *this*
    score?

    Preferred path: ``shap.TreeExplainer``, the exact polynomial-time TreeSHAP
    algorithm the manuscript used, which walks the forest's decision paths and
    returns exact Shapley values for the ensemble.

    Fallback path: a permutation estimator of the same quantity,

        phi_j = E_pi [ f(S_pi,j U {j}) - f(S_pi,j) ]

    averaged over random feature orderings pi, with absent features filled from
    the training background. This is the Monte-Carlo form of the Shapley
    definition in the manuscript; it converges to TreeSHAP's answer and needs no
    extra dependency.

    Returns (phi per feature, base value, method label). Shapley values satisfy
    local accuracy, so base_value + sum(phi) == f(x) in both paths.
    """
    x = np.asarray(hf.feature_vector(seq), dtype=float)

    # ---- exact TreeSHAP, when available ----------------------------------
    try:
        import shap
        explainer = shap.TreeExplainer(gatekeeper.model)
        raw = explainer.shap_values(x.reshape(1, -1))
        # API differences across shap versions: list-of-arrays, or 3-D array.
        if isinstance(raw, list):
            phi = np.asarray(raw[1]).ravel()
            base = float(np.ravel(explainer.expected_value)[1])
        else:
            arr = np.asarray(raw)
            phi = arr[0, :, 1].ravel() if arr.ndim == 3 else arr.ravel()
            ev = np.ravel(explainer.expected_value)
            base = float(ev[1] if ev.size > 1 else ev[0])
        return dict(zip(hf.FEATURE_ORDER, phi.tolist())), base, "TreeSHAP (exact)"
    except Exception:
        pass

    # ---- permutation Shapley fallback (vectorised) ------------------------
    # Every coalition this estimator needs is built up front and scored in ONE
    # predict_proba call. The naive loop issues n_permutations * n_features
    # single-row predictions - thousands of calls against a 400-tree forest,
    # which takes well over a minute and is unusable behind a UI. Batching
    # turns it into one call over a few thousand rows: ~100x faster, identical
    # arithmetic.
    rng = np.random.default_rng(RANDOM_STATE)
    background = _background_matrix()
    base = float(gatekeeper.model.predict_proba(background)[:, 1].mean())

    n_feat = len(hf.FEATURE_ORDER)
    n_perm = int(n_permutations)

    orders = np.argsort(rng.random((n_perm, n_feat)), axis=1)      # random pi
    refs = background[rng.integers(len(background), size=n_perm)]  # fillers

    # rank[p, j] is the position at which feature j is revealed in permutation
    # p, so feature j is present in coalition step k exactly when rank < k.
    # Expressing the reveal as a boolean mask avoids the mixed basic/advanced
    # indexing that silently pairs the wrong axes.
    rank = np.argsort(orders, axis=1)                     # (n_perm, n_feat)
    steps = np.arange(n_feat + 1)[None, :, None]          # (1, n_feat+1, 1)
    present = rank[:, None, :] < steps                    # (n_perm, n_feat+1, n_feat)

    # rows[p, k] is the coalition after revealing the first k features of
    # permutation p: k = 0 is the pure reference, k = n_feat is the full input.
    rows = np.where(present, x[None, None, :], refs[:, None, :])

    flat = rows.reshape(-1, n_feat)
    probs = gatekeeper.model.predict_proba(flat)[:, 1].reshape(n_perm, n_feat + 1)

    # The k-th increment is the marginal contribution of orders[:, k-1].
    increments = np.diff(probs, axis=1)
    phi = np.zeros(n_feat)
    np.add.at(phi, orders.ravel(), increments.ravel())
    phi /= n_perm

    return (dict(zip(hf.FEATURE_ORDER, phi.tolist())), base,
            "Permutation Shapley (vectorised fallback)")


# ===========================================================================
# 4. ACPE - Adaptive Candidate Prioritization Engine
# ===========================================================================

def pareto_mask(objectives: np.ndarray) -> np.ndarray:
    """
    Non-dominated sort. ``objectives`` is (n_candidates, n_objectives) in
    *minimisation* convention - maximisation objectives must be negated first,
    which is why Table 9 of the manuscript lists p_active and aliphatic index as
    negative values.

    Candidate i dominates j when i is no worse on every objective and strictly
    better on at least one. i enters the Pareto set P* only when no j dominates
    it. Pareto dominance is used instead of a weighted sum because collapsing
    four conflicting criteria into one score imposes an exchange rate between
    potency and haemolytic risk that no data justifies.

    O(n^2) and exact. pymoo's NonDominatedSorting is available but unnecessary
    at n = 9, and this keeps the deployment dependency-light.
    """
    n = objectives.shape[0]
    non_dominated = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            no_worse = np.all(objectives[j] <= objectives[i])
            strictly_better = np.any(objectives[j] < objectives[i])
            if no_worse and strictly_better:
                non_dominated[i] = False
                break
    return non_dominated


def acpe_rank(records: Sequence[dict], thresholds: Dict[str, float],
              n_leads: int = 2) -> List[dict]:
    """
    Run the ACPE stage over a candidate pool.

    Objective vector, all converted to minimisation:
        -p_active      maximise predicted antimicrobial probability
        -aliphatic     maximise structural stability / thermostability
        +toxicity      minimise predicted cytotoxicity
        +hemolysis     minimise predicted haemolytic risk

    Three distinct verdicts are attached to each record, because conflating
    them would misrepresent the manuscript's own result:

    ``non_dominated``    Pareto membership. Seven of the paper's nine qualify,
                         so this alone cannot pick a lead.
    ``threshold_checks`` Compliance with the four Research-Question-4 criteria,
                         scored individually. For the published pool *no*
                         candidate clears all four: the aliphatic and activity
                         floors conflict directly with the toxicity and
                         haemolysis ceilings across this set, which is the
                         trade-off the whole Pareto stage exists to expose.
    ``selected_lead``    The manuscript's actual selection rule. Among the
                         non-dominated set, candidates are ranked by combined
                         predicted risk (toxicity + haemolysis) ascending and
                         the lowest ``n_leads`` are advanced. The paper states
                         this explicitly - selection "intentionally prioritized
                         predicted safety and physicochemical viability over
                         predicted potency", reasoning that a non-toxic
                         candidate of moderate activity is a more tractable
                         starting point than a potent one predicted to lyse host
                         cells. On the published pool this rule returns
                         Peptide_07 and Peptide_03, matching Section 6.
    """
    if not records:
        return []

    obj = np.array([
        [
            -r["p_active"],
            -r["ai"],
            r["toxicity"],
            r["hemolysis"],
        ]
        for r in records
    ], dtype=float)

    mask = pareto_mask(obj)
    out = []
    for rec, nd in zip(records, mask):
        checks = {
            "Activity floor": rec["p_active"] >= thresholds["p_active_min"],
            "Stability floor": rec["ai"] >= thresholds["ai_min"],
            "Toxicity ceiling": rec["toxicity"] < thresholds["toxicity_max"],
            "Hemolysis ceiling": rec["hemolysis"] < thresholds["hemolysis_max"],
        }
        enriched = dict(rec)
        enriched["non_dominated"] = bool(nd)
        enriched["threshold_checks"] = checks
        enriched["n_passed"] = sum(checks.values())
        enriched["all_thresholds"] = all(checks.values())
        enriched["risk_score"] = rec["toxicity"] + rec["hemolysis"]
        enriched["selected_lead"] = False
        out.append(enriched)

    # Safety-first lead selection among the non-dominated set.
    front = sorted(
        (r for r in out if r["non_dominated"]),
        key=lambda r: r["risk_score"],
    )
    for position, rec in enumerate(front, start=1):
        rec["risk_rank"] = position
        rec["selected_lead"] = position <= n_leads
    for rec in out:
        rec.setdefault("risk_rank", None)

    return out


def hypervolume_2d(points: np.ndarray, reference: Tuple[float, float]) -> float:
    """
    Exact 2-D hypervolume indicator: the area dominated by the front, bounded by
    a reference point. A single scalar for "how good is this front overall",
    useful for comparing fronts across pipeline runs.

    Points must be in minimisation convention and below the reference on both
    axes. Computed by sweeping sorted points and accumulating slice areas.
    """
    pts = np.asarray([p for p in points if p[0] < reference[0] and p[1] < reference[1]])
    if pts.size == 0:
        return 0.0
    pts = pts[np.argsort(pts[:, 0])]
    area = 0.0
    prev_y = reference[1]
    for x, y in pts:
        if y < prev_y:
            area += (reference[0] - x) * (prev_y - y)
            prev_y = y
    return float(area)
