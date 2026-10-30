"""
hygeia_features.py
==================
Physicochemical descriptor engine. Pure NumPy / stdlib, no modlAMP required.

Every function here reimplements an equation stated in the HYGEIA manuscript so
the dashboard reproduces the published descriptors exactly rather than deferring
to a third-party library whose scale choices may differ. modlAMP is a heavy,
frequently-broken wheel on hosted Python 3.12+, so it stays optional.

Descriptors implemented
-----------------------
net_charge          Henderson-Hasselbalch protonation fractions at a given pH
isoelectric_point   pH at which net charge crosses zero (bisection)
hydrophobic_moment  Eisenberg mu_H over a sliding helical window
gravy               Grand average of hydropathy, Kyte-Doolittle scale
aliphatic_index     Ikai thermostability / aggregation proxy
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

CANONICAL_AA = "ACDEFGHIKLMNPQRSTVWY"

# Kyte-Doolittle hydropathy index. Positive = hydrophobic.
# Used for both GRAVY and, as H_n, the hydrophobic moment.
KYTE_DOOLITTLE: Dict[str, float] = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

# Eisenberg NORMALISED consensus hydrophobicity (Eisenberg et al., 1984), the
# scale modlAMP uses for mu_H and the one this module uses for the moment.
#
# Convention note: several packages ship the *un-normalised* consensus scale,
# whose values are roughly 2.5x larger and which therefore yields moments
# roughly 1.6x larger than the figures in Table 6. Rank order is unaffected -
# the scales are linearly related - but absolute values are not comparable
# across conventions. The dashboard reports the manuscript's published values
# for the manuscript's own nine peptides and computes with this scale for
# anything new, so the two are never silently mixed.
EISENBERG: Dict[str, float] = {
    "A": 0.25, "R": -1.76, "N": -0.64, "D": -0.72, "C": 0.04,
    "Q": -0.69, "E": -0.62, "G": 0.16, "H": -0.40, "I": 0.73,
    "L": 0.53, "K": -1.10, "M": 0.26, "F": 0.61, "P": -0.07,
    "S": -0.26, "T": -0.18, "W": 0.37, "Y": 0.02, "V": 0.54,
}

# Side-chain pKa values plus terminal pKa values (EMBOSS set).
# Basic groups carry positive charge when protonated; acidic groups carry
# negative charge when deprotonated.
PKA_BASIC = {"K": 10.00, "R": 12.00, "H": 6.00}

# Default acidic set: aspartate and glutamate only. Cysteine and tyrosine are
# weakly acidic (pKa 9.0 and 10.0) and contribute ~0.01 charge units at pH 7,
# but they shift pI by a whole unit for a strongly cationic peptide. Restricting
# the set to D/E is the convention that reproduces BOTH the charge and the pI
# columns of Table 6; pass PKA_ACIDIC_EXTENDED to include the weak acids.
PKA_ACIDIC = {"D": 4.05, "E": 4.45}
PKA_ACIDIC_EXTENDED = {"D": 4.05, "E": 4.45, "C": 9.00, "Y": 10.00}
PKA_N_TERM = 9.60   # alpha-amino group, basic
PKA_C_TERM = 3.65   # alpha-carboxyl group, acidic

# Angular rotation per residue for an ideal alpha-helix (manuscript: delta = 100 deg).
HELIX_ANGLE_DEG = 100.0

# Empirical conversion between the two hydrophobic-moment conventions, for
# DISPLAY ONLY - never for computation. Across all nine published candidates the
# ratio (Table 6 value) / (value from the normalised scale above) is
# 1.572 +/- 0.034, i.e. the two scales are linearly related and rank order is
# identical. Multiplying a computed moment by this puts it on the same footing
# as Table 6, so a threshold can be quoted in either convention.
MOMENT_SCALE_RATIO = 1.572

# Aliphatic index coefficients: relative side-chain volume referenced to alanine.
AI_COEFF_VAL = 2.9   # a
AI_COEFF_ILE_LEU = 3.9  # b


# ---------------------------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------------------------

def clean_sequence(seq: str) -> str:
    """Upper-case, strip whitespace/digits/FASTA headers, keep canonical residues."""
    if not seq:
        return ""
    lines = [ln for ln in seq.splitlines() if not ln.strip().startswith(">")]
    merged = "".join(lines).upper()
    return "".join(ch for ch in merged if ch in CANONICAL_AA)


def validate_sequence(seq: str, min_len: int = 10, max_len: int = 30) -> Tuple[bool, str]:
    """
    Layer 1 of the two-layer gate: the canonical design window.

    The manuscript constrains candidates to 10-30 residues, the window over
    which the curated DBAASP training set has MIC coverage. Sequences outside
    it are out-of-distribution for the gatekeeper, so the gate rejects them
    before any descriptor is computed.
    """
    cleaned = clean_sequence(seq)
    if not cleaned:
        return False, "No canonical amino acid residues found."
    if len(cleaned) < min_len:
        return False, f"Length {len(cleaned)} is below the {min_len}-residue design floor."
    if len(cleaned) > max_len:
        return False, f"Length {len(cleaned)} exceeds the {max_len}-residue design ceiling."
    return True, "Sequence accepted into the design window."


# ---------------------------------------------------------------------------
# NET CHARGE - Henderson-Hasselbalch
# ---------------------------------------------------------------------------

def net_charge(seq: str, ph: float = 7.0, c_term_amide: bool = True,
               acidic: Dict[str, float] | None = None) -> float:
    """
    Net charge z at a given pH from Henderson-Hasselbalch protonation fractions.

        z = sum_basic  1 / (1 + 10^(pH - pKa))     [fraction still protonated]
          - sum_acidic 1 / (1 + 10^(pKa - pH))     [fraction deprotonated]

    This is the fractional form, not an integer count of K/R/D/E. It is why the
    manuscript reports charges such as +4.975 and +5.934 rather than whole
    numbers, and it matters mechanistically: net charge is the top-ranked
    TreeSHAP feature (mean |phi| = 0.0747), driving electrostatic attraction to
    the anionic lipopolysaccharide surface of K. pneumoniae.

    ``c_term_amide`` defaults to True, modelling a C-terminally amidated
    peptide. Amidation is near-universal in synthetic AMP design because it
    removes the terminal negative charge and so raises net charge by about +1,
    and the manuscript's reported charges are only reproducible with the
    carboxyl suppressed. Set it False for a free acid C-terminus.
    """
    s = clean_sequence(seq)
    if not s:
        return 0.0

    positive = 1.0 / (1.0 + 10 ** (ph - PKA_N_TERM))      # free alpha-amino terminus
    for residue, pka in PKA_BASIC.items():
        positive += s.count(residue) / (1.0 + 10 ** (ph - pka))

    negative = 0.0
    if not c_term_amide:                                  # free alpha-carboxyl
        negative += 1.0 / (1.0 + 10 ** (PKA_C_TERM - ph))
    for residue, pka in (acidic or PKA_ACIDIC).items():
        negative += s.count(residue) / (1.0 + 10 ** (pka - ph))

    return positive - negative


def isoelectric_point(seq: str, lo: float = 0.0, hi: float = 14.0,
                      tol: float = 1e-4, c_term_amide: bool = False) -> float:
    """
    Isoelectric point: the pH where net_charge crosses zero.

    Net charge is monotonically decreasing in pH, so bisection converges
    unconditionally. No analytic inverse of the Henderson-Hasselbalch sum
    exists, which is why every pI implementation is iterative.

    Note the different default from ``net_charge``: pI is computed with a FREE
    C-terminal carboxyl. This is the universal convention (Expasy and every
    standard pI calculator assume free termini) and it is what the manuscript's
    pI column reproduces. It also matters numerically - a strongly cationic
    peptide with no acidic residues and an amidated C-terminus has no zero
    crossing at all below pH 14, so bisection would simply return the bound.
    """
    s = clean_sequence(seq)
    if not s:
        return 0.0
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if net_charge(s, mid, c_term_amide=c_term_amide) > 0:
            lo = mid        # still net positive, move right
        else:
            hi = mid        # already net negative, move left
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# HYDROPHOBIC MOMENT - Eisenberg
# ---------------------------------------------------------------------------

def hydrophobic_moment(seq: str, window: int | None = None,
                       angle_deg: float = HELIX_ANGLE_DEG,
                       scale: Dict[str, float] | None = None) -> float:
    """
    Eisenberg hydrophobic moment mu_H, maximum over all sliding windows.

        mu_H = (1/N) * sqrt( (sum H_n sin(delta_n))^2 + (sum H_n cos(delta_n))^2 )

    Each residue is placed at delta_n = n * 100 deg around a helical wheel and
    weighted by its hydrophobicity H_n. The vector sum is large only when
    hydrophobic residues cluster on one face of the helix, so mu_H quantifies
    amphipathicity: the facial segregation a peptide needs to bury a hydrophobic
    face in the bilayer while keeping a hydrophilic face solvent-exposed.

    TreeSHAP ranks mu_H fourth globally (0.0262) but Table 4 shows its action is
    non-linear: Spearman rho = 0.9087 against Pearson r = 0.7036. Low
    amphipathicity produces the largest negative attributions anywhere in the
    model (phi ~= -0.20), while high amphipathicity yields only moderate
    positive attributions. It is a necessary, not sufficient, condition.

    ``window`` defaults to the whole sequence, matching the manuscript's
    formulation, which sums over "the sequence window". Pass an integer (11 is
    the usual choice, one helical turn plus a residue) to scan sliding windows
    instead and return the maximum - useful for finding a short amphipathic
    stretch inside an otherwise unremarkable sequence.
    """
    s = clean_sequence(seq)
    if not s:
        return 0.0
    scale = scale or EISENBERG
    win = len(s) if window is None else min(window, len(s))
    radians = math.radians(angle_deg)

    best = 0.0
    for start in range(0, len(s) - win + 1):
        sin_sum = cos_sum = 0.0
        for n, residue in enumerate(s[start:start + win]):
            h = scale.get(residue, 0.0)
            sin_sum += h * math.sin(n * radians)
            cos_sum += h * math.cos(n * radians)
        # Normalising by window length makes windows of different size comparable.
        best = max(best, math.hypot(sin_sum, cos_sum) / win)
    return best


# ---------------------------------------------------------------------------
# GRAVY and ALIPHATIC INDEX
# ---------------------------------------------------------------------------

def gravy(seq: str) -> float:
    """
    Grand average of hydropathy: GRAVY = (1/L) * sum h_i (Kyte-Doolittle).

    Overall hydrophobicity, unlike mu_H which measures its spatial arrangement.
    Positive GRAVY means a net-hydrophobic peptide. In the candidate set only
    ranks 1, 2 and 6 are positive; the rest are hydrophilic down to -1.544.
    """
    s = clean_sequence(seq)
    if not s:
        return 0.0
    return sum(KYTE_DOOLITTLE.get(r, 0.0) for r in s) / len(s)


def aliphatic_index(seq: str) -> float:
    """
    Ikai aliphatic index, the manuscript's structural-stability objective.

        AI = X_Ala + a * X_Val + b * (X_Ile + X_Leu),  a = 2.9, b = 3.9

    X are mole percentages and a, b weight side-chain volume relative to
    alanine. A high AI means a thermostable but aggregation-prone peptide, so
    AI is simultaneously the stability objective to maximise (ACPE floor: 50)
    and a liability: the paper's top-ranked candidate is also the most aliphatic
    (AI = 131.74), which is precisely the trade-off that forces a Pareto stage
    rather than ranking on predicted activity alone.
    """
    s = clean_sequence(seq)
    if not s:
        return 0.0
    L = len(s)
    mol_pct = lambda r: 100.0 * s.count(r) / L  # noqa: E731
    return (
        mol_pct("A")
        + AI_COEFF_VAL * mol_pct("V")
        + AI_COEFF_ILE_LEU * (mol_pct("I") + mol_pct("L"))
    )


def aggregation_band(ai: float) -> str:
    """Map an aliphatic index onto the manuscript's three aggregation bands."""
    if ai < 50.0:
        return "Low aggregation risk"
    if ai < 80.0:
        return "Normal behaviour"
    return "High aggregation risk"


# ---------------------------------------------------------------------------
# AGGREGATE DESCRIPTOR VECTOR
# ---------------------------------------------------------------------------

#: Feature order fed to the gatekeeper. Fixed, because a Random Forest indexes
#: columns positionally - reordering this silently corrupts every prediction.
FEATURE_ORDER: List[str] = [
    "length", "charge", "mu_h", "gravy", "ai", "pI",
    "frac_cationic", "frac_hydrophobic", "frac_aromatic", "frac_gly_pro",
    "frac_cys", "frac_anionic",
]

HYDROPHOBIC_SET = set("AILMFVWY")
AROMATIC_SET = set("FWY")
CATIONIC_SET = set("KRH")
ANIONIC_SET = set("DE")


def describe(seq: str, ph: float = 7.0,
             c_term_amide: bool = True) -> Dict[str, float]:
    """
    Compute the full physicochemical descriptor block for one sequence.

    Returns the twelve-element block corresponding to the manuscript's
    "physicochemical only" ablation arm (Table 3, n_features = 12). The fused
    172-dimension matrix additionally carries PepFoundry embedding dimensions,
    which require the embedding model and are therefore represented in this
    dashboard by their published SHAP attributions rather than recomputed.
    """
    s = clean_sequence(seq)
    if not s:
        return {k: 0.0 for k in FEATURE_ORDER}
    L = len(s)
    frac = lambda members: sum(1 for r in s if r in members) / L  # noqa: E731

    return {
        "length": float(L),
        "charge": net_charge(s, ph, c_term_amide=c_term_amide),
        "mu_h": hydrophobic_moment(s),
        "gravy": gravy(s),
        "ai": aliphatic_index(s),
        # pI keeps the free-carboxyl convention regardless of the charge
        # convention above - see isoelectric_point's docstring.
        "pI": isoelectric_point(s),
        "frac_cationic": frac(CATIONIC_SET),
        "frac_hydrophobic": frac(HYDROPHOBIC_SET),
        "frac_aromatic": frac(AROMATIC_SET),
        "frac_gly_pro": frac(set("GP")),
        "frac_cys": frac(set("C")),
        "frac_anionic": frac(ANIONIC_SET),
    }


def feature_vector(seq: str, ph: float = 7.0) -> List[float]:
    """Descriptor dict flattened into FEATURE_ORDER, ready for the classifier."""
    d = describe(seq, ph)
    return [d[k] for k in FEATURE_ORDER]


# ---------------------------------------------------------------------------
# PROFILE FILE PARSING
# ---------------------------------------------------------------------------

def parse_profile_file(raw: bytes | str) -> Tuple[List[float], List[float]]:
    """
    Parse an iMODS deformability / B-factor export into (index, value).

    Accepts whitespace-, tab- or comma-separated text, with or without a header
    row, comment lines, or trailing columns. One column is read as values with
    indices generated; two or more columns are read as index plus value.

    Hand-rolled rather than handed to ``pandas.read_csv(sep=None)``: the
    delimiter sniffer misreads a single column of decimals, splitting on the
    decimal point and returning one garbage row. Scanning tokens line by line
    cannot make that mistake.

    Raises ValueError when no numeric rows are found, so the caller can report
    a parse failure instead of plotting an empty axis.
    """
    text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw

    idx: List[float] = []
    val: List[float] = []
    for line in text.splitlines():
        line = line.split("#")[0].strip()
        if not line:
            continue
        tokens = [t for t in line.replace(",", " ").replace(";", " ").split() if t]
        numbers: List[float] = []
        for tok in tokens[:3]:
            try:
                numbers.append(float(tok))
            except ValueError:
                break           # a header row, or a trailing text label
        if not numbers:
            continue            # header or comment, skip without complaint
        if len(numbers) == 1:
            idx.append(len(val) + 1.0)
            val.append(numbers[0])
        else:
            idx.append(numbers[0])
            val.append(numbers[1])

    if not val:
        raise ValueError("No numeric rows found — expected one or two columns "
                         "of numbers, optionally with a header or comments.")
    return idx, val


#: Human-readable labels for the SHAP panel.
FEATURE_LABELS = {
    "length": "Sequence length",
    "charge": "Net charge (z)",
    "mu_h": "Hydrophobic moment",
    "gravy": "GRAVY",
    "ai": "Aliphatic index",
    "pI": "Isoelectric point",
    "frac_cationic": "Cationic fraction",
    "frac_hydrophobic": "Hydrophobic fraction",
    "frac_aromatic": "Aromatic fraction",
    "frac_gly_pro": "Gly/Pro fraction",
    "frac_cys": "Cysteine fraction",
    "frac_anionic": "Anionic fraction",
}
