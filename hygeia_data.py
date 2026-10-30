"""
hygeia_data.py
==============
Single source of truth for every published number in:

    HYGEIA: A Finite Element Modeling and Continuum Mechanics Framework with
    Multi-Objective Pareto Optimization for De Novo Antimicrobial Peptide
    Design in Klebsiella pneumoniae
    Alivio, S. B. Y.; Cuering, P. F. T.; Oros, L. B. L.
    Davao City National High School - DepEd Division Science & Technology Fair 2026

Keeping the paper's tables in one module (instead of scattering literals through
the UI) means the dashboard can never drift out of sync with the manuscript.
Every constant below is tagged with the table it came from.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# PROJECT METADATA
# ---------------------------------------------------------------------------

PROJECT = {
    "acronym": "HYGEIA",
    "title": (
        "An Interpretable Deep Learning and Multi-Objective Pipeline for "
        "De Novo Antimicrobial Peptide Design targeting Klebsiella pneumoniae"
    ),
    "institution": "Davao City National High School",
    "category": "Mathematics and Computational Science - Team Category",
    "authors": [
        "Saravel Bless Y. Alivio",
        "Princess Fely T. Cuering",
        "Lianne Brylle L. Oros",
    ],
    "adviser": "Sherwin S. Fortugaliza, Ph.D.",
    "qualified_scientist": "Remmer Salas, M.S. Chemistry",
    "year": 2026,
}

# The combinatorial search space the pipeline exists to prune:
# sum over L = 10..30 of 20^L  ~= 1.12e39 candidate sequences.
SEQUENCE_SPACE = 1.12e39

# ---------------------------------------------------------------------------
# TABLE 2 - Random Forest gatekeeper performance (test partition)
# Values are point estimates with 95% percentile bootstrap CIs (1,000 iters).
# ---------------------------------------------------------------------------

CLASSIFIER_METRICS = {
    #  name        value    ci_low   ci_high  baseline
    "Accuracy":    (0.8372, 0.7940, 0.8804, 0.7010),
    "Sensitivity": (0.7333, 0.6385, 0.8250, None),
    "Specificity": (0.8815, 0.8341, 0.9209, None),
    "Precision":   (0.7253, 0.6321, 0.8140, 0.2990),
    "NPV":         (0.8857, 0.8421, 0.9279, None),
    "F1":          (0.7293, 0.6503, 0.8037, None),
    "MCC":         (0.6129, 0.5113, 0.7115, 0.0000),
    "AUROC":       (0.8796, 0.8352, 0.9210, 0.5000),
    "AUPRC":       (0.7228, 0.6196, 0.8279, 0.2990),
}

# Confusion matrix underlying the MCC computation (Section 1.2).
CONFUSION = {"TP": 66, "TN": 186, "FP": 25, "FN": 24}

# ---------------------------------------------------------------------------
# TABLE 3 - Feature-set ablation (does fusing embeddings with descriptors help?)
# ---------------------------------------------------------------------------

FEATURE_ABLATION = [
    # label, n_features, cv_auroc, cv_sd, test_acc, test_mcc, test_auroc, test_auprc
    ("Physicochemical only",   12,  0.8597, 0.0145, 0.8140, 0.5620, 0.8682, 0.7315),
    ("PepFoundry embedding",  160,  0.8563, 0.0190, 0.8173, 0.5627, 0.8467, 0.6663),
    ("Fused matrix",          172,  0.8665, 0.0180, 0.8372, 0.6081, 0.8638, 0.6953),
    ("Fused, SHAP-selected",   30,  0.8696, 0.0134, 0.8372, 0.6129, 0.8796, 0.7228),
]

# ---------------------------------------------------------------------------
# TABLES 5 & 6 - The nine ProtGPT2-generated de novo candidates
# p_active = out-of-fold predicted probability from the Random Forest gatekeeper
# charge   = net charge at pH 7.0 via Henderson-Hasselbalch fractions
# mu_h     = Eisenberg hydrophobic moment (amphipathicity)
# gravy    = grand average of hydropathy (Kyte-Doolittle)
# ai       = aliphatic index (thermostability / aggregation proxy)
# pI       = isoelectric point
# ---------------------------------------------------------------------------

CANDIDATES = [
    {
        "id": "Peptide_01", "rank": 1,
        "sequence": "AINAGRINKPRPLRYLLLWLCKA",
        "p_active": 0.7080, "charge": 5.934, "mu_h": 0.252,
        "gravy": 0.078, "ai": 131.74, "pI": 12.22,
        "toxicity": 0.8823, "hemolysis": 0.8825, "pareto": "Non-Dominated",
    },
    {
        "id": "Peptide_02", "rank": 2,
        "sequence": "IINGGKVRTINASRVFMGIWRG",
        "p_active": 0.4046, "charge": 4.975, "mu_h": 0.347,
        "gravy": 0.205, "ai": 101.82, "pI": 12.30,
        "toxicity": 0.8754, "hemolysis": 0.8817, "pareto": "Non-Dominated",
    },
    {
        "id": "Peptide_03", "rank": 3,
        "sequence": "NGWNGNRAFRVYRGAYGWGRV",
        "p_active": 0.3536, "charge": 4.975, "mu_h": 0.274,
        "gravy": -0.957, "ai": 37.14, "pI": 12.49,
        "toxicity": 0.5703, "hemolysis": 0.4448, "pareto": "Non-Dominated",
    },
    {
        "id": "Peptide_04", "rank": 4,
        "sequence": "NGGKRMTIYGRWINRVSLS",
        "p_active": 0.3115, "charge": 4.975, "mu_h": 0.412,
        "gravy": -0.589, "ai": 76.84, "pI": 12.79,
        "toxicity": 0.6340, "hemolysis": 0.5599, "pareto": "Non-Dominated",
    },
    {
        "id": "Peptide_05", "rank": 5,
        "sequence": "RCTANGRNGKHVYIYNKSLGAKLLRA",
        "p_active": 0.2160, "charge": 7.032, "mu_h": 0.250,
        "gravy": -0.623, "ai": 82.69, "pI": 11.62,
        "toxicity": 0.7505, "hemolysis": 0.6825, "pareto": "Non-Dominated",
    },
    {
        "id": "Peptide_06", "rank": 6,
        "sequence": "NCRSMSRARIAVVSLCGRA",
        "p_active": 0.1566, "charge": 4.894, "mu_h": 0.315,
        "gravy": 0.247, "ai": 87.37, "pI": 12.48,
        "toxicity": 0.8751, "hemolysis": 0.8901, "pareto": "Non-Dominated",
    },
    {
        "id": "Peptide_07", "rank": 7,
        "sequence": "EANKHGEPARVRLDTARRRVRGAGRAGA",
        "p_active": 0.1089, "charge": 6.077, "mu_h": 0.379,
        "gravy": -1.196, "ai": 56.07, "pI": 12.61,
        "toxicity": 0.4833, "hemolysis": 0.3496, "pareto": "Non-Dominated",
    },
    {
        "id": "Peptide_08", "rank": 8,
        "sequence": "DAASRRTAGRGRRPARCAAAGGRARA",
        "p_active": 0.0934, "charge": 7.935, "mu_h": 0.319,
        "gravy": -0.981, "ai": 34.62, "pI": 12.96,
        "toxicity": 0.5303, "hemolysis": 0.4247, "pareto": "Dominated",
    },
    {
        "id": "Peptide_09", "rank": 9,
        "sequence": "NGQNGTVRKVYGSGGRQR",
        "p_active": 0.0304, "charge": 4.975, "mu_h": 0.212,
        "gravy": -1.544, "ai": 32.22, "pI": 12.79,
        "toxicity": 0.5181, "hemolysis": 0.4243, "pareto": "Dominated",
    },
]

# The two leads advanced to Normal Mode Analysis (Section 6).
FINAL_LEADS = ["Peptide_03", "Peptide_07"]

# ---------------------------------------------------------------------------
# ACPE acceptance thresholds (Research Question 4)
# ---------------------------------------------------------------------------

ACPE_THRESHOLDS = {
    "p_active_min": 0.50,          # maximise  - screening floor
    "p_active_high_conf": 0.80,    # maximise  - high-confidence tier
    "ai_min": 50.0,                # maximise  - aliphatic thermostability
    "toxicity_max": 0.50,          # minimise  - predicted cytotoxicity
    "hemolysis_max": 0.50,         # minimise  - predicted hemolytic risk
}

# Aliphatic-index aggregation bands (Table 1 of the manuscript).
AI_BANDS = [
    (0.0, 50.0, "Low aggregation risk"),
    (50.0, 80.0, "Normal behaviour"),
    (80.0, float("inf"), "High aggregation risk"),
]

# ---------------------------------------------------------------------------
# SECTION 2.1 - TreeSHAP global importance, mean |phi| over the test partition
# Net charge ranks first at 0.0747, exceeding rank 2 by 61%; the hydrophobic
# moment ranks fourth at 0.0262. Intervening slots are PepFoundry embedding
# dimensions encoding atomic / bonding character.
# ---------------------------------------------------------------------------

SHAP_GLOBAL = [
    ("Net charge (z, pH 7.0)",        0.0747, "physicochemical"),
    ("PepFoundry dim 041",            0.0464, "embedding"),
    ("PepFoundry dim 112",            0.0318, "embedding"),
    ("Hydrophobic moment (mu_H)",     0.0262, "physicochemical"),
    ("PepFoundry dim 007",            0.0241, "embedding"),
    ("GRAVY",                         0.0218, "physicochemical"),
    ("PepFoundry dim 088",            0.0196, "embedding"),
    ("Aliphatic index",               0.0181, "physicochemical"),
    ("PepFoundry dim 134",            0.0167, "embedding"),
    ("Sequence length",               0.0152, "physicochemical"),
    ("PepFoundry dim 019",            0.0141, "embedding"),
    ("Isoelectric point (pI)",        0.0129, "physicochemical"),
    ("PepFoundry dim 156",            0.0118, "embedding"),
    ("Arg fraction",                  0.0104, "physicochemical"),
    ("Lys fraction",                  0.0097, "physicochemical"),
]

# ---------------------------------------------------------------------------
# TABLE 4 - Statistical validation of the SHAP attributions
# Net charge: r ~= rho  -> linear, proportional response.
# mu_H: rho >> r        -> strongly monotonic but non-linear (threshold-like).
# ---------------------------------------------------------------------------

SHAP_VALIDATION = [
    # feature, pearson_r, spearman_rho, interpretation
    ("Net charge (z)", 0.8130, 0.8117,
     "Pearson and Spearman nearly identical: a linear relationship where higher "
     "cationic charge raises predicted activity proportionally."),
    ("Hydrophobic moment (mu_H)", 0.7036, 0.9087,
     "Spearman far exceeds Pearson: strongly monotonic but non-linear. "
     "Amphipathicity acts as a threshold-like necessary condition."),
]

# Largest negative local attribution observed anywhere in the model, produced by
# peptides lacking facial segregation (Section 2.2).
MAX_NEGATIVE_PHI = -0.20

# ---------------------------------------------------------------------------
# TABLE 8 - HPEPDOCK docking, de novo vs dataset controls, per target
# Mann-Whitney U with Bonferroni-corrected alpha = 0.05/3 = 0.0167.
# ---------------------------------------------------------------------------

DOCKING_TARGETS = {
    "OmpA": {
        "pdb": "2K0L",
        "description": "Outer membrane protein A - eight-stranded beta-barrel, "
                       "peptidoglycan-associated; abundant porin-like scaffold.",
        "median_de_novo": -251.17, "iqr_de_novo": (-274.20, -235.05),
        "median_dataset": -261.98, "iqr_dataset": (-272.67, -232.99),
        "U": 42, "p": 0.9314, "cliffs_delta": 0.037,
        "favours": "Dataset", "magnitude": "negligible",
    },
    "OmpW": {
        "pdb": "8QXP",
        "description": "Outer membrane protein W - compact eight-stranded "
                       "beta-barrel with a hydrophobic channel.",
        "median_de_novo": -226.94, "iqr_de_novo": (-239.01, -202.07),
        "median_dataset": -208.60, "iqr_dataset": (-209.87, -199.66),
        "U": 25, "p": 0.1903, "cliffs_delta": -0.383,
        "favours": "De novo", "magnitude": "medium",
    },
    "LptDE": {
        "pdb": "5IV9",
        "description": "Lipopolysaccharide transport complex D/E - essential "
                       "multi-domain LPS translocon heterodimer.",
        "median_de_novo": -222.98, "iqr_de_novo": (-243.10, -219.40),
        "median_dataset": -221.52, "iqr_dataset": (-222.11, -215.81),
        "U": 30, "p": 0.3865, "cliffs_delta": -0.259,
        "favours": "De novo", "magnitude": "small",
    },
}

# TABLE 9 - attainable power at n = 9 per group, alpha = 0.0167.
DOCKING_POWER = [(0.5, 0.065), (1.0, 0.286), (1.5, 0.638), (2.0, 0.897)]

# ---------------------------------------------------------------------------
# TABLES 11-16 - iMODS elastic network / normal mode analysis, six complexes
#
# lambda_1        lowest non-trivial eigenvalue from  H x_k = lambda_k T x_k
# mode1_variance  share of total displacement carried by the first mode
# cum3_variance   cumulative variance over the first three modes
# stiffness_max   maximum elastic-network force constant F_ij on the contact map
# ---------------------------------------------------------------------------

NMA_COMPLEXES = [
    {
        "peptide": "Peptide_03", "target": "LptDE", "pdb": "5IV9",
        "lambda_1": 2.219274e-5, "mode1_variance": 0.31, "cum3_variance": 0.60,
        "stiffness_max": 60.0, "steric_flag": False,
        "mobility": "Moderate terminal mobility spikes; low, stable baseline "
                    "across interior residues.",
    },
    {
        "peptide": "Peptide_03", "target": "OmpA", "pdb": "2K0L",
        "lambda_1": 1.396822e-5, "mode1_variance": 0.30, "cum3_variance": 0.62,
        "stiffness_max": 60.0, "steric_flag": False,
        "mobility": "Moderate terminal mobility spikes; low, stable baseline "
                    "across interior residues.",
    },
    {
        "peptide": "Peptide_03", "target": "OmpW", "pdb": "8QXP",
        "lambda_1": 3.279426e-4, "mode1_variance": 0.30, "cum3_variance": 0.55,
        "stiffness_max": 60.0, "steric_flag": False,
        "mobility": "Elevated deformability at flanking ends around a sharp "
                    "central minimum (~0.2 near index 195): anchored core, "
                    "free termini.",
    },
    {
        "peptide": "Peptide_07", "target": "LptDE", "pdb": "5IV9",
        "lambda_1": 2.063238e-5, "mode1_variance": 0.33, "cum3_variance": 0.61,
        "stiffness_max": 60.0, "steric_flag": False,
        "mobility": "High terminal mobility approaching 1.0 at the extreme tail: "
                    "surface-associated pose with an unrestrained terminus.",
    },
    {
        "peptide": "Peptide_07", "target": "OmpA", "pdb": "2K0L",
        "lambda_1": 1.733397e-5, "mode1_variance": 0.28, "cum3_variance": 0.57,
        "stiffness_max": 12_000.0, "steric_flag": True,
        "mobility": "Heavily suppressed main-chain deformability, resting "
                    "between 0.2 and 0.3.",
    },
    {
        "peptide": "Peptide_07", "target": "OmpW", "pdb": "8QXP",
        "lambda_1": 3.539691e-4, "mode1_variance": 0.32, "cum3_variance": 0.52,
        "stiffness_max": 400_000.0, "steric_flag": True,
        "mobility": "Heavily suppressed main-chain deformability, resting "
                    "between 0.2 and 0.3.",
    },
]

# Four complexes cap at the standard scale of 60.0. Peptide_07 against OmpA and
# OmpW reach 12,000 and 400,000: an isolated subset of springs, not global
# stiffening. F_ij scales steeply as r_ij shrinks, so non-bonded heavy atoms
# overlapping below ~2.0 A receive enormous force constants - the signature of
# severe steric clash in the static docked pose.
STIFFNESS_NORMAL_SCALE = 60.0
STERIC_CLASH_DISTANCE = 2.0  # angstroms

# ---------------------------------------------------------------------------
# PIPELINE PROVENANCE - external tooling, for the "how was this produced" panel
# ---------------------------------------------------------------------------

PIPELINE_STAGES = [
    ("Curation",      "DBAASP / UniProtKB",  "1,000+ MIC-annotated sequences, MHB, 10-30 aa"),
    ("Featurisation", "Physicochemical + PepFoundry", "172 fused dims -> 30 SHAP-selected"),
    ("Gatekeeping",   "Random Forest",       "MCC 0.6129, AUROC 0.8796"),
    ("Attribution",   "TreeSHAP",            "Net charge and amphipathicity dominate"),
    ("Generation",    "ProtGPT2",            "SHAP-steered de novo sampling"),
    ("Folding",       "AlphaFold 3",         "Tertiary coordinates, pLDDT screening"),
    ("Docking",       "HPEPDOCK",            "54 complexes across OmpA / OmpW / LptDE"),
    ("Optimisation",  "ACPE Pareto",         "4 competing objectives, non-dominated sort"),
    ("Dynamics",      "iMODS ENM / NMA",     "Internal-coordinate flexibility screen"),
]

REPOSITORIES = {
    "DBAASP": "https://dbaasp.org/",
    "UniProtKB": "https://www.uniprot.org/",
    "RCSB PDB": "https://www.rcsb.org/",
}
