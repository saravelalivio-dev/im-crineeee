"""
HYGEIA - Interactive research dashboard
=======================================
An Interpretable Deep Learning and Multi-Objective Pipeline for De Novo
Antimicrobial Peptide Design targeting Klebsiella pneumoniae

Alivio, S. B. Y. | Cuering, P. F. T. | Oros, L. B. L.
Davao City National High School - DepEd Division Science & Technology Fair 2026

Run with:   streamlit run app.py

Module map
----------
hygeia_data.py      every published table, in one place
hygeia_features.py  physicochemical descriptor equations
hygeia_models.py    gatekeeper, SHAP attribution, ACPE Pareto sort
hygeia_theme.py     pink design system, CSS, Plotly template
app.py              this file - layout and interaction only

Charting conventions held throughout: one y-axis per chart, a legend whenever
two or more series are present, direct labels on the points that matter, and a
single-hue ramp for magnitude. No chart relies on colour alone to carry meaning.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import hygeia_features as hf
import hygeia_models as hm
import hygeia_theme as theme
from hygeia_data import (
    ACPE_THRESHOLDS,
    CANDIDATES,
    CLASSIFIER_METRICS,
    CONFUSION,
    DOCKING_POWER,
    DOCKING_TARGETS,
    FEATURE_ABLATION,
    FINAL_LEADS,
    MAX_NEGATIVE_PHI,
    NMA_COMPLEXES,
    PIPELINE_STAGES,
    PROJECT,
    REPOSITORIES,
    SEQUENCE_SPACE,
    SHAP_GLOBAL,
    SHAP_VALIDATION,
    STERIC_CLASH_DISTANCE,
    STIFFNESS_NORMAL_SCALE,
)

# ===========================================================================
# PAGE SETUP
# ===========================================================================

st.set_page_config(
    page_title="HYGEIA | AMP Design Pipeline",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded",
)
theme.inject(st)


@st.cache_resource(show_spinner="Preparing the Random Forest gatekeeper...")
def get_gatekeeper() -> hm.Gatekeeper:
    """Cached so the forest is fitted once per server process, not per rerun."""
    return hm.load_gatekeeper()


GK = get_gatekeeper()

# Session state holds the generated pool so it survives widget reruns.
st.session_state.setdefault("pool", [])
st.session_state.setdefault("focus_peptide", "Peptide_03")


def plot(fig: go.Figure, key: str, height: int = 430, **kwargs):
    """
    Render a Plotly figure, returning the selection payload when the installed
    Streamlit supports click selection and None otherwise.
    """
    fig.update_layout(height=height)
    try:
        return st.plotly_chart(
            fig, use_container_width=True, key=key,
            on_select="rerun", selection_mode=("points",), **kwargs,
        )
    except TypeError:
        # Streamlit < 1.35 has no selection API; the selectbox controls remain.
        st.plotly_chart(fig, use_container_width=True, key=key, **kwargs)
        return None


# ===========================================================================
# HERO
# ===========================================================================

st.markdown(
    f"""
    <div class="hygeia-hero">
        <h1>🌸 {PROJECT['acronym']}</h1>
        <div class="sub">{PROJECT['title']}</div>
        <div class="byline">
            {' · '.join(PROJECT['authors'])}<br>
            {PROJECT['institution']} — {PROJECT['category']} — {PROJECT['year']}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ===========================================================================
# SIDEBAR
# ===========================================================================

with st.sidebar:
    st.markdown("## 🎀 Pipeline controls")

    if GK.is_surrogate:
        st.warning(
            "**Surrogate gatekeeper active.** The published Random Forest "
            "artefact is not bundled, so a surrogate forest was fitted at "
            "runtime from the directional relationships validated in Table 4. "
            "Published peptides still report their manuscript `p_active`. Drop "
            "`hygeia_rf_gatekeeper.joblib` beside `app.py` to use the real model.",
            icon="⚠️",
        )
    else:
        st.success("Production Random Forest gatekeeper loaded.", icon="✅")

    theme.rule(st, "Chemistry")
    ph = st.slider(
        "pH for charge calculation", 5.0, 9.0, 7.0, 0.1,
        help="Henderson-Hasselbalch protonation fractions are pH-dependent. "
             "The manuscript reports net charge at pH 7.0.",
    )

    theme.rule(st, "ACPE thresholds")
    st.caption("Acceptance criteria from Research Question 4.")
    p_min = st.slider("Minimum p_active", 0.0, 1.0,
                      ACPE_THRESHOLDS["p_active_min"], 0.01)
    ai_min = st.slider("Minimum aliphatic index", 0.0, 140.0,
                       ACPE_THRESHOLDS["ai_min"], 1.0)
    tox_max = st.slider("Maximum toxicity", 0.0, 1.0,
                        ACPE_THRESHOLDS["toxicity_max"], 0.01)
    hem_max = st.slider("Maximum hemolysis", 0.0, 1.0,
                        ACPE_THRESHOLDS["hemolysis_max"], 0.01)

    thresholds = {
        "p_active_min": p_min,
        "p_active_high_conf": ACPE_THRESHOLDS["p_active_high_conf"],
        "ai_min": ai_min,
        "toxicity_max": tox_max,
        "hemolysis_max": hem_max,
    }

    theme.rule(st, "Gatekeeper benchmark")
    st.metric("MCC", f"{CLASSIFIER_METRICS['MCC'][0]:.4f}",
              help="Selection criterion: uses all four confusion cells, so it "
                   "stays honest under the 70/30 class imbalance.")
    st.metric("AUROC", f"{CLASSIFIER_METRICS['AUROC'][0]:.4f}")
    st.caption(
        f"Confusion matrix — TP {CONFUSION['TP']}, TN {CONFUSION['TN']}, "
        f"FP {CONFUSION['FP']}, FN {CONFUSION['FN']}"
    )

    with st.expander("🧪 Pipeline provenance"):
        for stage, tool, detail in PIPELINE_STAGES:
            st.markdown(f"**{stage}** · `{tool}`  \n<small>{detail}</small>",
                        unsafe_allow_html=True)

    with st.expander("📚 Source repositories"):
        for name, url in REPOSITORIES.items():
            st.markdown(f"- [{name}]({url})")
        st.caption(
            "Entirely in silico. No human participants, personal health records "
            "or live animal models were used."
        )

# ===========================================================================
# TABS
# ===========================================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "🧬  Generation & Gate",
    "💗  SHAP Attribution",
    "🎯  Pareto / ACPE",
    "🔬  Structural Dynamics",
])

# ---------------------------------------------------------------------------
# TAB 1 - DE NOVO GENERATION AND GATE SCREENING
# ---------------------------------------------------------------------------
with tab1:
    st.markdown("### De novo sequence generation & two-layer gate screening")
    st.caption(
        f"The design space over 10-30 residues holds roughly "
        f"{SEQUENCE_SPACE:.2e} sequences. The gate exists to make that tractable."
    )

    left, right = st.columns([1, 1.45], gap="large")

    with left:
        theme.rule(st, "Generation parameters")

        template = st.text_input(
            "Template / seed motif (optional)",
            value="",
            placeholder="e.g. NGWNGNR — held fixed as a prefix",
            help="Supplied residues are held fixed and the remainder sampled, "
                 "so a known motif can be elongated instead of sampled afresh.",
        )

        c1, c2 = st.columns(2)
        with c1:
            len_lo = st.slider("Minimum length", 10, 30, 18)
        with c2:
            len_hi = st.slider("Maximum length", 10, 30, 28)

        temperature = st.slider(
            "Temperature", 0.20, 2.00, 1.00, 0.05,
            help="Rescales the residue distribution as p^(1/T), the same "
                 "transformation applied to transformer logits. Below 1 sharpens "
                 "toward the cationic/amphipathic prior; above 1 flattens toward "
                 "uniform for exploration.",
        )
        batch = st.slider("Batch size", 3, 40, 9,
                          help="The manuscript advanced nine candidates.")
        seed = st.number_input("Random seed", 0, 10_000, 42, 1)

        theme.rule(st, "Gate thresholds")
        g1, g2 = st.columns(2)
        with g1:
            min_charge = st.slider("Min net charge", 0.0, 10.0, 2.0, 0.25)
        with g2:
            min_mu = st.slider(
                "Min μ_H", 0.0, 0.45, 0.13, 0.01,
                help="On the normalised Eisenberg scale. The default 0.13 is "
                     "the equivalent of ≈0.20 on the scale behind Table 6, "
                     "just under the lowest published candidate.",
            )
        st.caption(
            f"μ_H floor {min_mu:.2f} ≈ **{min_mu * hf.MOMENT_SCALE_RATIO:.2f}** "
            "on the Table 6 scale. Thresholds are always evaluated on computed "
            "descriptors, so published and generated candidates are judged alike."
        )

        run = st.button("🌸  Generate & screen", use_container_width=True)
        load_paper = st.button("📄  Load the nine published candidates",
                               use_container_width=True)

    if run:
        seqs = hm.generate_sequences(
            n=int(batch),
            length_range=(int(len_lo), int(len_hi)),
            temperature=float(temperature),
            template=template,
            seed=int(seed),
        )
        st.session_state["pool"] = hm.apply_gate(
            seqs, GK, min_charge=float(min_charge), min_mu_h=float(min_mu)
        )

    if load_paper:
        st.session_state["pool"] = hm.apply_gate(
            [c["sequence"] for c in CANDIDATES], GK,
            min_charge=float(min_charge), min_mu_h=float(min_mu),
        )

    pool = st.session_state["pool"]

    with right:
        if not pool:
            st.info(
                "Generate a batch, or load the nine published candidates to "
                "reproduce Tables 5 and 6 of the manuscript.",
                icon="🌷",
            )
        else:
            passed = [r for r in pool if r["passed"]]
            theme.rule(st, "Screening summary")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Generated", len(pool))
            m2.metric("Cleared the gate", len(passed),
                      delta=f"{len(passed)/len(pool):.0%} yield")
            if passed:
                best = max(passed, key=lambda r: r["p_active"])
                m3.metric("Best p_active", f"{best['p_active']:.4f}")
                m4.metric("Mean net charge",
                          f"{np.mean([r['charge'] for r in passed]):+.2f}")

            # Descriptor table. Rows are the gate's own verdicts, so rejected
            # candidates stay visible with their reason rather than vanishing.
            df = pd.DataFrame([{
                "ID": r["published_id"] or "—",
                "Sequence": r["sequence"],
                "Values": "manuscript" if r["descriptor_source"] == "manuscript"
                          else "computed",
                "L": int(r["length"]),
                "p_active": r["p_active"],
                "z (charge)": round(r["charge"], 3),
                "μ_H": round(r["mu_h"], 3),
                "GRAVY": round(r["gravy"], 3),
                "AI": round(r["ai"], 2),
                "pI": round(r["pI"], 2),
                "Aggregation": r["aggregation"],
                "Gate": "Pass" if r["passed"] else "Reject",
            } for r in pool]).sort_values(
                "p_active", ascending=False, na_position="last"
            )

            st.dataframe(
                df, use_container_width=True, hide_index=True,
                column_config={
                    "p_active": st.column_config.ProgressColumn(
                        "p_active", min_value=0.0, max_value=1.0, format="%.4f",
                    ),
                },
            )

            rejected = [r for r in pool if not r["passed"]]
            if rejected:
                with st.expander(f"⚠️ {len(rejected)} sequence(s) rejected by the gate"):
                    for r in rejected:
                        st.markdown(
                            f'<div class="hygeia-seq">{r["sequence"] or "(empty)"}</div>',
                            unsafe_allow_html=True,
                        )
                        for reason in r["reasons"]:
                            st.caption(f"↳ {reason}")

    # ---- Classification output card ---------------------------------------
    if pool:
        theme.rule(st, "Classification output")

        options = [r["sequence"] for r in pool if r["passed"]]
        if not options:
            st.warning("No sequence cleared the gate, so there is nothing to score.")
        else:
            labels = {
                r["sequence"]: f"{r['published_id'] or 'Generated'} — {r['sequence'][:22]}…"
                for r in pool if r["passed"]
            }
            chosen = st.selectbox(
                "Candidate", options, format_func=lambda s: labels[s],
                key="tab1_candidate",
            )
            rec = next(r for r in pool if r["sequence"] == chosen)

            # Published values at pH 7.0, recomputed at any other pH - charge
            # and pI are pH-dependent, so the sidebar control stays live.
            d, d_source = hm.descriptors_for(chosen, ph=ph)

            cc1, cc2 = st.columns([1, 1.9], gap="large")
            with cc1:
                tier = ("High confidence" if rec["p_active"] >= 0.80
                        else "Above screening floor" if rec["p_active"] >= 0.50
                        else "Below screening floor")
                tier_kind = ("pass" if rec["p_active"] >= 0.50 else "warn")
                st.markdown(
                    f"""
                    <div class="hygeia-verdict">
                        <div class="cap">Predicted antimicrobial activity</div>
                        <div class="big">{rec['p_active']:.4f}</div>
                        <div class="cap">p_active · Random Forest gatekeeper</div>
                        <div style="margin-top:10px">
                            {theme.chip(tier, tier_kind)}
                            {theme.chip(rec['aggregation'],
                                        'warn' if 'High' in rec['aggregation'] else 'pass')}
                            {theme.chip(rec['published_id'], 'lead') if rec['published_id'] else ''}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown(f'<div class="hygeia-seq">{chosen}</div>',
                            unsafe_allow_html=True)

            with cc2:
                g1, g2, g3 = st.columns(3)
                g1.metric("Net charge z", f"{d['charge']:+.3f}", help=f"At pH {ph:.1f}")
                g2.metric("Hydrophobic moment", f"{d['mu_h']:.3f}")
                g3.metric("GRAVY", f"{d['gravy']:+.3f}")
                g4, g5, g6 = st.columns(3)
                g4.metric("Aliphatic index", f"{d['ai']:.2f}")
                g5.metric("Isoelectric point", f"{d['pI']:.2f}")
                g6.metric("Length", f"{int(d['length'])} aa")
                st.caption(
                    "Descriptors as published in Table 6."
                    if d_source == "manuscript" else
                    "Descriptors computed from the manuscript's equations. "
                    "Note the hydrophobic moment uses the normalised Eisenberg "
                    "scale, ~1.57× smaller in absolute terms than Table 6's "
                    "scale; rank order is identical."
                )

                # Residue composition. Single series, so no legend is needed -
                # the title names it; magnitude is carried by the rose ramp.
                counts = pd.Series(list(chosen)).value_counts().reindex(
                    list(hf.CANONICAL_AA), fill_value=0
                )
                fig = go.Figure(go.Bar(
                    x=counts.index.tolist(), y=counts.values.tolist(),
                    marker=dict(
                        color=counts.values.tolist(),
                        colorscale=theme.ROSE_SEQUENTIAL,
                        line=dict(color=theme.SURFACE, width=2),  # 2px surface gap
                    ),
                    text=[c if c else "" for c in counts.values],
                    textposition="outside",
                    hovertemplate="%{x}: %{y} residue(s)<extra></extra>",
                ))
                fig.update_layout(
                    title="Residue composition",
                    yaxis_title="Count", xaxis_title="Amino acid",
                    showlegend=False, bargap=0.28,
                )
                plot(fig, key="composition", height=265)

        with st.expander("📐 How each descriptor is computed"):
            st.markdown(
                r"""
**Net charge (Henderson-Hasselbalch).** Protonation fractions, not an integer
count of K/R/D/E:

$$z = \sum_{\text{basic}} \frac{1}{1 + 10^{\,pH - pK_a}} - \sum_{\text{acidic}} \frac{1}{1 + 10^{\,pK_a - pH}}$$

This is why the manuscript reports charges such as +4.975. Net charge is the
top-ranked TreeSHAP feature (mean $|\phi|$ = 0.0747), driving electrostatic
attraction to the anionic lipopolysaccharide surface of *K. pneumoniae*.

**Hydrophobic moment (Eisenberg).** Each residue is placed at
$\delta_n = n \times 100^\circ$ around a helical wheel and weighted by its
hydrophobicity:

$$\mu_H = \frac{1}{N}\sqrt{\Big(\sum_n H_n \sin \delta_n\Big)^2 + \Big(\sum_n H_n \cos \delta_n\Big)^2}$$

Large only when hydrophobic residues cluster on one helical face, so $\mu_H$
quantifies the facial segregation required for bilayer insertion.

**GRAVY.** $\text{GRAVY} = \frac{1}{L}\sum_i h_i$ on the Kyte-Doolittle scale —
overall hydrophobicity, as distinct from its spatial arrangement.

**Aliphatic index.**
$\text{AI} = X_{Ala} + 2.9\,X_{Val} + 3.9\,(X_{Ile} + X_{Leu})$, with $X$ in mole
percent. High AI means thermostable *and* aggregation-prone, which is exactly
why selection cannot rank on predicted activity alone.

**Isoelectric point.** The pH at which $z = 0$, found by bisection: the
Henderson-Hasselbalch sum has no analytic inverse.
                """
            )

# ---------------------------------------------------------------------------
# TAB 2 - EXPLAINABLE AI (SHAP) ATTRIBUTION DASHBOARD
# ---------------------------------------------------------------------------
with tab2:
    st.markdown("### Explainable AI — TreeSHAP attribution")
    st.caption(
        "Shapley values answer a question accuracy cannot: *why* did this "
        "peptide receive this score, and did the model learn biophysics or "
        "dataset artefacts?"
    )

    theme.rule(st, "Global feature importance")
    gl, gr = st.columns([1.5, 1], gap="large")

    with gl:
        # Two categories -> legend present, and both are direct-labelled.
        labels = [s[0] for s in SHAP_GLOBAL][::-1]
        values = [s[1] for s in SHAP_GLOBAL][::-1]
        kinds = [s[2] for s in SHAP_GLOBAL][::-1]

        fig = go.Figure()
        for kind, colour, name in [
            ("physicochemical", theme.CATEGORICAL[0], "Physicochemical descriptor"),
            ("embedding", theme.CATEGORICAL[1], "PepFoundry embedding dim"),
        ]:
            mask = [k == kind for k in kinds]
            fig.add_trace(go.Bar(
                y=[l for l, m in zip(labels, mask) if m],
                x=[v for v, m in zip(values, mask) if m],
                name=name, orientation="h",
                marker=dict(color=colour, line=dict(color=theme.SURFACE, width=2)),
                text=[f"{v:.4f}" for v, m in zip(values, mask) if m],
                textposition="outside",
                hovertemplate="%{y}<br>mean |φ| = %{x:.4f}<extra></extra>",
            ))
        fig.update_layout(
            title="Mean |φ| across the test partition",
            xaxis_title="Mean absolute SHAP value",
            barmode="overlay", bargap=0.35,
            xaxis=dict(range=[0, max(values) * 1.3]),
        )
        plot(fig, key="shap_global", height=520)

    with gr:
        theme.card(st, "What the ranking shows", f"""
        <p><b>Net charge ranks first</b> at mean |φ| = 0.0747, exceeding the
        second-ranked feature by 61%. The hydrophobic moment ranks fourth at
        0.0262; the intervening slots are PepFoundry embedding dimensions
        encoding atomic and bonding character.</p>
        <p>The model received <b>no biological instructions</b>, yet
        independently elevated net charge and amphipathicity — the two
        properties established across decades of AMP research as the primary
        determinants of membrane disruption. It converged on genuine
        biophysics rather than dataset artefacts.</p>
        <p>Global importance measures <i>magnitude</i>, not direction. The
        signed view below is what reveals the asymmetry.</p>
        """)
        st.metric("Largest negative local attribution", f"{MAX_NEGATIVE_PHI:.2f}",
                  help="Produced by peptides lacking facial segregation — the "
                       "model penalises low amphipathicity heavily.")

    theme.rule(st, "Statistical validation of the attributions")
    vdf = pd.DataFrame(
        [(f, r, rho, f"{rho - r:+.4f}", note) for f, r, rho, note in SHAP_VALIDATION],
        columns=["Feature", "Pearson r", "Spearman ρ", "ρ − r", "Interpretation"],
    )
    st.dataframe(vdf, use_container_width=True, hide_index=True)
    st.caption(
        "Net charge: r ≈ ρ, a linear and proportional response. "
        "μ_H: ρ ≫ r, strongly monotonic but non-linear — amphipathicity acts "
        "as a threshold-like necessary condition, not a proportional one. "
        "All attributions significant under Bonferroni correction (p from 1e-21 to 1e-115)."
    )

    theme.rule(st, "Local attribution — why this peptide scored as it did")

    pool = st.session_state["pool"]
    sources = ([(r["published_id"] or r["sequence"][:18], r["sequence"])
                for r in pool if r["passed"]]
               or [(c["id"], c["sequence"]) for c in CANDIDATES])

    pick = st.selectbox(
        "Peptide", sources, format_func=lambda t: t[0], key="tab2_pick",
    )
    seq = pick[1]

    with st.spinner("Computing Shapley values..."):
        phi, base, method = hm.local_attribution(GK, seq)

    ll, lr = st.columns([1.6, 1], gap="large")

    with ll:
        # Waterfall: signed contributions, so a diverging encoding with a
        # neutral baseline. Sorted by magnitude, largest nearest the output.
        items = sorted(phi.items(), key=lambda kv: abs(kv[1]), reverse=True)[:10]
        items = items[::-1]
        names = [hf.FEATURE_LABELS.get(k, k) for k, _ in items]
        vals = [v for _, v in items]

        fig = go.Figure(go.Waterfall(
            orientation="h",
            y=names, x=vals,
            measure=["relative"] * len(vals),
            base=base,
            increasing=dict(marker=dict(color=theme.CATEGORICAL[0])),
            decreasing=dict(marker=dict(color=theme.CATEGORICAL[1])),
            connector=dict(line=dict(color=theme.PETAL, width=1.5)),
            text=[f"{v:+.4f}" for v in vals],
            textposition="outside",
            hovertemplate="%{y}<br>φ = %{x:+.4f}<extra></extra>",
        ))
        fig.add_vline(x=base, line=dict(color=theme.INK_MUTED, dash="dot", width=1.5),
                      annotation_text=f"base {base:.3f}", annotation_position="top")
        fig.update_layout(
            title=f"Local SHAP contributions — {pick[0]}",
            xaxis_title="Contribution to p_active (φ)",
        )
        plot(fig, key="shap_local", height=470)

    with lr:
        total = base + sum(phi.values())
        st.metric("Base value E[f(x)]", f"{base:.4f}")
        st.metric("Σ φ (all features)", f"{sum(phi.values()):+.4f}")
        st.metric("Reconstructed f(x)", f"{total:.4f}")
        st.caption(
            f"Method: **{method}**. Shapley values satisfy local accuracy, so "
            "base + Σφ reconstructs the model output exactly."
        )
        st.markdown(f'<div class="hygeia-seq">{seq}</div>', unsafe_allow_html=True)

        with st.expander("The Shapley definition"):
            st.markdown(
                r"""
$$\phi_j = \sum_{S \subseteq F \setminus \{j\}} \frac{|S|!\,(|F|-|S|-1)!}{|F|!}\big[f_x(S \cup \{j\}) - f_x(S)\big]$$

- $F$ — the complete feature set
- $S$ — any subset excluding feature $j$
- $f_x(S)$ — predicted probability using only the features in $S$

The bracketed term is the change produced by adding feature $j$ to subset $S$;
the factorial coefficient weights each subset so every feature is assessed
fairly across all orderings. A positive $\phi_j$ pushes the prediction toward
activity, a negative one away. TreeSHAP computes this exactly for tree
ensembles in polynomial time.
                """
            )

    theme.rule(st, "Feature-set ablation")
    adf = pd.DataFrame(FEATURE_ABLATION, columns=[
        "Feature set", "n features", "CV AUROC", "CV SD",
        "Test accuracy", "Test MCC", "Test AUROC", "Test AUPRC",
    ])
    st.dataframe(adf, use_container_width=True, hide_index=True)
    st.caption(
        "Fusing descriptors with embeddings lifts test MCC from 0.5620 and "
        "0.5627 to 0.6081. Reducing 172 dimensions to 30 SHAP-selected features "
        "raises CV AUROC to 0.8696 while cutting fold-to-fold variance from "
        "0.0180 to 0.0134 — uninformative dimensions removed without discarding "
        "predictive signal."
    )

# ---------------------------------------------------------------------------
# TAB 3 - MULTI-OBJECTIVE PARETO OPTIMISATION (ACPE)
# ---------------------------------------------------------------------------
with tab3:
    st.markdown("### Adaptive Candidate Prioritization Engine — Pareto front")
    st.caption(
        "Four objectives in conflict: maximise predicted activity and aliphatic "
        "stability, minimise cytotoxicity and haemolytic risk. Pareto dominance "
        "avoids imposing an exchange rate between potency and safety that no "
        "data justifies."
    )

    ranked = hm.acpe_rank(CANDIDATES, thresholds)
    pdf = pd.DataFrame(ranked)

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Candidates in pool", len(ranked))
    s2.metric("Non-dominated", int(pdf["non_dominated"].sum()),
              help="No other candidate is at least as good on all four "
                   "objectives and strictly better on one.")
    s3.metric("Clear all four thresholds", int(pdf["all_thresholds"].sum()),
              help="The activity and stability floors conflict directly with "
                   "the toxicity and haemolysis ceilings across this pool — "
                   "which is the trade-off the Pareto stage exists to expose.")
    s4.metric("Selected as leads", int(pdf["selected_lead"].sum()),
              help="Lowest combined predicted risk (toxicity + haemolysis) "
                   "within the non-dominated set.")

    theme.rule(st, "Objective space")

    axis_options = {
        "Predicted activity (p_active)": "p_active",
        "Aliphatic index": "ai",
        "Predicted toxicity": "toxicity",
        "Predicted hemolysis": "hemolysis",
        "Net charge": "charge",
        "Hydrophobic moment": "mu_h",
        "GRAVY": "gravy",
    }

    cc = st.columns([1, 1, 1, 1.1])
    with cc[0]:
        x_label = st.selectbox("X axis", list(axis_options), index=0)
    with cc[1]:
        y_label = st.selectbox("Y axis", list(axis_options), index=3)
    with cc[2]:
        z_label = st.selectbox("Z axis (3-D)", list(axis_options), index=1)
    with cc[3]:
        view = st.radio("View", ["2-D", "3-D"], horizontal=True)

    xk, yk, zk = axis_options[x_label], axis_options[y_label], axis_options[z_label]

    nd = [r for r in ranked if r["non_dominated"]]
    dom = [r for r in ranked if not r["non_dominated"]]
    leads = [r for r in ranked if r["id"] in FINAL_LEADS]

    hover = (
        "<b>%{customdata[0]}</b><br>%{customdata[1]}<br>"
        "p_active %{customdata[2]:.4f}<br>AI %{customdata[3]:.2f}<br>"
        "Toxicity %{customdata[4]:.4f}<br>Hemolysis %{customdata[5]:.4f}"
        "<extra></extra>"
    )
    cd = lambda rs: [[r["id"], r["sequence"], r["p_active"], r["ai"],  # noqa: E731
                      r["toxicity"], r["hemolysis"]] for r in rs]

    if view == "2-D":
        fig = go.Figure()
        # Dominated first so non-dominated marks draw on top.
        if dom:
            fig.add_trace(go.Scatter(
                x=[r[xk] for r in dom], y=[r[yk] for r in dom],
                mode="markers+text", name="Dominated",
                marker=dict(size=14, color=theme.CATEGORICAL[1], symbol="x-thin",
                            line=dict(color=theme.CATEGORICAL[1], width=2.5)),
                text=[r["id"].replace("Peptide_", "P") for r in dom],
                textposition="bottom center",
                textfont=dict(size=10, color=theme.INK_MUTED),
                customdata=cd(dom), hovertemplate=hover,
            ))
        fig.add_trace(go.Scatter(
            x=[r[xk] for r in nd], y=[r[yk] for r in nd],
            mode="markers+text", name="Non-dominated (P*)",
            marker=dict(size=16, color=theme.CATEGORICAL[0], symbol="circle",
                        line=dict(color=theme.SURFACE, width=2)),  # 2px surface ring
            text=[r["id"].replace("Peptide_", "P") for r in nd],
            textposition="top center",
            textfont=dict(size=11, color=theme.ROSE_DEEP),
            customdata=cd(nd), hovertemplate=hover,
        ))
        fig.add_trace(go.Scatter(
            x=[r[xk] for r in leads], y=[r[yk] for r in leads],
            mode="markers+text", name="Final leads",
            marker=dict(size=26, color="rgba(0,0,0,0)", symbol="star-diamond-open",
                        line=dict(color=theme.ROSE_DEEP, width=3)),
            text=[r["id"] for r in leads], textposition="middle right",
            textfont=dict(size=12, color=theme.ROSE_DEEP, family=theme.FONT_STACK),
            customdata=cd(leads), hovertemplate=hover,
        ))

        # Threshold guides, drawn only when the chosen axis is a gated objective.
        guides = {"p_active": p_min, "ai": ai_min,
                  "toxicity": tox_max, "hemolysis": hem_max}
        if xk in guides:
            fig.add_vline(x=guides[xk], line=dict(color=theme.ROSE_LIGHT,
                          dash="dash", width=2),
                          annotation_text="ACPE threshold")
        if yk in guides:
            fig.add_hline(y=guides[yk], line=dict(color=theme.ROSE_LIGHT,
                          dash="dash", width=2),
                          annotation_text="ACPE threshold")

        fig.update_layout(title=f"{y_label} against {x_label}",
                          xaxis_title=x_label, yaxis_title=y_label)
        event = plot(fig, key="pareto2d", height=540)
    else:
        fig = go.Figure()
        for group, name, colour, symbol in [
            (nd, "Non-dominated (P*)", theme.CATEGORICAL[0], "circle"),
            (dom, "Dominated", theme.CATEGORICAL[1], "x"),
        ]:
            if not group:
                continue
            fig.add_trace(go.Scatter3d(
                x=[r[xk] for r in group], y=[r[yk] for r in group],
                z=[r[zk] for r in group],
                mode="markers+text", name=name,
                marker=dict(size=9, color=colour, symbol=symbol,
                            line=dict(color=theme.SURFACE, width=1.5)),
                text=[r["id"].replace("Peptide_", "P") for r in group],
                textfont=dict(size=10, color=theme.INK_SOFT),
                customdata=cd(group), hovertemplate=hover,
            ))
        if leads:
            fig.add_trace(go.Scatter3d(
                x=[r[xk] for r in leads], y=[r[yk] for r in leads],
                z=[r[zk] for r in leads],
                mode="markers+text", name="Final leads",
                marker=dict(size=15, color=theme.ROSE_DEEP, symbol="diamond",
                            line=dict(color=theme.SURFACE, width=2)),
                text=[r["id"] for r in leads],
                textfont=dict(size=11, color=theme.ROSE_DEEP),
                customdata=cd(leads), hovertemplate=hover,
            ))
        fig.update_layout(
            title=f"{x_label} / {y_label} / {z_label}",
            scene=dict(xaxis_title=x_label, yaxis_title=y_label, zaxis_title=z_label),
        )
        theme.style_3d(fig)
        event = plot(fig, key="pareto3d", height=600)

    # Click selection, when the installed Streamlit supports it.
    clicked = None
    try:
        pts = (event or {}).get("selection", {}).get("points", [])
        if pts:
            cdata = pts[0].get("customdata")
            if cdata:
                clicked = cdata[0]
    except Exception:
        clicked = None

    theme.rule(st, "Candidate inspector")
    ids = [r["id"] for r in ranked]
    default_idx = ids.index(clicked) if clicked in ids else ids.index(FINAL_LEADS[0])
    focus_id = st.selectbox(
        "Select a candidate (or click a point above)", ids,
        index=default_idx, key="tab3_focus",
    )
    rec = next(r for r in ranked if r["id"] == focus_id)
    st.session_state["focus_peptide"] = focus_id

    il, ir = st.columns([1, 1.5], gap="large")
    with il:
        chips = "".join(
            theme.chip(f"{k}: {'pass' if v else 'fail'}", "pass" if v else "fail")
            for k, v in rec["threshold_checks"].items()
        )
        verdict = ("Selected lead" if rec["selected_lead"]
                   else "Non-dominated (P*)" if rec["non_dominated"]
                   else "Dominated")
        kind = ("pass" if rec["selected_lead"]
                else "warn" if rec["non_dominated"] else "fail")
        risk_chip = (
            theme.chip(f"Risk rank {rec['risk_rank']} of {int(pdf['non_dominated'].sum())}",
                       "pass" if rec["risk_rank"] and rec["risk_rank"] <= 2 else "")
            if rec["risk_rank"] else ""
        )
        st.markdown(
            f"""
            <div class="hygeia-verdict">
                <div class="cap">{rec['id']}</div>
                <div class="big">{rec['p_active']:.4f}</div>
                <div class="cap">p_active</div>
                <div style="margin-top:10px">
                    {theme.chip(verdict, kind)}
                    {risk_chip}
                    {theme.chip('Advanced to NMA', 'lead') if rec['id'] in FINAL_LEADS else ''}
                </div>
                <div style="margin-top:8px">{chips}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(f'<div class="hygeia-seq">{rec["sequence"]}</div>',
                    unsafe_allow_html=True)

    with ir:
        # Objective profile on a common 0-1 scale. AI is divided by its pool
        # maximum rather than plotted raw, because mixing a 0-1 probability with
        # a 0-132 index on one axis would make the probabilities invisible -
        # and a second y-axis is never the answer.
        ai_max = max(r["ai"] for r in ranked)
        bars = [
            ("Predicted activity", rec["p_active"], "maximise"),
            ("Aliphatic index (scaled)", rec["ai"] / ai_max, "maximise"),
            ("Predicted toxicity", rec["toxicity"], "minimise"),
            ("Predicted hemolysis", rec["hemolysis"], "minimise"),
        ]
        fig = go.Figure()
        for direction, colour, nm in [
            ("maximise", theme.CATEGORICAL[0], "Maximise"),
            ("minimise", theme.CATEGORICAL[1], "Minimise"),
        ]:
            sel = [(n, v) for n, v, d in bars if d == direction]
            fig.add_trace(go.Bar(
                y=[n for n, _ in sel], x=[v for _, v in sel],
                name=nm, orientation="h",
                marker=dict(color=colour, line=dict(color=theme.SURFACE, width=2)),
                text=[f"{v:.3f}" for _, v in sel], textposition="outside",
                hovertemplate="%{y}: %{x:.4f}<extra></extra>",
            ))
        fig.update_layout(
            title=f"Objective profile — {rec['id']}",
            xaxis=dict(range=[0, 1.18], title="Normalised objective value"),
            barmode="overlay", bargap=0.4,
        )
        plot(fig, key="objective_profile", height=330)

        hv = hm.hypervolume_2d(
            np.array([[-r["p_active"], r["hemolysis"]] for r in nd]),
            reference=(0.0, 1.0),
        )
        st.metric("Front hypervolume (p_active × hemolysis)", f"{hv:.4f}",
                  help="Area dominated by the non-dominated front against the "
                       "reference point (0, 1). One scalar for overall front "
                       "quality across pipeline runs.")

    theme.rule(st, "Full ACPE table")
    show = pdf[["id", "sequence", "p_active", "ai", "toxicity", "hemolysis",
                "risk_score", "non_dominated", "n_passed", "selected_lead"]].copy()
    show.columns = ["ID", "Sequence", "p_active", "Aliphatic index", "Toxicity",
                    "Hemolysis", "Combined risk", "Non-dominated",
                    "Thresholds passed", "Selected lead"]
    st.dataframe(
        show.sort_values(["Selected lead", "Non-dominated", "Combined risk"],
                         ascending=[False, False, True]),
        use_container_width=True, hide_index=True,
        column_config={
            "Thresholds passed": st.column_config.ProgressColumn(
                "Thresholds passed", min_value=0, max_value=4, format="%d/4",
            ),
        },
    )

    st.info(
        "**Seven of the nine candidates are non-dominated**, so Pareto "
        "membership alone cannot pick a lead. Nor does threshold filtering "
        "settle it: across this pool the activity and stability floors conflict "
        "directly with the toxicity and haemolysis ceilings, and no candidate "
        "clears all four at once. That conflict *is* the finding — it is why the "
        "pipeline needs a Pareto stage instead of a weighted score.\n\n"
        "The selection therefore ranks the front by **combined predicted risk** "
        "and advances the two lowest: **Peptide_07** (0.833) and **Peptide_03** "
        "(1.015). This deliberately prioritises predicted safety over predicted "
        "potency, on the reasoning that a non-toxic candidate of moderate "
        "activity is a more tractable starting point for optimisation than a "
        "potent one predicted to lyse host cells. Ranking on predicted "
        "probability alone would instead have advanced Peptide_01 — the most "
        "aliphatic and most aggregation-prone sequence in the pool.",
        icon="🎀",
    )
    st.caption(
        "A Pareto-versus-dominated statistical comparison was omitted: at n = 9 "
        "only 36 distinct rank arrangements exist, giving a minimum attainable "
        "two-sided p of 0.0556 — above the corrected threshold of 0.0125."
    )

# ---------------------------------------------------------------------------
# TAB 4 - STRUCTURAL DYNAMICS AND CONTINUUM MECHANICS
# ---------------------------------------------------------------------------
with tab4:
    st.markdown("### Structural dynamics — elastic network model & normal mode analysis")
    st.caption(
        "Docking reports whether a pose is complementary; it does not report "
        "whether that pose survives thermal motion. NMA in internal coordinates "
        "is the flexibility screen between docking and molecular dynamics."
    )

    theme.rule(st, "Molecular docking against outer-membrane targets")

    tcols = st.columns(3)
    for col, (name, t) in zip(tcols, DOCKING_TARGETS.items()):
        with col:
            favours_kind = "pass" if t["favours"] == "De novo" else "warn"
            theme.card(st, f"{name} · {t['pdb']}", f"""
            <p>{t['description']}</p>
            <p><b>De novo median</b> {t['median_de_novo']:.2f}<br>
               <b>Dataset median</b> {t['median_dataset']:.2f}</p>
            <p>U = {t['U']} · p = {t['p']:.4f}<br>
               Cliff's δ = {t['cliffs_delta']:+.3f} ({t['magnitude']})</p>
            {theme.chip(f"Favours: {t['favours']}", favours_kind)}
            <p><a href="https://www.rcsb.org/structure/{t['pdb']}" target="_blank">
               RCSB PDB {t['pdb']} ↗</a></p>
            """)

    dl, dr = st.columns([1.5, 1], gap="large")
    with dl:
        # Grouped bars, two series -> legend plus direct labels. Scores are
        # negative, so lower is better; the axis title says so.
        names = list(DOCKING_TARGETS)
        fig = go.Figure()
        for key, colour, nm in [
            ("median_de_novo", theme.CATEGORICAL[0], "De novo candidates"),
            ("median_dataset", theme.CATEGORICAL[1], "Dataset controls"),
        ]:
            iqr_key = "iqr_de_novo" if "de_novo" in key else "iqr_dataset"
            vals = [DOCKING_TARGETS[n][key] for n in names]
            lo = [abs(DOCKING_TARGETS[n][iqr_key][0] - DOCKING_TARGETS[n][key])
                  for n in names]
            hi = [abs(DOCKING_TARGETS[n][iqr_key][1] - DOCKING_TARGETS[n][key])
                  for n in names]
            fig.add_trace(go.Bar(
                x=names, y=vals, name=nm,
                marker=dict(color=colour, line=dict(color=theme.SURFACE, width=2)),
                error_y=dict(type="data", symmetric=False, array=hi, arrayminus=lo,
                             color=theme.INK_MUTED, thickness=1.5, width=6),
                text=[f"{v:.1f}" for v in vals], textposition="outside",
                hovertemplate="%{x} — %{y:.2f}<extra></extra>",
            ))
        fig.update_layout(
            title="HPEPDOCK median binding score by target (IQR whiskers)",
            yaxis_title="Binding score — more negative is more favourable",
            xaxis_title="Outer-membrane target", bargap=0.3, barmode="group",
        )
        plot(fig, key="docking", height=430)

    with dr:
        theme.card(st, "Reading the non-significance", """
        <p>All 54 complexes produced strongly negative scores (group medians
        −208.60 to −261.98), so both groups formed stable complexes with every
        target.</p>
        <p>No test reached the Bonferroni-corrected α = 0.0167. That is a
        <b>power</b> result, not an equivalence result: at n = 9 per group the
        design could only detect very large differences.</p>
        """)
        powdf = pd.DataFrame(
            [(f"{d:.1f} SD", f"{p:.1%}") for d, p in DOCKING_POWER],
            columns=["True difference", "Attainable power"],
        )
        st.dataframe(powdf, use_container_width=True, hide_index=True)
        st.caption(
            "The largest effect observed (Cliff's δ = −0.383 against OmpW) is "
            "about 0.7 SD, where attainable power is below 20%. The meaningful "
            "finding is parity: sequences designed without reference to any "
            "target structure docked comparably to experimentally validated "
            "actives."
        )

    theme.rule(st, "Elastic network normal mode analysis")

    st.markdown(
        r"""
The complex is coarse-grained into a harmonic graph where spatial proximity
places Hookean springs, with total elastic potential energy

$$V = \tfrac{1}{2}\sum_{i<j} F_{ij}\big(r_{ij} - r_{ij}^{0}\big)^2$$

Modes come from the generalised eigenvalue problem $H x_k = \lambda_k T x_k$,
where $H$ is the Hessian (the stiffness landscape), $T$ the kinetic-energy
matrix (the torsional analogue of a moment of inertia), $x_k$ the eigenvector
giving the direction and relative amplitude of motion, and
$\lambda_k = \omega_k^2$ the energetic cost of that deformation. Working in
internal coordinates restricts the variables to backbone $\phi/\psi$ dihedrals
with bond lengths and angles as holonomic constraints, reducing $3N$ degrees of
freedom to roughly $2N_{res}$, guaranteeing stereochemically valid
conformations and removing the six trivial rigid-body modes without post-hoc
filtering.
        """
    )

    ndf = pd.DataFrame(NMA_COMPLEXES)
    ndf["complex"] = ndf["peptide"].str.replace("Peptide_0", "P") + " × " + ndf["target"]

    e1, e2 = st.columns([1.35, 1], gap="large")

    with e1:
        # Log axis: eigenvalues span 1.4e-5 to 3.5e-4, so a linear axis would
        # flatten five of six bars to nothing.
        fig = go.Figure()
        for pep, colour in [("Peptide_03", theme.CATEGORICAL[0]),
                            ("Peptide_07", theme.CATEGORICAL[2])]:
            sub = ndf[ndf["peptide"] == pep]
            fig.add_trace(go.Bar(
                x=sub["target"], y=sub["lambda_1"], name=pep,
                marker=dict(color=colour, line=dict(color=theme.SURFACE, width=2)),
                text=[f"{v:.2e}" for v in sub["lambda_1"]], textposition="outside",
                hovertemplate="%{x}<br>λ₁ = %{y:.6e}<extra></extra>",
            ))
        fig.update_layout(
            title="Lowest non-trivial eigenvalue λ₁ by complex",
            yaxis=dict(title="λ₁ (log scale)", type="log"),
            xaxis_title="Target", barmode="group", bargap=0.3,
        )
        plot(fig, key="eigen", height=400)
        st.caption(
            "Thermal amplitude scales inversely with λ under equipartition, so a "
            "low eigenvalue denotes motion excited at little energetic cost. "
            "Stiffness tracks the **receptor architecture**, not the peptide: "
            "the compact eight-stranded OmpW barrel occupies the highest tier "
            "(≈3.3–3.5 × 10⁻⁴) while the larger OmpA barrel and the multi-domain "
            "LptDE heterodimer return ≈1.4–2.2 × 10⁻⁵. All six spectra ascend "
            "smoothly across 20 modes with no degenerate, zero-frequency or "
            "negative eigenvalues — none of the complexes has an unconstrained "
            "global unfolding trajectory."
        )

    with e2:
        # Variance is a share of a whole: two series on one 0-100% axis.
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=ndf["complex"], x=ndf["mode1_variance"] * 100,
            name="Mode 1", orientation="h",
            marker=dict(color=theme.CATEGORICAL[0],
                        line=dict(color=theme.SURFACE, width=2)),
            text=[f"{v*100:.0f}%" for v in ndf["mode1_variance"]],
            textposition="inside", insidetextanchor="middle",
            hovertemplate="%{y}<br>Mode 1: %{x:.0f}%<extra></extra>",
        ))
        fig.add_trace(go.Bar(
            y=ndf["complex"],
            x=(ndf["cum3_variance"] - ndf["mode1_variance"]) * 100,
            name="Modes 2–3", orientation="h",
            marker=dict(color=theme.CATEGORICAL[4],
                        line=dict(color=theme.SURFACE, width=2)),
            text=[f"{v*100:.0f}%" for v in ndf["cum3_variance"]],
            textposition="outside",
            hovertemplate="%{y}<br>Cumulative 3 modes<extra></extra>",
        ))
        fig.update_layout(
            title="Variance captured by the lowest modes",
            xaxis=dict(title="Share of total dynamic variance (%)", range=[0, 78]),
            barmode="stack", bargap=0.35,
        )
        plot(fig, key="variance", height=400)
        st.caption(
            "Variance is inversely proportional to eigenvalue, so this is a "
            "direct accounting of how total displacement partitions across the "
            "spectrum. Mode 1 carries 28–33% and the first three modes 52–62% in "
            "every complex — the candidates participate in the collective "
            "vibrational motion of the binding pocket rather than fluctuating as "
            "uncorrelated noise."
        )

    theme.rule(st, "Harmonic spring stiffness & the steric-clash anomaly")

    s1, s2 = st.columns([1.35, 1], gap="large")
    with s1:
        # Status colours (reserved, never reused as series hues) and both a
        # label and a legend, so the flag is never colour-alone.
        fig = go.Figure()
        for flagged, colour, nm in [
            (False, theme.STATUS["good"], "Physically realistic packing"),
            (True, theme.STATUS["critical"], "Steric clash detected"),
        ]:
            sub = ndf[ndf["steric_flag"] == flagged]
            if sub.empty:
                continue
            fig.add_trace(go.Bar(
                x=sub["complex"], y=sub["stiffness_max"], name=nm,
                marker=dict(color=colour, line=dict(color=theme.SURFACE, width=2)),
                text=[f"{v:,.0f}" for v in sub["stiffness_max"]],
                textposition="outside",
                hovertemplate="%{x}<br>max F_ij = %{y:,.0f}<extra></extra>",
            ))
        fig.add_hline(
            y=STIFFNESS_NORMAL_SCALE,
            line=dict(color=theme.ROSE, dash="dash", width=2),
            annotation_text=f"normal scale {STIFFNESS_NORMAL_SCALE:.0f}",
        )
        fig.update_layout(
            title="Maximum elastic-network force constant F_ij",
            yaxis=dict(title="max F_ij (log scale)", type="log"),
            xaxis_title="Complex", bargap=0.32,
        )
        plot(fig, key="stiffness", height=420)

    with s2:
        theme.card(st, "Why 400,000 is a red flag, not a strong bond", f"""
        <p>Four complexes — Peptide_03 across all three targets and Peptide_07
        against LptDE — cap at the standard scale of
        {STIFFNESS_NORMAL_SCALE:.0f}, with uniform contact maps corresponding to
        realistic van der Waals and electrostatic packing.</p>
        <p>Peptide_07 against OmpA and OmpW reach 12,000 and 400,000. In both
        anomalous maps nearly every contact stays light, proving the scale is
        driven by an <b>isolated subset of springs</b>, not global stiffening.</p>
        <p>Because the spring constant F<sub>ij</sub> rises steeply as the
        interatomic distance r<sub>ij</sub> shrinks, non-bonded heavy atoms
        sitting below ~{STERIC_CLASH_DISTANCE:.1f} Å receive enormous force
        constants. These unphysical springs pin the backbone
        inside the harmonic network — the signature of a severe steric clash in
        the static docked pose, which is precisely what a flexibility screen is
        supposed to catch before an expensive production run.</p>
        """)

    theme.rule(st, "Deformability & B-factor profiles")

    complex_names = ndf["complex"].tolist()
    # Default to the complex whose profile the manuscript describes in most
    # detail. Positional index into the list, not a DataFrame index label.
    default_complex = ("P3 × OmpW" if "P3 × OmpW" in complex_names
                       else complex_names[0])
    sel_complex = st.selectbox(
        "Complex", complex_names,
        index=complex_names.index(default_complex),
        key="tab4_complex",
    )
    row = ndf[ndf["complex"] == sel_complex].iloc[0]

    pl, pr = st.columns([1.6, 1], gap="large")
    with pl:
        uploaded = st.file_uploader(
            "Upload the iMODS deformability export for this complex (optional)",
            type=["txt", "dat", "csv"],
            help="Two columns — residue index and deformability. When supplied, "
                 "the measured profile is plotted instead of the schematic.",
        )

        measured = None
        if uploaded is not None:
            try:
                cols, vals = hf.parse_profile_file(uploaded.getvalue())
                measured = (np.asarray(cols, dtype=float),
                            np.asarray(vals, dtype=float))
                st.success(f"Parsed {len(vals):,} residues from {uploaded.name}.",
                           icon="✅")
            except Exception as exc:
                st.error(f"Could not parse that file: {exc}")

        if measured is not None:
            idx, defo = measured
            source_note = "Measured iMODS export"
        else:
            # ---- SCHEMATIC ONLY --------------------------------------------
            # Reconstructed from the manuscript's *verbal* description of each
            # profile so the shape can be discussed in the UI. It is NOT iMODS
            # output and is labelled as such on the chart and in the caption.
            rng = np.random.default_rng(abs(hash(sel_complex)) % (2**31))
            n_receptor, n_peptide = 182, 23
            idx = np.arange(1, n_receptor + n_peptide + 1)

            # Receptor: low baseline punctuated by periodic loop peaks - the
            # contrast between rigid hydrogen-bonded transmembrane beta-strands
            # and flexible extracellular / periplasmic loops.
            defo = 0.18 + 0.05 * rng.random(len(idx))
            loops = 0.42 * np.abs(np.sin(idx / 9.0)) ** 6
            defo[:n_receptor] += loops[:n_receptor]

            tail = np.arange(n_peptide)
            if row["peptide"] == "Peptide_03" and row["target"] == "OmpW":
                # Anchored core, free termini: elevated ends around a sharp
                # central minimum of ~0.2 near index 195.
                centre = n_peptide / 2
                defo[n_receptor:] = 0.20 + 0.55 * ((tail - centre) / centre) ** 2
            elif row["peptide"] == "Peptide_07" and row["target"] == "LptDE":
                # Surface-associated pose: terminal mobility approaching 1.0.
                defo[n_receptor:] = 0.25 + 0.72 * (tail / max(1, n_peptide - 1)) ** 2.4
            elif row["steric_flag"]:
                # Pinned by unphysical springs: suppressed, 0.2-0.3 throughout.
                defo[n_receptor:] = 0.21 + 0.09 * rng.random(n_peptide)
            else:
                # Moderate terminal spikes over a low, stable interior.
                defo[n_receptor:] = 0.22 + 0.30 * (
                    np.exp(-tail / 2.6) + np.exp(-(n_peptide - 1 - tail) / 2.6)
                )
            defo = np.clip(defo, 0.0, 1.05)
            source_note = "Schematic reconstruction — not iMODS output"

        n_receptor_line = 182 if measured is None else None

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=idx, y=defo, mode="lines", name="Main-chain deformability",
            line=dict(color=theme.CATEGORICAL[0], width=2),
            fill="tozeroy", fillcolor="rgba(216,27,116,0.10)",
            hovertemplate="Residue %{x}<br>Deformability %{y:.3f}<extra></extra>",
        ))
        # B-factor is the same mobility rescaled by 8π²/3 to compare against
        # crystallographic atomic displacement parameters. Plotted normalised,
        # because two y-axes are never the answer.
        bfac = defo * (8 * np.pi**2 / 3)
        fig.add_trace(go.Scatter(
            x=idx, y=bfac / bfac.max() * defo.max(), mode="lines",
            name="B-factor profile (normalised, ×8π²/3)",
            line=dict(color=theme.CATEGORICAL[4], width=2, dash="dot"),
            hovertemplate="Residue %{x}<extra></extra>",
        ))
        if n_receptor_line:
            fig.add_vline(
                x=n_receptor_line,
                line=dict(color=theme.ROSE_DEEP, dash="dash", width=2),
                annotation_text="receptor ends / peptide begins",
                annotation_position="top left",
            )
        fig.update_layout(
            title=f"{sel_complex} — {source_note}",
            xaxis_title="Residue index", yaxis_title="Deformability (arbitrary units)",
        )
        plot(fig, key="deformability", height=420)

        if measured is None:
            st.warning(
                "This trace is a **schematic reconstruction** drawn from the "
                "manuscript's written description of the profile, so the shape "
                "is discussable but the values are not measurements. Upload the "
                "iMODS export above to plot the real profile.",
                icon="📐",
            )

    with pr:
        st.metric("λ₁", f"{row['lambda_1']:.4e}")
        st.metric("Mode 1 variance", f"{row['mode1_variance']*100:.0f}%")
        st.metric("Cumulative, 3 modes", f"{row['cum3_variance']*100:.0f}%")
        st.metric("Max F_ij", f"{row['stiffness_max']:,.0f}",
                  delta="steric clash" if row["steric_flag"] else "normal packing",
                  delta_color="inverse" if row["steric_flag"] else "normal")
        theme.card(st, "Reported mobility pattern", f"<p>{row['mobility']}</p>")
        st.caption(
            "Main-chain deformability measures each residue's intrinsic capacity "
            "to deform under collective mode excitation, locating flexible hinges "
            "and rigid anchors. Scaling by 8π²/3 converts normal-mode mobility "
            "into B-factors comparable with crystallographic displacement "
            "parameters. For Peptide_03, the calculated curves closely trace the "
            "experimental B-factor topologies deposited for LptDE and OmpW, "
            "confirming the physical fidelity of the elastic network."
        )

    theme.rule(st, "All six complexes")
    tbl = ndf[["complex", "peptide", "target", "pdb", "lambda_1",
               "mode1_variance", "cum3_variance", "stiffness_max", "steric_flag"]].copy()
    tbl.columns = ["Complex", "Peptide", "Target", "PDB", "λ₁",
                   "Mode 1 var.", "Cum. 3 modes", "max F_ij", "Steric clash"]
    tbl["λ₁"] = tbl["λ₁"].map(lambda v: f"{v:.4e}")
    tbl["Mode 1 var."] = tbl["Mode 1 var."].map(lambda v: f"{v*100:.0f}%")
    tbl["Cum. 3 modes"] = tbl["Cum. 3 modes"].map(lambda v: f"{v*100:.0f}%")
    tbl["max F_ij"] = tbl["max F_ij"].map(lambda v: f"{v:,.0f}")
    st.dataframe(tbl, use_container_width=True, hide_index=True)

    st.success(
        "**Peptide_03 is the more robust lead.** It holds normal force constants "
        "against all three targets, its calculated mobility reproduces deposited "
        "experimental B-factor topologies, and its profile shows an anchored core "
        "with free termini. Peptide_07 remains viable against LptDE but its OmpA "
        "and OmpW poses need re-docking or refinement before molecular dynamics.",
        icon="🌸",
    )

    with st.expander("🧊 Docked-complex structure viewer"):
        st.markdown(
            "Drop a docked complex (`.pdb` / `.cif`) here to inspect it in 3-D. "
            "Rendering uses `stmol` + `py3Dmol` when both are installed; "
            "otherwise the file summary below is shown."
        )
        struct = st.file_uploader("Docked complex", type=["pdb", "cif", "ent"],
                                  key="structure_upload")
        if struct is not None:
            raw = struct.getvalue().decode("utf-8", errors="replace")
            try:
                import py3Dmol
                from stmol import showmol

                view = py3Dmol.view(width=900, height=520)
                view.addModel(raw, struct.name.rsplit(".", 1)[-1])
                # Receptor as a pink cartoon, peptide ligand highlighted.
                view.setStyle({"cartoon": {"color": "spectrum"}})
                view.addStyle({"chain": "B"},
                              {"stick": {"colorscheme": "magentaCarbon"}})
                view.setBackgroundColor("#FFF5F9")
                view.zoomTo()
                showmol(view, height=520, width=900)
            except ImportError:
                atoms = sum(1 for ln in raw.splitlines() if ln.startswith("ATOM"))
                chains = sorted({ln[21] for ln in raw.splitlines()
                                 if ln.startswith("ATOM") and len(ln) > 21})
                c1, c2, c3 = st.columns(3)
                c1.metric("File", struct.name)
                c2.metric("ATOM records", f"{atoms:,}")
                c3.metric("Chains", ", ".join(chains) or "—")
                st.info(
                    "Install the optional viewer to render this inline:  \n"
                    "`pip install stmol py3Dmol`",
                    icon="💡",
                )

# ===========================================================================
# FOOTER
# ===========================================================================

st.markdown("---")
st.caption(
    f"🌸 **{PROJECT['acronym']}** — {PROJECT['institution']} · "
    f"{PROJECT['category']} · {PROJECT['year']}  \n"
    f"Proponents: {', '.join(PROJECT['authors'])} · "
    f"Research adviser: {PROJECT['adviser']} · "
    f"Qualified scientist: {PROJECT['qualified_scientist']}  \n"
    "Entirely in silico, built on open-access repositories (DBAASP, UniProtKB, "
    "RCSB PDB). Predicted values are computational hypotheses for prioritising "
    "wet-lab work, not evidence of antimicrobial efficacy or safety — "
    "experimental validation is required."
)
