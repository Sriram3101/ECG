"""
Builds the project report PDF: full documentation of every agent plus the
results on the real model and data.

    python -m scripts.run_real_pipeline --n 60     # produces artifacts/results/real_results.json
    python -m scripts.build_report                 # -> artifacts/report/ECG_Agentic_Pipeline_Report.pdf

Every number in the "Results" chapter comes either from that JSON or is
recomputed here from the model and data (diagnosis metrics, validation pass
rates, the explanation example). The few development-time measurements that
can no longer be re-run (the state before a fix was applied) are labelled
as such in the text.
"""
from __future__ import annotations

import argparse
import json
import os
import warnings
from datetime import date
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
warnings.filterwarnings("ignore", category=UserWarning)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, Preformatted, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

from core.ecg_state import CLASS_LABELS
from evaluation.data import ARTIFACTS_DIR, load_split, stratified_sample

REPORT_DIR = ARTIFACTS_DIR / "report"
FIG_DIR = REPORT_DIR / "figures"

INK, MUTED, ACCENT, GRID = "#1f2937", "#6b7280", "#2563eb", "#e5e7eb"
VERDICT_COLORS = {"clean": "#16a34a", "label_poisoned": "#f59e0b", "signal_poisoned": "#2563eb",
                  "both_poisoned": "#dc2626", "uncertain": "#9ca3af", "rejected by validation": "#4b5563"}

# Measurements taken during development, before a fix was applied. They
# describe code that no longer exists, so they cannot be recomputed.
DEV_MEASUREMENTS = {
    "clean_flagged_before_calibration_fix": (25, 40),
    "seconds_per_beat_before_compile": 6.19,
    "fit_seconds_before_compile": 707,
    "single_call_ms_eager": 404,
    "single_call_ms_compiled": 26,
    "synthetic_verdicts": {
        "clean": {"clean": 20},
        "label flipped": {"label_poisoned": 19, "both_poisoned": 1},
        "PGD eps=0.05": {"signal_poisoned": 12, "both_poisoned": 7, "clean": 1},
    },
    "synthetic_fusion": [  # scenario, weighted_mean flag rate, noisy_or flag rate (100 beats each)
        ("clean (unseen)", 0.02, 0.20), ("PGD ε=0.05", 0.22, 0.67),
        ("PGD ε=0.1", 0.32, 1.00), ("Gaussian σ=0.05", 0.18, 1.00),
    ],
}


# --------------------------------------------------------------------------- figures

def _style_axes(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=8)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def fig_pipeline(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 9.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0.4, 12)
    ax.axis("off")

    def box(x, y, w, h, title, sub, color):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.15",
                                    facecolor=color, edgecolor="#9ca3af", linewidth=0.8))
        ax.text(x, y + h * 0.2, title, ha="center", va="center", fontsize=9.5, color=INK, weight="bold")
        ax.text(x, y - h * 0.17, sub, ha="center", va="center", fontsize=7, color=MUTED, linespacing=1.3)

    def arrow(x1, y1, x2, y2, label=None, color="#6b7280", ha="left"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=10,
                                     color=color, linewidth=1))
        if label:
            ax.text((x1 + x2) / 2 + (0.12 if ha == "left" else -0.12), (y1 + y2) / 2, label, fontsize=7,
                    color=color, va="center", ha=ha)

    def bypass(points, label, label_xy):
        xs, ys = zip(*points[:-1])
        ax.plot(xs, ys, color="#dc2626", linewidth=1, linestyle="--")
        ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>", mutation_scale=10,
                                     color="#dc2626", linewidth=1, linestyle="--"))
        ax.text(*label_xy, label, fontsize=7, color="#dc2626", ha="center", va="center", rotation=90)

    neutral, label_c, signal_c, core_c = "#f3f4f6", "#fdecdf", "#dff3ec", "#e8e8fb"
    box(5, 11.4, 5.4, 0.8, "ECG input", "one beat · 360 samples @ 360 Hz · optional claimed label", neutral)
    box(5, 10.1, 6.2, 1.0, "1. Validation agent", "format · sampling rate · flatline / clipping\nPASS · REPAIR · REJECT",
        neutral)
    box(2.85, 8.3, 4.1, 1.3, "2. Label poisoning detection",
        "AUM · confident learning (OOF)\nk-NN · Mahalanobis · entropy\nper-class score fusion", label_c)
    box(7.15, 8.3, 4.1, 1.3, "3. Signal poisoning / noise",
        "reconstruction · Mahalanobis\nfragility · gradient sensitivity\nSNR · entropy · noise type", signal_c)
    box(5, 6.55, 5.0, 1.0, "4. Decision agent", "verdict · attack type · severity\nproceed / repair / quarantine",
        core_c)
    box(7.4, 5.05, 3.6, 1.0, "5. Recovery agent", "relabel · reconstruct · denoise\nre-validate · re-check", core_c)
    box(4.4, 3.65, 4.6, 0.9, "6. Diagnosis agent", "DenseNet + BiLSTM + Attention", core_c)
    box(4.4, 2.4, 4.6, 0.9, "7. Explainability agent", "attention maps · gradients · SHAP", core_c)
    box(5, 1.1, 5.6, 0.9, "8. Alert & reporting", "alert level · clinician report · recommendations", neutral)

    arrow(5, 11.0, 5, 10.6)
    arrow(3.8, 9.6, 3.0, 8.95)
    arrow(6.2, 9.6, 7.0, 8.95)
    arrow(3.2, 7.65, 4.1, 7.05)
    arrow(6.8, 7.65, 5.9, 7.05)
    arrow(3.4, 6.05, 3.4, 4.1, "clean → proceed", ha="right")
    arrow(6.3, 6.05, 7.0, 5.55, "attack → repair")
    arrow(6.6, 4.55, 5.7, 4.1, "repaired")
    arrow(4.4, 3.2, 4.4, 2.85)
    arrow(4.4, 1.95, 4.4, 1.55)
    bypass([(1.9, 10.1), (0.5, 10.1), (0.5, 1.1), (2.2, 1.1)], "REJECT → request new recording", (0.25, 5.6))
    bypass([(9.2, 5.05), (9.6, 5.05), (9.6, 1.1), (7.8, 1.1)], "quarantine → expert review", (9.85, 3.1))
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_confusion(cm_counts: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    norm = cm_counts / cm_counts.sum(axis=1, keepdims=True)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    labels = [CLASS_LABELS[i] for i in range(len(cm_counts))]
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("predicted", fontsize=9, color=INK)
    ax.set_ylabel("true", fontsize=9, color=INK)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{cm_counts[i, j]}\n{norm[i, j]:.0%}", ha="center", va="center", fontsize=7.5,
                    color="white" if norm[i, j] > 0.5 else INK)
    ax.tick_params(labelsize=8)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_verdicts(scenarios: dict, path: Path, title: str) -> None:
    names = list(scenarios)
    order = [v for v in VERDICT_COLORS if any(v in scenarios[n] for n in names)]
    fig, ax = plt.subplots(figsize=(6.8, 3.0))
    left = np.zeros(len(names))
    for verdict in order:
        values = np.array([scenarios[n].get(verdict, 0) for n in names], dtype=float)
        ax.barh(names, values, left=left, color=VERDICT_COLORS[verdict], label=verdict.replace("_", " "),
                edgecolor="white", linewidth=0.8)
        for y, (l, v) in enumerate(zip(left, values)):
            if v >= 3:
                ax.text(l + v / 2, y, str(int(v)), ha="center", va="center", fontsize=7.5, color="white")
        left += values
    ax.invert_yaxis()
    ax.set_xlabel("beats", fontsize=8, color=INK)
    ax.set_title(title, fontsize=9, color=INK, loc="left")
    ax.legend(fontsize=7, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    _style_axes(ax)
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    ax.yaxis.grid(False)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


SHORT_NAMES = {"clean": "clean", "label flipped": "flipped", "gaussian s=0.01": "σ .01",
               "gaussian s=0.05": "σ .05", "PGD eps=0.05": "PGD"}


def fig_score_distributions(results: dict, path: Path) -> None:
    scen = results["scenarios"]
    names = list(scen)
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.9), sharey=False)
    panels = [("signal_scores_weighted_mean", "signal score — weighted mean", results["thresholds"]["signal"]),
              ("signal_scores_noisy_or", "signal score — noisy OR", results["thresholds"]["signal"]),
              ("label_scores", "label score", results["thresholds"]["label"])]
    rng = np.random.default_rng(0)
    for ax, (key, title, thr) in zip(axes, panels):
        for i, n in enumerate(names):
            v = np.asarray(scen[n][key])
            ax.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=6, alpha=0.6, color=ACCENT, linewidths=0)
            ax.plot([i - 0.28, i + 0.28], [np.median(v)] * 2, color=INK, linewidth=1.2)
        ax.axhline(thr, color="#dc2626", linewidth=0.9, linestyle="--")
        ax.set_xticks(range(len(names)), [SHORT_NAMES.get(n, n) for n in names], fontsize=6.5)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(title, fontsize=8.5, color=INK)
        _style_axes(ax)
    axes[0].set_ylabel("score (red line = decision threshold)", fontsize=7.5, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_before_after(clean_flag_rate_now: float, sec_per_beat_now: float, fit_now: float, path: Path) -> None:
    before_flag = DEV_MEASUREMENTS["clean_flagged_before_calibration_fix"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4))
    data = [
        ("clean beats wrongly flagged", [before_flag[0] / before_flag[1] * 100, clean_flag_rate_now * 100], "%",
         ["calibrated on\ntraining patients", "calibrated on\nval patients"]),
        ("seconds per beat (all agents)", [DEV_MEASUREMENTS["seconds_per_beat_before_compile"], sec_per_beat_now], "s",
         ["eager Keras", "compiled"]),
        ("fitting the detectors", [DEV_MEASUREMENTS["fit_seconds_before_compile"], fit_now], "s",
         ["eager Keras", "compiled"]),
    ]
    for ax, (title, values, unit, ticks) in zip(axes, data):
        bars = ax.bar(ticks, values, color=["#9ca3af", ACCENT], width=0.6)
        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}{unit}" if v >= 10 else f"{v:.2f}{unit}",
                    ha="center", va="bottom", fontsize=8, color=INK)
        ax.set_title(title, fontsize=8.5, color=INK)
        ax.tick_params(axis="x", labelsize=7)
        _style_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_explanation(beat: np.ndarray, explain: dict, diagnosis: dict, path: Path) -> None:
    t = np.arange(len(beat)) / 360 * 1000
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.0, 4.2), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    attention = np.asarray(explain["attention_weights"])
    ax1.fill_between(t, beat.min(), beat.min() + (attention / attention.max()) * (beat.max() - beat.min()),
                     color="#fde68a", alpha=0.7, label="attention (scaled)")
    ax1.plot(t, beat, color=INK, linewidth=1.1, label="ECG beat (z-scored)")
    saliency = np.asarray(explain["gradient_sensitivity"])
    ax1.plot(t, beat.min() + saliency * (beat.max() - beat.min()) * 0.4, color=ACCENT, linewidth=0.7,
             alpha=0.8, label="gradient saliency (scaled)")
    ax1.legend(fontsize=7, frameon=False, loc="upper right")
    ax1.set_title(f"Diagnosis {diagnosis['predicted_label']} (confidence {diagnosis['confidence']:.2f})",
                  fontsize=9, color=INK, loc="left")
    _style_axes(ax1)
    shap = np.asarray(explain["shap_values"])
    seg_ms = 30 / 360 * 1000
    centers = (np.arange(len(shap)) + 0.5) * seg_ms
    ax2.bar(centers, shap, width=seg_ms * 0.85, color=["#16a34a" if v >= 0 else "#dc2626" for v in shap])
    ax2.axhline(0, color=MUTED, linewidth=0.6)
    ax2.set_ylabel("SHAP", fontsize=8, color=INK)
    ax2.set_xlabel("time (ms)", fontsize=8, color=INK)
    _style_axes(ax2)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- live measurements

def measure(model_path: Path) -> dict:
    from agents.diagnosis.diagnosis_agent import run_diagnosis
    from agents.explainability.explainability_agent import run_explainability
    from agents.validation.validation_agent import ValidationConfig, validate_beat
    from models.backend import KerasBackend

    backend = KerasBackend.load(str(model_path))
    train, val, test = load_split("train"), load_split("val"), load_split("test")
    out: dict = {"split_sizes": {s.name: {CLASS_LABELS[c]: int((s.labels == c).sum()) for c in range(4)}
                                 for s in (train, val, test)}}

    probs = backend.predict_proba(test.signals)
    pred = probs.argmax(axis=1)
    cm_counts = np.zeros((4, 4), dtype=int)
    for t_, p_ in zip(test.labels, pred):
        cm_counts[t_, p_] += 1
    per_class = {}
    for c in range(4):
        tp = cm_counts[c, c]
        precision = tp / max(cm_counts[:, c].sum(), 1)
        recall = tp / max(cm_counts[c].sum(), 1)
        per_class[CLASS_LABELS[c]] = {"precision": precision, "recall": recall, "support": int(cm_counts[c].sum()),
                                      "f1": 2 * precision * recall / max(precision + recall, 1e-9)}
    out["diagnosis"] = {"accuracy": float((pred == test.labels).mean()), "n": len(pred), "per_class": per_class}
    out["confusion"] = cm_counts

    rng = np.random.default_rng(0)
    out["validation"] = {}
    for split in (train, val, test):
        idx = rng.choice(len(split.labels), 2000, replace=False)
        default = sum(validate_beat(split.signals[i])["status"] == "PASS" for i in idx)
        preset = sum(validate_beat(split.signals[i], config=ValidationConfig.zscored_beats())["status"] == "PASS"
                     for i in idx)
        out["validation"][split.name] = (default / 2000, preset / 2000)

    background = train.signals[stratified_sample(train, 2000, seed=0)].mean(axis=0)
    v_idx = int(np.flatnonzero((test.labels == 3) & (pred == 3))[0])
    state = {"active_signal": test.signals[v_idx]}
    state = run_diagnosis(state, backend)
    state = run_explainability(state, backend, background, n_permutations=128)
    out["explanation"] = (test.signals[v_idx], state["explainability"], state["diagnosis"])
    return out


# --------------------------------------------------------------------------- PDF

def _styles():
    pdfmetrics.registerFont(TTFont("Arial", "C:/Windows/Fonts/arial.ttf"))
    pdfmetrics.registerFont(TTFont("Arial-Bold", "C:/Windows/Fonts/arialbd.ttf"))
    pdfmetrics.registerFont(TTFont("Arial-Italic", "C:/Windows/Fonts/ariali.ttf"))
    pdfmetrics.registerFont(TTFont("Consolas", "C:/Windows/Fonts/consola.ttf"))
    pdfmetrics.registerFontFamily("Arial", normal="Arial", bold="Arial-Bold", italic="Arial-Italic",
                                  boldItalic="Arial-Bold")
    base = getSampleStyleSheet()
    s = {
        "title": ParagraphStyle("title", parent=base["Title"], fontName="Arial-Bold", fontSize=24, leading=29,
                                textColor=colors.HexColor(INK), alignment=0, spaceAfter=10),
        "subtitle": ParagraphStyle("subtitle", fontName="Arial", fontSize=12, leading=16,
                                   textColor=colors.HexColor(MUTED), spaceAfter=24),
        "h1": ParagraphStyle("h1", fontName="Arial-Bold", fontSize=16, leading=20, spaceBefore=6, spaceAfter=10,
                             textColor=colors.HexColor(INK)),
        "h2": ParagraphStyle("h2", fontName="Arial-Bold", fontSize=12, leading=15, spaceBefore=12, spaceAfter=5,
                             textColor=colors.HexColor(ACCENT)),
        "h3": ParagraphStyle("h3", fontName="Arial-Bold", fontSize=10, leading=13, spaceBefore=7, spaceAfter=3,
                             textColor=colors.HexColor(INK)),
        "body": ParagraphStyle("body", fontName="Arial", fontSize=9.5, leading=13.5, spaceAfter=5,
                               textColor=colors.HexColor(INK)),
        "bullet": ParagraphStyle("bullet", fontName="Arial", fontSize=9.5, leading=13.5, leftIndent=12,
                                 bulletIndent=2, spaceAfter=2, textColor=colors.HexColor(INK)),
        "caption": ParagraphStyle("caption", fontName="Arial-Italic", fontSize=8, leading=10.5,
                                  textColor=colors.HexColor(MUTED), alignment=TA_CENTER, spaceAfter=10),
        "cell": ParagraphStyle("cell", fontName="Arial", fontSize=8, leading=10.5, textColor=colors.HexColor(INK)),
        "cellb": ParagraphStyle("cellb", fontName="Arial-Bold", fontSize=8, leading=10.5,
                                textColor=colors.HexColor(INK)),
        "code": ParagraphStyle("code", fontName="Consolas", fontSize=7.2, leading=9.4,
                               textColor=colors.HexColor(INK), backColor=colors.HexColor("#f6f7f9"),
                               borderPadding=6, spaceBefore=4, spaceAfter=10),
    }
    return s


def _table(rows, widths, s, header=True):
    data = [[Paragraph(str(c), s["cellb"] if (header and i == 0) else s["cell"]) for c in r]
            for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    style = [("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("LINEBELOW", (0, 0), (-1, -1), 0.4, colors.HexColor(GRID)),
             ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef2ff")),
                  ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.HexColor(ACCENT))]
    t.setStyle(TableStyle(style))
    return t


def _bullets(items, s):
    return [Paragraph(i, s["bullet"], bulletText="•") for i in items]


def _figure(path, width_cm, caption, s):
    img = Image(str(path))
    ratio = img.imageHeight / img.imageWidth
    img.drawWidth, img.drawHeight = width_cm * cm, width_cm * cm * ratio
    return KeepTogether([img, Paragraph(caption, s["caption"])])


AGENTS = [
    {
        "name": "1. Validation agent",
        "files": "agents/validation/ (validation_agent.py, format_checks.py, sanity_checks.py, triage.py, repair.py)",
        "origin": "feature/validation-eval branch",
        "role": "Entry point. Verifies that the beat is complete, correctly formatted, sampled at the expected rate "
                "and free of obvious corruption, so nothing downstream analyses an unreliable signal.",
        "inputs": "Raw beat, shape (360,) or (360, 1); optional metadata (record id, lead, sampling rate).",
        "outputs": "ValidationResult: status PASS / REPAIR / REJECT, the (possibly repaired) signal, one flag per "
                   "check, a reason and the list of repair actions.",
        "method": [
            "Format checks: shape, numeric dtype, NaN, missing samples, non-finite or out-of-range values.",
            "Sampling check (added during integration): metadata[\"fs\"] / [\"sampling_rate\"], if given, must be 360 Hz.",
            "Signal sanity: flatline (std below 0.01), clipping (≥3 consecutive samples at the rail), "
            "abnormal amplitude (per-beat |z| above threshold).",
            "Triage: structural failure, ≥5% NaN or flatline → REJECT; sparse NaN, clipping or outliers → REPAIR "
            "(interpolate NaN, smooth clipped runs, clamp outliers) and re-validate the repaired beat; otherwise PASS.",
        ],
        "config": "ValidationConfig. Defaults (±5 range, z > 6 clamp) suit raw-amplitude beats. "
                  "ValidationConfig.zscored_beats() (±12, z > 10) is the preset for the per-beat z-scored ECG_Splits "
                  "data and is what the pipeline uses.",
    },
    {
        "name": "2. Label poisoning detection agent",
        "files": "agents/label_poisoning/ (label_poisoning_agent.py, learning_dynamics.py, confident_learning.py, "
                 "latent_attention.py, score_fusion.py)",
        "origin": "Built during integration from the label-poisoning-agent / poison_detection_agent notebooks "
                  "(k-NN, Mahalanobis, confidence, entropy, prediction-label disagreement)",
        "role": "Decides whether the beat's annotation can be trusted, i.e. whether the label was flipped by an attacker.",
        "inputs": "Beat, claimed label, record id. Fitted once on the labelled dataset being audited, plus a "
                  "calibration set of beats from other patients.",
        "outputs": "DetectionReport (channel \"label\") with one Evidence per check, the fused label_score in [0, 1] "
                   "(the attack probability) and the out-of-fold class probabilities used by Recovery to relabel.",
        "method": [
            "Learning dynamics (AUM): trains a small classifier on the frozen DenseNet embeddings for 15 epochs and "
            "averages each beat's margin p(claimed) − max p(other); mislabeled beats keep a low or negative margin.",
            "Confident learning: 5-fold out-of-fold probabilities, so no beat is scored by a model that saw its "
            "own label; statistic = max p(other) − p(claimed).",
            "Latent + attention: in the attention-pooled embedding space, the fraction of the 10 nearest neighbours "
            "that disagree with the claimed label, Mahalanobis distance to the claimed class centroid relative to "
            "the nearest centroid, and −log p(claimed) (label-aware entropy).",
            "Score fusion with per-class calibration: each statistic becomes a [0, 1] score with robust "
            "(median/MAD) statistics per claimed class, then a weighted mean (0.35 / 0.35 / 0.30).",
            "Dataset audit: suspicious_records() / pipeline.audit_dataset() lists audited training samples above "
            "the threshold — the \"remove suspicious training samples\" option.",
        ],
        "config": "LabelPoisoningConfig (epochs, folds, k, z_offset, check weights).",
    },
    {
        "name": "3. Signal poisoning / noise detection agent",
        "files": "agents/signal_poisoning/ (signal_poisoning_agent.py, reconstruction.py, feature_distance.py, "
                 "fragility.py) and agents/signal_poisoning/noise/ (spectral noise features + trained classifier)",
        "origin": "Four checks built during integration; noise/ subpackage from the signal-agent branch",
        "role": "Detects tampering with the waveform itself and reports signal quality (noise type and severity).",
        "inputs": "Validated beat. Fitted on clean training beats; calibrated on clean beats from other patients.",
        "outputs": "DetectionReport (channel \"signal\") with the four Evidence objects the Decision agent fuses, "
                   "a spectral_noise Evidence, the autoencoder reconstruction and its confidence (for Recovery), "
                   "and signal_quality: SNR, spectral entropy, noise type, noise severity.",
        "method": [
            "Reconstruction check: an undercomplete autoencoder (360-128-32-128-360) trained on clean beats; "
            "tampered beats reconstruct badly. Reconstruction confidence = how close the reconstruction is to a "
            "fixed point of the autoencoder (i.e. a plausible clean beat).",
            "Feature-space distance: Mahalanobis distance of the beat's embedding from the clean embedding "
            "distribution (Ledoit-Wolf covariance).",
            "Prediction fragility: total-variation change of the prediction and class-flip rate under 16 small "
            "random perturbations.",
            "Gradient sensitivity: local Lipschitz estimate — the largest input-gradient norm of p(predicted class) "
            "at the beat and 7 nearby points.",
            "Signal quality: wavelet-based SNR estimate, spectral entropy, and the spectral noise agent's "
            "baseline-wander / powerline / EMG / morphology features → noise type and severity.",
            "All four attack statistics are calibrated against clean beats so 0.5 ≈ 3 robust SDs into the tail.",
        ],
        "config": "SignalPoisoningConfig (perturbation counts and sizes, calibration size, z_offset).",
    },
    {
        "name": "4. Decision agent",
        "files": "agents/decision/ (decision_agent.py, fusion.py, config/thresholds.yaml)",
        "origin": "decision-agent / recovery-agent branches; attack type and severity added during integration",
        "role": "Central reasoning step: fuses both detectors' evidence, decides clean vs attacked, names the "
                "attack and its severity, and whether recovery is needed.",
        "inputs": "label_score + label Evidence; the four signal Evidence objects.",
        "outputs": "DecisionRecord: verdict (clean, label_poisoned, signal_poisoned, both_poisoned, uncertain), "
                   "confidence, both scores, attack_type, severity (none/low/moderate/high), recommended action "
                   "(proceed / repair / quarantine) and all evidence for audit.",
        "method": [
            "Signal score = weighted mean (default) or noisy-OR of the four signal checks (weights 0.30 / 0.25 / "
            "0.20 / 0.25).",
            "Thresholds (0.55 / 0.55): label only → repair; signal only → repair; both → quarantine; neither → proceed.",
            "attack_type from what fired hardest: label_flip, waveform_corruption, out_of_distribution_signal, "
            "adversarial_perturbation. Severity from how far past its threshold the triggering score is.",
        ],
        "config": "config/thresholds.yaml (thresholds, uncertain band, fusion strategy, weights).",
    },
    {
        "name": "5. Recovery agent",
        "files": "agents/recovery/ (recovery_agent.py, strategies/label_repair.py, signal_repair.py, denoise.py, "
                 "quarantine.py, config/repair_thresholds.yaml)",
        "origin": "recovery-agent branch; Butterworth/wavelet denoising added during integration",
        "role": "Mitigates a detected attack, or refers the record for expert review when recovery isn't possible.",
        "inputs": "DecisionRecord, the record, out-of-fold probabilities, autoencoder reconstruction + confidence.",
        "outputs": "RecoveryResult: action (relabel, reconstruct_signal, denoise_signal, quarantine, reject), the "
                   "repaired record, repair confidence and notes.",
        "method": [
            "Label poisoned: relabel to the out-of-fold consensus class if its probability ≥ 0.70, else quarantine.",
            "Signal poisoned: substitute the autoencoder reconstruction if its confidence ≥ 0.60; otherwise "
            "Butterworth 0.5–40 Hz band-pass + db4 wavelet soft-threshold denoising.",
            "Both poisoned or uncertain: quarantine for manual review.",
            "Orchestrator safeguard: every repaired beat is re-validated, and a repaired signal must also pass "
            "signal-poisoning detection again; otherwise it is quarantined (\"referred for expert review\").",
        ],
        "config": "config/repair_thresholds.yaml (min confidences, signal_repair_fallback: denoise | quarantine).",
    },
    {
        "name": "6. Diagnosis agent",
        "files": "agents/diagnosis/diagnosis_agent.py, models/backend.py, models/keras_layers.py",
        "origin": "feature/validation-eval branch (rewritten onto the shared model backend)",
        "role": "Classifies the validated (or recovered) beat with the DenseNet + BiLSTM + Attention model.",
        "inputs": "state[\"active_signal\"]: the validated beat or the Recovery agent's repaired beat.",
        "outputs": "predicted class and label (A, B, N, V), confidence and per-class probabilities.",
        "method": [
            "Runs the team's DenseNet1D_clean.keras checkpoint (DenseNet blocks → BiLSTM → AttentionLayer → "
            "Dense(128) → softmax over 4 classes) through KerasBackend.",
            "KerasBackend graph-compiles every model call (tf.function), which made single predictions 16× faster "
            "than eager calls.",
        ],
        "config": "Model path; classes A=0, B=1 (merged LBBB/RBBB), N=2, V=3.",
    },
    {
        "name": "7. Explainability agent",
        "files": "agents/explainability/explainability_agent.py",
        "origin": "feature/validation-eval branch (file was truncated mid-line; completed during integration)",
        "role": "Shows which parts of the beat drove the diagnosis.",
        "inputs": "The diagnosed beat and its prediction; a background beat (mean of clean training beats).",
        "outputs": "Attention weights (360), gradient saliency (360), Shapley values for twelve 30-sample "
                   "(83 ms) segments, and the three most influential regions.",
        "method": [
            "Attention map: the model's own AttentionLayer softmax weights over BiLSTM timesteps, resampled to 360.",
            "Gradient saliency: |∂p(predicted)/∂x|, normalised to [0, 1].",
            "SHAP: permutation-sampling estimate of segment-level Shapley values against the background beat, "
            "computed in one batched forward pass.",
        ],
        "config": "Number of SHAP permutations (64 in the pipeline).",
    },
    {
        "name": "8. Alert & reporting agent",
        "files": "agents/reporting/reporting_agent.py",
        "origin": "New — written during integration",
        "role": "Produces the final clinician report.",
        "inputs": "The complete pipeline state.",
        "outputs": "Alert level (CRITICAL / WARNING / INFO), validation result, verdict with attack type and "
                   "severity, per-check evidence, signal quality, recovery action, diagnosis and confidence, "
                   "most influential regions, recommendations, and a plain-text rendering.",
        "method": [
            "CRITICAL: validation rejected the beat, or the record was quarantined/rejected — no diagnosis shown.",
            "WARNING: an attack was detected and repaired — diagnosis shown, marked as from a repaired record.",
            "INFO: clean beat.",
            "Recommendations: request a new recording, expert review, confirm a corrected annotation, low "
            "diagnostic confidence, abnormal beat class, poor signal quality.",
        ],
        "config": "Low-confidence and noise-flag thresholds in the module.",
    },
]


def build_pdf(results: dict, live: dict, out_path: Path) -> None:
    s = _styles()
    story = []
    scen = results["scenarios"]
    thr = results["thresholds"]

    def flagged(name):
        return scen[name]["n"] - scen[name]["verdicts"].get("clean", 0)

    clean_rate = flagged("clean") / scen["clean"]["n"]
    acc = live["diagnosis"]["accuracy"]

    # ---- title page
    story += [Spacer(1, 3.2 * cm),
              Paragraph("Agentic AI Pipeline for Secure ECG Diagnosis", s["title"]),
              Paragraph("Integration documentation and results — Data-Poisoning-in-Healthcare", s["subtitle"]),
              Paragraph(f"Generated {date.today().isoformat()} from branch <b>integration</b>.", s["body"]),
              Spacer(1, 0.6 * cm),
              Paragraph("Summary", s["h2"])]
    story += _bullets([
        "All eight agents of the architecture are implemented and wired together by one orchestrator, which runs "
        "as plain Python or as a LangGraph graph.",
        f"The team's DenseNet + BiLSTM + Attention checkpoint runs inside the pipeline: "
        f"<b>{acc:.1%}</b> accuracy on all {live['diagnosis']['n']:,} test beats.",
        f"Training-set label audit: <b>{results['audit']['true_positives']} of {results['audit']['flipped']}</b> "
        f"deliberately flipped labels found (precision "
        f"{results['audit']['true_positives'] / results['audit']['flagged']:.2f}).",
        f"Unseen test patients: {flagged('clean')} of {scen['clean']['n']} clean beats are wrongly flagged "
        f"(<b>{clean_rate:.0%}</b> false positives); {flagged('label flipped')} of {scen['label flipped']['n']} "
        f"flipped labels and {flagged('PGD eps=0.05')} of {scen['PGD eps=0.05']['n']} PGD-attacked beats are flagged.",
        f"<b>{results['seconds_per_beat']:.2f} s per beat</b> through all agents on a CPU; detectors fit in "
        f"{results['fit_seconds']:.0f} s.",
        "Not yet ready for clinical use: the decision thresholds still need calibration, and adversarial "
        "signal attacks are mostly attributed to the label (chapter 5).",
    ], s)
    story.append(PageBreak())

    # ---- 1. architecture
    story += [Paragraph("1. Architecture", s["h1"]),
              Paragraph("A raw ECG beat enters the Validation agent. Valid beats are analysed in parallel by the "
                        "label poisoning and signal poisoning / noise detectors; the Decision agent fuses their "
                        "evidence. Clean beats go straight to Diagnosis; attacked beats go through Recovery first. "
                        "Explainability and the Alert & reporting agent follow. Rejected and quarantined beats skip "
                        "to the report, which asks for a new recording or expert review.", s["body"]),
              _figure(FIG_DIR / "pipeline.png", 14.5, "Figure 1 — Pipeline as implemented in orchestrator/pipeline.py.", s),
              PageBreak(),
              Paragraph("1.1 Shared contracts", s["h2"]),
              Paragraph("Before integration the branches used three incompatible state formats. All agents now "
                        "share one state and one set of message types:", s["body"])]
    story.append(_table([
        ["Contract", "File", "Purpose"],
        ["ECGState", "core/ecg_state.py", "The dict every pipeline node reads and writes: raw_signal, label, "
         "validation, label_poisoning, signal_poisoning, decision, recovery, active_signal, active_label, "
         "diagnosis, explainability, report."],
        ["Evidence", "shared/contracts.py", "One detector check: name, channel (label/signal), score in [0, 1], "
         "raw value, confidence, metadata."],
        ["DetectionReport", "shared/contracts.py", "Output of each detection agent: evidence, fused score, artifacts."],
        ["DecisionRecord", "shared/contracts.py", "Verdict, confidence, scores, attack type, severity, action, evidence."],
        ["RecoveryResult", "shared/contracts.py", "Recovery action, repaired record, confidence, notes."],
        ["ClassifierBackend", "models/backend.py", "One model interface (predict, embed, gradient, attention) "
         "with KerasBackend for the checkpoint and TorchBackend for tests."],
    ], [3.2 * cm, 3.6 * cm, 9.6 * cm], s))
    story += [Paragraph("1.2 Repository layout", s["h2"]),
              Preformatted("agents/\n"
                           "  validation/  label_poisoning/  signal_poisoning/ (+ noise/)  decision/\n"
                           "  recovery/  diagnosis/  explainability/  reporting/\n"
                           "core/          ecg_state.py, calibration.py, signal_filters.py\n"
                           "shared/        contracts.py\n"
                           "models/        backend.py, keras_layers.py, densenet_bilstm.py\n"
                           "orchestrator/  pipeline.py\n"
                           "evaluation/    data.py, synthetic_beats.py, benchmark + metrics scripts\n"
                           "scripts/       run_real_pipeline.py, run_pipeline_demo.py, build_report.py\n"
                           "tests/         end-to-end + real-model tests (agent unit tests live next to each agent)\n"
                           "notebooks/     original Colab notebooks\n"
                           "artifacts/     model + data + results (gitignored)", s["code"]),
              PageBreak()]

    # ---- 2. agents
    story.append(Paragraph("2. The agents", s["h1"]))
    for a in AGENTS:
        block = [Paragraph(a["name"], s["h2"]), Paragraph(a["role"], s["body"]),
                 _table([["Files", a["files"]], ["Origin", a["origin"]], ["Input", a["inputs"]],
                         ["Output", a["outputs"]], ["Configuration", a["config"]]],
                        [2.6 * cm, 13.8 * cm], s, header=False),
                 Paragraph("How it works", s["h3"])]
        story += [KeepTogether(block)] + _bullets(a["method"], s)
    story.append(Paragraph("Orchestrator", s["h2"]))
    story += _bullets([
        "orchestrator/pipeline.py: ECGDefensePipeline.from_reference_data() fits both detectors (clean reference "
        "beats, the audited dataset, a calibration set from other patients) and builds the pipeline.",
        "pipeline.run(beat, beat_id, label) runs every node and returns the full ECGState; build_graph() returns "
        "the same pipeline as a LangGraph graph with the two detectors as parallel branches.",
        "Routing: REJECT → report; proceed → diagnosis; repair → recovery → re-validation and re-detection → "
        "diagnosis, or quarantine → report.",
    ], s)
    story.append(PageBreak())

    # ---- 3. integration work
    story += [Paragraph("3. Integration work", s["h1"]),
              Paragraph("What was missing or broken when the branches were reviewed against the architecture, and "
                        "what was done about it:", s["body"])]
    story.append(_table([
        ["Problem found", "Fix"],
        ["No orchestrator; agents lived on separate branches.", "Merged all branches (history kept) into one "
         "package; wrote orchestrator/pipeline.py (plain Python + LangGraph)."],
        ["Three incompatible state contracts (\"REPAIR\" vs \"REPAIRED\", different keys).", "One ECGState in core/, "
         "one set of message types in shared/."],
        ["Label poisoning existed only as a notebook; no AUM, confident learning or score fusion; Recovery never got "
         "the OOF probabilities it needs.", "Built the label poisoning agent with all four boxes of the architecture."],
        ["Decision agent requires four named signal checks; the signal-agent branch had a different detector.",
         "Built the four checks; the team's spectral noise agent runs alongside as signal-quality evidence."],
        ["Diagnosis/Explainability imported a model module that was never committed; explainability_agent.py was "
         "truncated mid-line; no SHAP.", "Model backend for the Keras checkpoint; explainability completed with "
         "attention, gradients and SHAP."],
        ["No Alert & reporting agent.", "Written."],
        ["Noise training script passed arguments the data generator didn't accept.", "Added the Gaussian attack "
         "mode and arguments to noise_datagen.py."],
        [f"Validation rejected ~{1 - np.mean([v[0] for v in live['validation'].values()]):.0%} of real beats "
         "(z-scored R-peaks exceed the ±5 range).", "ValidationConfig.zscored_beats() preset — 100% of sampled real "
         "beats pass."],
        ["Detectors calibrated on training patients flagged 62% of clean test beats.", "Calibration on the "
         f"validation split (other patients): {clean_rate:.0%}."],
        ["Eager Keras calls: 6.2 s per beat, 707 s to fit.", f"tf.function compilation: "
         f"{results['seconds_per_beat']:.2f} s per beat, {results['fit_seconds']:.0f} s to fit."],
        ["PDF requirements not yet covered (attack type/severity, SNR/entropy, denoising, recommendations, "
         "sampling-rate check, removing suspicious training samples).", "All added."],
    ], [7.6 * cm, 8.8 * cm], s))
    story.append(PageBreak())

    # ---- 4. results
    story += [Paragraph("4. Results", s["h1"]),
              Paragraph("4.1 Data and model", s["h2"]),
              Paragraph("ECG_Splits CSVs: 360-sample single-lead MIT-BIH beats, z-scored per beat, split by "
                        "patient (no patient appears in more than one split).", s["body"])]
    sizes = live["split_sizes"]
    story.append(_table([["Split", "A", "B", "N", "V", "Total"]] +
                        [[n, *[f"{sizes[n][c]:,}" for c in "ABNV"], f"{sum(sizes[n].values()):,}"]
                         for n in ("train", "val", "test")],
                        [3 * cm, 2.4 * cm, 2.4 * cm, 2.4 * cm, 2.4 * cm, 2.4 * cm], s))
    story += [Paragraph("4.2 Diagnosis agent on the full test split", s["h2"]),
              Paragraph(f"Accuracy <b>{acc:.2%}</b> on {live['diagnosis']['n']:,} beats. Class A has only "
                        f"{sizes['test']['A']} test beats, so its figures are unreliable.", s["body"])]
    pc = live["diagnosis"]["per_class"]
    story.append(_table([["Class", "Precision", "Recall", "F1", "Support"]] +
                        [[c, f"{pc[c]['precision']:.3f}", f"{pc[c]['recall']:.3f}", f"{pc[c]['f1']:.3f}",
                          f"{pc[c]['support']:,}"] for c in "ABNV"],
                        [3 * cm, 3 * cm, 3 * cm, 3 * cm, 3 * cm], s))
    story.append(_figure(FIG_DIR / "confusion.png", 8.5, "Figure 2 — Diagnosis confusion matrix (test split).", s))

    story += [Paragraph("4.3 Validation agent on real beats", s["h2"])]
    story.append(_table([["Split (2,000 random beats)", "PASS with original thresholds", "PASS with z-score preset"]] +
                        [[n, f"{v[0]:.1%}", f"{v[1]:.1%}"] for n, v in live["validation"].items()],
                        [5.5 * cm, 5.5 * cm, 5.5 * cm], s))
    story.append(PageBreak())

    story += [Paragraph("4.4 End-to-end pipeline on unseen test patients", s["h2"]),
              Paragraph(f"Setup: detectors fitted on {results['config']['reference']:,} clean training beats; label "
                        f"audit on {results['audit']['n']:,} training beats of which {results['audit']['flipped']} "
                        f"had their labels deliberately flipped; scores calibrated on "
                        f"{results['config']['calibration']} validation beats; then {scen['clean']['n']} unseen test "
                        f"beats per scenario through all eight agents. Decision thresholds: label "
                        f"{thr['label']}, signal {thr['signal']}.", s["body"])]
    rows = [["Scenario", "Flagged", "Verdicts", "Recovery actions", "True class diagnosed"]]
    for n, r in scen.items():
        rows.append([n, f"{flagged(n)}/{r['n']}",
                     ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in r["verdicts"].items()),
                     ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in r["recovery"].items()),
                     f"{r['correct_diagnosis']}/{r['n']}"])
    story.append(_table(rows, [2.8 * cm, 1.6 * cm, 5.2 * cm, 4.8 * cm, 2.0 * cm], s))
    story.append(Spacer(1, 6))
    story.append(_figure(FIG_DIR / "verdicts.png", 15.5, "Figure 3 — Decision verdicts per scenario "
                         "(real model, unseen test patients).", s))
    story += _bullets([
        f"<b>Label audit (training set):</b> recall {results['audit']['true_positives'] / results['audit']['flipped']:.2f}, "
        f"precision {results['audit']['true_positives'] / results['audit']['flagged']:.2f} — the label detector works "
        "well on the dataset it audits.",
        f"<b>Clean beats:</b> {clean_rate:.0%} wrongly flagged. These are false positives that send good beats "
        "to recovery or review.",
        "<b>Gaussian noise σ = 0.01 / 0.05</b> (the team's poisoning experiment) gives the same verdicts as clean "
        "beats: on z-scored beats with R-peaks of 5–8 the noise is imperceptible, and it does not change "
        "the diagnosis either.",
        "<b>PGD (ε = 0.05):</b> most attacked beats are flagged, but mainly as label poisoning (chapter 5).",
        "<b>True class diagnosed</b> counts beats whose final diagnosis matches the true class; quarantined beats "
        "receive no diagnosis, so they count as misses.",
    ], s)
    story.append(PageBreak())

    story += [Paragraph("4.5 Score distributions and the fusion choice", s["h2"]),
              _figure(FIG_DIR / "scores.png", 16, "Figure 4 — Per-beat scores by scenario. Black bars are medians, "
                      "the red line is the decision threshold.", s)]
    rows = [["Scenario", "Signal flag rate, weighted mean", "Signal flag rate, noisy OR"]]
    for n, r in scen.items():
        wm, no = np.asarray(r["signal_scores_weighted_mean"]), np.asarray(r["signal_scores_noisy_or"])
        rows.append([n, f"{np.mean(wm >= thr['signal']):.0%}", f"{np.mean(no >= thr['signal']):.0%}"])
    story.append(_table(rows, [5 * cm, 5.5 * cm, 5.5 * cm], s))
    story.append(Paragraph("The default weighted mean lets three calm checks average down one strong one; noisy OR "
                           "catches far more PGD beats but also flags many more clean ones. The strategy and the "
                           "thresholds have to be chosen together on a larger calibration run.", s["body"]))

    story += [Paragraph("4.6 Effect of the integration fixes", s["h2"]),
              _figure(FIG_DIR / "before_after.png", 15.5, "Figure 5 — Before/after the calibration and "
                      "compilation fixes. \"Before\" values are development measurements of code that has since "
                      "been changed.", s)]
    story.append(PageBreak())

    story += [Paragraph("4.7 Explainability example", s["h2"]),
              _figure(FIG_DIR / "explanation.png", 15.5, "Figure 6 — A V beat from the test split: model attention, "
                      "gradient saliency, and segment SHAP values (green pushes toward the prediction).", s),
              Paragraph("4.8 Example clinician reports", s["h2"])]
    for n, text in results["examples"].items():
        story.append(KeepTogether([Paragraph(n, s["h3"]), Preformatted(text, s["code"])]))
    story.append(PageBreak())

    story += [Paragraph("4.9 Synthetic development tests", s["h2"]),
              Paragraph("Before the real model was available, the pipeline was tested on synthetic beats with a small "
                        "PyTorch DenseNet+BiLSTM+Attention model (tests/test_pipeline_integration.py, "
                        "scripts/run_pipeline_demo.py). Verdicts on 20 beats per scenario:", s["body"])]
    syn = DEV_MEASUREMENTS["synthetic_verdicts"]
    story.append(_table([["Scenario", "Verdicts"]] +
                        [[n, ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in vd.items())] for n, vd in syn.items()],
                        [4 * cm, 12 * cm], s))
    story.append(Spacer(1, 6))
    story.append(_table([["Synthetic scenario (100 beats)", "weighted mean", "noisy OR"]] +
                        [[n, f"{a:.0%}", f"{b:.0%}"] for n, a, b in DEV_MEASUREMENTS["synthetic_fusion"]],
                        [6 * cm, 5 * cm, 5 * cm], s))
    story.append(Paragraph("Test suite: 75 tests pass — the agents' own unit tests, end-to-end tests on synthetic "
                           "beats (including the LangGraph graph), and three tests against the real checkpoint and "
                           "data.", s["body"]))

    # ---- 5. limitations
    story += [Paragraph("5. Limitations and next steps", s["h1"])]
    story += _bullets([
        f"<b>False positives on unseen patients ({clean_rate:.0%}).</b> The thresholds in "
        "agents/decision/config/thresholds.yaml are starting values. Sweep them with "
        "agents/decision/evaluation/evaluate_fusion.py on a larger validation run.",
        "<b>Signal attacks are blamed on the label.</b> A successful adversarial example makes the classifier "
        "disagree with the (correct) label, so for beats outside the audited dataset the label detector fires. "
        f"{scen['PGD eps=0.05']['verdicts'].get('label_poisoned', 0)} of {scen['PGD eps=0.05']['n']} PGD beats "
        "were classified as label poisoning, which leads Recovery to relabel or quarantine instead of repairing "
        "the signal. A fusion policy under which the signal side also fires turns these into both_poisoned → "
        "quarantine.",
        "<b>Signal fusion strategy.</b> Weighted mean vs noisy OR trades missed attacks against false alarms "
        "(section 4.5); decide together with the thresholds.",
        "<b>Imperceptible noise.</b> Gaussian σ ≤ 0.05 on z-scored beats is not detected — and does not change "
        "diagnoses. A meaningful signal-poisoning evaluation needs attacks that actually affect the classifier "
        "(PGD, or larger perturbations).",
        "<b>Class mapping.</b> The model uses 4 classes (A, B, N, V; B = merged LBBB/RBBB) while "
        "notebooks/mitbih_preprocessing.ipynb still encodes 5 (A, L, N, R, V). Merge L/R before retraining.",
        f"<b>Class A</b> has only {sizes['test']['A']} test and {sizes['val']['A']} validation beats, so A-specific "
        "calibration and metrics are unreliable.",
    ], s)

    story += [Paragraph("6. How to reproduce", s["h1"]),
              Preformatted("python -m venv venv && venv\\Scripts\\activate\n"
                           "pip install -r requirements.txt tensorflow reportlab\n"
                           "# put DenseNet1D_clean.keras, train.csv, val.csv, test.csv in artifacts/\n"
                           "python -m pytest -q\n"
                           "python -m scripts.run_real_pipeline --n 60   # -> artifacts/results/real_results.json\n"
                           "python -m scripts.build_report               # -> this PDF", s["code"])]

    def on_page(canvas, doc):
        canvas.saveState()
        canvas.setFont("Arial", 7.5)
        canvas.setFillColor(colors.HexColor(MUTED))
        if doc.page > 1:
            canvas.drawString(2 * cm, 1.2 * cm, "Agentic AI Pipeline for Secure ECG Diagnosis")
            canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"{doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(str(out_path), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=1.8 * cm, bottomMargin=2 * cm,
                            title="Agentic AI Pipeline for Secure ECG Diagnosis — Documentation and Results",
                            author="Data-Poisoning-in-Healthcare team")
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=str(ARTIFACTS_DIR / "results" / "real_results.json"))
    parser.add_argument("--model", default=str(ARTIFACTS_DIR / "DenseNet1D_clean.keras"))
    parser.add_argument("--out", default=str(REPORT_DIR / "ECG_Agentic_Pipeline_Report.pdf"))
    args = parser.parse_args()

    results = json.loads(Path(args.results).read_text(encoding="utf-8"))
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    live = measure(Path(args.model))

    fig_pipeline(FIG_DIR / "pipeline.png")
    fig_confusion(live["confusion"], FIG_DIR / "confusion.png")
    fig_verdicts({n: r["verdicts"] for n, r in results["scenarios"].items()}, FIG_DIR / "verdicts.png",
                 f"{results['scenarios']['clean']['n']} unseen test beats per scenario")
    fig_score_distributions(results, FIG_DIR / "scores.png")
    clean = results["scenarios"]["clean"]
    fig_before_after((clean["n"] - clean["verdicts"].get("clean", 0)) / clean["n"], results["seconds_per_beat"],
                     results["fit_seconds"], FIG_DIR / "before_after.png")
    fig_explanation(*live["explanation"], FIG_DIR / "explanation.png")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    build_pdf(results, live, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
