#!/usr/bin/env python3
"""
generate_paper_figures.py - Generate publication-quality figures (300 DPI).
Figures: cross-database bar chart, fragment curves, ROC, training, confusion matrices.
"""
import os, sys, json, logging
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(PROJECT_ROOT, "results")
FIGURES = os.path.join(RESULTS, "figures")
LOG_FILE = os.path.join(RESULTS, "pipeline.log")
os.makedirs(FIGURES, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"), logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

# Style
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 11, "axes.linewidth": 0.8, "figure.dpi": 300,
    "savefig.dpi": 300, "savefig.bbox": "tight",
})
COLORS = {"RGI": "#4C72B0", "DeepARG": "#DD8452", "DNABERT-2": "#55A868"}


def load_results():
    """Load baseline and DNABERT-2 results."""
    baseline_path = os.path.join(RESULTS, "baseline_results.json")
    dnabert_path = os.path.join(RESULTS, "dnabert2_results.json")
    baseline, dnabert = {}, {}
    if os.path.isfile(baseline_path):
        with open(baseline_path) as f:
            baseline = json.load(f)
    if os.path.isfile(dnabert_path):
        with open(dnabert_path) as f:
            dnabert = json.load(f)
    return baseline, dnabert


def get_metric(results, model, test_key, metric="F1"):
    """Extract a metric value from results, handling missing data."""
    if model in results and test_key in results[model]:
        return results[model][test_key].get(metric, 0.0)
    return 0.0


def fig1_cross_database(all_results):
    """Figure 1: Cross-database generalization bar chart."""
    logger.info("Generating Figure 1: Cross-database generalization...")
    fig, ax = plt.subplots(figsize=(7, 5))
    databases = ["CARD", "MEGARes", "SARG"]
    test_keys = ["test_card_full", "test_megares_full", "test_sarg_full"]
    models = ["RGI", "DeepARG", "DNABERT-2"]
    x = np.arange(len(databases))
    width = 0.22
    for i, model in enumerate(models):
        vals = [get_metric(all_results, model, tk) for tk in test_keys]
        bars = ax.bar(x + i * width, vals, width, label=model,
                     color=COLORS[model], edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("F1 Score")
    ax.set_title("Cross-Database Generalization Performance")
    ax.set_xticks(x + width)
    ax.set_xticklabels(databases)
    ax.set_ylim(0, 1.15)
    ax.legend(frameon=True, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    path = os.path.join(FIGURES, "fig1_cross_database_generalization.png")
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"  Saved: {path}")


def fig2_fragment_performance(all_results):
    """Figure 2: Short fragment performance curves."""
    logger.info("Generating Figure 2: Fragment performance...")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    databases = ["card", "megares", "sarg"]
    titles = ["CARD", "MEGARes", "SARG"]
    frag_sizes = [150, 300, 500]
    x_labels = ["150bp", "300bp", "500bp", "Full"]
    models = ["RGI", "DeepARG", "DNABERT-2"]
    for ax, db, title in zip(axes, databases, titles):
        for model in models:
            vals = []
            for fs in frag_sizes:
                vals.append(get_metric(all_results, model, f"test_{db}_{fs}bp"))
            vals.append(get_metric(all_results, model, f"test_{db}_full"))
            ax.plot(range(4), vals, "o-", label=model, color=COLORS[model],
                   linewidth=2, markersize=6)
        ax.set_title(title, fontweight="bold")
        ax.set_xticks(range(4))
        ax.set_xticklabels(x_labels)
        ax.set_ylim(0, 1.05)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("F1 Score")
    axes[1].set_xlabel("Fragment Length")
    axes[2].legend(frameon=True, framealpha=0.9, loc="lower right")
    plt.tight_layout()
    path = os.path.join(FIGURES, "fig2_short_fragment_performance.png")
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"  Saved: {path}")


def fig3_roc_curves():
    """Figure 3: ROC curves for full-length sequences."""
    logger.info("Generating Figure 3: ROC curves...")
    roc_path = os.path.join(RESULTS, "roc_data.json")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    databases = ["card", "megares", "sarg"]
    titles = ["CARD", "MEGARes", "SARG"]
    if os.path.isfile(roc_path):
        with open(roc_path) as f:
            roc_data = json.load(f)
        for ax, db, title in zip(axes, databases, titles):
            key = f"test_{db}_full"
            if key in roc_data:
                labels = np.array(roc_data[key]["labels"])
                probs = np.array(roc_data[key]["probs"])
                fpr, tpr, _ = roc_curve(labels, probs)
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, color=COLORS["DNABERT-2"], linewidth=2,
                       label=f"DNABERT-2 (AUC={roc_auc:.3f})")
            ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=1)
            ax.set_title(title, fontweight="bold")
            ax.set_xlabel("False Positive Rate")
            ax.legend(loc="lower right", frameon=True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
    else:
        for ax, title in zip(axes, titles):
            ax.text(0.5, 0.5, "No ROC data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)
    axes[0].set_ylabel("True Positive Rate")
    plt.tight_layout()
    path = os.path.join(FIGURES, "fig3_roc_curves.png")
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"  Saved: {path}")


def fig4_training_curves():
    """Figure 4: Training loss and F1 curves."""
    logger.info("Generating Figure 4: Training curves...")
    hist_path = os.path.join(RESULTS, "training_history.json")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.5))
    if os.path.isfile(hist_path):
        with open(hist_path) as f:
            h = json.load(f)
        epochs = range(1, len(h.get("train_loss", [])) + 1)
        if h.get("train_loss"):
            ax1.plot(epochs, h["train_loss"], "o-", color="#E74C3C", label="Train", linewidth=2)
            ax1.plot(epochs, h["val_loss"], "o-", color="#3498DB", label="Val", linewidth=2)
        if h.get("val_f1"):
            ax2.plot(epochs, h["val_f1"], "o-", color="#55A868", label="Val F1", linewidth=2)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.set_title("Loss")
    ax1.legend(); ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("F1 Score"); ax2.set_title("Validation F1")
    ax2.legend(); ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
    plt.tight_layout()
    path = os.path.join(FIGURES, "fig4_training_curves.png")
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"  Saved: {path}")


def fig5_confusion_matrices():
    """Figure 5: Normalized confusion matrices for DNABERT-2."""
    logger.info("Generating Figure 5: Confusion matrices...")
    roc_path = os.path.join(RESULTS, "roc_data.json")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    databases = ["card", "megares", "sarg"]
    titles = ["CARD", "MEGARes", "SARG"]
    if os.path.isfile(roc_path):
        with open(roc_path) as f:
            roc_data = json.load(f)
        for ax, db, title in zip(axes, databases, titles):
            key = f"test_{db}_full"
            if key in roc_data:
                labels = np.array(roc_data[key]["labels"])
                probs = np.array(roc_data[key]["probs"])
                preds = (probs >= 0.5).astype(int)
                from sklearn.metrics import confusion_matrix as cm_func
                cm = cm_func(labels, preds, normalize="true")
                sns.heatmap(cm, annot=True, fmt=".2%", cmap="Blues", ax=ax,
                           xticklabels=["Not Res.", "Resistant"],
                           yticklabels=["Not Res.", "Resistant"], cbar=False)
                ax.set_title(title, fontweight="bold")
            else:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(title)
    axes[0].set_ylabel("Actual")
    fig.supxlabel("Predicted", y=0.02)
    plt.tight_layout()
    path = os.path.join(FIGURES, "fig5_confusion_matrices.png")
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"  Saved: {path}")


def main():
    """Generate all publication figures."""
    logger.info(f"\n{'='*60}\nFigure Generation — {datetime.now().isoformat()}\n{'='*60}")
    baseline, dnabert = load_results()
    all_results = {}
    for model_key in ["RGI", "DeepARG"]:
        if model_key in baseline:
            all_results[model_key] = baseline[model_key]
    if "DNABERT-2" in dnabert:
        all_results["DNABERT-2"] = dnabert["DNABERT-2"]
    fig1_cross_database(all_results)
    fig2_fragment_performance(all_results)
    fig3_roc_curves()
    fig4_training_curves()
    fig5_confusion_matrices()
    logger.info(f"\nAll figures saved to {FIGURES}")
    logger.info("Next step: python scripts/generate_paper_tables.py")


if __name__ == "__main__":
    main()
