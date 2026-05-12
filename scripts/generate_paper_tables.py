#!/usr/bin/env python3
"""
generate_paper_tables.py - Generate LaTeX tables for the paper.
Tables: main results, fragment results, data summary.
"""
import os, sys, json, logging
from datetime import datetime
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(PROJECT_ROOT, "results")
TABLES = os.path.join(RESULTS, "tables")
SPLITS = os.path.join(PROJECT_ROOT, "data", "splits")
LOG_FILE = os.path.join(RESULTS, "pipeline.log")
os.makedirs(TABLES, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"), logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)


def load_all_results():
    """Load baseline and DNABERT-2 results into unified dict."""
    all_r = {}
    for fname, key_prefix in [("baseline_results.json", ""), ("dnabert2_results.json", "")]:
        fpath = os.path.join(RESULTS, fname)
        if os.path.isfile(fpath):
            with open(fpath) as f:
                data = json.load(f)
            for model, tests in data.items():
                all_r[model] = tests
    return all_r


def bold_best(values, fmt=".3f"):
    """Return list of formatted strings with best value bolded."""
    if not values or all(v == 0 for v in values):
        return [f"{v:{fmt}}" for v in values]
    best = max(values)
    return [f"\\textbf{{{v:{fmt}}}}" if v == best else f"{v:{fmt}}" for v in values]


def bold_best_speed(values, fmt=".1f"):
    """Bold the lowest (fastest) speed value."""
    if not values or all(v == 0 for v in values):
        return [f"{v:{fmt}}" for v in values]
    best = min(v for v in values if v > 0) if any(v > 0 for v in values) else 0
    return [f"\\textbf{{{v:{fmt}}}}" if v == best else f"{v:{fmt}}" for v in values]


def table1_main_results(all_r):
    """Generate Table 1: Main results on full-length sequences."""
    logger.info("Generating Table 1: Main results...")
    models = ["RGI", "DeepARG", "DNABERT-2"]
    metrics = ["F1", "MCC", "AUROC", "Precision", "Recall", "ms_per_seq"]
    databases = [("CARD", "test_card_full"), ("MEGARes", "test_megares_full"), ("SARG", "test_sarg_full")]
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Performance comparison on full-length ARG sequences across three databases.}",
        "\\label{tab:main_results}",
        "\\begin{tabular}{l cccccc}",
        "\\toprule",
        "Model & F1 & MCC & AUROC & Precision & Recall & Speed (ms/seq) \\\\",
    ]
    for db_name, test_key in databases:
        lines.append("\\midrule")
        lines.append(f"\\multicolumn{{7}}{{l}}{{\\textit{{{db_name} Test Set}}}} \\\\")
        for m in ["F1", "MCC", "AUROC", "Precision", "Recall"]:
            vals = [all_r.get(model, {}).get(test_key, {}).get(m, 0.0) for model in models]
            formatted = bold_best(vals)
            # Build row for this metric across models (transpose later)
        # Build rows per model
        for model in models:
            d = all_r.get(model, {}).get(test_key, {})
            vals = [d.get(m, 0.0) for m in metrics]
            # Format
            row_strs = []
            for i, (m, v) in enumerate(zip(metrics, vals)):
                if m == "ms_per_seq":
                    row_strs.append(f"{v:.1f}")
                else:
                    row_strs.append(f"{v:.3f}")
            lines.append(f"{model} & {' & '.join(row_strs)} \\\\")
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])
    tex = "\n".join(lines)
    path = os.path.join(TABLES, "table1_main_results.tex")
    with open(path, "w", encoding="utf-8") as f:
        f.write(tex)
    logger.info(f"  Saved: {path}")


def table2_fragment_results(all_r):
    """Generate Table 2: Performance on 150bp fragments."""
    logger.info("Generating Table 2: Fragment results...")
    models = ["RGI", "DeepARG", "DNABERT-2"]
    metrics = ["F1", "MCC", "AUROC", "Precision", "Recall", "ms_per_seq"]
    databases = [("CARD", "test_card_150bp"), ("MEGARes", "test_megares_150bp"), ("SARG", "test_sarg_150bp")]
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Performance comparison on 150bp short-read fragments.}",
        "\\label{tab:fragment_results}",
        "\\begin{tabular}{l cccccc}",
        "\\toprule",
        "Model & F1 & MCC & AUROC & Precision & Recall & Speed (ms/seq) \\\\",
    ]
    for db_name, test_key in databases:
        lines.append("\\midrule")
        lines.append(f"\\multicolumn{{7}}{{l}}{{\\textit{{{db_name} — 150bp Fragments}}}} \\\\")
        for model in models:
            d = all_r.get(model, {}).get(test_key, {})
            vals = [d.get(m, 0.0) for m in metrics]
            row_strs = []
            for m, v in zip(metrics, vals):
                row_strs.append(f"{v:.1f}" if m == "ms_per_seq" else f"{v:.3f}")
            lines.append(f"{model} & {' & '.join(row_strs)} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    path = os.path.join(TABLES, "table2_fragment_results.tex")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"  Saved: {path}")


def table3_data_summary():
    """Generate Table 3: Dataset statistics."""
    logger.info("Generating Table 3: Data summary...")
    splits_info = []
    for name, label in [("train", "Train"), ("dev", "Dev"), ("test_card", "Test-CARD"),
                         ("test_megares", "Test-MEGARes"), ("test_sarg", "Test-SARG")]:
        fpath = os.path.join(SPLITS, f"{name}.csv")
        if os.path.isfile(fpath):
            df = pd.read_csv(fpath)
            n_pos = int((df["label"] == 1).sum())
            n_neg = int((df["label"] == 0).sum())
            n_classes = df["drug_class"].nunique() if "drug_class" in df.columns else 0
            avg_len = int(df["sequence"].str.len().mean()) if "sequence" in df.columns else 0
            source = df["source_db"].iloc[0] if "source_db" in df.columns else "mixed"
            splits_info.append((label, len(df), n_pos, n_neg, n_classes, avg_len, source))
        else:
            splits_info.append((label, 0, 0, 0, 0, 0, "N/A"))
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Dataset statistics after preprocessing and leakage removal.}",
        "\\label{tab:data_summary}",
        "\\begin{tabular}{l rrrrrr}",
        "\\toprule",
        "Split & Total & Positive & Negative & Drug Classes & Avg Length (bp) & Source \\\\",
        "\\midrule",
    ]
    for label, total, pos, neg, nc, al, src in splits_info:
        lines.append(f"{label} & {total:,} & {pos:,} & {neg:,} & {nc} & {al} & {src} \\\\")
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\vspace{2mm}",
        "\\footnotesize{Sequences with $>$90\\% k-mer overlap to training data were removed from external test sets.}",
        "\\end{table}",
    ])
    path = os.path.join(TABLES, "table3_data_summary.tex")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"  Saved: {path}")


def main():
    """Generate all LaTeX tables."""
    logger.info(f"\n{'='*60}\nTable Generation — {datetime.now().isoformat()}\n{'='*60}")
    all_r = load_all_results()
    table1_main_results(all_r)
    table2_fragment_results(all_r)
    table3_data_summary()
    logger.info(f"\nAll tables saved to {TABLES}")
    logger.info("Pipeline complete!")


if __name__ == "__main__":
    main()
