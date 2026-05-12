# Results Interpretation Guide

How to read each output file and what to write in the paper.

---

## results/dnabert2_results.json & baseline_results.json

**What it contains**: F1, MCC, AUROC, Precision, Recall, and speed (ms/seq) for every model on every test set.

**Good results**:
- F1 > 0.90 on CARD test = strong in-distribution performance
- F1 > 0.80 on MEGARes/SARG = good cross-database generalization
- MCC > 0.80 = reliable even with class imbalance

**What to write**:
> "DNABERT-2 achieved an F1 score of X.XX on the CARD test set, outperforming RGI (F1=X.XX) and DeepARG (F1=X.XX). On the cross-database MEGARes test set, DNABERT-2 retained Y% of its in-distribution performance, compared to Z% for alignment-based methods."

---

## results/figures/fig1_cross_database_generalization.png

**Key finding**: If DNABERT-2's bars drop less steeply from CARD→MEGARes→SARG than baselines, this is your main contribution.

**Write**:
> "DNABERT-2 retained X% of its F1 score when evaluated on MEGARes, compared to Y% retention for RGI, demonstrating superior cross-database generalization (Figure 1)."

---

## results/figures/fig2_short_fragment_performance.png

**Key finding**: How well each model handles short reads (150bp is typical Illumina).

**Good result**: DNABERT-2 line stays above baselines as fragment size decreases.

**Write**:
> "At 150bp fragment length simulating short metagenomic reads, DNABERT-2 maintained F1=X.XX while RGI's F1 dropped to X.XX, demonstrating robustness to partial gene sequences (Figure 2)."

---

## results/figures/fig3_roc_curves.png

**Key finding**: AUC values across databases. Higher AUC = better discrimination.

**Write**:
> "DNABERT-2 achieved AUC values of X.XX, X.XX, and X.XX on CARD, MEGARes, and SARG respectively (Figure 3)."

---

## results/figures/fig4_training_curves.png

**What to check**: No overfitting (val loss should not increase sharply while train loss decreases). F1 should plateau.

---

## results/figures/fig5_confusion_matrices.png

**Key finding**: False positive vs false negative rates. For AMR detection, false negatives (missing real resistance) are worse than false positives.

---

## results/DATA_LEAKAGE_REPORT.md

**Copy directly into Methods section**. This demonstrates rigorous evaluation methodology.

---

## results/tables/table1_main_results.tex

Paste directly into LaTeX. Requires `\usepackage{booktabs}`.

---

## Interpreting MCC (Matthews Correlation Coefficient)

MCC is preferred over F1 for imbalanced datasets:
- MCC > 0.8 = excellent
- MCC 0.6-0.8 = good
- MCC 0.4-0.6 = moderate
- MCC < 0.4 = poor

Report MCC alongside F1 in all tables for reviewers who prefer it.
