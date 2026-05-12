#!/usr/bin/env python3
"""
evaluate.py - Evaluate DNABERT-2 on all test sets (full + fragments).
"""
import os, sys, json, time, logging
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.special import softmax
from sklearn.metrics import (f1_score, matthews_corrcoef, roc_auc_score,
    precision_score, recall_score, confusion_matrix)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)
torch.manual_seed(42)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLITS = os.path.join(PROJECT_ROOT, "data", "splits")
MODELS = os.path.join(PROJECT_ROOT, "models")
RESULTS = os.path.join(PROJECT_ROOT, "results")
FIGURES = os.path.join(RESULTS, "figures")
LOG_FILE = os.path.join(RESULTS, "pipeline.log")
os.makedirs(FIGURES, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"), logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)


class AMRDataset(Dataset):
    """Dataset for evaluation."""
    def __init__(self, csv_path, tokenizer, max_length=512, max_samples=None):
        """Load sequences and labels from CSV."""
        self.df = pd.read_csv(csv_path)
        if max_samples and max_samples < len(self.df):
            self.df = self.df.sample(n=max_samples, random_state=42).reset_index(drop=True)
        self.sequences = self.df["sequence"].tolist()
        self.labels = self.df["label"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        """Return dataset size."""
        return len(self.sequences)

    def __getitem__(self, idx):
        """Return tokenized item."""
        encoding = self.tokenizer(
            str(self.sequences[idx]), max_length=self.max_length,
            padding="max_length", truncation=True, return_tensors="pt")
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def evaluate_test_set(model, tokenizer, csv_path, device, test_name, max_samples=None):
    """Evaluate model on a single test set and return metrics + predictions."""
    dataset = AMRDataset(csv_path, tokenizer, max_length=512, max_samples=max_samples)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
    model.eval()
    all_logits, all_labels = [], []
    start_time = time.time()
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            if torch.cuda.is_available():
                with torch.amp.autocast("cuda"):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            all_logits.append(outputs.logits.cpu().numpy())
            all_labels.append(batch["labels"].numpy())
    elapsed = time.time() - start_time
    all_logits = np.concatenate(all_logits)
    all_labels = np.concatenate(all_labels)
    preds = np.argmax(all_logits, axis=-1)
    probs = softmax(all_logits, axis=-1)[:, 1]
    ms_per_seq = (elapsed * 1000) / max(len(dataset), 1)
    metrics = {
        "F1": round(float(f1_score(all_labels, preds, average="binary", zero_division=0)), 4),
        "MCC": round(float(matthews_corrcoef(all_labels, preds)), 4),
        "Precision": round(float(precision_score(all_labels, preds, zero_division=0)), 4),
        "Recall": round(float(recall_score(all_labels, preds, zero_division=0)), 4),
        "ms_per_seq": round(ms_per_seq, 1),
    }
    try:
        metrics["AUROC"] = round(float(roc_auc_score(all_labels, probs)), 4)
    except ValueError:
        metrics["AUROC"] = 0.0
    # Save confusion matrix
    cm = confusion_matrix(all_labels, preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Not Resistant", "Resistant"],
                yticklabels=["Not Resistant", "Resistant"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix: {test_name}")
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES, f"cm_{test_name}.png"), dpi=200)
    plt.close(fig)
    return metrics, all_labels, probs


def main():
    """Evaluate DNABERT-2 on all test sets."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples per test set")
    args = parser.parse_args()
    
    logger.info(f"\n{'='*60}\nEvaluation -- {datetime.now().isoformat()}\n{'='*60}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    # Load model
    best_dir = os.path.join(MODELS, "dnabert2_amr_best")
    if not os.path.isdir(best_dir) or not os.listdir(best_dir):
        logger.error(f"Best model not found at {best_dir}. Run train_dnabert2.py first.")
        sys.exit(1)
    # Add scripts dir to path for the loader
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from dnabert2_loader import load_dnabert2_classifier
    
    MODEL_NAME = "zhihan1996/DNABERT-2-117M"
    logger.info(f"Loading model from {best_dir}")
    weights_path = os.path.join(best_dir, "pytorch_model.bin")
    if not os.path.isfile(weights_path):
        logger.error(f"pytorch_model.bin not found in {best_dir}")
        sys.exit(1)
    model, tokenizer = load_dnabert2_classifier(weights_path)
    logger.info("Loaded best model weights")
    model = model.to(device)
    # Define test sets
    test_sets = []
    for db in ["card", "megares", "sarg"]:
        for sfx in ["", "_150bp", "_300bp", "_500bp"]:
            csv_name = f"test_{db}{sfx}.csv"
            csv_path = os.path.join(SPLITS, csv_name)
            if os.path.isfile(csv_path):
                label = f"test_{db}{'_full' if not sfx else sfx}"
                test_sets.append((csv_path, label))
    if not test_sets:
        logger.error("No test CSVs found. Run preprocess.py first.")
        sys.exit(1)

    results = {"DNABERT-2": {}}
    roc_data = {}
    # Print header
    logger.info(f"\n{'Test Set':<25s} | {'F1':>6s} | {'MCC':>6s} | {'AUROC':>6s} | {'ms/seq':>6s}")
    logger.info("-" * 65)
    for csv_path, label in test_sets:
        metrics, labels, probs = evaluate_test_set(model, tokenizer, csv_path, device, label, max_samples=args.max_samples)
        results["DNABERT-2"][label] = metrics
        roc_data[label] = {"labels": labels.tolist(), "probs": probs.tolist()}
        logger.info(f"{label:<25s} | {metrics['F1']:6.4f} | {metrics['MCC']:6.4f} | "
                    f"{metrics['AUROC']:6.4f} | {metrics['ms_per_seq']:6.1f}")
    # Save results
    out_path = os.path.join(RESULTS, "dnabert2_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    roc_path = os.path.join(RESULTS, "roc_data.json")
    with open(roc_path, "w") as f:
        json.dump(roc_data, f)
    logger.info(f"\nResults saved to {out_path}")
    logger.info("Next step: python scripts/generate_paper_figures.py")


if __name__ == "__main__":
    main()
