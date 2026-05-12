#!/usr/bin/env python3
"""
train_dnabert2.py - Fine-tune DNABERT-2 for AMR gene detection.
Supports full fine-tuning and LoRA (--use_lora flag).
"""
import os, sys, json, time, logging, argparse, random
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from scipy.special import softmax
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score, precision_score, recall_score

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLITS = os.path.join(PROJECT_ROOT, "data", "splits")
MODELS = os.path.join(PROJECT_ROOT, "models")
RESULTS = os.path.join(PROJECT_ROOT, "results")
LOG_FILE = os.path.join(RESULTS, "pipeline.log")
os.makedirs(RESULTS, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"), logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

MODEL_NAME = "zhihan1996/DNABERT-2-117M"
MAX_LENGTH = 512


class AMRDataset(Dataset):
    """PyTorch dataset for AMR sequence classification."""
    def __init__(self, csv_path, tokenizer, max_length=512, max_samples=None):
        """Initialize dataset from CSV file.
        
        Args:
            csv_path: Path to CSV with 'sequence' and 'label' columns.
            tokenizer: HuggingFace tokenizer instance.
            max_length: Maximum token length for truncation/padding.
            max_samples: Optional limit on number of samples to use.
        """
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
        """Return tokenized sequence and label at index."""
        seq = str(self.sequences[idx])
        encoding = self.tokenizer(
            seq, max_length=self.max_length, padding="max_length",
            truncation=True, return_tensors="pt")
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


class WeightedTrainer:
    """Custom trainer with class-weighted loss for imbalanced data."""
    def __init__(self, model, train_dataset, val_dataset, args, compute_metrics_fn,
                 pos_weight=1.0, device="cpu"):
        """Initialize trainer.
        
        Args:
            model: The model to train.
            train_dataset: Training dataset.
            val_dataset: Validation dataset.
            args: Training arguments dict.
            compute_metrics_fn: Function to compute metrics.
            pos_weight: Weight for positive class.
            device: Device to train on.
        """
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.args = args
        self.compute_metrics_fn = compute_metrics_fn
        self.pos_weight = pos_weight
        self.device = device
        self.history = {"train_loss": [], "val_loss": [], "val_f1": [],
                       "val_mcc": [], "val_auroc": [], "epoch_time": []}
        self.best_f1 = 0.0

    def train(self):
        """Run the training loop."""
        from torch.utils.data import DataLoader
        pin_mem = torch.cuda.is_available()
        train_loader = DataLoader(self.train_dataset, batch_size=self.args["batch_size"],
                                  shuffle=True, num_workers=0, pin_memory=pin_mem)
        val_loader = DataLoader(self.val_dataset, batch_size=self.args["eval_batch_size"],
                                shuffle=False, num_workers=0, pin_memory=pin_mem)
        optimizer = torch.optim.AdamW(self.model.parameters(),
                                       lr=self.args["learning_rate"],
                                       weight_decay=self.args["weight_decay"])
        total_steps = len(train_loader) * self.args["num_epochs"]
        warmup_steps = int(total_steps * self.args["warmup_ratio"])
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=warmup_steps)
        loss_weights = torch.tensor([1.0, self.pos_weight], dtype=torch.float32).to(self.device)
        loss_fn = torch.nn.CrossEntropyLoss(weight=loss_weights)
        # Disable pin_memory on CPU
        pin_mem = torch.cuda.is_available()
        scaler = None
        use_amp = self.args.get("fp16", False) and torch.cuda.is_available()
        if use_amp:
            scaler = torch.amp.GradScaler("cuda")
        logger.info(f"Training: {self.args['num_epochs']} epochs, {len(train_loader)} batches/epoch")
        logger.info(f"Device: {self.device}, AMP: {use_amp}, Pos weight: {self.pos_weight:.2f}")

        for epoch in range(self.args["num_epochs"]):
            epoch_start = time.time()
            self.model.train()
            total_loss = 0.0
            n_batches = 0
            for batch in train_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                optimizer.zero_grad()
                if use_amp:
                    with torch.amp.autocast("cuda"):
                        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                        loss = loss_fn(outputs.logits, labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = loss_fn(outputs.logits, labels)
                    loss.backward()
                    optimizer.step()
                scheduler.step()
                total_loss += loss.item()
                n_batches += 1
            avg_train_loss = total_loss / max(n_batches, 1)

            # Validation
            val_metrics, val_loss = self._evaluate(val_loader, loss_fn, use_amp)
            epoch_time = time.time() - epoch_start
            self.history["train_loss"].append(round(avg_train_loss, 4))
            self.history["val_loss"].append(round(val_loss, 4))
            self.history["val_f1"].append(round(val_metrics["f1"], 4))
            self.history["val_mcc"].append(round(val_metrics["mcc"], 4))
            self.history["val_auroc"].append(round(val_metrics["auroc"], 4))
            self.history["epoch_time"].append(round(epoch_time, 1))

            logger.info(
                f"Epoch {epoch+1}/{self.args['num_epochs']} | "
                f"Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"F1: {val_metrics['f1']:.4f} | MCC: {val_metrics['mcc']:.4f} | "
                f"AUROC: {val_metrics['auroc']:.4f} | Time: {epoch_time/60:.1f}min")

            # Save best model
            if val_metrics["f1"] > self.best_f1:
                self.best_f1 = val_metrics["f1"]
                best_dir = os.path.join(MODELS, "dnabert2_amr_best")
                os.makedirs(best_dir, exist_ok=True)
                self.model.save_pretrained(best_dir)
                logger.info(f"  * New best model saved (F1={self.best_f1:.4f})")

        # Save final model
        final_dir = os.path.join(MODELS, "dnabert2_amr_final")
        os.makedirs(final_dir, exist_ok=True)
        self.model.save_pretrained(final_dir)
        logger.info(f"Final model saved to {final_dir}")
        return self.history

    def _evaluate(self, loader, loss_fn, use_amp):
        """Run evaluation on a dataloader."""
        self.model.eval()
        all_logits, all_labels = [], []
        total_loss = 0.0
        n_batches = 0
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                if use_amp:
                    with torch.amp.autocast("cuda"):
                        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                        loss = loss_fn(outputs.logits, labels)
                else:
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = loss_fn(outputs.logits, labels)
                total_loss += loss.item()
                n_batches += 1
                all_logits.append(outputs.logits.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        all_logits = np.concatenate(all_logits, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        preds = np.argmax(all_logits, axis=-1)
        probs = softmax(all_logits, axis=-1)[:, 1]
        metrics = {
            "f1": f1_score(all_labels, preds, average="binary", zero_division=0),
            "mcc": matthews_corrcoef(all_labels, preds),
            "precision": precision_score(all_labels, preds, zero_division=0),
            "recall": recall_score(all_labels, preds, zero_division=0),
        }
        try:
            metrics["auroc"] = roc_auc_score(all_labels, probs)
        except ValueError:
            metrics["auroc"] = 0.0
        return metrics, total_loss / max(n_batches, 1)


def main():
    """Fine-tune DNABERT-2 for AMR detection."""
    parser = argparse.ArgumentParser(description="Train DNABERT-2 for AMR detection")
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA for memory-efficient training")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--max_samples", type=int, default=None, help="Max training samples (for quick testing)")
    args = parser.parse_args()

    logger.info(f"\n{'='*60}\nDNABERT-2 Training — {datetime.now().isoformat()}\n{'='*60}")

    # Check data
    train_csv = os.path.join(SPLITS, "train.csv")
    dev_csv = os.path.join(SPLITS, "dev.csv")
    if not os.path.isfile(train_csv) or not os.path.isfile(dev_csv):
        logger.error("Train/dev CSVs not found. Run preprocess.py first.")
        sys.exit(1)

    train_df = pd.read_csv(train_csv)
    n_pos = (train_df["label"] == 1).sum()
    n_neg = (train_df["label"] == 0).sum()
    pos_weight = n_neg / max(n_pos, 1)
    logger.info(f"Training data: {len(train_df)} sequences ({n_pos} pos, {n_neg} neg)")
    logger.info(f"Positive class weight: {pos_weight:.2f}")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load tokenizer and model
    logger.info(f"Loading model: {MODEL_NAME}")
    
    # Add scripts dir to path for the loader
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from dnabert2_loader import load_dnabert2_classifier, load_dnabert2_base, DNABERT2ForClassification
    
    best_dir = os.path.join(MODELS, "dnabert2_amr_best")
    weights_path = os.path.join(best_dir, "pytorch_model.bin")
    
    if args.resume and os.path.isfile(weights_path):
        logger.info(f"Resuming from checkpoint: {best_dir}")
        model, tokenizer = load_dnabert2_classifier(weights_path)
    else:
        base_model, config, tokenizer = load_dnabert2_base()
        model = DNABERT2ForClassification(base_model, config.hidden_size, num_labels=2)
        logger.info(f"Model params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    # LoRA
    if args.use_lora:
        logger.info("Applying LoRA configuration...")
        try:
            from peft import LoraConfig, get_peft_model, TaskType
            lora_config = LoraConfig(
                task_type=TaskType.SEQ_CLS, r=16, lora_alpha=32,
                target_modules=["query", "value"], lora_dropout=0.1, bias="none")
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
        except ImportError:
            logger.warning("peft not installed. Training without LoRA.")

    # Datasets
    train_dataset = AMRDataset(train_csv, tokenizer, MAX_LENGTH, max_samples=args.max_samples)
    val_dataset = AMRDataset(dev_csv, tokenizer, MAX_LENGTH, max_samples=args.max_samples // 4 if args.max_samples else None)
    logger.info(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    # Save tokenizer
    for save_dir in [os.path.join(MODELS, "dnabert2_amr_best"),
                     os.path.join(MODELS, "dnabert2_amr_final")]:
        os.makedirs(save_dir, exist_ok=True)
        tokenizer.save_pretrained(save_dir)

    # Train
    training_args = {
        "learning_rate": args.lr, "num_epochs": args.epochs,
        "batch_size": args.batch_size, "eval_batch_size": 16,
        "warmup_ratio": 0.1, "weight_decay": 0.01, "fp16": True,
    }
    trainer = WeightedTrainer(
        model=model, train_dataset=train_dataset, val_dataset=val_dataset,
        args=training_args, compute_metrics_fn=None,
        pos_weight=pos_weight, device=device)
    history = trainer.train()

    # Save training history
    history_path = os.path.join(RESULTS, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"Training history saved to {history_path}")
    logger.info(f"Best validation F1: {trainer.best_f1:.4f}")
    logger.info("Next step: python scripts/evaluate.py")


if __name__ == "__main__":
    main()
