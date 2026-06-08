#!/usr/bin/env python3
"""
train_dnabert2.py  –  Fine-tune DNABERT-2 for AMR gene detection.
==================================================================

Pipeline (mirrors the 5-step description):

  STEP 1 – TOKENIZATION (BPE)
    Raw DNA string  →  BPE subword tokens  →  integer IDs + attention mask.
    The tokenizer reduces ~1000 bp sequences to ~200 tokens (≈5x compression).

  STEP 2 – FORWARD PASS  (12 transformer layers, ALiBi positional encoding)
    token IDs  →  768-dim embeddings  →  12 self-attention layers
    →  contextual hidden states  →  [CLS] token representation (768-dim vector).

  STEP 3 – CLASSIFICATION HEAD
    [CLS] vector  →  Dropout(0.1)  →  Linear(768 → 2)  →  logits
    logits represent P(Non-Resistant) and P(Resistant).

  STEP 4 – BACKPROPAGATION & OPTIMISATION
    Loss  : Weighted Cross-Entropy (class weights compensate for CARD imbalance)
    Optim : AdamW  (lr=2e-5, weight_decay=0.01)
    Sched : Linear warm-up for 10% of steps → linear decay to 0
    Epochs: 3–5 (default 3 to preserve pre-trained knowledge)
    Grad clip: max_norm=1.0 (prevents exploding gradients)
    AMP   : FP16 mixed-precision on CUDA (2× speed, half the VRAM)

  STEP 5 – LoRA (memory optimisation)
    Freezes all 117M BERT weights.
    Injects trainable rank-16 adapters into every query & value matrix.
    Trainable params: ~1.3M  (vs 117M for full fine-tuning) – ~60% VRAM savings.

Usage:
    python scripts/train_dnabert2.py                # full fine-tuning (3 epochs)
    python scripts/train_dnabert2.py --use_lora     # LoRA (consumer GPU)
    python scripts/train_dnabert2.py --epochs 5 --lr 2e-5 --batch_size 16
    python scripts/train_dnabert2.py --resume       # resume from best checkpoint
    python scripts/train_dnabert2.py --max_samples 500  # quick smoke-test
"""

import os, sys, json, time, logging, argparse, random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.special import softmax
from sklearn.metrics import (
    f1_score, matthews_corrcoef, roc_auc_score,
    precision_score, recall_score,
)

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLITS   = PROJECT_ROOT / "data"   / "splits"
MODELS   = PROJECT_ROOT / "models"
RESULTS  = PROJECT_ROOT / "results"
LOG_FILE = RESULTS / "pipeline.log"
RESULTS.mkdir(exist_ok=True)

# Add scripts dir to path (for dnabert2_loader)
SCRIPTS = PROJECT_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

MODEL_NAME = "zhihan1996/DNABERT-2-117M"
MAX_LENGTH = 512   # BPE tokens; ~2500 bp raw DNA after 5x compression


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 – DATASET / TOKENIZATION
# ══════════════════════════════════════════════════════════════════════════════

class AMRDataset(Dataset):
    """Wraps a CSV (columns: 'sequence', 'label') and tokenizes on-the-fly.

    Tokenization uses DNABERT-2's BPE tokenizer:
      - Variable-length DNA subwords  (up to MAX_LENGTH tokens)
      - Padding to MAX_LENGTH with [PAD] tokens
      - Truncation at MAX_LENGTH for very long genes
    """

    def __init__(self, csv_path: str, tokenizer, max_length: int = MAX_LENGTH,
                 max_samples: int = None):
        self.df = pd.read_csv(csv_path)
        if max_samples and max_samples < len(self.df):
            self.df = self.df.sample(n=max_samples, random_state=SEED).reset_index(drop=True)
        self.sequences  = self.df["sequence"].tolist()
        self.labels     = self.df["label"].tolist()
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = str(self.sequences[idx])
        # BPE tokenization: returns input_ids + attention_mask
        enc = self.tokenizer(
            seq,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),       # (MAX_LENGTH,)
            "attention_mask": enc["attention_mask"].squeeze(0),   # (MAX_LENGTH,)
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 + 3 – MODEL: DNABERT-2 ENCODER + CLASSIFICATION HEAD
# ══════════════════════════════════════════════════════════════════════════════

class DNABERT2Classifier(nn.Module):
    """DNABERT-2 (12 ALiBi transformer layers) + binary classification head.

    Architecture:
      Input IDs
        → BPE Embeddings (768-dim)
          → 12x Self-Attention (ALiBi positional bias – handles variable length)
            → [CLS] hidden state (768-dim, represents the whole sequence)
              → Dropout(p=0.1)
                → Linear(768 → num_labels)  ← the "binary decision switch"
                  → logits  [P(Non-Resistant), P(Resistant)]
    """

    def __init__(self, base_model, hidden_size: int, num_labels: int = 2):
        super().__init__()
        self.bert       = base_model
        self.dropout    = nn.Dropout(p=0.1)
        self.classifier = nn.Linear(hidden_size, num_labels)

        # Xavier initialisation for the new classification head
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        """
        Returns a simple namespace with .logits for compatibility with trainer.

        STEP 2 – Forward pass:
          All 12 transformer layers run internally inside self.bert().
        STEP 3 – Classification head:
          We take only the [CLS] position (index 0) of the final hidden state.
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Unpack: DNABERT-2 returns (hidden_states,) tuple or ModelOutput
        if isinstance(outputs, tuple):
            hidden_states = outputs[0]               # (B, seq_len, 768)
        elif hasattr(outputs, "last_hidden_state"):
            hidden_states = outputs.last_hidden_state
        else:
            hidden_states = outputs[0]

        cls_vec = hidden_states[:, 0, :]             # [CLS] token → (B, 768)
        cls_vec = self.dropout(cls_vec)
        logits  = self.classifier(cls_vec)           # (B, 2)

        return type("Out", (), {"logits": logits})()

    def save_pretrained(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 – BACKPROPAGATION & OPTIMISATION
# ══════════════════════════════════════════════════════════════════════════════

class AMRTrainer:
    """Training loop implementing the described backpropagation scheme.

    Each epoch:
      1. Forward pass  → logits
      2. Weighted cross-entropy loss  (compensates class imbalance)
      3. loss.backward()  → gradients through all 117M weights (or LoRA only)
      4. Gradient clipping  (max_norm=1.0)
      5. AdamW step  →  weights updated
      6. LR scheduler step  (linear warmup → linear decay)
    """

    def __init__(self, model, train_ds, val_ds, training_args: dict,
                 pos_weight: float, device: torch.device):
        self.model      = model.to(device)
        self.train_ds   = train_ds
        self.val_ds     = val_ds
        self.args       = training_args
        self.pos_weight = pos_weight
        self.device     = device
        self.best_f1    = 0.0
        self.history    = {
            "train_loss": [], "val_loss": [],
            "val_f1": [], "val_mcc": [], "val_auroc": [],
            "epoch_time_min": [],
        }

    # ── helpers ───────────────────────────────────────────────────────────────

    def _make_loaders(self):
        pin = torch.cuda.is_available()
        train_loader = DataLoader(
            self.train_ds, batch_size=self.args["batch_size"],
            shuffle=True, num_workers=0, pin_memory=pin)
        val_loader = DataLoader(
            self.val_ds, batch_size=self.args["eval_batch_size"],
            shuffle=False, num_workers=0, pin_memory=pin)
        return train_loader, val_loader

    def _make_optimizer_scheduler(self, total_steps: int):
        """AdamW + linear-warmup / linear-decay scheduler."""
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.args["lr"],
            weight_decay=self.args["weight_decay"],
        )
        warmup_steps = int(total_steps * self.args["warmup_ratio"])

        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / max(warmup_steps, 1)
            progress = float(step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return max(0.0, 1.0 - progress)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return optimizer, scheduler

    def _log_trainable(self):
        total   = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(
            f"[Params] Total: {total/1e6:.1f}M | "
            f"Trainable: {trainable/1e6:.2f}M ({100*trainable/total:.1f}%)"
        )

    # ── main train loop ───────────────────────────────────────────────────────

    def train(self) -> dict:
        train_loader, val_loader = self._make_loaders()
        total_steps = len(train_loader) * self.args["epochs"]
        optimizer, scheduler = self._make_optimizer_scheduler(total_steps)

        # Class-weighted cross-entropy (STEP 4 – loss function)
        loss_weights = torch.tensor(
            [1.0, self.pos_weight], dtype=torch.float32, device=self.device)
        loss_fn = nn.CrossEntropyLoss(weight=loss_weights)

        # FP16 AMP scaler
        use_amp = self.args.get("fp16", False) and torch.cuda.is_available()
        scaler  = torch.amp.GradScaler("cuda") if use_amp else None

        self._log_trainable()
        logger.info(
            f"[Train] {self.args['epochs']} epochs | "
            f"{len(train_loader)} batches/epoch | "
            f"lr={self.args['lr']} | device={self.device} | AMP={use_amp}"
        )

        for epoch in range(self.args["epochs"]):
            epoch_start = time.time()
            self.model.train()
            total_loss, n_batches = 0.0, 0

            for batch in train_loader:
                ids   = batch["input_ids"].to(self.device)
                mask  = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                optimizer.zero_grad()

                # ── Forward pass (STEP 2 + 3) ──────────────────────────────
                if use_amp:
                    with torch.amp.autocast("cuda"):
                        out  = self.model(input_ids=ids, attention_mask=mask)
                        loss = loss_fn(out.logits, labels)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.args["max_grad_norm"])
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    out  = self.model(input_ids=ids, attention_mask=mask)
                    loss = loss_fn(out.logits, labels)
                    # ── Backward pass (STEP 4) ──────────────────────────────
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.args["max_grad_norm"])
                    optimizer.step()

                scheduler.step()
                total_loss += loss.item()
                n_batches  += 1

            avg_train_loss = total_loss / max(n_batches, 1)
            val_metrics, val_loss = self._evaluate(val_loader, loss_fn, use_amp)
            elapsed_min = (time.time() - epoch_start) / 60

            self.history["train_loss"].append(round(avg_train_loss, 4))
            self.history["val_loss"].append(round(val_loss, 4))
            self.history["val_f1"].append(round(val_metrics["f1"], 4))
            self.history["val_mcc"].append(round(val_metrics["mcc"], 4))
            self.history["val_auroc"].append(round(val_metrics["auroc"], 4))
            self.history["epoch_time_min"].append(round(elapsed_min, 2))

            logger.info(
                f"Epoch {epoch+1}/{self.args['epochs']} | "
                f"TrainLoss={avg_train_loss:.4f} | ValLoss={val_loss:.4f} | "
                f"F1={val_metrics['f1']:.4f} | MCC={val_metrics['mcc']:.4f} | "
                f"AUROC={val_metrics['auroc']:.4f} | "
                f"Time={elapsed_min:.1f}min"
            )

            # Save best checkpoint
            if val_metrics["f1"] > self.best_f1:
                self.best_f1 = val_metrics["f1"]
                best_dir = str(MODELS / "dnabert2_amr_best")
                self.model.save_pretrained(best_dir)
                logger.info(f"  [*] New best model saved (F1={self.best_f1:.4f})")

        # Save final model
        final_dir = str(MODELS / "dnabert2_amr_final")
        os.makedirs(final_dir, exist_ok=True)
        self.model.save_pretrained(final_dir)
        logger.info(f"Final model saved -> {final_dir}")
        return self.history

    def _evaluate(self, loader, loss_fn, use_amp):
        self.model.eval()
        all_logits, all_labels, total_loss, n = [], [], 0.0, 0
        with torch.no_grad():
            for batch in loader:
                ids    = batch["input_ids"].to(self.device)
                mask   = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                if use_amp:
                    with torch.amp.autocast("cuda"):
                        out  = self.model(input_ids=ids, attention_mask=mask)
                        loss = loss_fn(out.logits, labels)
                else:
                    out  = self.model(input_ids=ids, attention_mask=mask)
                    loss = loss_fn(out.logits, labels)
                total_loss += loss.item(); n += 1
                all_logits.append(out.logits.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        logits = np.concatenate(all_logits)
        labels = np.concatenate(all_labels)
        preds  = np.argmax(logits, axis=-1)
        probs  = softmax(logits, axis=-1)[:, 1]

        metrics = {
            "f1":        f1_score(labels, preds, average="binary", zero_division=0),
            "mcc":       matthews_corrcoef(labels, preds),
            "precision": precision_score(labels, preds, zero_division=0),
            "recall":    recall_score(labels, preds, zero_division=0),
            "auroc":     0.0,
        }
        try:
            metrics["auroc"] = roc_auc_score(labels, probs)
        except ValueError:
            pass
        return metrics, total_loss / max(n, 1)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 – LoRA  (optional memory-efficient fine-tuning)
# ══════════════════════════════════════════════════════════════════════════════

def apply_lora(model: DNABERT2Classifier) -> DNABERT2Classifier:
    """Freeze all BERT weights and inject LoRA adapters into Q and V matrices.

    - All 117M base-model parameters: requires_grad = False  (frozen)
    - LoRA adapters (rank=16, alpha=32): requires_grad = True  (~1.3M params)
    - Classification head: requires_grad = True  (always trained from scratch)

    This cuts VRAM usage by ~60% with negligible accuracy loss.
    """
    try:
        from peft import LoraConfig, get_peft_model, TaskType
    except ImportError:
        logger.warning("[LoRA] peft not installed. Run: pip install peft. Skipping LoRA.")
        return model

    lora_cfg = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=16,                        # rank of the low-rank decomposition
        lora_alpha=32,               # scaling factor (alpha/r = 2)
        target_modules=["Wqkv"],     # Combined QKV projection matrix in DNABERT-2
        lora_dropout=0.1,
        bias="none",
    )

    # Wrap only the BERT backbone with LoRA (not the classifier head)
    model.bert = get_peft_model(model.bert, lora_cfg)

    # Ensure the classification head remains trainable
    for param in model.dropout.parameters():
        param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    logger.info(f"[LoRA] Applied. Trainable: {trainable/1e6:.2f}M / {total/1e6:.1f}M total.")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Fine-tune DNABERT-2 for AMR detection")
    parser.add_argument("--use_lora",    action="store_true",
                        help="STEP 5: Apply LoRA (60%% VRAM reduction)")
    parser.add_argument("--epochs",      type=int,   default=3,
                        help="Training epochs (default: 3)")
    parser.add_argument("--batch_size",  type=int,   default=8,
                        help="Training batch size")
    parser.add_argument("--lr",          type=float, default=2e-5,
                        help="AdamW learning rate (default: 2e-5, slow to preserve pre-training)")
    parser.add_argument("--resume",      action="store_true",
                        help="Resume from best checkpoint")
    parser.add_argument("--max_samples", type=int,   default=None,
                        help="Limit training samples (quick smoke-test)")
    args = parser.parse_args()

    logger.info(f"\n{'='*65}")
    logger.info(f"  DNABERT-2 AMR Fine-Tuning  --  {datetime.now().isoformat()}")
    logger.info(f"{'='*65}")

    # ── Check preprocessed data exists ─────────────────────────────────────
    train_csv = SPLITS / "train.csv"
    dev_csv   = SPLITS / "dev.csv"
    if not train_csv.is_file() or not dev_csv.is_file():
        logger.error("Train/dev CSVs not found. Run preprocess.py first.")
        sys.exit(1)

    train_df  = pd.read_csv(train_csv)
    n_pos     = int((train_df["label"] == 1).sum())
    n_neg     = int((train_df["label"] == 0).sum())
    pos_weight = n_neg / max(n_pos, 1)
    logger.info(f"[Data] {len(train_df)} sequences  |  pos={n_pos}  neg={n_neg}")
    logger.info(f"[Data] Positive class weight: {pos_weight:.3f}  "
                f"(higher = more penalty for missing resistant genes)")

    # ── Device ─────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"[Device] {device}" +
                (f"  --  {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else ""))

    # ── STEP 1: Load tokenizer (BPE) ───────────────────────────────────────
    from dnabert2_loader import load_dnabert2_base, DNABERT2ForClassification
    logger.info("[Step 1] Loading BPE tokenizer from local cache...")
    base_model, config, tokenizer = load_dnabert2_base()
    logger.info(f"[Step 1] Tokenizer ready | vocab_size={tokenizer.vocab_size}")

    # ── STEP 2+3: Build classifier model ───────────────────────────────────
    logger.info("[Step 2+3] Building DNABERT-2 + Classification Head...")
    best_weights = MODELS / "dnabert2_amr_best" / "pytorch_model.bin"

    if args.resume and best_weights.is_file():
        logger.info(f"[Step 2+3] Resuming from checkpoint: {best_weights}")
        model = DNABERT2Classifier(base_model, config.hidden_size, num_labels=2)
        state = torch.load(str(best_weights), map_location="cpu")
        model.load_state_dict(state, strict=False)
    else:
        model = DNABERT2Classifier(base_model, config.hidden_size, num_labels=2)

    logger.info(
        f"[Step 2+3] Model built | "
        f"hidden={config.hidden_size} | "
        f"params={sum(p.numel() for p in model.parameters())/1e6:.1f}M"
    )

    # ── STEP 5: LoRA (optional) ─────────────────────────────────────────────
    if args.use_lora:
        logger.info("[Step 5] Applying LoRA — freezing base BERT, training adapters only...")
        model = apply_lora(model)
    else:
        logger.info("[Step 5] Full fine-tuning (all parameters trainable). "
                    "Use --use_lora on consumer GPUs.")

    # ── Build datasets (BPE tokenization happens in __getitem__) ───────────
    max_val = (args.max_samples // 4) if args.max_samples else None
    train_ds = AMRDataset(str(train_csv), tokenizer, MAX_LENGTH, args.max_samples)
    val_ds   = AMRDataset(str(dev_csv),   tokenizer, MAX_LENGTH, max_val)
    logger.info(f"[Data] Train={len(train_ds)} | Val={len(val_ds)}")

    # Save tokenizer alongside model checkpoints
    for d in ["dnabert2_amr_best", "dnabert2_amr_final"]:
        save_dir = str(MODELS / d)
        os.makedirs(save_dir, exist_ok=True)
        tokenizer.save_pretrained(save_dir)

    # ── STEP 4: Training args (AdamW + warmup + cross-entropy) ─────────────
    training_args = {
        "epochs":        args.epochs,
        "lr":            args.lr,         # 2e-5 default (slow, preserve pre-training)
        "batch_size":    args.batch_size,
        "eval_batch_size": 16,
        "warmup_ratio":  0.10,            # 10% warm-up steps
        "weight_decay":  0.01,            # AdamW regularisation
        "max_grad_norm": 1.0,             # gradient clipping
        "fp16":          True,            # AMP on CUDA
    }
    logger.info(
        f"[Step 4] AdamW lr={training_args['lr']} | "
        f"warmup={training_args['warmup_ratio']*100:.0f}% | "
        f"weight_decay={training_args['weight_decay']} | "
        f"grad_clip={training_args['max_grad_norm']} | "
        f"fp16={training_args['fp16']}"
    )

    # ── Train ───────────────────────────────────────────────────────────────
    trainer = AMRTrainer(
        model=model, train_ds=train_ds, val_ds=val_ds,
        training_args=training_args,
        pos_weight=pos_weight, device=device,
    )
    history = trainer.train()

    # ── Save training history ───────────────────────────────────────────────
    hist_path = RESULTS / "training_history.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"[Done] Training history saved -> {hist_path}")
    logger.info(f"[Done] Best validation F1 : {trainer.best_f1:.4f}")
    logger.info("[Done] Next step: python scripts/evaluate.py")


if __name__ == "__main__":
    main()
