"""
dnabert2_loader.py - Reliable DNABERT-2 model loader using Hugging Face Hub.

Downloads and caches DNABERT-2 (zhihan1996/DNABERT-2-117M) locally on first
run via huggingface_hub.snapshot_download, then loads from the local cache
on every subsequent run — no internet required after the first download.

Resolves the Windows config-class mismatch that arises when using
AutoModel.from_pretrained() directly on Windows with trust_remote_code=True,
by pre-fetching the model files and loading from the local snapshot path.
"""

import os
import sys
import logging
import torch
from pathlib import Path

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────
MODEL_NAME = "zhihan1996/DNABERT-2-117M"

# Local cache directory: AMR-LM/models/dnabert2_pretrained/
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCAL_CACHE_DIR = str(_PROJECT_ROOT / "models" / "dnabert2_pretrained")


# ─────────────────────────────────────────────────────────
# MODEL DOWNLOAD
# ─────────────────────────────────────────────────────────

def download_model(cache_dir: str = LOCAL_CACHE_DIR, force: bool = False) -> str:
    """
    Download DNABERT-2 from Hugging Face Hub to a local directory.

    Uses huggingface_hub.snapshot_download to mirror the full model
    repository (weights, tokenizer, remote code) into ``cache_dir``.
    On subsequent calls the cached snapshot is returned immediately.

    Args:
        cache_dir: Destination directory for the downloaded snapshot.
        force: If True, re-download even when local copy already exists.

    Returns:
        str: Absolute path to the local snapshot directory.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required. Install it with:\n"
            "    pip install huggingface_hub"
        )

    # Check if already downloaded (sentinel: config.json present)
    config_path = os.path.join(cache_dir, "config.json")
    if os.path.isfile(config_path) and not force:
        logger.info(f"[HF] DNABERT-2 already cached at: {cache_dir}")
        return cache_dir

    logger.info(f"[HF] Downloading '{MODEL_NAME}' -> {cache_dir}")
    logger.info("[HF] This may take a few minutes on first run (~500 MB)...")

    os.makedirs(cache_dir, exist_ok=True)
    snapshot_download(
        repo_id=MODEL_NAME,
        local_dir=cache_dir,
        local_dir_use_symlinks=False,   # Windows-safe: real files, not symlinks
        ignore_patterns=["*.msgpack", "*.h5", "flax_model*"],  # skip Flax/TF weights
    )

    logger.info(f"[HF] Download complete. Files saved to: {cache_dir}")
    return cache_dir


# ─────────────────────────────────────────────────────────
# CLASSIFICATION HEAD
# ─────────────────────────────────────────────────────────

class DNABERT2ForClassification(torch.nn.Module):
    """DNABERT-2 encoder + binary classification head.

    Wraps the raw BertModel returned by AutoModel and adds:
        - dropout (p=0.1)
        - a linear classifier over the [CLS] token representation

    The ``.logits`` attribute on the forward output is compatible with
    the existing WeightedTrainer in train_dnabert2.py.
    """

    def __init__(self, base_model, hidden_size: int, num_labels: int = 2):
        """
        Args:
            base_model: Loaded AutoModel instance (DNABERT-2 BertModel).
            hidden_size: Encoder hidden size (from config.hidden_size).
            num_labels: Number of output classes (default: 2).
        """
        super().__init__()
        self.bert = base_model
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        """
        Args:
            input_ids: (batch, seq_len) token IDs.
            attention_mask: (batch, seq_len) mask tensor.

        Returns:
            Namespace-like object with ``.logits`` shape (batch, num_labels).
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # DNABERT-2 returns tuple OR ModelOutput; handle both
        if isinstance(outputs, tuple):
            hidden_states = outputs[0]
        elif hasattr(outputs, "last_hidden_state"):
            hidden_states = outputs.last_hidden_state
        else:
            hidden_states = outputs[0]

        pooled = hidden_states[:, 0]          # [CLS] token
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)

        # Return an object with .logits so existing trainer code needs no changes
        return type("Output", (), {"logits": logits})()

    def save_pretrained(self, save_dir: str):
        """Save full model state dict to ``save_dir/pytorch_model.bin``."""
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))
        logger.info(f"[Checkpoint] Model state dict saved to {save_dir}")


# ─────────────────────────────────────────────────────────
# LOADER FUNCTIONS (public API — unchanged signature)
# ─────────────────────────────────────────────────────────

def load_dnabert2_base(cache_dir: str = LOCAL_CACHE_DIR):
    """Load base DNABERT-2 encoder, config, and tokenizer from local cache.

    Downloads the model on first call if not already present.

    Args:
        cache_dir: Local directory containing (or to store) the model snapshot.

    Returns:
        tuple: (base_model, config, tokenizer)
            - base_model: ``AutoModel`` instance (BertModel backbone).
            - config: ``AutoConfig`` instance.
            - tokenizer: ``AutoTokenizer`` instance.
    """
    from transformers import AutoTokenizer, AutoConfig, AutoModel

    # 1. Ensure model is downloaded locally
    local_path = download_model(cache_dir=cache_dir)

    logger.info(f"[Loader] Loading tokenizer from: {local_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        local_path,
        trust_remote_code=True,
    )

    logger.info(f"[Loader] Loading config from: {local_path}")
    config = AutoConfig.from_pretrained(
        local_path,
        trust_remote_code=True,
    )

    logger.info(f"[Loader] Loading model weights from: {local_path}")
    base_model = AutoModel.from_pretrained(
        local_path,
        config=config,
        trust_remote_code=True,
    )

    logger.info(
        f"[Loader] DNABERT-2 loaded | hidden_size={config.hidden_size} | "
        f"params={sum(p.numel() for p in base_model.parameters()) / 1e6:.1f}M"
    )
    return base_model, config, tokenizer


def load_dnabert2_classifier(
    weights_path: str = None,
    cache_dir: str = LOCAL_CACHE_DIR,
    num_labels: int = 2,
):
    """Load DNABERT-2 with classification head, optionally restoring fine-tuned weights.

    Args:
        weights_path: Path to ``pytorch_model.bin`` with fine-tuned state dict.
                      If None or file not found, returns the freshly initialised head.
        cache_dir: Local directory for the pre-trained base model snapshot.
        num_labels: Number of output classes (default: 2).

    Returns:
        tuple: (model, tokenizer)
            - model: ``DNABERT2ForClassification`` instance.
            - tokenizer: ``AutoTokenizer`` instance.
    """
    base_model, config, tokenizer = load_dnabert2_base(cache_dir=cache_dir)
    model = DNABERT2ForClassification(base_model, config.hidden_size, num_labels=num_labels)

    if weights_path and os.path.isfile(weights_path):
        logger.info(f"[Loader] Restoring fine-tuned weights from: {weights_path}")
        state_dict = torch.load(weights_path, map_location="cpu")
        
        # Check if LoRA was used during training
        is_lora = any("lora_" in k or "base_model" in k for k in state_dict.keys())
        if is_lora:
            logger.info("[Loader] Detected LoRA weights in checkpoint. Applying PEFT wrapper...")
            try:
                from peft import LoraConfig, get_peft_model, TaskType
                lora_cfg = LoraConfig(
                    task_type=TaskType.FEATURE_EXTRACTION,
                    r=16,
                    lora_alpha=32,
                    target_modules=["Wqkv"],
                    lora_dropout=0.1,
                    bias="none",
                )
                model.bert = get_peft_model(model.bert, lora_cfg)
            except ImportError:
                logger.error("[Loader] Checkpoint uses LoRA but 'peft' library is not installed.")
                
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning(f"[Loader] Missing keys in checkpoint: {missing}")
        if unexpected:
            logger.warning(f"[Loader] Unexpected keys in checkpoint: {unexpected}")
        logger.info("[Loader] Fine-tuned weights loaded successfully.")
    else:
        logger.info("[Loader] No fine-tuned weights found -- using pre-trained base only.")

    return model, tokenizer


# ── Backward-compatibility alias ──────────────────────────────────────────────
# train_dnabert2.py and evaluate.py may import DNABERT2ForClassification from
# this module.  The canonical class now lives in train_dnabert2.py but we keep
# a re-export here so existing code continues to work.
try:
    from train_dnabert2 import DNABERT2Classifier as DNABERT2ForClassification  # noqa: F401
except ImportError:
    # Fallback: define it locally using the same logic
    class DNABERT2ForClassification(torch.nn.Module):  # type: ignore[no-redef]
        """Lightweight alias kept for backward compatibility."""
        def __init__(self, base_model, hidden_size, num_labels=2):
            super().__init__()
            self.bert = base_model
            self.dropout = torch.nn.Dropout(0.1)
            self.classifier = torch.nn.Linear(hidden_size, num_labels)

        def forward(self, input_ids, attention_mask=None, **kwargs):
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            hs = outputs[0] if isinstance(outputs, tuple) else outputs.last_hidden_state
            pooled = self.dropout(hs[:, 0])
            logits = self.classifier(pooled)
            return type("Out", (), {"logits": logits})()

        def save_pretrained(self, save_dir):
            import os
            os.makedirs(save_dir, exist_ok=True)
            torch.save(self.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))

