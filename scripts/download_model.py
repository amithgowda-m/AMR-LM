#!/usr/bin/env python3
"""
download_model.py - Pre-download DNABERT-2 from Hugging Face Hub.
=================================================================
Run this once before training to cache the model locally so that
train_dnabert2.py and evaluate.py never need an internet connection.

Usage:
    python scripts/download_model.py
    python scripts/download_model.py --force     # re-download even if cached
    python scripts/download_model.py --verify    # check cached files only

The model is saved to:
    AMR-LM/models/dnabert2_pretrained/
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Ensure the scripts dir is importable
SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# Use ASCII-safe stream to avoid cp1252 issues on Windows
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

MODEL_NAME = "zhihan1996/DNABERT-2-117M"
DEFAULT_CACHE = str(PROJECT_ROOT / "models" / "dnabert2_pretrained")

# Expected files after a successful download
EXPECTED_FILES = [
    "config.json",
    "tokenizer_config.json",
    "pytorch_model.bin",        # full weights (or model.safetensors)
    "special_tokens_map.json",
    "bert_layers.py",           # remote-code module
    "configuration_bert.py",    # remote-code config
]


def verify_cache(cache_dir: str) -> bool:
    """Check that all critical files are present in the local cache.

    Args:
        cache_dir: Path to the model snapshot directory.

    Returns:
        True if all expected files are found, False otherwise.
    """
    missing = []
    for fname in EXPECTED_FILES:
        fpath = os.path.join(cache_dir, fname)
        # pytorch_model.bin OR model.safetensors is acceptable
        if fname == "pytorch_model.bin":
            safe = os.path.join(cache_dir, "model.safetensors")
            if not os.path.isfile(fpath) and not os.path.isfile(safe):
                missing.append(f"{fname} (or model.safetensors)")
            continue
        if not os.path.isfile(fpath):
            missing.append(fname)

    if missing:
        logger.warning("Missing files in cache:")
        for f in missing:
            logger.warning(f"  [MISSING]  {f}")
        return False

    logger.info("Cache verification passed. All expected files are present:")
    for fname in EXPECTED_FILES:
        fpath = os.path.join(cache_dir, fname)
        if os.path.isfile(fpath):
            size_mb = os.path.getsize(fpath) / (1024 * 1024)
            logger.info(f"  [OK]  {fname}  ({size_mb:.1f} MB)")
    return True


def print_next_steps():
    """Print a summary of next pipeline steps."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("  DNABERT-2 download complete!")
    logger.info("=" * 60)
    logger.info("  Next steps:")
    logger.info("    1.  python scripts/setup.py")
    logger.info("    2.  python scripts/download_negatives_and_test_dbs.py")
    logger.info("    3.  python scripts/preprocess.py")
    logger.info("    4.  python scripts/train_dnabert2.py  [--use_lora]")
    logger.info("=" * 60)


def main():
    """Entry point -- parse args and trigger download/verify."""
    parser = argparse.ArgumentParser(
        description="Download DNABERT-2 from Hugging Face Hub to local cache."
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=DEFAULT_CACHE,
        help=f"Directory to save the model (default: {DEFAULT_CACHE})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if a local copy already exists.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Only verify the existing cache; do not re-download.",
    )
    args = parser.parse_args()

    logger.info(f"Model : {MODEL_NAME}")
    logger.info(f"Cache : {args.cache_dir}")
    logger.info("")

    # Verify-only mode
    if args.verify:
        ok = verify_cache(args.cache_dir)
        sys.exit(0 if ok else 1)

    # Check huggingface_hub
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        logger.error("huggingface_hub is not installed.")
        logger.error("Run:  pip install huggingface_hub")
        sys.exit(1)

    # Download via dnabert2_loader
    from dnabert2_loader import download_model
    download_model(cache_dir=args.cache_dir, force=args.force)

    # Quick load test
    logger.info("")
    logger.info("[Test] Running quick load verification...")
    try:
        from dnabert2_loader import load_dnabert2_base
        base_model, config, tokenizer = load_dnabert2_base(cache_dir=args.cache_dir)
        logger.info(f"[Test] OK - Tokenizer loaded  (vocab_size={tokenizer.vocab_size})")
        logger.info(f"[Test] OK - Config loaded     (hidden_size={config.hidden_size})")
        logger.info(
            f"[Test] OK - Model loaded      "
            f"({sum(p.numel() for p in base_model.parameters()) / 1e6:.1f}M params)"
        )
        del base_model  # free memory
    except Exception as exc:
        logger.error(f"[Test] Load verification failed: {exc}")
        sys.exit(1)

    # Verify file layout
    verify_cache(args.cache_dir)
    print_next_steps()


if __name__ == "__main__":
    main()
