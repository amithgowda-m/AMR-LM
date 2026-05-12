#!/usr/bin/env python3
"""
setup.py — AMR DNABERT-2 Pipeline Setup
========================================
Creates all project directories, installs dependencies, verifies CARD data,
and prints a readiness checklist.
"""

import os
import sys
import subprocess
import logging
from datetime import datetime

# ==============================================
# CONFIGURATION
# ==============================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CARD_DATA_DIR = os.path.join(os.path.dirname(PROJECT_ROOT), "card-data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
LOG_FILE = os.path.join(RESULTS_DIR, "pipeline.log")

# Directories to create
DIRECTORIES = [
    os.path.join(PROJECT_ROOT, "data", "raw", "card"),
    os.path.join(PROJECT_ROOT, "data", "raw", "megares"),
    os.path.join(PROJECT_ROOT, "data", "raw", "sarg"),
    os.path.join(PROJECT_ROOT, "data", "raw", "negatives"),
    os.path.join(PROJECT_ROOT, "data", "processed"),
    os.path.join(PROJECT_ROOT, "data", "splits"),
    os.path.join(PROJECT_ROOT, "models", "dnabert2_amr_best"),
    os.path.join(PROJECT_ROOT, "models", "dnabert2_amr_final"),
    os.path.join(RESULTS_DIR, "figures"),
    os.path.join(RESULTS_DIR, "tables"),
    os.path.join(PROJECT_ROOT, "scripts"),
    os.path.join(PROJECT_ROOT, "notebooks"),
]

# CARD files to verify
REQUIRED_CARD_FILES = [
    "nucleotide_fasta_protein_homolog_model.fasta",
    "nucleotide_fasta_protein_variant_model.fasta",
    "nucleotide_fasta_rRNA_gene_variant_model.fasta",
    "aro_index.tsv",
    "aro_categories.tsv",
    "aro_categories_index.tsv",
    "card.json",
]

OPTIONAL_CARD_FILES = [
    "shortname_antibiotics.tsv",
    "PMID.tsv",
    "snps.txt",
]


def setup_logging():
    """Configure dual logging to file and console."""
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


def create_directories(logger):
    """Create all project directories."""
    logger.info("Creating project directories...")
    created = 0
    for d in DIRECTORIES:
        if not os.path.exists(d):
            os.makedirs(d, exist_ok=True)
            created += 1
            logger.info(f"  Created: {os.path.relpath(d, PROJECT_ROOT)}")
        else:
            logger.info(f"  Exists:  {os.path.relpath(d, PROJECT_ROOT)}")
    logger.info(f"Directories ready ({created} newly created).")


def copy_card_files(logger):
    """Copy CARD data files to the project's data/raw/card/ directory."""
    import shutil

    dest_dir = os.path.join(PROJECT_ROOT, "data", "raw", "card")
    card_src = CARD_DATA_DIR
    logger.info(f"Copying CARD files from {card_src} -> {dest_dir}")

    if not os.path.isdir(card_src):
        logger.warning(f"CARD data directory not found at {card_src}")
        logger.warning("Trying alternative path: card-data.tar")
        alt_path = os.path.join(os.path.dirname(PROJECT_ROOT), "card-data.tar")
        if os.path.isdir(alt_path):
            logger.info(f"Found CARD data at {alt_path}")
            card_src = alt_path
        else:
            logger.error("Cannot find CARD data. Please check the path.")
            return False

    copied = 0
    for fname in REQUIRED_CARD_FILES + OPTIONAL_CARD_FILES:
        src = os.path.join(card_src, fname)
        dst = os.path.join(dest_dir, fname)
        if os.path.isfile(src):
            shutil.copy2(src, dst)
            copied += 1
        else:
            logger.warning(f"  Missing: {fname}")
    logger.info(f"Copied {copied} CARD files to data/raw/card/")
    return True


def install_dependencies(logger):
    """Install Python dependencies from requirements.txt."""
    req_file = os.path.join(PROJECT_ROOT, "requirements.txt")
    if not os.path.isfile(req_file):
        logger.error(f"requirements.txt not found at {req_file}")
        return False

    logger.info("Installing Python dependencies...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", req_file, "--quiet"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        logger.info("All dependencies installed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        logger.warning(f"pip install returned non-zero: {e}")
        logger.info("Attempting individual package installs...")
        with open(req_file, "r") as f:
            for line in f:
                pkg = line.strip()
                if pkg and not pkg.startswith("#"):
                    try:
                        subprocess.check_call(
                            [sys.executable, "-m", "pip", "install", pkg, "--quiet"],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE,
                        )
                    except subprocess.CalledProcessError:
                        logger.warning(f"  Failed to install: {pkg}")
        return True


def check_gpu(logger):
    """Check CUDA availability and print GPU info."""
    logger.info("Checking GPU availability...")
    try:
        import torch

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"  [OK] GPU available: {gpu_name}")
            logger.info(f"  [OK] VRAM: {vram:.1f} GB")
            logger.info(f"  [OK] CUDA version: {torch.version.cuda}")
            logger.info(f"  [OK] PyTorch version: {torch.__version__}")
        else:
            logger.warning("  [FAIL] No CUDA GPU detected. Training will use CPU (much slower).")
            logger.info(f"  PyTorch version: {torch.__version__}")
    except ImportError:
        logger.error("  PyTorch not installed. Run this script again after install.")


def verify_card_files(logger):
    """Verify CARD files exist and print a checklist."""
    logger.info("Verifying CARD data files...")
    card_dir = os.path.join(PROJECT_ROOT, "data", "raw", "card")
    all_found = True
    checklist = []

    for fname in REQUIRED_CARD_FILES:
        fpath = os.path.join(card_dir, fname)
        if os.path.isfile(fpath):
            size_mb = os.path.getsize(fpath) / (1024 * 1024)
            checklist.append(f"  [OK] {fname} ({size_mb:.1f} MB)")
        else:
            checklist.append(f"  [FAIL] {fname} — MISSING (REQUIRED)")
            all_found = False

    for fname in OPTIONAL_CARD_FILES:
        fpath = os.path.join(card_dir, fname)
        if os.path.isfile(fpath):
            size_mb = os.path.getsize(fpath) / (1024 * 1024)
            checklist.append(f"  [OK] {fname} ({size_mb:.1f} MB)")
        else:
            checklist.append(f"  [-] {fname} — optional, not found")

    for line in checklist:
        logger.info(line)

    if all_found:
        logger.info("All required CARD files verified.")
    else:
        logger.error("Some required CARD files are missing!")

    return all_found


def print_summary(logger, card_ok):
    """Print final setup summary."""
    logger.info("")
    logger.info("=" * 50)
    logger.info("  SETUP COMPLETE")
    logger.info("=" * 50)
    logger.info(f"  Project root:  {PROJECT_ROOT}")
    logger.info(f"  CARD data:     {'[OK] Ready' if card_ok else '[FAIL] Issues found'}")
    logger.info(f"  Log file:      {LOG_FILE}")
    logger.info("")
    logger.info("  Next step: python scripts/download_negatives_and_test_dbs.py")
    logger.info("=" * 50)


def main():
    """Run the complete setup pipeline."""
    logger = setup_logging()
    logger.info(f"AMR DNABERT-2 Pipeline Setup — {datetime.now().isoformat()}")
    logger.info(f"Python: {sys.version}")
    logger.info(f"Project root: {PROJECT_ROOT}")
    logger.info("")

    # Step 1: Create directories
    create_directories(logger)

    # Step 2: Copy CARD files
    copy_card_files(logger)

    # Step 3: Install dependencies
    install_dependencies(logger)

    # Step 4: Check GPU
    check_gpu(logger)

    # Step 5: Verify CARD files
    card_ok = verify_card_files(logger)

    # Summary
    print_summary(logger, card_ok)


if __name__ == "__main__":
    main()
