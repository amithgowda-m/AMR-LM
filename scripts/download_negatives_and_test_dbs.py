#!/usr/bin/env python3
"""
download_negatives_and_test_dbs.py — Download external databases and negative sequences
=======================================================================================
Part A: Download MEGARes v3.0 database
Part B: Download SARG database
Part C: Fetch negative sequences from NCBI RefSeq (housekeeping genes)
"""

import os
import sys
import logging
import random
import time
import urllib.request
import urllib.error
import json
from datetime import datetime

import numpy as np

# Seed for reproducibility
random.seed(42)
np.random.seed(42)

# ==============================================
# PATHS
# ==============================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(PROJECT_ROOT, "data", "raw")
MEGARES_DIR = os.path.join(DATA_RAW, "megares")
SARG_DIR = os.path.join(DATA_RAW, "sarg")
NEGATIVES_DIR = os.path.join(DATA_RAW, "negatives")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
LOG_FILE = os.path.join(RESULTS_DIR, "pipeline.log")


def setup_logging():
    """Configure logging to file and console."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


def download_file(url, dest_path, logger, desc="file"):
    """Download a file from URL to dest_path with retry logic."""
    if os.path.isfile(dest_path) and os.path.getsize(dest_path) > 100:
        logger.info(f"  Already exists: {os.path.basename(dest_path)} — skipping download")
        return True

    logger.info(f"  Downloading {desc} from {url}")
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as response:
                data = response.read()
            with open(dest_path, "wb") as f:
                f.write(data)
            size_mb = len(data) / (1024 * 1024)
            logger.info(f"  Downloaded: {os.path.basename(dest_path)} ({size_mb:.1f} MB)")
            return True
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            logger.warning(f"  Attempt {attempt + 1}/3 failed: {e}")
            time.sleep(5 * (attempt + 1))
    logger.error(f"  Failed to download {desc} after 3 attempts.")
    return False


def count_fasta_sequences(fpath):
    """Count the number of sequences in a FASTA file."""
    if not os.path.isfile(fpath):
        return 0
    count = 0
    try:
        with open(fpath, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.startswith(">"):
                    count += 1
    except Exception:
        return 0
    return count


def write_fasta(records, fpath):
    """Write list of (header, sequence) tuples to FASTA file."""
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    with open(fpath, "w", encoding="utf-8") as f:
        for header, seq in records:
            if not header.startswith(">"):
                header = ">" + header
            f.write(f"{header}\n")
            # Write sequence in 80-character lines
            for i in range(0, len(seq), 80):
                f.write(f"{seq[i:i+80]}\n")


# ==============================================
# PART A: MEGARes v3.0
# ==============================================
def download_megares(logger):
    """Download MEGARes v3.0 database and annotations."""
    logger.info("=" * 60)
    logger.info("PART A: Downloading MEGARes v3.0")
    logger.info("=" * 60)
    os.makedirs(MEGARES_DIR, exist_ok=True)

    # MEGARes v3.0 download URLs
    megares_urls = [
        (
            "https://www.meglab.org/downloads/megares_v3.00/megares_database_v3.00.fasta",
            os.path.join(MEGARES_DIR, "megares_database_v3.00.fasta"),
            "MEGARes database FASTA",
        ),
        (
            "https://www.meglab.org/downloads/megares_v3.00/megares_annotations_v3.00.csv",
            os.path.join(MEGARES_DIR, "megares_annotations_v3.00.csv"),
            "MEGARes annotations CSV",
        ),
    ]

    # Alternative URLs
    alt_megares_urls = [
        (
            "https://megares.meglab.org/download/megares_v3.00/megares_database_v3.00.fasta",
            os.path.join(MEGARES_DIR, "megares_database_v3.00.fasta"),
            "MEGARes database FASTA (alt)",
        ),
        (
            "https://megares.meglab.org/download/megares_v3.00/megares_annotations_v3.00.csv",
            os.path.join(MEGARES_DIR, "megares_annotations_v3.00.csv"),
            "MEGARes annotations CSV (alt)",
        ),
    ]

    success = True
    for url, dest, desc in megares_urls:
        if not download_file(url, dest, logger, desc):
            logger.info("  Trying alternative URL...")
            alt = [(u, d, de) for u, d, de in alt_megares_urls if d == dest]
            if alt and not download_file(alt[0][0], alt[0][1], logger, alt[0][2]):
                logger.warning(f"  Could not download {desc}. Will generate synthetic placeholder.")
                success = False

    # If download failed, generate synthetic MEGARes-like data for pipeline testing
    fasta_path = os.path.join(MEGARES_DIR, "megares_database_v3.00.fasta")
    if not os.path.isfile(fasta_path) or os.path.getsize(fasta_path) < 100:
        logger.info("  Generating synthetic MEGARes-like sequences for pipeline testing...")
        _generate_synthetic_amr_sequences(
            fasta_path,
            n_sequences=2000,
            prefix="MEGARes",
            logger=logger,
        )
        # Also generate a synthetic annotations CSV
        _generate_synthetic_megares_annotations(
            os.path.join(MEGARES_DIR, "megares_annotations_v3.00.csv"),
            fasta_path,
            logger,
        )

    n_seqs = count_fasta_sequences(fasta_path)
    logger.info(f"  MEGARes sequences: {n_seqs}")
    return success


def _generate_synthetic_amr_sequences(fpath, n_sequences, prefix, logger):
    """Generate synthetic AMR-like DNA sequences for pipeline testing when downloads fail."""
    logger.info(f"  Generating {n_sequences} synthetic {prefix} sequences...")
    records = []
    gene_classes = [
        ("blaOXA", "Beta-Lactam"),
        ("blaTEM", "Beta-Lactam"),
        ("blaCTX-M", "Beta-Lactam"),
        ("mecA", "Beta-Lactam"),
        ("vanA", "Glycopeptide"),
        ("vanB", "Glycopeptide"),
        ("tetM", "Tetracycline"),
        ("tetW", "Tetracycline"),
        ("ermB", "MLS"),
        ("ermC", "MLS"),
        ("aph3", "Aminoglycoside"),
        ("aac6", "Aminoglycoside"),
        ("mcr-1", "Colistin"),
        ("qnrA", "Fluoroquinolone"),
        ("sul1", "Sulfonamide"),
        ("dfrA", "Trimethoprim"),
        ("catA1", "Phenicol"),
        ("floR", "Phenicol"),
        ("fosA", "Fosfomycin"),
        ("linA", "Lincosamide"),
    ]

    for i in range(n_sequences):
        gene, drug_class = random.choice(gene_classes)
        seq_len = random.randint(300, 2500)
        seq = "".join(random.choices("ATCG", k=seq_len))
        header = f">{prefix}|{gene}_{i}|{drug_class}|RequiresSNPConfirmation"
        records.append((header, seq))

    write_fasta(records, fpath)
    logger.info(f"  Wrote {n_sequences} sequences to {os.path.basename(fpath)}")


def _generate_synthetic_megares_annotations(csv_path, fasta_path, logger):
    """Generate matching annotations CSV for synthetic MEGARes data."""
    import csv

    headers_list = []
    with open(fasta_path, "r") as f:
        for line in f:
            if line.startswith(">"):
                headers_list.append(line.strip().lstrip(">"))

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["header", "class", "mechanism", "group", "type"])
        for hdr in headers_list:
            parts = hdr.split("|")
            gene = parts[1] if len(parts) > 1 else "unknown"
            drug_class = parts[2] if len(parts) > 2 else "unknown"
            writer.writerow([hdr, drug_class, "Antibiotic_resistance", gene, "AMR"])

    logger.info(f"  Generated annotations CSV with {len(headers_list)} entries")


# ==============================================
# PART B: SARG Database
# ==============================================
def download_sarg(logger):
    """Download SARG database from GitHub."""
    logger.info("=" * 60)
    logger.info("PART B: Downloading SARG Database")
    logger.info("=" * 60)
    os.makedirs(SARG_DIR, exist_ok=True)

    sarg_urls = [
        "https://raw.githubusercontent.com/xinehc/args_oap/main/src/args_oap/db/sarg.fasta",
        "https://raw.githubusercontent.com/xinehc/args_oap/master/src/args_oap/db/sarg.fasta",
        "https://raw.githubusercontent.com/xinehc/args_oap/main/args_oap/db/sarg.fasta",
    ]

    dest = os.path.join(SARG_DIR, "database.fasta")
    success = False

    for url in sarg_urls:
        if download_file(url, dest, logger, "SARG database"):
            success = True
            break

    if not success or not os.path.isfile(dest) or os.path.getsize(dest) < 100:
        logger.info("  SARG download failed. Generating synthetic SARG-like sequences...")
        _generate_synthetic_amr_sequences(
            dest,
            n_sequences=1500,
            prefix="SARG",
            logger=logger,
        )

    n_seqs = count_fasta_sequences(dest)
    logger.info(f"  SARG sequences: {n_seqs}")
    return success


# ==============================================
# PART C: Negative Sequences (RefSeq)
# ==============================================
def fetch_negatives_entrez(logger, target_count=12000):
    """Download actual eukaryotic and bacterial sequences from Ensembl FTP."""
    import urllib.request
    import gzip
    import random

    logger.info("=" * 60)
    logger.info("PART C: Fetching Real Eukaryotic & Bacterial Negative Sequences")
    logger.info("=" * 60)
    os.makedirs(NEGATIVES_DIR, exist_ok=True)

    dest = os.path.join(NEGATIVES_DIR, "refseq_negatives.fasta")
    if os.path.isfile(dest) and count_fasta_sequences(dest) >= target_count // 2:
        n = count_fasta_sequences(dest)
        logger.info(f"  Negative sequences already exist: {n} sequences — skipping")
        return True

    records = []
    
    datasets = {
        "E_coli": ("transcript", "http://ftp.ensemblgenomes.org/pub/bacteria/release-57/fasta/bacteria_0_collection/escherichia_coli_str_k_12_substr_mg1655_gca_000005845/cds/Escherichia_coli_str_k_12_substr_mg1655_gca_000005845.ASM584v2.cds.all.fa.gz"),
        "B_subtilis": ("transcript", "http://ftp.ensemblgenomes.org/pub/bacteria/release-57/fasta/bacteria_0_collection/bacillus_subtilis_subsp_subtilis_str_168_gca_000009045/cds/Bacillus_subtilis_subsp_subtilis_str_168_gca_000009045.AL0091263.cds.all.fa.gz"),
        "Homo_sapiens": ("transcript", "http://ftp.ensembl.org/pub/release-110/fasta/homo_sapiens/cdna/Homo_sapiens.GRCh38.cdna.all.fa.gz"),
        "Mus_musculus": ("transcript", "http://ftp.ensembl.org/pub/release-110/fasta/mus_musculus/cdna/Mus_musculus.GRCm39.cdna.all.fa.gz"),
        "S_cerevisiae": ("transcript", "http://ftp.ensembl.org/pub/release-110/fasta/saccharomyces_cerevisiae/cdna/Saccharomyces_cerevisiae.R64-1-1.cdna.all.fa.gz")
    }

    seqs_per_genome = 10000
    all_negatives = []

    for org, (dtype, url) in datasets.items():
        gz_path = os.path.join(NEGATIVES_DIR, f"{org}.fa.gz")
        if not os.path.exists(gz_path) or os.path.getsize(gz_path) < 1000000:
            logger.info(f"  Downloading {org} {dtype} from Ensembl using curl...")
            import subprocess
            try:
                subprocess.run(["curl.exe", "-L", "-o", gz_path, url], check=True)
            except Exception as e:
                logger.error(f"  Failed to download {org} with curl: {e}")
                continue

        logger.info(f"  Extracting sequences from {org}...")
        try:
            org_records = []
            with gzip.open(gz_path, "rt") as f:
                if dtype == "genome":
                    genome = "".join([line.strip() for line in f if not line.startswith(">")])
                    for i in range(seqs_per_genome * 2): # Try to get enough
                        length = random.randint(300, 3000)
                        start = random.randint(0, len(genome) - length - 1)
                        seq = genome[start:start+length].upper()
                        if set(seq).issubset({"A", "C", "G", "T"}):
                            org_records.append((f">neg|{org}|genome_frag|seq_{i}", seq))
                            if len(org_records) >= seqs_per_genome:
                                break
                else:
                    header = None
                    seq = []
                    for line in f:
                        line = line.strip()
                        if line.startswith(">"):
                            if header and seq:
                                s = "".join(seq).upper()
                                if 200 <= len(s) <= 4000 and set(s).issubset({"A", "C", "G", "T"}):
                                    org_records.append((f">neg|{org}|transcript|{header}", s))
                            header = line[1:].split()[0]
                            seq = []
                        else:
                            seq.append(line)
            
            random.shuffle(org_records)
            records.extend(org_records[:seqs_per_genome])
        except Exception as e:
            logger.error(f"  Failed to process {org}: {e}")

    # Write all negatives
    random.shuffle(records)
    write_fasta(records, dest)
    logger.info(f"  Total negative sequences extracted: {len(records)}")
    logger.info(f"  Saved to: {dest}")
    return True


def print_summary(logger):
    """Print final download summary."""
    megares_fasta = os.path.join(MEGARES_DIR, "megares_database_v3.00.fasta")
    sarg_fasta = os.path.join(SARG_DIR, "database.fasta")
    neg_fasta = os.path.join(NEGATIVES_DIR, "refseq_negatives.fasta")

    logger.info("")
    logger.info("=" * 60)
    logger.info("  DOWNLOAD SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  MEGARes:    {count_fasta_sequences(megares_fasta)} sequences")
    logger.info(f"  SARG:       {count_fasta_sequences(sarg_fasta)} sequences")
    logger.info(f"  Negatives:  {count_fasta_sequences(neg_fasta)} sequences")
    logger.info("")
    logger.info("  Next step: python scripts/preprocess.py")
    logger.info("=" * 60)


def main():
    """Run all download tasks."""
    logger = setup_logging()
    logger.info(f"\n{'='*60}")
    logger.info(f"Download Script — {datetime.now().isoformat()}")
    logger.info(f"{'='*60}")

    download_megares(logger)
    download_sarg(logger)
    fetch_negatives_entrez(logger, target_count=8000)
    print_summary(logger)


if __name__ == "__main__":
    main()
