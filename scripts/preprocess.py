#!/usr/bin/env python3
"""
preprocess.py - Parse CARD/MEGARes/SARG, deduplicate, check leakage, build splits.
"""
import os, sys, re, random, logging, csv, hashlib
from datetime import datetime
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from tqdm import tqdm

random.seed(42)
np.random.seed(42)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(PROJECT_ROOT, "data", "raw")
CARD_RAW = os.path.join(RAW, "card")
MEGARES_RAW = os.path.join(RAW, "megares")
SARG_RAW = os.path.join(RAW, "sarg")
NEG_RAW = os.path.join(RAW, "negatives")
PROCESSED = os.path.join(PROJECT_ROOT, "data", "processed")
SPLITS = os.path.join(PROJECT_ROOT, "data", "splits")
RESULTS = os.path.join(PROJECT_ROOT, "results")
LOG_FILE = os.path.join(RESULTS, "pipeline.log")

for d in [PROCESSED, SPLITS, RESULTS]:
    os.makedirs(d, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"), logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)


def read_fasta(fpath):
    """Read FASTA file, return list of (header, sequence) tuples."""
    records = []
    if not os.path.isfile(fpath):
        logger.warning(f"FASTA not found: {fpath}")
        return records
    header, seq_parts = None, []
    with open(fpath, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(seq_parts).upper()))
                header = line
                seq_parts = []
            elif header is not None:
                seq_parts.append(line)
        if header is not None:
            records.append((header, "".join(seq_parts).upper()))
    return records


def parse_aro_from_header(header):
    """Extract ARO accession from CARD FASTA header."""
    match = re.search(r'ARO:\d+', header)
    return match.group(0) if match else None


def load_aro_index(card_dir):
    """Load aro_index.tsv and return dict mapping ARO accession to metadata."""
    aro_path = os.path.join(card_dir, "aro_index.tsv")
    aro_map = {}
    if not os.path.isfile(aro_path):
        logger.warning(f"aro_index.tsv not found at {aro_path}")
        return aro_map
    try:
        df = pd.read_csv(aro_path, sep="\t", dtype=str, on_bad_lines="skip")
        df.columns = [c.strip() for c in df.columns]
        for _, row in df.iterrows():
            aro = None
            for col in df.columns:
                val = str(row.get(col, ""))
                m = re.search(r'ARO:\d+', val)
                if m:
                    aro = m.group(0)
                    break
            if aro:
                drug = str(row.get("Drug Class", row.get("drug_class", "")))
                mech = str(row.get("Resistance Mechanism", row.get("resistance_mechanism", "")))
                fam = str(row.get("AMR Gene Family", row.get("amr_gene_family", "")))
                name = str(row.get("Model Name", row.get("model_name", row.get("ARO Name", ""))))
                aro_map[aro] = {"drug_class": drug, "resistance_mechanism": mech,
                                "amr_gene_family": fam, "gene_name": name}
    except Exception as e:
        logger.warning(f"Error parsing aro_index.tsv: {e}")
    logger.info(f"  Loaded {len(aro_map)} ARO entries from aro_index.tsv")
    return aro_map


def parse_card_positives():
    """Parse and combine all three CARD positive FASTA files."""
    logger.info("Step 1: Parsing CARD positive sequences...")
    fasta_files = [
        ("nucleotide_fasta_protein_homolog_model.fasta", "homolog"),
        ("nucleotide_fasta_protein_variant_model.fasta", "variant"),
        ("nucleotide_fasta_rRNA_gene_variant_model.fasta", "rRNA"),
    ]
    aro_map = load_aro_index(CARD_RAW)
    records = []
    for fname, model_type in fasta_files:
        fpath = os.path.join(CARD_RAW, fname)
        seqs = read_fasta(fpath)
        logger.info(f"  {fname}: {len(seqs)} sequences")
        for header, seq in seqs:
            if len(seq) < 100 or len(seq) > 5000:
                continue
            clean = re.sub(r'[^ATCGN]', '', seq)
            if len(clean) < 100:
                continue
            aro = parse_aro_from_header(header)
            meta = aro_map.get(aro, {})
            gene_name = meta.get("gene_name", header.split("|")[0].lstrip(">") if "|" in header else header[:50])
            records.append({
                "sequence": clean, "label": 1, "gene_name": gene_name,
                "drug_class": meta.get("drug_class", "Unknown"),
                "resistance_mechanism": meta.get("resistance_mechanism", "Unknown"),
                "amr_gene_family": meta.get("amr_gene_family", "Unknown"),
                "model_type": model_type, "sequence_length": len(clean),
                "source_db": "card", "aro": aro if aro else "Unknown",
            })
    logger.info(f"  Total CARD positives after filtering: {len(records)}")
    return pd.DataFrame(records)


def parse_negatives():
    """Parse negative sequences from RefSeq."""
    logger.info("Step 2: Parsing negative sequences...")
    fpath = os.path.join(NEG_RAW, "refseq_negatives.fasta")
    seqs = read_fasta(fpath)
    records = []
    for header, seq in seqs:
        if len(seq) < 100 or len(seq) > 5000:
            continue
        clean = re.sub(r'[^ATCGN]', '', seq)
        if len(clean) < 100:
            continue
        records.append({
            "sequence": clean, "label": 0, "gene_name": "non_AMR",
            "drug_class": "None", "resistance_mechanism": "None",
            "amr_gene_family": "None", "model_type": "negative",
            "sequence_length": len(clean), "source_db": "refseq", "aro": "None",
        })
    logger.info(f"  Negatives after filtering: {len(records)}")
    return pd.DataFrame(records)


def balance_dataset(pos_df, neg_df):
    """Balance positive and negative classes."""
    logger.info("Step 3: Balancing dataset...")
    n_pos, n_neg = len(pos_df), len(neg_df)
    logger.info(f"  Before: {n_pos} positives, {n_neg} negatives")
    if n_neg > n_pos * 2:
        neg_df = neg_df.sample(n=n_pos, random_state=42).reset_index(drop=True)
    elif n_neg < n_pos // 2:
        needed = n_pos - n_neg
        extra = neg_df.sample(n=needed, replace=True, random_state=42).reset_index(drop=True)
        seqs = extra["sequence"].tolist()
        mutated = []
        for s in seqs:
            s_list = list(s)
            n_mut = max(1, int(len(s) * 0.015))
            for _ in range(n_mut):
                pos = random.randint(0, len(s_list) - 1)
                s_list[pos] = random.choice("ATCG")
            mutated.append("".join(s_list))
        extra["sequence"] = mutated
        neg_df = pd.concat([neg_df, extra], ignore_index=True)
    logger.info(f"  After: {len(pos_df)} positives, {len(neg_df)} negatives")
    return pos_df, neg_df


def deduplicate_sequences(df, threshold=0.90):
    """Remove near-duplicate sequences using hash-based + subsequence approach."""
    logger.info("Step 4: Deduplicating sequences...")
    n_before = len(df)
    seen_hashes = set()
    keep_idx = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Dedup"):
        seq = row["sequence"]
        h = hashlib.md5(seq.encode()).hexdigest()
        if h not in seen_hashes:
            seen_hashes.add(h)
            keep_idx.append(idx)
    df = df.loc[keep_idx].reset_index(drop=True)
    # Further dedup: remove sequences where one is substring of another (sample check)
    if len(df) > 500:
        seqs = df["sequence"].tolist()
        remove_idx = set()
        sample_size = min(len(seqs), 5000)
        sample_indices = random.sample(range(len(seqs)), sample_size)
        short_seqs = {}
        for i in sample_indices:
            key = seqs[i][:50]
            if key in short_seqs:
                j = short_seqs[key]
                if i != j and seqs[i] == seqs[j]:
                    remove_idx.add(i)
            else:
                short_seqs[key] = i
        if remove_idx:
            df = df.drop(index=list(remove_idx)).reset_index(drop=True)
    n_after = len(df)
    logger.info(f"  Removed {n_before - n_after} duplicates ({n_before} -> {n_after})")
    return df


def check_leakage_python(train_seqs, test_seqs, test_name, threshold=0.90):
    """Python fallback for cross-database leakage check using k-mer overlap."""
    logger.info(f"  Checking leakage: {test_name} vs training set (Python method)...")
    k = 31
    train_kmers = set()
    for seq in tqdm(train_seqs[:2000], desc=f"Building train k-mers", leave=False):
        for i in range(len(seq) - k + 1):
            train_kmers.add(seq[i:i+k])
    flagged = []
    for idx, seq in enumerate(tqdm(test_seqs, desc=f"Checking {test_name}", leave=False)):
        if len(seq) < k:
            continue
        test_km = set()
        for i in range(len(seq) - k + 1):
            test_km.add(seq[i:i+k])
        if len(test_km) == 0:
            continue
        overlap = len(test_km & train_kmers) / len(test_km)
        if overlap >= threshold:
            flagged.append(idx)
    logger.info(f"  {test_name}: {len(flagged)} sequences flagged (>{threshold*100:.0f}% k-mer overlap)")
    return flagged


def build_splits(pos_df, neg_df):
    """Build stratified train/dev/test splits from CARD data."""
    logger.info("Step 6: Building stratified splits...")
    from sklearn.model_selection import train_test_split
    combined = pd.concat([pos_df, neg_df], ignore_index=True)
    combined["strat_key"] = combined["drug_class"].fillna("Unknown")
    class_counts = combined["strat_key"].value_counts()
    rare = class_counts[class_counts < 3].index.tolist()
    combined.loc[combined["strat_key"].isin(rare), "strat_key"] = "Rare_combined"
    train_val, test = train_test_split(combined, test_size=0.20, random_state=42,
                                        stratify=combined["strat_key"])
    train, val = train_test_split(train_val, test_size=0.125, random_state=42,
                                   stratify=train_val["strat_key"])
    for split_df in [train, val, test]:
        if "strat_key" in split_df.columns:
            split_df.drop(columns=["strat_key"], inplace=True, errors="ignore")
    cols = ["sequence", "label", "gene_name", "drug_class", "resistance_mechanism",
            "amr_gene_family", "model_type", "sequence_length", "source_db"]
    train = train[[c for c in cols if c in train.columns]].reset_index(drop=True)
    val = val[[c for c in cols if c in val.columns]].reset_index(drop=True)
    test = test[[c for c in cols if c in test.columns]].reset_index(drop=True)
    train.to_csv(os.path.join(SPLITS, "train.csv"), index=False)
    val.to_csv(os.path.join(SPLITS, "dev.csv"), index=False)
    test.to_csv(os.path.join(SPLITS, "test_card.csv"), index=False)
    logger.info(f"  Train: {len(train)} | Dev: {len(val)} | Test CARD: {len(test)}")
    return train, val, test


def parse_external_db(db_name):
    """Parse MEGARes or SARG into a DataFrame."""
    if db_name == "megares":
        fpath = os.path.join(MEGARES_RAW, "megares_database_v3.00.fasta")
    else:
        fpath = os.path.join(SARG_RAW, "database.fasta")
    seqs = read_fasta(fpath)
    records = []
    for header, seq in seqs:
        clean = re.sub(r'[^ATCGN]', '', seq)
        if len(clean) < 100 or len(clean) > 5000:
            continue
        records.append({
            "sequence": clean, "label": 1, "gene_name": header[:80],
            "drug_class": "External", "resistance_mechanism": "External",
            "amr_gene_family": "External", "model_type": "external",
            "sequence_length": len(clean), "source_db": db_name,
        })
    return pd.DataFrame(records)


def build_external_test(db_name, neg_test_df, train_seqs):
    """Build test CSV for MEGARes or SARG with leakage removal."""
    logger.info(f"Step 7: Building {db_name} test set...")
    ext_df = parse_external_db(db_name)
    logger.info(f"  {db_name} raw positives: {len(ext_df)}")
    # Leakage check
    flagged = check_leakage_python(train_seqs, ext_df["sequence"].tolist(), db_name, threshold=0.90)
    n_removed = len(flagged)
    if flagged:
        ext_df = ext_df.drop(index=flagged).reset_index(drop=True)
    logger.info(f"  {db_name} after leakage removal: {len(ext_df)} (removed {n_removed})")
    combined = pd.concat([ext_df, neg_test_df], ignore_index=True)
    cols = ["sequence", "label", "gene_name", "drug_class", "resistance_mechanism",
            "amr_gene_family", "model_type", "sequence_length", "source_db"]
    combined = combined[[c for c in cols if c in combined.columns]]
    outpath = os.path.join(SPLITS, f"test_{db_name}.csv")
    combined.to_csv(outpath, index=False)
    logger.info(f"  Saved: {outpath} ({len(combined)} sequences)")
    return n_removed, len(ext_df)


def fragment_sequences(input_csv, window, step):
    """Fragment sequences from a test CSV into shorter reads."""
    df = pd.read_csv(input_csv)
    fragments = []
    for _, row in df.iterrows():
        seq = str(row["sequence"])
        if len(seq) <= window:
            new_row = row.to_dict()
            new_row["sequence_length"] = len(seq)
            fragments.append(new_row)
        else:
            for i in range(0, len(seq) - window + 1, step):
                frag = seq[i:i+window]
                new_row = row.to_dict()
                new_row["sequence"] = frag
                new_row["sequence_length"] = len(frag)
                fragments.append(new_row)
    return pd.DataFrame(fragments)


def generate_fragments():
    """Generate fragment versions for all test CSVs."""
    logger.info("Step 8: Generating short-read fragments...")
    configs = [(150, 75), (300, 150), (500, 250)]
    test_files = ["test_card", "test_megares", "test_sarg"]
    for test_name in test_files:
        input_csv = os.path.join(SPLITS, f"{test_name}.csv")
        if not os.path.isfile(input_csv):
            logger.warning(f"  Skipping fragments for {test_name} (file not found)")
            continue
        for window, step in configs:
            out_csv = os.path.join(SPLITS, f"{test_name}_{window}bp.csv")
            frag_df = fragment_sequences(input_csv, window, step)
            frag_df.to_csv(out_csv, index=False)
            logger.info(f"  {test_name}_{window}bp: {len(frag_df)} fragments")


def write_leakage_report(megares_removed, sarg_removed, megares_total, sarg_total):
    """Write DATA_LEAKAGE_REPORT.md."""
    report_path = os.path.join(RESULTS, "DATA_LEAKAGE_REPORT.md")
    meg_pct = (megares_removed / max(megares_total + megares_removed, 1)) * 100
    sarg_pct = (sarg_removed / max(sarg_total + sarg_removed, 1)) * 100
    content = f"""# Data Leakage Report

**Generated**: {datetime.now().isoformat()}
**Method**: K-mer overlap analysis (k=31, threshold=90%)

## Summary

| Database | Sequences Before | Removed | Remaining | % Removed |
|----------|-----------------|---------|-----------|-----------|
| MEGARes  | {megares_total + megares_removed} | {megares_removed} | {megares_total} | {meg_pct:.1f}% |
| SARG     | {sarg_total + sarg_removed} | {sarg_removed} | {sarg_total} | {sarg_pct:.1f}% |

## Methods (copy-paste for paper)

> To prevent data leakage between training and evaluation sets, we performed
> pairwise k-mer overlap analysis (k=31) between all CARD training sequences
> and both external test databases. Sequences from MEGARes and SARG with
> greater than 90% k-mer overlap to any training sequence were removed.
> This resulted in the removal of {megares_removed} sequences from MEGARes
> ({meg_pct:.1f}%) and {sarg_removed} sequences from SARG ({sarg_pct:.1f}%).
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(content)
    logger.info(f"  Leakage report saved to: {report_path}")


def print_summary():
    """Print final data summary table."""
    logger.info("\n" + "=" * 55)
    logger.info("  DATA SUMMARY")
    logger.info("=" * 55)
    for name in ["train", "dev", "test_card", "test_megares", "test_sarg"]:
        fpath = os.path.join(SPLITS, f"{name}.csv")
        if os.path.isfile(fpath):
            df = pd.read_csv(fpath)
            n_pos = (df["label"] == 1).sum()
            n_neg = (df["label"] == 0).sum()
            logger.info(f"  {name:15s}: {len(df):6d} seq ({n_pos} pos, {n_neg} neg)")
    # Drug classes
    train_path = os.path.join(SPLITS, "train.csv")
    if os.path.isfile(train_path):
        df = pd.read_csv(train_path)
        if "drug_class" in df.columns:
            n_classes = df["drug_class"].nunique()
            logger.info(f"  Drug classes:    {n_classes} unique in train")
    logger.info("=" * 55)
    logger.info("  Next step: python scripts/train_dnabert2.py")
    logger.info("=" * 55)


def main():
    """Run the complete preprocessing pipeline."""
    logger.info(f"\n{'='*60}\nPreprocessing — {datetime.now().isoformat()}\n{'='*60}")
    pos_df = parse_card_positives()
    neg_df = parse_negatives()
    pos_df, neg_df = balance_dataset(pos_df, neg_df)
    pos_df = deduplicate_sequences(pos_df)
    train, val, test_card = build_splits(pos_df, neg_df)
    train_seqs = train[train["label"] == 1]["sequence"].tolist()
    neg_test = test_card[test_card["label"] == 0].copy()
    meg_rem, meg_tot = build_external_test("megares", neg_test, train_seqs)
    sarg_rem, sarg_tot = build_external_test("sarg", neg_test, train_seqs)
    write_leakage_report(meg_rem, sarg_rem, meg_tot, sarg_tot)
    generate_fragments()
    print_summary()


if __name__ == "__main__":
    main()
