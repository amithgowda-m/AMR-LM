#!/bin/bash
set -e
echo "═══════════════════════════════════════════════"
echo "  AMR DNABERT-2 Pipeline — Full Run"
echo "═══════════════════════════════════════════════"

echo "[1/8] Setting up project..."
python scripts/setup.py

echo "[2/8] Downloading databases and negatives..."
python scripts/download_negatives_and_test_dbs.py

echo "[3/8] Preprocessing data..."
python scripts/preprocess.py

echo "[4/8] Running baselines..."
python scripts/run_baselines.py || echo "WARNING: Baselines had issues, continuing..."

echo "[5/8] Training DNABERT-2..."
python scripts/train_dnabert2.py

echo "[6/8] Evaluating model..."
python scripts/evaluate.py

echo "[7/8] Generating figures..."
python scripts/generate_paper_figures.py

echo "[8/8] Generating tables..."
python scripts/generate_paper_tables.py

echo ""
echo "═══════════════════════════════════════════════"
echo "  Pipeline complete! Check results/ folder"
echo "═══════════════════════════════════════════════"
