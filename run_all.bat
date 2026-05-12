@echo off
echo ═══════════════════════════════════════════════
echo   AMR DNABERT-2 Pipeline — Full Run
echo ═══════════════════════════════════════════════
echo.

echo [1/8] Setting up project...
python scripts\setup.py
if errorlevel 1 (echo FAILED: setup.py & pause & exit /b 1)

echo [2/8] Downloading databases and negatives...
python scripts\download_negatives_and_test_dbs.py
if errorlevel 1 (echo FAILED: download_negatives_and_test_dbs.py & pause & exit /b 1)

echo [3/8] Preprocessing data...
python scripts\preprocess.py
if errorlevel 1 (echo FAILED: preprocess.py & pause & exit /b 1)

echo [4/8] Running baselines...
python scripts\run_baselines.py
if errorlevel 1 (echo WARNING: Baselines had issues, continuing...)

echo [5/8] Training DNABERT-2...
python scripts\train_dnabert2.py
if errorlevel 1 (echo FAILED: train_dnabert2.py & pause & exit /b 1)

echo [6/8] Evaluating model...
python scripts\evaluate.py
if errorlevel 1 (echo FAILED: evaluate.py & pause & exit /b 1)

echo [7/8] Generating figures...
python scripts\generate_paper_figures.py

echo [8/8] Generating tables...
python scripts\generate_paper_tables.py

echo.
echo ═══════════════════════════════════════════════
echo   Pipeline complete! Check results\ folder
echo ═══════════════════════════════════════════════
pause
