@echo off
echo ═══════════════════════════════════════════════
echo   AMR DNABERT-2 Pipeline — Full Run
echo ═══════════════════════════════════════════════
echo.

echo Activating Python 3.11 Virtual Environment...
call venv\Scripts\activate.bat
if errorlevel 1 (echo FAILED: Could not activate venv. Run setup first. & exit /b 1)
echo.

echo [1/9] Setting up project...
python scripts\setup.py
if errorlevel 1 (echo FAILED: setup.py & exit /b 1)

echo [2/9] Downloading DNABERT-2 model from Hugging Face...
python scripts\download_model.py
if errorlevel 1 (echo FAILED: download_model.py & exit /b 1)

echo [3/9] Downloading databases and negatives...
python scripts\download_negatives_and_test_dbs.py
if errorlevel 1 (echo FAILED: download_negatives_and_test_dbs.py & exit /b 1)

echo [4/9] Preprocessing data...
python scripts\preprocess.py
if errorlevel 1 (echo FAILED: preprocess.py & exit /b 1)

echo [5/9] Running baselines...
python scripts\run_baselines.py
if errorlevel 1 (echo WARNING: Baselines had issues, continuing...)

echo [6/9] Training DNABERT-2...
python scripts\train_dnabert2.py --use_lora --batch_size 2
if errorlevel 1 (echo FAILED: train_dnabert2.py & exit /b 1)

echo [7/9] Evaluating model...
python scripts\evaluate.py
if errorlevel 1 (echo FAILED: evaluate.py & exit /b 1)

echo [8/9] Generating figures...
python scripts\generate_paper_figures.py

echo [9/9] Generating tables...
python scripts\generate_paper_tables.py

echo.
echo ═══════════════════════════════════════════════
echo   Pipeline complete! Check results\ folder
echo ═══════════════════════════════════════════════
