@echo off
REM Download SGCC dataset from Kaggle (Windows version)
REM Requires Kaggle API credentials

echo [INFO] Downloading SGCC dataset from Kaggle...

REM Check if kaggle is installed
where kaggle >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Kaggle CLI not found. Install with: pip install kaggle
    exit /b 1
)

REM Create data directory if it doesn't exist
if not exist "data" mkdir data

REM Download dataset
echo [INFO] Downloading from bensalem14/sgcc-dataset...
kaggle datasets download -d bensalem14/sgcc-dataset -p data --unzip

if %errorlevel% equ 0 (
    echo [SUCCESS] Dataset downloaded to data/
    echo [INFO] Downloaded files:
    dir data\
    echo [INFO] Dataset ready for processing!
) else (
    echo [ERROR] Failed to download dataset. Check your Kaggle credentials.
    exit /b 1
)
