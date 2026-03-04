@echo off
setlocal

echo === Watermark Removal - Windows Setup ===
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Install Python 3.10-3.13 from https://python.org
    exit /b 1
)

if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
)

call .venv\Scripts\activate.bat
python -m pip install --upgrade pip --quiet

echo.
echo Select your hardware:
echo   1) CPU only (no GPU)
echo   2) NVIDIA GPU - CUDA 12.1
echo   3) NVIDIA GPU - CUDA 11.8
echo   4) Skip PyTorch install (already installed)
echo.
set /p choice="Enter choice [1-4]: "

if "%choice%"=="1" (
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
) else if "%choice%"=="2" (
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
) else if "%choice%"=="3" (
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
) else if "%choice%"=="4" (
    echo Skipping PyTorch install.
) else (
    echo Invalid choice. Defaulting to CPU.
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
)

echo.
echo Installing dependencies...
pip install -r pipeline\requirements.txt
pip install -r training\requirements.txt
pip install -e pipeline\

echo.
echo === Setup complete ===
echo.
echo Generate the dataset (run from pipeline\ directory):
echo   cd pipeline
echo   ..\venv\Scripts\wm-generate
echo   cd ..
echo.
echo Train the model:
echo   .venv\Scripts\python.exe training\train.py
echo.
echo Run inference:
echo   .venv\Scripts\python.exe training\infer.py --checkpoint training\checkpoints\epoch_XXXX.pth --watermarked IMAGE --mask MASK --output result.png
echo.

endlocal
