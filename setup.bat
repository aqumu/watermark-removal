@echo off

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

echo Activating virtual environment...
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip --quiet

echo.
echo Select your hardware:
echo   1) CPU only (no GPU)
echo   2) NVIDIA GPU - CUDA 12.1
echo   3) NVIDIA GPU - CUDA 11.8
echo   4) Skip PyTorch install (already installed)
echo.
set /p SETUP_CHOICE="Enter choice [1-4]: "

if "%SETUP_CHOICE%"=="1" (
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
) else if "%SETUP_CHOICE%"=="2" (
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
) else if "%SETUP_CHOICE%"=="3" (
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
) else if "%SETUP_CHOICE%"=="4" (
    echo Skipping PyTorch install.
) else (
    echo Invalid choice. Defaulting to CPU.
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
)

echo.
echo Installing dependencies...
pip install -r pipeline\requirements.txt
pip install -r training\requirements.txt

echo.
echo Installing CLI commands (wm-generate, wm-preview)...
pip install -e pipeline\

echo.
echo === Setup complete. Virtual environment is now active. ===
echo.
echo First-time workflow on a new machine:
echo.
echo   1) Generate the dataset:
echo        wm-generate --config pipeline\configs\default.yaml
echo.
echo   2) Train the model:
echo        cd training
echo        python train.py
echo.
echo   3) Run inference:
echo        python infer.py --checkpoint checkpoints\best.pth ^
echo            --watermarked IMAGE --mask MASK --output result.png
echo.
