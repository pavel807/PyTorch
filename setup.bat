@echo off
setlocal
title AMD ROCm 7.1.1 + PyTorch Setup (Python 3.12)

echo ========================================================
echo      AMD ROCm 7.1.1 Installation Assistant
echo      Target Python: 3.12
echo ========================================================
echo.

:: 1. ПРОВЕРКА ВЕРСИИ PYTHON (Критично для cp312 whl файлов)
python --version 2>nul | findstr /C:"3.12" >nul
if %errorlevel% neq 0 (
    echo [ERROR] Kriticheskaya oshibka!
    echo Vashi ssylki prednaznacheny dlya Python 3.12.
    echo Vasha tekushchaya versiya:
    python --version
    echo.
    echo Pozhaluysta, ustanovite Python 3.12.x i zapustite skript snova.
    pause
    exit /b
)

:: 2. НАСТРОЙКА ОКРУЖЕНИЯ
echo [SETUP] Primenenie peremennykh sredy dlya stabil'nosti GPU...

:: Фикс для большинства карт Radeon (RX 6xxx/7xxx)
setx HSA_OVERRIDE_GFX_VERSION 10.3.0 >nul
set "HSA_OVERRIDE_GFX_VERSION=10.3.0"

:: Выбор основной видеокарты
setx HIP_VISIBLE_DEVICES 0 >nul
set "HIP_VISIBLE_DEVICES=0"

echo [OK] HSA_OVERRIDE_GFX_VERSION = 10.3.0
echo [OK] HIP_VISIBLE_DEVICES = 0

:: 3. ОЧИСТКА СТАРЫХ ВЕРСИЙ
echo.
echo [CLEANUP] Udalenie starykh bibliotek torch...
pip uninstall -y torch torchvision torchaudio rocm-sdk-core rocm-sdk-devel rocm-sdk-libraries-custom
echo [OK] Ochistka zavershena.

:: 4. УСТАНОВКА ROCm SDK (Ваши ссылки 1-4)
echo.
echo [INSTALL] Ustanovka ROCm SDK Components...
echo Step 1/4: Core...
pip install --no-cache-dir https://repo.radeon.com/rocm/windows/rocm-rel-7.1.1/rocm_sdk_core-0.1.dev0-py3-none-win_amd64.whl
echo Step 2/4: Devel...
pip install --no-cache-dir https://repo.radeon.com/rocm/windows/rocm-rel-7.1.1/rocm_sdk_devel-0.1.dev0-py3-none-win_amd64.whl
echo Step 3/4: Libraries Custom...
pip install --no-cache-dir https://repo.radeon.com/rocm/windows/rocm-rel-7.1.1/rocm_sdk_libraries_custom-0.1.dev0-py3-none-win_amd64.whl
echo Step 4/4: ROCm Tarball...
pip install --no-cache-dir https://repo.radeon.com/rocm/windows/rocm-rel-7.1.1/rocm-0.1.dev0.tar.gz

:: 5. УСТАНОВКА PYTORCH (Ваши ссылки 5-7)
echo.
echo [INSTALL] Ustanovka PyTorch 2.9.0 (ROCm)...
echo Eto bolshoy fayl, naberites' terpeniya...
echo.
echo Installing Torch...
pip install --no-cache-dir https://repo.radeon.com/rocm/windows/rocm-rel-7.1.1/torch-2.9.0+rocmsdk20251116-cp312-cp312-win_amd64.whl

echo Installing Torchaudio...
pip install --no-cache-dir https://repo.radeon.com/rocm/windows/rocm-rel-7.1.1/torchaudio-2.9.0+rocmsdk20251116-cp312-cp312-win_amd64.whl

echo Installing Torchvision...
pip install --no-cache-dir https://repo.radeon.com/rocm/windows/rocm-rel-7.1.1/torchvision-0.24.0+rocmsdk20251116-cp312-cp312-win_amd64.whl

:: 6. ПРОВЕРКА
echo.
echo [TEST] Zapusk proverki PyTorch + ROCm...
(
echo import torch
echo try:
echo     print("-" * 50^)
echo     print("PyTorch Version: " + torch.__version__^)
echo     # print("ROCm Version: " + str(torch.version.hip^)^) # In newer builds implies cuda version
echo     print("-" * 50^)
echo     if torch.cuda.is_available(^):
echo         print("SUCCESS: AMD GPU detected via HIP!"^)
echo         print("Device: " + torch.cuda.get_device_name(0^)^)
echo         x = torch.tensor([1.0, 2.0]^).to('cuda'^)
echo         print("Tensor Test: OK " + str(x^)^)
echo     else:
echo         print("ERROR: GPU not detected. You are running on CPU."^)
echo except Exception as e:
echo     print("CRITICAL ERROR: " + str(e^)^)
) > verify_rocm.py

python verify_rocm.py

echo.
echo ========================================================
echo Esli sverkhu napisano SUCCESS - zapuskayte vash chatbot!
echo Vash kod dolzhen ispol'zovat' DEVICE = 'cuda'
echo ========================================================
pause
del verify_rocm.py
