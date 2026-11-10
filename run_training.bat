@echo off
REM ===========================================================================
REM  run_training.bat - double-click to train. Resumes automatically.
REM
REM  Usage:
REM     double-click                 -> trains 2q_cnn
REM     run_training.bat 5q_cnn      -> trains that one experiment
REM     run_training.bat all         -> all four, one after another
REM
REM  Valid names: 2q_cnn  5q_cnn  2q_stt  5q_stt  all
REM
REM  It activates the venv, checks the GPU, starts CARLA itself, and continues
REM  from the newest checkpoint - so stopping with Ctrl+C never loses progress.
REM ===========================================================================

setlocal

REM ---- edit these two lines if your folders ever move ----
set "PROJECT=D:\safety-aware-drl-carla\safety-aware-drl-carla-update-v1"
set "CARLA_EXE=D:\WORK\Carla\CarlaUE4.exe"
REM --------------------------------------------------------

set "EXPERIMENT=%~1"
if "%EXPERIMENT%"=="" set "EXPERIMENT=2q_cnn"

if not exist "%PROJECT%\CARLA-RL-Agents" (
    echo [ERROR] Cannot find "%PROJECT%\CARLA-RL-Agents"
    echo Edit the PROJECT line at the top of this file.
    pause
    exit /b 1
)
if not exist "%PROJECT%\venv\Scripts\activate.bat" (
    echo [ERROR] Cannot find the venv at "%PROJECT%\venv"
    pause
    exit /b 1
)
if not exist "%PROJECT%\train_all.py" (
    echo [ERROR] train_all.py is missing from "%PROJECT%"
    pause
    exit /b 1
)

cd /d "%PROJECT%\CARLA-RL-Agents"
call "%PROJECT%\venv\Scripts\activate.bat"

echo.
echo ============================ GPU CHECK ============================
python -c "import torch; ok=torch.cuda.is_available(); print('torch   :', torch.__version__); print('CUDA    :', ok); print('device  :', torch.cuda.get_device_name(0) if ok else 'CPU ONLY - training will be slow')"
echo ===================================================================
echo.
echo If the device line says CPU ONLY, close this and reinstall torch with:
echo    python -m pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128
echo.

if not exist "%CARLA_EXE%" (
    echo [WARN] CARLA not found at "%CARLA_EXE%"
    echo        Start CARLA yourself and leave it open - the runner will wait for it.
    echo.
    if /i "%EXPERIMENT%"=="all" (
        python "%PROJECT%\train_all.py"
    ) else (
        python "%PROJECT%\train_all.py" --only %EXPERIMENT%
    )
) else (
    if /i "%EXPERIMENT%"=="all" (
        python "%PROJECT%\train_all.py" --carla-exe "%CARLA_EXE%"
    ) else (
        python "%PROJECT%\train_all.py" --only %EXPERIMENT% --carla-exe "%CARLA_EXE%"
    )
)

echo.
echo ===================================================================
echo  Stopped. Run this file again any time - it continues from the
echo  newest checkpoint instead of starting the agent over.
echo ===================================================================
pause
endlocal
