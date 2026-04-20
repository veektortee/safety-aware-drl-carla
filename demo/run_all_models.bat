@echo off
REM Batch script to test all SAC model variants
REM Usage: run_all_models.bat [num_episodes]

setlocal enabledelayedexpansion

REM Get number of episodes from command line (default: 2)
set NUM_EPISODES=2
if not "%~1"=="" set NUM_EPISODES=%~1

echo.
echo ==============================================================================
echo SAC MODEL INFERENCE - AUTOMATIC TEST ALL VARIANTS
echo ==============================================================================
echo Episodes per model: %NUM_EPISODES%
echo ==============================================================================
echo.

REM Get current directory
set DEMO_DIR=%~dp0
cd /d "%DEMO_DIR%"

REM Color codes for output
set COLOR_GREEN=[92m
set COLOR_YELLOW=[93m
set COLOR_RED=[91m
set COLOR_RESET=[0m

REM Test each model
set MODELS=5cnn 2cnn 5stt 2stt
set MODEL_COUNT=0
set COMPLETED_COUNT=0
set FAILED_COUNT=0

for %%M in (%MODELS%) do (
    set /a MODEL_COUNT+=1
    echo.
    echo ==============================================================================
    echo Model %%M - Episode set %MODEL_COUNT% of 4
    echo ==============================================================================
    
    python run_model_inference.py --model %%M --episodes %NUM_EPISODES%
    
    if errorlevel 1 (
        set /a FAILED_COUNT+=1
        echo [ERROR] Model %%M failed
    ) else (
        set /a COMPLETED_COUNT+=1
        echo [OK] Model %%M completed
    )
    
    REM Wait between models
    if not "%%M"=="2stt" (
        echo.
        echo Waiting 5 seconds before next model...
        timeout /t 5 /nobreak
    )
)

REM Print summary
echo.
echo ==============================================================================
echo TEST SUMMARY
echo ==============================================================================
echo Total models tested:  %MODEL_COUNT%
echo Completed:           %COMPLETED_COUNT%
echo Failed:              %FAILED_COUNT%
echo ==============================================================================
echo.

REM Compare models if all succeeded
if %FAILED_COUNT% equ 0 (
    echo Running model comparison...
    python compare_models.py --export model_comparison_results.json
) else (
    echo Some tests failed. Skipping comparison.
)

echo.
echo Results saved. Check results_*.json and model_comparison_results.json
echo.
pause
