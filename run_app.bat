@echo off
setlocal
cd /d "%~dp0"

echo ============================================
echo   RAE-1B Real-Time Testing App
echo ============================================
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo Python was not found on PATH. Install Python 3.10+ and try again.
    pause
    exit /b 1
)

if not exist "checkpoints\app_bundle.pt" (
    echo No trained model bundle found - training now.
    echo This happens once and takes a few minutes ^(faster with a GPU^)...
    echo.
    python train_and_export.py
    if errorlevel 1 (
        echo.
        echo Training failed - see the errors above.
        pause
        exit /b 1
    )
)

echo Starting the server in a new window...
start "RAE-1B Server" cmd /k "python app.py"

echo Waiting for it to come up...
timeout /t 10 /nobreak >nul

echo Opening http://localhost:7860 in your browser...
start "" http://localhost:7860

echo.
echo Done. The server keeps running in the "RAE-1B Server" window -
echo close that window (or Ctrl+C inside it) to stop the app.
pause
