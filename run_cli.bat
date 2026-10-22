@echo off
echo ===================================================
echo Starting AI Smart Surveillance System (CLI Mode)...
echo ===================================================
py -3.10 main.py --source samples/real_surveillance.mp4
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Attempting default python launcher...
    python main.py --source samples/real_surveillance.mp4
)
pause
