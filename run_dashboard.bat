@echo off
echo ===================================================
echo Starting AI Smart Surveillance System Dashboard...
echo ===================================================
py -3.10 -m streamlit run app.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Attempting default python launcher...
    python -m streamlit run app.py
)
pause
