@echo off
echo ========================================
echo Universal Self-RAG Setup
echo ========================================

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.8+
    exit /b 1
)

echo Python found
python --version

REM Install dependencies
echo.
echo Installing dependencies...
pip install -r requirements.txt

REM Create .env if not exists
if not exist .env (
    echo.
    echo Creating .env file...
    copy .env.example .env
    echo WARNING: Please edit .env and add your GOOGLE_API_KEY
) else (
    echo .env file already exists
)

REM Create documents folder
if not exist documents mkdir documents

echo.
echo ========================================
echo Setup complete!
echo.
echo Next steps:
echo 1. Get your FREE Gemini API key: https://makersuite.google.com/app/apikey
echo 2. Edit .env and add your API key
echo 3. Add PDF files to documents/ folder
echo 4. Run: python example.py
echo.
echo ========================================
pause
