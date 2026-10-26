@echo off
title Universal AI-CAD Generator - Streamlit Engine
echo ========================================================
echo   Launching Universal AI-CAD Generator Web Application
echo ========================================================
echo.
cd /d "%~dp0"
call .venv\Scripts\activate.bat
streamlit run app.py
pause
