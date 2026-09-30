@echo off
title Local RAG MCP - Attivazione venv
echo ========================================================
echo Attivazione Ambiente Virtuale Python (venv)...
echo ========================================================
if exist "%~dp0venv\Scripts\activate.bat" (
    call "%~dp0venv\Scripts\activate.bat"
    echo.
    echo Ambiente virtuale attivato con successo!
    echo Ora puoi eseguire comandi come:
    echo   python ingest.py
    echo   python query.py
    echo.
    cmd /k
) else (
    echo [ERRORE] Cartella venv non trovata in %~dp0
    echo Assicurati di aver creato l'ambiente virtuale prima di eseguire questo file.
    echo.
    pause
)
