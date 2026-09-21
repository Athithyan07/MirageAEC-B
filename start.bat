@echo off
echo Starting Mirage-AEC...

:: Start Backend in a new CMD window
echo Starting Backend...
start "Mirage-AEC Backend" cmd /k "if exist backend\.venv\Scripts\activate.bat (call backend\.venv\Scripts\activate.bat) && python -m uvicorn backend.main:app --reload"

:: Start Frontend in a new CMD window
echo Starting Frontend...
start "Mirage-AEC Frontend" cmd /k "cd frontend && npm run dev"

echo Both servers are starting in separate windows.
