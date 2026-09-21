Write-Host "Starting Mirage-AEC..." -ForegroundColor Cyan

# Start Backend in a new window
Write-Host "Starting Backend..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd backend; if (Test-Path .venv\Scripts\python.exe) { .venv\Scripts\python.exe -m uvicorn main:app --reload } else { python -m uvicorn main:app --reload }"

# Start Frontend in a new window
Write-Host "Starting Frontend..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd frontend; npm run dev"

Write-Host "Both servers are starting in separate windows." -ForegroundColor Cyan
