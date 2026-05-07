# start_eval.ps1

$root = "C:\git\Master-Thesis-GEOAI"
$eval = "$root\evaluation"
$python = "$root\.venv\Scripts\python.exe"

# Backend
Start-Process powershell -ArgumentList @"
cd $eval
$python -m uvicorn server:app --port 8001 --reload
"@

# Frontend
Start-Process powershell -ArgumentList @"
cd $eval
$python -m http.server 8000
"@

Write-Host "Servers started:"
Write-Host "Frontend: http://localhost:8000"
Write-Host "Backend:  http://localhost:8001"