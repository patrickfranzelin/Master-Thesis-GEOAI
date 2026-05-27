$root = "C:\git\Master-Thesis-GEOAI"
$eval = "$root\evaluation"
$python = "$root\.venv\Scripts\python.exe"

Start-Process powershell -ArgumentList @"
cd $eval
$python -m uvicorn semantic_server:app --port 8002 --reload
"@

Start-Process powershell -ArgumentList @"
cd $eval
$python -m http.server 8000
"@

Write-Host "Semantic evaluation servers started:"
Write-Host "Frontend: http://localhost:8000/semantic_index.html"
Write-Host "Backend:  http://localhost:8002"
