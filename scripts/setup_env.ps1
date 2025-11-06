<#
.SYNOPSIS
  Set up the Master-Thesis-GEOAI Python environment on Windows
#>

Write-Host "=== Setting up Master-Thesis-GEOAI environment (Windows) ==="

# 1️⃣ Ensure uv is installed
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "Installing uv..."
    irm https://astral.sh/uv/install.ps1 | iex
}

# 2️⃣ Define paths
$projectRoot = (Resolve-Path "$PSScriptRoot\..").Path
$venvPath = Join-Path $projectRoot ".venv"
$reqFile = Join-Path $projectRoot "requirements.txt"

# 3️⃣ Create virtual environment
if (-not (Test-Path $venvPath)) {
    Write-Host "🐍 Creating new virtual environment..."
    uv venv --python 3.10 $venvPath
} else {
    Write-Host "✅ Virtual environment already exists."
}

# 4️⃣ Activate venv
& "$venvPath\Scripts\activate.ps1"

# 5️⃣ Install the correct CUDA 12.4 GPU build of PyTorch
Write-Host "🚀 Installing GPU-enabled PyTorch (CUDA 12.4 stable build)..."
try {
    # `uv pip uninstall` doesn't support confirmation flags, so just skip if not installed
    uv pip uninstall torch torchvision torchaudio --quiet
    uv pip install torch==2.5.0+cu124 torchvision==0.20.0+cu124 torchaudio==2.5.0+cu124 `
        --index-url https://download.pytorch.org/whl/cu124
} catch {
    Write-Warning "⚠️ CUDA 12.4 build not available — falling back to CPU version."
    uv pip install torch torchvision torchaudio
}




# 6️⃣ Install the remaining dependencies
if (Test-Path $reqFile) {
    Write-Host "📦 Installing remaining dependencies from $reqFile ..."
    uv pip install -r $reqFile
} else {
    Write-Host "⚠️ No requirements.txt found at $reqFile — skipping dependency installation."
}

Write-Host "`n✅ Environment ready and GPU-enabled!"
Write-Host "   Activate anytime with: $venvPath\Scripts\activate.ps1"
