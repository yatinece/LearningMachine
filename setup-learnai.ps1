# setup-learnai.ps1

# Step 1: Define the folder structure
$basePath = "learnai_ready"
$subfolders = "api", "core", "cli", "terraform", "tests"

# Step 2: Create base directory and subdirectories
foreach ($folder in $subfolders) {
    $fullPath = Join-Path -Path $basePath -ChildPath $folder
    New-Item -ItemType Directory -Path $fullPath -Force | Out-Null
}

# Step 3: Change directory to the project folder
Set-Location -Path $basePath

# Step 4: (Optional) Create virtual environment
# python -m venv venv

# Step 5: (Optional) Activate virtual environment
# & .\venv\Scripts\Activate.ps1

# Step 6: Install dependencies
pip install fastapi uvicorn pydantic pyyaml typer python-multipart


##.\setup-learnai.ps1
