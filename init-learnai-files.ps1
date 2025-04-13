# init-learnai-files.ps1

# Step 1: Create root-level files
New-Item -ItemType File -Path "README.md" -Force | Out-Null
New-Item -ItemType File -Path "setup.py" -Force | Out-Null

# Step 2: Create __init__.py in each directory
$initFiles = @(
    "learnai_ready/__init__.py",
    "learnai_ready/api/__init__.py",
    "learnai_ready/core/__init__.py",
    "learnai_ready/cli/__init__.py",
    "learnai_ready/terraform/__init__.py",
    "learnai_ready/tests/__init__.py"
)

foreach ($file in $initFiles) {
    $folder = Split-Path $file
    if (-not (Test-Path $folder)) {
        New-Item -ItemType Directory -Path $folder -Force | Out-Null
    }
    New-Item -ItemType File -Path $file -Force | Out-Null
}
