#!/bin/bash

# Step 1: Create folder structure
mkdir -p learnai_ready/{api,core,cli,terraform,tests}

# Step 2: Create root-level files
touch README.md
touch setup.py

# Step 3: Create __init__.py files
touch learnai_ready/__init__.py
touch learnai_ready/api/__init__.py
touch learnai_ready/core/__init__.py
touch learnai_ready/cli/__init__.py
touch learnai_ready/terraform/__init__.py
touch learnai_ready/tests/__init__.py
