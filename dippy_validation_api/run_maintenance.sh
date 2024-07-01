#!/bin/bash

# Virtual environment name
VENV_NAME="model_validation_venv"

# check if .env file exists
if [ ! -f ../.env ]; then
    echo "Error: .env file does not exist."
    exit 1
fi

# Export environment variables from .env file
export $(grep -v '^#' ../.env | xargs)

# check if ADMIN_KEY is loaded
echo "ADMIN_KEY: $ADMIN_KEY"
echo "DIPPY_KEY: $DIPPY_KEY"

# Start the validation_api
echo "Starting maintenance script"
./../$VENV_NAME/bin/python3 maintenance.py

