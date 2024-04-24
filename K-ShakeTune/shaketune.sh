#!/usr/bin/env bash

# This script is used to run the Shake&Tune Python scripts as a module
# from the project root directory using its virtual environment
# Usage: ./shaketune.sh <args>

source ~/klippain_shaketune-env/bin/activate
cd ~/klippain_shaketune
python -m src.is_workflow "$@"
deactivate
