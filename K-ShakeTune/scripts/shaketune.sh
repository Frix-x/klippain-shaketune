#!/usr/bin/env bash

source ~/klippain_shaketune-env/bin/activate
python ~/klippain_shaketune/K-ShakeTune/scripts/is_workflow.py "$@"
deactivate
