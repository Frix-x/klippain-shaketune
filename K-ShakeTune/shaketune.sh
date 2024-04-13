#!/usr/bin/env bash

source ~/klippain_shaketune-env/bin/activate
python ~/klippain_shaketune/src/is_workflow.py "$@"
deactivate
