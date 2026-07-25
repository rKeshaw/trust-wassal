#!/bin/bash

# Define the skip_strategies and skip_budgets
SKIP_STRATEGIES="AL_WITHSOFT WASSAL WASSAL_WITHSOFT"
SKIP_METHODS="WASSAL"
SKIP_BUDGETS=""
DEVICE_ID="1"
EXPERIMENT_NAME="onlyal"
SOFT_LOSS_HYPERPARAM="0.3"
# Call the Python script with the defined arguments
python3 -u tutorials/All_Wassal/wassal_stl10.py "$SKIP_STRATEGIES" "$SKIP_METHODS" "$SKIP_BUDGETS" "$DEVICE_ID" "$EXPERIMENT_NAME" "$SOFT_LOSS_HYPERPARAM" 2>&1 | tee tutorials/results/onlywassal/stl10/AL/stl10ALnew2.log