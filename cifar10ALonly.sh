#!/bin/bash

# Define the skip_strategies and skip_budgets
SKIP_STRATEGIES="AL_WITHSOFT WASSAL WASSAL_WITHSOFT"
SKIP_METHODS=""
SKIP_BUDGETS="25 50 100 125 150 175 200"
DEVICE_ID="0"
EXPERIMENT_NAME="onlyal"
SOFT_LOSS_HYPERPARAM="0.3"
# Call the Python script with the defined arguments
python3 -u tutorials/All_Wassal/wassal_cifar10_multiclass_vanilla2.py "$SKIP_STRATEGIES" "$SKIP_METHODS" "$SKIP_BUDGETS" "$DEVICE_ID" "$EXPERIMENT_NAME" "$SOFT_LOSS_HYPERPARAM" 2>&1 | tee tutorials/results/onlyal/cifar10/cifar10onlyalnew.log