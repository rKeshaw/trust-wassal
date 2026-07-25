#!/bin/bash

# Define the skip_strategies and skip_budgets
SKIP_STRATEGIES="random AL AL_WITHSOFT WASSAL"
SKIP_METHODS="WASSAL_WITHSOFT"
SKIP_BUDGETS=""
DEVICE_ID="0"
EXPERIMENT_NAME="onlywassal"
SOFT_LOSS_HYPERPARAM="0.3"
# Call the Python script with the defined arguments
python3 -u tutorials/All_Wassal/wassal_cifar10_multiclass_vanilla2.py "$SKIP_STRATEGIES" "$SKIP_METHODS" "$SKIP_BUDGETS" "$DEVICE_ID" "$EXPERIMENT_NAME" "$SOFT_LOSS_HYPERPARAM" 2>&1 | tee tutorials/results/onlywassal/cifar10/cifar10wassalsoftnew.log