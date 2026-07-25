SKIP_STRATEGIES="random AL AL_WITHSOFT WASSAL"
SKIP_METHODS="WASSAL"
SKIP_BUDGETS="25 50 125 150 175 200"
DEVICE_ID="1"
EXPERIMENT_NAME="onlywassal"
SOFT_LOSS_HYPERPARAM="0.3"
# Call the Python script with the defined arguments
python3 -u tutorials/All_Wassal/wassal_stl10.py "$SKIP_STRATEGIES" "$SKIP_METHODS" "$SKIP_BUDGETS" "$DEVICE_ID" "$EXPERIMENT_NAME" "$SOFT_LOSS_HYPERPARAM" 2>&1 | tee tutorials/results/onlywassal/stl10/stl10wassalsoftB100new.log