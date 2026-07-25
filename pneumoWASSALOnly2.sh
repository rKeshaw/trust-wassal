# Define the skip_strategies and skip_budgets
SKIP_STRATEGIES="random AL WASSAL_WITHSOFT AL_WITHSOFT"
SKIP_METHODS="WASSAL"
SKIP_BUDGETS="70 80 90 100"
DEVICE_ID="1"
EXPERIMENT_NAME="onlywassal"
SOFT_LOSS_HYPERPARAM="0.3"
python3 -u tutorials/All_Wassal/wassal_pneumonia_multiclass_vanilla.py "$SKIP_STRATEGIES" "$SKIP_METHODS" "$SKIP_BUDGETS" "$DEVICE_ID" "$EXPERIMENT_NAME" "$SOFT_LOSS_HYPERPARAM" 2>&1 | tee tutorials/results/onlywassal/pneumowassalonly2.log