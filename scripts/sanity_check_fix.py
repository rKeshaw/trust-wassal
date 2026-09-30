"""Pre-launch sanity check: canonical WASSAL_Multiclass (trust/strategies/wassal_multiclass.py)
with the FIXED driver args (lr=0.001, 10 iters) on the pneumonia round-0 state.
Verifies: select() dispatch, output tuple structure, selection class composition."""
import os, sys, random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SEED = 0
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
device = "cuda:0" if torch.cuda.is_available() else "cpu"

from trust.utils.custom_dataset_medmnist import load_biodataset_custom
from trust.utils.models.resnet import ResNet18
from trust.utils.custom_dataset import LabeledToUnlabeledDataset
from trust.utils.utils import SubsetWithTargets
from trust.strategies.wassal_multiclass import WASSAL_Multiclass

split_cfg = {
    "sel_cls_idx": [0, 1],
    "per_imbclass_train": {0: 50, 1: 50},
    "per_imbclass_val": {0: 50, 1: 50},
    "per_imbclass_lake": {0: 1000, 1: 3000},
    "per_imbclass_test": {0: 300, 1: 300},
}
train_set, val_set, test_set, lake_set, sel_cls_idx, _ = load_biodataset_custom(
    "data", "pneumoniamnist", "classimb", split_cfg, False, False
)
lake_targets = np.array([int(t) for t in lake_set.targets])

model = ResNet18(2).to(device)
opt0 = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
crit = torch.nn.CrossEntropyLoss()
loader = DataLoader(train_set, batch_size=1000, shuffle=True)
model.train()
for ep in range(200):
    for x, y in loader:
        x, y = x.to(device), y.to(device, dtype=torch.long)
        opt0.zero_grad(); crit(model(x), y).backward(); opt0.step()
model.eval()

# driver-faithful query set: train_set via recipe="asis" (all train pts, targets = labels)
for_query_set = SubsetWithTargets(
    train_set, list(range(len(train_set))), torch.Tensor(train_set.targets).long()
)
strategy_args = {"batch_size": 1000, "device": device, "embedding_type": "features",
                 "keep_embedding": True, "lr": 0.001, "wassal_iterations": 10,
                 "step_size": 10, "min_iteration": 5}
unl = LabeledToUnlabeledDataset(lake_set)
strat = WASSAL_Multiclass(train_set, unl, for_query_set, model, 2, strategy_args)
subset, soft_out = strat.select(100)
comp = np.bincount(lake_targets[np.array(subset)], minlength=2)
print(f"\nSELECTED per class (c0,c1): {comp.tolist()}  | n={len(subset)} unique={len(set(subset))}")
print(f"soft output: {len(soft_out)} tuples; tuple len={len(soft_out[0])}; "
      f"class_idx={[t[2] for t in soft_out]}; q1 nonzero={int((soft_out[0][0]>0).sum())}; "
      f"S1 zeroed in q1: {bool((soft_out[0][0][subset]==0).all())}")
assert len(subset) == 100 and comp[0] > 10 and comp[1] > 10, "SELECTION STILL DEGENERATE"
print("SANITY CHECK PASSED")
