"""Controlled experiment: why does PneumoniaMNIST WASSAL select 100% class 0 every round?

Rebuilds the exact round-0 state of tutorials/All_Wassal/wassal_pneumonia_multiclass_vanilla.py:
  - deterministic splits (np.random.seed(42) inside create_class_imb_bio_with_testset)
  - init model: train to >=99% train acc (cap 100 epochs) with the driver's optimizer
Then compares selection methods on the SAME model/split:
  A. v2 shared simplex, ascending sort, lr in {0.1 (driver), 0.001, 0.0001}, 10 iters
  B. canonical per-class simplex (Eq. 4), query = argmin_i max_j w_ji, lr in {0.001, 0.1}, 10 iters
Logs per-class composition of the 100 selected points + weight diagnostics.
Run on a free GPU:  CUDA_VISIBLE_DEVICES=0 python scripts/experiment_pneumo_selection.py
"""
import os, sys, random, math
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SEED = 0
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = "cuda:0" if torch.cuda.is_available() else "cpu"

from trust.utils.custom_dataset_medmnist import load_biodataset_custom
from trust.utils.models.resnet import ResNet18
from trust.strategies.wassal_multiclass_v2 import WASSAL_Multiclass as WASSAL_v2

split_cfg = {
    "sel_cls_idx": [0, 1],
    "per_imbclass_train": {0: 50, 1: 50},
    "per_imbclass_val": {0: 50, 1: 50},
    "per_imbclass_lake": {0: 1000, 1: 3000},
    "per_imbclass_test": {0: 300, 1: 300},
}
num_cls = 2

train_set, val_set, test_set, lake_set, sel_cls_idx, _ = load_biodataset_custom(
    "data", "pneumoniamnist", "classimb", split_cfg, False, False
)
lake_targets = np.array([int(t) for t in lake_set.targets])
print(f"lake size={len(lake_set)}, class-0 positions={np.where(lake_targets==0)[0][:5]}..., "
      f"class-1 first position={np.where(lake_targets==1)[0][0]}, "
      f"counts={np.bincount(lake_targets)}")

# ---- init model: exactly as the driver (99% train acc, cap 100 epochs) ----
model = ResNet18(num_cls).to(device)
optimizer = optim.SGD(model.parameters(), lr=3e-4, momentum=0.9, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()
trainloader = DataLoader(train_set, batch_size=1000, shuffle=True, pin_memory=True)

model.train()
ep = 0; full_acc = 0.0
while full_acc < 0.99 and ep < 100:
    ep += 1
    for x, y in trainloader:
        x, y = x.to(device), y.to(device, dtype=torch.long)
        optimizer.zero_grad(); loss = criterion(model(x), y); loss.backward(); optimizer.step()
    model.eval(); correct = total = 0
    with torch.no_grad():
        for x, y in trainloader:
            x, y = x.to(device), y.to(device, dtype=torch.long)
            p = model(x).max(1)[1]; total += y.size(0); correct += p.eq(y).sum().item()
    full_acc = correct / total
    print(f"init epoch {ep}: train acc {full_acc:.4f}", end="\r")
print(f"\ninit done after {ep} epochs (train acc {full_acc:.4f})")

def per_class_acc(m):
    m.eval()
    pc = np.zeros(num_cls); pt = np.zeros(num_cls)
    with torch.no_grad():
        for x, y in DataLoader(test_set, batch_size=1000):
            p = m(x.to(device)).max(1)[1].cpu().numpy(); y = y.numpy().astype(int)
            for c in range(num_cls):
                mk = (y == c); pt[c] += mk.sum(); pc[c] += (p[mk] == c).sum()
    return (pc / np.maximum(pt, 1) * 100).round(1).tolist()

print("round-0 test per-class acc (driver-faithful init):", per_class_acc(model))

# ---- wrap the strategy exactly like the driver does ----
from trust.utils.custom_dataset import LabeledToUnlabeledDataset
from torch.utils.data import Subset
class SubsetWithTargets4Q(torch.utils.data.Dataset):
    """query dataset: val_set with targets (driver passes for_query_set built from val/train)."""
    def __init__(self, base, idx, targets):
        self.base = base; self.idx = idx; self.targets = targets
    def __getitem__(self, i):
        x, _ = self.base[self.idx[i]]; return x, self.targets[i]
    def __len__(self): return len(self.idx)

val_targets = torch.Tensor(val_set.targets)
q_idx = []
for c in [0, 1]:
    q_idx += list(torch.where(val_targets == c)[0].cpu().numpy())
for_query_set = SubsetWithTargets4Q(val_set, q_idx, val_targets[q_idx].long())

unlabeled_lake_set = LabeledToUnlabeledDataset(lake_set)

def class_composition(idxs):
    sel = lake_targets[np.array(idxs)]
    return np.bincount(sel, minlength=num_cls).tolist()

def run_v2(lr):
    torch.manual_seed(SEED)
    strat_args = {"batch_size": 1000, "device": device, "embedding_type": "features",
                  "keep_embedding": True, "lr": lr, "wassal_iterations": 10,
                  "step_size": 10, "min_iteration": 5}
    strat = WASSAL_v2(train_set, unlabeled_lake_set, for_query_set, model, num_cls, strat_args)
    idxs, soft = strat.select(100)
    comp = class_composition(idxs)
    idx_arr = np.array(idxs)
    print(f"[v2  lr={lr:<7}] sel per class (c0,c1) = {comp}   idx: min={idx_arr.min()} max={idx_arr.max()} #in[0,999]={int((idx_arr<1000).sum())}")
    return comp

print("\n=== CONDITION A: driver-faithful init (undertrained, class-biased) ===")
run_v2(0.1)      # driver setting
run_v2(0.001)
run_v2(0.0001)
run_v2(0.0)      # pure tie-break baseline: weights never move

# ---- canonical Eq.4/5 selection on the same model (per-class simplexes, min-max rule) ----
from geomloss import SamplesLoss
loss_func = SamplesLoss("sinkhorn", p=2, blur=0.05, scaling=0.7, backend="tensorized")

@torch.no_grad()
def _feats(dataset, labeled):
    out = []
    for batch in DataLoader(dataset, batch_size=1000, shuffle=False):
        x = batch[0] if labeled else batch
        e = model(x.to(device), last=True)[1]
        out.append(e.view(x.size(0), -1).cpu())
    return torch.vstack(out).to(device)

def run_canonical(lr):
    torch.manual_seed(SEED)
    m = model
    qf = _feats(for_query_set, labeled=True)
    classes = torch.stack([item[1] for item in for_query_set]).to(device)
    uf = _feats(unlabeled_lake_set, labeled=False)
    n = len(unlabeled_lake_set)
    W = []
    for c in range(num_cls):
        w = (torch.ones(n, device=device) / n).clone().detach().requires_grad_(True)
        opt = optim.Adam([w], lr=lr)
        cf = qf[classes == c]
        beta = torch.ones(cf.size(0), device=device) / cf.size(0)
        for it in range(10):
            opt.zero_grad()
            loss = loss_func(w, uf, beta, cf)
            loss.backward(); opt.step()
            with torch.no_grad():
                # project to simplex
                v = w.detach().view(1, -1)
                mu = torch.flip(torch.sort(v, dim=1)[0], dims=(1,))
                cs = torch.cumsum(mu, dim=1)
                j = torch.arange(1, n + 1, dtype=mu.dtype, device=mu.device).unsqueeze(0)
                rho = torch.sum(mu * j - cs + 1 > 0.0, dim=1, keepdim=True) - 1
                theta = (cs[0, rho[0, 0]] - 1) / (rho[0, 0] + 1)
                w.data = torch.clamp(v - theta, min=0.0).view(-1)
        W.append(w.detach())
    maxw = torch.vstack(W).max(0)[0]
    s1 = torch.argsort(maxw)[:100].cpu().numpy()          # Eq. 5: min over max_j
    comp = class_composition(s1)
    print(f"[canon lr={lr:<7}] sel per class (c0,c1) = {comp}   maxw: min={maxw.min():.2e} p50={maxw.median():.2e} max={maxw.max():.2e}")
    return comp

print("\n=== canonical on driver-faithful init ===")
run_canonical(0.001)
run_canonical(0.1)

# ---- CONDITION B: converged init (99% train acc, healthy lr) ----
print("\n=== CONDITION B: converged init ===")
model2 = ResNet18(num_cls).to(device)
opt2 = optim.SGD(model2.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
model2.train(); ep = 0; full_acc = 0.0
while full_acc < 0.99 and ep < 200:
    ep += 1
    for x, y in trainloader:
        x, y = x.to(device), y.to(device, dtype=torch.long)
        opt2.zero_grad(); loss = criterion(model2(x), y); loss.backward(); opt2.step()
    model2.eval(); correct = total = 0
    with torch.no_grad():
        for x, y in trainloader:
            x, y = x.to(device), y.to(device, dtype=torch.long)
            p = model2(x).max(1)[1]; total += y.size(0); correct += p.eq(y).sum().item()
    full_acc = correct / total
print(f"converged init: {ep} epochs, train acc {full_acc:.4f}, test per-class {per_class_acc(model2)}")
model = model2

run_v2(0.1)
run_v2(0.001)
run_v2(0.0)
print("\n=== canonical on converged init ===")
run_canonical(0.001)
run_canonical(0.1)
