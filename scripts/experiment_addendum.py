"""Addendum to scripts/experiment_pneumo_selection.py.

Part 1: v2 shared-simplex weight distribution by class -> why ascending sort picks class 0.
Part 2: canonical Eq.4/5 on CIFAR-10 round-0 config: is 10-iteration selection converged or ~random?
"""
import os, sys, random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SEED = 0
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
torch.cuda.manual_seed_all(SEED)
device = "cuda:0" if torch.cuda.is_available() else "cpu"

from geomloss import SamplesLoss
loss_func = SamplesLoss("sinkhorn", p=2, blur=0.05, scaling=0.7, backend="tensorized")

# ================= PART 1: v2 shared-simplex class profile (pneumonia) =================
from trust.utils.custom_dataset_medmnist import load_biodataset_custom
from trust.utils.models.resnet import ResNet18
from trust.utils.custom_dataset import LabeledToUnlabeledDataset
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

# quick healthy init (no need to fully retrain; converged-ish is enough for weight profile)
model = ResNet18(num_cls).to(device)
opt0 = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
crit = torch.nn.CrossEntropyLoss()
trainloader = DataLoader(train_set, batch_size=1000, shuffle=True)
model.train()
for ep in range(200):
    for x, y in trainloader:
        x, y = x.to(device), y.to(device, dtype=torch.long)
        opt0.zero_grad(); crit(model(x), y).backward(); opt0.step()

class SubsetQ(torch.utils.data.Dataset):
    def __init__(self, base, idx, targets):
        self.base = base; self.idx = idx; self.targets = targets
    def __getitem__(self, i):
        x, _ = self.base[self.idx[i]]; return x, self.targets[i]
    def __len__(self): return len(self.idx)

vt = torch.Tensor(val_set.targets)
q_idx = []
for c in [0, 1]:
    q_idx += list(torch.where(vt == c)[0].cpu().numpy())
for_query_set = SubsetQ(val_set, q_idx, vt[q_idx].long())
unlabeled_lake_set = LabeledToUnlabeledDataset(lake_set)

print("=" * 70)
print("PART 1: v2 shared simplex (Eq.4 step) weight profile, pneumonia, lr=0.1, 10 iters")
torch.manual_seed(SEED)
strat_args = {"batch_size": 1000, "device": device, "embedding_type": "features",
              "keep_embedding": True, "lr": 0.1, "wassal_iterations": 10,
              "step_size": 10, "min_iteration": 5}
strat = WASSAL_v2(train_set, unlabeled_lake_set, for_query_set, model, num_cls, strat_args)
idxs, soft = strat.select(100)
# get_query_simplex is broken on v2 (AttributeError), so instead re-run Eq.4 inline:
from trust.strategies.wassal_multiclass_v2 import WASSAL_Multiclass as _V2

def shared_simplex_profile(lr, iterations):
    torch.manual_seed(SEED)
    qf = strat._compute_features(for_query_set, "features", "avgpool", None, True).to(device)
    uf = strat._compute_features(unlabeled_lake_set, "features", "avgpool", None, False).to(device)
    classes = torch.stack([item[1] for item in for_query_set]).to(device)
    n = len(unlabeled_lake_set)
    w = (torch.ones(n, device=device) / n).clone().detach().requires_grad_(True)
    opt = optim.Adam([w], lr=lr)
    sched = optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.1)
    for it in range(iterations):
        opt.zero_grad()
        loss = 0.0
        for cls in classes.unique():
            cf = qf[classes == cls]
            beta = torch.ones(cf.size(0), device=device) / cf.size(0)
            loss = loss + loss_func(w, uf, beta, cf)
        loss.backward(); opt.step(); sched.step()
        with torch.no_grad():
            v = w.detach().view(1, -1)
            mu = torch.flip(torch.sort(v, dim=1)[0], dims=(1,))
            cs = torch.cumsum(mu, dim=1)
            j = torch.arange(1, n + 1, dtype=mu.dtype, device=mu.device).unsqueeze(0)
            rho = torch.sum(mu * j - cs + 1 > 0.0, dim=1, keepdim=True) - 1
            theta = (cs[0, rho[0, 0]] - 1) / (rho[0, 0] + 1)
            w.data = torch.clamp(v - theta, min=0.0).view(-1)
    wd = w.detach().cpu().numpy()
    c0, c1 = wd[lake_targets == 0], wd[lake_targets == 1]
    print(f"  lr={lr}: class-0 block: mean={c0.mean():.3e} med={np.median(c0):.3e} max={c0.max():.3e} zeros={np.mean(c0==0)*100:.0f}%")
    print(f"          class-1 block: mean={c1.mean():.3e} med={np.median(c1):.3e} max={c1.max():.3e} zeros={np.mean(c1==0)*100:.0f}%")
    print(f"          total mass: c0={c0.sum():.3f} c1={c1.sum():.3f} (pool shares: 0.25/0.75)")
    # what ascending sort of budget=100 picks:
    order = np.argsort(wd)
    sel = order[:100]
    print(f"          ascending top-100: {np.bincount(lake_targets[sel], minlength=2).tolist()} (c0,c1); idx range {sel.min()}..{sel.max()}")

shared_simplex_profile(0.1, 10)
shared_simplex_profile(0.001, 10)

# ================= PART 2: canonical on CIFAR-10 round-0 =================
print("=" * 70)
print("PART 2: canonical Eq.4/5, CIFAR-10 round-0 (20/cls init), 10 iters, lr=0.001")
import torchvision
import torchvision.transforms as T

SEED2 = 48  # driver seed[0]
torch.manual_seed(SEED2); np.random.seed(SEED2); random.seed(SEED2)

tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
tf_test = T.Compose([T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
full = torchvision.datasets.CIFAR10("data", train=True, download=True, transform=tf)
targets = np.array(full.targets)

rng = np.random.RandomState(0)
train_idx, val_idx, lake_idx = [], [], []
for c in range(10):
    idx_c = np.where(targets == c)[0]
    rng.shuffle(idx_c)
    train_idx += idx_c[:20].tolist()
    val_idx += idx_c[20:30].tolist()
    lake_idx += idx_c[30:4730].tolist()  # 4700 per class -> 47k pool
lake_t = targets[lake_idx]

from trust.utils.utils import SubsetWithTargets
train_set2 = SubsetWithTargets(full, train_idx, torch.Tensor(targets[train_idx]).long())
lake_set2 = SubsetWithTargets(full, lake_idx, torch.Tensor(lake_t).long())
# query set from train (canonical driver uses val; 100 pts either way, balanced 10/class)
val_set2 = SubsetWithTargets(full, val_idx, torch.Tensor(targets[val_idx]).long())

model2 = ResNet18(10).to(device)
opt0 = optim.SGD(model2.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
loader2 = DataLoader(train_set2, batch_size=20, shuffle=True)
model2.train()
for ep in range(120):
    for x, y in loader2:
        x, y = x.to(device), y.to(device, dtype=torch.long)
        opt0.zero_grad(); crit(model2(x), y).backward(); opt0.step()
model2.eval()
corr = tot = 0
with torch.no_grad():
    for x, y in DataLoader(train_set2, batch_size=100):
        p = model2(x.to(device)).max(1)[1].cpu(); corr += p.eq(y).sum().item(); tot += len(y)
print(f"  init train acc (memorization check): {corr/tot:.3f}")

def feats(m, ds, labeled):
    out = []
    with torch.no_grad():
        for batch in DataLoader(ds, batch_size=1000, shuffle=False):
            x = batch[0] if labeled else batch
            e = m(x.to(device), last=True)[1]
            out.append(e.view(x.size(0), -1).cpu())
    return torch.vstack(out).to(device)

def canonical_cifar(lr, iterations, m):
    torch.manual_seed(SEED)
    qf = feats(m, val_set2, True)
    classes = torch.stack([item[1] for item in val_set2]).to(device)
    uf = feats(m, LabeledToUnlabeledDataset(lake_set2), False)
    n = len(lake_set2)
    W = []
    for c in range(10):
        w = (torch.ones(n, device=device) / n).clone().detach().requires_grad_(True)
        opt = optim.Adam([w], lr=lr)
        cf = qf[classes == c]
        beta = torch.ones(cf.size(0), device=device) / cf.size(0)
        for it in range(iterations):
            opt.zero_grad()
            loss_func(w, uf, beta, cf).backward(); opt.step()
            with torch.no_grad():
                v = w.detach().view(1, -1)
                mu = torch.flip(torch.sort(v, dim=1)[0], dims=(1,))
                cs = torch.cumsum(mu, dim=1)
                j = torch.arange(1, n + 1, dtype=mu.dtype, device=mu.device).unsqueeze(0)
                rho = torch.sum(mu * j - cs + 1 > 0.0, dim=1, keepdim=True) - 1
                theta = (cs[0, rho[0, 0]] - 1) / (rho[0, 0] + 1)
                w.data = torch.clamp(v - theta, min=0.0).view(-1)
        W.append(w.detach().cpu())
    maxw = torch.vstack(W).max(0)[0]
    s1 = torch.argsort(maxw)[:100].numpy()
    comp = np.bincount(lake_t[s1], minlength=10)
    # concentration: how peaked is each class simplex?
    l1mass = sum(int((w > 0).sum()) for w in W)
    print(f"  lr={lr}, iters={iterations}: sel per class={comp.tolist()} | nonzero w entries total={l1mass}/47000 | "
          f"maxw med={maxw.median():.2e}")
    # overlap with random selection: expected overlap of two random 100-subsets of 47000 = 100*100/47000 ~= 0.21
    return s1

s1_a = canonical_cifar(0.001, 10, model2)   # driver config
s1_b = canonical_cifar(0.001, 500, model2)  # converged reference
ov = len(set(s1_a.tolist()) & set(s1_b.tolist()))
print(f"  overlap(10-iter, 500-iter) = {ov}/100  (two random picks would share ~0.2)")
