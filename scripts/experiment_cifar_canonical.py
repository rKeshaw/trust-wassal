"""Part 2 standalone: canonical Eq.4/5 on CIFAR-10 round-0 — is the 10-iteration
selection converged, seed-stable, and better than random?"""
import os, sys, random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
device = "cuda:0" if torch.cuda.is_available() else "cpu"

from geomloss import SamplesLoss
loss_func = SamplesLoss("sinkhorn", p=2, blur=0.05, scaling=0.7, backend="tensorized")

import torchvision
import torchvision.transforms as T
from trust.utils.models.resnet import ResNet18
from trust.utils.utils import SubsetWithTargets
from trust.utils.custom_dataset import LabeledToUnlabeledDataset

SEED2 = 48
torch.manual_seed(SEED2); np.random.seed(SEED2); random.seed(SEED2)

tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
full = torchvision.datasets.CIFAR10("data", train=True, download=True, transform=tf)
targets = np.array(full.targets)

rng = np.random.RandomState(0)
train_idx, val_idx, lake_idx = [], [], []
for c in range(10):
    idx_c = np.where(targets == c)[0]
    rng.shuffle(idx_c)
    train_idx += idx_c[:20].tolist()
    val_idx += idx_c[20:30].tolist()
    lake_idx += idx_c[30:4730].tolist()
lake_t = targets[lake_idx]

train_set2 = SubsetWithTargets(full, train_idx, torch.Tensor(targets[train_idx]).long())
lake_set2 = SubsetWithTargets(full, lake_idx, torch.Tensor(lake_t).long())
val_set2 = SubsetWithTargets(full, val_idx, torch.Tensor(targets[val_idx]).long())

model2 = ResNet18(10).to(device)
opt0 = optim.SGD(model2.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
crit = torch.nn.CrossEntropyLoss()
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
print(f"init train acc: {corr/tot:.3f}", flush=True)

def feats(ds, labeled):
    out = []
    with torch.no_grad():
        for batch in DataLoader(ds, batch_size=2000, shuffle=False):
            x = batch[0] if labeled else batch
            e = model2(x.to(device), last=True)[1]
            out.append(e.view(x.size(0), -1).cpu())
    return torch.vstack(out).to(device)

qf = feats(val_set2, True)
classes = torch.stack([item[1] for item in val_set2]).to(device)
uf = feats(LabeledToUnlabeledDataset(lake_set2), False)
n = len(lake_set2)
print(f"features: pool {tuple(uf.shape)}, query {tuple(qf.shape)}", flush=True)

def canonical(lr, iterations, torch_seed):
    torch.manual_seed(torch_seed)
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
        W.append(w.detach())
    maxw = torch.vstack(W).max(0)[0]
    return torch.argsort(maxw)[:100].cpu().numpy()

s_10 = canonical(0.001, 10, 0)
s_100 = canonical(0.001, 100, 0)
s_10b = canonical(0.001, 10, 1)
print(f"sel(10 iters)  per-class: {np.bincount(lake_t[s_10], minlength=10).tolist()}", flush=True)
print(f"sel(100 iters) per-class: {np.bincount(lake_t[s_100], minlength=10).tolist()}", flush=True)
print(f"overlap 10 vs 100 iters (same seed): {len(set(s_10.tolist()) & set(s_100.tolist()))}/100", flush=True)
print(f"overlap 10 iters, seed0 vs seed1:    {len(set(s_10.tolist()) & set(s_10b.tolist()))}/100", flush=True)
print(f"baseline: two independent random 100-subsets of 47000 overlap ~= {100*100/47000:.2f}", flush=True)
# round-1 utility proxy: mean distance of selected features to query-class centroids vs random
cent = torch.stack([qf[classes == c].mean(0) for c in range(10)]).cpu()
def dist_to_nearest_centroid(idx):
    x = uf.cpu()[torch.as_tensor(idx)]
    d = torch.cdist(x, cent).min(1)[0]
    return d.mean().item()
rand_idx = rng.choice(n, 100, replace=False)
print(f"mean dist to nearest class centroid: wassal(10it)={dist_to_nearest_centroid(s_10):.4f} "
      f"wassal(100it)={dist_to_nearest_centroid(s_100):.4f} random={dist_to_nearest_centroid(rand_idx):.4f}", flush=True)
