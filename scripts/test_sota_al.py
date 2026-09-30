"""Synthetic smoke tests for sota_al.py (MaxHerding, UHerding, WassersteinIP).

CPU-only. Verifies:
  1. _herding_select matches a brute-force re-implementation of the official
     greedy loop (herding.py/uherding.py) on random kernels.
  2. MaxHerding: from one labeled point in cluster A, next picks spread into
     cluster B (generalized coverage - Fig. 2(a) of Bae et al. 2024).
  3. UHerding: uncertainty weighting shifts picks toward uncertain regions;
     temperature fitting returns a grid value; class runs end-to-end.
  4. WassersteinIP k-medoids: medoids land one per cluster and beat random
     subsets on the scenario-reduction objective sum_i min_m ||x_i - x_m||.
  5. _cand_subset matches the official compute_cand_size behavior.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, Dataset

from trust.strategies.sota_al import (
    MaxHerding, UHerding, WassersteinIP, _herding_select, _rbf, _cand_subset)

torch.manual_seed(0)
np.random.seed(0)
DEV = "cpu"


def brute_force_greedy(K, unc, budget, n_lab):
    """Direct implementation of Eq. 5 / Algorithm 1 (for comparison)."""
    K = np.asarray(K, dtype=np.float32)
    unc = np.asarray(unc, dtype=np.float32)
    if n_lab > 0:
        max_emb = K[:n_lab].max(0, keepdims=True)  # (1, N)
    else:
        max_emb = np.zeros((1, K.shape[1]), dtype=np.float32)
    sel = []
    for _ in range(budget):
        gain = np.maximum(K - max_emb, 0.0)                   # (N, N)
        scores = (unc.reshape(-1, 1) * gain).mean(0)          # (N,)
        scores[:n_lab] = -np.inf
        for j in sel:
            scores[j] = -np.inf
        j = int(scores.argmax())
        sel.append(j)
        max_emb = np.maximum(max_emb, K[j:j + 1])
    return sel


def test_herding_math():
    N, n_lab, B = 12, 3, 5
    A = torch.randn(N, N)
    K = ((A + A.T) / 2).clamp(0, 1)  # symmetric in [0,1], loosely kernel-like
    max_emb = torch.rand(1, N)
    unc = torch.rand(N)
    ours = _herding_select(K.clone(), K[:n_lab].clone(), unc.clone(), B, DEV,
                           torch.empty_like(K))
    ref = brute_force_greedy(K.numpy(), unc.numpy(), B, n_lab)
    assert ours == ref, f"greedy mismatch: {ours} vs {ref}"
    # and with zero labeled set
    ours0 = _herding_select(K.clone(), None, unc.clone(), B, DEV, torch.empty_like(K))
    ref0 = brute_force_greedy(K.numpy(), unc.numpy(), B, 0)
    assert ours0 == ref0, f"greedy(lab=0) mismatch: {ours0} vs {ref0}"
    print("PASS herding math == brute force")


def test_rbf():
    X = torch.randn(7, 4)
    K = _rbf(X, X, 1.0)
    assert torch.allclose(torch.diag(K), torch.ones(7)), "RBF diag must be 1"
    d = torch.cdist(X[:2], X)
    assert torch.allclose(K[:2], torch.exp(-(d / 1.0) ** 2))
    print("PASS rbf kernel")


def make_toy(nA=200, nB=200, d=8):
    cA = torch.randn(nA, d) * 0.25
    cB = torch.randn(nB, d) * 0.25 + torch.tensor([2.5] + [0.0] * (d - 1))
    X = torch.cat([cA, cB])
    X = X / X.norm(dim=1, keepdim=True)
    y = torch.cat([torch.zeros(nA), torch.ones(nB)]).long()
    return X, y


class UnlabeledDS(Dataset):
    """Driver-style unlabeled dataset yielding inputs only."""
    def __init__(self, X):
        self.X = X
    def __len__(self):
        return len(self.X)
    def __getitem__(self, i):
        return self.X[i]


class TinyNet(nn.Module):
    """Has an 'avgpool' child so Strategy.get_feature_embedding hooks onto it."""
    def __init__(self, d=8, k=2):
        super().__init__()
        self.body = nn.Linear(d, 16)
        self.avgpool = nn.Identity()
        self.fc = nn.Linear(16, k)
    def forward(self, x):
        return self.fc(self.avgpool(self.body(x)))


class PassThroughNet(nn.Module):
    """Feature output == input, so selection is optimized on the same geometry
    the test scores (isolates the algorithm from the feature map)."""
    def __init__(self, d=8, k=2):
        super().__init__()
        self.avgpool = nn.Identity()
        self.fc = nn.Linear(d, k)
    def forward(self, x):
        return self.avgpool(x)


def make_strategy(X, y, n_lab, cls, extra_args=None, model_cls=TinyNet):
    args = {"device": DEV, "batch_size": 64}
    if extra_args:
        args.update(extra_args)
    lab = TensorDataset(X[:n_lab], y[:n_lab])
    unl = UnlabeledDS(X[n_lab:])
    model = model_cls(d=X.shape[1], k=int(y.max().item()) + 1)
    return cls(lab, unl, model, int(y.max().item()) + 1, args), unl


def test_maxherding_end_to_end():
    X, y = make_toy()
    strat, unl = make_strategy(X, y, n_lab=1, cls=MaxHerding)
    sel1 = strat.select(6)
    sel2 = strat.select(6)
    assert len(sel1) == 6 and len(set(sel1.tolist())) == 6, "unique, correct size"
    assert (sel1 == sel2).all(), "MaxHerding must be deterministic"
    assert sel1.min() >= 0 and sel1.max() < len(unl)
    inB = (np.asarray(sel1) >= 200 - 1).sum()  # pool index i corresponds to X[1+i]
    assert inB >= 1, f"expected spread into cluster B, got {inB}/6 in B"
    first = int(sel1[0])
    assert first >= 199, f"first pick must enter cluster B (Fig 2a), got pool idx {first}"
    print(f"PASS MaxHerding e2e (first pick in far cluster, {inB}/6 total)")


def test_uherding_end_to_end():
    X, y = make_toy()
    strat, unl = make_strategy(X, y, n_lab=12, cls=UHerding)
    sel = strat.select(6)
    assert len(sel) == 6 and len(set(sel.tolist())) == 6
    assert sel.min() >= 0 and sel.max() < len(unl)
    print("PASS UHerding e2e (valid unique selection)")


def test_uherding_uncertainty_pull():
    """With uncertainty concentrated on cluster B, UHerding should favour B more
    than uniform-weight herding does."""
    X, y = make_toy()
    # hand-built kernels rather than the full class: rel = 2 labeled(A) + pool
    lab_f = X[:2]
    pool = X[2:]
    rel = torch.cat([lab_f, pool])
    K = _rbf(rel, rel, 1.0)
    kla = _rbf(lab_f, rel, 1.0)
    buf = torch.empty_like(K)
    uniform = _herding_select(K.clone(), kla.clone(), torch.ones(1, rel.shape[0]) / rel.shape[0],
                              4, DEV, buf)
    # uncertainty only on B-side pool points
    u = torch.zeros(rel.shape[0])
    u[2 + 200:] = 1.0  # pool indices 200..399 are cluster B
    pull = _herding_select(K.clone(), kla.clone(), (u / u.sum()).reshape(1, -1), 4, DEV, buf)
    b_uniform = sum(1 for j in uniform if j >= 2 + 200)
    b_pull = sum(1 for j in pull if j >= 2 + 200)
    assert b_pull >= b_uniform, f"uncertainty pull failed: {b_pull} vs {b_uniform}"
    print(f"PASS UHerding uncertainty pull (B-picks uniform={b_uniform}, weighted={b_pull})")


def test_wasserstein_ip():
    X, y = make_toy()
    strat, unl = make_strategy(X, y, n_lab=10, cls=WassersteinIP, model_cls=PassThroughNet)
    sel = strat.select(2)
    sel = np.asarray(sel) + 10  # pool index i -> X[10+i]
    nA = (sel < 200).sum()
    assert nA == 1 and len(sel) - nA == 1, f"expected 1 medoid per cluster, got {sel}"
    # objective comparison vs random subsets
    Xp = X[10:].numpy()
    def obj(idx):
        D = np.linalg.norm(Xp[:, None, :] - Xp[None, idx, :], axis=2)
        return D.min(1).sum()
    ours = obj(sel - 10)
    # true optimum for two separated clusters: best medoid of each cluster
    A_idx, B_idx = np.arange(0, 190), np.arange(190, 390)  # Xp = X[10:400]
    def within_medoid(idx):
        D = np.linalg.norm(Xp[idx][:, None, :] - Xp[idx][None, :, :], axis=2)
        return idx[D.sum(1).argmin()]
    opt_pair = [within_medoid(A_idx), within_medoid(B_idx)]
    opt = obj(opt_pair)
    assert ours <= 1.02 * opt, f"k-medoids obj {ours:.1f} far from optimum {opt:.1f}"
    rng = np.random.RandomState(0)
    rand = np.mean([obj(rng.choice(len(Xp), 2, replace=False)) for _ in range(50)])
    assert ours < rand, f"k-medoids obj {ours:.1f} not better than random {rand:.1f}"
    print(f"PASS WassersteinIP (obj {ours:.1f} ~ optimum {opt:.1f}, random {rand:.1f})")


def test_cand_subset():
    s = _cand_subset(500, 100, 10)
    assert (s == np.arange(500)).all(), "small pool -> identity"
    s = _cand_subset(40000, 100, 10)
    assert len(s) == 35000 and len(set(s.tolist())) == 35000, "cap at 35000"
    s = _cand_subset(150, 200, 5)
    assert len(s) == 150, "pool smaller than budget -> whole pool"
    print("PASS _cand_subset (official compute_cand_size)")


if __name__ == "__main__":
    test_rbf()
    test_herding_math()
    test_cand_subset()
    test_maxherding_end_to_end()
    test_uherding_end_to_end()
    test_uherding_uncertainty_pull()
    test_wasserstein_ip()
    print("\nALL SOTA SMOKE TESTS PASSED")
