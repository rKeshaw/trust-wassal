"""SOTA baselines (2022-2025) for comparison with WASSAL.

  - MaxHerding    (Bae et al., ECCV 2024) - greedy maximization of generalized
    coverage (kernel herding with a "max kernel"). Faithful port of the official
    implementation in BorealisAI/uherding, deep-al/pycls/al/herding.py.
  - UHerding      (Bae et al., ICLR 2025) - uncertainty-weighted herding with
    temperature-calibrated margin uncertainty and adaptive kernel bandwidth
    (sigma = min pairwise labeled distance). Port of official
    BorealisAI/uherding, deep-al/pycls/al/uherding.py (Algorithm 1 of the paper).
  - WassersteinIP (Mahmood et al., ICLR 2022) - no official code was released.
    We implement the paper's own practical heuristic: k-medoids scenario
    reduction (Heitsch & Romisch 2003) on frozen features, i.e. choose the
    budget-sized subset minimizing sum_i min_m ||x_i - x_m|| (the empirical W1
    between pool and coreset). The paper's exact method is a MILP solved by
    Generalized Benders Decomposition (~3-6 h/selection), infeasible for our
    3-seed x budget x dataset grid and without POT/CBC in this environment.

All operate on the current model's L2-normalized avgpool features (task-model
feature regime - the same adaptation used by our other baselines and by the
official repos' 'classifier'-feature configurations).

Deviations from the official protocol (documented in AUDIT_REPORT.md):
  - candidate pools are subsampled to <= 35k points with a fixed seed (official
    code caps candidates at 35k via compute_cand_size but uses the global RNG);
  - UHerding's temperature is fit by minimizing ECE on a 30% held-out split of
    the labeled set using the current round model, instead of retraining a
    fresh model on 70% of the labels (protocol consistency with our other
    baselines; same objective).
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .strategy import Strategy
from .modern_al import _features, _random_fill, _kmeans_torch

_CAND_MAX = 35000  # official compute_cand_size cap


def _cand_subset(n_pool, budget, n_labeled, seed=0):
    """Official candidate-subset sizing (pycls/utils/io.py compute_cand_size)."""
    ub = 45000 * (_CAND_MAX + 10000)
    cand = int((ub + (n_labeled + budget) ** 2 / 4) ** 0.5 - 1.5 * (n_labeled + budget))
    cand = max(min(_CAND_MAX, cand), budget)
    if n_pool <= cand:
        return np.arange(n_pool)
    return np.sort(np.random.RandomState(seed).permutation(n_pool)[:cand])


def _rbf(XA, XB, h, block=512):
    """Official RBFKernel: exp(-(d/h)^2), computed in row blocks."""
    rows = []
    for i in range(0, XA.shape[0], block):
        d = torch.cdist(XA[i:i + block], XB)
        rows.append(torch.exp(-1.0 * (d / h) ** 2))
    return torch.cat(rows, dim=0)


def _herding_select(K_all, kernel_la, unc, budget, device, buf):
    """Official greedy loop (herding.py/uherding.py select_samples).

    scores_j = mean_n(unc_n * max(K[n,j] - max_emb_n, 0)); pick argmax over
    unlabeled candidates; max_emb <- max(max_emb, K[:,j]) via the incremental
    update max_emb += clamp(K[:,j] - max_emb, 0). `unc` is pre-normalized so
    that unc @ buf equals the official (unc * buf).mean(-1).
    """
    N = K_all.shape[0]
    if kernel_la is not None:
        max_emb = kernel_la.max(dim=0, keepdim=True).values  # (1, N)
    else:
        max_emb = torch.zeros(1, N, device=device)
    w = unc.reshape(1, N)  # already divided by N
    selectable = torch.ones(N, dtype=torch.bool, device=device)
    if kernel_la is not None:
        selectable[:kernel_la.shape[0]] = False
    selected = []
    for _ in range(budget):
        if not selectable.any():
            break
        torch.sub(K_all, max_emb, out=buf)
        buf.clamp_min_(0)
        scores = torch.mm(w, buf).squeeze(0)
        scores[~selectable] = -1.0
        j = int(scores.argmax())
        selected.append(j)
        selectable[j] = False
        max_emb += buf[j].unsqueeze(0)
    return selected


def _pool_logits(strat, device, batch_size=1000):
    """Model logits over [labeled..., unlabeled...] in order (official computes
    uncertainties over relevant_indices = [lSet, uSet])."""
    strat.model.eval()
    outs = []
    with torch.no_grad():
        for ds in (strat.labeled_dataset, strat.unlabeled_dataset):
            for batch in DataLoader(ds, batch_size=batch_size, shuffle=False):
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                outs.append(strat.model(x.to(device)).cpu())
    return torch.cat(outs, dim=0).to(device)


def _fit_temperature(strat, device, grid=(0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0)):
    """Temperature minimizing ECE on a 30% held-out split of the labeled set
    (official train_al.py: temp_valSet = lSet[-num_temp_vSet:], num = 30%)."""
    y = torch.tensor([int(item[1]) for item in strat.labeled_dataset], device=device)
    logits = _pool_logits(strat, device)[:len(y)]
    n_val = max(1, int(0.3 * len(y)))
    val_logits, val_y = logits[-n_val:], y[-n_val:]

    def ece(t):
        p = F.softmax(val_logits / t, dim=1)
        conf, pred = p.max(1)
        acc = (pred == val_y).float()
        bins = torch.linspace(0, 1, 16, device=device)
        e = 0.0
        for b in range(15):
            m = (conf > bins[b]) & (conf <= bins[b + 1])
            if m.any():
                e += m.float().mean().item() * abs(acc[m].mean().item() - conf[m].mean().item())
        return e

    return min(grid, key=ece)


# ------------------------------------------------------------------ MaxHerding
class MaxHerding(Strategy):
    """Greedy generalized-coverage maximization with RBF kernel (h=1.0 default,
    robust on normalized features - Bae et al. 2024, Sec 3.2). Uncertainty is
    uniform (official herding.py uses ones over labeled+pool)."""

    def select(self, budget):
        device = self.device
        uf = _features(self, self.unlabeled_dataset, device, labeled=False)
        lf = _features(self, self.labeled_dataset, device, labeled=True)
        cand = _cand_subset(uf.shape[0], budget, lf.shape[0])
        rel = torch.cat([lf, uf[cand]], dim=0) if lf.shape[0] else uf[cand]

        h = 1.0
        K_all = _rbf(rel, rel, h)
        kernel_la = _rbf(lf, rel, h) if lf.shape[0] else None
        unc = torch.ones(1, rel.shape[0], device=device) / rel.shape[0]
        buf = torch.empty_like(K_all)

        sel = _herding_select(K_all, kernel_la, unc, budget, device, buf)
        pool_idx = cand[np.array(sel) - lf.shape[0]]
        return _random_fill(list(pool_idx), budget, uf.shape[0])


# -------------------------------------------------------------------- UHerding
class UHerding(Strategy):
    """Uncertainty Herding (Bae et al., ICLR 2025, Algorithm 1): herding greedy
    loop with margin uncertainty U = 1 - (p1 - p2) of the temperature-scaled
    softmax; temperature minimizes ECE on a labeled val split; kernel bandwidth
    sigma* = min pairwise distance within the labeled set (adaptive delta)."""

    def select(self, budget):
        device = self.device
        uf = _features(self, self.unlabeled_dataset, device, labeled=False)
        lf = _features(self, self.labeled_dataset, device, labeled=True)
        cand = _cand_subset(uf.shape[0], budget, lf.shape[0])
        rel = torch.cat([lf, uf[cand]], dim=0) if lf.shape[0] else uf[cand]

        # adaptive delta (official uherding.py: min labeled pairwise distance)
        h = 1.0
        if lf.shape[0] > 1:
            d = torch.cdist(lf, lf)
            d_tril = torch.tril(d, diagonal=-1)
            pos = d_tril[d_tril > 0]
            if pos.numel():
                h = max(pos.min().item(), 1e-6)

        K_all = _rbf(rel, rel, h)
        kernel_la = _rbf(lf, rel, h) if lf.shape[0] else None

        # margin uncertainty with temperature-scaled softmax (official: UNC over
        # relevant indices, zeros on labeled points)
        n_lab = lf.shape[0]
        if n_lab > 0:
            temp = _fit_temperature(self, device)
            logits = _pool_logits(self, device)
            n_rel = rel.shape[0]
            cand_t = torch.from_numpy(cand).to(logits.device)
            rel_logits = torch.cat([logits[:n_lab], logits[n_lab:][cand_t]])
            p = F.softmax(rel_logits / temp, dim=1)
            top2 = p.topk(2, dim=1).values
            u = 1.0 - (top2[:, 0] - top2[:, 1])
            u[:n_lab] = 0.0
        else:
            u = torch.ones(rel.shape[0], device=device)
        unc = u.reshape(1, -1) / rel.shape[0]
        buf = torch.empty_like(K_all)

        sel = _herding_select(K_all, kernel_la, unc, budget, device, buf)
        pool_idx = cand[np.array(sel) - n_lab]
        return _random_fill(list(pool_idx), budget, uf.shape[0])


# --------------------------------------------------------------- Wasserstein IP
class WassersteinIP(Strategy):
    """Mahmood et al. (ICLR 2022) Wasserstein-IP via the paper's k-medoids
    scenario-reduction heuristic on features: minimize sum_i min_m ||x_i - x_m||
    (uniform-weight empirical W1 between pool and coreset). Deterministic
    farthest-first seeding + Voronoi swap refinement (PAM-style local search)."""

    def select(self, budget):
        device = self.device
        uf = _features(self, self.unlabeled_dataset, device, labeled=False)
        cand = _cand_subset(uf.shape[0], budget, len(self.labeled_dataset))
        X = uf[cand]
        n, B = X.shape[0], min(budget, X.shape[0])

        # Seed medoids from k-means assignments (central, deterministic; avoids
        # the peripheral-point local optima of farthest-first seeding), then
        # Voronoi refinement (alternating k-medoids).
        assign0, _, _ = _kmeans_torch(X, B, seed=0, device=device)
        medoids = []
        for c in range(B):
            members = torch.where(assign0 == c)[0]
            if members.numel() == 0:
                medoids.append(None)
                continue
            dd = torch.cdist(X[members], X[members])
            medoids.append(int(members[dd.sum(1).argmin()]))
        if any(m is None for m in medoids):  # degenerate seeding fallback
            col_sum = torch.zeros(n, device=device)
            for i in range(0, n, 512):
                col_sum[i:i + 512] = torch.cdist(X[i:i + 512], X).sum(0)
            g = int(col_sum.argmin())
            medoids = [m if m is not None else g for m in medoids]
        prev_sum = torch.cdist(X, X[medoids]).min(1).values.sum().item()

        # Voronoi swap refinement (alternating k-medoids)
        for _ in range(8):
            assign = torch.empty(n, dtype=torch.long, device=device)
            best_d = torch.full((n,), float("inf"), device=device)
            for c, m in enumerate(medoids):
                d = torch.cdist(X, X[m:m + 1]).squeeze(1)
                closer = d < best_d
                best_d[closer] = d[closer]
                assign[closer] = c
            new_medoids = []
            for c in range(len(medoids)):
                members = torch.where(assign == c)[0]
                if members.numel() == 0:
                    new_medoids.append(medoids[c])
                    continue
                dd = torch.cdist(X[members], X[members])
                new_medoids.append(int(members[dd.sum(1).argmin()]))
            if set(new_medoids) == set(medoids):
                medoids = new_medoids
                break
            medoids = new_medoids
            d_min = torch.cdist(X, X[medoids]).min(1).values
            cur = d_min.sum().item()
            if cur >= prev_sum - 1e-9:
                break
            prev_sum = cur

        return _random_fill(list(cand[np.array(medoids)]), budget, uf.shape[0])
