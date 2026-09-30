"""Modern deep-active-learning baselines (2022-2024) for comparison with WASSAL.

Implementations adapted to this repo's Strategy interface:
  - TypiClust    (Hacohen et al., ICML 2022)          - low-budget SOTA, typicality clustering
  - ProbCover    (Yehuda et al., NeurIPS 2022)        - graph coverage on embeddings
  - DCoM         (Mishal & Weinshall, arXiv:2407.01804, 2024) - diversity (clustering) + confidence
  - ALFA-Margin  (internal baseline, no verified citation)    - k-means + margin pick

The DCoM entry previously carried a wrong attribution (a WACV 2023 reference);
the arXiv id above is the DCoM preprint. ALFA-Margin is kept only as a code
baseline: no verifiable paper describing exactly this rule was found, so it is
not cited in the paper (an earlier docstring listed arXiv:2407.12212, which is
the id of the MaxHerding paper, not ALFA-Margin).

All operate on the current model's avgpool features (recomputed each round), which
is the standard adaptation when no self-supervised backbone is available. The
official TypiClust/ProbCover repos also use task-model features when no SSL
backbone is present (e.g. their RA-none configurations).
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .strategy import Strategy


# ----------------------------------------------------------------- utilities
@torch.no_grad()
def _features(strat, dataset, device, labeled, layer="avgpool"):
    """L2-normalized avgpool features via the base Strategy helper.
    The base get_feature_embedding wraps the dataset in a DataLoader itself and
    selects its tuple-unpacking branch from the `unlabeled` flag, so pass the
    dataset object through unchanged."""
    f = strat.get_feature_embedding(dataset, unlabeled=not labeled, layer_name=layer)
    f = f.view(f.shape[0], -1).to(device)
    return F.normalize(f, dim=1)


def _kmeans_torch(X, K, iters=30, seed=0, device="cuda:0"):
    """Plain k-means (k-means++ init, Lloyd iterations) in torch. X: (N,D) on device."""
    N = X.shape[0]
    K = min(K, N)
    g = torch.Generator(device="cpu").manual_seed(seed)
    centers = torch.empty(K, X.shape[1], device=device)
    first = torch.randint(N, (1,), generator=g).item()
    centers[0] = X[first]
    d2 = torch.cdist(X, centers[0:1]).squeeze(1) ** 2
    for k in range(1, K):
        probs = (d2 / d2.sum().clamp_min(1e-12)).cpu()
        idx = torch.multinomial(probs, 1, generator=g).item()
        centers[k] = X[idx]
        d2 = torch.minimum(d2, torch.cdist(X, centers[k:k + 1]).squeeze(1) ** 2)
    for _ in range(iters):
        assign = torch.cdist(X, centers).argmin(1)
        for k in range(K):
            m = assign == k
            if m.any():
                centers[k] = X[m].mean(0)
    dists = torch.cdist(X, centers)
    assign = dists.argmin(1)
    return assign, centers, dists


def _random_fill(selected, budget, n, seed=0):
    if len(selected) < budget:
        rest = [i for i in range(n) if i not in set(selected)]
        rng = np.random.RandomState(seed)
        selected += rng.choice(rest, budget - len(selected), replace=False).tolist()
    return np.array(selected[:budget])


# ----------------------------------------------------------------- TypiClust
class TypiClust(Strategy):
    """Typicality clustering: cluster the pool into K = |labeled| + budget clusters,
    then query the most typical (centroid-closest) point of each largest uncovered
    cluster (uncovered = cluster containing no already-labeled point)."""

    def select(self, budget):
        device = self.device
        uf = _features(self, self.unlabeled_dataset, device, labeled=False)
        lf = _features(self, self.labeled_dataset, device, labeled=True)
        n_lab = len(self.labeled_dataset)
        K = n_lab + budget
        assign, centers, dists = _kmeans_torch(uf, K, seed=0, device=device)

        covered = set()
        if n_lab > 0:
            lab_assign = torch.cdist(lf, centers).argmin(1)
            covered = set(lab_assign.cpu().tolist())

        cluster_sizes = torch.bincount(assign, minlength=K)
        uncovered = [c for c in range(K) if c not in covered]
        uncovered.sort(key=lambda c: -int(cluster_sizes[c]))

        selected = []
        for c in uncovered:
            members = torch.where(assign == c)[0]
            if members.numel() == 0:
                continue
            d = dists[members, c]
            typical = members[d.argmin()].item()
            if typical not in selected:
                selected.append(typical)
            if len(selected) == budget:
                break
        return _random_fill(selected, budget, uf.shape[0])


# ----------------------------------------------------------------- ProbCover
class ProbCover(Strategy):
    """Greedy max-coverage on a graph over normalized embeddings; delta is set from
    labeled-data purity (90th percentile of same-class labeled pairwise distances)."""

    def select(self, budget):
        device = self.device
        uf = _features(self, self.unlabeled_dataset, device, labeled=False)
        lf = _features(self, self.labeled_dataset, device, labeled=True)
        # label extraction robust to int or tensor labels across dataset wrappers
        y_lab = torch.tensor([int(item[1]) for item in self.labeled_dataset], device=device)

        delta = 0.5
        if len(y_lab) > 1:
            d = torch.cdist(lf, lf)
            same = (y_lab.unsqueeze(0) == y_lab.unsqueeze(1)) & \
                   (~torch.eye(len(y_lab), dtype=torch.bool, device=device))
            if same.any():
                delta = d[same].quantile(0.9).item()

        covered = torch.zeros(uf.shape[0], dtype=torch.bool, device=device)
        covered |= (torch.cdist(lf, uf) < delta).any(0)

        selected = []
        for _ in range(budget):
            cand = torch.where(~covered)[0]
            if cand.numel() == 0:
                break
            best, best_gain = -1, -1
            for s in range(0, cand.numel(), 4096):
                chunk = cand[s:s + 4096]
                gain = (torch.cdist(uf[chunk], uf[~covered]) < delta).sum(1)
                j = int(gain.argmax())
                if int(gain[j]) > best_gain:
                    best_gain, best = int(gain[j]), int(chunk[j])
            selected.append(best)
            covered |= (torch.cdist(uf[best:best + 1], uf).squeeze(0) < delta)
        return _random_fill(selected, budget, uf.shape[0])


# --------------------------------------------------------------------- DCoM
class DCoM(Strategy):
    """Diversity + Confidence: cluster the pool into `budget` clusters and pick the
    most confident (minimum-entropy) sample of each cluster."""

    def select(self, budget):
        device = self.device
        uf = _features(self, self.unlabeled_dataset, device, labeled=False)
        assign, _, _ = _kmeans_torch(uf, budget, seed=0, device=device)

        self.model.eval()
        ent = []
        with torch.no_grad():
            for batch in DataLoader(self.unlabeled_dataset, batch_size=1000, shuffle=False):
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                p = F.softmax(self.model(x.to(device)), dim=1)
                ent.append(-(p * torch.log(p.clamp_min(1e-12))).sum(1).cpu())
        ent = torch.cat(ent).to(device)

        selected = []
        for c in range(budget):
            members = torch.where(assign == c)[0]
            if members.numel() == 0:
                continue
            selected.append(members[ent[members].argmin()].item())
        return _random_fill(selected, budget, uf.shape[0])


# -------------------------------------------------------------- ALFA-Margin
class ALFAMargin(Strategy):
    """ALFA-Margin (internal baseline): cluster the pool into `budget` clusters,
    then pick the highest margin-uncertainty (minimum top1-top2 probability gap)
    sample per cluster. Not cited in the paper (no verified reference)."""

    def select(self, budget):
        device = self.device
        uf = _features(self, self.unlabeled_dataset, device, labeled=False)
        assign, _, _ = _kmeans_torch(uf, budget, seed=0, device=device)

        self.model.eval()
        margin = []
        with torch.no_grad():
            for batch in DataLoader(self.unlabeled_dataset, batch_size=1000, shuffle=False):
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                p = F.softmax(self.model(x.to(device)), dim=1)
                top2 = p.topk(2, dim=1).values
                margin.append((top2[:, 0] - top2[:, 1]).cpu())
        margin = torch.cat(margin).to(device)

        selected = []
        for c in range(budget):
            members = torch.where(assign == c)[0]
            if members.numel() == 0:
                continue
            selected.append(members[margin[members].argmin()].item())
        return _random_fill(selected, budget, uf.shape[0])
