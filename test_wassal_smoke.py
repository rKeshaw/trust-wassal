# test_wassal_smoke.py
"""End-to-end smoke test for the WASSAL multiclass selection loop.

Runs WASSAL_Multiclass.select_only_for_query on small dummy data (CPU by
default) and checks that the selection and per-class simplexes are valid.
As a qualitative signal, reports the sliced Wasserstein distance between the
query set and (a) the WASSAL-selected subset vs (b) a random subset.

Usage: python test_wassal_smoke.py
"""
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import wasserstein_distance

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trust.strategies.wassal_multiclass import WASSAL_Multiclass

SEED = 0
EMB_DIM = 16
N_CLASSES = 3
POOL_SIZE = 120
BUDGET = 8


def sliced_wasserstein_np(X, Y, n_projections=64, seed=0):
    """Approximate multi-dim Wasserstein by averaging 1D Wasserstein over
    random projections."""
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(n_projections):
        v = rng.normal(size=X.shape[1])
        v /= (np.linalg.norm(v) + 1e-12)
        vals.append(wasserstein_distance(X.dot(v), Y.dot(v)))
    return float(np.mean(vals))


class LabeledSet(torch.utils.data.Dataset):
    """Returns (image, label) pairs with class-dependent means so WASSAL has
    real structure to match."""

    def __init__(self, n, d=EMB_DIM, nclasses=N_CLASSES):
        g = torch.Generator().manual_seed(SEED + n)
        self.targets = torch.randint(0, nclasses, (n,), generator=g)
        offsets = 3.0 * torch.arange(nclasses).float().unsqueeze(1)
        self.data = (torch.randn(n, d, generator=g)
                     + offsets[self.targets])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


class UnlabeledSet(LabeledSet):
    """Same distribution but returns the image only, as the strategy expects
    for the unlabeled pool."""

    def __getitem__(self, idx):
        return self.data[idx]


class DummyNet(torch.nn.Module):
    def __init__(self, d_in=EMB_DIM, d_out=N_CLASSES):
        super().__init__()
        self.fc = torch.nn.Linear(d_in, d_out)

    def forward(self, x):
        return self.fc(x)


def run_smoke_test():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Running WASSAL smoke test on {device}")

    labeled_dataset = LabeledSet(n=20)
    unlabeled_dataset = UnlabeledSet(n=POOL_SIZE)
    query_dataset = LabeledSet(n=12)
    net = DummyNet().to(device)

    args = {
        "device": device,
        "batch_size": 32,
        "embedding_type": "features",
        "layer_name": "fc",  # DummyNet exposes its logits layer as features
        "wassal_iterations": 30,
        "lr": 0.01,
    }
    strat = WASSAL_Multiclass(labeled_dataset, unlabeled_dataset,
                              query_dataset, net, N_CLASSES, args)

    selected_indices, per_class_output = strat.select_only_for_query(BUDGET)
    print(f"\nselected {len(selected_indices)} indices: "
          f"{sorted(int(i) for i in selected_indices)}")

    # --- Validate the selection ---
    assert len(selected_indices) == BUDGET, \
        f"expected {BUDGET} selections, got {len(selected_indices)}"
    assert len(set(int(i) for i in selected_indices)) == BUDGET, \
        "selected indices are not unique"
    assert all(0 <= int(i) < POOL_SIZE for i in selected_indices), \
        "selected index out of pool range"

    # --- Validate the per-class simplexes ---
    assert len(per_class_output) == N_CLASSES, \
        f"expected {N_CLASSES} per-class outputs, got {len(per_class_output)}"
    for simplex_query, simplex_refrain, class_idx in per_class_output:
        assert simplex_query.shape[0] == POOL_SIZE
        assert (simplex_query >= 0).all(), \
            f"class {class_idx}: simplex has negative weights"

    # --- Qualitative check: selected subset vs random subset ---
    pool = unlabeled_dataset.data.numpy()
    query = query_dataset.data.numpy()
    rng = np.random.RandomState(SEED)
    random_subset = pool[rng.choice(POOL_SIZE, BUDGET, replace=False)]
    selected_subset = pool[np.asarray(selected_indices, dtype=int)]

    sw_selected = sliced_wasserstein_np(query, selected_subset)
    sw_random = sliced_wasserstein_np(query, random_subset)
    print(f"sliced Wasserstein to query set: "
          f"selected={sw_selected:.4f}, random={sw_random:.4f}")

    print("\nSmoke test PASSED")


if __name__ == "__main__":
    run_smoke_test()
