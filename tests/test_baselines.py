"""CPU tests of the CriticalKV selection on the Ada-KV cluster (adaptive_snapkv)."""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from adaptive_snapkv.monkeypatch.snapkv_utils import (  # noqa: E402
    CRITICAL_EPS, AdaptiveSnapKVCluster, criticalkv_select, projected_value_l1)


def test_projected_value_l1_matches_definition():
    """||v W_O^h||_1, averaged over the query heads of the group."""
    torch.manual_seed(0)
    n, d, H, hidden = 9, 4, 4, 6
    v = torch.randn(n, d, dtype=torch.float32)
    w_o = torch.randn(hidden, H * d, dtype=torch.float32)
    ref = sum((v @ w_o[:, h * d:(h + 1) * d].T).abs().sum(-1) for h in (2, 3)) / 2
    assert torch.allclose(projected_value_l1(v, w_o, [2, 3]), ref, rtol=1e-5)


def test_two_stage_selection():
    """Stage 1 keeps the top half by attention; stage 2 fills by (a + eps) * ||v W_O||_1."""
    a = torch.tensor([[0.50, 0.30, 0.10, 0.06, 0.04]])
    order = a.argsort(dim=-1, descending=True).view(1, 1, -1)
    l1 = torch.tensor([0.1, 0.1, 1.0, 9.0, 1.0])
    keep = criticalkv_select(a, order, 4, l1)
    assert keep.shape == (1, 1, 4)
    # stage 1: positions 0 and 1; stage 2: (a + eps) * l1 ranks 3 (0.54), then 2 (0.10)
    assert sorted(keep.flatten().tolist()) == [0, 1, 2, 3]
    keep = criticalkv_select(a, order, 3, l1)
    assert sorted(keep.flatten().tolist()) == [0, 2, 3]       # stage 1 keeps int(1.5) = 1
    s2 = (a[0] + CRITICAL_EPS) * l1
    assert s2[3] > s2[2] > s2[4]


def _run_cluster(o_proj_weight, seed=0, H=4, n=96, d=8, B=48, w=8):
    torch.manual_seed(seed)
    c = AdaptiveSnapKVCluster(window_size=w, kernel_size=7, pooling="maxpool", base_capacity=B,
                              floor_alpha=0.2, skip=0, layer_idx=0, num_hidden_layers=1)
    c.o_proj_weight = o_proj_weight
    k, q, v = (torch.randn(1, H, n, d) for _ in range(3))
    keys, _ = c.update_kv(k, q, v)
    return c, keys, k


@pytest.mark.parametrize("seed", [0, 1])
def test_critical_keeps_ada_budgets_and_changes_selection(seed):
    """CriticalKV uses the same per-head budgets as Ada-KV and only changes which entries are kept."""
    H, d, hidden = 4, 8, 16
    ada, ada_keys, _ = _run_cluster(None, seed)
    torch.manual_seed(100 + seed)
    w_o = torch.randn(hidden, H * d)
    crit, crit_keys, k = _run_cluster(w_o, seed)
    assert torch.equal(ada.head_lens, crit.head_lens)
    assert ada_keys.shape == crit_keys.shape
    assert not torch.equal(ada_keys, crit_keys)
    # The kept rows are rows of the original keys, and the window is kept.
    offsets = [0] + crit.head_lens.cumsum(0).tolist()
    for h in range(H):
        rows = crit_keys[offsets[h]:offsets[h + 1]]
        assert torch.equal(rows[-8:], k[0, h, -8:])
        match = (rows[:, None, :] == k[0, h][None]).all(-1).any(-1)
        assert bool(match.all())
