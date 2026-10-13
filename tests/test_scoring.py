"""🧪 CPU checks of LORE-KV's estimator and scoring against the paper's equations."""
import math

import pytest
import torch

from lorekv.monkeypatch.scoring import TestTimeLookaheadKVCluster as Cluster
from lorekv.monkeypatch.scoring import (
    divergence_from_query_stacks,
    reliability_weights,
    standardize_disagreement,
    step_stats,
)

torch.manual_seed(0)


def _cluster(window=8, budget=16, chunk=4, H=2, D=8, hidden=12, **kw):
    args = dict(window_size=window, base_capacity=budget, chunk_size=chunk, skip=0,
                layer_idx=0, num_hidden_layers=1, num_particles=2, lookahead_len=3,
                particle_beta=1.0, particle_gamma=1.3, robust_alpha=0.0,
                robust_topk_beta=1.0, lambda_lookahead=1.0, lambda_k=0.0, lambda_v=0.0,
                budget_tau=0.0, lookahead_temperature=1.0)
    args.update(kw)
    c = Cluster(**args)
    c.head_dim = D
    c.o_proj_weight = torch.randn(hidden, H * D)
    return c


# --------------------------------------------------------------------------- Eq. 2
@pytest.mark.parametrize("top_k,top_p,temp", [(None, None, 1.0), (5, None, 1.0),
                                              (None, 0.8, 1.0), (None, None, 0.7)])
def test_centered_score_has_zero_conditional_mean(top_k, top_p, temp):
    """E_{y~q}[c(y)] = 0 under the distribution the token is sampled from (Eq. 2)."""
    logits = torch.randn(3, 50)
    vocab = torch.arange(50)
    cs, qs = [], []
    for y in vocab:
        lp, H, V, c = step_stats(logits, y.repeat(3), temperature=temp, top_k=top_k, top_p=top_p)
        cs.append(c)
        qs.append(lp.exp())
    c = torch.stack(cs, -1)
    q = torch.stack(qs, -1)
    expectation = torch.where(q > 0, q * c, torch.zeros_like(c)).sum(-1)
    assert torch.allclose(expectation, torch.zeros(3), atol=1e-5)


def test_varentropy_matches_definition():
    """V_u = Var_{y~q}[log q(y)] (Eq. 9)."""
    logits = torch.randn(1, 30)
    _, H, V, _ = step_stats(logits, torch.tensor([0]))
    logq = torch.log_softmax(logits, -1)
    q = logq.exp()
    var = (q * logq.pow(2)).sum() - (q * logq).sum().pow(2)
    assert torch.allclose(V, var.reshape(1), atol=1e-5)
    assert torch.allclose(H, -(q * logq).sum().reshape(1), atol=1e-5)


# ------------------------------------------------------------------------ Eqs. 3-4
def test_reliability_weights_follow_eqs_3_and_4():
    M, T = 4, 6
    c, H, V = torch.randn(M, T), torch.rand(M, T), torch.rand(M, T)
    d_hat = torch.randn(M)
    w, parts = reliability_weights(c, H, V, d_hat, beta=1.0, gamma=1.3, lambda_d=0.3)
    R = c.mean(-1) - 1.3 * H.mean(-1) - 0.3 * d_hat
    assert torch.allclose(parts["R"], R)
    assert torch.allclose(w, torch.softmax(R, 0))
    assert math.isclose(float(w.sum()), 1.0, rel_tol=1e-6)


def test_uniform_and_logprob_settings():
    """(0,0,0) gives uniform weights; (1,1,0) weights by mean log-probability (App. C.3)."""
    M, T = 5, 4
    logp, H = -torch.rand(M, T) * 3, torch.rand(M, T) * 2
    c = logp + H
    w0, _ = reliability_weights(c, H, torch.zeros(M, T), None, 0.0, 0.0, 0.0)
    assert torch.allclose(w0, torch.full((M,), 1.0 / M))
    w1, _ = reliability_weights(c, H, torch.zeros(M, T), None, 1.0, 1.0, 0.0)
    assert torch.allclose(w1, torch.softmax(logp.mean(-1), 0), atol=1e-6)


# ------------------------------------------------------------------- Eq. 10, C.3
def test_disagreement_and_standardization():
    M, Hq, T, D = 4, 3, 5, 8
    q = torch.randn(1, Hq, T, D).repeat(M, 1, 1, 1)          # identical trajectories
    assert torch.allclose(divergence_from_query_stacks([q], M), torch.zeros(M), atol=1e-6)

    q = torch.randn(M, Hq, T, D)
    d = divergence_from_query_stacks([q, torch.randn(M, Hq, T, D)], M)
    qn = q / q.norm(dim=-1, keepdim=True)
    cos = torch.einsum("mhtd,rhtd->mrht", qn, qn).mean((-1, -2))
    assert d.shape == (M,) and torch.all(d >= 0)
    z = standardize_disagreement(d)
    assert abs(float(z.mean())) < 1e-5 and abs(float(z.std(unbiased=False)) - 1.0) < 1e-4
    assert torch.allclose(1 - cos.diagonal(), torch.zeros(M), atol=1e-5)

    assert torch.equal(divergence_from_query_stacks([torch.randn(2, Hq, T, D)], 2), torch.zeros(2))
    assert torch.equal(standardize_disagreement(torch.tensor([0.3, 0.7])), torch.zeros(2))


# ------------------------------------------------------------------------ Eqs. 5-6
def test_leave_one_out_deletion_effect_is_exact():
    """o_q - o_{q,-i} = a_{q,i} / (1 - a_{q,i}) (v_i - o_q) when eps = 0 (Eq. 5)."""
    n, D = 10, 6
    q, K, V = torch.randn(D), torch.randn(n, D), torch.randn(n, D)
    a = torch.softmax(K @ q / math.sqrt(D), 0)
    o = a @ V
    for i in range(n):
        keep = torch.arange(n) != i
        o_minus = torch.softmax(K[keep] @ q / math.sqrt(D), 0) @ V[keep]
        closed = a[i] / (1 - a[i]) * (V[i] - o)
        assert torch.allclose(o - o_minus, closed, atol=1e-5)


@pytest.mark.parametrize("objective", ["projected", "unprojected", "attention"])
def test_projected_scores_match_brute_force(objective):
    H, D, n, Qn = 2, 8, 20, 5
    cl = _cluster(H=H, D=D, score_objective=objective)
    K, V = torch.randn(1, H, n, D), torch.randn(1, H, n, D)
    qb = torch.randn(1, H, Qn, D)
    (out,) = cl._project_and_score(K, V, [qb], [False])
    for h in range(H):
        a = torch.softmax(qb[0, h] @ K[0, h].T / math.sqrt(D), -1)        # [Q, n]
        if objective == "attention":
            ref = a.mean(0)
        else:
            o = a @ V[0, h]
            W = cl.o_proj_weight[:, h * D:(h + 1) * D] if objective == "projected" else torch.eye(D)
            diff = (V[0, h][None] - o[:, None]) @ W.T                     # [Q, n, *]
            ref = (a / (1 - a + cl.causal_eps) * diff.norm(dim=-1)).mean(0)
        assert torch.allclose(out[h], ref, atol=1e-4, rtol=1e-4)


# ------------------------------------------------------------------- Eq. 7-8, 11
def _compress(cl, n=64, H=2, D=8, M=2, T=3):
    cl.num_key_value_groups = 1
    cl.full_key_states = torch.randn(1, H, n, D)
    cl.full_value_states = torch.randn(1, H, n, D)
    cl.window_queries = torch.randn(1, H, cl.window_size, D)
    cl._lookahead_buffers = [[torch.randn(1, H, 1, D) for _ in range(T)] for _ in range(M)]
    cl.particle_weights = torch.full((M,), 1.0 / M)
    return cl.compress()


def test_compress_keeps_budget_and_protected_set():
    H, n = 2, 64
    cl = _cluster(H=H, budget=16, chunk=4, window=8)
    k, v, meta = _compress(cl, n=n, H=H)
    assert meta["head_lens"].tolist() == [16] * H                          # B tokens per head
    assert k.shape == (16 * H, 8) and v.shape == k.shape
    first = cl.full_key_states[0, 0, :4]
    window = cl.full_key_states[0, 0, -8:]
    kept = k[:16]
    for row in torch.cat([first, window]):                                 # P is always kept
        assert any(torch.equal(row, r) for r in kept)


def test_score_combines_window_and_weighted_lookahead(monkeypatch):
    """S = psi(E_W) + lambda_L sum_m omega_m psi(E_m) (Eq. 7) drives selection."""
    cl = _cluster(H=1, budget=12, chunk=4, window=4, lambda_lookahead=2.0)
    seen = {}
    real_score, real_select = cl._project_and_score, cl._select_chunks

    def spy_score(*a, **k):
        seen["E"] = real_score(*a, **k)
        return seen["E"]

    def spy_select(score, *a, **k):
        seen["S"] = score.clone()
        return real_select(score, *a, **k)

    monkeypatch.setattr(cl, "_project_and_score", spy_score)
    monkeypatch.setattr(cl, "_select_chunks", spy_select)
    _compress(cl, n=40, H=1, M=2)
    E_W, E_1, E_2 = seen["E"]
    assert torch.allclose(seen["S"], E_W + 2.0 * (0.5 * E_1 + 0.5 * E_2))


def test_adaptive_head_budget_conserves_layer_budget():
    """Eq. 11 with tau > 0 redistributes, but never changes, the layer's total budget."""
    H = 4
    cl = _cluster(H=H, budget=24, chunk=4, window=8, budget_tau=1.0)
    _, _, meta = _compress(cl, n=96, H=H)
    assert int(meta["head_lens"].sum()) == 24 * H
