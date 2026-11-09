"""JEPA masking: v1 predictions for masked peaks are identical within a spectrum (the bug); v2 queries carry m/z."""
import os, sys
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from casmi.jepa import JEPA  # noqa: E402


def batch(B=8, N=32):
    torch.manual_seed(0)
    mz = torch.sort(torch.rand(B, N) * 400 + 50, 1).values
    pad = torch.zeros(B, N, dtype=torch.bool); pad[:, 28:] = True
    return dict(mz=mz, it=torch.rand(B, N), pad=pad, prec=torch.full((B,), 500.), adduct=torch.zeros(B, dtype=torch.long),
                mode=torch.ones(B, dtype=torch.long), ce=torch.full((B,), 40.))


def spread(query):
    m = JEPA(d=64, layers=2, query=query).eval()
    with torch.no_grad():
        return float(m.loss(batch())[1]["spread"])


def test_v1_masked_predictions_collapse():
    assert spread("none") < 1e-5


def test_v2_masked_predictions_differ():
    assert spread("mz") > 0.05


def test_v2_context_cannot_see_masked_peaks():
    """Changing a masked peak's intensity must not change the context encoder's visible outputs."""
    m = JEPA(d=64, layers=2, query="mz").eval(); b = batch()
    from casmi.jepa import block_mask
    torch.manual_seed(1); mask = block_mask(b["mz"], b["pad"])
    run = lambda it: m.ctx(b["mz"], it, b["pad"] | mask, b["prec"], b["adduct"], b["mode"], b["ce"], return_tokens=True)
    it2 = torch.where(mask, torch.rand_like(b["it"]), b["it"])
    with torch.no_grad():
        (c1, t1), (c2, t2) = run(b["it"]), run(it2)
    vis = ~(b["pad"] | mask)
    assert torch.allclose(c1, c2, atol=1e-5) and torch.allclose(t1[vis], t2[vis], atol=1e-5)


def test_ctx_free_control_runs():
    m = JEPA(d=64, layers=2, query="mz", ctx_free=True)
    loss, info = m.loss(batch()); loss.backward()
    assert float(info["cls"]) == 0.0
