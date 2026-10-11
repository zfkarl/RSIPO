import torch

from verl.trainer.ppo.core_algos import compute_opd_loss


def _inputs():
    torch.manual_seed(0)
    log_prob = torch.randn(4, 6, requires_grad=True)
    teacher = torch.randn(4, 6)
    mask = torch.ones(4, 6)
    mask[1, 4:] = 0
    step_mask = torch.tensor([1.0, 1.0, 0.0, 1.0])
    return log_prob, teacher, mask, step_mask


def test_absent_or_unit_weight_reproduces_seed():
    log_prob, teacher, mask, step_mask = _inputs()
    base = compute_opd_loss(log_prob, teacher, mask, step_mask)
    unit = compute_opd_loss(log_prob, teacher, mask, step_mask, token_weight=torch.ones(4, 1))
    for a, b in zip(base, unit):
        assert torch.equal(a, b)


def test_row_weight_scales_numerator_only():
    log_prob, teacher, mask, step_mask = _inputs()
    weight = torch.tensor([[0.0], [2.0], [1.5], [0.5]])
    loss, active, *_ = compute_opd_loss(log_prob, teacher, mask, step_mask, token_weight=weight)
    gate = torch.sigmoid(5.0 * (teacher - log_prob.detach()))
    opd_mask = mask * step_mask[:, None]
    expected = (weight * gate * (teacher - log_prob) * opd_mask).sum() / opd_mask.sum()
    assert torch.allclose(loss, expected)
    base_active = compute_opd_loss(log_prob, teacher, mask, step_mask)[1]
    assert torch.equal(active, base_active)


def test_weight_is_not_differentiated_and_scales_gradient():
    log_prob, teacher, mask, step_mask = _inputs()
    weight = torch.full((4, 1), 3.0, requires_grad=True)
    loss = compute_opd_loss(log_prob, teacher, mask, step_mask, token_weight=weight)[0]
    loss.backward()
    assert weight.grad is None
    scaled = log_prob.grad.clone()
    log_prob.grad = None
    compute_opd_loss(log_prob, teacher, mask, step_mask)[0].backward()
    assert torch.allclose(scaled, 3.0 * log_prob.grad)
