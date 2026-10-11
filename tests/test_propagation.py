import torch

from afterglow import afterglow, decayed_running_max


def naive(scores, gamma, n_sink):
    out = scores.clone()
    for t in range(scores.shape[-1]):
        for j in range(n_sink, t):
            out[..., t] = torch.maximum(out[..., t], gamma ** (t - j) * scores[..., j])
    return out


def test_closed_form_matches_loop():
    torch.manual_seed(0)
    s = torch.rand(2, 3, 50) ** 4
    for gamma in (0.5, 0.9, 0.95):
        for n_sink in (0, 4):
            assert torch.allclose(afterglow(s, gamma, n_sink), naive(s, gamma, n_sink), rtol=1e-5, atol=1e-7)


def test_recurrence():
    # without sinks, s'(t) = max(s(t), gamma * s'(t-1))
    s = torch.rand(1, 1, 30)
    out = decayed_running_max(s, 0.9)
    for t in range(1, 30):
        assert torch.isclose(out[0, 0, t], torch.maximum(s[0, 0, t], 0.9 * out[0, 0, t - 1]), rtol=1e-5)


def test_sinks_are_not_sources():
    s = torch.zeros(1, 1, 20)
    s[..., 0] = 1000.0  # attention sink
    out = afterglow(s, 0.95, n_sink=4)
    assert out[..., 0] == 1000.0 and (out[..., 1:] < 1e-20).all()  # zeros are clamped to 1e-30 in log space


def test_span_continuation_inherits_start():
    s = torch.full((1, 1, 40), 0.01)
    s[..., 10] = 1.0  # high score at the start of a span
    out = afterglow(s, 0.95, n_sink=4)
    assert torch.allclose(out[..., 10:20], 0.95 ** torch.arange(10.0))
    assert torch.equal(out[..., :10], s[..., :10])  # nothing propagates backward
