import math

import torch


def decayed_running_max(scores: torch.Tensor, gamma: float) -> torch.Tensor:
    """
    out[..., t] = max_{j <= t} gamma^(t - j) * scores[..., j], along the last dimension.

    Computed in closed form as one cumulative max in log space:
    log out(t) = t log(gamma) + cummax_{j <= t} (log s(j) - j log(gamma)).
    Scores must be >= 0 (they are clamped at 1e-30 before the log).
    """
    seq_len = scores.shape[-1]
    log_gamma = math.log(gamma)
    positions = torch.arange(seq_len, device=scores.device, dtype=torch.float32)
    log_scores = scores.float().clamp_min(1e-30).log()
    return (torch.cummax(log_scores - positions * log_gamma, dim=-1).values + positions * log_gamma).exp()


def afterglow(scores: torch.Tensor, gamma: float = 0.95, n_sink: int = 4) -> torch.Tensor:
    """
    Afterglow: s'(t) = max(s(t), max_{j < t, j >= n_sink} gamma^(t - j) s(j)), per layer and KV head.

    scores : (..., seq_len) non-negative importance scores of the context tokens (e.g. (batch, kv_heads, seq_len)).
    gamma  : decay; a score c times the keep threshold carries over about log(c) / log(1 / gamma) tokens.
    n_sink : the first n_sink tokens (attention sinks) keep their own score but are not propagation sources.

    Returns a tensor of the same shape and dtype. The eviction method then selects from s' exactly as from s.
    """
    if (scores < 0).any():
        raise ValueError("Afterglow needs scores >= 0 (apply a rank-preserving positive transform first)")
    sources = scores
    if n_sink > 0:
        sources = scores.clone()
        sources[..., :n_sink] = 0
    return torch.maximum(scores, decayed_running_max(sources, gamma).to(scores.dtype))
