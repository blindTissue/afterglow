from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from kvpress import ScorerPress

from afterglow.propagation import afterglow, decayed_running_max


@dataclass
class AfterglowPress(ScorerPress):
    """
    Afterglow on top of any kvpress ScorerPress: the wrapped press scores the context as usual, Afterglow
    post-processes the scores of each layer and KV head, and the selection runs on the new scores at the same budget.

    Composes with kvpress' budget-allocating wrappers, which then allocate from the new scores:
        AfterglowPress(press=SnapKVPress())                        # SnapKV + Afterglow
        AdaKVPress(AfterglowPress(press=SnapKVPress()))            # Ada-KV + Afterglow
        CriticalAdaKVPress(AfterglowPress(press=SnapKVPress()))    # CriticalAdaKV + Afterglow

    direction is the shape ablation of the paper (appendix): "forward" is Afterglow; "backward", "both" (forward and
    backward decay) and "pool" (symmetric max-pooling of width pool_kernel) are the alternatives it is compared to.
    In those variants the sinks and the observation window are not sources either.
    """

    press: ScorerPress = field(default=None)
    gamma: float = 0.95
    n_sink: int = 4
    direction: str = "forward"
    pool_kernel: int = 31

    def __post_init__(self):
        super().__post_init__()
        assert isinstance(self.press, ScorerPress), "AfterglowPress requires a ScorerPress"
        assert 0 < self.gamma < 1, "gamma must be in (0, 1)"
        assert self.direction in ("forward", "backward", "both", "pool"), f"unknown direction {self.direction}"

    def score(self, module, hidden_states, keys, values, attentions, kwargs):
        s = self.press.score(module, hidden_states, keys, values, attentions, kwargs)
        if self.direction == "forward":
            return afterglow(s, self.gamma, self.n_sink)
        if (s < 0).any():
            raise ValueError("AfterglowPress needs scores >= 0")
        src = s.clone()
        src[..., : self.n_sink] = 0
        window = getattr(self.press, "window_size", 0)
        if window:
            src[..., -window:] = 0
        backward = decayed_running_max(src.flip(-1), self.gamma).flip(-1)
        if self.direction == "backward":
            spread = backward
        elif self.direction == "both":
            spread = torch.maximum(decayed_running_max(src, self.gamma), backward)
        else:
            k = self.pool_kernel
            spread = F.max_pool1d(src.float().reshape(-1, 1, src.shape[-1]), k, stride=1, padding=k // 2)
            spread = spread.reshape(src.shape)
        return torch.maximum(s, spread.to(s.dtype))
