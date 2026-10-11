import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Generator

import torch
from kvpress.presses.base_press import BasePress, is_prefilling
from kvpress.utils import compute_n_kept, extract_keys_and_values, get_prerope_query_states
from torch import nn
from transformers import PreTrainedModel
from transformers.models.llama.modeling_llama import rotate_half

from afterglow.propagation import afterglow


@dataclass
class KVzipScorerPress(BasePress):
    """
    The "KVzip+ scorer" of the paper: query-aware, single-pass scoring with one budget across layers and heads.

    1. For every layer and KV head, the attention of the last `window_size` prompt tokens (which contain the question)
       over the context, max over these queries and over the query heads of the group, with the KVzip+ normalization
       (divided by ||h|| of the query, multiplied by ||W_o v|| of the key's value).
    2. Optionally Afterglow (use_afterglow=True) on these scores, per layer and KV head.
    3. One top-k over all (layer, KV head, token) once every layer is scored; the first n_sink tokens are always kept.
       Pruned pairs are masked through kvpress' `module.masked_key_indices` (the cache is not physically shrunk).

    Requires the question at the end of the compressed context (query-aware compression).
    """

    compression_ratio: float = 0.0
    window_size: int = 64
    n_sink: int = 4
    use_afterglow: bool = False
    gamma: float = 0.95

    scores: dict = field(init=False, default_factory=dict, repr=False)
    modules: dict = field(init=False, default_factory=dict, repr=False)

    def __post_init__(self):
        assert 0 <= self.compression_ratio < 1, "compression_ratio must be in [0, 1)"
        assert 0 < self.gamma < 1, "gamma must be in (0, 1)"

    @staticmethod
    def value_output_norm(module: nn.Module, values: torch.Tensor) -> torch.Tensor:
        """||W_o v|| for each query head of the group, via the Gram matrix of W_o. (bsz, kv_heads, groups, seq_len)."""
        num_kv_heads, head_dim = values.shape[1], values.shape[-1]
        num_groups = module.config.num_attention_heads // num_kv_heads
        Wo = module.o_proj.weight.view(module.config.hidden_size, num_kv_heads, num_groups, head_dim).float()
        gram = torch.einsum("jhgi,jhgk->hgik", Wo, Wo)
        v = values.float()
        return torch.einsum("bhti,hgik,bhtk->bhgt", v, gram, v).clamp_min(0).sqrt()

    def question_scores(self, module: nn.Module, hidden_states, keys, values, kwargs) -> torch.Tensor:
        """Normalized attention of the last window_size queries, max over queries and group heads. (bsz, H, seq_len)."""
        bsz, num_kv_heads, k_len, head_dim = keys.shape
        num_groups = module.config.num_attention_heads // num_kv_heads
        window_size = min(self.window_size, k_len)

        queries = get_prerope_query_states(module, hidden_states[:, -window_size:])
        cos, sin = kwargs["position_embeddings"]
        cos, sin = cos[:, -window_size:].unsqueeze(1), sin[:, -window_size:].unsqueeze(1)
        queries = (queries * cos) + (rotate_half(queries) * sin)
        queries = queries.view(bsz, num_kv_heads, num_groups, window_size, head_dim)

        attn = torch.einsum("bhgqd,bhkd->bhgqk", queries.float(), keys.float()) / math.sqrt(head_dim)
        causal_mask = torch.ones(window_size, k_len, dtype=torch.bool, device=keys.device)
        attn.masked_fill_(causal_mask.triu(k_len - window_size + 1), float("-inf"))
        attn = attn.softmax(dim=-1)

        h_norm = hidden_states[:, -window_size:].float().norm(dim=-1)
        attn = attn / h_norm[:, None, None, :, None]
        attn = attn * self.value_output_norm(module, values)[:, :, :, None, :]
        return attn.amax(dim=(2, 3))

    def forward_hook(self, module: nn.Module, input: list[torch.Tensor], kwargs: dict, output: list):
        """Only score during prefill; the budget is allocated once all layers are scored (see __call__)."""
        hidden_states = kwargs["hidden_states"]
        if self.compression_ratio == 0 or not is_prefilling(kwargs["cache_position"], hidden_states.shape[1]):
            return output
        keys, values = extract_keys_and_values(kwargs["past_key_values"], module.layer_idx)
        self.scores[module.layer_idx] = self.question_scores(module, hidden_states, keys, values, kwargs)
        self.modules[module.layer_idx] = module
        return output

    def compress_all(self):
        """Afterglow (optional), then one top-k over (layer, KV head, token); pruned pairs are masked."""
        layer_indices = sorted(self.scores)
        scores = []
        for layer_idx in layer_indices:
            s = self.scores[layer_idx].float()
            scores.append(afterglow(s, self.gamma, self.n_sink) if self.use_afterglow else s)
        scores = torch.stack(scores, dim=1)  # (bsz, n_layers, num_kv_heads, seq_len)
        scores[..., : self.n_sink] = float("inf")

        bsz, n_layers, num_kv_heads, seq_len = scores.shape
        n_pruned = n_layers * num_kv_heads * (seq_len - compute_n_kept(seq_len, self.compression_ratio))
        pruned = torch.topk(-scores.reshape(bsz, -1), n_pruned, dim=-1).indices
        batch_indices = torch.arange(bsz, device=pruned.device)[:, None].expand_as(pruned)
        layer_of_pruned = pruned // (num_kv_heads * seq_len)
        for i, layer_idx in enumerate(layer_indices):
            mask = layer_of_pruned == i
            flat = pruned[mask] % (num_kv_heads * seq_len)
            self.modules[layer_idx].masked_key_indices = (batch_indices[mask], flat // seq_len, flat % seq_len)

    @contextmanager
    def __call__(self, model: PreTrainedModel) -> Generator:
        self.scores, self.modules = {}, {}
        try:
            with super().__call__(model):
                yield
            if self.scores:
                self.compress_all()
        finally:
            self.scores, self.modules = {}, {}
