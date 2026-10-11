# Afterglow: Forward Span Propagation for Long-Context Recall under Query-Aware KV Cache Eviction

Query-aware eviction methods (SnapKV and the methods built on it) score each context token by the attention the question pays to it, and keep the highest-scoring tokens. This attention concentrates on the *first* tokens of the span the question asks about. So eviction keeps the start of a value, identifier or title but drops its continuation, and the model copies a truncated or corrupted answer.

Afterglow is a training-free plug-in that lets each token inherit the decayed score of the tokens before it before the method's own selection at the same budget:

```
s'(t) = max( s(t), max_{j < t, j >= n_sink} gamma^(t - j) * s(j) )      gamma = 0.95, n_sink = 4
```

It is one cumulative max in log space over the score tensor the method already computes
([`afterglow/propagation.py`](afterglow/propagation.py)).

## Install

```bash
pip install -e .                       # installs kvpress==0.5.5 from PyPI
git clone --branch v0.5.5 --depth 1 https://github.com/NVIDIA/kvpress   # datasets and scorers for evaluation
pip install "kvpress[eval]==0.5.5"     # scorer dependencies (LongBench metrics)
pytest tests                           # optional (needs pytest)
```

## Use

```python
from kvpress import AdaKVPress, SnapKVPress
from transformers import pipeline
from afterglow import AfterglowPress, KVzipScorerPress

press = AfterglowPress(press=SnapKVPress())                 # SnapKV + Afterglow
# press = AdaKVPress(AfterglowPress(press=SnapKVPress()))   # Ada-KV + Afterglow
# press = KVzipScorerPress(use_afterglow=True)              # KVzip+ scorer + Afterglow
press.compression_ratio = 0.875                             # share of the cache evicted

pipe = pipeline("kv-press-text-generation", model="meta-llama/Llama-3.1-8B-Instruct", device="cuda:0",
                model_kwargs={"dtype": "auto"})
# query-aware compression: the question is the end of the compressed context
answer = pipe(context + question, question="", answer_prefix="Answer:", press=press)["answer"]
```

`AfterglowPress` wraps any kvpress `ScorerPress` whose scores are non-negative. For your own scorer, call
`afterglow(scores, gamma=0.95, n_sink=4)` on scores of shape `(..., seq_len)`.

## Reproduce the paper

`scripts/evaluate.py` runs the methods of the paper without and with Afterglow (greedy decoding, bf16, query-aware,
observation window = last 64 prompt tokens unless `--true_window`). Results go to
`results/<model>/<dataset>/<method>__<ratio or budget>/` (`predictions.csv`, `metrics.json`).

| Paper | Command |
|---|---|
| Tables 1–2 (RULER-4K, full test set; Table 1 = the 8 `niah_*` tasks of `metrics.json`) | `python scripts/evaluate.py --model <model> --dataset ruler --data_dir 4096 --ratios 0.5 0.75 0.875 0.9375` |
| RULER at other lengths (test subsets) | `--data_dir 16384 --per_task 50` (32K: `--per_task 20`, 64K: `--per_task 10`) |
| Tool-output retrieval | `python scripts/evaluate.py --model <model> --dataset toolout --ratios 0.875 0.9375` |
| LongBench | `python scripts/evaluate.py --model <model> --dataset longbench --truncate_tokens 16000 --true_window --ratios 0.875` (or `--budgets 512`) |

Models: `Qwen/Qwen3-8B` (kvpress' pipeline disables thinking by default), `meta-llama/Llama-3.1-8B-Instruct`,
`Qwen/Qwen3-4B-Instruct-2507`, `Qwen/Qwen3-14B`. Methods: `full`, `snapkv`, `adakv`, `criticaladakv`, `kvzip_scorer`,
each also as `<method>+afterglow`. Pass `--kvpress_eval_dir kvpress/evaluation` (default) to point to the cloned
evaluation code.

The tool-output data and how it was built are described in [`data/README.md`](data/README.md).

## Notes

- **Plugin location** SnapKV and the KVzip+ scorer: on the per-token score. Ada-KV: before its head-wise
  allocation. CriticalAdaKV: on the inner SnapKV score, because CriticalKV writes the largest float into scores it protects.
- **Negative scores.** Afterglow works in log space and needs scores >= 0. Scorers with negative scores (e.g. Knorm, KeyDiff) need a rank-preserving positive transform first.

License: Apache-2.0.
