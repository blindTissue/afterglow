"""
Evaluate query-aware KV cache eviction with and without Afterglow (kvpress pipeline, greedy decoding).

Datasets and scorers come from kvpress' own evaluation code (clone of NVIDIA/kvpress at tag v0.5.5, see README).

Examples (the settings of the paper):
  # RULER-4K, full test set (6,500 samples), 87.5% evicted
  python scripts/evaluate.py --model meta-llama/Llama-3.1-8B-Instruct --dataset ruler --data_dir 4096 \
      --methods full snapkv snapkv+afterglow --ratios 0.875
  # tool-output retrieval (data/toolout_4096.parquet, scored with RULER's string match)
  python scripts/evaluate.py --model meta-llama/Llama-3.1-8B-Instruct --dataset toolout \
      --methods snapkv snapkv+afterglow --ratios 0.875 0.9375
  # LongBench, 16 English tasks, contexts middle-truncated to 16K tokens, question tokens as observation window,
  # fixed budget of 512 tokens per layer and KV head
  python scripts/evaluate.py --model Qwen/Qwen3-8B --dataset longbench --truncate_tokens 16000 --true_window \
      --methods snapkv snapkv+afterglow --budgets 512
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from kvpress import AdaKVPress, CriticalAdaKVPress, SnapKVPress
from transformers import AutoTokenizer, pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from afterglow import AfterglowPress, KVzipScorerPress  # noqa: E402

LONGBENCH_EN = ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", "gov_report", "qmsum",
                "multi_news", "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
TOOLOUT = Path(__file__).resolve().parents[1] / "data" / "toolout_4096.parquet"

# name -> factory of a fresh press (None: full cache). Afterglow: gamma = 0.95, the first 4 tokens are not sources.
METHODS = {
    "full": lambda: None,
    "snapkv": lambda: SnapKVPress(),
    "snapkv+afterglow": lambda: AfterglowPress(press=SnapKVPress()),
    "adakv": lambda: AdaKVPress(SnapKVPress()),
    "adakv+afterglow": lambda: AdaKVPress(AfterglowPress(press=SnapKVPress())),
    "criticaladakv": lambda: CriticalAdaKVPress(SnapKVPress()),
    "criticaladakv+afterglow": lambda: CriticalAdaKVPress(AfterglowPress(press=SnapKVPress())),
    "kvzip_scorer": lambda: KVzipScorerPress(),
    "kvzip_scorer+afterglow": lambda: KVzipScorerPress(use_afterglow=True),
}


def load_df(dataset, data_dirs, per_task, seed, registry):
    """All test samples (per_task=0, the full test sets of the paper) or per_task samples per task drawn with seed."""
    if dataset == "toolout":
        df = pd.read_parquet(TOOLOUT)
    else:
        dfs = []
        for d in data_dirs:
            df = load_dataset(registry[dataset], data_dir=d, split="test").to_pandas()
            if "task" not in df.columns:
                df["task"] = d
            dfs.append(df)
        df = pd.concat(dfs, ignore_index=True)
    if per_task:
        df = df.groupby("task", group_keys=False).apply(lambda g: g.sample(n=min(per_task, len(g)), random_state=seed))
    return df


def question_window(tok, question, default=64, cap=256):
    """Tokens of the question at the end of the prompt: from the last "Question:"/"Query:" marker if present, else the
    whole question field; at most `cap` tokens, `default` if there is no question."""
    starts = [question.rfind(m) for m in ("Question:", "Query:") if m in question]
    if starts:
        question = question[max(starts):]
    n = len(tok.encode(question, add_special_tokens=False))
    return min(n, cap) if n > 0 else default


def set_window(press, window):
    """Set the observation window of a press and of any press it wraps."""
    while press is not None:
        if hasattr(press, "window_size"):
            press.window_size = window
        press = getattr(press, "press", None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", default="ruler", choices=["ruler", "longbench", "toolout"])
    p.add_argument("--data_dir", nargs="+", default=None, help="ruler: context length (4096); longbench: tasks")
    p.add_argument("--per_task", type=int, default=0, help="0: full test set")
    p.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    p.add_argument("--ratios", nargs="+", type=float, default=[0.875], help="share of the cache evicted")
    p.add_argument("--budgets", nargs="+", type=int, default=None,
                   help="fixed budgets (tokens kept per layer and KV head) instead of --ratios")
    p.add_argument("--true_window", action="store_true",
                   help="observation window = the question tokens of each sample (default: the last 64 prompt tokens)")
    p.add_argument("--truncate_tokens", type=int, default=0, help="middle-truncate the context (before the question)")
    p.add_argument("--out", default="results")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--kvpress_eval_dir", default=os.environ.get("KVPRESS_EVAL_DIR", "kvpress/evaluation"),
                   help="the evaluation/ directory of a clone of NVIDIA/kvpress at tag v0.5.5")
    args = p.parse_args()

    sys.path.insert(0, args.kvpress_eval_dir)
    from evaluate_registry import DATASET_REGISTRY, SCORER_REGISTRY

    data_dirs = args.data_dir or {"ruler": ["4096"], "longbench": LONGBENCH_EN, "toolout": ["toolout_4096"]}[args.dataset]
    scorer = SCORER_REGISTRY["longbench" if args.dataset == "longbench" else "ruler"]
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    df = load_df(args.dataset, data_dirs, args.per_task, args.seed, DATASET_REGISTRY)
    tok = AutoTokenizer.from_pretrained(args.model)
    if args.truncate_tokens:
        def middle_truncate(text):
            ids = tok.encode(text, add_special_tokens=False)
            if len(ids) <= args.truncate_tokens:
                return text
            half = args.truncate_tokens // 2
            return tok.decode(ids[:half]) + tok.decode(ids[-half:])

        df["context"] = df["context"].apply(middle_truncate)
    if args.true_window:
        df["question_len"] = df["question"].apply(lambda q: question_window(tok, q))
    # query-aware compression: the question is part of the compressed context
    df["context"] = df["context"] + df["question"]
    df["question"] = ""
    print(f"{len(df)} samples, {df['context'].nunique()} contexts", flush=True)

    pipe = pipeline("kv-press-text-generation", model=args.model, device="cuda:0", model_kwargs={"dtype": "auto"})
    pipe.model.eval()

    for name in args.methods:
        for setting in (args.budgets or args.ratios):
            if name == "full" and setting != (args.budgets or args.ratios)[0]:
                continue
            label = f"b{setting}" if args.budgets else f"{setting:.4f}"
            out_dir = Path(args.out) / args.model.replace("/", "--") / args.dataset / f"{name}__{label}"
            if (out_dir / "metrics.json").exists():
                print(f"skip {out_dir}", flush=True)
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            press = METHODS[name]()
            if press is not None and not args.budgets:
                press.compression_ratio = setting
            preds, ratios = {}, {}
            for context, g in df.groupby("context", sort=False):
                if args.budgets and press is not None:
                    kw = pipe._sanitize_parameters(questions=g["question"].tolist(),
                                                   answer_prefix=g["answer_prefix"].iloc[0])[0]
                    n = pipe.preprocess(context, **kw)["context_ids"].shape[1]
                    # keep exactly `setting` tokens per layer and KV head (+0.5 against float rounding)
                    press.compression_ratio = 0.0 if n <= setting else 1 - (setting + 0.5) / n
                    ratios.update({idx: press.compression_ratio for idx in g.index})
                if args.true_window:
                    set_window(press, int(g["question_len"].iloc[0]))
                with torch.inference_mode():
                    out = pipe(context, questions=g["question"].tolist(), answer_prefix=g["answer_prefix"].iloc[0],
                               press=press, max_new_tokens=int(g["max_new_tokens"].iloc[0]))
                preds.update(zip(g.index, out["answers"]))
            res = df.drop(columns=["context"]).copy()
            res["predicted_answer"] = pd.Series(preds)
            res["compression_ratio"] = pd.Series(ratios) if args.budgets else (setting if press else 0.0)
            res.to_csv(out_dir / "predictions.csv", index_label="sample")  # sample: row of the full test set
            if args.dataset == "longbench":
                metrics = {task: scorer(g.copy()) for task, g in res.groupby("task")}
            else:
                metrics = scorer(res.copy())
            json.dump(metrics, open(out_dir / "metrics.json", "w"), indent=2)
            flat = [v["string_match"] if isinstance(v, dict) and "string_match" in v else v for v in metrics.values()]
            flat = [v for v in flat if isinstance(v, (int, float))]
            print(f"{name} {label}: mean over tasks {np.mean(flat):.2f}", flush=True)
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
