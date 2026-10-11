"""
Tool-output retrieval: an agent has run a command and must return an exact string from its output. Real data only:
  tool_issues  GitHub API response (JSON) of huggingface/datasets issues (`lewtun/github-issues`): the title of issue #N
  tool_gitlog  `git log` of psf/requests (author names only, no e-mails): the message of commit <7-char hash>
  tool_grep    `grep -rn "def "` over installed Python packages: the signature of a function, name + parameters
Each context is a contiguous slice of the tool output of about --tokens tokens (Llama-3.1 tokenizer, the RULER-4K
scale); the target record sits at a uniformly random position, its key occurs once, and its answer (>= 4 words or, for
grep, >= 2 parameters) occurs once and verbatim. Same columns as RULER, scored with the RULER scorer (string match).

    git clone --bare https://github.com/psf/requests requests.git
    python make_toolout.py --gitdir requests.git --site <python site-packages directory> --out toolout_4096.parquet

The released toolout_4096.parquet is the exact file used in the paper. Rebuilding it gives a different file unless
the psf/requests history and the installed package versions match (see data/README.md).
"""

import argparse
import ast
import json
import random
import re
import subprocess
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer

PREFACE = "Below is the output of a command that an assistant ran while working on a task.\n\n"
SUFFIX = "Answer with the exact text from the output."
PACKAGES = ["transformers/generation", "transformers/pipelines", "datasets", "pandas/core", "torch/nn", "accelerate"]


def issues_records():
    d = load_dataset("lewtun/github-issues", split="train").to_pandas().sort_values("number", ascending=False)
    recs = []
    for r in d.itertuples():
        user = r.user["login"] if isinstance(r.user, dict) else ast.literal_eval(str(r.user))["login"]
        labels = [lab["name"] for lab in r.labels] if len(r.labels) else []
        created = pd.to_datetime(int(r.created_at), unit="ms").strftime("%Y-%m-%dT%H:%M:%SZ")
        rec = {"number": int(r.number), "title": r.title, "user": user, "state": r.state, "created_at": created,
               "comments": len(r.comments) if r.comments is not None else 0, "labels": labels,
               "html_url": r.html_url}
        text = "  " + json.dumps(rec, indent=2, ensure_ascii=False).replace("\n", "\n  ")
        recs.append(dict(text=text, key=f"\"number\": {r.number},", answer=r.title.strip(),
                         question=f"What is the title of issue #{r.number}?"))
    return "$ curl -s 'https://api.github.com/repos/huggingface/datasets/issues?state=all&per_page=100'\n", recs, "[\n", \
        ",\n", "\n]"


def gitlog_records(gitdir):
    fmt = "%H%x1f%an%x1f%ad%x1f%s"
    out = subprocess.run(["git", "-C", gitdir, "log", "--date=format:%a %b %d %H:%M:%S %Y", f"--format={fmt}"],
                         capture_output=True, text=True, check=True).stdout
    recs = []
    for line in out.splitlines():
        h, author, date, subject = line.split("\x1f")
        text = f"commit {h}\nAuthor: {author}\nDate:   {date}\n\n    {subject}\n"
        answer = re.sub(r"\s*\(#\d+\)$", "", subject).strip()  # the PR number is not required
        recs.append(dict(text=text, key=h[:7], answer=answer, question=f"What is the message of commit {h[:7]}?"))
    return "$ git log\n", recs, "", "\n", ""


def grep_records(site):
    recs = []
    for pkg in PACKAGES:
        root = Path(site) / pkg
        out = subprocess.run(["grep", "-rn", "--include=*.py", "def ", "."], cwd=root, capture_output=True,
                             text=True).stdout
        for line in sorted(out.splitlines()):
            m = re.match(r"(\./[^:]+):(\d+):(\s*)def (\w+)\((.*)\)( -> [^:]+)?:\s*$", line)
            if not m:
                continue
            name, params = m.group(4), m.group(5)
            path = f"{pkg}/{m.group(1)[2:]}"
            answer = f"{name}({params})"
            recs.append(dict(text=f"{path}:{m.group(2)}:{m.group(3)}def {answer}{m.group(6) or ''}:", key=name,
                             answer=answer, question=f"What is the signature of the function `{name}`?",
                             ok=len([p for p in params.split(",") if p.strip() and p.strip() != "self"]) >= 2,
                             block=pkg))
    return "$ grep -rn \"def \" {block}/\n", recs, "", "\n", ""


def build(task, header, recs, opening, sep, closing, tok, n, target, rng, max_new_tokens):
    rows, tries = [], 0
    ntok = [len(tok.encode(r["text"] + sep, add_special_tokens=False)) for r in recs]
    while len(rows) < n:
        tries += 1
        assert tries < 100 * n, f"{task}: too few valid samples"
        start = rng.randrange(len(recs))
        end, total = start, len(tok.encode(PREFACE + header + opening + closing, add_special_tokens=False))
        while end < len(recs) and total + ntok[end] <= target and recs[end].get("block") == recs[start].get("block"):
            total += ntok[end]
            end += 1
        window = recs[start:end]
        if total < 0.9 * target:
            continue
        t = rng.randrange(len(window))
        rec = window[t]
        if not rec.get("ok", len(rec["answer"].split()) >= 4):
            continue
        body = opening + sep.join(r["text"] for r in window) + closing
        if body.count(rec["key"]) != 1 or body.count(rec["answer"]) != 1:
            continue
        head = header.format(block=rec["block"]) if "block" in rec else header
        rows.append(dict(context=PREFACE + head + body + "\n\n", question=f"{rec['question']} {SUFFIX}",
                         answer_prefix="Answer:", answer=[rec["answer"]], task=task, max_new_tokens=max_new_tokens,
                         depth=t / max(1, len(window) - 1)))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gitdir", required=True)
    p.add_argument("--site", required=True, help="site-packages directory with transformers, datasets, pandas, torch, "
                                                 "accelerate installed")
    p.add_argument("--tokenizer", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--tokens", type=int, default=4096)
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    rows = []
    for i, (task, src) in enumerate((("tool_issues", issues_records()), ("tool_gitlog", gitlog_records(args.gitdir)),
                                     ("tool_grep", grep_records(args.site)))):
        rng = random.Random(args.seed + i)
        rows += build(task, *src, tok, args.n, args.tokens - 200, rng, 64)  # 200: question and chat template
        print(task, sum(r["task"] == task for r in rows), flush=True)
    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out)
    print(df.groupby("task").context.apply(lambda c: c.str.len().mean()).round(), "->", args.out)


if __name__ == "__main__":
    main()
