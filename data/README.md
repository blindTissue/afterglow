# Tool-output retrieval data

`toolout_4096.parquet` is the exact file used in the paper: 300 questions over real command outputs, 100 per source.
An assistant has run a command and must copy one exact string from its output. Same columns as kvpress' RULER data
(`context`, `question`, `answer_prefix`, `answer`, `task`, `max_new_tokens`, plus `depth`, the relative position of
the target record in the context), scored with RULER's string match.

| task | command output | question | answer |
|---|---|---|---|
| `tool_issues` | GitHub API response (JSON array) for issues of `huggingface/datasets` | "What is the title of issue #N?" | the issue title |
| `tool_gitlog` | default `git log` of `psf/requests` (author names, no e-mail) | "What is the message of commit <7-char hash>?" | the commit subject, without a trailing `(#PR)` |
| `tool_grep` | `grep -rn "def "` over one installed package directory | "What is the signature of the function `name`?" | `name(parameters)` |

## Construction (`make_toolout.py`)

- Each context is one sentence saying that it is the output of a command an assistant ran, the command line, and a
  contiguous slice of the command's output of about 4K Llama-3.1 tokens (3,896 tokens of output; 200 tokens are left
  for the question and chat template).
- The target record is at a uniformly random position in the slice. A sample is kept only if its key (issue number,
  commit hash, function name) and its answer each occur exactly once in the slice, and the answer has at least four
  words (for signatures, at least two parameters besides `self`).
- The prompt ends with the question and "Answer with the exact text from the output."; the answer prefix is
  "Answer:".
- Seeds: 0, 1 and 2 for the three sources.
- No design choice of Afterglow was made on this task.

## Sources

- **Issues:** the public Hugging Face dataset `lewtun/github-issues` (issues and pull requests of
  `huggingface/datasets`, numbers 1 to 2862 appear), rendered as a GitHub API response: number, title, user login,
  state, creation time, number of comments, labels, URL. Newest first.
- **Git log:** the public history of `psf/requests` (commits from 2011 to July 2026 appear), with
  `--date=format:%a %b %d %H:%M:%S %Y`. Only author names are included, no e-mail addresses. Some commit subjects
  contain e-mail-like strings verbatim.
- **Grep:** `grep -rn --include=*.py "def "` over `datasets`, `pandas/core`, `torch/nn` and `accelerate`. The script
  also lists `transformers/generation` and `transformers/pipelines`, but no sample from them passed the filters. We
  used transformers 5.2.0, datasets 5.0.1, pandas 2.3.3, torch 2.14.0 and accelerate 1.15.0. Line numbers and
  signatures depend on these versions.

Rebuilding from the sources gives a different file unless the `psf/requests` history and the package versions match.
Use the released parquet to reproduce the paper.

All contents are public. The code and history come from open-source projects under their own licenses (`psf/requests`
and the `datasets` and `accelerate` packages: Apache-2.0; `pandas` and `torch`: BSD-3-Clause). The issue metadata is
public GitHub content, taken as distributed in `lewtun/github-issues`.
