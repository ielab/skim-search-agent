# Reproducing the paper end-to-end

This document goes from a fresh clone to every table and figure in the paper. The stages are
independent, so stop after any one of them. Every command assumes the repository root as the
working directory, with the `envs/` virtualenv activated.

Contents:
1. [Environment](#1-environment)
2. [Data](#2-data)
3. [Indexes](#3-indexes)
4. [Serving the backbone](#4-serving-the-backbone)
5. [Running evaluation cells](#5-running-evaluation-cells)
6. [Sharded cluster runs](#6-sharded-cluster-runs)
7. [Forced-answer recovery](#7-forced-answer-recovery)
8. [LLM-as-judge scoring](#8-llm-as-judge-scoring)
9. [Tables, statistics, figures](#9-tables-statistics-figures)
10. [Configuration reference](#10-configuration-reference)

---

> **The short version.** Every paper setting already exists as a complete experiment file under
> [`configs/paper/`](../configs/paper/), with every knob spelled out.
> `skimsearchagent run configs/paper/hotpotqa_structured_sieve.yaml` does in one command what the
> longer invocations below do by hand, and the file's contents land in the run's `config.json`.
> Run `skimsearchagent validate configs/paper/hotpotqa_structured_sieve.yaml` first. It reports
> what is missing (dense cache, Lucene index, Java, API keys) before any GPU time is spent.
> `skimsearchagent` is the installed launcher; `python run.py` is the same entry point from a
> source checkout.

## 1. Environment

Python 3.10, a JDK 21 or newer, and a CUDA GPU for vLLM serving and dense encoding.

The Java version matters. Pyserini 1.2 and Lucene require the `jdk.incubator.vector` module, and
on an older JVM the process dies with no Python traceback. Check `java -version` first. Under
conda, `conda install "openjdk>=21"` into the environment works; set
`JAVA_HOME=$CONDA_PREFIX/lib/jvm` if activation does not set it.

```bash
pip install -r requirements.txt
```

The BQL core and the local scoring path need only the standard library plus pytest. Pyserini,
sentence-transformers, FAISS and vLLM are required for the paper-scale runs.

API keys and every knob mentioned below are templated in [`.env.example`](../.env.example). Copy
it to `.env`, fill in the values in use, then `set -a; source .env; set +a`. Do not commit the
filled-in copy.

Quick sanity check:

```bash
python -m pytest tests/
```

That covers the BQL parser, the executor and the scoring units. Once the Pyserini JVM starts it
swallows pytest's terminal output, so pass `--junitxml=report.xml` to read the results.

**The token ruler.** Every cap in the library is a model-token count on tiktoken's `o200k_base`
(see CONFIGURATION.md). tiktoken downloads that encoding once and caches it in the process's
temp directory, which on a compute node is empty and offline. Seed the shared cache once from
a node with network access:

```bash
python -m agent_search.tokens --seed     # copies the file into indexes/tiktoken_cache/
```

Every shard refuses to start on the whitespace fallback (`AGENT_SEARCH_REQUIRE_TIKTOKEN=1`,
set by `scripts/shard_cell.sh`), and each run record names the ruler it used (`token_ruler`
in `config.json`). A cell whose record says `whitespace` was cut in words, not tokens, and is
not comparable to the others.

## 2. Data

The paper's corpora are published on Hugging Face as `wshuai190/browsecomp-plus-structured-full`,
`wshuai190/hotpotqa-structured` and `wshuai190/musique-structured`. Pulling them into `data/` is
faster than rebuilding. [`corpus_build/README.md`](../corpus_build/README.md) has the
download-and-stage commands.

To rebuild from scratch, `corpus_build/` has both pipelines:

- `corpus_build/wikipedia/` builds the HotpotQA and MuSiQue twins by matching benchmark documents
  against `wikimedia/structured-wikipedia`. The sections are native to that mirror, so no model is
  involved.
- `corpus_build/browsecomp_plus/` builds the BrowseComp-Plus twin from each document's own
  frontmatter plus one LLM sectioning batch. The published `sections.jsonl` re-assembles the
  corpus without repeating that batch.

Each folder's README has the exact commands. Dataset names are registered in
`agent_search/evaluation/datasets/`. The paper's BrowseComp-Plus experiments use
`browsecomp_plus_structured_full` and `browsecomp_plus_flat_full`, which is the complete
100,195-document collection (the plain `browsecomp_plus_structured` / `_flat` pair is the smaller
67,707-document pooled corpus the local builder produces).

## 3. Indexes

```bash
bash scripts/build_indexes.sh
```

For each dataset that builds three things: the Lucene structured index that the BQL executor and
the structured backend read, the Pyserini BM25 index, and the dense embedding cache. The default
encoder is `BAAI/bge-base-en-v1.5` at a 1024-token sequence length. Select a dataset and retriever
with `DATASET=` and `RETRIEVER=`; the script's header comment lists the rest.

The alternative encoders for the sensitivity study go through `scripts/embed_full.sbatch` with
`EMBED_MODEL=<hf-id>`. Each one lands in its own cache at
`indexes/dense/<model>-sl1024/<dataset>/`, so encoders never overwrite each other.

## 4. Serving the backbone

Every paper run serves its model locally with vLLM. The primary backbone is
`Alibaba-NLP/Tongyi-DeepResearch-30B-A3B`. The transfer backbones are `Qwen-AgentWorld-35B-A3B`
(serve with `--gdn-prefill-backend triton`) and `OpenResearcher/OpenResearcher-30B-A3B`
(`--trust-remote-code`). Every backbone runs the paper prompt unchanged, the same 98,304-token
serving window and the same 86,000-token history budget, so the backbone is the only thing that
changes between rows.

Five more backbones have experiment files under `configs/paper/` for the two baselines
(`browsecomp_plus_structured_search_visit_*.yaml`, `..._search_fetch_*.yaml`): Qwen3.5 4B, 9B
and 27B (thinking on, `--tool-call-parser qwen3_xml --reasoning-parser qwen3`) and gpt-oss 20b
and 120b (native function calling through the Responses driver, `--tool-call-parser openai
--reasoning-parser openai_gptoss`, `GPU_UTIL=0.8`, two GPUs for 120b). gpt-oss serving reads its
vocabulary from `TIKTOKEN_ENCODINGS_BASE`, which the shard script sets and `python -m
agent_search.tokens --seed` fills.

`scripts/run.sh` and `scripts/shard_cell.sh` start their own server per job on a distinct port, so
no manual server management is needed. For interactive work, any OpenAI-compatible endpoint works:
point `--api-base` at it and pass `--backend api`.

## 5. Running evaluation cells

A **cell** is one (dataset, model, condition) triple. Conditions are the interface variants from
Table 1. The paper uses these:

| condition | interface |
|---|---|
| `agent_research_bm25` / `_dense` / `_hybrid` | Search-Visit (whole-document reading), 3 rankers |
| `agent_research_bm25_autoread` / `_dense_autoread` | Search-AutoRead |
| `agent_research_dci` / `agent_research_bm25_dci` | DCI / BM25-bounded DCI (RISE-style) |
| `agent_research_bm25_fetch_snip` / `_dense_fetch` / `_hybrid_fetch_snip` | Search-Fetch (section reading), 3 rankers |
| `agent_research_snip` / `_bql_donly_snip` / `_bql_dense_snip` | **Sieve** (Boolean-filtered BM25 / Dense / BM25+Dense) |
| `agent_research_bql_dense_fetch` | Sieve without snippets (ablation) |
| `agent_research_indri_snip` | Indri-executor comparison |
| one-shot floors | the `rag_bm25`, `rag_dense` and `rag_hybrid` strategies (`skimsearchagent run`) |

### Retrieval floors on BrowseComp-Plus structured

The floors rank once with the raw question and no agent (`strategy=bm25`, `dense`, `hybrid`,
`reranked`; `python scripts/checks_matrix.py write_floors` writes the files, `scripts/slurm/floors.sbatch`
runs them). All 830 questions, seed 42, gold documents are the questions' evidence documents;
`recall@10` is the share of gold documents in the top 10, `hit@k` whether any gold document is
in the top k. These are the numbers a fresh clone of `dev` reproduces.

| floor | encoder or reranker | recall@10 | hit@5 | hit@10 |
|---|---|---|---|---|
| `bm25` | Lucene BM25 (Pyserini, k1=0.9, b=0.4) | 0.027 | 0.036 | 0.049 |
| `dense` | `BAAI/bge-base-en-v1.5` | 0.092 | 0.102 | 0.152 |
| `dense` | `ielabgroup/ITER-Qwen3-Embedding-0.6B` (i9 query format, weights of 2026-09-11) | 0.215 | 0.277 | 0.359 |
| `dense` | `ielabgroup/ITER-Qwen3-Embedding-4B` (i9 query format) | 0.339 | 0.435 | 0.527 |
| `hybrid` | BM25 + bge-base, RRF k=60, pools of 100 | 0.074 | 0.083 | 0.127 |
| `hybrid` | BM25 + ITER-0.6B, RRF k=60, pools of 100 | 0.161 | 0.173 | 0.281 |
| `reranked` | BM25 pool of 100, `BAAI/bge-reranker-v2-m3` | 0.046 | 0.060 | 0.080 |

A single-shot ranking barely reaches the evidence on this collection, which is why every
agent strategy searches many times. The ITER encoders are served the way they were trained
(`agent_search/retrievers/dense/decoder_encoder.py`: the end token kept, last position pooled,
documents at 512 tokens, queries at 8192) and queried in their i9 format with the reasoning field
empty, since a floor has no agent; the 0.6B checkpoint more than doubles bge-base's recall and the
4B checkpoint adds half again. Fusing BM25 into ITER by reciprocal rank lowers recall: BM25 is far weaker here and its
votes dilute the dense ranking.

### Full-set results

The full-set results table is being redone and is not in this page yet. Every paper condition is
rerun on one code version with the paper's result depth (k=5 on every search) before the table
returns here in one piece; partial or mixed-protocol numbers are not published.

To run one cell locally:

```bash
python -m agent_search.evaluation.run_eval \
  --dataset browsecomp_plus_structured_full \
  --retriever agent_research_bql_dense_snip \
  --runs-dir runs/<tier> [--limit N] [--only-instances ids.txt]
```

The paper's condition names carry the paper's prompt; the friendly names (`sieve`, `sieve_bm25`,
`search_visit`) run the same strategies under the library's default prompt, so a reproduction names
the condition (`skimsearchagent --help` lists both):

```bash
skimsearchagent dataset=browsecomp_plus_structured_full strategy=sieve runs_dir=runs/<tier>
```

**Two departures from the paper's prompts (0.3.1).** Both apply to Sieve and to the Search-Fetch
controls alike, so the comparison stays matched. First, `fetch` is declared as a rank and a section,
`{"rank": 1, "section": "Career"}`, in place of the paper's `{"specs": [[rank, section]]}`
list of pairs: the backbone mis-closed the nested list in a quarter of Sieve's fetch calls and
three quarters of Search-Fetch's, and 99.9% of the calls carried one pair. The tool also
resolves requests the paper's tool refused (`body` returns the whole document under the same
12,000-token cap as a visit, `infobox` returns the document's facts, a section that sits on
another row of the listing is read from there and the reply says so). Second, the BrowseComp Sieve
manual describes the sectioned corpus. The paper's manual said the corpus had no sections and
told the agent to fetch `body`, which the paper's appendix on instruction mismatches
acknowledges; half of Sieve's failed fetches on BrowseComp-Plus were those `body` requests.
The paper's manual is kept verbatim as the condition `agent_research_bql_dense_snip_papermanual`.
`tests/test_prompt_fidelity.py` pins every condition's prompt after these two changes.

Rows append to `runs/<tier>/agent/<dataset>/<model>/<condition>/rows.jsonl` as they finish, so
re-running the same command resumes and skips the instance ids that already scored. The
strict-Boolean ablation is the same Sieve condition with `BQL_SOFT_FALLBACK=0`. The dense-encoder
ladder swaps `DENSE_MODEL=<hf-id>`.

**Base configuration guard.** Every paper cell runs with `MAX_VISIT_TOKENS=12000
MAX_SECTION_TOKENS=12000` in the environment, plus
`--max-steps 100` on the command line (`max_steps=100` with the key=value launcher), `k=5` results
per search, temperature 0.6 and seed 42. Note `--max-steps`: `run_eval` defaults to 50, so a paper
run that omits the flag halves its step budget. The launchers and the `configs/paper/` files pin
all of this. When launching another way, export the knobs explicitly and check the first rows'
recorded `env_knobs` and `max_steps` before scaling up.

## 6. Sharded cluster runs

Full collections run as SLURM arrays with one vLLM server per shard:

```bash
DATASET=browsecomp_plus_structured_full RUNS_DIR=runs/<tier> \
CONDITION=agent_research_bql_dense_snip NUM_SHARDS=10 WORKERS=2 \
MODEL=Alibaba-NLP/Tongyi-DeepResearch-30B-A3B \
bash scripts/shard_cell.sh          # submitter: splits remaining ids, submits itself as an array
```

When every shard has finished:

```bash
python scripts/merge_shards.py --runs-dir runs/<tier> --dataset <ds> \
  --condition <cond> --num-shards N [--model <model-id>]
```

The merge dedups by instance id into the canonical `rows.jsonl`. Check three things after every
merge: the exact `n`, that ids are unique, and that `max_steps` is the same across all rows.
`WORKERS` is vLLM concurrency and only changes throughput, never results. Long-context datasets
need fewer workers.

## 7. Forced-answer recovery

Long BrowseComp episodes sometimes spend their whole step budget without emitting an `<answer>`.
The recovery pass replays each empty-answer row's terminal context with an assistant-prefill
forced decode:

```bash
python scripts/force_answer_backfill.py runs/<tier>/agent/<ds>/<model>/<cond>
```

It takes one or more cell directories as positional arguments, so a whole batch can be passed at
once. Add `--dry-run` to list what it would recover without spending anything.

The SLURM form is `scripts/force_answer_backfill.sbatch`. It writes `recovered_answers.jsonl`
**next to** `rows.jsonl`, never into it. It is idempotent and append-only, and everything
downstream overlays it automatically. Run it after a BrowseComp cell lands and before judging.

## 8. LLM-as-judge scoring

BrowseComp-Plus accuracy is the benchmark's own LLM-judge verdict with an exact-match short
circuit. The judge model defaults to `gpt-4o-mini` (`agent_search/evaluation/llm_judge.py`, or set
`LLM_JUDGE_MODEL`). The wiki collections are exact-match only and never get judged.

```bash
export OPENAI_API_KEY=...
python scripts/judge_cells.py --datasets browsecomp                 # registry cells
python scripts/judge_cells.py --datasets browsecomp --extra-cell runs/<tier>/agent/<ds>/<model>/<cond>   # non-registry (e.g. transfer backbones)
```

Judgments cache per cell as sidecar files, so re-running only fills the gaps. For a long campaign,
`scripts/judge_daemon.sh` wraps this in a loop that re-judges every 20 minutes by default.

## 9. Tables, statistics, figures

The statistics are paired exact McNemar for accuracy and paired t-tests for tokens and calls. Both
live in `scripts/compare_cells.py` and everything else calls into it, so no metric logic is
duplicated anywhere.

`analysis/make_paper_tables.py` recomputes every numeric table cell from `runs/`:

```bash
python analysis/make_paper_tables.py --check          # recompute and diff against the live .tex
python analysis/make_paper_tables.py --emit OUTDIR    # regenerate the tables into OUTDIR
python analysis/make_paper_tables.py --selftest       # cross-check the metric path
```

It reads the paper's table sources from `Boolean_agent_paper/tables/` at the repo root, and that
tree is not part of this code release. Run these with the paper's table files in place; they ship
with the camera-ready. The figure scripts read through the identical loaders.

## 10. Configuration reference

| knob | paper value | where |
|---|---|---|
| results per search | 5 | the `listing:` knobs, `BM25_VISIT_TOPK` / `DENSE_VISIT_TOPK` for the visit strategies and `BM25_FETCH_TOPK` / `DENSE_FETCH_TOPK` / `HYBRID_FETCH_TOPK` for the fetch strategies (all 5 by default since 0.3.1; the earlier fetch default of 10 is kept as the ablation `runs/bcp_s_fetch_k10`) |
| rank-metric cutoffs | 1, 3, 5, 10 | `--k` (report cutoffs only; it does not change what the agent sees) |
| read ceiling (visit and section) | 12,000 tokens | `MAX_VISIT_TOKENS` / `MAX_SECTION_TOKENS` |
| step cap | 100 | `--max-steps` (a run_eval flag, `max_steps=` in the launcher; there's no environment variable for it, and the default is 50) |
| temperature / seed | 0.6 / 42 | `--temperature` / `--seed` |
| BM25 and the structured index | Lucene (the only document engines) | built once with `skimsearchagent-build-indexes` |
| default dense encoder | `BAAI/bge-base-en-v1.5` | `DENSE_MODEL` |
| Boolean soft fallback | on | `BQL_SOFT_FALLBACK` (0 = strict ablation) |
| snippet length | 32 model tokens | `SNIPPET_TOKENS` (the paper's code cut 25 whitespace words with a character clip; the library cuts model tokens) |

Every run directory records what it resolved, in `config.json` and the per-row
`env_knobs`. When those disagree with the intended invocation, the recorded values are correct.
