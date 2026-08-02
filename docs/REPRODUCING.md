# Reproducing the paper end-to-end

This walks from a clean checkout to every table and figure in the paper. Stages are independent —
you can stop after any of them. Commands assume the repo root as the working directory and the
`envs/` virtualenv activated.

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

## 1. Environment

- Python 3.10, Java 11+ (Pyserini/Lucene), CUDA GPU for vLLM serving and dense encoding.
- `pip install -r requirements.txt`. The BQL core and local scoring need only the standard
  library + pytest; Pyserini, sentence-transformers, FAISS, and vLLM are needed for the
  paper-scale runs.
- Sanity check: `pytest tests/` (BQL parser/executor and scoring unit tests).

## 2. Data

Pull the published flat/structured twins from Hugging Face into `data/` (see the dataset table in
the top-level README), or rebuild them from scratch with the pipelines in `corpus_build/`:

- `corpus_build/wikipedia/` builds the HotpotQA and MuSiQue twins by matching benchmark documents
  against `wikimedia/structured-wikipedia` (native sections; no LLM involved).
- `corpus_build/browsecomp_plus/` builds the BrowseComp-Plus twin from the shipped frontmatter
  plus a one-time LLM sectioning batch; the published `sections.jsonl` lets you re-assemble the
  corpus **without** re-running that batch.

Each folder's README has the exact commands. Dataset names used throughout the harness are
registered in `evaluation/datasets.py`; the paper's BrowseComp-Plus experiments use
`browsecomp_plus_structured_full` / `browsecomp_plus_flat_full` (the complete 100,195-document
collection).

## 3. Indexes

```bash
bash scripts/build_indexes.sh
```

builds, per dataset: the Lucene structured index (BQL executor + structured backend), the
Pyserini BM25 index, and the dense embedding cache (default encoder `BAAI/bge-base-en-v1.5`,
1024-token sequence length). Alternative dense encoders for the sensitivity study are built with
`scripts/embed_full.sbatch` (`EMBED_MODEL=<hf-id>`); each lands in its own cache under
`indexes/dense/<model>-sl1024/<dataset>/`, so encoders never overwrite each other.

## 4. Serving the backbone

All paper runs use local vLLM serving. The primary backbone is
`Alibaba-NLP/Tongyi-DeepResearch-30B-A3B` at a 131,072-token context window; transfer backbones
(`Qwen-AgentWorld-35B-A3B`, `OpenResearcher/OpenResearcher-30B-A3B`) serve at 262,144. The
launchers in `scripts/` (`run.sh`, `shard_cell.sh`) start their own server per job on a distinct
port; for interactive use, any OpenAI-compatible endpoint works via the standard vLLM flags.

## 5. Running evaluation cells

A **cell** = (dataset, model, condition). Conditions are the interface variants from Table 1;
the ones used in the paper:

| condition | interface |
|---|---|
| `agent_research_bm25` / `_dense` / `_hybrid` | Search–Visit (whole-document reading), 3 rankers |
| `agent_research_bm25_autoread` / `_dense_autoread` | Search–AutoRead |
| `agent_research_dci` / `agent_research_bm25_dci` | DCI / BM25-bounded DCI (RISE-style) |
| `agent_research_bm25_fetch_snip` / `_dense_fetch` / `_hybrid_fetch_snip` | Search–Fetch (section reading), 3 rankers |
| `agent_research_snip` / `_bql_donly_snip` / `_bql_dense_snip` | **Sieve** (Boolean-filtered BM25 / Dense / BM25+Dense) |
| `agent_research_bql_dense_fetch` | Sieve without snippets (ablation) |
| `agent_research_indri_snip` | Indri-executor comparison |
| one-shot floors | `scripts/oneshot_rag.py` (`runs/_oneshot/...`) |

Single cell, locally:

```bash
python -m evaluation.run_eval \
  --dataset browsecomp_plus_structured_full \
  --retriever agent_research_bql_dense_snip \
  --runs-dir runs/<tier> [--limit N] [--only-instances ids.txt]
```

Rows append to `runs/<tier>/agent/<dataset>/<model>/<condition>/rows.jsonl`; re-running resumes
(already-scored instance ids are skipped). The strict-Boolean ablation is the same Sieve
condition with `BQL_SOFT_FALLBACK=0`; the dense-encoder ladder swaps `DENSE_MODEL=<hf-id>`.

**Base configuration guard.** Every paper cell runs `MAX_VISIT_TOKENS=12000
MAX_SECTION_TOKENS=12000 BM25_BACKEND=pyserini STRUCTURED_BACKEND=lucene MAX_STEPS=100` with
`k=5` results per search, temperature 0.6, seed 42. The launchers pin these; if you launch
another way, export them explicitly and verify the first rows' recorded `env_knobs`/`max_steps`
before scaling up.

## 6. Sharded cluster runs

Full collections are run as SLURM arrays with per-shard vLLM servers:

```bash
DATASET=browsecomp_plus_structured_full RUNS_DIR=runs/<tier> \
CONDITION=agent_research_bql_dense_snip NUM_SHARDS=10 WORKERS=2 \
MODEL=Alibaba-NLP/Tongyi-DeepResearch-30B-A3B \
bash scripts/shard_cell.sh          # submitter: splits remaining ids, submits itself as an array
```

then, when all shards finish:

```bash
python scripts/merge_shards.py --runs-dir runs/<tier> --dataset <ds> \
  --condition <cond> --num-shards N [--model <model-id>]
```

Merging dedups by instance id into the canonical `rows.jsonl`. Verify after every merge: exact
n, unique ids, uniform `max_steps` across rows. `WORKERS` (vLLM concurrency) only affects
throughput, not results; long-context datasets want fewer workers.

## 7. Forced-answer recovery

Long BrowseComp episodes can exhaust their step budget without emitting an `<answer>`. The
recovery pass (`scripts/force_answer_backfill.py`, launcher
`scripts/force_answer_backfill.sbatch`) replays each empty-answer row's terminal context with an
assistant-prefill forced decode and writes `recovered_answers.jsonl` **next to** (never into)
`rows.jsonl`. It is idempotent and append-only; all downstream scoring overlays it
automatically. Run it after a BCP cell lands and before judging.

## 8. LLM-as-judge scoring

BrowseComp-Plus accuracy is the benchmark's LLM-judge verdict with an exact-match short circuit
(judge model configured in `evaluation/llm_judge.py`; wiki collections are exact-match only and
are never judged):

```bash
export OPENAI_API_KEY=...
python scripts/judge_cells.py --datasets browsecomp                 # registry cells
python scripts/judge_cells.py --datasets browsecomp --extra-cell runs/<tier>/agent/<ds>/<model>/<cond>   # non-registry (e.g. transfer backbones)
```

Judgments cache per cell as sidecar files; re-running only fills gaps. `scripts/judge_daemon.sh`
wraps this in a periodic loop for long campaigns.

## 9. Tables, statistics, figures

Statistics: paired exact McNemar for accuracy, paired t-tests for tokens/calls, implemented in
`scripts/compare_cells.py` and reused everywhere — no metric logic is duplicated.

`analysis/make_paper_tables.py` recomputes every numeric table cell of the paper from `runs/`
(`--check` diffs against the paper's LaTeX tables, `--emit` regenerates them, `--selftest`
cross-checks the metric path). It reads the paper's table sources from `Boolean_agent_paper/tables/`
at the repo root; that tree is not part of this code release, so run these commands with the
paper's table files in place (released with the camera-ready). The paper's figure scripts draw
from the identical loaders.

## 10. Configuration reference

| knob | paper value | where |
|---|---|---|
| results per search (`k`) | 5 | run config |
| read ceiling (visit & section) | 12,000 tokens | `MAX_VISIT_TOKENS` / `MAX_SECTION_TOKENS` |
| step cap | 100 | `MAX_STEPS` |
| temperature / seed | 0.6 / 42 | run config |
| BM25 backend | `pyserini` (Lucene) | `BM25_BACKEND` |
| structured/BQL backend | `lucene` | `STRUCTURED_BACKEND` |
| default dense encoder | `BAAI/bge-base-en-v1.5` | `DENSE_MODEL` |
| Boolean soft fallback | on (paper default) | `BQL_SOFT_FALLBACK` (0 = strict ablation) |
| snippet length | 25 tokens | condition config |

Every run directory records its resolved configuration (`config.json`, per-row `env_knobs`) —
trust what the worker recorded over what you intended to launch.
