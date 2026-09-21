# Reproducing the paper end to end

Copy and paste your way from `git clone` to a scored number. Every step is a command. Run them
from the repository root with the project environment active.

Two papers share this pipeline. Steps 1 to 8 cover both. [SIEVE.md](SIEVE.md) and
[ITER.md](ITER.md) add the few settings that differ.

Contents:
1. [Install](#1-install)
2. [Data](#2-data)
3. [Indexes](#3-indexes)
4. [Serve the backbone](#4-serve-the-backbone)
5. [Run one cell](#5-run-one-cell)
6. [Sharded cluster runs](#6-sharded-cluster-runs)
7. [Forced-answer recovery](#7-forced-answer-recovery)
8. [Judging and exact match](#8-judging-and-exact-match)
9. [Read the results](#9-read-the-results)
10. [Reference results](#10-reference-results)
11. [Configuration reference](#11-configuration-reference)

---

## 1. Install

**1.1 Clone and make an environment.** Python 3.10 or newer.

```bash
git clone https://github.com/ielab/skim-search-agent.git
cd skim-search-agent
python -m venv .venv && source .venv/bin/activate
```

**1.2 Install the package and the extras you need.**

```bash
python -m pip install -e ".[retrieval,api,eval,dev]"    # everything a document run needs
python -m pip install -e ".[serve]"                     # add this to serve a backbone with vLLM
```

`requirements.txt` holds the same pins as one flat list (`pip install -r requirements.txt`). The
extras are defined in `pyproject.toml`: `retrieval` (Pyserini, sentence-transformers, FAISS,
torch), `api` (the OpenAI client), `eval` (datasets, numpy, scipy, tqdm), `serve` (vLLM), `dev`
(pytest), `train` (retriever training, which needs its own environment because FlagEmbedding pins
an older transformers), `all`.

**1.3 Point `JAVA_HOME` at a JDK 21 or newer.** Pyserini 1.2 and Lucene need the
`jdk.incubator.vector` module. On an older JVM the process dies with no Python traceback, so check
this before anything else. Every document run and the test suite import Pyserini.

```bash
conda install -y "openjdk>=21"                  # or use a JDK already on the machine
export JAVA_HOME=$CONDA_PREFIX/lib/jvm
export PATH="$JAVA_HOME/bin:$PATH"
java -version                                   # must print 21 or higher
```

**1.4 Set the API keys you need.** Only the LLM judge and hosted backbones need one. A locally
served model needs no key, but the OpenAI client still wants a non-empty value.

```bash
cp .env.example .env
# edit .env, then:
set -a; source .env; set +a
```

`OPENAI_API_KEY` is used by the default judge (`gpt-4o-mini`) and by any `--model gpt-*`.
`GEMINI_API_KEY` is used by `--model gemini-*`. For a served backbone, export
`OPENAI_API_KEY=dummy`.

**1.5 Seed the token ruler.** Every cap in the library is a model-token count on tiktoken's
`o200k_base`. tiktoken downloads that encoding once into a temp directory, which is empty and
offline on a compute node. Seed the shared cache from a machine with network access:

```bash
python -m agent_search.tokens --seed
```

That writes `indexes/tiktoken_cache/` and `indexes/tiktoken_encodings/` (the second one is the
same vocabulary under the plain name `openai_harmony` reads when serving gpt-oss).
`scripts/shard_cell.sh` sets `AGENT_SEARCH_REQUIRE_TIKTOKEN=1`, so a shard refuses to start on the
whitespace fallback. Each run record names the ruler it used in `config.json` (`token_ruler`). A
cell whose record says `whitespace` was cut in words, not tokens, and isn't comparable to the
others.

**1.6 Check the install.**

```bash
python -m pytest tests/ --junitxml=report.xml
skimsearchagent run configs/smoke_doc_fixture_sieve_bm25.yaml   # three inline docs, no model, no keys
```

Pass `--junitxml` because the Pyserini JVM swallows pytest's terminal output once it starts.

## 2. Data

Every dataset is a directory under `data/`. Nothing is downloaded at run time.

**2.1 Pull the published corpora.** This is the fast path and it's what the paper ran.

```bash
huggingface-cli download wshuai190/browsecomp-plus-structured-full --repo-type dataset --local-dir data/_hf/bcp
cp -r data/_hf/bcp/structured data/browsecomp_plus_structured_full
cp -r data/_hf/bcp/flat       data/browsecomp_plus_flat_full

huggingface-cli download wshuai190/hotpotqa-structured --repo-type dataset --local-dir data/_hf/hotpotqa
cp -r data/_hf/hotpotqa/structured data/hotpotqa_structured
cp -r data/_hf/hotpotqa/flat       data/hotpotqa_flat

huggingface-cli download wshuai190/musique-structured --repo-type dataset --local-dir data/_hf/musique
cp -r data/_hf/musique/structured data/musique_structured
cp -r data/_hf/musique/flat       data/musique_flat
```

**2.2 The on-disk layout.** Each directory is BEIR-style. The loader wants `corpus.jsonl`,
`queries.jsonl` and `qrels/test.tsv` (it also accepts `qrels/dev.tsv`, `qrels.tsv` or
`qrels.jsonl`).

```
data/browsecomp_plus_structured_full/corpus.jsonl     6.1 GB   100,195 documents
data/browsecomp_plus_structured_full/queries.jsonl    517 KB   830 questions, answers inline
data/browsecomp_plus_structured_full/qrels/test.tsv
data/browsecomp_plus_structured_full/sections.jsonl   2.0 GB   the raw {_id, sections} map (optional)
data/hotpotqa_structured/                             1.0 GB   7,343 questions
data/musique_structured/                              378 MB   2,409 questions
```

`browsecomp_plus_structured` (without `_full`) is the smaller 67,707-document pooled corpus the
local builder produces, about 5.5 GB. The backbone-transfer cells run on that one; the headline
cells run on `_full`.

**2.3 The dataset names the harness knows.** Pass one of these to `--dataset` or `dataset.name`.

| name | what it is |
|---|---|
| `browsecomp_plus_structured_full` / `browsecomp_plus_flat_full` | the complete 100,195-document BrowseComp-Plus collection, structured and flat |
| `browsecomp_plus_structured` / `browsecomp_plus_flat` | the 67,707-document pooled pair |
| `hotpotqa_structured` / `hotpotqa_flat` | HotpotQA twin |
| `musique_structured` / `musique_flat` | MuSiQue twin |
| `2wiki_structured` / `2wiki_flat` | 2WikiMultiHopQA twin, not published, rebuild it |
| `browsecomp_plus_chunks`, `infoseek_eval`, `infoseek_train` | the ITER sets, see [ITER.md](ITER.md) |
| `doc_fixture`, `browsecomp_plus_fixture`, `hotpotqa_fixture`, `musique_fixture` | tiny inline fixtures for smoke runs |

`skimsearchagent-eval --help` prints the full list from the registry.

**2.4 Rebuild from scratch instead.** `corpus_build/` has both builders and each folder's README
has the exact commands. Run them from inside their own folder.

```bash
cd corpus_build/wikipedia
hf download wikimedia/structured-wikipedia --repo-type dataset --include "enwiki/data/*.parquet" --local-dir ./sw
python build.py build --dataset hotpotqa --sw-path ./sw --dump-misses hotpotqa_misses.txt
python build.py build --dataset musique  --sw-path ./sw --dump-misses musique_misses.txt
```

The wiki builder needs the flat staging at `data/<name>/corpus.jsonl` first
(`scripts/stage_multihop.py` writes it). The BrowseComp builder costs one `gpt-5.4-nano` Batch
sectioning pass, so prefer the published `sections.jsonl`:

```bash
cd corpus_build/browsecomp_plus
python build.py corpus --sections ../../data/_hf/bcp/sections.jsonl
```

`scripts/download_data.sh` only stages the SWE-bench code datasets. It exits with an error for
document corpora and points here.

## 3. Indexes

Three kinds of persistent index live under `indexes/`. Build only the ones your strategy reads.

| strategy | Lucene structured | Pyserini BM25 | dense cache |
|---|---|---|---|
| `sieve`, `sieve_dense`, `sieve_nosnip` | yes | no | yes |
| `sieve_bm25`, `indri` | yes | no | no |
| `search_visit`, `search_fetch`, `autoread`, `bounded_dci` | no | yes | no |
| `search_visit_dense`, `search_fetch_dense`, `autoread_dense`, `dedup_dense` | no | no | yes |
| `search_visit_hybrid`, `search_fetch_hybrid` | no | yes | yes |
| `dci` | no | no | no |

`skimsearchagent validate <file>` prints the same answer for one experiment file, so check there
if you're unsure.

**3.1 Build them.** One command per kind. All three are idempotent and skip work already on disk.

```bash
DS=browsecomp_plus_structured_full

# the Lucene structured index (BQL and Indri read it); needs JDK 21
skimsearchagent-build-indexes --dataset $DS --retriever search_lucene --index-root indexes

# the Pyserini BM25 index; needs JDK 21
skimsearchagent-build-indexes --dataset $DS --retriever bm25_pyserini --index-root indexes

# the dense embedding cache; wants a GPU
skimsearchagent-build-indexes --dataset $DS --retriever dense --model BAAI/bge-base-en-v1.5 --index-root indexes
```

`--retriever` takes exactly those three names. Add `--rebuild` to force a rebuild,
`--corpus-limit N` to cap the corpus. `python -m agent_search.retrievers.lucene.index_builder
--dataset $DS` is the structured builder's own entry point and does the same thing.

`scripts/build_indexes.sh` wraps the second and third of these and reads `DATASET=`,
`RETRIEVER=`, `DENSE_MODEL=`, `INDEX_ROOT=`, `REBUILD=`, `LIMIT=`, `CORPUS_LIMIT=` and `NSHARDS=`
from the environment.

**3.2 The same three as one SLURM job.**

```bash
sbatch --account=YOUR_ACCOUNT --partition=h24gpu --gres=gpu:1 \
  --export=ALL,DATASET=browsecomp_plus_structured_full,DENSE_MODEL=BAAI/bge-base-en-v1.5,WHICH=dense,lucene,pyserini \
  scripts/slurm/build_indexes.sbatch
```

`WHICH` defaults to `dense,lucene,pyserini`. Set it to a subset to build one kind.

**3.3 Where the artifacts land.** Roughly, for the 100k-document collection: 1.1 GB structured
index, 700 MB BM25 index, 200 MB dense cache with bge-base.

```
indexes/lucene_structured/<dataset>_v3/
indexes/bm25_pyserini/<dataset>/
indexes/dense/<model>-sl<seq_length>[-eos][-<dtype>]/<dataset>/
indexes/external/<name>/              # a prebuilt index you import, see ITER.md
```

Because the dense cache path carries the encoder, the sequence length and the precision, two
encoders never overwrite each other. Swap encoders with `DENSE_MODEL=<hf-id>` or
`retrieval.dense_model`. The encoder sweep for the sensitivity study goes through
`scripts/embed_full.sbatch`, which builds one cache over `browsecomp_plus_structured_full`:

```bash
sbatch --account=YOUR_ACCOUNT --export=ALL,EMBED_MODEL=Qwen/Qwen3-Embedding-0.6B scripts/embed_full.sbatch
```

## 4. Serve the backbone

Every paper run talks to an OpenAI-compatible server. Start one, note its port, then point the run
at it with `--api-base` (or `model.api_base`).

**4.1 The default backbone, Tongyi-DeepResearch-30B-A3B.**

```bash
export VLLM_USE_FLASHINFER_MOE_FP16=0     # load-bearing on H100/SM90: the FlashInfer CUTLASS MoE JIT fails there
vllm serve Alibaba-NLP/Tongyi-DeepResearch-30B-A3B \
  --tensor-parallel-size 1 --port 8000 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 131072 \
  --compilation-config '{"cudagraph_mode":"PIECEWISE"}'
```

Set `--max-model-len` to the experiment file's `agent.ctx_window`. The two Tongyi full-corpus
files use 131072; the backbone-transfer files use 98304.

**4.2 The other backbones.** Same command with the flags below added. Each one is what the
experiment file's `env.VLLM_ARGS` records.

| backbone | model id | tp | extra `vllm serve` flags |
|---|---|---|---|
| Qwen-AgentWorld | `Qwen/Qwen-AgentWorld-35B-A3B` | 1 | `--language-model-only --gdn-prefill-backend triton` |
| OpenResearcher | `OpenResearcher/OpenResearcher-30B-A3B` | 1 | `--trust-remote-code` |
| Qwen3.5 4B / 9B / 27B | `Qwen/Qwen3.5-4B`, `-9B`, `-27B` | 1 | `--enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3 --gdn-prefill-backend triton` |
| gpt-oss 20b | `openai/gpt-oss-20b` | 1 | `--enable-auto-tool-choice --tool-call-parser openai --reasoning-parser openai_gptoss`, with `--gpu-memory-utilization 0.8` |
| gpt-oss 120b | `openai/gpt-oss-120b` | 2 | the same as 20b |

`--language-model-only` is load-bearing for AgentWorld: its config declares a vision tower that
the checkpoint doesn't carry, and vLLM crashes on load without it. `--gdn-prefill-backend triton`
avoids the FlashInfer gated-delta-net kernel, which is JIT-compiled with nvcc and fails on CUDA
12.4. gpt-oss reads its vocabulary from `TIKTOKEN_ENCODINGS_BASE`, which step 1.5 fills and
`scripts/shard_cell.sh` exports.

**4.3 Let a script serve for you.** `scripts/run.sh`, `scripts/shard_cell.sh` and
`scripts/slurm/serve_and_run.sbatch` each start their own server on their own port and shut it
down on exit, so no manual server management is needed.

```bash
sbatch --account=YOUR_ACCOUNT --partition=h24gpu --gres=gpu:1 \
  --export=ALL,EXPERIMENT=configs/paper/hotpotqa_structured_sieve.yaml,MODEL=Alibaba-NLP/Tongyi-DeepResearch-30B-A3B,TP=1,MAX_MODEL_LEN=131072,VLLM_PYTHON=/path/to/vllm-env/bin/python \
  scripts/slurm/serve_and_run.sbatch
```

`VLLM_PYTHON` names the interpreter that has vLLM installed, when it isn't the active one.
`VLLM_EXTRA_ARGS` passes the flags from the table above. `OVERRIDES` passes
`section.key=value` overrides to the run.

**4.4 A hosted model instead.** `--model gpt-4o-mini` and `--model gemini-*` route to their own
APIs and need no server, just the key from step 1.4.

## 5. Run one cell

A cell is one (dataset, backbone, condition) triple.

**5.1 Validate the experiment file first.** It reports what's missing before any GPU time is
spent.

```bash
skimsearchagent validate configs/paper/browsecomp_plus_structured_full_sieve_tongyi.yaml
```

It prints the resolved `run_eval` command, the environment knobs, any keys the strategy doesn't
read, and the prerequisites: the dense cache, the Lucene index, Java 21, the API keys.

**5.2 Run it.**

```bash
skimsearchagent run configs/paper/browsecomp_plus_structured_full_sieve_tongyi.yaml
```

Add `section.key=value` arguments to change anything in the file for this run only. They're
recorded in `config.json`.

```bash
skimsearchagent run configs/paper/hotpotqa_structured_sieve.yaml \
  model.name=Alibaba-NLP/Tongyi-DeepResearch-30B-A3B \
  model.api_base=http://127.0.0.1:8000/v1 \
  output.runs_dir=runs/mine
```

`skimsearchagent` is the installed launcher. `python run.py` is the same entry point from a source
checkout.

**5.3 The paper's experiment files.** Each one is complete: every knob its strategy reads is
spelled out, and the file's content lands in the run's `config.json`.

| file pattern under `configs/paper/` | dataset | condition | backbone |
|---|---|---|---|
| `browsecomp_plus_structured_full_sieve_tongyi.yaml` | BCP full | Sieve, fused ranker | Tongyi |
| `browsecomp_plus_structured_full_search_visit_tongyi.yaml` | BCP full | Search-Visit, BM25 | Tongyi |
| `hotpotqa_structured_sieve.yaml`, `hotpotqa_structured_sieve_bm25.yaml`, `hotpotqa_structured_search_visit.yaml` | HotpotQA | Sieve fused, Sieve BM25, Search-Visit | none named, pass `model.name=` |
| `musique_structured_sieve.yaml` | MuSiQue | Sieve, fused | none named |
| `browsecomp_plus_structured_sieve_{agentworld,openresearcher}.yaml` | BCP pooled | Sieve, fused | the two transfer backbones |
| `browsecomp_plus_structured_search_visit_<backbone>.yaml` | BCP pooled | Search-Visit, BM25 | agentworld, openresearcher, qwen35_4b/9b/27b, gptoss_20b/120b |
| `browsecomp_plus_structured_search_fetch_<backbone>.yaml` | BCP pooled | Search-Fetch, hybrid | the same seven |

`configs/iter/` holds the ITER settings ([ITER.md](ITER.md)), `configs/samples/` holds small
sample runs, and `configs/smoke_*.yaml` runs on the fixtures with no model.

**5.4 The flag form.** `skimsearchagent-eval` is the same harness without the file.

```bash
skimsearchagent-eval \
  --dataset browsecomp_plus_structured_full \
  --retriever agent_research_bql_dense_snip \
  --policy llm --backend api --api-base http://127.0.0.1:8000/v1 \
  --model Alibaba-NLP/Tongyi-DeepResearch-30B-A3B \
  --max-steps 100 --temperature 0.6 --seed 42 \
  --workers 8 --k 1 3 5 10 \
  --runs-dir runs/mine
```

`python -m agent_search.evaluation.run_eval` is the same command. There's also a `key=value` form
that maps onto it and exports the environment knobs for you:

```bash
skimsearchagent dataset=browsecomp_plus_structured_full strategy=sieve \
  model=Alibaba-NLP/Tongyi-DeepResearch-30B-A3B api_base=http://127.0.0.1:8000/v1 \
  max_steps=100 snippet_tokens=32 runs_dir=runs/mine
```

**5.5 The conditions.** The paper's condition names carry the paper's prompt. The friendly names
run the same strategies under the library's default prompt, so a reproduction names the condition.
`skimsearchagent --help` lists both.

| paper condition (`--retriever agent_<name>`) | friendly name | interface |
|---|---|---|
| `research_bm25` / `research_dense` / `research_hybrid` | `search_visit` / `_dense` / `_hybrid` | Search-Visit, whole-document reading, 3 rankers |
| `research_bm25_autoread` / `research_dense_autoread` | `autoread` / `autoread_dense` | Search-AutoRead |
| `research_dci` / `research_bm25_dci` | `dci` / `bounded_dci` | DCI and BM25-bounded DCI (RISE-style) |
| `research_bm25_fetch_snip` / `research_dense_fetch` / `research_hybrid_fetch_snip` | `search_fetch` / `_dense` / `_hybrid` | Search-Fetch, section reading, 3 rankers |
| `research_snip` / `research_bql_donly_snip` / `research_bql_dense_snip` | `sieve_bm25` / `sieve_dense` / `sieve` | Sieve, Boolean-filtered BM25 / dense / fused |
| `research_bql_dense_fetch` | `sieve_nosnip` | Sieve without snippets (ablation) |
| `research_indri_snip` | `indri` | Indri-executor comparison |
| `rag_bm25` / `rag_dense` / `rag_hybrid` | same | one-shot floors: rank once, one model call |
| `bm25` / `dense` / `hybrid` / `reranked` | same | retrieval-only floors, no agent |

**5.6 The base configuration.** Every paper cell runs with these. The launchers and the
`configs/paper/` files pin all of it.

```bash
export MAX_VISIT_TOKENS=12000 MAX_SECTION_TOKENS=12000 SNIPPET_TOKENS=32
# on the command line: --max-steps 100 --temperature 0.6 --seed 42 --k 1 3 5 10
```

Watch `--max-steps`. `run_eval` defaults to 50, so a paper run that omits the flag halves its step
budget. `k` is 5 results per search, set by the `listing:` knobs. When you launch another way,
check the run's `config.json` (`env_knobs` and `max_steps`) before scaling up.

**5.7 Resume.** Rows append to the run directory as they finish, so re-running the same command
skips the instance ids that already scored. Re-running a *different* setting into the same
directory is refused unless you pass `--allow-config-drift`.

### Two departures from the paper's prompts (0.3.1)

Both apply to Sieve and to the Search-Fetch controls alike, so the comparison stays matched. First,
`fetch` is declared as a rank and a section, `{"rank": 1, "section": "Career"}`, in place of the
paper's `{"specs": [[rank, section]]}` list of pairs: the backbone mis-closed the nested list in a
quarter of Sieve's fetch calls and three quarters of Search-Fetch's, and 99.9% of the calls
carried one pair. The tool also resolves requests the paper's tool refused (`body` returns the
whole document under the same 12,000-token cap as a visit, `infobox` returns the document's facts,
a section that sits on another row of the listing is read from there and the reply says so).
Second, the Sieve manual is the reference manual: the query language, the fields, the fetch call
and worked examples. The paper's manual described an unsegmented corpus (it said there were no
sections and told the agent to fetch `body`, which the paper's appendix on instruction mismatches
acknowledges) and carried three advice sections on how to search, hop and avoid mistakes. A manual
ablation on BrowseComp-Plus (eight variants, all else matched) found every cut of that manual
scoring above it and the reference alone scoring highest, so the reference is the default and the
advice sections are ablation variants (`scripts/compose_manuals.py`, [SIEVE.md](SIEVE.md)).
`tests/test_prompt_fidelity.py` pins every condition's prompt after these two changes.

## 6. Sharded cluster runs

A full collection runs as a SLURM array, one vLLM server per shard. The submitter splits the
not-yet-scored ids into `NUM_SHARDS` files and submits itself as an array.

**6.1 Submit.** Export the whole block. A bare shell without the vLLM environment on `PATH` exits
127.

```bash
export DATASET=browsecomp_plus_structured_full
export RUNS_DIR=runs/mine                       # the TIER root, not the cell dir
export CONDITION=agent_research_bql_dense_snip  # an agent_* condition
export MODEL=Alibaba-NLP/Tongyi-DeepResearch-30B-A3B
export NUM_SHARDS=10
export WORKERS=8
export QOS=normal
export JOB_TIME=24:00:00
export MAX_STEPS=100
export SLURM_ACCOUNT=YOUR_ACCOUNT
export PREBUILD=0                               # build the indexes once yourself, step 3
bash scripts/shard_cell.sh
```

**6.2 Size `WORKERS` by dataset and tool family.** It's vLLM concurrency per shard, so it changes
throughput and never results. It's KV-bound: the live context per step is what the pages fill it
with. Measured on 24-question express smokes:

| family | live context | workers |
|---|---|---|
| wiki Search-Fetch, Sieve, bounded DCI | 15k | 12 |
| wiki Search-Visit | 24k | 12 |
| BrowseComp Search-Visit, Search-Fetch, Sieve | 25k | 8 |
| DCI | 35k to 45k, 81k at p90 | 4 |
| AutoRead | 60k to 75k | 2 |
| one-shot RAG | one call | 6 |

**6.3 Other knobs.** `TP` (GPUs per shard), `GPU_UTIL`, `VLLM_ARGS` (the serving flags from step
4.2), `MAX_MODEL_LEN`, `SNIPPET_TOKENS` (32), `DEDUP_SNIPPET_TOKENS` (64), `SEED` (42),
`TEMPERATURE` (0.6), `GPU_PARTITION` (h24gpu), `JOB_CPUS`, `JOB_MEM`, `INDEX_ROOT`,
`ARRAY_SPEC=3-9` to resubmit only some shards. `MAX_VISIT_TOKENS` and `MAX_SECTION_TOKENS` default
to 12000 inside the script and are carried into the array export.

**6.4 Merge when every shard has finished.**

```bash
python scripts/merge_shards.py --runs-dir runs/mine \
  --dataset browsecomp_plus_structured_full \
  --condition agent_research_bql_dense_snip --num-shards 10
```

It backs up the canonical `rows.jsonl`, dedups by instance id, and is idempotent. Add `--dry-run`
to see what it would fold in, and `--model <id>` when the model tag can't be auto-discovered.
Check three things after every merge: the exact `n`, that ids are unique, and that `max_steps` is
the same across all rows.

## 7. Forced-answer recovery

Long BrowseComp episodes sometimes spend the whole step budget without emitting an `<answer>`. The
recovery pass replays each empty-answer row's terminal context with an assistant-prefill forced
decode. Run it after a BrowseComp cell lands and before judging.

```bash
python scripts/force_answer_backfill.py --dry-run runs/mine/agent/<ds>/<model>/<cond>
python scripts/force_answer_backfill.py runs/mine/agent/<ds>/<model>/<cond>
```

It takes one or more cell directories, so a whole batch goes in one call. It writes
`recovered_answers.jsonl` next to `rows.jsonl` and never into it. It's idempotent and
append-only, and everything downstream overlays it automatically. The SLURM form is
`scripts/force_answer_backfill.sbatch`, which serves Tongyi on the node first.

## 8. Judging and exact match

**8.1 Exact match is automatic.** HotpotQA and MuSiQue are scored by `answer_em` and `answer_f1`
during the run (`agent_search/evaluation/doc_scoring.py`). They never get judged. The numbers are
already in `rows.jsonl` and `results.json` when the run ends.

**8.2 BrowseComp-Plus needs the judge.** It's the benchmark's own LLM verdict with an exact-match
short circuit, using the BrowseComp Appendix F prompt. The default model is `gpt-4o-mini`.

```bash
export OPENAI_API_KEY=...
skimsearchagent-judge --results-dir runs/mine/agent/<ds>/<model>/<cond> \
  --judge-model gpt-4o-mini --workers 16
```

That adds `judge_correct` to each row and writes `judge_summary.json` next to it. Other flags:
`--judge-api-base` for a served judge, `--judge-prompt` to pick a prompt, `--tag NAME` to keep a
second judge's verdicts beside the default's (`judge_summary_<tag>.json`), `--unfinished-ok` to
judge a run that has no `results.json` yet.

**8.3 Judge a whole campaign.** `scripts/judge_cells.py` walks the cell registry, caches verdicts
per cell keyed on the answer's hash, and only spends on answers it hasn't seen.

```bash
python scripts/judge_cells.py --datasets browsecomp --dry-run          # preview, no API calls
python scripts/judge_cells.py --datasets browsecomp --workers 16       # registry cells
python scripts/judge_cells.py --datasets browsecomp --extra-cell runs/mine/agent/<ds>/<model>/<cond>
```

`--extra-cell` judges a directory the registry can't express, such as a transfer-backbone cell.
`scripts/judge_daemon.sh` wraps the same call in a loop that re-judges every 20 minutes.

## 9. Read the results

**9.1 Where a run lands.** `<runs_dir>/<kind>/<dataset>/<model>/<retriever>/`, where `kind` is
`agent` for any `agent_*` retriever and `retrieval_only` otherwise, and `<model>` is the last path
segment of the model id.

```
runs/paper/agent/browsecomp_plus_structured_full/Tongyi-DeepResearch-30B-A3B/agent_research_bql_dense_snip/
  config.json               every resolved knob, env_knobs, token_ruler, prompt_sha256, git_rev
  rows.jsonl                one JSON object per question, the full trajectory included
  results.json              the aggregate: n, n_skipped, n_errors, and the mean of every row metric
  judge_summary.json        written by the judge (BrowseComp only)
  recovered_answers.jsonl   written by the recovery pass (BrowseComp only)
```

Steps, seed and `max_steps` are not in the path, so point an ablation on any of them at its own
`runs_dir`. One exception: `--seeds 0,1,2` adds a `seed=<N>` segment per seed, since that's a
variance band rather than one run. The published Sieve runs use a curated tree instead,
`runs/sieve/<dataset>/<strategy>/<agent>/<retriever>/`, with one cell per leaf.

**9.2 The row columns that matter.**

| column | meaning |
|---|---|
| `instance_id` | `<dataset>__<query id>` |
| `question`, `gold_answer`, `final_answer` | the question, the gold answer, what the agent answered |
| `answer_em`, `answer_f1` | exact match and token F1 against the gold answer |
| `grounded`, `grounded_em`, `grounded_f1` | the same, but zero unless a gold document was surfaced |
| `judge_correct`, `judge_extracted`, `judge_reasoning` | the LLM judge's verdict, what it read as the answer, and why, added by step 8 |
| `n_steps`, `llm_calls`, `tool_errors`, `stopped` | trajectory length, model calls, failed calls, why the episode ended |
| `total_tokens_once` | the episode total counted once, so a prompt resent next turn isn't counted twice |
| `prompt_tokens`, `completion_tokens`, `output_tokens`, `reasoning_tokens`, `cached_input_tokens` | the raw usage counters |
| `read_tokens`, `retrieved_doc_tokens`, `context_once_tokens` | how much document text entered the context |
| `recall@k`, `hit@k`, `acc@k`, `ndcg@k`, `map@k`, `mrr@10` | rank metrics over the documents the agent surfaced |
| `gold_ids`, `surfaced_docs`, `gold_doc_coverage` | the evidence documents, what the agent saw, the share it covered |
| `queries`, `actions`, `hits_per_step` | one entry per step |
| `trajectory` | every step with its action, arguments and full observation |
| `max_steps`, `tool_condition`, `retriever` | the setting this row ran under |

`results.json` carries `n`, `n_skipped`, `n_errors`, `level`, `rows_file` and a `metrics` object
holding the mean of every numeric row column plus `timeout_rate`, the share of episodes that ended
on a budget (`max_steps`, `ctx_budget` or `max_turns`). `judge_summary.json` carries `judge_model`,
`judge_prompt`, `n_judged`, `n_correct`, `n_judge_errors`, `n_newly_judged` and `judge_accuracy`.

**9.3 Tabulate and compare.** `scripts/summarize_runs.py` walks a runs tree and prints one table
per dataset, floors first, agents grouped by step budget and model.

```bash
python scripts/summarize_runs.py --dir runs/mine            # the tables
python scripts/summarize_runs.py --dir runs/mine --csv      # one flat row per condition
python scripts/summarize_runs.py --dir runs/mine --no-sig   # skip the t-tests, and skip reading the huge rows.jsonl files
```

Other flags: `--compare` adds cross-cutting tables, `--full` shows every metric column instead of
the curated set. `scripts/compare_cells.py` holds the statistics themselves, paired exact McNemar
for accuracy and paired t-tests for tokens and calls. Everything else calls into it, so no metric
logic is duplicated.

**9.4 Regenerate the paper's tables.**

```bash
PYTHONPATH=. python analysis/make_paper_tables.py --check          # recompute and diff against the live .tex
PYTHONPATH=. python analysis/make_paper_tables.py --emit OUTDIR    # regenerate the tables into OUTDIR
PYTHONPATH=. python analysis/make_paper_tables.py --selftest       # cross-check the metric path
```

It reads the paper's table sources from `Boolean_agent_paper/tables/` at the repository root, and
that tree isn't part of this code release. Run these with the paper's table files in place; they
ship with the camera-ready. The figure scripts read through the identical loaders.

## 10. Reference results

These are the numbers a fresh clone of `dev` reproduces. Nothing below changes how you run
anything; it's what the cells should score.

### Retrieval floors on BrowseComp-Plus structured

The floors rank once with the raw question and no agent (`strategy=bm25`, `dense`, `hybrid`,
`reranked`). Write the files with `python scripts/checks_matrix.py write_floors`, then run them
with `scripts/slurm/floors.sbatch`. All 830 questions, seed 42, gold documents are the questions'
evidence documents; `recall@10` is the share of gold documents in the top 10, `hit@k` whether any
gold document is in the top k.

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
4B checkpoint adds half again. Fusing BM25 into ITER by reciprocal rank lowers recall: BM25 is far
weaker here and its votes dilute the dense ranking.

### Full-set results

Every cell below is a full collection on one code version: 830 questions for BrowseComp-Plus
structured, 2,409 for MuSiQue, 7,343 for HotpotQA. The backbone is Tongyi-DeepResearch-30B-A3B
throughout. BrowseComp-Plus is scored by gpt-4o-mini with the BrowseComp Appendix F prompt, and the
two wiki collections by exact match. Steps are the mean trajectory length. Tokens are the episode
total counted once, so a prompt that gets resent on the next turn isn't counted twice.

| method | BCP acc | steps | tokens | MuSiQue EM | steps | tokens | HotpotQA EM | steps | tokens |
|---|---|---|---|---|---|---|---|---|---|
| Search-Visit, BM25 | 36.4 | 54.4 | 57.3k | 25.0 | 41.4 | 42.5k | 43.9 | 22.5 | 20.6k |
| Search-Visit, dense | 36.5 | 63.9 | 48.9k | 24.6 | 44.9 | 38.9k | 43.2 | 24.4 | 19.9k |
| Search-Visit, hybrid | 43.1 | 55.5 | 52.4k | 26.0 | 43.5 | 41.0k | 43.3 | 23.2 | 19.8k |
| Search-Fetch, BM25 | 39.2 | 59.2 | 48.7k | 26.7 | 42.4 | 32.2k | 42.9 | 25.1 | 16.1k |
| Search-Fetch, dense | 37.3 | 63.9 | 44.7k | 25.8 | 45.0 | 30.2k | 42.1 | 26.4 | 16.0k |
| Search-Fetch, hybrid | 45.1 | 56.0 | 45.3k | 26.2 | 43.5 | 30.6k | 43.8 | 25.6 | 15.9k |
| DCI | 20.0 | 34.8 | 86.1k | 27.0 | 28.1 | 58.1k | 43.8 | 16.7 | 32.5k |
| BM25-bounded DCI | 32.4 | 52.8 | 65.5k | 27.1 | 34.7 | 44.8k | 43.5 | 20.6 | 20.8k |
| Sieve, BM25 | 45.8 | 58.6 | 45.3k | 27.2 | 44.1 | 30.8k | 44.2 | 26.1 | 15.6k |
| Sieve, dense | 46.5 | 57.2 | 45.6k | 28.3 | 43.3 | 31.1k | 44.6 | 25.8 | 15.7k |
| Sieve, fused | 48.8 | 58.0 | 45.6k | 27.6 | 43.8 | 31.5k | 44.6 | 26.8 | 16.2k |

Read it by column. On BrowseComp-Plus the best Sieve beats the best baseline by 3.7 points at the
same token cost. On MuSiQue the margin is 1.2 points over bounded DCI, and it's dense Sieve rather
than the fused one that gets there. On HotpotQA every method lands between 42 and 45, so what
separates Sieve is the 15.6k tokens it spends against Search-Visit's 20.6k.

Which ranker wins depends on the collection. Fusion is worth 2.3 points over dense alone on
BrowseComp-Plus, and it's worth nothing on the two wiki collections, where dense alone ties or
wins. Run the ranker you can afford: on HotpotQA plain BM25 Sieve is within 0.4 of the fused one.

## 11. Configuration reference

| knob | paper value | where |
|---|---|---|
| results per search | 5 | the `listing:` knobs, `BM25_VISIT_TOPK` / `DENSE_VISIT_TOPK` for the visit strategies and `BM25_FETCH_TOPK` / `DENSE_FETCH_TOPK` / `HYBRID_FETCH_TOPK` for the fetch strategies (all 5 by default since 0.3.1; the earlier fetch default of 10 is kept as the ablation `runs/bcp_s_fetch_k10`) |
| rank-metric cutoffs | 1, 3, 5, 10 | `--k` (report cutoffs only; it does not change what the agent sees) |
| read ceiling (visit and section) | 12,000 tokens | `MAX_VISIT_TOKENS` / `MAX_SECTION_TOKENS` |
| step cap | 100 | `--max-steps` (a run_eval flag, `max_steps=` in the launcher; there's no environment variable for it, and the default is 50) |
| temperature / seed | 0.6 / 42 | `--temperature` / `--seed` |
| BM25 and the structured index | Lucene (the only document engines) | built once in step 3 |
| default dense encoder | `BAAI/bge-base-en-v1.5` | `DENSE_MODEL` |
| Boolean soft fallback | on | `BQL_SOFT_FALLBACK` (0 = strict ablation) |
| snippet length | 32 model tokens | `SNIPPET_TOKENS` (the paper's code cut 25 whitespace words with a character clip; the library cuts model tokens) |

Every run directory records what it resolved in `config.json`, under `env_knobs` and the flag keys
next to it. When those disagree with the intended invocation, the recorded values are correct. The
full knob list is in [CONFIGURATION.md](CONFIGURATION.md).
