# Training a retriever from run records

Every run is a full trajectory: the searches the agent issued, the documents it read, what it
generated after each read, and whether the answer was right. This page turns that record into a
dense retriever that conditions on the agent's history (the ITER recipe: memory-conditioned
queries, tiered negatives, a patched FlagEmbedding trainer) and plugs the result back into any
strategy. Heavy steps run as SLURM jobs.

```
runs/…/rows.jsonl ──build-triples──▶ train_data/*.jsonl ──train-retriever──▶ models/my-retriever
        ▲                                                                            │
        └──────────── skimsearchagent run … retrieval.dense_model=models/my-retriever ◀┘
```

## 1. Collect trajectories

Use runs where the agent reads documents, including wrong ones; the negatives come from those
reads. The paper configs are suitable:

```bash
sbatch --account=YOUR_ACCOUNT --partition=h24gpu --gres=gpu:2 \
    --export=ALL,EXPERIMENT=configs/paper/hotpotqa_structured_search_visit.yaml,MODEL=Alibaba-NLP/Tongyi-DeepResearch-30B-A3B \
    scripts/slurm/serve_and_run.sbatch
```

Rows written by this version carry `hit_ids` (what each search listed) and `read_ids` (what
each read opened) on every step. Older rows work; the listing is parsed from the observation
text.

## 2. Build the triples

```bash
skimsearchagent-build-triples \
    --runs runs/paper/agent/hotpotqa_structured/Tongyi-DeepResearch-30B-A3B/agent_research_bm25 \
    --dataset hotpotqa_structured --out train_data/hotpotqa_i2.jsonl \
    --query-style i2 --labeller oracle
```

`--runs` accepts run directories or `rows.jsonl` files. `--dataset` supplies the document texts;
positives and negatives are stored as full text, the format FlagEmbedding reads. A
`.summary.json` with counts is written next to the output. The command exits non-zero when no
example was produced.

For each search the agent issued, the builder writes one example if it has a positive and at
least one negative:

| field | content |
|---|---|
| `query` | the retriever query rendered from the agent's history at that search (style below) |
| `pos` | a document read for the first time after this search, judged relevant, and listed by some search in the episode |
| `neg_diversity` | documents read before this search and judged relevant (the retriever learns not to return them again) |
| `neg_hard` | documents read before this search and judged irrelevant |
| `neg_weak` | documents this search listed that the agent never read in the episode |
| `reweight_rate` | ITER's per-example weight, from the length of the agent's generation after reading the positive |

Relevance comes from a labeller. `oracle` uses the gold document ids in the run record. `answer`
checks whether the gold answer string appears in the document. `judge:<model>` is ITER's
protocol, an LLM classifying the agent's post-read generation; use it when there are no gold
ids (for example `judge:gpt-4o-mini`; the model is routed like any other).

Query styles (ITER's names):

| style | the retriever query contains |
|---|---|
| `plain` | the sub-query only |
| `i2` | main question, current sub-query, previous sub-queries (default) |
| `i3` / `i6` | `i2` plus the documents visited under each previous sub-query |
| `i4` / `i7` | `i3` plus the agent's cleaned notes on those documents |
| `i5` | main question, sub-query, notes only |
| `mem` / `docs` | ITER's earlier `[Q] / [Now] / [Memory]` and `[Prev]` formats |

The training style is the serving style. The builder records it in every example and the trainer
writes it into the checkpoint's serving note.

## 3. Train

Training runs FlagEmbedding's decoder-only embedder trainer with ITER's patch (three tier weights
in the InfoNCE loss and the per-example weight). FlagEmbedding 1.3.5 requires an older
`transformers` than the evaluation environment pins, so it gets its own environment. Create it
once on a node with internet:

```bash
python -m venv envs-train && source envs-train/bin/activate
pip install -e ".[train]"                      # FlagEmbedding 1.3.5, peft, accelerate
skimsearchagent-train-retriever patch          # applies agent_search/training/patches/*.patch (idempotent)
skimsearchagent-train-retriever check          # prints the FlagEmbedding location and patch state
```

Write a training file and submit the job:

```bash
skimsearchagent-train-retriever template > train.yaml   # every knob with ITER's defaults
# set train_data, output_dir, query_style (same as step 2), base_model
sbatch --account=YOUR_ACCOUNT --partition=h24gpu --gres=gpu:1 \
    --export=ALL,TRAIN=train.yaml,TRAIN_ENV=$PWD/envs-train scripts/slurm/train_retriever.sbatch
```

Defaults are the paper's: `Qwen/Qwen3-Embedding-0.6B`, learning rate 1e-6, 2 epochs, group size
10, batch 32, temperature 0.02, last-token pooling, normalised embeddings, tier weights 3.0 /
1.0 / 0.3, caps 3 / 3, 8192-token queries for history styles (512 for `plain`), 512-token
passages. The 0.6B model takes about nine hours on one H100.
`skimsearchagent-train-retriever run train.yaml --dry-run` prints the `torchrun` command without
running it.

The trainer writes `skimsearchagent_dense.json` into the output directory: the query
instruction, pooling, normalisation and lengths the model was trained with.
`skimsearchagent-train-retriever serving-note train.yaml` regenerates it from the config.

## 4. Use the checkpoint

```bash
skimsearchagent-build-indexes --dataset hotpotqa_structured --retriever dense --model models/my-retriever
skimsearchagent run configs/paper/hotpotqa_structured_sieve.yaml \
    retrieval.dense_model=models/my-retriever retrieval.dense_query_style=i2 output.runs_dir=runs/mine
```

The dense retriever reads the serving note, builds the same pooling pipeline, prefixes every query
with the trained instruction, and for any style other than `plain` renders the query from the
live episode's history with the same code the builder used. One model serves every dense arm:
Sieve's ranker and its fallback, the dense and hybrid baselines, Indri's dense belief. Both
knobs are recorded in `config.json` and are part of the run's identity.

## 5. Reproducing ITER

ITER (Zhou et al., github.com/ielab/ITER) trains a retriever from search-agent trajectories and
evaluates it inside the agent loop. Everything it needs is in the library; this section maps the
paper's setup onto files here.

### Strategy

ITER's agent has two tools: `search(query)` and `get_document(docid)`. A search over-fetches a
pool of 100, drops every document an earlier search already surfaced in this episode, and shows
the top 10 of the rest with a 64-token snippet. Documents that would have ranked but were shown
before are listed under "Already-seen" so the agent can reopen them. That is `strategy=dedup_dense`
(the run's dense model behind `search`) or `dedup_bm25`, with the task template
`agent_search/prompts/tasks/research_dedup.md`. Listing knobs: `listing.dedup_topk` (10),
`listing.dedup_pool_k` (100); snippet: `budgets.snippet_tokens: 64`; `get_document` returns at
most 512 tokens in ITER (`budgets.max_visit_tokens: 512`; the wiki chunks are 512 tokens, so this
only matters on a corpus with longer documents).

### Backbones

The backbone is `model.name` plus `model.backend`. ITER ran Tongyi-DeepResearch-30B-A3B,
WebExplorer-8B, Qwen3-30B-A3B-Thinking and gpt-oss through vLLM; every one of them emits the
`<tool_call>{"name": ..., "arguments": ...}</tool_call>` text format the loop parses, so they are
one line each:

```yaml
model:
  name: Alibaba-NLP/Tongyi-DeepResearch-30B-A3B   # or hkust-nlp/WebExplorer-8B, Qwen/Qwen3-30B-A3B-Thinking-2507, openai/gpt-oss-120b
  backend: api                # a served model; the loop talks to it over the OpenAI protocol
  api_base: http://127.0.0.1:8000/v1
```

`scripts/slurm/serve_and_run.sbatch` starts the vLLM server on the node and runs the file
(`TP=2` for gpt-oss-120b; `VLLM_PYTHON=` names the interpreter that has vLLM). A hosted API
model needs no GPU, but only if the compute nodes have network egress; on a cluster without it,
serve the backbone on the node.

### Retrievers

Any embedding model is `retrieval.dense_model`. ITER's released checkpoints
(`ielabgroup/ITER-Qwen3-Embedding-0.6B`, `-4B`), the plain `Qwen/Qwen3-Embedding-*` baselines and
LRAT (`Yuqi-Zhou/LRAT-Qwen3-Embedding-0.6B`) are decoder checkpoints without a sentence-transformers
config; the library detects that from `config.json` and serves them with last-token pooling and
normalisation (`retrieval.dense_pooling` overrides). The query side is the pair
`retrieval.dense_query_style` + `retrieval.dense_query_instruction`: `i2` with ITER's instruction
for the trained models, `plain` for the baselines. Pull a released checkpoint once on a node with
internet (`hf download ielabgroup/ITER-Qwen3-Embedding-0.6B`); the runs themselves
stay offline.

### Corpus and datasets

ITER's files drop into `data/`:

```
data/corpora/wiki25_512/corpus.jsonl        # {"docid": 0, "text": "Title\n..."}   11.2M chunks, 18 GB
data/infoseek_eval/topics.tsv               # id <TAB> question <TAB> answer       300 questions, answer-only
data/infoseek_train/topics.tsv              # id <TAB> question <TAB> answer       9,894 questions
data/browsecomp_plus_chunks/{corpus.jsonl,topics.tsv,qrels.txt}   # chunks with TREC evidence qrels
```

Registered as `infoseek_eval`, `infoseek_train` (both over `wiki25_512`) and
`browsecomp_plus_chunks`. A corpus above 1 GiB is served from disk: the loader builds a
byte-offset index next to the file once and looks documents up by id afterwards, so the harness
never holds the corpus in memory. Retrieval over such a corpus goes through prebuilt indexes:

```yaml
retrieval:
  dense_model: ielabgroup/ITER-Qwen3-Embedding-0.6B
  dense_index: indexes/external/iter06b_wiki25_512     # index.faiss + index.lookup.pkl (ITER's build_ann_index.py layout)
  ann_ef_search: 256
```

InfoSeek has answers but no document labels. Such an answer-only set runs like any other: rank
metrics are left out of the rows, the answer metrics and the LLM judge (`evaluation.judge_model`)
score it.

### Sample a paper setting first

`skimsearchagent-sample-dataset` cuts a small dataset out of a big one in the same layout: a few
questions with their answers and qrels, every gold document of those questions, an optional BM25
pool per question (`--pool-bm25-index`, for answer-only sets) and random chunks. The sample is a
dataset like any other (a folder under `data/` with `topics.tsv` and `corpus.jsonl` is picked up
without code), so a retriever or a backbone can be tried on ITER's setting in minutes:

```bash
skimsearchagent-sample-dataset --dataset browsecomp_plus_chunks --out browsecomp_plus_chunks_sample --n-topics 20 --n-docs 20000
skimsearchagent-sample-dataset --dataset infoseek_eval --out infoseek_eval_sample --n-topics 20 --n-docs 20000 \
    --pool-bm25-index indexes/external/wiki25_512_lucene --pool-k 100
sbatch --account=ACCT --qos=express --export=ALL,VLLM_PYTHON=/path/to/vllm-env/bin/python scripts/slurm/iter_sample.sbatch
```

The job indexes both samples with the released ITER retriever, serves Tongyi, and runs
`configs/iter/sample_*_iter06b_tongyi.yaml`. Judge the answer-only set afterwards where the API
is reachable: `skimsearchagent-judge --results-dir runs/iter_sample/... --judge-model gpt-4o-mini`.

### What was verified

These runs were made on a SLURM cluster on 2026-09-09 with the launchers above (one GPU per
job, express queue):

| run | setting | result |
|---|---|---|
| smoke, stage 1 | 8 InfoSeek training questions, `dedup_dense`, Tongyi via vLLM, DIVER's i2 checkpoint and its HNSW index over all 11.2M wiki chunks | 8/8 scored, 0 errors, 23 steps per question, 49 min |
| smoke, stage 2 | 9 triples (answer labeller), Qwen3-Embedding-0.6B, patched FlagEmbedding, 1 epoch | 4 steps, checkpoint with serving note |
| smoke, stage 3 | trained vs base retriever on the triples, 3,010-document subset | recall@1 0.667 both; novelty@5 0.822 vs 0.844 (four steps do not move a retriever; the loop works) |
| InfoSeek-Eval sample | 20 questions, released `ielabgroup/ITER-Qwen3-Embedding-0.6B`, Tongyi, 40 steps | judged accuracy 70% (14/20), 24 steps per question |
| BrowseComp-Plus sample | 20 questions with qrels, the same retriever and backbone, 40 steps | judged accuracy 25% (5/20), hit@1 0.20, gold-document coverage 0.41; 90% of episodes used all 40 steps (the paper allows 100) |

The sample numbers are smoke checks over 20 questions each, not paper results.

### Evaluate a retriever without an agent

```bash
skimsearchagent-eval-retriever --triples train_data/infoseek_i2.jsonl --dataset infoseek_train \
    --dense-model models/my-retriever --k 1,5,10 --subset 5000
```

Recall@k of the positive and novelty@k (the share of the top k the agent had not read) for every
triple; `--subset N` encodes only the triples' documents plus the first N chunks, the quick check
after a smoke run.

### The whole loop, on SLURM

The three launchers under `scripts/slurm/` are the pipeline this repository was tested with
(never on a login node):

```bash
sbatch --account=ACCT --export=ALL scripts/slurm/iter_smoke_traj.sbatch            # trajectories: configs/iter/smoke_infoseek_train_gpt4omini.yaml
sbatch --account=ACCT --export=ALL,RUNS=runs/iter_smoke/agent/infoseek_train/... scripts/slurm/iter_smoke_train.sbatch   # triples + fine-tune
sbatch --account=ACCT --export=ALL scripts/slurm/iter_smoke_eval.sbatch            # trained vs base retriever on the triples
```

The paper-scale files are `configs/iter/infoseek_train_trajectories_i2.yaml` (trajectory
generation with Tongyi), `configs/iter/infoseek_eval_dedup_dense_iter06b_tongyi.yaml` and the
BrowseComp-Plus counterpart. Re-indexing the wiki corpus with a new checkpoint is a sharded GPU
job outside this library (ITER's `build_faiss_index.sh` + `build_ann_index.py`); point
`retrieval.dense_index` at its output.

## Components

| component | location |
|---|---|
| query rendering (train and serve) | `agent_search/training/queries.py` |
| triples from run records, labellers | `agent_search/training/triples.py`, `build_triples.py` |
| trainer command, environment check, patch, serving note | `agent_search/training/retriever.py` |
| episode context at inference | `agent_search/training/history.py` |
| ITER's FlagEmbedding changes | `agent_search/training/patches/flagembedding-1.3.5-iter.patch` |
| ITER's search strategy (dedup search, get_document) | `agent_search/agent/tools/doc_dedup.py`, `prompts/tasks/research_dedup.md` |
| on-disk corpus, prebuilt index loading | `agent_search/corpus/docstore.py`, `retrievers/dense/vector_index.py` (`ExternalFaissIndex`) |
| ITER datasets (topics, qrels, answer-only) | `agent_search/evaluation/datasets.py` (`_load_topics_qrels`) |
| retriever-only evaluation | `agent_search/training/retriever_eval.py` |
| ITER experiment files | `configs/iter/` |
| SLURM jobs | `scripts/slurm/train_retriever.sbatch`, `scripts/slurm/iter_smoke_*.sbatch` |
| tests | `tests/test_training_pipeline.py`, `tests/test_iter_integration.py` |

This recipe trains the retriever only. The run record contains what an SFT or RL policy trainer
needs (`docs/RUN_RECORD.md`, "Using the record for training"); a policy trainer is not included.
