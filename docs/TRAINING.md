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
instruction, pooling, normalisation, lengths and precision the model was trained with. A model
trained with `bf16: true` is loaded and served in bfloat16, and the embedding caches it builds
are kept apart from float32 ones; `retrieval.dense_dtype` overrides the note.
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

## 5. ITER

ITER's own setting (its search tools, backbones, released retrievers, datasets, the sample
runs and what was verified) has its own page: [ITER.md](ITER.md).

## Components

| component | location |
|---|---|
| query rendering (train and serve) | `agent_search/training/queries.py` |
| triples from run records, labellers | `agent_search/training/triples.py`, `build_triples.py` |
| trainer command, environment check, patch, serving note | `agent_search/training/retriever.py` |
| episode context at inference | `agent_search/training/history.py` |
| ITER's FlagEmbedding changes | `agent_search/training/patches/flagembedding-1.3.5-iter.patch` |
| ITER's search strategy (dedup search, get_document) | `agent_search/strategies/dedup.py`, `agent_search/tools/search_dedup/`, `agent_search/tasks/research_dedup/prompt.md` |
| on-disk corpus, prebuilt index loading | `agent_search/corpus/docstore.py`, `retrievers/dense/vector_index.py` (`ExternalFaissIndex`) |
| ITER datasets (topics, qrels, answer-only) | `agent_search/evaluation/datasets/topics.py` (`_load_topics_qrels`) |
| retriever-only evaluation | `agent_search/training/retriever_eval.py` |
| ITER experiment files | `configs/iter/` |
| SLURM jobs | `scripts/slurm/train_retriever.sbatch`, `scripts/slurm/iter_smoke_*.sbatch` |
| tests | `tests/test_training_pipeline.py`, `tests/test_iter_integration.py` |

This recipe trains the retriever only. The run record contains what an SFT or RL policy trainer
needs (`docs/ARCHITECTURE.md`, "The run record"); a policy trainer is not included.
