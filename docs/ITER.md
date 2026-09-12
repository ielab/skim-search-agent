# ITER in SkimSearchAgent

Paper: *ITER: Interaction-Aware Retrieval for Agentic Search* (Chen, Wang, Yin, Zhuang, Zuccon,
Leelanupab, 2026), https://arxiv.org/abs/2608.27912; code and released checkpoints at
https://github.com/ielab/ITER.


ITER (Chen et al., https://github.com/ielab/ITER) trains a dense retriever from the trajectories
of a search agent and evaluates it inside the agent loop: the retriever is conditioned on what
the agent already searched, and it is trained to return documents the agent has not read yet.
This page maps the paper's setup onto this library: the strategy, the backbones, the retrievers,
the datasets and corpus, the training recipe, the evaluation, and the runs made to verify it.
The generic training recipe (triples from run records, the trainer, plugging a checkpoint back
in) is in [TRAINING.md](TRAINING.md). This page covers the ITER-specific part.

## Strategy

ITER's agent has two tools: `search(query)` and `get_document(docid)`. A search over-fetches a
pool of 100, drops every document an earlier search already surfaced in this episode, and shows
the top 10 of the rest, each as its passage text cut to 64 model tokens (ITER's runs passed
`--snippet-max-tokens 64`; ITER cuts with the served model's tokenizer, the library on its own
token ruler, so the counts are close, not identical). Documents that would have ranked but were shown
before are listed under "Already-seen" so the agent can reopen them. That is `strategy=dedup_dense`
(the run's dense model behind `search`) or `dedup_bm25`.

The prompt is the one DIVER evaluated Tongyi-DeepResearch with: Tongyi's deep-research system
prompt, DIVER's strict tool rules, and its dedup notice (`agent_search/tasks/research_dedup/prompt.md`,
condition `agent_research_dedup_dense`). DIVER's `--strong` prompt for general backbones, a
meticulous multi-constraint research agent, is the task `research_dedup_strong` (condition
`agent_research_dedup_dense_strong`). DIVER capped an episode at 50 LLM calls (`MAX_LLM_CALL_PER_RUN`),
so ITER cells run with `max_steps: 50`, not the 100 of the Sieve experiments. Its BrowseComp-Plus
runs use the full 100,195-document corpus (`browsecomp_plus_structured` here), not a chunked one.

Listing knobs: `listing.dedup_topk` (10), `listing.dedup_pool_k` (100) and
`listing.dedup_snippet_tokens` (64). `get_document` returns at most 512 model tokens in ITER
(`budgets.max_visit_tokens: 512`, cut on the same ruler). The wiki chunks are already 512
tokens, so both limits only matter on a corpus with longer documents.

## Backbones

The backbone is `model.name` plus `model.backend`. ITER ran Tongyi-DeepResearch-30B-A3B,
WebExplorer-8B, Qwen3-30B-A3B-Thinking and gpt-oss through vLLM. Every one of them emits the
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
model needs no GPU, but only if the compute nodes have network egress. On a cluster without it,
serve the backbone on the node.

## Retrievers

Any embedding model is `retrieval.dense_model`. ITER's released checkpoints
(`ielabgroup/ITER-Qwen3-Embedding-0.6B`, `-4B`), the plain `Qwen/Qwen3-Embedding-*` baselines, and
LRAT (`Yuqi-Zhou/LRAT-Qwen3-Embedding-0.6B`) are decoder checkpoints without a sentence-transformers
config. The library detects that from `config.json` and serves them with last-token pooling and
normalisation (`retrieval.dense_pooling` overrides). ITER encoded its corpora in bfloat16, so the
shipped files set `retrieval.dense_dtype: bfloat16` for these checkpoints. The query side is the pair
`retrieval.dense_query_style` + `retrieval.dense_query_instruction`: `i2` with ITER's instruction
for the trained models, `plain` for the baselines. Pull a released checkpoint once on a node with
internet (`hf download ielabgroup/ITER-Qwen3-Embedding-0.6B`). The runs themselves stay offline.

## Corpus and datasets

ITER's files drop into `data/`:

```
data/corpora/wiki25_512/corpus.jsonl        # {"docid": 0, "text": "Title\n..."}   11.2M chunks, 18 GB
data/infoseek_eval/topics.tsv               # id <TAB> question <TAB> answer       300 questions, answer-only
data/infoseek_train/topics.tsv              # id <TAB> question <TAB> answer       9,894 questions
data/browsecomp_plus_chunks/{corpus.jsonl,topics.tsv,qrels.txt}   # chunks with TREC evidence qrels
```

Registered as `infoseek_eval`, `infoseek_train` (both over `wiki25_512`) and
`browsecomp_plus_chunks`. A corpus above 1 GiB is served from disk: the loader builds a
byte-offset index next to the file once and looks documents up by id afterwards, so a run
never holds the corpus in memory. Retrieval over such a corpus goes through prebuilt indexes:

```yaml
retrieval:
  dense_model: ielabgroup/ITER-Qwen3-Embedding-0.6B
  dense_index: indexes/external/iter06b_wiki25_512     # index.faiss + index.lookup.pkl (ITER's build_ann_index.py layout)
  ann_ef_search: 256
```

InfoSeek has answers but no document labels. Such an answer-only set runs like any other: rank
metrics are left out of the rows. The answer metrics and the LLM judge (`evaluation.judge_model`)
score it instead.

## A corpus you already embedded

ITER's `encode.py` (Tevatron) writes one pickle per shard, `(embeddings, doc_ids)`.
`scripts/import_tevatron_index.py` turns those shards into the prebuilt index layout the
library serves (`index.faiss` + `index.lookup.pkl`) and checks every id against the corpus. Point
`retrieval.dense_index` at the result. The corpus is then served from disk and never re-embedded.
The query side still needs the encoder (`retrieval.dense_model`).

```bash
python scripts/import_tevatron_index.py --shards "/path/to/index-*.pkl" \
    --out indexes/external/iter06b_bcp_chunks --corpus data/browsecomp_plus_chunks/corpus.jsonl
```

## Sample a paper setting first

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

## What was verified

The smoke pipeline, the sample runs, the training check and the held-out check above all ran
end to end on a SLURM cluster with the launchers in `scripts/slurm/`, each stage producing its
record. Numbers from those runs are not reported here: they come from 20-question samples and
say nothing about the paper's results. Run the full sets with the files under `configs/iter/`.

## Evaluate a retriever without an agent

```bash
skimsearchagent-eval-retriever --triples train_data/infoseek_i2.jsonl --dataset infoseek_train \
    --dense-model models/my-retriever --k 1,5,10 --subset 5000
```

The command reports Recall@k of the positive and novelty@k (the share of the top k the agent had
not read) for every triple. `--subset N` encodes only the triples' documents plus the first N
chunks, a quick check to run after a smoke run.

## The whole loop, on SLURM

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
job outside this library (ITER's `build_faiss_index.sh` + `build_ann_index.py`). Point
`retrieval.dense_index` at its output.

## The setting in one file

ITER's tools with its released retriever:

```yaml
strategy: dedup_dense         # or dedup_bm25
retrieval:
  dense_model: ielabgroup/ITER-Qwen3-Embedding-0.6B
  dense_dtype: bfloat16       # ITER encoded its corpora in bfloat16
  dense_query_style: i2       # the query carries the earlier sub-queries, as the model was trained
  dense_query_instruction: "Given the main question, the current sub-query, and the sub-queries already tried in previous interactions, retrieve documents relevant to the current sub-query that provide NEW information not yet found."
```

A corpus that does not fit in memory, served from disk and searched through prebuilt indexes:

```yaml
dataset:
  name: infoseek_eval         # data/infoseek_eval/topics.tsv over data/corpora/wiki25_512/corpus.jsonl (11.2M chunks)
retrieval:
  dense_index: indexes/external/iter06b_wiki25_512   # index.faiss + index.lookup.pkl, ITER's layout
  bm25_index: indexes/external/wiki25_512_lucene     # for dedup_bm25 / search_visit
```

The complete files are under `configs/iter/`.
