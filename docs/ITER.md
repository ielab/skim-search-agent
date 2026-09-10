# ITER in SkimSearchAgent

Paper: *ITER: Interaction-Aware Retrieval for Agentic Search* (Chen, Wang, Yin, Zhuang, Zuccon,
Leelanupab, 2026), https://arxiv.org/abs/2608.27912; code and released checkpoints at
https://github.com/ielab/ITER.


ITER (Zhou et al., https://github.com/ielab/ITER) trains a dense retriever from the trajectories
of a search agent and evaluates it inside the agent loop: the retriever is conditioned on what
the agent already searched, and it is trained to return documents the agent has not read yet.
This page maps the paper's setup onto this library: the strategy, the backbones, the retrievers,
the datasets and corpus, the training recipe, the evaluation, and the runs made to verify it.
The generic training recipe (triples from run records, the trainer, plugging a checkpoint back
in) is in [TRAINING.md](TRAINING.md); this page is the ITER-specific part.

## Strategy

ITER's agent has two tools: `search(query)` and `get_document(docid)`. A search over-fetches a
pool of 100, drops every document an earlier search already surfaced in this episode, and shows
the top 10 of the rest with a 64-token snippet. Documents that would have ranked but were shown
before are listed under "Already-seen" so the agent can reopen them. That is `strategy=dedup_dense`
(the run's dense model behind `search`) or `dedup_bm25`, with the task template
`agent_search/tasks/research_dedup/prompt.md`. Listing knobs: `listing.dedup_topk` (10),
`listing.dedup_pool_k` (100); snippet: `budgets.snippet_tokens: 64`; `get_document` returns at
most 512 tokens in ITER (`budgets.max_visit_tokens: 512`; the wiki chunks are 512 tokens, so this
only matters on a corpus with longer documents).

## Backbones

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

## Retrievers

Any embedding model is `retrieval.dense_model`. ITER's released checkpoints
(`ielabgroup/ITER-Qwen3-Embedding-0.6B`, `-4B`), the plain `Qwen/Qwen3-Embedding-*` baselines and
LRAT (`Yuqi-Zhou/LRAT-Qwen3-Embedding-0.6B`) are decoder checkpoints without a sentence-transformers
config; the library detects that from `config.json` and serves them with last-token pooling and
normalisation (`retrieval.dense_pooling` overrides). ITER encoded its corpora in bfloat16, so the
shipped files set `retrieval.dense_dtype: bfloat16` for these checkpoints. The query side is the pair
`retrieval.dense_query_style` + `retrieval.dense_query_instruction`: `i2` with ITER's instruction
for the trained models, `plain` for the baselines. Pull a released checkpoint once on a node with
internet (`hf download ielabgroup/ITER-Qwen3-Embedding-0.6B`); the runs themselves
stay offline.

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

These runs were made on a SLURM cluster on 2026-09-09 with the launchers above (one GPU per
job, express queue):

| run | setting | result |
|---|---|---|
| smoke, stage 1 | 8 InfoSeek training questions, `dedup_dense`, Tongyi via vLLM, DIVER's i2 checkpoint and its HNSW index over all 11.2M wiki chunks | 8/8 scored, 0 errors, 23 steps per question, 49 min |
| smoke, stage 2 | 9 triples (answer labeller), Qwen3-Embedding-0.6B, patched FlagEmbedding, 1 epoch | 4 steps, checkpoint with serving note |
| smoke, stage 3 | trained vs base retriever on the triples, 3,010-document subset | recall@1 0.667 both; novelty@5 0.822 vs 0.844 (four steps do not move a retriever; the loop works) |
| training check | 48 triples from every trajectory above (oracle labels on BrowseComp-Plus, answer labels on InfoSeek), Qwen3-Embedding-0.6B, bf16, 3 epochs, lr 5e-6 | loss per epoch 1.34, 0.29, 0.09 (72 steps); the checkpoint's serving note records bfloat16, last-token pooling, the i2 style |
| training check, eval | the trained checkpoint loaded in bfloat16 vs its base, on the 19 BrowseComp-Plus triples (in-sample) over the 20,092-chunk sample corpus | recall@1 0.26 vs 0.11, recall@5 0.68 vs 0.37, recall@10 0.74 vs 0.47 |
| training check, agent | the trained checkpoint (bfloat16) as the retriever of `dedup_dense` on the BrowseComp-Plus sample, Tongyi, 40 steps | 20/20 scored, 0 errors, hit@5 0.45, judged 15% (3/20); a 48-triple model, not a contender, the loop closes |
| InfoSeek-Eval sample | 20 questions, released `ielabgroup/ITER-Qwen3-Embedding-0.6B`, Tongyi, 40 steps | judged accuracy 70% (14/20), 24 steps per question |
| BrowseComp-Plus sample | 20 questions with qrels, the same retriever and backbone, 40 steps | judged accuracy 25% (5/20), hit@1 0.20, gold-document coverage 0.41; 90% of episodes used all 40 steps (the paper allows 100) |
| held-out training | 99 InfoSeek training trajectories (97 triples, answer labels), Qwen3-Embedding-0.6B, bf16, 3 epochs | loss decreases each epoch; the checkpoint serves in bfloat16 |
| held-out eval, retriever | the trained checkpoint vs its base vs the released ITER-0.6B on 20 InfoSeek-Eval triples the training never saw | recall@1 0.45 / 0.30 / 0.45; recall@5 0.60 / 0.75 / 0.75; recall@10 0.70 / 0.85 / 0.85 |
| held-out eval, agent | `dedup_dense` on the 20-question InfoSeek-Eval sample, Tongyi, 40 steps, the trained checkpoint vs its base | judged accuracy 65% (13/20) vs 60% (12/20); 25.7 vs 27.6 steps per question |
| after the 0.3 restructure | the InfoSeek-Eval sample again with the released ITER-0.6B, its index rebuilt in bfloat16 | judged accuracy 50% (10/20), 25 steps per question; replaying the earlier run's 20 trajectories through the old and the new code gives identical observations on all 486 steps (`scripts/replay_check.py`), so the difference from the 70% above is the index precision and sampling |
| one-shot RAG | `rag_bm25` on the same sample, Tongyi: BM25 top 5 in one prompt, one call | judged accuracy 70% (14/20), 3 empty answers |

The sample numbers are smoke checks over 20 questions each, not paper results. The held-out
rows show the whole loop closing on data the model never trained on (trajectories in, a
checkpoint out, that checkpoint behind the agent); 97 triples and 20 questions are far too few to
rank the retrievers.

## Evaluate a retriever without an agent

```bash
skimsearchagent-eval-retriever --triples train_data/infoseek_i2.jsonl --dataset infoseek_train \
    --dense-model models/my-retriever --k 1,5,10 --subset 5000
```

Recall@k of the positive and novelty@k (the share of the top k the agent had not read) for every
triple; `--subset N` encodes only the triples' documents plus the first N chunks, the quick check
after a smoke run.

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
job outside this library (ITER's `build_faiss_index.sh` + `build_ann_index.py`); point
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
  bm25_index: indexes/external/wiki25_512_lucene     # for dedup_bm25 / search_visit with bm25_backend: pyserini
```

The complete files are under `configs/iter/`.
