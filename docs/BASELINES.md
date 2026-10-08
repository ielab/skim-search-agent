# Baselines on BrowseComp-Plus

Retrievers and rerankers under one agent setting. Each row changes only the retriever behind
`search` (or the reranker that reorders its pool). Everything else is fixed. Every row has its own
experiment file under `configs/baselines/`, and that file is what produced its numbers.

## The setting

- **Backbone:** `Alibaba-NLP/Tongyi-DeepResearch-30B-A3B`, temperature 0.6, seed 42.
- **Corpus:** `browsecomp_plus`, the original BrowseComp-Plus text (100,195 documents, front
  matter included, no inserted headings), with all 830 questions.
- **Tools:** `search` returns the top 10 documents with a 64-token snippet each, no
  de-duplication. `get_document` reads a document, cut at 12,000 tokens.
- **Budget:** 100 turns, a 131,072-token context window (the run stops at 90% of 125,000).
- **Encoding:** every learned retriever encodes a document's first 512 tokens.
- **Rerankers:** each reorders the top 100 of Qwen3-Embedding-0.6B. The agent sees the top 10.

Tongyi often sends `query` as a list. Every search tool here runs the first query of a list and
drops the rest. The official BrowseComp-Plus harness instead returns an
error for a list of two or more queries. About 14% to 21% of search calls in these cells carry
more than one query.

## Results

Accuracy is over all 830 questions. An answer a judge returns no verdict on counts as wrong (the
Qwen3-32B judge leaves 13 to 26 per cell). Three judges score each cell:

- **Qwen3-32B (official):** the BrowseComp-Plus leaderboard's judge and prompt. Rows are ranked by it.
- **Qwen3-30B-A3B-Thinking-2507:** the same answers under DIVER's judge template.
- **gpt-4o-mini:** the BrowseComp Appendix F prompt.

`searches` is the mean number of search calls per question.

Every judge reads the answer text only: reasoning tails and tool calls are removed before judging.
An answer without a verdict counts as wrong.

### First-stage retrievers

| retriever | file | Qwen3-32B (official) | Qwen3-30B-A3B-Thinking-2507 | gpt-4o-mini | searches |
|---|---|---:|---:|---:|---:|
| ITER-Qwen3-Embedding-4B (generic instruction) | `iter4b_generic` | 59.8 | 60.6 | 56.7 | 48.9 |
| ITER-Qwen3-Embedding-4B | `iter4b` | 58.2 | 59.4 | 54.5 | 50.8 |
| AgentIR-4B | `agentir4b` | 57.0 | 57.7 | 53.9 | 50.3 |
| ITER-Qwen3-Embedding-0.6B | `iter06b` | 56.4 | 57.5 | 53.6 | 53.5 |
| ColBERTv2 | `colbert` | 54.5 | 55.7 | 52.5 | 52.9 |
| LRAT-Qwen3-Embedding-0.6B | `lrat06b` | 50.2 | 50.8 | 47.2 | 55.7 |
| Qwen3-Embedding-4B | `qwen3emb4b` | 50.0 | 51.7 | 47.2 | 58.0 |
| SPLADE++ (CoCondenser-EnsembleDistil) | `splade` | 49.9 | 50.6 | 47.0 | 55.9 |
| Qwen3-Embedding-8B | `qwen3emb8b` | 49.0 | 50.4 | 47.2 | 57.5 |
| DiffRetriever, dense | `diffretriever` | 48.6 | 49.8 | 46.4 | 57.5 |
| DiffRetriever, sparse | `diffretriever_sparse` | 46.7 | 47.8 | 44.0 | 58.9 |
| RepLLaMA | `repllama` | 44.8 | 45.1 | 41.9 | 61.4 |
| Qwen3-Embedding-0.6B | `qwen3emb06b` | 41.7 | 43.0 | 38.8 | 62.2 |
| BM25 | `bm25` | 37.8 | 38.9 | 35.3 | 61.0 |

`iter4b_generic` is the same 4B checkpoint in its own query format, with the generic
Qwen3-Embedding instruction ("Given a web search query, retrieve relevant passages that answer the
query") in place of the ITER instruction it was trained with. On the same first queries it retrieves
more evidence: gold hit about 0.50 against 0.39. The 0.6B checkpoint shows no such difference.

One row changes the encoding length, so it sits outside the table: ITER-Qwen3-Embedding-4B with
documents encoded at 4,096 tokens instead of 512 (`iter4b_d4096`) scores 58.9 on Qwen3-32B
(official), 60.5 on Qwen3-30B-A3B-Thinking-2507 and 55.9 on gpt-4o-mini, with 49.4 searches per
question. It needs its own index, built with `DENSE_SEQ_LENGTH=4096`.

### Rerankers over the Qwen3-Embedding-0.6B pool

| reranker | file | Qwen3-32B (official) | Qwen3-30B-A3B-Thinking-2507 | gpt-4o-mini | searches |
|---|---|---:|---:|---:|---:|
| monoT5-3B | `rerank_monot5` | 56.1 | 57.0 | 53.5 | 53.3 |
| bge-reranker-v2-m3 | `rerank_bge_m3` | 52.5 | 53.7 | 49.8 | 54.4 |
| Qwen3-Reranker-0.6B | `rerank_qwen3_06b` | 49.9 | 51.1 | 48.1 | 57.0 |
| none (the pool's own order) | `qwen3emb06b` | 41.7 | 43.0 | 38.8 | 62.2 |
| Laya typed-decisions | `rerank_laya_typed` | 35.5 | 36.7 | 33.7 | 72.3 |
| Laya (English) | `rerank_laya` | 34.2 | 35.4 | 32.0 | 71.6 |
| Laya multilingual | `rerank_laya_multilingual` | 27.1 | 28.1 | 24.9 | 77.3 |

### Rerankers over the ITER pools

The same three rerankers, and Qwen3-Reranker-4B, over the pool of 100 from the ITER checkpoints:

| retriever | reranker | file | Qwen3-32B (official) | Qwen3-30B-A3B-Thinking-2507 | gpt-4o-mini | searches |
|---|---|---|---:|---:|---:|---:|
| ITER-Qwen3-Embedding-4B | monoT5-3B | `rerank_monot5_iter4b` | 62.0 | 63.0 | 58.6 | 47.5 |
| ITER-Qwen3-Embedding-0.6B | monoT5-3B | `rerank_monot5_iter06b` | 61.7 | 63.4 | 59.2 | 49.6 |
| ITER-Qwen3-Embedding-4B | bge-reranker-v2-m3 | `rerank_bge_m3_iter4b` | 61.2 | 61.9 | 59.0 | 46.1 |
| ITER-Qwen3-Embedding-0.6B | bge-reranker-v2-m3 | `rerank_bge_m3_iter06b` | 60.1 | 60.6 | 57.5 | 49.5 |
| ITER-Qwen3-Embedding-0.6B | Qwen3-Reranker-4B | `rerank_qwen3_4b_iter06b` | 59.2 | 60.7 | 56.0 | 49.9 |
| ITER-Qwen3-Embedding-4B | Qwen3-Reranker-0.6B | `rerank_qwen3_06b_iter4b` | 58.7 | 58.9 | 55.4 | 49.2 |
| ITER-Qwen3-Embedding-0.6B | Qwen3-Reranker-0.6B | `rerank_qwen3_06b_iter06b` | 57.5 | 58.1 | 55.3 | 51.0 |

Without a reranker the two retrievers score 56.4 (`iter06b`) and 58.2 (`iter4b`). These
rows ran as 12 shard jobs each. A row needs its retriever's index (`iter06b` or `iter4b` in step
4). `rerank_monot5_iter4b` runs the reranker 4 pairs at a time: the 4B encoder, monoT5-3B and the
model server share one GPU. `rerank_qwen3_4b_iter06b` runs it 8 pairs at a time and gives the model
server 0.77 of the GPU (`GPU_MEM_UTIL=0.77`).

The file column is short for `configs/baselines/browsecomp_plus_<file>_tongyi.yaml`.

The same rows are on the project site's leaderboard, https://ielab.io/skim-search-agent/leaderboard/,
with the Qwen3-32B accuracy, the evidence recall and the search calls that the BrowseComp-Plus
leaderboard reports, and the tokens per question with each token counted once.
`scripts/leaderboard_row.py` computes them from a judged cell.

### Agent backbones

The questions, the corpus, the retriever and the three judges stay as above. The agent model
changes. Each model runs with its own sampling settings and tool-call format, all written in its
experiment file.

General models (Tongyi-DeepResearch, gpt-oss, Qwen3.5, Qwen-AgentWorld) use the library's search
and fetch tools and the 100-turn budget. Four agents were fine-tuned with their own tools and
prompt, and they run in that interface: the tool names, the result layout, the system prompt and
the history format are the ones in each agent's released code. The turn budget is not: every
agent gets 100 turns, so a row differs from another in the agent only. The library's retriever
answers the searches, and the corpus answers the page reads.

| agent | tools | its own code's limit (not used) | sampling |
|---|---|---|---|
| OpenResearcher-30B-A3B | `browser.search`, `browser.open`, `browser.find` | 200 turns | temperature 1.0 |
| QUEST-35B-RL | `search` (several queries, five 512-token passages each) | 400 calls | temperature 1.0, presence penalty 1.1 |
| MiroThinker-1.7-mini | `google_search`, `scrape_and_extract_info` (the served model reads the page) | 300 turns, the newest five results kept | temperature 1.0, top-p 0.95, repetition penalty 1.05 |
| OpenSeeker-v2-30B-SFT | `search` (several queries), `visit` with a goal (the served model reads the page) | 200 calls | temperature 0.6, top-p 0.95, top-k 20 |

With Qwen3-Embedding-0.6B behind the search tool:

| backbone | file | Qwen3-32B (official) | Qwen3-30B-A3B-Thinking-2507 | gpt-4o-mini | searches |
|---|---|---:|---:|---:|---:|
| MiroThinker-1.7-mini | `qwen3emb06b_mirothinker` | 52.3 | 55.1 | 50.7 | 52.8 |
| QUEST-35B-RL | `qwen3emb06b_quest` | 42.2 | 41.7 | 39.6 | 42.3 |
| Tongyi-DeepResearch-30B-A3B | `qwen3emb06b_tongyi` | 41.7 | 43.0 | 38.8 | 62.2 |
| OpenResearcher-30B-A3B | `qwen3emb06b_openresearcher` | 39.6 | 42.0 | 38.8 | 43.5 |
| gpt-oss-120b | `qwen3emb06b_gptoss_120b` | 38.1 | 38.7 | 36.3 | 21.8 |
| gpt-oss-20b | `qwen3emb06b_gptoss_20b` | 36.5 | 34.6 | 33.5 | 44.5 |
| Qwen-AgentWorld-35B-A3B | `qwen3emb06b_agentworld` | 28.4 | 29.3 | 27.6 | 21.7 |
| Qwen3.5-9B | `qwen3emb06b_qwen35_9b` | 25.4 | 24.8 | 22.0 | 25.3 |
| Qwen3.5-27B | `qwen3emb06b_qwen35_27b` | 22.4 | 22.3 | 20.1 | 18.3 |
| Qwen3.5-4B | `qwen3emb06b_qwen35_4b` | 16.0 | 16.4 | 14.6 | 19.9 |

With ITER-Qwen3-Embedding-4B behind the search tool:

| backbone | file | Qwen3-32B (official) | Qwen3-30B-A3B-Thinking-2507 | gpt-4o-mini | searches |
|---|---|---:|---:|---:|---:|
| Tongyi-DeepResearch-30B-A3B | `iter4b_tongyi` | 58.2 | 59.4 | 54.5 | 50.8 |
| Qwen3.5-9B | `iter4b_qwen35_9b` | 44.3 | 44.5 | 40.6 | 19.4 |
| Qwen3.5-27B | `iter4b_qwen35_27b` | 38.3 | 37.8 | 34.5 | 13.0 |
| Qwen3.5-4B | `iter4b_qwen35_4b` | 31.2 | 31.9 | 28.8 | 15.1 |

The file column is short for `configs/baselines/browsecomp_plus_<file>.yaml`. `searches` counts
the agent's search tool under its own name (`browser.search`, `google_search`, `search`).

Read the rows with these points in mind:

- The first runs of the three fine-tuned agents used the limits of their own code (300, 400 and
  200 turns) and were removed: a larger budget is a second difference from the other rows. All
  three rows are now 100-turn runs.
- A 100-turn row keeps every episode of the first run that ended by itself within 98 model calls
  and reruns the other questions with the 100-turn limit.
- The token column of the leaderboard counts the largest prompt plus the generated tokens.
  MiroThinker-1.7-mini keeps only its newest five tool results in the prompt, and the tokens its
  page reader spends are not in that count. Its real cost is higher than the column shows.
- MiroThinker-1.7-mini's prompt states today's date, so a run resumed on a later day is refused
  as a different experiment. Set `MIROTHINKER_DATE` (for example `2026-10-06`) in the file's `env`
  block to the day the run started.
- The OpenSeeker-v2-30B-SFT file ships without a row. Each `visit` asks the served model to read
  the page, and a full run was too slow to finish.
- The fine-tuned agents' clients send no sampling seed. One seed on every request replays the
  same random draws each turn, and a repeated turn then repeats for the rest of the episode. Their
  files set `LLM_SEED_PER_CALL: '1'`: the seed is 42 plus the number of model calls the episode has
  made, so a run is still reproducible.
- The gpt-oss rows run through the Responses API (`model.driver: responses`) at reasoning effort
  high. A turn with no tool call and no `Exact Answer:` line is not an answer: the model is
  reminded and takes the turn again, three times at most.

Serve each backbone with the flags its file carries in `env.VLLM_ARGS`:

| backbone | model | extra `vllm serve` flags |
|---|---|---|
| Tongyi-DeepResearch-30B-A3B | `Alibaba-NLP/Tongyi-DeepResearch-30B-A3B` | none |
| OpenSeeker-v2-30B-SFT | `PolarSeeker/OpenSeeker-v2-30B-SFT` | none |
| OpenResearcher-30B-A3B | `OpenResearcher/OpenResearcher-30B-A3B` | none |
| MiroThinker-1.7-mini | `miromind-ai/MiroThinker-1.7-mini` | none |
| Qwen-AgentWorld-35B-A3B | `Qwen/Qwen-AgentWorld-35B-A3B` | `--language-model-only --gdn-prefill-backend triton` |
| QUEST-35B-RL | `osunlp/QUEST-35B-RL` | `--language-model-only --gdn-prefill-backend triton --enable-prefix-caching --mamba-cache-mode align` |
| Qwen3.5-4B, Qwen3.5-9B, Qwen3.5-27B, Qwen3.6-27B, Qwen3.8-27B | `Qwen/Qwen3.5-4B` and so on | `--enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3 --gdn-prefill-backend triton` |
| gpt-oss-20b, gpt-oss-120b | `openai/gpt-oss-20b`, `openai/gpt-oss-120b` | `--enable-auto-tool-choice --tool-call-parser openai --reasoning-parser openai_gptoss` |

Notes for single backbones:

- **MiroThinker-1.7-mini and OpenSeeker-v2-30B-SFT** read pages with a model. Point the reader at
  the served backbone before the run:
  `export VISIT_READER_API_BASE=http://127.0.0.1:8000/v1 VISIT_READER_MODEL=miromind-ai/MiroThinker-1.7-mini`.
- **QUEST-35B-RL** names a tokenizer class in `tokenizer_config.json` that vLLM 0.18 does not
  load. Copy the snapshot to a folder, set `tokenizer_class` to `Qwen2TokenizerFast` there, and
  serve the folder with `--served-model-name osunlp/QUEST-35B-RL`. Give the server 0.92 of the GPU.
- **Qwen3.6-27B and Qwen3.8-27B** have experiment files and no row. Their files carry
  `--enable-prefix-caching --mamba-cache-mode align`: without them the server recomputes the whole prompt every
  turn. With the flags a turn still takes about 40 seconds, because the models write 1,000 to
  2,400 thinking tokens per turn.

## Run one

The steps run from a fresh clone. Steps 1 to 3 are shared. Step 4 depends on the row.

**1. Install.** Needs JDK 21 on `JAVA_HOME` and GPUs.

```bash
python -m pip install -e ".[retrieval,api,eval,serve]"
python -m agent_search.tokens --seed
```

**2. Stage the corpus.** On a node with internet. This writes `data/browsecomp_plus/`, with the
gold qrels the runs read (`qrels/test.tsv`) and the evidence qrels recall is scored against
(`qrels/evidence.tsv`).

```bash
python scripts/stage_browsecomp_plus.py
```

**3. Download the checkpoints** the row needs (the backbone, the retriever, the reranker), also on
a node with internet. Runs stay offline after this.

```bash
hf download Alibaba-NLP/Tongyi-DeepResearch-30B-A3B
hf download Qwen/Qwen3-Embedding-0.6B      # the retriever or reranker pool of your row
```

**4. Build the row's index.** Each build is a GPU job, except BM25. The index directory name
carries the model and the encoding settings, so a run finds the index from its file.

| rows | build |
|---|---|
| `bm25` | `skimsearchagent-build-indexes --dataset browsecomp_plus --retriever bm25_pyserini --index-root indexes` |
| `qwen3emb06b`, `qwen3emb4b`, `qwen3emb8b`, every `rerank_*` | `sbatch --export=ALL,EMBED_MODEL=Qwen/Qwen3-Embedding-0.6B,DATASET=browsecomp_plus,DENSE_SEQ_LENGTH=512,DENSE_DTYPE=bfloat16 scripts/embed_full.sbatch` (`-4B` or `-8B` for those rows) |
| `lrat06b` | the same with `EMBED_MODEL=Yuqi-Zhou/LRAT-Qwen3-Embedding-0.6B,DENSE_POOLING=last_token` |
| `iter06b`, `iter4b`, `rerank_*_iter06b`, `rerank_*_iter4b` | the same with `EMBED_MODEL=ielabgroup/ITER-Qwen3-Embedding-0.6B` (or `-4B`) `,DENSE_POOLING=last_token` |
| `iter4b_d4096` | the `iter4b` build with `DENSE_SEQ_LENGTH=4096` |
| `agentir4b` | the same with `EMBED_MODEL=Tevatron/AgentIR-4B,DENSE_POOLING=last_token,DENSE_DTYPE=float16` |
| `repllama` | merge the adapter first (below), then `EMBED_MODEL=models/repllama-v1-7b-passage-merged,DENSE_POOLING=last_token,DENSE_DTYPE=float16` |
| `splade` | `sbatch --export=ALL,RETRIEVER=splade,DATASET=browsecomp_plus,SPLADE_DOC_LENGTH=512 scripts/embed_full.sbatch` |
| `colbert` | `sbatch --export=ALL,RETRIEVER=colbert,DATASET=browsecomp_plus,COLBERT_DOC_LENGTH=512 scripts/embed_full.sbatch` |
| `diffretriever` | `sbatch --export=ALL,DATASET=browsecomp_plus,DIFFRETRIEVER_DOC_LENGTH=512,DIFFRETRIEVER_PYTHON=/path/to/env/bin/python scripts/slurm/diffretriever_index.sbatch` |
| `diffretriever_sparse` | the `diffretriever` build, then the same command again with `DIFFRETRIEVER_MODE=sparse` (it adds `sparse.npz` and keeps the vectors) |

Before every `sbatch`, pass `--account=YOUR_ACCOUNT` and any partition your cluster needs.

Notes for single rows:

- **RepLLaMA** ships as a LoRA adapter. Merge it once, in an environment that has `peft`:
  `python scripts/merge_lora.py --adapter castorini/repllama-v1-7b-lora-passage --out models/repllama-v1-7b-passage-merged --dtype float16`.
- **DiffRetriever** runs in its own environment (transformers 4.54 and peft), behind
  `scripts/serve_diffretriever.py`. Serve it on a second GPU during the run and export
  `DIFFRETRIEVER_URL`. `scripts/slurm/diffretriever_index.sbatch` shows the commands.
- **Laya** loads its code from the main repo and its weights from the checkpoint. Download both:
  `hf download convaiinnovations/laya` and, for the other rows, `convaiinnovations/laya-typed-decisions`
  or `convaiinnovations/laya-multilingual`.

Check a file before you run it. `validate` prints every setting and fails if the index is missing:

```bash
skimsearchagent validate configs/baselines/browsecomp_plus_colbert_tongyi.yaml
```

**5. Serve the backbone.**

```bash
export VLLM_USE_FLASHINFER_MOE_FP16=0
vllm serve Alibaba-NLP/Tongyi-DeepResearch-30B-A3B --tensor-parallel-size 1 --port 8000 \
  --max-model-len 131072 --gpu-memory-utilization 0.80 --trust-remote-code
```

**6. Run the row.**

```bash
F=configs/baselines/browsecomp_plus_colbert_tongyi.yaml
skimsearchagent run $F model.backend=api model.api_base=http://127.0.0.1:8000/v1
```

This writes the cell to the file's `output.runs_dir` (`runs/baseline/colbert_tongyi` here).

Every published row ran as 10 shard jobs on one GPU each: the backbone, the retriever and the
reranker share that GPU. Shard `k` takes every tenth question starting at `k`, in the loader's
order. Write the id files:

```bash
R=runs/baseline/colbert_tongyi                       # the file's output.runs_dir
S=$R/__shards/browsecomp_plus/agent_research_iter_colbert   # the condition is the file's strategy
mkdir -p $S && python - $S <<'EOF'
import sys
import agent_search.evaluation.datasets as D
ids = [i.instance_id for i in D._DATASETS["browsecomp_plus"](limit=None)]
for k in range(10):
    open(f"{sys.argv[1]}/shard_{k}of10.txt", "w").write("\n".join(ids[k::10]) + "\n")
EOF
```

Each shard is the step 6 command with two more overrides, against its own backbone server:

```bash
skimsearchagent run $F model.backend=api model.api_base=http://127.0.0.1:8000/v1 \
  dataset.only_instances=$S/shard_${k}of10.txt output.runs_dir=$S/shard_${k}of10
```

When every shard has finished, fold them into the cell:

```bash
python scripts/merge_shards.py --runs-dir $R --dataset browsecomp_plus \
  --condition agent_research_iter_colbert --num-shards 10 --model Alibaba-NLP/Tongyi-DeepResearch-30B-A3B
```

**7. Judge it three times.** Run the judges one after another, never two at once on one cell.
The two Qwen3 judges rewrite `rows.jsonl` and each adds its own column. gpt-4o-mini writes its
verdicts to `judge_cache.jsonl` next to it.

```bash
D=$R/agent/browsecomp_plus/Tongyi-DeepResearch-30B-A3B/agent_research_iter_colbert

# Qwen3-32B with the official BrowseComp-Plus prompt
vllm serve Qwen/Qwen3-32B --port 8104 --max-model-len 32768 --max-num-seqs 16 &
OPENAI_API_KEY=EMPTY python -m agent_search.evaluation.llm_judge --results-dir $D \
  --judge-model Qwen/Qwen3-32B --judge-api-base http://127.0.0.1:8104/v1 \
  --judge-prompt bcp --tag officialbcp --workers 16 --max-tokens 16384 --unfinished-ok

# Qwen3-30B-A3B-Thinking-2507 with DIVER's template
vllm serve Qwen/Qwen3-30B-A3B-Thinking-2507 --port 8103 --max-model-len 32768 --max-num-seqs 32 &
OPENAI_API_KEY=EMPTY python -m agent_search.evaluation.llm_judge --results-dir $D \
  --judge-model Qwen/Qwen3-30B-A3B-Thinking-2507 --judge-api-base http://127.0.0.1:8103/v1 \
  --judge-prompt diver --tag diver --workers 16 --max-tokens 8192 --unfinished-ok

# gpt-4o-mini
OPENAI_API_KEY=sk-... python scripts/judge_cells.py --extra-cell $D --workers 8
```

Stop each `vllm serve` before you start the next one. Read the three accuracies over all 830
questions, with a missing verdict counted as wrong:

```bash
python - $D <<'EOF'
import json, sys
d = sys.argv[1]
rows = [json.loads(l) for l in open(d + "/rows.jsonl")]
gpt = {}
for l in open(d + "/judge_cache.jsonl"):
    r = json.loads(l)
    gpt[r["instance_id"]] = r.get("judge_correct") is True
for name, ok in [("Qwen3-32B (official)", lambda r: r.get("judge_correct_officialbcp") is True),
                 ("Qwen3-30B-A3B-Thinking-2507", lambda r: r.get("judge_correct_diver") is True),
                 ("gpt-4o-mini", lambda r: gpt.get(r["instance_id"], False))]:
    print(f"{name}: {100 * sum(ok(r) for r in rows) / len(rows):.1f}")
EOF
```
