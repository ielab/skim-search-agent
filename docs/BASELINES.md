# Baselines on BrowseComp-Plus

Retrievers and rerankers under one agent setting. Each row changes only the retriever behind
`search` (or the reranker that reorders its pool). Everything else is fixed. Every row has its own
experiment file under `configs/iter/`, and that file is what produced its numbers.

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

### First-stage retrievers

| retriever | file | Qwen3-32B (official) | Qwen3-30B-A3B-Thinking-2507 | gpt-4o-mini | searches |
|---|---|---:|---:|---:|---:|
| AgentIR-4B | `agentir4b` | 56.6 | 57.5 | 53.9 | 50.3 |
| ColBERTv2 | `colbert` | 54.0 | 55.1 | 52.5 | 52.9 |
| LRAT-Qwen3-Embedding-0.6B | `lrat06b` | 50.0 | 50.7 | 47.2 | 55.7 |
| Qwen3-Embedding-4B | `qwen3emb4b` | 49.9 | 51.4 | 47.3 | 58.0 |
| SPLADE++ (CoCondenser-EnsembleDistil) | `splade` | 49.5 | 49.9 | 46.9 | 55.9 |
| Qwen3-Embedding-8B | `qwen3emb8b` | 48.4 | 50.2 | 47.2 | 57.5 |
| DiffRetriever, dense | `diffretriever` | 48.3 | 49.9 | 46.4 | 57.5 |
| DiffRetriever, sparse | `diffretriever_sparse` | 46.3 | 47.2 | 44.0 | 58.9 |
| RepLLaMA | `repllama` | 44.3 | 44.5 | 41.9 | 61.4 |
| Qwen3-Embedding-0.6B | `qwen3emb06b` | 41.7 | 42.4 | 38.8 | 62.2 |
| BM25 | `bm25` | 37.7 | 38.1 | 35.3 | 61.0 |

ITER-Qwen3-Embedding-0.6B and -4B (`iter06b`, `iter4b`), and ITER-4B with documents encoded at
4,096 tokens (`iter4b_d4096`), are running. Their rows go here when all three judges finish.

### Rerankers over the Qwen3-Embedding-0.6B pool

| reranker | file | Qwen3-32B (official) | Qwen3-30B-A3B-Thinking-2507 | gpt-4o-mini | searches |
|---|---|---:|---:|---:|---:|
| monoT5-3B | `rerank_monot5` | 55.9 | 56.3 | 53.4 | 53.3 |
| bge-reranker-v2-m3 | `rerank_bge_m3` | 52.2 | 54.0 | 49.8 | 54.4 |
| Qwen3-Reranker-0.6B | `rerank_qwen3_06b` | 49.8 | 50.2 | 48.0 | 57.0 |
| none (the pool's own order) | `qwen3emb06b` | 41.7 | 42.4 | 38.8 | 62.2 |
| Laya (English) | `rerank_laya` | 33.4 | 34.8 | 32.0 | 71.6 |

The Laya typed-decisions and multilingual checkpoints (`rerank_laya_typed`,
`rerank_laya_multilingual`) are waiting on the Qwen3-32B judge.

The file column is short for `configs/iter/browsecomp_plus_baseline_<file>_tongyi.yaml`.

## Run one

The steps run from a fresh clone. Steps 1 to 3 are shared. Step 4 depends on the row.

**1. Install.** Needs JDK 21 on `JAVA_HOME` and GPUs.

```bash
python -m pip install -e ".[retrieval,api,eval,serve]"
python -m agent_search.tokens --seed
```

**2. Stage the corpus.** On a node with internet. This writes `data/browsecomp_plus/`.

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
| `iter06b`, `iter4b` | the same with `EMBED_MODEL=ielabgroup/ITER-Qwen3-Embedding-0.6B` (or `-4B`) `,DENSE_POOLING=last_token` |
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
skimsearchagent validate configs/iter/browsecomp_plus_baseline_colbert_tongyi.yaml
```

**5. Serve the backbone.**

```bash
export VLLM_USE_FLASHINFER_MOE_FP16=0
vllm serve Alibaba-NLP/Tongyi-DeepResearch-30B-A3B --tensor-parallel-size 1 --port 8000 \
  --max-model-len 131072 --gpu-memory-utilization 0.80 --trust-remote-code
```

**6. Run the row.**

```bash
F=configs/iter/browsecomp_plus_baseline_colbert_tongyi.yaml
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
