# ITER in SkimSearchAgent

Paper: *ITER: Interaction-Aware Retrieval for Agentic Search* (Chen, Wang, Yin, Zhuang, Zuccon,
Leelanupab, 2026), https://arxiv.org/abs/2608.27912; code and released checkpoints at
https://github.com/ielab/ITER.

ITER trains a dense retriever from the trajectories of a search agent and evaluates it inside the
agent loop: the retriever is conditioned on what the agent already searched, and it's trained to
return documents the agent has not read yet. This page maps that setup onto this library. The
generic training recipe (triples from run records, the trainer, plugging a checkpoint back in) is
in [TRAINING.md](TRAINING.md). The install notes, SLURM sharding and the run-directory layout are shared with the Sieve paper
and spelled out in [SIEVE.md](SIEVE.md#reproduce-it).

## Reproduce it

Six steps from a fresh clone to the published ITER numbers on BrowseComp-Plus. Step 5 is the exact
command that produced the table in [Results](#results-on-browsecomp-plus).

**1. Install.** Needs JDK 21 on `JAVA_HOME` and a GPU for serving.

```bash
python -m pip install -e ".[retrieval,api,eval,serve]"
python -m agent_search.tokens --seed
```

**2. Pull the retriever checkpoint and the corpus.** Do this on a node with internet. The runs
themselves stay offline.

```bash
hf download ielabgroup/ITER-Qwen3-Embedding-0.6B     # or -4B
huggingface-cli download wshuai190/browsecomp-plus-structured-full --repo-type dataset --local-dir data/_hf/bcp
cp -r data/_hf/bcp/structured data/browsecomp_plus_structured_full
```

**3. Build the dense cache with the ITER checkpoint.** Documents encode at 512 tokens in bfloat16
with last-token pooling, which is how ITER trained and indexed. The cache directory name carries the
encoder, the length and the precision, so it can't collide with another cache.

```bash
DENSE_SEQ_LENGTH=512 DENSE_DTYPE=bfloat16 DENSE_POOLING=last_token \
skimsearchagent-build-indexes \
  --dataset browsecomp_plus_structured_full \
  --retriever dense --model ielabgroup/ITER-Qwen3-Embedding-0.6B \
  --index-root indexes
```

Check it: `indexes/dense/ielabgroup__ITER-Qwen3-Embedding-0.6B-sl512-eos-bfloat16/browsecomp_plus_structured_full/meta.json`
must say `"n": 100195`.

**4. Serve the backbone.**

```bash
export VLLM_USE_FLASHINFER_MOE_FP16=0
vllm serve Alibaba-NLP/Tongyi-DeepResearch-30B-A3B \
  --tensor-parallel-size 1 --port 8000 --gpu-memory-utilization 0.9 \
  --max-model-len 131072 --compilation-config '{"cudagraph_mode":"PIECEWISE"}' \
  --enable-prefix-caching
```

**5. Run the cell.** These are the settings the published cells ran with. The strategy is
`agent_research_iter_dense`: ten unfiltered results per search, no de-duplication, then
`get_document`. The dense encoder runs on CPU so it doesn't compete with vLLM for the GPU.

```bash
export MAX_VISIT_TOKENS=512 MAX_SECTION_TOKENS=12000 SNIPPET_TOKENS=32 \
       DENSE_QUERY_STYLE=i9 DENSE_POOLING=last_token DENSE_DTYPE=bfloat16 \
       DENSE_SEQ_LENGTH=512 DENSE_QUERY_SEQ_LENGTH=8192 \
       AGENT_CTX_WINDOW=100000 AGENT_CTX_TOKENS=95000 \
       DEDUP_TOPK=10 DEDUP_POOL_K=100 AGENT_SEARCH_DENSE_DEVICE=cpu \
       OPENAI_API_KEY=dummy

skimsearchagent-eval \
  --dataset browsecomp_plus_structured_full \
  --retriever agent_research_iter_dense \
  --dense-model ielabgroup/ITER-Qwen3-Embedding-0.6B \
  --policy llm --backend api --api-base http://127.0.0.1:8000/v1 \
  --model Alibaba-NLP/Tongyi-DeepResearch-30B-A3B \
  --max-steps 50 --temperature 0.6 --seed 42 --workers 2 \
  --k 1 3 5 10 --runs-dir runs/iter/paper_setting_iter06b
```

For the 4B checkpoint, change `--dense-model` to `ielabgroup/ITER-Qwen3-Embedding-4B` and
`--runs-dir` to `runs/iter/paper_setting_iter4b`, after building its cache in step 3.

On SLURM the same run takes ten shards with one GPU each, about an hour per shard. Export the same
knobs plus the ones below, then submit. `scripts/shard_cell.sh` serves vLLM on each node itself.

```bash
export DATASET=browsecomp_plus_structured_full CONDITION=agent_research_iter_dense \
       MODEL=Alibaba-NLP/Tongyi-DeepResearch-30B-A3B DENSE_MODEL=ielabgroup/ITER-Qwen3-Embedding-0.6B \
       RUNS_DIR=runs/iter/paper_setting_iter06b NUM_SHARDS=10 WORKERS=2 MAX_STEPS=50 \
       MAX_MODEL_LEN=131072 JOB_TIME=04:00:00 PREBUILD=0 SLURM_ACCOUNT=YOUR_ACCOUNT \
       VLLM_ARGS="--enable-prefix-caching"
bash scripts/shard_cell.sh
# when every shard has finished:
python scripts/merge_shards.py --runs-dir runs/iter/paper_setting_iter06b \
  --dataset browsecomp_plus_structured_full --condition agent_research_iter_dense \
  --num-shards 10 --model Alibaba-NLP/Tongyi-DeepResearch-30B-A3B
```

The experiment file `configs/iter/browsecomp_plus_full_iter_iter06b_tongyi.yaml` is not the
published setting. It adds DIVER's client behaviour: per-turn budgets of 4096, 2048 and 1024 tokens,
thinking on, DIVER's final-turn nudge and a 10,000-token forced answer. The published cells ran
with the library defaults instead: a flat 4,000-token turn budget, the template's thinking default,
the library's nudge and a 2,000-token forced answer.

**6. Judge it, twice.** Each cell carries two scores. gpt-4o-mini with the BrowseComp Appendix F
prompt is this library's judge:

```bash
D=runs/iter/paper_setting_iter06b/agent/browsecomp_plus_structured_full/Tongyi-DeepResearch-30B-A3B/agent_research_iter_dense
export OPENAI_API_KEY=sk-...
skimsearchagent-judge --results-dir $D --judge-model gpt-4o-mini --workers 16
```

The paper's judge is Qwen3-30B-A3B-Thinking-2507 with DIVER's template, served locally. It writes
`judge_summary_diver.json` next to the first summary.

```bash
vllm serve Qwen/Qwen3-30B-A3B-Thinking-2507 --port 8103 \
  --gpu-memory-utilization 0.9 --max-model-len 32768 --max-num-seqs 64 &
OPENAI_API_KEY=EMPTY python -m agent_search.evaluation.llm_judge --results-dir $D \
  --judge-model Qwen/Qwen3-30B-A3B-Thinking-2507 --judge-api-base http://127.0.0.1:8103/v1 \
  --judge-prompt diver --tag diver --workers 64 --max-tokens 8192 --unfinished-ok
```

Read both numbers:

```bash
python -c "import json; [print(f, round(100*json.load(open(f'$D/'+f))['judge_accuracy'],1)) for f in ('judge_summary.json','judge_summary_diver.json')]"
```

Expect about 44.7 and 48.6 for ITER-0.6B. A second run of the same setup gave 44.3 and 48.2, so a
result within half a point of these matches.

### The ITER experiment files

| file under `configs/iter/` | dataset | strategy | retriever |
|---|---|---|---|
| `browsecomp_plus_full_iter_iter06b_tongyi.yaml` | BCP full, 100,195 docs | `agent_research_iter_dense` | ITER-0.6B |
| `browsecomp_plus_full_iter_iter4b_tongyi.yaml` | BCP full | `agent_research_iter_dense` | ITER-4B |
| `browsecomp_plus_dedup_dense_iter06b_tongyi.yaml` | BCP pooled | `dedup_dense` | ITER-0.6B |
| `browsecomp_plus_dedup_dense_iter06b_{qwen35_4b,qwen35_9b,qwen35_27b}.yaml` | BCP pooled | `agent_research_dedup_dense_qwen` | ITER-0.6B |
| `browsecomp_plus_dedup_dense_iter06b_{gptoss_20b,gptoss_120b}.yaml` | BCP pooled | `agent_research_dedup_dense_strong` | ITER-0.6B |
| `browsecomp_plus_chunks_dedup_dense_iter06b_tongyi.yaml` | BCP chunks | `dedup_dense` | ITER-0.6B |
| `infoseek_eval_dedup_dense_iter06b_tongyi.yaml` | InfoSeek eval, wiki chunks | `dedup_dense` | ITER-0.6B |
| `infoseek_eval_dedup_bm25_tongyi.yaml` | InfoSeek eval | `dedup_bm25` | Lucene |
| `infoseek_train_trajectories_i2.yaml` | InfoSeek train | `dedup_dense` | trajectory generation |
| `sample_*.yaml`, `smoke_*.yaml` | the samples and smokes | | |

## The setting in one file

The three blocks below are what an ITER file sets that an ordinary file doesn't. The complete
files are under `configs/iter/`.

**The tools and the retriever.**

```yaml
strategy: agent_research_iter_dense   # the paper's evaluation setting, no dedup
retrieval:
  dense_model: ielabgroup/ITER-Qwen3-Embedding-0.6B
  dense_pooling: last_token   # Tevatron's --pooling eos, how ITER trained and indexed
  dense_dtype: bfloat16       # ITER encoded its corpora in bfloat16
  dense_seq_length: 512       # documents cut to 512 encoder tokens
  dense_query_seq_length: 8192
  dense_query_style: i9       # the model card's reasoning-augmented query format
  dense_query_instruction: null   # null = the built-in i9 instruction
  ann_ef_search: 256
```

**The budgets.** They're DIVER's, not the Sieve experiments'.

```yaml
agent:
  max_steps: 50               # DIVER's MAX_LLM_CALL_PER_RUN, not the 100 of the Sieve cells
  ctx_tokens: 95000
  ctx_window: 100000
  ctx_stop_frac: 0.9          # force the answer once the prompt passes 90,000 tokens
  forced_answer_tokens: 10000
  forced_answer_nudge: 'Retrieval complete. You are forbidden to call any tools now. Based only on the information already collected above, provide your best final answer.'
budgets:
  max_visit_tokens: 512       # get_document returns at most 512 model tokens
listing:
  dedup_topk: 10              # a search returns the top 10
  dedup_pool_k: 100           # the dedup setting only: over-fetch 100, then drop what was seen
  dedup_snippet_tokens: 64    # each result's passage text, cut to 64 model tokens
model:
  max_tokens: 4096
  max_tokens_schedule: '4096,2048,1024'   # 4096 on turn one, 2048 on turn two, 1024 after
  thinking: true
```

**A corpus too large for memory.** Above 1 GiB
(`AGENT_SEARCH_DOCSTORE_MIN_BYTES`) the loader writes a byte-offset index next to the corpus once
and looks documents up by seek afterwards, so a run never holds the corpus in memory. Retrieval
then goes through prebuilt indexes named in the file.

```yaml
dataset:
  name: infoseek_eval         # data/infoseek_eval/topics.tsv over data/corpora/wiki25_512/corpus.jsonl
retrieval:
  dense_index: indexes/external/iter06b_wiki25_512   # index.faiss + index.lookup.pkl, ITER's layout
  bm25_index: indexes/external/wiki25_512_lucene     # for dedup_bm25 and search_visit
```

## Strategy

ITER's agent has two tools: `search(query)` and `get_document(docid)`. A search returns the top 10
documents, each as its DocID, title and passage text cut to 64 model tokens (ITER cuts with the
served model's tokenizer, the library on its own token ruler, so the counts are close, not
identical). `get_document` returns one document by its DocID, at most 512 model tokens.

The paper evaluates every retriever with unfiltered, non-de-duplicated rankings (Sec. 5.3). That's
`strategy=agent_research_iter_dense` (the run's dense model behind `search`) or
`agent_research_iter_bm25`, both under the task `research_tongyi`: Tongyi's deep-research system
prompt with DIVER's strict tool rules (`agent_search/tasks/research_tongyi/prompt.md`).

The paper collected its training trajectories in a de-duplicated setting (Sec. 4.2.1): a search
over-fetches a pool of 100, drops every document an earlier search already surfaced in the
episode, shows the top 10 of the rest, and lists the hidden documents under "Already-seen" so the
agent can reopen them. That's `strategy=dedup_dense` (or `dedup_bm25`) under the task
`research_dedup`, the same prompt plus DIVER's dedup notice. DIVER's `--strong` prompt for general
backbones is the task `research_dedup_strong`.

DIVER's Tongyi client caps an episode at 50 LLM calls, budgets each turn (a turn cut off
mid-thought is discarded and the next runs with thinking off) and forces the answer once the
conversation passes 90,000 tokens. The keys above carry all of that. Its search tool runs the
first query when the backbone passes a list, as the library's tools do. The paper's
BrowseComp-Plus corpus is the official 100,195-document collection
(`browsecomp_plus_structured_full`, Sec. 5.1); `browsecomp_plus_structured` is the 67,707-document
pooled version.

## Backbones

The backbone is `model.name` plus `model.backend`. DIVER did not run every backbone through the
same client, and the library keeps that split, one task and one driver per family:

| backbone family | task | driver | what DIVER's client does |
|---|---|---|---|
| Tongyi-DeepResearch-30B | `research_dedup` | `loop` | Tongyi's deep-research persona, strict tool rules, answer tags, tool calls as `<tool_call>` text |
| Qwen3.5 (4B, 9B, 27B), WebExplorer | `research_dedup_qwen` | `loop` | "You are a helpful assistant." plus the tools block; no answer tags, the first reply without a tool call is the answer; temperature 1.0, top-k 20, presence penalty 1.5, thinking on, generation budgets 4096 / 2048 / 1024 by turn, a turn cut off mid-thought is discarded and the next runs with thinking off, the answer is forced once the conversation passes 26k tokens |
| gpt-oss (20B, 120B) | `research_dedup_strong` | `responses` | the strong prompt with the dedup notice as `instructions`, DIVER's question template as the user turn, native function calling through `/v1/responses`, reasoning effort medium, 10,000 output tokens per turn, the final turn made with no tools |

The dedup notice is appended to every family's prompt. The Qwen and gpt-oss files set the sampling
keys (`model.top_k`, `presence_penalty`, `max_tokens_schedule`, `thinking`) and the driver;
nothing else differs from the Tongyi file.

vLLM has to be started the way DIVER started it for the tool-call and reasoning parsers to apply.
Add these to `vllm serve`, or pass them through `VLLM_ARGS` when `scripts/shard_cell.sh` serves
the model:

```bash
# Qwen3.5
--enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3 --gdn-prefill-backend triton
# gpt-oss (also --gpu-memory-utilization 0.8)
--enable-auto-tool-choice --tool-call-parser openai --reasoning-parser openai_gptoss
```

The FlashInfer gated-delta-net kernel is JIT-compiled with nvcc and fails on CUDA 12.4, which is
why Qwen3.5 gets `--gdn-prefill-backend triton`. A hosted API model needs no GPU, but only if the
compute nodes have network egress. On a cluster without it, serve the backbone on the node.

## Retrievers

Any embedding model goes in `retrieval.dense_model`. ITER's released checkpoints
(`ielabgroup/ITER-Qwen3-Embedding-0.6B`, `-4B`), the plain `Qwen/Qwen3-Embedding-*` baselines, and
LRAT (`Yuqi-Zhou/LRAT-Qwen3-Embedding-0.6B`) are decoder checkpoints without a sentence-transformers
config. The library serves them with its plain decoder encoder
(`agent_search/retrievers/dense/decoder_encoder.py`): the checkpoint's tokenizer with its end token,
left padding, the last position pooled, unit-length vectors, documents and queries cut at 512
tokens (`retrieval.dense_seq_length`). That is Tevatron's `--pooling eos` encoding, which is how
ITER trained and indexed; it reproduces ITER's published vectors to 0.997 cosine.

The query side is the pair `retrieval.dense_query_style` + `retrieval.dense_query_instruction`.
Use `i9` for the released checkpoints, the model card's reasoning-augmented format: the main
question, the agent's pre-search reasoning on one line, the sub-query, then the sub-queries already
tried. The paper calls it ITER-i7. Leave the instruction `null` to get the built-in i9 text:
"Given the main question, the agent's reasoning and the current sub-query it led to, and the
sub-queries already tried in previous interactions, retrieve documents relevant to the current
sub-query that provide NEW information not yet found." Use `plain` for the baselines. Queries run
to 8192 tokens (`retrieval.dense_query_seq_length`), documents to 512.

The ITER-0.6B hub revision of 2026-09-11 replaced the weights. A cache records the weights
revision and is rebuilt when it changes.

## Corpus and datasets

ITER's files drop into `data/` (symlinks are fine):

```
data/corpora/wiki25_512/corpus.jsonl        # {"docid": 0, "text": "Title\n..."}   11.2M chunks, 18 GB
data/infoseek_eval/topics.tsv               # id <TAB> question <TAB> answer       300 questions, answer-only
data/infoseek_train/topics.tsv              # id <TAB> question <TAB> answer       9,894 questions
data/browsecomp_plus_chunks/corpus.jsonl    # 3.3 GB of chunks
data/browsecomp_plus_chunks/topics.tsv      # id <TAB> question <TAB> answer
data/browsecomp_plus_chunks/qrels.txt       # TREC: qid Q0 docid rel
```

They register as `infoseek_eval`, `infoseek_train` (both over `wiki25_512`) and
`browsecomp_plus_chunks`. To add your own topic set over an existing corpus, one line does it
(`corpus=` lets several topic sets share one corpus and its indexes):

```python
from agent_search.evaluation.datasets import register_dataset, _topics_qrels_loader
register_dataset("my_topics", domain="general")(_topics_qrels_loader("my_topics", corpus="wiki25_512"))
```

InfoSeek has answers but no document labels. Such an answer-only set runs like any other: rank
metrics are left out of the rows. The answer metrics and the LLM judge (`evaluation.judge_model`)
score it instead.

## Import a corpus you already embedded

ITER's `encode.py` (Tevatron) writes one pickle per shard, `(embeddings, doc_ids)`.
`scripts/import_tevatron_index.py` turns those shards into the prebuilt index layout the library
serves (`index.faiss` + `index.lookup.pkl`) and checks every id against the corpus.

```bash
python scripts/import_tevatron_index.py --shards "/path/to/index-*.pkl" \
    --out indexes/external/iter06b_bcp_chunks --corpus data/browsecomp_plus_chunks/corpus.jsonl
```

Point `retrieval.dense_index` at the result. The corpus is then served from disk and never
re-embedded. The query side still needs the encoder (`retrieval.dense_model`).

Re-indexing the whole wiki corpus with a new checkpoint is a sharded GPU job outside this library
(ITER's `build_faiss_index.sh` plus `build_ann_index.py`). Point `retrieval.dense_index` at its
output.

## Sample a paper setting first

`skimsearchagent-sample-dataset` cuts a small dataset out of a big one in the same layout: a few
questions with their answers and qrels, every gold document of those questions, an optional BM25
pool per question and random chunks. A folder under `data/` with `topics.tsv` and `corpus.jsonl`
is picked up without code, so a retriever or a backbone can be tried on ITER's setting in minutes.

```bash
skimsearchagent-sample-dataset --dataset browsecomp_plus_chunks --out browsecomp_plus_chunks_sample \
    --n-topics 20 --n-docs 20000
skimsearchagent-sample-dataset --dataset infoseek_eval --out infoseek_eval_sample \
    --n-topics 20 --n-docs 20000 --pool-bm25-index indexes/external/wiki25_512_lucene --pool-k 100
sbatch --account=YOUR_ACCOUNT --qos=express \
    --export=ALL,VLLM_PYTHON=/path/to/vllm-env/bin/python scripts/slurm/iter_sample.sbatch
```

The job indexes both samples with the released ITER retriever, serves Tongyi, and runs
`configs/iter/sample_*_iter06b_tongyi.yaml`. Judge the answer-only set afterwards where the API is
reachable:

```bash
skimsearchagent-judge --results-dir runs/iter/sample_runs/agent/infoseek_eval_sample/<model>/<condition> \
    --judge-model gpt-4o-mini
```

## Evaluate a retriever without an agent

```bash
skimsearchagent-eval-retriever --triples train_data/infoseek_i2.jsonl --dataset infoseek_train \
    --dense-model models/my-retriever --k 1,5,10 --subset 5000
```

It reports Recall@k of the positive and novelty@k (the share of the top k the agent had not read)
for every triple. `--subset N` encodes only the triples' documents plus the first N chunks, a quick
check to run after a smoke run. `--bm25` replaces `--dense-model` to score the lexical baseline.

## The training loop, on SLURM

Three stages, each its own job. Never on a login node. The recipe itself is in
[TRAINING.md](TRAINING.md) section 5.

```bash
# 1. trajectories: configs/iter/smoke_infoseek_train_gpt4omini.yaml
sbatch --account=YOUR_ACCOUNT --export=ALL scripts/slurm/iter_smoke_traj.sbatch
# 2. triples + fine-tune (uses envs-train; TRAIN_PYTHON= points it elsewhere)
sbatch --account=YOUR_ACCOUNT \
  --export=ALL,RUNS=runs/iter/smoke/agent/infoseek_train/<model>/agent_research_dedup_dense \
  scripts/slurm/iter_smoke_train.sbatch
# 3. trained vs base retriever on the triples
sbatch --account=YOUR_ACCOUNT --export=ALL scripts/slurm/iter_smoke_eval.sbatch
```

The paper-scale files are `configs/iter/infoseek_train_trajectories_i2.yaml` (trajectory
generation with Tongyi), `configs/iter/infoseek_eval_dedup_dense_iter06b_tongyi.yaml` and the
BrowseComp-Plus counterpart.

## Results on BrowseComp-Plus

Two cells in the paper's evaluation setting: the official 100,195-document corpus, documents encoded
at 512 tokens, unfiltered top-10 rankings with no de-duplication, 50 search calls,
Tongyi-DeepResearch-30B as the backbone, all 830 questions. Each one is scored twice, by the paper's
judge and by ours, because the judge choice moves the number by about four points.

| retriever | paper's judge | gpt-4o-mini | steps | tokens | paper |
|---|---|---|---|---|---|
| ITER-Qwen3-Embedding-0.6B | 48.6 | 44.7 | 43.9 | 46.5k | 49.2 |
| ITER-Qwen3-Embedding-4B | 51.1 | 47.7 | 40.4 | 43.0k | 51.2 |

Both cells land within a point of the paper on the paper's own judge: 48.6 against 49.2, and 51.1
against 51.2. A second run of the 0.6B cell with the same setup scored 48.2, so run-to-run noise is
about half a point.

The two rows come from different code. The 0.6B cell ran on 21 September. The 4B cell ran on 14
September, before three fixes to the agent loop: repairs for malformed tool calls, a filled skeleton
for an empty tool call, and a forced answer when the context window fills instead of a dropped
question. On the 0.6B cell those fixes were worth 2.5 points under either judge, so read the 4B row as
a lower bound for the current code.

Use the judge column that matches what you're comparing against. DIVER's own released answers for the
i2 setting score 38.9 under gpt-4o-mini against the 44.5 they report, a 5.6-point gap in the same
direction, so a number judged by gpt-4o-mini reads low next to any published ITER figure.
`runs/iter/diver_i2_rescored/` holds that reference so you can re-check it.

The backbone cells (Qwen3.5 at 4B, 9B and 27B, gpt-oss at 20B and 120B) ran under DIVER's de-duplicated
setting with documents encoded at 1,024 tokens, which a later fix corrected to 512. Those runs were
deleted and the numbers retracted rather than published on a superseded setting. Rerunning them in the
setting above is open work. The smoke pipeline, sample runs, training check and held-out check ran end
to end on SLURM with the launchers in `scripts/slurm/`; their 20-question numbers aren't reported.
