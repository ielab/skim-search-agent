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

ITER's agent has two tools: `search(query)` and `get_document(docid)`. A search returns the top 10
documents, each as its DocID, title and passage text cut to 64 model tokens (ITER cuts with the
served model's tokenizer, the library on its own token ruler, so the counts are close, not
identical). `get_document` returns one document by its DocID, at most 512 model tokens. The paper
evaluates every retriever this way, with unfiltered, non-de-duplicated rankings (Sec. 5.3): that
is `strategy=agent_research_iter_dense` (the run's dense model behind `search`) or
`agent_research_iter_bm25`, under the task `research_tongyi`: Tongyi's deep-research system prompt
with DIVER's strict tool rules (`agent_search/tasks/research_tongyi/prompt.md`).

The paper collected its training trajectories in a de-duplicated setting (Sec. 4.2.1): a search
over-fetches a pool of 100, drops every document an earlier search already surfaced in the
episode, shows the top 10 of the rest, and lists the hidden documents under "Already-seen" so the
agent can reopen them. That is `strategy=dedup_dense` (or `dedup_bm25`) under the task
`research_dedup`, the same prompt plus DIVER's dedup notice. DIVER's `--strong` prompt for general
backbones is the task `research_dedup_strong`.

DIVER's Tongyi client caps an episode at 50 LLM calls (`MAX_LLM_CALL_PER_RUN`), so ITER cells run
with `max_steps: 50`, not the 100 of the Sieve experiments. It also budgets each turn (4,096
generation tokens on the first turn, 2,048 on the second, 1,024 after; a turn cut off mid-thought
is discarded and the next runs with thinking off) and forces the answer once the conversation
passes 90,000 tokens with "Retrieval complete. You are forbidden to call any tools now. ...", with
a 10,000-token budget and an `<answer>` prefill (`model.max_tokens_schedule`, `model.thinking`,
`agent.ctx_window`, `agent.forced_answer_nudge`, `agent.forced_answer_tokens`); the ITER files
carry all of that. Its search tool runs the first query when the backbone passes a list, as the
library's tools do. The paper's BrowseComp-Plus corpus is the official 100,195-document collection
(`browsecomp_plus_structured_full`, Sec. 5.1); `browsecomp_plus_structured` is the 67,707-document
pooled version.

Listing knobs: `listing.dedup_topk` (10), `listing.dedup_pool_k` (100, the dedup setting only) and
`listing.dedup_snippet_tokens` (64). `get_document` returns at most 512 model tokens
(`budgets.max_visit_tokens: 512`, cut on the same ruler). The wiki chunks are already 512 tokens,
so both limits only matter on a corpus with longer documents.

## Backbones

The backbone is `model.name` plus `model.backend`. DIVER did not run every backbone through the
same client, and the library keeps that split, one task and one driver per family:

| backbone family | task | driver | what DIVER's client does |
|---|---|---|---|
| Tongyi-DeepResearch-30B | `research_dedup` | `loop` | Tongyi's deep-research persona, strict tool rules, answer tags, tool calls as `<tool_call>` text |
| Qwen3.5 (4B, 9B, 27B), WebExplorer | `research_dedup_qwen` | `loop` | "You are a helpful assistant." plus the tools block; no answer tags, the first reply without a tool call is the answer; temperature 1.0, top-k 20, presence penalty 1.5, thinking on, generation budgets 4096 / 2048 / 1024 by turn, a turn cut off mid-thought is discarded and the next runs with thinking off, the answer is forced once the conversation passes 26k tokens |
| gpt-oss (20B, 120B) | `research_dedup_strong` | `responses` | the strong prompt with the dedup notice as `instructions`, DIVER's question template as the user turn, native function calling through `/v1/responses`, reasoning effort medium, 10,000 output tokens per turn, the final turn made with no tools |

The dedup notice is appended to every family's prompt. The three settings are complete files:
`configs/iter/browsecomp_plus_dedup_dense_iter06b_tongyi.yaml`,
`..._qwen35_4b.yaml` (9b, 27b) and `..._gptoss_20b.yaml` (120b). The Qwen and gpt-oss files set
the sampling keys (`model.top_k`, `presence_penalty`, `max_tokens_schedule`, `thinking`) and the
driver; nothing else differs from the Tongyi file.

vLLM has to be started the way DIVER started it for the tool-call and reasoning parsers to apply,
through `VLLM_ARGS` when the shard script serves the model:

```bash
# Qwen3.5: --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3 --gdn-prefill-backend triton
#          (the FlashInfer gated-delta-net kernel is JIT-compiled with nvcc and fails on CUDA 12.4; Triton needs no compile)
# gpt-oss: --enable-auto-tool-choice --tool-call-parser openai --reasoning-parser openai_gptoss   (GPU_UTIL=0.8)
```

```yaml
model:
  name: Alibaba-NLP/Tongyi-DeepResearch-30B-A3B   # or Qwen/Qwen3.5-27B, openai/gpt-oss-120b
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
config. The library serves them with its plain decoder encoder
(`agent_search/retrievers/dense/decoder_encoder.py`): the checkpoint's tokenizer with its end token,
left padding, the last position pooled, unit-length vectors, documents and queries cut at 512
tokens (`retrieval.dense_seq_length`). That is Tevatron's `--pooling eos` encoding, which is how
ITER trained and indexed; it reproduces ITER's published vectors to 0.997 cosine. ITER encoded its
corpora in bfloat16, so the shipped files set `retrieval.dense_dtype: bfloat16`. The query side is
the pair `retrieval.dense_query_style` + `retrieval.dense_query_instruction`: `i9` for the released
checkpoints (the model card's reasoning-augmented format: main question, the agent's pre-search
reasoning on one line, the sub-query, the previous sub-queries; the paper calls it ITER-i7), with
the built-in i9 instruction; `plain` for the baselines. Queries run to 8192 tokens
(`retrieval.dense_query_seq_length`), documents to 512. The ITER-0.6B hub revision of 2026-09-11
replaced the weights; a cache records the weights revision and is rebuilt when it changes. Pull a released checkpoint once on a
node with internet (`hf download ielabgroup/ITER-Qwen3-Embedding-0.6B`). The runs themselves stay
offline.

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

## Results on BrowseComp-Plus

The ITER protocol cells ran on this code with Tongyi-DeepResearch-30B-A3B as the backbone: DIVER's
Tongyi prompt with the dedup notice, 50 LLM calls, dedup on (pool 100, top 10), 64-model-token
snippets, 512-token reads, i9 queries at 8,192 tokens, the encoder served as trained (end token
kept, last position pooled, bfloat16), the forced answer at 10,000 tokens, seed 42, temperature 0.6.
830 questions, the 67,707-document structured corpus. Accuracy is our gpt-4o-mini judge. The paper
column is DIVER's own judge; that judge scored DIVER's released answers 5.6 points higher than ours
on the same trajectories, so read the two columns with that gap in mind.

| retriever and prompt | accuracy % (our judge) | paper (DIVER's judge) | steps | tokens once (k) | at the 50-call cap | runs dir |
|---|---|---|---|---|---|---|
| ITER-Qwen3-Embedding-0.6B, DIVER prompt | 41.2 | 44.8 | 44.7 | 49.3 | 521 | `bcp_s_iter` |
| ITER-Qwen3-Embedding-4B, DIVER prompt | 40.7 | 45.7 | 44.4 | 47.6 | 538 | `bcp_s_iter_4b` |
| Qwen3-Embedding-0.6B (untrained), DIVER prompt | 34.3 | 40.6 | 45.7 | 52.4 | 582 | `bcp_s_iter_qwen06b` |
| ITER-0.6B, combined prompt | 35.3 | - | 45.4 | 48.0 | 593 | `bcp_s_prompt` |
| ITER-0.6B, paper prompt | 34.0 | - | 43.7 | 45.6 | 451 | `bcp_s_prompt` |

The three retrievers land where the paper puts them once the judge gap is applied. Inside the
trajectories the trained retriever is the difference: final recall@10 is 0.278 for ITER-0.6B,
0.361 for ITER-4B and 0.104 for the untrained Qwen3-Embedding-0.6B; gold documents first surface
at the second search with ITER and at the fifth with Qwen, and a quarter of the Qwen episodes never
surface one (8% with ITER-0.6B). The 4B retriever finds more yet the judged accuracy is the same
within noise (standard error 1.7 points): Tongyi compensates for the weaker 0.6B retriever.

The prompt matters in the other direction from the Sieve tools. On ITER's `search` and
`get_document`, DIVER's own prompt beats the library's combined prompt by six points and the Sieve
paper's prompt by seven, with identical retrieval in all three cells; on the Sieve tools the
combined prompt beats the paper prompt by eight. That is why the ITER task keeps DIVER's prompt
and the other research strategies use the combined one.

DIVER's prompt is not the same for every backbone. Tongyi gets the deep-research persona, the
answer tags and the strict tool rules; Qwen3.5 and WebExplorer get "You are a helpful assistant."
plus the tools block; gpt-oss gets a one-line system prompt with the tools passed as native
function definitions. The library carries all three (the Backbones section above), and the table
below runs them on the same retriever.

### Backbones on the ITER protocol

Same protocol and retriever as above (ITER-Qwen3-Embedding-0.6B, i9 queries, dedup on, 50 calls),
each backbone under the task and driver DIVER used for it. Our judge is gpt-4o-mini; the paper
column is DIVER's own backbone table, which scores a different retriever (DIVER, or LRAT) with
string-match accuracy, so it shows the order the paper found, not a like-for-like number.

| backbone | task, driver | accuracy % (our judge) | DIVER paper (DIVER / LRAT retriever, string match) | steps | tokens once (k) | at the 50-call cap |
|---|---|---|---|---|---|---|
| Tongyi-DeepResearch-30B-A3B | research_dedup (Tongyi prompt), loop | 41.2 | 44.8 (ITER paper) | 44.7 | 49.3 | 521 |
| Qwen3.5-27B | research_dedup_qwen, loop | 33.5 | 48.3 / 42.2 | 29.2 | 25.4 | 142 |
| Qwen3.5-9B | research_dedup_qwen, loop | 37.7 | 32.4 / 32.8 | 27.0 | 33.3 | 41 |
| Qwen3.5-4B | research_dedup_qwen, loop | 25.1 | 24.6 / 26.7 | 26.9 | 30.0 | 71 |
| gpt-oss-120b | research_dedup_strong, responses | 39.4 | 37.5 / 32.0 | 17.9 | 21.4 | 1 |
| gpt-oss-20b | research_dedup_strong, responses | 29.2 | 21.6 / 22.2 | 29.5 | 37.8 | 0 |

Tongyi stays first on its own retriever. gpt-oss-120b is second with the fewest calls and tokens
of any backbone by a wide margin, and the 4B model trails, as in the paper. The one departure from
the paper's order is Qwen3.5-27B, which the paper ranks first and which lands third here. Its
trajectories show why: under DIVER's turn budgets (4096, 2048, then 1024 generation tokens) the
27B model keeps reasoning in the open after its thinking is switched off, so a turn is cut before
it reaches a tool call 1,188 times in 830 episodes, 17% of its episodes run out of calls against 5%
for the 9B, and its gold coverage ends below the 4B's. The behaviour is the protocol's, not a
serving fault: the chat template honours the thinking switch (the follow-up turns carry no think
block) and the budgets are DIVER's.

Every backbone family needed one serving detail to run at all: Tongyi's text protocol needed the
tool-call repairs described on the [reproduction page](REPRODUCING.md), Qwen3.5 needed the reasoning channel read as the reply
when vLLM's Qwen3 parser filed a reply without a closing think tag as reasoning, and gpt-oss needed
a turn the server refused to re-parse dropped and taken again. Those are library behaviour now, not
per-run patches.

The earlier smoke pipeline, sample runs, training check and held-out check ran end to end on
SLURM with the launchers in `scripts/slurm/`; their 20-question numbers are not reported.

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
  dense_query_style: i9       # main question, pre-search reasoning, sub-query, earlier sub-queries, as the released model was trained
  dense_query_instruction: null   # the built-in i9 instruction: "Given the main question, the agent's reasoning and the current sub-query it led to, and the sub-queries already tried in previous interactions, retrieve documents relevant to the current sub-query that provide NEW information not yet found."
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
