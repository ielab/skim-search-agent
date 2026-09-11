# Verification

Every strategy, four datasets, the code fixture, the served backbones of both papers and the
training path were run on the cluster after the 0.3 restructure, one GPU job per check, with
`scripts/checks_matrix.py`. This page records what was run and what came out. The numbers are
smoke checks over 20 questions each at seed 42, not paper results; a 20-question sample moves by
two or three questions between runs of the same code because served sampling is not
bit-reproducible.

## How to rerun

```bash
export SSA_SLURM_ACCOUNT=<account> SSA_SLURM_QOS=<qos> VLLM_PYTHON=/path/to/vllm-env/bin/python \
       JAVA_HOME_OVERRIDE=/path/to/jdk-21
python scripts/checks_matrix.py write      # one experiment file per check under configs/checks/
python scripts/checks_matrix.py submit     # one job per file; `submit <dataset|backbone|stem>` for a subset
python scripts/checks_matrix.py table      # rows, errors, steps and accuracy per run under runs/checks
```

Judge the document runs from a node that reaches the API:
`skimsearchagent-judge --results-dir <run dir> --judge-model gpt-4o-mini`. The training path is
`scripts/slurm/iter_smoke_train.sbatch` then `iter_smoke_eval.sbatch`, and the trained checkpoint
behind the agent is `configs/iter/sample_infoseek_eval_trained_tongyi.yaml`.

## Datasets and backbones

| dataset | what it is | strategies checked |
|---|---|---|
| `infoseek_eval_sample` | 20 InfoSeek-Eval questions over 21,997 flat Wikipedia chunks, gold answers | every strategy, every backbone, the floors, one-shot RAG |
| `browsecomp_plus_chunks_sample` | 20 BrowseComp-Plus questions with qrels over 20,092 flat chunks | Search-Visit, ITER's loop |
| `browsecomp_plus_structured` | 20 questions over 67,707 documents with sections, author and date (browsecomp field profile) | Sieve, fused Sieve, Search-Fetch, Indri |
| `hotpotqa_structured` | 20 questions over 59,833 documents with sections and infobox (wiki field profile) | Sieve, Search-Fetch, Indri, Search-Visit |
| `code_fixture` | one bug report over two repository files | codefix, codefix_grep, codefix_patch |

Backbones served with vLLM on the node: Tongyi-DeepResearch-30B-A3B (both papers),
Qwen3-30B-A3B-Instruct-2507, Qwen3-30B-A3B-Thinking-2507, Qwen3-8B, gpt-oss-120b (four GPUs).
gpt-4o-mini was checked from the login node, the only place here with API egress: Sieve and
Search-Visit on the fixture, both correct. Qwen-AgentWorld-35B-A3B is not in the matrix: its
released checkpoint has no vision weights while its config declares the vision-language
architecture, and vLLM 0.18 has no text-only loader for it.

## Results

Every row scored all of its questions with no errors. `judged` is gpt-4o-mini's verdict on the
answer; `acc@5` is the rank metric where the dataset has qrels; `fix ok` is the code task's score.

| dataset | retriever | backbone | rows | errors | steps | acc@5 | fix ok | judged |
|---|---|---|---|---|---|---|---|---|
| browsecomp_plus_chunks_sample | agent_research_bm25 | Tongyi-DeepResearch-30B-A3B | 20 | 0 | 26.95 | 0.00 |  | 8/20 |
| browsecomp_plus_chunks_sample | agent_research_dedup_dense | Tongyi-DeepResearch-30B-A3B | 20 | 0 | 41.00 | 0.00 |  | 5/20 |
| browsecomp_plus_structured | agent_research_bm25_fetch_snip | Tongyi-DeepResearch-30B-A3B | 20 | 0 | 32.00 | 0.00 |  | 6/20 |
| browsecomp_plus_structured | agent_research_bql_dense_snip | Tongyi-DeepResearch-30B-A3B | 20 | 0 | 28.85 | 0.00 |  | 8/20 |
| browsecomp_plus_structured | agent_research_indri_snip | Tongyi-DeepResearch-30B-A3B | 20 | 0 | 34.10 | 0.05 |  | 8/20 |
| browsecomp_plus_structured | agent_research_snip | Tongyi-DeepResearch-30B-A3B | 20 | 0 | 28.00 | 0.05 |  | 7/20 |
| code_fixture | agent_codefix | Tongyi-DeepResearch-30B-A3B | 1 | 0 | 8.00 | 0.00 | 1.00 |  |
| code_fixture | agent_codefix_grep | Tongyi-DeepResearch-30B-A3B | 1 | 0 | 40.00 | 0.00 |  |  |
| code_fixture | agent_codefix_patch | Tongyi-DeepResearch-30B-A3B | 1 | 0 | 3.00 | 0.00 | 1.00 |  |
| hotpotqa_structured | agent_research_bm25 | Tongyi-DeepResearch-30B-A3B | 20 | 0 | 19.55 | 0.30 |  | 7/20 |
| hotpotqa_structured | agent_research_bm25_fetch_snip | Tongyi-DeepResearch-30B-A3B | 20 | 0 | 18.85 | 0.35 |  | 9/20 |
| hotpotqa_structured | agent_research_indri_snip | Tongyi-DeepResearch-30B-A3B | 20 | 0 | 24.80 | 0.35 |  | 8/20 |
| hotpotqa_structured | agent_research_snip | Tongyi-DeepResearch-30B-A3B | 20 | 0 | 18.05 | 0.55 |  | 9/20 |
| infoseek_eval_sample | agent_rag_bm25 | Qwen3-30B-A3B-Instruct-2507 | 20 | 0 | 1.00 |  |  | 15/20 |
| infoseek_eval_sample | agent_research_bm25 | Qwen3-30B-A3B-Instruct-2507 | 20 | 0 | 4.95 |  |  | 14/20 |
| infoseek_eval_sample | agent_research_dedup_dense | Qwen3-30B-A3B-Instruct-2507 | 20 | 0 | 8.45 |  |  | 11/20 |
| infoseek_eval_sample | agent_research_snip | Qwen3-30B-A3B-Instruct-2507 | 20 | 0 | 5.65 |  |  | 12/20 |
| infoseek_eval_sample | agent_research_bm25 | Qwen3-30B-A3B-Thinking-2507 | 20 | 0 | 3.35 |  |  | 12/20 |
| infoseek_eval_sample | agent_research_dedup_dense | Qwen3-30B-A3B-Thinking-2507 | 20 | 0 | 5.30 |  |  | 11/20 |
| infoseek_eval_sample | agent_research_snip | Qwen3-30B-A3B-Thinking-2507 | 20 | 0 | 4.15 |  |  | 9/20 |
| infoseek_eval_sample | agent_rag_bm25 | Qwen3-8B | 20 | 0 | 1.00 |  |  | 10/20 |
| infoseek_eval_sample | agent_research_bm25 | Qwen3-8B | 20 | 0 | 20.00 |  |  | 12/20 |
| infoseek_eval_sample | agent_research_dedup_dense | Qwen3-8B | 20 | 0 | 31.30 |  |  | 12/20 |
| infoseek_eval_sample | agent_research_snip | Qwen3-8B | 20 | 0 | 15.30 |  |  | 13/20 |
| infoseek_eval_sample | agent_rag_bm25 | Tongyi-DeepResearch-30B-A3B | 20 | 0 | 1.00 |  |  | 13/20 |
| infoseek_eval_sample | agent_rag_dense | Tongyi-DeepResearch-30B-A3B | 20 | 0 | 1.00 |  |  | 12/20 |
| infoseek_eval_sample | agent_rag_hybrid | Tongyi-DeepResearch-30B-A3B | 20 | 0 | 1.00 |  |  | 13/20 |
| infoseek_eval_sample | agent_research_bm25 | Tongyi-DeepResearch-30B-A3B | 20 | 0 | 16.40 |  |  | 13/20 |
| infoseek_eval_sample | agent_research_bm25_autoread | Tongyi-DeepResearch-30B-A3B | 20 | 0 | 8.80 |  |  | 16/20 |
| infoseek_eval_sample | agent_research_bm25_dci | Tongyi-DeepResearch-30B-A3B | 20 | 0 | 15.40 |  |  | 15/20 |
| infoseek_eval_sample | agent_research_bm25_fetch_snip | Tongyi-DeepResearch-30B-A3B | 20 | 0 | 13.45 |  |  | 13/20 |
| infoseek_eval_sample | agent_research_bql_dense_fetch | Tongyi-DeepResearch-30B-A3B | 20 | 0 | 17.50 |  |  | 13/20 |
| infoseek_eval_sample | agent_research_bql_dense_snip | Tongyi-DeepResearch-30B-A3B | 20 | 0 | 14.95 |  |  | 14/20 |
| infoseek_eval_sample | agent_research_bql_donly_snip | Tongyi-DeepResearch-30B-A3B | 20 | 0 | 13.15 |  |  | 17/20 |
| infoseek_eval_sample | agent_research_dci | Tongyi-DeepResearch-30B-A3B | 20 | 0 | 14.60 |  |  | 12/20 |
| infoseek_eval_sample | agent_research_dedup_bm25 | Tongyi-DeepResearch-30B-A3B | 20 | 0 | 22.90 |  |  | 11/20 |
| infoseek_eval_sample | agent_research_dedup_dense | Tongyi-DeepResearch-30B-A3B | 20 | 0 | 27.90 |  |  | 13/20 |
| infoseek_eval_sample | agent_research_dense | Tongyi-DeepResearch-30B-A3B | 20 | 0 | 16.65 |  |  | 12/20 |
| infoseek_eval_sample | agent_research_dense_autoread | Tongyi-DeepResearch-30B-A3B | 20 | 0 | 13.90 |  |  | 14/20 |
| infoseek_eval_sample | agent_research_dense_fetch | Tongyi-DeepResearch-30B-A3B | 20 | 0 | 16.30 |  |  | 14/20 |
| infoseek_eval_sample | agent_research_hybrid | Tongyi-DeepResearch-30B-A3B | 20 | 0 | 24.10 |  |  | 11/20 |
| infoseek_eval_sample | agent_research_hybrid_fetch_snip | Tongyi-DeepResearch-30B-A3B | 20 | 0 | 15.35 |  |  | 14/20 |
| infoseek_eval_sample | agent_research_indri_snip | Tongyi-DeepResearch-30B-A3B | 20 | 0 | 14.15 |  |  | 13/20 |
| infoseek_eval_sample | agent_research_snip | Tongyi-DeepResearch-30B-A3B | 20 | 0 | 16.15 |  |  | 12/16 |
| infoseek_eval_sample | agent_research_bm25 | gpt-oss-120b | 20 | 0 | 11.05 |  |  | 15/20 |
| infoseek_eval_sample | agent_research_dedup_dense | gpt-oss-120b | 12 | 0 | 39.50 |  |  | 2/12 |
| infoseek_eval_sample | agent_research_snip | gpt-oss-120b | 20 | 0 | 13.90 |  |  | 16/20 |
| infoseek_eval_sample | bm25_local | ITER-Qwen3-Embedding-0.6B | 20 | 0 |  |  |  |  |
| infoseek_eval_sample | bql | ITER-Qwen3-Embedding-0.6B | 20 | 0 |  |  |  |  |
| infoseek_eval_sample | dense | ITER-Qwen3-Embedding-0.6B | 20 | 0 |  |  |  |  |

The training path: 9 smoke triples, one epoch, the checkpoint saved with its serving note;
evaluated against its base on the same triples (recall@10 0.89 vs 1.0, four steps do not move a
retriever); the checkpoint trained on 99 InfoSeek trajectories behind the agent on the InfoSeek
sample scored 12/20. The three floors (`bm25`, `dense`, `bql`) scored all 20 questions.

## The old code and the new code

The restructure kept the paper's behaviour. Checked three ways at commit 9dfa5b2, where both
codes were present:

- Every pinned system prompt renders byte for byte, every tool matches its old workspace
  observation for observation, and a stub episode is identical end to end for every condition.
- Replay: the recorded generations of a run are fed to both codes over the same index, and the
  observations are compared step by step. On the InfoSeek sample (ITER's loop, 486 steps) and on
  the structured BrowseComp-Plus corpus (Sieve, 646 steps) the two codes agree on every step.
- The same settings run through both codes, same index, seed 42, Tongyi, judged by gpt-4o-mini:

| dataset | strategy | old code | new code |
|---|---|---|---|
| InfoSeek-Eval sample | Sieve | 15/20 | 15/20 |
| InfoSeek-Eval sample | Search-Visit, Lucene BM25 | 13/19 | 14/20 |
| InfoSeek-Eval sample | ITER's loop | 13/20, 13/20 | 13/20, 10/20 |
| BrowseComp-Plus chunk sample | Sieve | 7/20 | 5/20 |
| BrowseComp-Plus chunk sample | Search-Visit | 6/20 | 5/20 |
| BrowseComp-Plus structured | Sieve, three runs each | 10/19, 8/20, 7/20 | 5/20, 8/20, 9/20 |

The first structured pair looked like a gap; two more runs of each code and the replay showed it
was the spread of the same distribution.

## What the matrix found and fixed

- Several jobs on one node served their models on the same port, so one job answered another's
  requests. Each launcher now derives its port from the job id.
- A hub refresh left the newest cache snapshot of the ITER checkpoint holding only a README, and
  the loader served it; every dense search after that failed. The loader now serves a complete
  snapshot.
- A run directory left by an attempt that scored nothing blocked the rerun as "a different
  experiment"; it is a fresh start now. The served endpoint's address is not part of a run's
  identity.
- Eleven strategies existed in code without a condition to run them; every strategy now runs
  under its own name.
- gpt-oss-120b needs four GPUs here (the weights dequantise to bf16) and a smaller batch; with
  ITER's prompt it rarely emits a tool call, so its ITER row is weak while its Sieve and
  Search-Visit rows are the best in the table.
