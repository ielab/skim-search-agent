# Cluster launchers

Heavy work goes through SLURM and never runs on the login node: model serving, corpus embedding,
index builds over real corpora, training. Every script here is dual-mode. Run it with `bash` and it
executes in the current shell, which suits a compute-node debug session. Run it with `sbatch` and
it submits.

Nothing here hardcodes a site. Pass the account and partition on the command line:

```bash
sbatch --account=YOUR_ACCOUNT --partition=h24gpu --qos=normal scripts/slurm/smoke_suite.sbatch
sbatch --account=YOUR_ACCOUNT --partition=h24gpu --gres=gpu:2 \
    --export=ALL,EXPERIMENT=configs/paper/hotpotqa_structured_sieve.yaml,MODEL=Alibaba-NLP/Tongyi-DeepResearch-30B-A3B \
    scripts/slurm/serve_and_run.sbatch
sbatch --account=YOUR_ACCOUNT --partition=h24gpu --gres=gpu:1 \
    --export=ALL,DATASET=hotpotqa_structured scripts/slurm/build_indexes.sbatch
```

What each one does:

- `smoke_suite.sbatch` runs the full test suite plus every strategy over the fixtures. No GPU
  needed. The Lucene tests require Java 21 or newer, so pass `JAVA_HOME_OVERRIDE=/path/to/jdk`
  through `--export` if the node's default JDK is older.
- `build_indexes.sbatch` builds the persistent indexes one dataset needs: the dense embedding
  cache (GPU), the Lucene structured index and the Pyserini BM25 index (CPU, Java 21+). `DATASET`
  is required. `DENSE_MODEL` defaults to `BAAI/bge-base-en-v1.5`, and `WHICH` defaults to
  `dense,lucene,pyserini`; set it to build a subset.
- `serve_and_run.sbatch` starts vLLM inside the job, runs one experiment file against it, and
  stops the server on the way out. `EXPERIMENT` is required. Add `MODEL` to serve an open-weight
  model, or leave it unset to use whatever `model.name` the experiment file already names (an API
  model needs no server at all). `TP`, `PORT`, `MAX_MODEL_LEN` and `OVERRIDES` are optional.
- `train_retriever.sbatch` trains a dense retriever from trajectory triples. `TRAIN` is required
  and points at a training YAML. Training lives in its own environment because FlagEmbedding pins
  an older transformers than the eval env; pass it as `TRAIN_ENV`.

Every launcher sources `_common.sh` first, which cds to the repo root, activates `envs/`, sets
`HF_HUB_OFFLINE=1` and `HF_DATASETS_OFFLINE=1`, and creates `slurm_logs/`. `serve_and_run.sbatch`
and `iter_sample.sbatch` start vLLM with `VLLM_PYTHON` (the interpreter that has vLLM installed)
when it differs from the active env.

Compute nodes have no internet. Models, tokenizers and data must be on the shared filesystem before
the job starts. Logs land in `slurm_logs/`. The run record resumes, so a job that hits its time
limit is restarted by resubmitting the same command.

## ITER smoke pipeline

Three stages, each its own job (see docs/TRAINING.md section 5):

```bash
# 8 InfoSeek questions, dedup_dense over the wiki corpus, Tongyi served on the node (200 GB RAM: the HNSW index is read into memory)
sbatch --account=ACCT --gres=gpu:1 --mem=200g --cpus-per-task=16 --time=03:00:00 \
    --export=ALL,EXPERIMENT=configs/iter/smoke_infoseek_train_tongyi.yaml,MODEL=Alibaba-NLP/Tongyi-DeepResearch-30B-A3B,TP=1,MAX_MODEL_LEN=98304,VLLM_PYTHON=/path/to/vllm-env/bin/python,OVERRIDES="evaluation.workers=2" \
    scripts/slurm/serve_and_run.sbatch
# the same with an API backbone (only where compute nodes have network egress):
sbatch --account=ACCT --export=ALL scripts/slurm/iter_smoke_traj.sbatch
sbatch --account=ACCT --export=ALL,RUNS=runs/iter_smoke/agent/infoseek_train/<model>/agent_research_dedup_dense scripts/slurm/iter_smoke_train.sbatch
sbatch --account=ACCT --export=ALL scripts/slurm/iter_smoke_eval.sbatch     # trained vs base on the triples (1 GPU)
```

`iter_smoke_train.sbatch` uses `envs-train` (a venv over the main env with FlagEmbedding 1.3.5 and
`skimsearchagent-train-retriever patch` applied); `TRAIN_PYTHON=` points it elsewhere.

## ITER paper settings on samples

`iter_sample.sbatch` indexes the sample datasets with a retriever, serves the backbone and runs
the sample experiment files (see docs/ITER.md, "Sample a paper setting first"):

```bash
sbatch --account=ACCT --qos=express --export=ALL,VLLM_PYTHON=/path/to/vllm-env/bin/python scripts/slurm/iter_sample.sbatch
```
