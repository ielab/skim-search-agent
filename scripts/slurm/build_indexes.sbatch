#!/usr/bin/env bash
#SBATCH --job-name=ssa-index
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128g
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
# Build the persistent indexes one dataset needs: the dense embedding cache (GPU), the Lucene
# structured index and the Pyserini BM25 index (CPU, Java 21+). --export=ALL,DATASET=...,
# DENSE_MODEL=... (default BAAI/bge-base-en-v1.5), WHICH=dense,lucene,pyserini (default all).
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/slurm/_common.sh"
DATASET="${DATASET:?set DATASET=<registered dataset> via --export}"
DENSE_MODEL="${DENSE_MODEL:-BAAI/bge-base-en-v1.5}"
WHICH="${WHICH:-dense,lucene,pyserini}"
INDEX_ROOT="${INDEX_ROOT:-indexes}"
case ",$WHICH," in *,dense,*)    skimsearchagent-build-indexes --dataset "$DATASET" --retriever dense --model "$DENSE_MODEL" --index-root "$INDEX_ROOT";; esac
case ",$WHICH," in *,lucene,*)   python -m agent_search.retrievers.structural.lucene.index_builder --dataset "$DATASET" --index-root "$INDEX_ROOT";; esac
case ",$WHICH," in *,pyserini,*) skimsearchagent-build-indexes --dataset "$DATASET" --retriever bm25_pyserini --index-root "$INDEX_ROOT";; esac
echo ">> indexes built for $DATASET under $INDEX_ROOT"
