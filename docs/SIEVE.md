# Sieve — the Boolean-filtered search–inspect–fetch setting

Sieve is one interface setting in the SkimSearchAgent harness — the one proposed in
*"Search, Inspect, Fetch: Revisiting Boolean Retrieval for Deep-Research Agents."* It composes
four separable stages on top of the shared agent loop:

<p align="center">
  <img src="assets/architecture.png" width="85%" alt="Sieve architecture: filter, rank, inspect, fetch"/>
</p>

1. **Filter.** The agent writes a BQL query — field-scoped terms (`title`, `section`, `date`,
   `infobox`, `body`), `AND/OR/NOT`, quoted phrases, prefixes — which compiles to a Lucene
   filter and *selects* the candidate set. If the filter admits nothing, a designed
   **soft-fallback** relaxes to ranked term retrieval instead of returning an empty page
   (`BQL_SOFT_FALLBACK=1`, the paper default).
2. **Rank.** An interchangeable ranker orders only the admitted candidates: BM25, dense, or
   reciprocal-rank fusion of both.
3. **Inspect.** Each result is a structure-rich card — title, section headings, matched
   fields, and a 25-token query-focused snippet — so the agent chooses what to read before
   spending any reading budget.
4. **Fetch.** The agent reads one named section, not the whole document.

## Running Sieve

The three ranker variants are ordinary conditions of the harness:

```bash
# Boolean-filtered BM25
python -m evaluation.run_eval --dataset browsecomp_plus_structured_full \
  --retriever agent_research_snip --runs-dir runs/demo

# Boolean-filtered Dense
python -m evaluation.run_eval --dataset browsecomp_plus_structured_full \
  --retriever agent_research_bql_donly_snip --runs-dir runs/demo

# Boolean-filtered BM25+Dense (the paper default)
python -m evaluation.run_eval --dataset browsecomp_plus_structured_full \
  --retriever agent_research_bql_dense_snip --runs-dir runs/demo
```

## Ablation knobs

| ablation | how | paper finding |
|---|---|---|
| Strict Boolean (no fallback) | `BQL_SOFT_FALLBACK=0` | accuracy drops well below the Search–Visit baseline — the fallback is load-bearing |
| Without snippets | condition `agent_research_bql_dense_fetch` | 2.9–6.8 accuracy points lost, matched everything else |
| Dense encoder sweep | `DENSE_MODEL=<hf-id>` (bge-small/base/large, Qwen3-Embedding 0.6B/4B/8B) | token saving stable across 33M–8B; accuracy varies within ~3 points |
| Ranker swap | the three conditions above | fusion is best on BrowseComp-Plus; dense best on HotpotQA |
| Engine swap | `agent_research_indri_snip` | Indri-QL executor comparison |

## Results

<p align="center">
  <img src="assets/pareto.png" width="60%" alt="Accuracy vs tokens for every method on BrowseComp-Plus-Structured"/>
</p>
<p align="center"><em>Every method on full-corpus BrowseComp-Plus (n=830): Sieve sits on the
accuracy–cost Pareto front.</em></p>

| | |
|---|---|
| <img src="assets/gainloss.png" alt="per-query gains and token deltas"/> | <img src="assets/retriever_sensitivity.png" alt="dense-encoder sweep 33M-8B"/> |
| *Per-query wins/losses and token change vs. Search–Visit.* | *Dense-encoder sweep (bge + Qwen3, 33M–8B): the saving is not an encoder artifact.* |

## Headline results

Against the BM25 Search–Visit baseline under identical budgets (k=5, 12,000-token reads,
100 steps):

| collection | accuracy (baseline → Sieve) | tokens/episode |
|---|---|---|
| HotpotQA (7,343) | 43.7 → 45.3 EM | 19.9k → 13.9k (−30.4%) |
| MuSiQue (2,409) | 26.1 → 29.2 EM | 43.0k → 21.3k (−50.6%) |
| BrowseComp-Plus full (830) | 34.7 → 37.2 judge | 68.1k → 46.0k (−32.4%) |

The effect transfers across agent backbones (Qwen-AgentWorld, OpenResearcher) with the largest
token savings where the baseline uses the most context. Full tables, ablations, and statistics:
the paper, and [`REPRODUCING.md`](REPRODUCING.md) to regenerate them.
