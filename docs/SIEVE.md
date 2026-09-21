# Sieve: the Boolean-filtered search, inspect, fetch setting

Sieve is one interface setting in the SkimSearchAgent framework, the one proposed in *"Search,
Inspect, Fetch: Exploiting Boolean Retrieval for Deep-Research Agents."* It adds four separable
stages on top of the shared agent loop.

<p align="center">
  <img src="assets/architecture.png" width="85%" alt="Sieve architecture: filter, rank, inspect, fetch"/>
</p>

## Run it end to end

Four commands take you from a fresh clone to a Sieve number on one collection.
[REPRODUCING.md](REPRODUCING.md) has the full grid: install details, the other datasets, the
sharded cluster path and the judge.

**1. Install and stage the data.** Needs JDK 21 on `JAVA_HOME` (Pyserini and Lucene).

```bash
python -m pip install -e ".[retrieval,api,eval,serve]"
python -m agent_search.tokens --seed
huggingface-cli download wshuai190/hotpotqa-structured --repo-type dataset --local-dir data/_hf/hotpotqa
cp -r data/_hf/hotpotqa/structured data/hotpotqa_structured
```

**2. Build the two indexes Sieve reads.** The Lucene structured index selects, the dense cache
ranks.

```bash
skimsearchagent-build-indexes --dataset hotpotqa_structured --retriever search_lucene --index-root indexes
skimsearchagent-build-indexes --dataset hotpotqa_structured --retriever dense --model BAAI/bge-base-en-v1.5 --index-root indexes
```

`sieve_bm25` needs only the first one. See [REPRODUCING.md](REPRODUCING.md) step 3 for the full
table of which strategy reads which index.

**3. Serve the backbone.**

```bash
export VLLM_USE_FLASHINFER_MOE_FP16=0
vllm serve Alibaba-NLP/Tongyi-DeepResearch-30B-A3B \
  --tensor-parallel-size 1 --port 8000 --gpu-memory-utilization 0.9 \
  --max-model-len 131072 --compilation-config '{"cudagraph_mode":"PIECEWISE"}'
```

**4. Run the cell.** Validate first; it reports anything missing before GPU time is spent.

```bash
skimsearchagent validate configs/paper/hotpotqa_structured_sieve.yaml
skimsearchagent run configs/paper/hotpotqa_structured_sieve.yaml \
  model.name=Alibaba-NLP/Tongyi-DeepResearch-30B-A3B \
  model.api_base=http://127.0.0.1:8000/v1 \
  output.runs_dir=runs/mine
```

HotpotQA and MuSiQue are scored by exact match during the run, so `results.json` already has
`answer_em`. BrowseComp-Plus needs one more step:

```bash
export OPENAI_API_KEY=...
skimsearchagent-judge --results-dir runs/mine/agent/<dataset>/<model>/<condition> \
  --judge-model gpt-4o-mini --workers 16
```

### The Sieve experiment files

| file under `configs/paper/` | collection | ranker |
|---|---|---|
| `browsecomp_plus_structured_full_sieve_tongyi.yaml` | BrowseComp-Plus, 100,195 docs | fused |
| `hotpotqa_structured_sieve.yaml` | HotpotQA | fused |
| `hotpotqa_structured_sieve_bm25.yaml` | HotpotQA | BM25 |
| `musique_structured_sieve.yaml` | MuSiQue | fused |
| `browsecomp_plus_structured_sieve_agentworld.yaml`, `..._openresearcher.yaml` | BrowseComp-Plus pooled | fused, transfer backbones |

### The three rankers as flags

```bash
# Boolean-filtered BM25
skimsearchagent-eval --dataset browsecomp_plus_structured_full --retriever agent_research_snip --runs-dir runs/demo
# Boolean-filtered dense
skimsearchagent-eval --dataset browsecomp_plus_structured_full --retriever agent_research_bql_donly_snip --runs-dir runs/demo
# Boolean-filtered BM25 + dense, the paper default
skimsearchagent-eval --dataset browsecomp_plus_structured_full --retriever agent_research_bql_dense_snip --runs-dir runs/demo
```

Those three also have aliases on the `skimsearchagent` launcher: `sieve_bm25`, `sieve_dense` and
`sieve`. The paper's condition names carry the paper's prompt; the aliases run the same strategies
under the library's default prompt, so a reproduction names the condition.

## How it works

1. **Filter.** The agent writes a BQL query: field-scoped terms (`title`, `section`, `date`,
   `infobox`, `body`), `AND`/`OR`/`NOT`, quoted phrases, prefixes. That compiles to a Lucene filter
   and *selects* the candidate set. When the filter admits nothing, a **soft fallback** relaxes to
   ranked term retrieval instead of returning an empty page (`BQL_SOFT_FALLBACK=1`, the paper
   default).
2. **Rank.** An interchangeable ranker orders only the candidates that got through: BM25, dense, or
   reciprocal-rank fusion of both.
3. **Inspect.** Each result comes back as a structure-rich card with the title, section headings,
   which fields matched, and a query-focused snippet (`SNIPPET_TOKENS`, 32 tokens by default, with
   no character cap). The agent selects what to read before spending any reading budget.
4. **Fetch.** The agent reads one named section, not the whole document.

The fetch call names a rank and a section, `{"rank": 1, "section": "Career"}`, one section per
call. The manual the agent reads is the reference manual: the query language, the fields, the
fetch call and worked examples, about 680 words. It lives at
`agent_search/tools/search_bql/bql_browsecomp.md` for BrowseComp-Plus and `bql_doc.md` for the
wiki collections. The manual ablation below found it the best of eight variants; the paper's
longer manual, which added search, hop and mistake advice, is the variant
`agent_research_bql_dense_snip_reference_howto_hops_mistakes`. REPRODUCING.md, "Two departures
from the paper's prompts", explains both changes.

## Ablation knobs

| ablation | how | what this library measured |
|---|---|---|
| Strict Boolean (no fallback) | `BQL_SOFT_FALLBACK=0` | 44.2 against 48.8 on BrowseComp-Plus, and steps rise from 58 to 62, so the fallback is load-bearing |
| Without snippets | condition `agent_research_bql_dense_fetch` | 43.4 against 48.8 on BrowseComp-Plus, the largest single component |
| Manual ablation | conditions `agent_research_bql_dense_snip_card` (a hundred-word card), `_nomanual` (a 21-word stub), `_reference_howto`, `_reference_hops`, `_reference_mistakes` (the reference plus one advice section), `_reference_howto_hops_mistakes` (the full manual) and `_reference_howto_hops_mistakes_noconstruct`; composed by `scripts/compose_manuals.py`; paper prompt, everything else matched | not in the paper; on BrowseComp-Plus every cut of the full manual scores above it and the reference alone scores highest, so the reference is the default |
| Snippet width sweep | `snippet_tokens=32 / 64 / 128 / 256 / 512` on the `skimsearchagent` launcher, or `SNIPPET_TOKENS=` in the environment. Default 32. It governs BOTH arms' listing snippets, Sieve's query-biased window, and the visit baselines' opening window. A sweep moves the whole comparison, not just one side of it | not yet run; listing size grows roughly linearly with the window |
| Dense encoder sweep | `DENSE_MODEL=<hf-id>` (bge-small/base/large, Qwen3-Embedding 0.6B/4B/8B) | 33M to 8B moves accuracy by 3.5 points with no trend by size, and tokens not at all. Sweep dense-only Sieve, not the fused ranker, or the fusion hides the encoder |
| Ranker swap | the three conditions above | fusion wins on BrowseComp-Plus by 2.3 points over dense; on MuSiQue and HotpotQA dense alone ties or wins |
| Engine swap | `agent_research_indri_snip` | Indri-QL executor comparison |

## Results

<p align="center">
  <img src="assets/pareto.png" width="60%" alt="Accuracy vs tokens for every method on BrowseComp-Plus-Structured"/>
</p>
<p align="center"><em>Every method on full-corpus BrowseComp-Plus (n=830). Sieve sits on the
accuracy/cost Pareto front.</em></p>

| | |
|---|---|
| <img src="assets/gainloss.png" alt="per-query gains and token deltas"/> | <img src="assets/retriever_sensitivity.png" alt="dense-encoder sweep 33M-8B"/> |
| *Per-query wins and losses, plus the token change against Search-Visit.* | *Dense-encoder sweep (bge and Qwen3, 33M to 8B): the saving isn't an encoder artifact.* |

## Headline results

Every number here comes from this library on full collections, with the same budget on both arms:
5 results per search, 12,000-token reads, 32-token listing snippets, 100 steps, seed 42. The
backbone is Tongyi-DeepResearch-30B-A3B. BrowseComp-Plus is judged by gpt-4o-mini, the wiki
collections by exact match. The Sieve column is the best of the three rankers, named in brackets.

| collection | BM25 Search-Visit | best Search-Fetch | Sieve | tokens, Search-Visit to Sieve |
|---|---|---|---|---|
| BrowseComp-Plus (830) | 36.4 | 45.1 | **48.8** (fused) | 57.3k to 45.6k |
| MuSiQue (2,409) | 25.0 | 26.7 | **28.3** (dense) | 42.5k to 31.1k |
| HotpotQA (7,343) | 43.9 | 43.8 | **44.6** (dense) | 20.6k to 15.7k |

Sieve wins all three, and it costs less context everywhere. The margin isn't the same size in each
column. BrowseComp-Plus is where the Boolean filter earns its keep, because the questions name
entities and dates that make good filter terms. HotpotQA is the hard case for the claim: everything
lands within a point, so the accuracy result there rests on the token saving, not the score.

The effect carries across agent backbones. With Qwen-AgentWorld, Sieve is 40.4 against 24.8 for BM25
Search-Visit on BrowseComp-Plus, and 29.8 against 28.4 on MuSiQue. With OpenResearcher it's 27.5
against 20.7, and 23.8 against 21.0. The weaker the backbone is at writing searches, the more the
structured listing helps it.

The per-component ablations, the manual ablation and the encoder sweep are the "Ablation knobs"
table above. To regenerate any of it, see [REPRODUCING.md](REPRODUCING.md).

### Where the published runs live

One cell per leaf, under `runs/sieve/<dataset>/<strategy>/<agent>/<retriever>/`. Each leaf holds
`rows.jsonl` (every question's trajectory and scores) and either `judge_summary.json`
(BrowseComp-Plus, gpt-4o-mini) or the exact-match scores in the rows themselves (MuSiQue,
HotpotQA). Strategy names are the library's: `sieve`, `search_fetch`, `search_visit`, `autoread`,
`dci`, `bounded_dci`, `rag`, plus the Sieve variants `sieve_bm25`, `sieve_dense`, `sieve_nosnip`,
`sieve_fill`, `sieve_strict` and the manual ablation `sieve_card`, `sieve_nomanual`,
`sieve_reference_*`. The retriever segment names the ranking model: `bm25`, `bge-base`,
`bm25+bge-base`, and the encoder sweep `bm25+bge-small` and so on.

A run you launch yourself lands in the harness layout instead,
`<runs_dir>/agent/<dataset>/<model>/<condition>/`. See [REPRODUCING.md](REPRODUCING.md) step 9.

## Ranking invariant: Boolean filters, one model ranks

In Sieve the Boolean query (BQL) only **selects** candidates. The **order** comes from one ranking
model: corpus BM25 for `sieve_bm25`, reciprocal-rank fusion of BM25 and dense similarity for
`sieve`, and dense similarity alone for `sieve_dense`. Two rules are enforced in code so an
experiment cannot mix rankers:

1. **The 0-hit fallback ranks with the same model over the same index as the exact path.**
   When a Boolean query matches nothing, `soft_topk` builds a pool of the lexically closest
   documents. If a dense belief is attached, it adds the dense side's nearest neighbours to that
   pool. The arm's own fusion rule then orders the pool (`StructuralExecutor._fuse_soft`,
   `DenseOnlyStructuralExecutor._fuse_soft`). The coverage-tier fallback for a many-clause AND
   fuses within tiers by the same rule. Pool size: `BQL_SOFT_POOL` (default 100).
2. **Dense similarity always comes from the persisted embedding cache** built by
   `skimsearchagent-build-indexes --retriever dense --model <dense_model>`. The run refuses
   to start a dense arm without that cache. It never encodes documents online: only the query
   gets encoded, once per search call. The model is the run's `dense_model` (`--dense-model` on
   the CLI, or `DENSE_MODEL` in the environment; default `BAAI/bge-base-en-v1.5`). Every dense arm
   uses that same model, so Sieve's ranker, its fallback, and the dense and hybrid baselines all
   read the same index for the same model.

`tests/test_sieve_fallback_consistency.py` pins both rules.

## Authors

<p align="center">
  <img src="assets/team.png" width="100%" alt="Shuai Wang, Haodong Chen, Yu Yin, Shengyao Zhuang, Bevan Koopman and Guido Zuccon"/>
</p>

<div align="center">

[Shuai Wang](https://shuaiwang.io)<sup>1</sup> ·
[Haodong Chen](https://donovan0243.github.io/)<sup>1</sup> ·
[Yu Yin](https://yinyubb.github.io/)<sup>1</sup> ·
[Shengyao Zhuang](https://arvinzhuang.github.io/) ·
[Bevan Koopman](https://bevankoopman.github.io/)<sup>2,1</sup> ·
[Guido Zuccon](https://ielab.io/people/guido-zuccon.html)<sup>1</sup>

<sup>1</sup>[ielab](https://ielab.io), The University of Queensland ·
<sup>2</sup>Australian e-Health Research Centre, CSIRO

</div>
