# Aggregate-dollar cost accounting -- full method vs. BM25 baseline (M6)

**Assumed price** (parameterizable, NOT a verified invoice -- see script docstring): **$0.20 / 1M input tokens, $0.80 / 1M output tokens**. Two accounting bases per cell: **count-once** (each real token the model ever saw, counted exactly once -- what the paper's headline 30-51% token-savings number uses) and **step-summed** (the raw cumulative sum of every LLM call's input+output tokens across the episode -- what an UNCACHED backend actually bills, and the basis sensitive to the method's LLM-call count).

| Dataset | Cell | n | LLM calls/ep | $/query (count-once) | $/query (step-summed) | $ over full eval set (count-once) | $ over full eval set (step-summed) |
|---|---|---:|---:|---:|---:|---:|---:|
| browsecomp_plus_structured | baseline (bm25) | 830 | 62.5 | $0.0209 | $0.4772 | $17.34 | $396.05 |
| browsecomp_plus_structured | full method (bql+dense+snip) | 830 | 60.7 | $0.0186 | $0.3151 | $15.41 | $261.50 |
| browsecomp_plus_structured | **method vs baseline, %Δ** | | **-2.8%** | **-11.2%** | **-34.0%** | | |
| hotpotqa_structured | baseline (bm25) | 7343 | 15.9 | $0.0052 | $0.0602 | $38.51 | $442.33 |
| hotpotqa_structured | full method (bql+dense+snip) | 7343 | 20.9 | $0.0043 | $0.0463 | $31.70 | $339.99 |
| hotpotqa_structured | **method vs baseline, %Δ** | | **+31.6%** | **-17.7%** | **-23.1%** | | |
| musique_structured | baseline (bm25) | 2409 | 29.6 | $0.0111 | $0.1759 | $26.79 | $423.71 |
| musique_structured | full method (bql+dense+snip) | 2409 | 32.6 | $0.0066 | $0.0879 | $15.97 | $211.78 |
| musique_structured | **method vs baseline, %Δ** | | **+9.9%** | **-40.4%** | **-50.0%** | | |

**Key sentences (paper-ready, hedged).**

- On browsecomp_plus_structured, at $0.20/$0.80 per 1M in/out tokens, the full method uses -2.8% LLM calls versus the BM25 baseline and costs -11.2% under count-once dollar accounting and -34.0% under step-summed (call-count-sensitive) dollar accounting -- the method **remains cheaper** under the accounting basis that is sensitive to call count.
- On hotpotqa_structured, at $0.20/$0.80 per 1M in/out tokens, the full method uses +31.6% LLM calls versus the BM25 baseline and costs -17.7% under count-once dollar accounting and -23.1% under step-summed (call-count-sensitive) dollar accounting -- the method **remains cheaper** under the accounting basis that is sensitive to call count.
- On musique_structured, at $0.20/$0.80 per 1M in/out tokens, the full method uses +9.9% LLM calls versus the BM25 baseline and costs -40.4% under count-once dollar accounting and -50.0% under step-summed (call-count-sensitive) dollar accounting -- the method **remains cheaper** under the accounting basis that is sensitive to call count.

**Overall.** Even under the step-summed, call-count-sensitive accounting -- the basis the paper does not currently report, and the one most analogous to the standard the paper itself applies to DCI's cost claim -- the full method remains cheaper than the BM25 baseline on all three datasets at the assumed price above, DESPITE making more LLM calls on two of the three (HotpotQA, MuSiQue): the per-call context the method resends is enough shorter (bounded section-level fetches vs. whole-document visits) that extra calls do not flip the aggregate-dollar verdict. This should be stated explicitly, with the assumed rate disclosed, alongside the existing count-once headline number.

**Why the $ delta is smaller than the raw-token-count delta (count-once basis).** `analysis/paper_analyses.py`'s Analysis 3 reports the method's count-once TOKEN savings as -29.6%/-30.4%/-50.9% on browsecomp/hotpotqa/musique respectively (input+output tokens combined, unweighted) -- noticeably larger in magnitude than this script's count-once DOLLAR deltas above (-11.2%/-17.7%/-40.4%). The reason: at a 4x output:input price ratio, the method's INPUT tokens drop sharply (structured fetch reads far less text than whole-document visit) but its OUTPUT tokens are somewhat HIGHER than the baseline's on every dataset (more calls / more structured-query reformulation text generated) -- a token-mix shift toward the more expensive side that a plain (unweighted) token-count delta cannot see, but a dollar accounting does. This is exactly the kind of basis-sensitivity the paper already applies to DCI's cost claim (`related_work.tex` Sec 4); it should be applied symmetrically to the paper's own count-once headline number too.

**Sensitivity note.** These dollar figures move linearly with the assumed price ratio; the SIGN of the method-vs-baseline delta on each basis does not depend on the absolute rate (both cells are priced identically), only on the input:output price RATIO, which is held fixed at the CLI-supplied rate for both cells here. Re-run with `--price-in-per-1m`/`--price-out-per-1m` set to any other public list price to confirm the qualitative verdict is rate-independent.
