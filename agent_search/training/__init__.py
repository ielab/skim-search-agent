"""Training from the run record.

Every episode the harness runs is a full trajectory, so the run record also serves as training
data. This package turns it into retriever training data and back into a deployed retriever,
following ITER (history-conditioned queries, tiered negatives, a patched FlagEmbedding trainer):

* `queries` renders the retriever query from the agent's history (identical at train and serve),
* `triples` extracts positives and tiered negatives from `rows.jsonl`,
* `retriever` builds and launches the training command and writes the serving note,
* `history` feeds the live episode's context to the dense retriever at inference.

See docs/TRAINING.md for the walk-through.
"""
from agent_search.training.queries import INSTRUCTIONS, STYLES, render_query
from agent_search.training.triples import build_triples, oracle_labeller

__all__ = ["INSTRUCTIONS", "STYLES", "render_query", "build_triples", "oracle_labeller"]
