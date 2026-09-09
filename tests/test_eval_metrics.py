import math

from agent_search.evaluation.metrics import recall_at_k, acc_at_k, mrr_at_k, ndcg_at_k


def test_recall_counts_gold_in_topk():
    assert recall_at_k(["a", "b", "c", "d"], {"b", "d"}, 4) == 1.0
    assert recall_at_k(["a", "b", "c", "d"], {"b", "d"}, 2) == 0.5  # only b in top-2


def test_recall_zero_when_none_found():
    assert recall_at_k(["a", "b"], {"x"}, 2) == 0.0


def test_acc_is_all_or_nothing():
    assert acc_at_k(["a", "b", "c"], {"a", "b"}, 3) == 1.0
    assert acc_at_k(["a", "b", "c"], {"a", "z"}, 3) == 0.0   # z missing entirely
    # LocAgent semantics: success needs min(|gold|, k) gold in the top-k, so a
    # multi-gold instance can still score Acc@1 when the entire top-1 is gold
    assert acc_at_k(["a", "b", "c"], {"a", "b"}, 1) == 1.0
    assert acc_at_k(["z", "a", "b"], {"a", "b"}, 1) == 0.0   # top-1 not gold


def test_mrr_uses_first_gold_rank():
    assert mrr_at_k(["a", "b", "c"], {"b"}, 3) == 0.5        # gold at rank 2
    assert mrr_at_k(["a", "b", "c"], {"a", "c"}, 3) == 1.0   # first gold at rank 1
    assert mrr_at_k(["a", "b", "c"], {"z"}, 3) == 0.0
    assert mrr_at_k(["a", "b", "c"], {"c"}, 2) == 0.0        # gold outside top-2


def test_ndcg_rewards_higher_rank():
    assert ndcg_at_k(["g", "x", "y"], {"g"}, 3) == 1.0                 # rank 1
    v = ndcg_at_k(["x", "g", "y"], {"g"}, 3)
    assert abs(v - 1 / math.log2(3)) < 1e-9                            # rank 2


def test_hit_precision_f1_map():
    from agent_search.evaluation.metrics import (hit_at_k, precision_at_k, f1_at_k,
                                        average_precision_at_k)
    r = ["a", "b", "c", "d"]
    # hit: any gold in top-k
    assert hit_at_k(r, {"c"}, 3) == 1.0 and hit_at_k(r, {"c"}, 2) == 0.0
    assert hit_at_k(r, set(), 3) == 0.0
    # precision@k = |gold∩topk| / k
    assert precision_at_k(r, {"a", "b"}, 4) == 0.5
    assert precision_at_k(r, {"a"}, 1) == 1.0
    # f1 = harmonic mean of p@k and r@k
    f = f1_at_k(r, {"a", "b"}, 2)               # p=1.0, r=1.0 -> 1.0
    assert abs(f - 1.0) < 1e-9
    # average precision: gold at ranks 1 and 3 -> (1/1 + 2/3)/2
    ap = average_precision_at_k(r, {"a", "c"}, 4)
    assert abs(ap - ((1 / 1 + 2 / 3) / 2)) < 1e-9
    # perfect ranking scores 1.0; normalized by min(|gold|,k)
    assert abs(average_precision_at_k(r, {"a", "b"}, 4) - 1.0) < 1e-9


def test_cutoff_free_set_metrics():
    from agent_search.evaluation.metrics import set_recall, set_precision, set_f1
    # a tight, surgical set: 2 returned, 1 gold among them
    tight = ["a.py::foo", "a.py::bar"]
    gold = {"a.py::foo"}
    assert set_recall(tight, gold) == 1.0
    assert set_precision(tight, gold) == 0.5          # 1 of 2 -> surgical
    # a broad grep-like set: 100 returned, same 1 gold
    broad = [f"x{i}" for i in range(99)] + ["a.py::foo"]
    assert set_recall(broad, gold) == 1.0             # still covers gold
    assert set_precision(broad, gold) == 1 / 100      # but far less pure
    assert set_f1(tight, gold) > set_f1(broad, gold)  # surgicality shows here
    assert set_recall([], gold) == 0.0 and set_precision([], gold) == 0.0
