"""The multi-hop release -> BEIR converter (scripts/stage_multihop.py) and its
round-trip through the general-domain dataset loader. Covers both the raw release schema
and HuggingFace's dict-of-parallel-lists schema."""
import importlib.util
import os
import tempfile

from evaluation.datasets import _load_beir_style, available_datasets

_SM = os.path.join(os.path.dirname(__file__), "..", "scripts", "stage_multihop.py")
_spec = importlib.util.spec_from_file_location("stage_multihop", _SM)
sm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sm)


def test_hotpotqa_raw_schema_dedups_corpus_and_maps_gold():
    examples = [
        {"_id": "q1", "question": "Q1?", "answer": "A1",
         "context": [["Alpha", ["a1.", "a2."]], ["Beta", ["b1."]], ["Distract", ["d."]]],
         "supporting_facts": [["Alpha", 0], ["Beta", 0]]},
        {"_id": "q2", "question": "Q2?", "answer": "A2",
         "context": [["Alpha", ["a1.", "a2."]], ["Gamma", ["g."]]],   # Alpha repeats
         "supporting_facts": [["Gamma", 0]]},
    ]
    corpus, queries, qrels = sm.build_beir(examples)
    assert {d["title"] for d in corpus.values()} == {"Alpha", "Beta", "Distract", "Gamma"}
    assert len(corpus) == 4                                    # Alpha deduped across queries
    by_title = {d["title"]: i for i, d in corpus.items()}
    assert qrels == {("q1", by_title["Alpha"]), ("q1", by_title["Beta"]),
                     ("q2", by_title["Gamma"])}
    assert queries["q1"]["answer"] == "A1" and queries["q1"]["text"] == "Q1?"


def test_hotpotqa_huggingface_dict_schema():
    """HF represents context/supporting_facts as dict-of-parallel-lists, not list-of-lists."""
    examples = [{
        "id": "h1", "question": "HQ?", "answer": "HA",
        "context": {"title": ["Alpha", "Beta"], "sentences": [["a."], ["b."]]},
        "supporting_facts": {"title": ["Beta"], "sent_id": [0]},
    }]
    corpus, queries, qrels = sm.build_beir(examples)
    by_title = {d["title"]: i for i, d in corpus.items()}
    assert set(by_title) == {"Alpha", "Beta"}
    assert qrels == {("h1", by_title["Beta"])}                # only the supporting title
    assert queries["h1"]["answer"] == "HA"


def test_musique_uses_is_supporting_for_gold():
    examples = [{
        "id": "m1", "question": "MQ?", "answer": "MA",
        "paragraphs": [
            {"idx": 0, "title": "P1", "paragraph_text": "t1", "is_supporting": True},
            {"idx": 1, "title": "P2", "paragraph_text": "t2", "is_supporting": False},
            {"idx": 2, "title": "P3", "paragraph_text": "t3", "is_supporting": True},
        ]}]
    corpus, queries, qrels = sm.build_beir(examples)
    assert len(corpus) == 3
    by_title = {d["title"]: i for i, d in corpus.items()}
    assert qrels == {("m1", by_title["P1"]), ("m1", by_title["P3"])}


def test_round_trip_loads_as_shared_corpus_instances():
    examples = [
        {"_id": "q1", "question": "Which?", "answer": "Treaty",
         "context": [["Alpha", ["a."]], ["Beta", ["b."]]],
         "supporting_facts": [["Alpha", 0]]},
    ]
    corpus, queries, qrels = sm.build_beir(examples)
    with tempfile.TemporaryDirectory() as d:
        sm.write_beir(d, corpus, queries, qrels)
        assert os.path.exists(os.path.join(d, "corpus.jsonl"))
        assert os.path.exists(os.path.join(d, "qrels", "test.tsv"))
        insts = _load_beir_style(d, "hotpotqa")
    assert len(insts) == 1
    inst = insts[0]
    assert inst.instance_id == "hotpotqa__q1"
    assert inst.answer == "Treaty"
    alpha_id = next(i for i, doc in corpus.items() if doc["title"] == "Alpha")
    assert inst.gold_doc_ids == {alpha_id}                    # gold = the supporting para
    assert inst.docs and len(inst.docs) == 2                  # the shared corpus rides along


def test_multihop_datasets_are_registered_general_domain():
    from evaluation.datasets import dataset_domain
    for name in ("hotpotqa", "2wiki", "musique"):
        assert name in available_datasets()
        assert dataset_domain(name) == "general"
