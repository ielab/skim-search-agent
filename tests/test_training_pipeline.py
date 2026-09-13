"""agent_search.training: run record -> triples -> trainer command -> checkpoint plugged back in."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from agent_search.agent.loop import Step
from agent_search.training import history as H
from agent_search.training import queries as Q
from agent_search.training import retriever as R
from agent_search.training import triples as T
from lucene_support import require_jvm

REPO = Path(__file__).resolve().parent.parent
CORPUS = {"d1": "Treaty of Guadalupe Hidalgo\nEnded the Mexican-American War in 1848.",
          "d2": "Treaty of Paris (1898)\nEnded the Spanish-American War.",
          "d3": "Adams-Onis Treaty\nCeded Florida in 1819.",
          "d4": "Unrelated\nNothing here."}


def _row():
    """A synthetic episode: search 1 lists d2,d3 and reads d3 (irrelevant); search 2 lists
    d1,d2,d4, reads d1 (gold) then re-reads d3; the answer follows."""
    return {
        "instance_id": "x__1", "question": "Which treaty ended the Mexican-American War?",
        "gold_ids": ["d1"], "gold_answer": "Treaty of Guadalupe Hidalgo", "final_answer": "Treaty of Guadalupe Hidalgo",
        "trajectory": [
            {"action": "search_s", "args": {"query": "treaty Florida"}, "query": "treaty Florida",
             "observation": "search: ...\n  1  d2  'Treaty of Paris'\n  2  d3  'Adams-Onis'", "raw_output": "<think>start</think><tool_call>{}</tool_call>",
             "hit_ids": ["d2", "d3"], "read_ids": []},
            {"action": "fetch_s", "args": {"specs": [[2, "(intro)"]]}, "query": "", "observation": "fetch: d3 ...",
             "raw_output": "<tool_call>{}</tool_call>", "hit_ids": [], "read_ids": ["d3"]},
            {"action": "search_s", "args": {"query": "Mexican-American War treaty"}, "query": "Mexican-American War treaty",
             "observation": "search: ...\n  1  d1  'Guadalupe'\n  2  d2  'Paris'\n  3  d4  'Unrelated'",
             "raw_output": "<think>Florida is not it, the doc was about 1819 and unrelated to the war. Let's search the war itself.</think><tool_call>{}</tool_call>",
             "hit_ids": ["d1", "d2", "d4"], "read_ids": []},
            {"action": "fetch_s", "args": {"specs": [[1, "(intro)"]]}, "query": "", "observation": "fetch: d1 ...",
             "raw_output": "<tool_call>{}</tool_call>", "hit_ids": [], "read_ids": ["d1"]},
            {"action": "fetch_s", "args": {"specs": [[2, "(intro)"]]}, "query": "", "observation": "fetch: d2 ...",
             "raw_output": "<think>The Guadalupe Hidalgo document says the treaty ended the war in 1848, which answers the question directly.</think><tool_call>{}</tool_call>",
             "hit_ids": [], "read_ids": ["d2"]},
            {"action": "answer", "args": {}, "query": "", "observation": "(episode ended)",
             "raw_output": "<think>Paris was a different war.</think><answer>Treaty of Guadalupe Hidalgo</answer>", "hit_ids": [], "read_ids": []},
        ],
    }


def test_query_styles_render_like_iter():
    inter = [{"query": "treaty Florida", "visits": [("d3", CORPUS["d3"], "Florida doc, not the war. Let's search the war.")]}]
    assert Q.render_query("plain", "Q?", "sub", inter) == "sub"
    i2 = Q.render_query("i2", "Q?", "sub", inter)
    assert i2 == "Main Question: Q?\nCurrent Subquery: sub\nPrevious Interactions:\nPrevious SubQuery 1: treaty Florida"
    i3 = Q.render_query("i3", "Q?", "sub", inter)
    assert "Visited Documents: [docs_id:d3] Adams-Onis Treaty Ceded Florida in 1819." in i3
    i4 = Q.render_query("i4", "Q?", "sub", inter)
    assert "Visited Document Notes: [docs_id:d3] Florida doc, not the war." in i4      # planning sentence dropped
    assert Q.render_query("i2", "Q?", "sub", []).endswith("Previous Interactions: <empty>")
    mem = Q.render_query("mem", "Q?", "sub", inter)
    assert mem.startswith("[Q] Q?\n[Now] sub")
    docs = Q.render_query("docs", "Q?", "sub", inter)
    assert "[Prev] treaty Florida -> Adams-Onis Treaty:" in docs
    assert Q.instruction_prefix(Q.INSTRUCTIONS["i2"]).startswith("Instruct: Given the main question")
    with pytest.raises(ValueError):
        Q.render_query("i99", "Q?", "sub", inter)


def test_triples_follow_iters_tier_rules():
    samples = T.build_triples([_row()], CORPUS.get, labeller=T.oracle_labeller, query_style="i2")
    assert len(samples) == 1
    s = samples[0]
    assert s["pos_id"] == ["d1"] and s["pos"] == [CORPUS["d1"]]
    assert s["neg_hard_id"] == ["d3"]          # read before this search, irrelevant
    assert s["neg_diversity_id"] == []         # nothing relevant had been read yet
    assert s["neg_weak_id"] == ["d4"]          # returned by this search, never read in the episode
    assert s["query"].startswith("Main Question: Which treaty ended the Mexican-American War?")
    assert "Previous SubQuery 1: treaty Florida" in s["query"]
    assert s["reasoning_len"] > 0 and s["reweight_rate"] == pytest.approx(1.0)
    assert s["instance_id"] == "x__1" and s["query_style"] == "i2"


def test_triples_parse_listings_and_ranks_for_old_rows():
    row = _row()
    for st in row["trajectory"]:
        st.pop("hit_ids"); st.pop("read_ids")
    assert T.listed_ids(row["trajectory"][2]) == ["d1", "d2", "d4"]
    assert T.read_ids(row["trajectory"][3], ["d1", "d2", "d4"]) == ["d1"]
    samples = T.build_triples([row], CORPUS.get)
    assert samples and samples[0]["pos_id"] == ["d1"] and samples[0]["neg_weak_id"] == ["d4"]


def test_labellers():
    row = _row()
    assert T.oracle_labeller("d1", "", row) and not T.oracle_labeller("d3", "", row)
    ans = T.make_answer_labeller(CORPUS.get)
    assert ans("d1", "", row) and not ans("d2", "", row)
    judge = T.make_llm_judge(lambda m: "<think>hm</think>\nNOT_RELEVANT")
    assert judge("d1", "some analysis", row) is False
    judge = T.make_llm_judge(lambda m: "RELEVANT because it names the treaty")
    assert judge("d1", "some analysis", row) is True


def test_history_context_renders_the_same_query_at_inference():
    ctx = H.QueryContext(question="Which treaty ended the Mexican-American War?", text_of=CORPUS.get, style="i3")
    steps = [Step(name="search_s", args={"query": "treaty Florida"}, observation="", raw_output=""),
             Step(name="fetch_s", args={"specs": [[2, "(intro)"]]}, observation="", raw_output=""),
             Step(name="search_s", args={"query": "Mexican-American War treaty"}, observation="",
                  raw_output="<think>Florida is not it.</think>")]
    ctx.observe(steps[0], ["d2", "d3"])
    ctx.observe(steps[1], ["d2", "d3"])
    ctx.observe(steps[2], ["d1", "d2", "d4"])
    rendered = ctx.render("Mexican-American War treaty")
    assert rendered == Q.render_query("i3", ctx.question, "Mexican-American War treaty",
                                      [{"query": "treaty Florida", "visits": [("d3", CORPUS["d3"], "Florida is not it.")]},
                                       {"query": "Mexican-American War treaty", "visits": []}])
    token = H.CURRENT.set(ctx)
    try:
        assert H.current_query_for("x").startswith("Main Question:")
    finally:
        H.CURRENT.reset(token)
    assert H.current_query_for("x") == "x"                          # outside an episode: plain


def test_run_record_carries_hit_and_read_ids(tmp_path):
    require_jvm()
    from agent_search import cli
    assert cli.main([f"runs_dir={tmp_path / 'runs'}", f"index_root={tmp_path / 'idx'}"]) == 0
    row = json.loads(next((tmp_path / "runs").rglob("rows.jsonl")).read_text().splitlines()[0])
    search = row["trajectory"][0]
    assert T.is_search_action(search["action"]) and search["hit_ids"]
    reads = [s for s in row["trajectory"] if T.is_read_action(s["action"])]
    assert reads and reads[0]["read_ids"]


def test_trainer_command_and_config():
    cfg = R.TrainConfig(train_data="t.jsonl", output_dir="models/x")
    cmd = R.build_command(cfg, python="python")
    assert cmd[:6] == ["python", "-m", "torch.distributed.run", "--nproc_per_node", "1", "-m"]
    assert "FlagEmbedding.finetune.embedder.decoder_only.base" in cmd
    assert cmd[cmd.index("--query_instruction_for_retrieval") + 1] == Q.INSTRUCTIONS[Q.DEFAULT_STYLE]
    assert cmd[cmd.index("--neg_w_div") + 1] == "3.0" and cmd[cmd.index("--temperature") + 1] == "0.02"
    assert "--bf16" in cmd and "--gradient_checkpointing" in cmd
    text = R.template()
    parsed = yaml.safe_load(text)
    assert parsed["base_model"] == "Qwen/Qwen3-Embedding-0.6B" and parsed["query_style"] == Q.DEFAULT_STYLE
    assert yaml.safe_load(R.template("plain"))["query_max_len"] == 512


def test_training_file_round_trip_and_serving_note(tmp_path):
    f = tmp_path / "train.yaml"
    f.write_text(R.template())
    cfg = R.load(str(f))
    note = R.write_serving_note(cfg, str(tmp_path / "ckpt"))
    data = json.loads(Path(note).read_text())
    assert data["pooling"] == "last_token" and data["query_instruction"] == Q.INSTRUCTIONS[Q.DEFAULT_STYLE]
    from agent_search.retrievers.dense import DenseRetriever
    assert (DenseRetriever(str(tmp_path / "ckpt"), encoder=object()).query_prefix_for()
            == Q.instruction_prefix(Q.INSTRUCTIONS[Q.DEFAULT_STYLE]))
    assert DenseRetriever("BAAI/bge-base-en-v1.5", encoder=object()).query_prefix_for() == ""
    bad = tmp_path / "bad.yaml"
    bad.write_text("train_data: x\nlearning_rat: 1\n")
    with pytest.raises(ValueError, match="unknown training keys"):
        R.load(str(bad))


def test_patch_file_ships_and_environment_check_runs():
    assert R.PATCH_PATH.exists() and "neg_w_div" in R.PATCH_PATH.read_text()
    info = R.check_environment()
    assert "patched" in info and info["patch_sha256"]
    out = subprocess.run([sys.executable, "-m", "agent_search.training.retriever", "template"],
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0 and "base_model:" in out.stdout


def test_build_triples_cli_on_the_fixture(tmp_path):
    require_jvm()
    from agent_search import cli
    assert cli.main(["dataset=doc_fixture", "strategy=search_visit", f"runs_dir={tmp_path / 'runs'}",
                     f"index_root={tmp_path / 'idx'}"]) == 0
    run_dir = next((tmp_path / "runs").rglob("rows.jsonl")).parent
    from agent_search.training.build_triples import main as bt
    out = tmp_path / "triples.jsonl"
    rc = bt(["--runs", str(run_dir), "--dataset", "doc_fixture", "--out", str(out), "--query-style", "i2"])
    # the scripted policy reads the top hit right after one search: no read before the search,
    # so no negatives can exist and the builder reports an empty (rc=1) but well-formed output
    assert rc in (0, 1)
    assert out.with_suffix(".summary.json").exists()


def test_i9_renders_the_released_checkpoints_format():
    """The released ITER checkpoints expect i9 (the paper's ITER-i7): i2's fields plus the agent's
    pre-search reasoning on one line before the sub-query, `<empty>` when there is none."""
    inter = [{"query": "treaty Florida", "visits": []}]
    q = Q.render_query("i9", "Q?", "sub", inter, pre_reasoning="Results are\n not helpful.\n  Try the ship.")
    assert q == ("Main Question: Q?\nCurrent Reasoning: Results are not helpful. Try the ship.\n"
                 "Current Subquery: sub\nPrevious Interactions:\nPrevious SubQuery 1: treaty Florida")
    assert Q.render_query("i9", "Q?", "sub", []) == ("Main Question: Q?\nCurrent Reasoning: <empty>\n"
                                                      "Current Subquery: sub\nPrevious Interactions: <empty>")
    assert Q.INSTRUCTIONS["i9"].startswith("Given the main question, the agent's reasoning")
    assert Q.DEFAULT_STYLE == "i9"


def test_query_context_carries_the_current_turns_reasoning():
    from agent_search.training.history import QueryContext
    ctx = QueryContext(question="Q?", text_of=lambda d: "", style="i9")
    ctx.note('<think>\nSearch the ship.\n</think>\n<tool_call>{"name":"search","arguments":{"query":"ship"}}</tool_call>')
    assert ctx.render("ship").splitlines()[1] == "Current Reasoning: Search the ship."
