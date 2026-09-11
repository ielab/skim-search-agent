"""Offline tests for scripts/force_answer_backfill.py (no GPU/model/network needed):
prompt reconstruction from a synthetic row (full observations, not the 600-char-capped
trajectory field), the forced-prefill request shape + its one plain-ask fallback, idempotency
(recovered_answers.jsonl skip), the scoring-side merge overlay, and --dry-run.
"""
import json
import os
from types import SimpleNamespace

import pytest

from scripts.force_answer_backfill import (
    FORCE_MSG,
    call_plain_ask,
    call_prefill,
    elicit_answer,
    empty_answer_rows,
    infer_dataset_name,
    infer_field_profile,
    load_recovered_ids,
    load_rows_with_recovery,
    main,
    needs_recovery,
    process_condition_dir,
    prefill_messages_for,
    reconstruct_messages,
    resolve_condition_dirs,
)

CAPPED_PLACEHOLDER = "SHOULD NOT APPEAR (this is the 600-char-capped trajectory field)"
FULL_OBSERVATION = "FULL UNCAPPED OBSERVATION " + ("x" * 700)   # > 600 chars, distinct from the cap


def _synthetic_row(instance_id="inst_1", final_answer="", prompt_profile_path="research_dci"):
    """A minimal but schema-faithful row: trajectory[i]['observation'] deliberately holds a
    DIFFERENT (short) placeholder than observations[i], mirroring the real display-cap
    (`agent_search/evaluation/agent_runner.py::_trajectory_meta` truncates
    trajectory[i]['observation'] to 600 chars) so a test can assert the reconstruction used the
    FULL field, never the capped one."""
    trajectory = [
        {"action": "bash", "args": {"command": "grep -ril foo ."}, "query": "",
         "observation": CAPPED_PLACEHOLDER, "raw_output": "<tool_call>{\"name\":\"bash\",\"arguments\":"
         "{\"command\":\"grep -ril foo .\"}}</tool_call>"},
        {"action": "read", "args": {"path": "d1.txt"}, "query": "",
         "observation": CAPPED_PLACEHOLDER, "raw_output": "<tool_call>{\"name\":\"read\",\"arguments\":"
         "{\"path\":\"d1.txt\"}}</tool_call>"},
    ]
    observations = [FULL_OBSERVATION, FULL_OBSERVATION]
    return {
        "instance_id": instance_id, "question": "What is the answer?",
        "final_answer": final_answer, "gold_answer": "forty-two",
        "prompt_profile_path": prompt_profile_path, "tool_condition": f"agent_{prompt_profile_path}",
        "domain": "general", "stopped": "max_steps", "n_steps": len(trajectory),
        "trajectory": trajectory, "observations": observations,
    }


# --- prompt reconstruction ----------------------------------------------------------------------

def test_reconstruct_messages_uses_full_observations_not_the_capped_field():
    row = _synthetic_row()
    messages = reconstruct_messages(row, field_profile=None)
    assert messages[0]["role"] == "system" and messages[0]["content"]
    assert messages[1]["role"] == "user" and "What is the answer?" in messages[1]["content"]
    joined = "\n".join(m["content"] for m in messages)
    assert FULL_OBSERVATION in joined
    assert CAPPED_PLACEHOLDER not in joined
    # the forced-terminal instruction is appended last, as a user/tool_response turn.
    assert messages[-1]["role"] == "user"
    assert FORCE_MSG in messages[-1]["content"]


def test_reconstruct_messages_ctx_tokens_drops_oldest_pairs_first():
    row = _synthetic_row()
    # a tiny budget: only the newest (assistant, tool_response) pair (plus system/user/forced-turn)
    # can fit — the OLDER step's raw_output must be dropped, mirroring AgentPolicy.build_messages'
    # newest-first walk (agent_search/agent/policies.py).
    messages = reconstruct_messages(row, field_profile=None, ctx_tokens=50)
    joined = "\n".join(m["content"] for m in messages)
    assert "grep -ril foo" not in joined       # step 0 (oldest) dropped
    assert "d1.txt" in joined                  # step 1 (newest) kept


def test_prefill_messages_for_appends_open_answer_tag():
    row = _synthetic_row()
    base = reconstruct_messages(row, field_profile=None)
    prefilled = prefill_messages_for(base)
    assert prefilled[:-1] == base
    assert prefilled[-1] == {"role": "assistant", "content": "<answer>"}


def test_infer_dataset_and_field_profile_from_path():
    cond_dir = "runs/_headline_validation/agent/browsecomp_plus_structured/SomeModel/agent_research_indri"
    assert infer_dataset_name(cond_dir) == "browsecomp_plus_structured"
    assert infer_field_profile(cond_dir) == "browsecomp"
    flat_dir = "runs/agent/browsecomp_plus_flat/SomeModel/agent_research_bm25"
    assert infer_dataset_name(flat_dir) == "browsecomp_plus_flat"
    assert infer_field_profile(flat_dir) is None    # flat corpus inherits the domain's manual


# --- forced-prefill call shape + its one fallback ------------------------------------------------

def _fake_client(contents):
    """A stubbed OpenAI-compatible client (mirrors tests/test_agent_backend.py's `_fake_client`):
    returns `contents[i]` on the i-th `create()` call and records every call's kwargs."""
    calls = []

    def create(**kwargs):
        i = len(calls)
        calls.append(kwargs)
        text = contents[min(i, len(contents) - 1)]
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    client._calls = calls
    return client


def test_call_prefill_request_shape():
    client = _fake_client(["Paris"])
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "Q?"}]
    text = call_prefill(client, "my-model", messages, max_tokens=200)
    assert text == "Paris"
    kwargs = client._calls[0]
    assert kwargs["model"] == "my-model"
    assert kwargs["messages"][-1] == {"role": "assistant", "content": "<answer>"}
    assert kwargs["messages"][:-1] == messages
    assert kwargs["stop"] == ["</answer>"]
    assert kwargs["extra_body"] == {"add_generation_prompt": False, "continue_final_message": True}


def test_call_plain_ask_request_shape_has_no_prefill_flags():
    client = _fake_client(["<answer>Paris</answer>"])
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "Q?"}]
    text = call_plain_ask(client, "my-model", messages)
    assert text == "<answer>Paris</answer>"
    kwargs = client._calls[0]
    assert "extra_body" not in kwargs
    assert "stop" not in kwargs
    assert kwargs["messages"][-1]["role"] == "user"
    assert "Output ONLY <answer>your answer</answer>" in kwargs["messages"][-1]["content"]


def test_elicit_answer_uses_prefill_when_non_empty_single_call():
    client = _fake_client(["Paris"])
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "Q?"}]
    answer, n_attempts, raw = elicit_answer(messages, client, "my-model")
    assert answer == "Paris"
    assert n_attempts == 1
    assert raw == "Paris"
    assert len(client._calls) == 1                # single call — no fallback needed


def test_elicit_answer_falls_back_once_when_prefill_is_empty():
    # first call (prefill) returns whitespace only; second call (plain-ask fallback) answers.
    client = _fake_client(["   ", "<answer>Paris</answer>"])
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "Q?"}]
    answer, n_attempts, raw = elicit_answer(messages, client, "my-model")
    assert answer == "Paris"
    assert n_attempts == 2
    assert raw == "<answer>Paris</answer>"
    assert len(client._calls) == 2
    # the SECOND call must be the plain-ask shape (no prefill flags).
    assert "extra_body" not in client._calls[1]


def test_elicit_answer_strips_trailing_answer_tag_if_echoed():
    # belt-and-braces: some server config might echo the stop string anyway.
    client = _fake_client(["Paris</answer>"])
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "Q?"}]
    answer, n_attempts, _ = elicit_answer(messages, client, "my-model")
    assert answer == "Paris"
    assert n_attempts == 1


def test_elicit_answer_returns_empty_when_both_calls_fail():
    client = _fake_client(["", "no tags here, sorry"])
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "Q?"}]
    answer, n_attempts, _ = elicit_answer(messages, client, "my-model")
    assert answer == ""
    assert n_attempts == 2


# --- idempotency + end-to-end process_condition_dir (stubbed client) ----------------------------

def _write_rows(cond_dir, rows):
    os.makedirs(cond_dir, exist_ok=True)
    with open(os.path.join(cond_dir, "rows.jsonl"), "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def test_process_condition_dir_is_idempotent(tmp_path, monkeypatch):
    cond_dir = str(tmp_path / "runs" / "agent" / "browsecomp_plus_structured" / "M" / "agent_research_dci")
    empty_row = _synthetic_row(instance_id="empty_1", final_answer="")
    answered_row = _synthetic_row(instance_id="answered_1", final_answer="already answered")
    _write_rows(cond_dir, [empty_row, answered_row])

    calls = {"n": 0}

    def fake_build_client(api_base, api_key=None):
        calls["n"] += 1
        return _fake_client(["forty-two"])

    monkeypatch.setattr("scripts.force_answer_backfill.build_client", fake_build_client)

    summary1 = process_condition_dir(
        cond_dir, model="m", api_base="http://fake/v1", ctx_tokens=110_000,
        prefill_max_tokens=200, fallback_max_tokens=512, stamp="2026-01-01T00:00:00Z")
    assert summary1["n_empty"] == 1                 # only the empty row is a candidate
    assert summary1["n_already_recovered"] == 0
    assert summary1["n_pending"] == 1
    assert summary1["n_recovered"] == 1

    rec_path = os.path.join(cond_dir, "recovered_answers.jsonl")
    assert os.path.exists(rec_path)
    recs = [json.loads(l) for l in open(rec_path) if l.strip()]
    assert len(recs) == 1
    assert recs[0]["instance_id"] == "empty_1"
    assert recs[0]["recovered_answer"] == "forty-two"
    assert recs[0]["method"] == "forced_terminal_prefill_v1"
    assert recs[0]["n_attempts"] == 1
    assert "raw_continuation" in recs[0]

    # SECOND run over the SAME cond_dir: the already-recovered id must be skipped (idempotent) —
    # no new model call, no new line appended.
    summary2 = process_condition_dir(
        cond_dir, model="m", api_base="http://fake/v1", ctx_tokens=110_000,
        prefill_max_tokens=200, fallback_max_tokens=512, stamp="2026-01-02T00:00:00Z")
    assert summary2["n_already_recovered"] == 1
    assert summary2["n_pending"] == 0
    assert summary2.get("n_recovered", 0) == 0
    recs_after = [json.loads(l) for l in open(rec_path) if l.strip()]
    assert len(recs_after) == 1                      # unchanged


def test_process_condition_dir_selects_placeholder_rows_and_stays_idempotent(tmp_path, monkeypatch):
    """Placeholder rows ("...") are selected for recovery alongside truly-empty rows, and once a
    row (placeholder or empty) is covered by recovered_answers.jsonl, a second run skips it and
    makes no new model call — the sidecar format/idempotency contract is unchanged by widening the
    selection predicate."""
    cond_dir = str(tmp_path / "runs" / "agent" / "browsecomp_plus_structured" / "M" / "agent_research_dci")
    empty_row = _synthetic_row(instance_id="empty_1", final_answer="")
    placeholder_row = _synthetic_row(instance_id="placeholder_1", final_answer="...")
    answered_row = _synthetic_row(instance_id="answered_1", final_answer="already answered")
    _write_rows(cond_dir, [empty_row, placeholder_row, answered_row])

    def fake_build_client(api_base, api_key=None):
        return _fake_client(["forty-two"])

    monkeypatch.setattr("scripts.force_answer_backfill.build_client", fake_build_client)

    summary1 = process_condition_dir(
        cond_dir, model="m", api_base="http://fake/v1", ctx_tokens=110_000,
        prefill_max_tokens=200, fallback_max_tokens=512, stamp="2026-01-01T00:00:00Z")
    assert summary1["n_empty"] == 2                  # empty_1 AND placeholder_1 are candidates
    assert summary1["n_already_recovered"] == 0
    assert summary1["n_pending"] == 2
    assert summary1["n_recovered"] == 2

    rec_path = os.path.join(cond_dir, "recovered_answers.jsonl")
    recs = {json.loads(l)["instance_id"] for l in open(rec_path) if l.strip()}
    assert recs == {"empty_1", "placeholder_1"}

    # SECOND run: both are now already-recovered -> skipped, no new calls, no new lines.
    summary2 = process_condition_dir(
        cond_dir, model="m", api_base="http://fake/v1", ctx_tokens=110_000,
        prefill_max_tokens=200, fallback_max_tokens=512, stamp="2026-01-02T00:00:00Z")
    assert summary2["n_already_recovered"] == 2
    assert summary2["n_pending"] == 0
    assert summary2.get("n_recovered", 0) == 0
    recs_after = [json.loads(l) for l in open(rec_path) if l.strip()]
    assert len(recs_after) == 2                      # unchanged


# --- per-row overflow resilience (job 28673993: one vLLM 400 killed the whole pass) --------------

OVERFLOW_MSG = ("Error code: 400 - This model's maximum context length is 131072 tokens. "
                "However, your request contains at least 130873 input tokens.")


def test_overflow_row_retries_with_shrunken_context_then_recovers(tmp_path, monkeypatch):
    """Two overflow 400s then success: the row must be recovered, and each retry must have
    rebuilt the prompt with a 15%-smaller ctx_tokens (mirroring AgentPolicy.propose())."""
    import scripts.force_answer_backfill as fab
    cond_dir = str(tmp_path / "runs" / "agent" / "browsecomp_plus_structured" / "M" / "agent_research_dci")
    _write_rows(cond_dir, [_synthetic_row(instance_id="of_1", final_answer="")])

    ctx_seen = []
    real_reconstruct = fab.reconstruct_messages

    def spy_reconstruct(row, field_profile, ctx_tokens=fab.DEFAULT_CTX_TOKENS):
        ctx_seen.append(ctx_tokens)
        return real_reconstruct(row, field_profile, ctx_tokens=ctx_tokens)

    monkeypatch.setattr(fab, "reconstruct_messages", spy_reconstruct)

    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        if len(calls) <= 2:
            raise RuntimeError(OVERFLOW_MSG)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="forty-two"))])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(fab, "build_client", lambda api_base, api_key=None: client)

    summary = process_condition_dir(
        cond_dir, model="m", api_base="http://fake/v1", ctx_tokens=110_000,
        prefill_max_tokens=200, fallback_max_tokens=512, stamp="2026-01-01T00:00:00Z")
    assert summary["n_recovered"] == 1
    assert summary["n_skipped_overflow"] == 0
    assert summary["n_failed"] == 0
    # 3 attempts: full window, then x0.85, then x0.85^2 (int() after each shrink).
    assert ctx_seen == [110000, 93500, 79475]
    recs = [json.loads(l) for l in open(os.path.join(cond_dir, "recovered_answers.jsonl")) if l.strip()]
    assert len(recs) == 1 and recs[0]["instance_id"] == "of_1"
    assert recs[0]["recovered_answer"] == "forty-two"


def test_always_overflowing_row_is_skipped_and_pass_continues(tmp_path, monkeypatch):
    """A row that overflows on ALL 4 attempts (3 shrinks) is skipped without a sidecar record;
    the other pending rows are still processed and the job exits 0 (success)."""
    import scripts.force_answer_backfill as fab
    cond_dir = str(tmp_path / "runs" / "agent" / "browsecomp_plus_structured" / "M" / "agent_research_dci")
    bad = _synthetic_row(instance_id="pathological_1", final_answer="")
    bad["question"] = "OVERFLOW-MARKER what is it?"
    good = _synthetic_row(instance_id="good_1", final_answer="...")   # placeholder row, also pending
    _write_rows(cond_dir, [bad, good])

    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        if any("OVERFLOW-MARKER" in m["content"] for m in kwargs["messages"]):
            raise RuntimeError(OVERFLOW_MSG)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="forty-two"))])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(fab, "build_client", lambda api_base, api_key=None: client)

    log_lines = []
    summary = process_condition_dir(
        cond_dir, model="m", api_base="http://fake/v1", ctx_tokens=110_000,
        prefill_max_tokens=200, fallback_max_tokens=512, stamp="2026-01-01T00:00:00Z",
        log=log_lines.append)
    assert summary["n_pending"] == 2
    assert summary["n_recovered"] == 1                # good_1 still processed
    assert summary["n_skipped_overflow"] == 1         # pathological_1 skipped, not crashed
    assert summary["n_failed"] == 0
    # the skip is logged with instance_id + the final (smallest) ctx size tried.
    skip_lines = [l for l in log_lines if "SKIPPED context-overflow" in l]
    assert len(skip_lines) == 1
    assert "pathological_1" in skip_lines[0]
    final_ctx = int(int(int(110000 * 0.85) * 0.85) * 0.85)
    assert f"final_ctx_tokens={final_ctx}" in skip_lines[0]
    # exactly 4 attempts were made for the pathological row (1 + 3 shrinks).
    bad_calls = [k for k in calls if any("OVERFLOW-MARKER" in m["content"] for m in k["messages"])]
    assert len(bad_calls) == 4
    # NO sidecar record for the skipped row — a later pass can retry it.
    recs = [json.loads(l) for l in open(os.path.join(cond_dir, "recovered_answers.jsonl")) if l.strip()]
    assert {r["instance_id"] for r in recs} == {"good_1"}

    # end-to-end: main() over the same dir returns SUCCESS despite the (idempotently re-hit)
    # overflow skip — good_1 is already recovered, so only the pathological row is pending.
    monkeypatch.setenv("MODEL", "m")
    rc = main([cond_dir, "--api-base", "http://fake/v1"])
    assert rc == 0


def test_non_overflow_row_exception_is_logged_and_counted_not_fatal(tmp_path, monkeypatch):
    """Any other per-row exception (e.g. a transient connection error) must not kill the pass:
    it is logged, counted in n_failed, and the remaining rows are still processed."""
    import scripts.force_answer_backfill as fab
    cond_dir = str(tmp_path / "runs" / "agent" / "browsecomp_plus_structured" / "M" / "agent_research_dci")
    bad = _synthetic_row(instance_id="flaky_1", final_answer="")
    bad["question"] = "FLAKY-MARKER what is it?"
    good = _synthetic_row(instance_id="good_1", final_answer="")
    _write_rows(cond_dir, [bad, good])

    def create(**kwargs):
        if any("FLAKY-MARKER" in m["content"] for m in kwargs["messages"]):
            raise ValueError("connection reset by peer")   # NOT a context-length error
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="forty-two"))])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(fab, "build_client", lambda api_base, api_key=None: client)

    log_lines = []
    summary = process_condition_dir(
        cond_dir, model="m", api_base="http://fake/v1", ctx_tokens=110_000,
        prefill_max_tokens=200, fallback_max_tokens=512, stamp="2026-01-01T00:00:00Z",
        log=log_lines.append)
    assert summary["n_recovered"] == 1
    assert summary["n_skipped_overflow"] == 0
    assert summary["n_failed"] == 1
    fail_lines = [l for l in log_lines if "ROW FAILED" in l]
    assert len(fail_lines) == 1 and "flaky_1" in fail_lines[0]
    recs = [json.loads(l) for l in open(os.path.join(cond_dir, "recovered_answers.jsonl")) if l.strip()]
    assert {r["instance_id"] for r in recs} == {"good_1"}


def test_main_exits_nonzero_when_over_20pct_of_rows_fail(tmp_path, monkeypatch):
    """Systemic breakage (every call raising a non-overflow error -> 100% failure) makes the
    exit code nonzero; the run itself still completes (no crash)."""
    import scripts.force_answer_backfill as fab
    cond_dir = str(tmp_path / "runs" / "agent" / "browsecomp_plus_structured" / "M" / "agent_research_dci")
    _write_rows(cond_dir, [_synthetic_row(instance_id=f"r{i}", final_answer="") for i in range(5)])

    def create(**kwargs):
        raise ValueError("server is down")

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(fab, "build_client", lambda api_base, api_key=None: client)
    monkeypatch.setenv("MODEL", "m")
    rc = main([cond_dir, "--api-base", "http://fake/v1"])
    assert rc == 1
    assert not os.path.exists(os.path.join(cond_dir, "recovered_answers.jsonl"))


def test_overflow_resilience_in_threaded_mode(tmp_path, monkeypatch):
    """workers>1 path (how the crashed GPU job actually ran): a permanently-overflowing row and
    a failing row must not kill the pool; the good row is still recovered."""
    import scripts.force_answer_backfill as fab
    cond_dir = str(tmp_path / "runs" / "agent" / "browsecomp_plus_structured" / "M" / "agent_research_dci")
    of_row = _synthetic_row(instance_id="of_1", final_answer="")
    of_row["question"] = "OVERFLOW-MARKER q"
    bad_row = _synthetic_row(instance_id="flaky_1", final_answer="")
    bad_row["question"] = "FLAKY-MARKER q"
    good = _synthetic_row(instance_id="good_1", final_answer="")
    _write_rows(cond_dir, [of_row, bad_row, good])

    def create(**kwargs):
        blob = "\n".join(m["content"] for m in kwargs["messages"])
        if "OVERFLOW-MARKER" in blob:
            raise RuntimeError(OVERFLOW_MSG)
        if "FLAKY-MARKER" in blob:
            raise ValueError("boom")
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="forty-two"))])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(fab, "build_client", lambda api_base, api_key=None: client)

    summary = process_condition_dir(
        cond_dir, model="m", api_base="http://fake/v1", ctx_tokens=110_000,
        prefill_max_tokens=200, fallback_max_tokens=512, stamp="2026-01-01T00:00:00Z",
        workers=3, log=lambda m: None)
    assert summary["n_recovered"] == 1
    assert summary["n_skipped_overflow"] == 1
    assert summary["n_failed"] == 1
    recs = [json.loads(l) for l in open(os.path.join(cond_dir, "recovered_answers.jsonl")) if l.strip()]
    assert {r["instance_id"] for r in recs} == {"good_1"}


def test_empty_answer_rows_and_resolve_condition_dirs(tmp_path):
    cond_dir = tmp_path / "cell"
    _write_rows(str(cond_dir), [
        _synthetic_row(instance_id="a", final_answer=""),
        _synthetic_row(instance_id="b", final_answer="has one"),
        _synthetic_row(instance_id="c", final_answer="   "),   # whitespace-only counts as empty
        _synthetic_row(instance_id="d", final_answer="..."),   # placeholder counts as empty too
        _synthetic_row(instance_id="e", final_answer="."),     # single-dot placeholder
        _synthetic_row(instance_id="f", final_answer=".."),    # double-dot placeholder
        _synthetic_row(instance_id="g", final_answer="  ...  "),  # placeholder with surrounding ws
    ])
    empty = empty_answer_rows(str(cond_dir))
    assert {r["instance_id"] for r in empty} == {"a", "c", "d", "e", "f", "g"}

    dirs_by_dir_glob = resolve_condition_dirs([str(cond_dir)])
    dirs_by_rows_glob = resolve_condition_dirs([str(cond_dir / "rows.jsonl")])
    assert dirs_by_dir_glob == [os.path.normpath(str(cond_dir))]
    assert dirs_by_rows_glob == [os.path.normpath(str(cond_dir))]


# --- needs_recovery: single source of truth for "empty-ish" ------------------------------------

def test_needs_recovery_true_for_empty_and_placeholder_answers():
    for ans in ("", "   ", "...", ".", "..", "  ...  ", None):
        assert needs_recovery(ans) is True, repr(ans)


def test_needs_recovery_false_for_real_answers():
    for ans in ("Paris", "42", "...and more", ". . .", "no."):
        assert needs_recovery(ans) is False, repr(ans)


def test_rows_jsonl_tolerates_unparsable_trailing_line(tmp_path):
    cond_dir = tmp_path / "live_cell"
    os.makedirs(cond_dir)
    with open(cond_dir / "rows.jsonl", "w") as fh:
        fh.write(json.dumps(_synthetic_row(instance_id="ok_row", final_answer="")) + "\n")
        fh.write('{"instance_id": "partial", "final_ans')   # truncated mid-write, no trailing \n
    rows = empty_answer_rows(str(cond_dir))
    assert [r["instance_id"] for r in rows] == ["ok_row"]


# --- scoring-side merge overlay ------------------------------------------------------------------

def test_load_rows_with_recovery_overlays_only_empty_recovered_rows(tmp_path):
    cond_dir = tmp_path / "cell2"
    rows = [
        _synthetic_row(instance_id="recovered_and_empty", final_answer=""),
        _synthetic_row(instance_id="never_recovered_empty", final_answer=""),
        _synthetic_row(instance_id="already_answered", final_answer="untouched"),
    ]
    _write_rows(str(cond_dir), rows)
    with open(cond_dir / "recovered_answers.jsonl", "w") as fh:
        fh.write(json.dumps({"instance_id": "recovered_and_empty", "recovered_answer": "Paris",
                             "n_attempts": 1, "recovered_at_iso": "2026-01-01T00:00:00Z",
                             "method": "forced_terminal_prefill_v1"}) + "\n")
        # a recovered entry for an id that ISN'T empty in rows.jsonl must be ignored (never
        # overwrites an already-answered row).
        fh.write(json.dumps({"instance_id": "already_answered", "recovered_answer": "SHOULD NOT WIN",
                             "n_attempts": 1, "recovered_at_iso": "2026-01-01T00:00:00Z",
                             "method": "forced_terminal_prefill_v1"}) + "\n")

    merged = load_rows_with_recovery(str(cond_dir))
    by_id = {r["instance_id"]: r for r in merged}
    assert by_id["recovered_and_empty"]["final_answer"] == "Paris"
    assert by_id["recovered_and_empty"]["recovered"] is True
    assert by_id["never_recovered_empty"]["final_answer"] == ""
    assert by_id["never_recovered_empty"]["recovered"] is False
    assert by_id["already_answered"]["final_answer"] == "untouched"
    assert by_id["already_answered"]["recovered"] is False


def test_load_rows_with_recovery_overlays_placeholder_rows_too(tmp_path):
    """A row whose `final_answer` is a placeholder ("...") — not truly empty — must ALSO be
    overlaid by a recovered entry (the bug this task fixes: force_answer_backfill previously
    keyed its overlay on `not final_answer.strip()` only, so a recovered entry for a placeholder
    row would silently never apply even if one existed)."""
    cond_dir = tmp_path / "cell_placeholder"
    rows = [
        _synthetic_row(instance_id="placeholder_dots", final_answer="..."),
        _synthetic_row(instance_id="placeholder_single_dot", final_answer="."),
    ]
    _write_rows(str(cond_dir), rows)
    with open(cond_dir / "recovered_answers.jsonl", "w") as fh:
        fh.write(json.dumps({"instance_id": "placeholder_dots", "recovered_answer": "Paris",
                             "n_attempts": 1, "recovered_at_iso": "2026-01-01T00:00:00Z",
                             "method": "forced_terminal_prefill_v1"}) + "\n")

    merged = load_rows_with_recovery(str(cond_dir))
    by_id = {r["instance_id"]: r for r in merged}
    assert by_id["placeholder_dots"]["final_answer"] == "Paris"
    assert by_id["placeholder_dots"]["recovered"] is True
    # no recovered entry for this one -> stays as the placeholder, recovered=False.
    assert by_id["placeholder_single_dot"]["final_answer"] == "."
    assert by_id["placeholder_single_dot"]["recovered"] is False


def test_load_rows_with_recovery_ignores_placeholder_recovered_answer(tmp_path):
    """A sidecar entry whose recovered_answer is itself a placeholder ("...", ".", "..", "")
    must NOT overlay (real case: dci browsecomp_plus_structured__844 — the model prefilled
    "..." even under forced decoding). The row stays empty-ish (counted in empty%, still a
    future recovery target) instead of becoming a fake non-empty answer; a genuine recovered
    answer still overlays."""
    cond_dir = tmp_path / "cell_placeholder_recovery"
    rows = [
        _synthetic_row(instance_id="fake_recovery_dots", final_answer=""),
        _synthetic_row(instance_id="fake_recovery_empty", final_answer="..."),
        _synthetic_row(instance_id="real_recovery", final_answer=""),
    ]
    _write_rows(str(cond_dir), rows)
    with open(cond_dir / "recovered_answers.jsonl", "w") as fh:
        fh.write(json.dumps({"instance_id": "fake_recovery_dots", "recovered_answer": "...",
                             "n_attempts": 1, "recovered_at_iso": "2026-01-01T00:00:00Z",
                             "method": "forced_terminal_prefill_v1"}) + "\n")
        fh.write(json.dumps({"instance_id": "fake_recovery_empty", "recovered_answer": "",
                             "n_attempts": 2, "recovered_at_iso": "2026-01-01T00:00:00Z",
                             "method": "forced_terminal_prefill_v1"}) + "\n")
        fh.write(json.dumps({"instance_id": "real_recovery", "recovered_answer": "Paris",
                             "n_attempts": 1, "recovered_at_iso": "2026-01-01T00:00:00Z",
                             "method": "forced_terminal_prefill_v1"}) + "\n")

    merged = load_rows_with_recovery(str(cond_dir))
    by_id = {r["instance_id"]: r for r in merged}
    # placeholder recoveries do NOT overlay — rows keep their original (empty-ish) answers.
    assert by_id["fake_recovery_dots"]["final_answer"] == ""
    assert by_id["fake_recovery_dots"]["recovered"] is False
    assert by_id["fake_recovery_empty"]["final_answer"] == "..."
    assert by_id["fake_recovery_empty"]["recovered"] is False
    # genuine recovery still overlays.
    assert by_id["real_recovery"]["final_answer"] == "Paris"
    assert by_id["real_recovery"]["recovered"] is True
    # and both fake-recovery rows still count as needing recovery downstream.
    assert needs_recovery(by_id["fake_recovery_dots"]["final_answer"])
    assert needs_recovery(by_id["fake_recovery_empty"]["final_answer"])


def test_later_genuine_recovery_wins_over_earlier_placeholder_entry(tmp_path):
    """Re-attempt semantics: a placeholder entry does not suppress a later genuine one for the
    same id (append-only sidecar — the genuine retry appends after the placeholder record)."""
    cond_dir = tmp_path / "cell_retry_after_placeholder"
    _write_rows(str(cond_dir), [_synthetic_row(instance_id="retry_1", final_answer="")])
    with open(cond_dir / "recovered_answers.jsonl", "w") as fh:
        fh.write(json.dumps({"instance_id": "retry_1", "recovered_answer": "...",
                             "n_attempts": 1, "recovered_at_iso": "2026-01-01T00:00:00Z",
                             "method": "forced_terminal_prefill_v1"}) + "\n")
        fh.write(json.dumps({"instance_id": "retry_1", "recovered_answer": "Paris",
                             "n_attempts": 1, "recovered_at_iso": "2026-01-02T00:00:00Z",
                             "method": "forced_terminal_prefill_v1"}) + "\n")
    merged = load_rows_with_recovery(str(cond_dir))
    assert merged[0]["final_answer"] == "Paris"
    assert merged[0]["recovered"] is True


def test_placeholder_recovery_does_not_suppress_reattempt(tmp_path, monkeypatch):
    """process_condition_dir: a row whose ONLY sidecar entry is a placeholder recovery is
    still pending (n_already_recovered excludes it) and gets re-attempted; the new genuine
    answer appends and then wins."""
    import scripts.force_answer_backfill as fab
    cond_dir = str(tmp_path / "runs" / "agent" / "browsecomp_plus_structured" / "M" / "agent_research_dci")
    _write_rows(cond_dir, [_synthetic_row(instance_id="stuck_1", final_answer="")])
    with open(os.path.join(cond_dir, "recovered_answers.jsonl"), "w") as fh:
        fh.write(json.dumps({"instance_id": "stuck_1", "recovered_answer": "...",
                             "n_attempts": 1, "recovered_at_iso": "2026-01-01T00:00:00Z",
                             "method": "forced_terminal_prefill_v1"}) + "\n")

    assert load_recovered_ids(cond_dir) == set()      # placeholder entry is not coverage

    monkeypatch.setattr(fab, "build_client", lambda api_base, api_key=None: _fake_client(["forty-two"]))
    summary = process_condition_dir(
        cond_dir, model="m", api_base="http://fake/v1", ctx_tokens=110_000,
        prefill_max_tokens=200, fallback_max_tokens=512, stamp="2026-01-02T00:00:00Z")
    assert summary["n_already_recovered"] == 0
    assert summary["n_pending"] == 1                  # NOT suppressed by the placeholder entry
    assert summary["n_recovered"] == 1
    assert summary["n_placeholder_recoveries"] == 0

    recs = [json.loads(l) for l in open(os.path.join(cond_dir, "recovered_answers.jsonl")) if l.strip()]
    assert len(recs) == 2                             # append-only: placeholder kept for audit
    merged = load_rows_with_recovery(cond_dir)
    assert merged[0]["final_answer"] == "forty-two"
    assert merged[0]["recovered"] is True
    # now genuinely covered: a third run has nothing pending.
    summary2 = process_condition_dir(
        cond_dir, model="m", api_base="http://fake/v1", ctx_tokens=110_000,
        prefill_max_tokens=200, fallback_max_tokens=512, stamp="2026-01-03T00:00:00Z")
    assert summary2["n_already_recovered"] == 1
    assert summary2["n_pending"] == 0


def test_fresh_placeholder_recovery_is_recorded_but_not_counted_as_success(tmp_path, monkeypatch):
    """A live pass where the model prefill emits '...' anyway: the attempt is appended for
    audit but counted as n_placeholder_recoveries, not n_recovered — and it does not overlay."""
    import scripts.force_answer_backfill as fab
    cond_dir = str(tmp_path / "runs" / "agent" / "browsecomp_plus_structured" / "M" / "agent_research_dci")
    _write_rows(cond_dir, [_synthetic_row(instance_id="dots_1", final_answer="")])
    monkeypatch.setattr(fab, "build_client", lambda api_base, api_key=None: _fake_client(["..."]))

    log_lines = []
    summary = process_condition_dir(
        cond_dir, model="m", api_base="http://fake/v1", ctx_tokens=110_000,
        prefill_max_tokens=200, fallback_max_tokens=512, stamp="2026-01-01T00:00:00Z",
        log=log_lines.append)
    assert summary["n_recovered"] == 0
    assert summary["n_placeholder_recoveries"] == 1
    assert summary["n_failed"] == 0
    assert any("PLACEHOLDER RECOVERY" in l and "dots_1" in l for l in log_lines)
    recs = [json.loads(l) for l in open(os.path.join(cond_dir, "recovered_answers.jsonl")) if l.strip()]
    assert len(recs) == 1 and recs[0]["recovered_answer"] == "..."   # audit record kept
    merged = load_rows_with_recovery(cond_dir)
    assert merged[0]["final_answer"] == ""            # never overlaid with the placeholder
    assert merged[0]["recovered"] is False


def test_load_rows_with_recovery_no_recovered_file_is_a_noop(tmp_path):
    cond_dir = tmp_path / "cell3"
    _write_rows(str(cond_dir), [_synthetic_row(instance_id="x", final_answer="")])
    merged = load_rows_with_recovery(str(cond_dir))
    assert merged[0]["recovered"] is False
    assert merged[0]["final_answer"] == ""


# --- --dry-run -------------------------------------------------------------------------------

def test_dry_run_reports_counts_and_sample_without_calling_model(tmp_path, monkeypatch, capsys):
    cond_dir = tmp_path / "runs" / "agent" / "browsecomp_plus_structured" / "M" / "agent_research_dci"
    _write_rows(str(cond_dir), [_synthetic_row(instance_id="only_row", final_answer="")])

    def _explode(*a, **k):
        raise AssertionError("dry-run must never call build_client")

    monkeypatch.setattr("scripts.force_answer_backfill.build_client", _explode)
    rc = main([str(cond_dir), "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "DRY-RUN sample reconstructed prompt" in out
    assert "only_row" in out
    assert not os.path.exists(os.path.join(str(cond_dir), "recovered_answers.jsonl"))


def test_dry_run_with_no_pending_rows_prints_no_sample(tmp_path, capsys):
    cond_dir = tmp_path / "cell_all_answered"
    _write_rows(str(cond_dir), [_synthetic_row(instance_id="x", final_answer="done")])
    rc = main([str(cond_dir), "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "DRY-RUN sample reconstructed prompt" not in out
