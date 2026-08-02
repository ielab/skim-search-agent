"""Offline forced-terminal-elicitation backfill for empty-`final_answer` browsecomp rows.

WHY (see docs/factorial_snapshot_20260710.md): `agent_search/agent/loop.py`'s
`run_episode` reserves its LAST allowed turn to inject a "STEP BUDGET REACHED" nudge —
a ONE-shot instruction to stop searching and commit an `<answer>` NOW. On long-episode
browsecomp cells Tongyi ignores that nudge 60-78% of the time and keeps emitting tool
calls instead; the episode then exhausts at `n_steps == max_steps+1` /
`stopped == "max_steps"` with `final_answer == ""` (26-42% of rows in the affected
cells). We are NOT allowed to touch the episode loop (mixed-provenance rule: the rows
already collected must stay reproducible against the exact loop.py/policies.py that
produced them). This script is instead a UNIFORM OFFLINE RECOVERY PASS, run entirely
after the fact: for every already-terminated empty-answer row it replays the episode's
terminal context to the model ONE more time and FORCES an answer out of it. Results
land in a SIBLING file, `<condition_dir>/recovered_answers.jsonl` — `rows.jsonl` is
only ever read, never written or mutated.

FORCING MECHANISM — ASSISTANT PREFILL, not ask-and-retry (design supersedes an
earlier ask-and-retry draft of this module): asking the model to emit `<answer>` tags
and retrying on refusal is exactly the failure mode we are recovering FROM (the model
already ignored one such instruction inline, 60-78% of the time). Instead the PRIMARY
mechanism is deterministic, single-call FORCED DECODING: after the reconstructed
conversation + the "STEP BUDGET REACHED / tools disabled" instruction, we append an
`assistant`-role message whose content is already the open tag `"<answer>"`, and ask
vLLM's OpenAI-compatible `/v1/chat/completions` to CONTINUE that message rather than
start a new turn. vLLM's `ChatCompletionRequest` (see
`envs/lib/python3.10/site-packages/vllm/entrypoints/openai/chat_completion/protocol.py`,
this repo's vLLM 0.22.1) exposes exactly this as two request fields —
`continue_final_message: bool` ("the chat will be formatted so that the final message
... is open-ended ... allows you to 'prefill' part of the model's response") and
`add_generation_prompt: bool`, which the same protocol's validator requires be `False`
whenever `continue_final_message` is `True` (the two are mutually exclusive). Neither
field is part of the OpenAI SDK's typed `chat.completions.create` signature, so they
are passed via `extra_body={"add_generation_prompt": False, "continue_final_message":
True}`, which the `openai` Python client merges verbatim into the JSON payload — this
is the chat-completions route (not `/v1/completions` with a manually-applied chat
template — the chat route is simpler here and vLLM's own field support makes the
manual-template route unnecessary). Generation is capped at a small `max_tokens`
(the model only needs to emit the short answer span) and stopped at `"</answer>"` (a
normal `stop` sequence, excluded from the returned text by vLLM's default
`include_stop_str_in_output=False`) — so the raw continuation IS the answer span,
already tag-free, by construction; there is no tag to fail to emit. This is
deterministic and single-call: the model cannot "ignore" the instruction and keep
tool-calling, because there is no token position left at which a tool call could
start — it is mid-way through an already-open `<answer>` string.

FALLBACK (kept minimal, per the design brief): if the forced continuation still comes
back empty/whitespace (a genuinely blank completion, which forced decoding cannot
prevent), we make exactly ONE plain (non-prefilled) follow-up call over the same
reconstructed conversation, asking the model to emit its own `<answer>...</answer>`,
extracted with the rfind-last extractor (see EXTRACTION below). No further retries.

PROMPT RECONSTRUCTION — design choice (task spec allows the "acceptable
simplification": system + question + a compacted transcript, documented here):

  `rows.jsonl`'s `trajectory[i]["observation"]` is DISPLAY-CAPPED to 600 chars by
  `agent_search/agent/retriever.py::_trajectory_meta` — never usable for a faithful
  replay. The FULL, uncapped observation for step `i` lives in the row's top-level
  `observations[i]` (same order, same length; see `_trajectory_meta`), which is what
  we use instead. `trajectory[i]["raw_output"]` is already the FULL raw model
  generation for that step (not capped).

  Rather than hand-rolling the history-window/truncation logic, we import and reuse
  `agent_search.agent.policies.AgentPolicy.build_messages` directly: we rebuild the
  row's `trajectory`/`observations` into real `agent_search.agent.loop.Step` objects
  and a real `Task(query=row["question"])`, construct an `AgentPolicy` with the SAME
  `prompt_path` (`row["prompt_profile_path"]`, the condition name, e.g.
  `"research_indri"` — carried verbatim in every row) and the SAME `field_profile`
  (derived from the dataset name found in the condition-dir path via
  `evaluation.datasets.dataset_field_profile`, since the row itself does not carry
  the profile — see `_infer_field_profile`), then call `.build_messages(task, steps)`.
  This is the EXACT code path that built the system prompt + newest-first,
  whole-(assistant,tool_response)-pair, char-budget-windowed history the model
  actually saw during the live episode — so the replayed prompt matches the real
  loop/policies wiring by construction, not by a parallel reimplementation that could
  drift from it.

  CONTEXT CAP: `AgentPolicy`'s own default `ctx_chars=450_000` (~128k tokens by its
  own `chars/3.5` heuristic, see `policies.py`) is already close to the ~120k-token
  cap this task asks for. We pass a slightly tighter `ctx_chars=420_000`
  (120_000 tokens * 3.5 chars/token, the SAME chars/3.5 heuristic, applied explicitly
  here so the cap is visible and tunable via `--ctx-chars` rather than silently
  inherited) — same whole-pair-drop-oldest-first algorithm, just a dedicated budget
  for recovery reconstruction. We do NOT attempt to recover the byte-exact window the
  live episode held at an EARLIER step; we want the window as of the point the episode
  actually ended, which this walk (over the row's FULL stored trajectory) reconstructs
  correctly by definition.

  We do not reconstruct/replay the mid-episode "budget" nudge specially — it is
  already one of the trajectory's real steps (`action == "budget"`, an empty
  `raw_output` paired with the injected nudge observation, exactly as the live loop
  produced it) and is carried through like any other step. What we APPEND after the
  full reconstructed transcript is a NEW user turn (the "STEP BUDGET REACHED / tools
  DISABLED" instruction, mirroring the semantics of that inline nudge) followed by the
  forced-prefill assistant turn described above.

EXTRACTION: the prefill continuation IS the answer span by construction (see above);
for the (rare) plain-ask fallback we reuse `agent_search.agent.loop._extract_answer`
(rfind-LAST `<answer>` open tag, so a model that names the tag in prose before the
real one still extracts correctly — see that function's own docstring for the
~37%-mis-extraction bug it fixes). We do not reimplement this.

OUTPUT: `<condition_dir>/recovered_answers.jsonl`, one record per recovered row:
`{instance_id, recovered_answer, n_attempts, raw_continuation, recovered_at_iso,
method}`, `method == "forced_terminal_prefill_v1"`, `n_attempts` 1 (prefill
succeeded) or 2 (prefill was empty, fallback plain-ask used), `raw_continuation` the
raw text of whichever call produced `recovered_answer` (for audit). Idempotent (skips
instance_ids already GENUINELY recovered in that file — an entry whose
`recovered_answer` is itself empty/placeholder is kept for audit but does not count
as coverage, so the row is re-attempted next pass and never overlaid downstream) and
append-only (one record flushed per row, so a killed job loses at most the in-flight
row). `rows.jsonl` parsing tolerates
a live-appended file (unparsable trailing line -> skipped, mirrors
`evaluation.run_eval._load_rows`).

PER-ROW RESILIENCE (added after job 28673993 crashed): a "maximum context length"
error (the chars/3.5 heuristic can overshoot the model's true token window on
token-dense rows) retries THAT ROW with the reconstructed window shrunk 15% at a
time, up to 3 shrinks — the same loop `AgentPolicy.propose()` uses live
(`agent_search/agent/policies.py`); a row that still overflows is SKIPPED with a log
line (no sidecar record, so a later pass with a bigger window can retry it). Any
other per-row exception is logged + counted and the pass CONTINUES; the exit code is
nonzero only when >20% of processed rows failed (systemic breakage), never for
isolated rows.

INTEGRATION: `load_rows_with_recovery(cond_dir)` overlays `recovered_answers.jsonl`
onto `rows.jsonl` for OPT-IN downstream scoring (`scripts/make_results_table.py
--with-recovery`); it never mutates `rows.jsonl` on disk.

WHAT COUNTS AS "NEEDS RECOVERY" (`needs_recovery`, single source of truth): a truly
empty/whitespace-only `final_answer`, OR one of the placeholder strings the model
sometimes emits when it gives up instead of a genuine answer (`"..."`, `"."`, `".."`).
Row SELECTION (`empty_answer_rows`), the overlay's coverage check
(`load_rows_with_recovery`), and `scripts/compare_cells.py`'s `empty` metric all import
this ONE function so selection and reporting can never disagree. (Fixed bug: this
script used to select on empty-only while `compare_cells.py` counted placeholders as
empty too, so placeholder rows in some cells — up to ~9% in indri+dense visit — were
silently never backfilled while the baseline's true-empty rows were recovered.)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Callable, Optional, Sequence

from agent_search.agent.loop import Step, Task, _extract_answer
from agent_search.agent.policies import AgentPolicy
# The prefill-elicitation MECHANISM (call_prefill/call_plain_ask/prefill_messages_for +
# FORCE_MSG/FALLBACK_MSG + the default tuning knobs) now lives in agent_search.agent.forced_answer,
# shared with the LIVE inline elicitation in agent_search/agent/loop.py's terminal branch (see that
# module's docstring for the "why a shared module" rationale + the loop/SDK-driver asymmetry). This
# script re-exports the names below so existing imports (incl. this file's own tests) keep working
# unchanged — no behavior change here, only the mechanism's HOME module moved.
from agent_search.agent.forced_answer import (
    DEFAULT_FALLBACK_MAX_TOKENS,
    DEFAULT_PREFILL_MAX_TOKENS,
    DEFAULT_SEED,
    DEFAULT_TEMPERATURE,
    FALLBACK_MSG,
    FORCE_MSG,
    call_plain_ask,
    call_prefill,
    elicit_final_answer,
    prefill_messages_for,
)

METHOD = "forced_terminal_prefill_v1"
DEFAULT_CTX_CHARS = 420_000              # ~120k tokens at the chars/3.5 heuristic (documented above)


# --- tolerant rows.jsonl / recovered_answers.jsonl I/O -----------------------------------------

def load_rows_tolerant(rows_path: str) -> list:
    """Parse `rows.jsonl`, skipping unparsable trailing lines (a run still appending). Mirrors
    `evaluation.run_eval._load_rows`'s tolerance, but keeps duplicates as-is (rows.jsonl for a
    finished condition should have none; we don't second-guess a live one)."""
    rows: list = []
    if not rows_path or not os.path.exists(rows_path):
        return rows
    with open(rows_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue   # partial trailing write from a live job; skip, never crash
    return rows


def load_recovered_ids(cond_dir: str) -> set:
    """instance_ids GENUINELY covered by `<cond_dir>/recovered_answers.jsonl` (idempotency).
    An entry whose `recovered_answer` is itself a placeholder/empty (`needs_recovery` true —
    the model prefilled "..." even under forced decoding; real case: dci
    browsecomp_plus_structured__844) does NOT count as coverage: the row stays a recovery
    target on the next pass instead of being suppressed by a fake answer. Sidecar entries are
    never deleted (append-only audit trail); a later genuine recovery for the same id simply
    appends, and `load_rows_with_recovery` only overlays genuine entries."""
    path = os.path.join(cond_dir, "recovered_answers.jsonl")
    ids = set()
    for rec in load_rows_tolerant(path):
        iid = rec.get("instance_id")
        if iid and not needs_recovery(rec.get("recovered_answer")):
            ids.add(iid)
    return ids


def append_recovered(cond_dir: str, record: dict, lock: Optional[threading.Lock] = None) -> None:
    path = os.path.join(cond_dir, "recovered_answers.jsonl")
    line = json.dumps(record) + "\n"
    if lock is not None:
        with lock:
            _append_line(path, line)
    else:
        _append_line(path, line)


def _append_line(path: str, line: str) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(line)
        fh.flush()
        os.fsync(fh.fileno())


# --- condition-dir resolution + dataset/field-profile inference --------------------------------

def resolve_condition_dirs(patterns: Sequence[str]) -> list:
    """Expand glob pattern(s) into condition dirs (each containing a `rows.jsonl`). A pattern may
    match the condition dir itself or its `rows.jsonl` file directly; both resolve to the dir."""
    dirs: set = set()
    for pat in patterns:
        for m in sorted(glob.glob(pat)):
            if os.path.isdir(m):
                if os.path.exists(os.path.join(m, "rows.jsonl")):
                    dirs.add(os.path.normpath(m))
            elif os.path.basename(m) == "rows.jsonl":
                dirs.add(os.path.normpath(os.path.dirname(m)))
    return sorted(dirs)


def infer_dataset_name(cond_dir: str) -> Optional[str]:
    """The registered dataset name found in the condition-dir path (e.g.
    `browsecomp_plus_structured`), by membership rather than a fixed path position — robust to the
    `<root>/agent/<dataset>/<model>/agent_<cond>` layout varying which `<root>` it sits under."""
    from evaluation.datasets import available_datasets
    names = available_datasets()
    parts = os.path.normpath(cond_dir).split(os.sep)
    for part in parts:
        if part in names:
            return part
    return None


def infer_field_profile(cond_dir: str) -> Optional[str]:
    from evaluation.datasets import dataset_field_profile
    ds = infer_dataset_name(cond_dir)
    return dataset_field_profile(ds) if ds else None


def infer_model_name(cond_dir: str) -> Optional[str]:
    """The model directory component (`<root>/agent/<dataset>/<model>/agent_<cond>`), used as a
    fallback default when neither `--model` nor env `MODEL` is set."""
    parent = os.path.dirname(os.path.normpath(cond_dir))
    name = os.path.basename(parent)
    return name or None


# --- prompt reconstruction (see module docstring) ---------------------------------------------

def reconstruct_messages(row: dict, field_profile: Optional[str],
                         ctx_chars: int = DEFAULT_CTX_CHARS) -> list:
    """The message list `AgentPolicy.build_messages` would build TODAY over this row's full stored
    trajectory (FULL `observations`, never the 600-char-capped `trajectory[i]['observation']`),
    plus the forced-terminal-elicitation turn appended at the end. Raises KeyError if the row has
    no `prompt_profile_path` (an older/foreign row shape this tool does not support)."""
    prompt_path = row.get("prompt_profile_path") or (row.get("tool_condition") or "").removeprefix("agent_")
    if not prompt_path:
        raise KeyError(f"row {row.get('instance_id')!r} has neither prompt_profile_path nor tool_condition")
    task = Task(task_id=row.get("instance_id") or "q", query=row.get("question") or "")
    steps = _row_to_steps(row)
    # generate is never called through this policy object — only .build_messages is used, so a
    # stub is fine (and keeps this function import-cheap / offline-safe for --dry-run).
    policy = AgentPolicy(generate=lambda _msgs: "", prompt_path=prompt_path,
                         field_profile=field_profile, ctx_chars=ctx_chars)
    messages = policy.build_messages(task, steps)
    messages.append({"role": "user", "content": f"<tool_response>\n{FORCE_MSG}\n</tool_response>"})
    return messages


def _row_to_steps(row: dict) -> list:
    """Real `agent_search.agent.loop.Step` objects from `row['trajectory']`, with each step's
    observation swapped for the FULL, uncapped text in `row['observations']` (same index, same
    order — see `agent_search/agent/retriever.py::_trajectory_meta`). Falls back to the capped
    `trajectory[i]['observation']` only if `observations` is shorter (defensive; should not happen
    on a well-formed row)."""
    traj = row.get("trajectory") or []
    obs = row.get("observations") or []
    steps = []
    for i, s in enumerate(traj):
        full_obs = obs[i] if i < len(obs) else s.get("observation", "")
        steps.append(Step(name=s.get("action") or "", args=s.get("args") or {},
                          observation=full_obs, raw_output=s.get("raw_output") or ""))
    return steps


def _has_answer_tag(text: str) -> bool:
    """Same case-sensitive `<answer>` check `_extract_answer` itself anchors on (its `rfind`)."""
    return bool(text) and text.rfind("<answer>") >= 0


# --- SINGLE SOURCE OF TRUTH: what counts as "needs recovery" -----------------------------------

PLACEHOLDER_ANSWERS = ("", "...", ".", "..")


def needs_recovery(final_answer: Optional[str]) -> bool:
    """True if `final_answer` needs offline recovery: truly empty/whitespace-only, OR one of the
    placeholder strings the model sometimes emits when it gives up (`"..."`, `"."`, `".."`) instead
    of a genuine answer. This is the ONE place that definition lives — used by row SELECTION here
    (`empty_answer_rows`, and therefore `process_condition_dir`'s `pending` set), by the overlay's
    idempotency/coverage check (`load_rows_with_recovery`), and imported by
    `scripts/compare_cells.py`'s `empty` metric — so a row that gets backfilled and a row that gets
    counted as empty can never disagree (see module docstring: the original bug here was
    `force_answer_backfill.py` selecting on empty-only while `compare_cells.py` counted placeholders
    as empty too, so placeholder rows in some cells were silently never recovered)."""
    return (final_answer or "").strip() in PLACEHOLDER_ANSWERS


# --- model call: vLLM assistant-prefill continuation (primary) + one plain-ask fallback --------

def build_client(api_base: str, api_key: Optional[str] = None):
    """A plain OpenAI-compatible client against `api_base` (a served vLLM, or a real OpenAI-
    compatible endpoint). Deliberately NOT `agent_search.models.backends.openai_compat_generate`
    — that helper's stop-sequence/tag-repair post-processing is tool-call-LOOP-specific, and (more
    importantly here) it has no way to pass vLLM's `continue_final_message`/`add_generation_prompt`
    prefill fields through `extra_body`. Mirrors `scripts/oneshot_rag.py::make_generate`'s plain-
    client construction."""
    from openai import OpenAI
    return OpenAI(base_url=api_base, api_key=api_key or os.environ.get("OPENAI_API_KEY", "EMPTY"))


def elicit_answer(messages: list, client, model: str, *,
                  prefill_max_tokens: int = DEFAULT_PREFILL_MAX_TOKENS,
                  fallback_max_tokens: int = DEFAULT_FALLBACK_MAX_TOKENS,
                  temperature: float = DEFAULT_TEMPERATURE, seed: Optional[int] = DEFAULT_SEED) -> tuple:
    """Thin wrapper over the shared `agent_search.agent.forced_answer.elicit_final_answer` that
    keeps THIS script's original 3-tuple shape (`recovered_answer, n_attempts, raw_continuation`)
    — `n_attempts` is 1 when the primary prefill call worked, 2 when the plain-ask fallback was
    needed (matching the pre-extraction `elicit_answer`'s own contract, so every downstream caller
    and this file's own tests are unaffected by the mechanism's move to `forced_answer.py`)."""
    answer, method_tag, raw = elicit_final_answer(
        messages, client, model, prefill_max_tokens=prefill_max_tokens,
        fallback_max_tokens=fallback_max_tokens, temperature=temperature, seed=seed,
        extract_fn=_extract_answer)
    n_attempts = 1 if method_tag == "prefill" else 2
    return answer, n_attempts, raw


# --- scoring-side integration: opt-in merge overlay --------------------------------------------

def load_rows_with_recovery(cond_dir: str) -> list:
    """`rows.jsonl` rows from `cond_dir`, overlaid with `recovered_answers.jsonl` (if present):
    any row whose `final_answer` `needs_recovery` (empty/whitespace-only OR a placeholder like
    `"..."`) AND has a GENUINE recovered entry (whose `recovered_answer` does not itself
    `needs_recovery`) gets `final_answer` replaced with the recovered span and `recovered=True`;
    every other row gets `recovered=False`. A placeholder recovery never overlays — the row stays
    empty-ish (still shows in empty%, still a future recovery target) rather than becoming a fake
    non-empty answer. Never touches `rows.jsonl` on disk — this is a read-only, in-memory merge
    for OPT-IN scoring (`scripts/make_results_table.py --with-recovery`)."""
    rows = load_rows_tolerant(os.path.join(cond_dir, "rows.jsonl"))
    for r in rows:
        r.setdefault("recovered", False)
    rec_path = os.path.join(cond_dir, "recovered_answers.jsonl")
    if not os.path.exists(rec_path):
        return rows
    recovered_by_id = {}
    for rec in load_rows_tolerant(rec_path):
        iid = rec.get("instance_id")
        # Only GENUINE recoveries are overlay candidates: a sidecar entry whose
        # recovered_answer is itself a placeholder/empty ("...", ".", "..", "") must never
        # become a non-empty final_answer downstream (it would be scored as a real — wrong —
        # answer and vanish from empty%). Such rows stay as-is: still empty-ish, still a
        # future recovery target. Keeping only genuine entries here also makes duplicate
        # sidecar entries per id harmless (a later genuine retry wins over an earlier
        # placeholder one).
        if iid and not needs_recovery(rec.get("recovered_answer")):
            recovered_by_id[iid] = rec
    for r in rows:
        iid = r.get("instance_id")
        rec = recovered_by_id.get(iid)
        if rec is not None and needs_recovery(r.get("final_answer")):
            r["final_answer"] = rec.get("recovered_answer") or ""
            r["recovered"] = True
    return rows


# --- per-cell driver -----------------------------------------------------------------------------

def empty_answer_rows(cond_dir: str) -> list:
    rows = load_rows_tolerant(os.path.join(cond_dir, "rows.jsonl"))
    return [r for r in rows if needs_recovery(r.get("final_answer"))]


def process_condition_dir(cond_dir: str, *, model: Optional[str], api_base: Optional[str],
                          ctx_chars: int, prefill_max_tokens: int, fallback_max_tokens: int,
                          stamp: str, limit: Optional[int] = None, workers: int = 1,
                          dry_run: bool = False, log: Callable[[str], None] = print) -> dict:
    """Process one condition dir end to end; returns a small summary dict for reporting."""
    empty = empty_answer_rows(cond_dir)
    done_ids = load_recovered_ids(cond_dir)
    pending = [r for r in empty if r.get("instance_id") not in done_ids]
    if limit is not None:
        pending = pending[:limit]
    field_profile = infer_field_profile(cond_dir)
    summary = {"cond_dir": cond_dir, "n_empty": len(empty), "n_already_recovered": len(done_ids),
              "n_pending": len(pending), "field_profile": field_profile}
    if dry_run or not pending:
        return summary

    resolved_model = model or os.environ.get("MODEL") or infer_model_name(cond_dir)
    resolved_api_base = api_base or os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE")
    if not resolved_api_base:
        raise RuntimeError("no --api-base and no OPENAI_BASE_URL/OPENAI_API_BASE env set")
    client = build_client(resolved_api_base)
    lock = threading.Lock()
    n_recovered = 0
    n_skipped_overflow = 0
    n_failed = 0
    n_placeholder = 0

    def _work(row: dict) -> tuple:
        """Recover ONE row. Returns `(iid, status, payload)`: status `"ok"` with payload
        `(answer, n_attempts, raw)`, or `"overflow"` with payload the FINAL (smallest)
        `ctx_chars` tried. Mirrors `AgentPolicy.propose()`'s overflow handling
        (`agent_search/agent/policies.py`): the chars/3.5 budget can overshoot the model's
        true token window on token-dense content, so on a "maximum context length" error
        shrink the reconstructed window 15% and retry, up to 3 shrinks (4 attempts total);
        a row that STILL overflows is skipped — no sidecar record, just a log line — so one
        pathological row can never kill the whole pass again (job 28673993 crashed exactly
        this way: one vLLM 400 propagated uncaught through the ThreadPoolExecutor). Any
        OTHER exception propagates to `_consume`, which logs it and counts it as a per-row
        failure without stopping the pass."""
        iid = row.get("instance_id")
        ctx = ctx_chars
        for shrink in range(4):
            messages = reconstruct_messages(row, field_profile, ctx_chars=ctx)
            try:
                answer, n_attempts, raw = elicit_answer(
                    messages, client, resolved_model,
                    prefill_max_tokens=prefill_max_tokens, fallback_max_tokens=fallback_max_tokens)
                return iid, "ok", (answer, n_attempts, raw)
            except Exception as e:
                if "maximum context length" not in str(e):
                    raise
                if shrink < 3:
                    ctx = int(ctx * 0.85)
        return iid, "overflow", ctx

    def _record(iid, answer, n_attempts, raw):
        append_recovered(cond_dir, {
            "instance_id": iid, "recovered_answer": answer, "n_attempts": n_attempts,
            "raw_continuation": raw, "recovered_at_iso": stamp, "method": METHOD,
        }, lock=lock)

    def _consume(row: dict, result_fn: Callable[[], tuple]) -> None:
        """Fold one row's outcome into the counters. NEVER raises: overflow skips and any
        other per-row exception are logged + counted so the pass always continues."""
        nonlocal n_recovered, n_skipped_overflow, n_failed, n_placeholder
        try:
            iid, status, payload = result_fn()
        except Exception as e:
            n_failed += 1
            log(f"   [{cond_dir}] ROW FAILED instance_id={row.get('instance_id')!r}: "
                f"{type(e).__name__}: {e}")
            return
        if status == "overflow":
            n_skipped_overflow += 1
            log(f"   [{cond_dir}] SKIPPED context-overflow-after-3-shrinks "
                f"instance_id={iid!r} final_ctx_chars={payload}")
            return
        _record(iid, *payload)   # always appended — audit trail of the attempt
        answer = payload[0]
        if needs_recovery(answer):
            # The model prefilled a placeholder/empty even under forced decoding (real case:
            # dci browsecomp_plus_structured__844). NOT a success: the sidecar entry is kept
            # for audit, but `load_recovered_ids`/`load_rows_with_recovery` ignore it, so the
            # row stays empty downstream and is re-attempted on the next pass.
            n_placeholder += 1
            log(f"   [{cond_dir}] PLACEHOLDER RECOVERY (not counted as recovered) "
                f"instance_id={iid!r} answer={answer!r}")
            return
        n_recovered += 1

    if workers <= 1:
        for row in pending:
            _consume(row, lambda row=row: _work(row))
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(_work, row): row for row in pending}
            for fut in as_completed(futs):
                _consume(futs[fut], fut.result)
                done = n_recovered + n_skipped_overflow + n_failed
                if done % 10 == 0:
                    log(f"   [{cond_dir}] {done}/{len(pending)} processed "
                        f"({n_recovered} recovered)")
    summary["n_recovered"] = n_recovered
    summary["n_skipped_overflow"] = n_skipped_overflow
    summary["n_failed"] = n_failed
    summary["n_placeholder_recoveries"] = n_placeholder
    return summary


# --- CLI -----------------------------------------------------------------------------------------

def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cond_dirs", nargs="+",
                    help="condition dir glob pattern(s), e.g. "
                         "'runs/_headline_validation/agent/browsecomp_plus_structured/*/agent_research_indri'")
    ap.add_argument("--model", default=None, help="default: env MODEL, else the row's model dir")
    ap.add_argument("--api-base", default=None, help="default: env OPENAI_BASE_URL")
    ap.add_argument("--stamp", default=None, help="recovered_at_iso; default: now (UTC)")
    ap.add_argument("--prefill-max-tokens", type=int, default=DEFAULT_PREFILL_MAX_TOKENS,
                    help="max_tokens for the primary forced-continuation call")
    ap.add_argument("--fallback-max-tokens", type=int, default=DEFAULT_FALLBACK_MAX_TOKENS,
                    help="max_tokens for the ONE plain-ask fallback call")
    ap.add_argument("--ctx-chars", type=int, default=DEFAULT_CTX_CHARS)
    ap.add_argument("--limit", type=int, default=None, help="cap rows processed per cell (debug)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true",
                    help="report counts + one reconstructed sample prompt; no model call")
    args = ap.parse_args(argv)

    stamp = args.stamp or datetime.now(timezone.utc).isoformat()
    cond_dirs = resolve_condition_dirs(args.cond_dirs)
    if not cond_dirs:
        print(f"ERROR: no condition dirs matched {args.cond_dirs!r}", file=sys.stderr)
        return 1

    print(f">> {len(cond_dirs)} condition dir(s) matched", file=sys.stderr)
    summaries = []
    sample_printed = False
    total_pending = 0
    for cond_dir in cond_dirs:
        summary = process_condition_dir(
            cond_dir, model=args.model, api_base=args.api_base, ctx_chars=args.ctx_chars,
            prefill_max_tokens=args.prefill_max_tokens, fallback_max_tokens=args.fallback_max_tokens,
            stamp=stamp, limit=args.limit, workers=args.workers, dry_run=args.dry_run,
            log=lambda m: print(m, file=sys.stderr))
        summaries.append(summary)
        total_pending += summary["n_pending"]
        print(f"   {cond_dir}: empty={summary['n_empty']} already_recovered={summary['n_already_recovered']} "
             f"pending={summary['n_pending']} field_profile={summary['field_profile']!r}"
             + (f" recovered={summary.get('n_recovered')} "
                f"placeholder_recoveries={summary.get('n_placeholder_recoveries', 0)} "
                f"skipped_overflow={summary.get('n_skipped_overflow', 0)} "
                f"failed={summary.get('n_failed', 0)}" if not args.dry_run else ""),
             file=sys.stderr)
        if args.dry_run and not sample_printed and summary["n_pending"] > 0:
            empty = empty_answer_rows(cond_dir)
            done_ids = load_recovered_ids(cond_dir)
            sample_row = next(r for r in empty if r.get("instance_id") not in done_ids)
            field_profile = summary["field_profile"]
            base_messages = reconstruct_messages(sample_row, field_profile, ctx_chars=args.ctx_chars)
            messages = prefill_messages_for(base_messages)
            print("\n=== DRY-RUN sample reconstructed prompt, INCLUDING the forced-prefill "
                 f"assistant turn (instance_id={sample_row.get('instance_id')!r}, cond_dir={cond_dir!r}) ===")
            for m in messages:
                print(f"--- role={m['role']} ({len(m['content'])} chars) ---")
                print(m["content"][:2000])
            print("(primary call would pass extra_body={'add_generation_prompt': False, "
                 "'continue_final_message': True}, stop=['</answer>'] over the messages above)")
            sample_printed = True

    n_empty = sum(s["n_empty"] for s in summaries)
    n_already = sum(s["n_already_recovered"] for s in summaries)
    n_recovered = sum(s.get("n_recovered", 0) for s in summaries)
    n_skipped_overflow = sum(s.get("n_skipped_overflow", 0) for s in summaries)
    n_failed = sum(s.get("n_failed", 0) for s in summaries)
    n_placeholder = sum(s.get("n_placeholder_recoveries", 0) for s in summaries)
    print(f"\n>> TOTAL: {n_empty} empty-answer rows across {len(cond_dirs)} cells, "
         f"{n_already} already recovered, {total_pending} pending"
         + ("" if args.dry_run else f", {n_recovered} recovered this run, "
            f"{n_placeholder} placeholder recoveries (not counted), "
            f"{n_skipped_overflow} skipped (context overflow), {n_failed} failed"),
         file=sys.stderr)
    # Per-row resilience contract: overflow skips, placeholder recoveries, and isolated
    # failures are logged + counted but do NOT fail the job (they resume cleanly on a rerun —
    # genuinely-recovered ids are skipped). Only a SYSTEMIC failure rate (>20% of processed
    # rows raising non-overflow errors — e.g. the server is down or a code bug) makes the
    # exit code nonzero.
    n_processed = n_recovered + n_skipped_overflow + n_failed + n_placeholder
    if not args.dry_run and n_processed and n_failed > 0.2 * n_processed:
        print(f">> ERROR: {n_failed}/{n_processed} processed rows failed (>20%) — "
             f"treating as a systemic failure", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
