"""DIVER's judge prompt and free-text verdict, next to the Appendix F JSON judge."""
from types import SimpleNamespace

from agent_search.evaluation import llm_judge as J


def test_divers_prompt_is_used_and_the_verdict_line_is_read_after_the_think_block():
    seen = {}

    def gen(prompt):
        seen["prompt"] = prompt
        return ("<think>long deliberation... correct: no (tentative)</think>\n"
                "extracted_final_answer: Treaty of Guadalupe Hidalgo\n[correct_answer]: Guadalupe Hidalgo\n"
                "reasoning: string variation of the same treaty.\ncorrect: yes\nconfidence: 100")
    d = J.judge_answer_detail("which treaty?", "Guadalupe Hidalgo", "The Treaty of Guadalupe Hidalgo.", gen, prompt="diver")
    assert d["judge_correct"] is True and d["judge_extracted"] == "Treaty of Guadalupe Hidalgo"
    assert "allowing the extracted_final_answer to be string variations" in seen["prompt"]
    assert "[correct_answer]: Guadalupe Hidalgo" in seen["prompt"]
    assert "Return ONLY a compact JSON" not in seen["prompt"]


def test_a_reply_without_a_verdict_line_is_an_error_not_a_no():
    d = J.judge_answer_detail("q", "gold", "some answer", lambda p: "<think>...</think> I cannot decide.", prompt="diver")
    assert d["judge_correct"] is None and d["judge_error"]


def test_bcp_prompt_is_the_default_and_unchanged():
    seen = {}

    def gen(prompt):
        seen["prompt"] = prompt
        return '{"extracted_final_answer": "x", "reasoning": "", "correct": "no", "confidence": 100}'
    d = J.judge_answer_detail("q", "gold", "x", gen)
    assert d["judge_correct"] is False and "Return ONLY a compact JSON" in seen["prompt"]


def test_a_free_text_served_judge_gets_its_budget_and_no_json_mode():
    calls = []

    def create(**kw):
        calls.append(kw)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="correct: yes"))])
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    gen = J.make_judge("Qwen/Qwen3-30B-A3B-Thinking-2507", api_base="http://x", client=client, max_tokens=8192, json_mode=False)
    assert gen("p") == "correct: yes"
    assert calls[0]["max_tokens"] == 8192 and calls[0]["temperature"] == 0.0 and "response_format" not in calls[0]


def test_a_tagged_judge_keeps_its_verdicts_next_to_the_default_ones(tmp_path):
    import json
    rd = tmp_path / "cell"
    rd.mkdir()
    rows = [{"instance_id": f"q{i}", "question": "q", "gold_answer": "gold", "final_answer": "gold" if i % 2 else "other",
             "judge_correct": True} for i in range(10)]
    (rd / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (rd / "results.json").write_text("{}")
    calls = []

    def gen(prompt):
        calls.append(prompt)
        return "correct: yes" if "[response]: gold" in prompt else "correct: no"
    s = J.judge_run_dir(str(rd), gen, judge_model="served-judge", prompt="diver", tag="diver", workers=3)
    assert s["n_judged"] == 10 and s["n_correct"] == 5 and s["judge_prompt"].startswith("DIVER")
    assert (rd / "judge_summary_diver.json").exists() and not (rd / "judge_summary.json").exists()
    out = [json.loads(l) for l in (rd / "rows.jsonl").read_text().splitlines()]
    assert [r["instance_id"] for r in out] == [f"q{i}" for i in range(10)]      # order kept across the pool
    assert all(r["judge_correct"] is True for r in out)                         # the default verdicts untouched
    assert [r["judge_correct_diver"] for r in out] == [bool(i % 2) for i in range(10)]
    assert len(calls) == 5          # the five exact matches short-circuit without a judge call
    # idempotent: a second pass grades nothing new
    s2 = J.judge_run_dir(str(rd), gen, judge_model="served-judge", prompt="diver", tag="diver", workers=3)
    assert s2["n_newly_judged"] == 0 and len(calls) == 5
