"""An answer is read from the visible turn only: no reasoning (including a lone </think> whose
opening tag the chat template supplied) and no tool call. Each case is a shape seen in real runs."""
from agent_search.agent.answer_text import clean_answer, in_open_think, judge_text, visible_text
from agent_search.agent.backbone.text import _repair_open_tag
from agent_search.evaluation.llm_judge import judge_answer_detail

# OpenSeeker: the template opened <think>; the model names the answer tag, then searches again
OPENSEEKER = ("Now we need to enclose final answer in <answer> tags.\n</think>\n\n<tool_call>\n"
              '{"name": "search", "arguments": {"query": "Darlene Kittle"}}\n</tool_call>')
# Tongyi forced answer at the step budget: reasoning, then a tool call, after the <answer> prefill
FORCED = ("Now we have hit the step budget, so we must answer now. It is likely Tony Goldwyn.\n"
          '</think>\n\n<tool_call>\n{"name": "search", "arguments": {"query": "x"}}\n</tool_call>')


def test_visible_text_drops_a_lone_closing_think():
    assert visible_text("reasoning here</think>\n\nThe answer is X.") == "\n\nThe answer is X."
    assert visible_text("<think>a</think>B") == "B"
    assert visible_text("no reasoning at all") == "no reasoning at all"


def test_clean_answer_keeps_only_the_answer():
    assert clean_answer(OPENSEEKER) == ""
    assert clean_answer("Thinking...</think>\nGabriel Estaba") == "Gabriel Estaba"
    assert clean_answer("Carl Herrera <tool_call>{\"name\": \"search\"") == "Carl Herrera"


def test_judge_text_falls_back_to_the_reasoning_without_the_call():
    assert judge_text(FORCED).startswith("Now we have hit the step budget")
    assert "<tool_call>" not in judge_text(FORCED)
    assert judge_text("r</think>\nTony Goldwyn") == "Tony Goldwyn"


def test_no_closing_answer_tag_is_fabricated_after_a_tool_call():
    # generation stopped at </tool_call>; the prose mention of <answer> must stay open
    assert _repair_open_tag(OPENSEEKER).endswith("</tool_call>")
    # a real open answer is still closed
    assert _repair_open_tag("<answer>Galati") == "<answer>Galati</answer>"


def test_the_judge_reads_the_cleaned_answer():
    seen = []
    def judge(prompt):
        seen.append(prompt)
        return '{"extracted_final_answer": "Tony Goldwyn", "reasoning": "r", "correct": "yes", "confidence": 100}'
    out = judge_answer_detail("q", "Tony Goldwyn", FORCED.replace("It is likely", "It is likely"), judge)
    assert out["judge_correct"] is True
    assert "<tool_call>" not in seen[0] and "</think>" not in seen[0]
    # an answer written only inside the reasoning is judged from the reasoning, never with the call
    judge_answer_detail("q", "g", OPENSEEKER, judge)
    assert "<tool_call>" not in seen[-1] and "Now we need to enclose" in seen[-1]


def test_exact_match_reads_the_answer_without_a_tool_call():
    from agent_search.evaluation.doc_scoring import score_answer
    out = score_answer("reasoning</think>\n<answer>Galati</answer>", "Galati", ["... Galati ..."])
    assert out["answer_em"] == 1.0 and out["grounded"]
    out = score_answer('Galati <tool_call>{"name": "search"}</tool_call>', "Galati", ["Galati"])
    assert out["answer_em"] == 1.0


# Tongyi quoting the Sieve prompt's answer format while still thinking: generation stops at the
# </answer> stop string, mid-thought
CUT = ("<think>\nThus answer: 20,104. The instructions say: \"Give ONLY the short answer span "
       "inside `<answer>...")


def test_a_turn_cut_inside_reasoning_has_no_answer():
    assert in_open_think(CUT) and not in_open_think("<think>a</think><answer>X</answer>")
    assert visible_text(CUT) == "" and clean_answer(CUT) == ""
    # the backend must not close the quoted tag: that turned "..." into the final answer
    assert _repair_open_tag(CUT) == CUT


def test_the_backend_continues_a_turn_cut_inside_reasoning():
    from types import SimpleNamespace as NS
    from agent_search.agent.backbone.openai_chat import openai_compat_generate
    replies = [NS(content=CUT, finish="stop", stop="</answer>"),
               NS(content="` tags.\nSo I answer now.\n</think>\n\n<answer>20,104", finish="stop", stop="</answer>")]
    calls = []

    class Completions:
        def create(self, **kw):
            calls.append(kw)
            r = replies[len(calls) - 1]
            return NS(choices=[NS(message=NS(content=r.content), finish_reason=r.finish, stop_reason=r.stop)],
                      usage=NS(prompt_tokens=10, completion_tokens=20))

    gen = openai_compat_generate("m", base_url="http://x/v1", client=NS(chat=NS(completions=Completions())))
    out = gen([{"role": "user", "content": "q"}])
    assert len(calls) == 2
    assert calls[1]["messages"][-1] == {"role": "assistant", "content": CUT + "</answer>"}
    assert calls[1]["extra_body"]["continue_final_message"] is True
    assert out.endswith("</think>\n\n<answer>20,104</answer>")
    assert clean_answer(out) == "<answer>20,104</answer>"


def test_a_cut_turn_does_not_end_the_episode():
    from agent_search.agent.loop import _final_locations
    assert _final_locations(CUT + "</answer>", "", {}) is None
