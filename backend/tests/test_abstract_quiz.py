"""Abstract quiz (v3 story gate): the transcript of the visible manga, the three quiz calls, strict grading with the
citation cap, failed claims, the never-raise contract and the revision prompt. Engines are small local fakes."""

from __future__ import annotations

import asyncio
import json

import pytest

from engines.base import BaseEngine, TextResponse
from services import abstract_quiz as aq
from services import boards
from services.storyboarder_v2 import Bubble, Cast, CastMember, PanelV2, StoryboardV2, _v3_rules


class Fake(BaseEngine):
    """Text engine fake: ``route(prompt)`` returns a dict (sent as JSON), a str, or an exception to raise."""

    name = "fake"
    supports_image_generation = False
    supports_vision = False

    def __init__(self, route):
        super().__init__(api_key="test-key-not-real", base_url="http://fake.invalid", model="fake")
        self.route = route
        self.calls = []

    @property
    def prompts(self) -> list[str]:
        return [prompt for prompt, _ in self.calls]

    async def generate_text(self, messages, config=None):
        self.calls.append((messages[-1].content, config))
        await asyncio.sleep(0)
        reply = self.route(messages[-1].content)
        if isinstance(reply, BaseException):
            raise reply
        content = reply if isinstance(reply, str) else json.dumps(reply, ensure_ascii=False)
        return TextResponse(content=content, model="fake", usage={"prompt_tokens": 100, "completion_tokens": 10})

    async def generate_text_stream(self, messages, config=None):
        raise NotImplementedError


def queue(*replies):
    """A route that answers with ``replies`` in order."""
    pending = list(replies)
    return lambda prompt: pending.pop(0)


def items_of(prompt: str, marker: str) -> list[dict]:
    """The JSON list a quiz prompt ends with (QUESTIONS: / ITEMS:)."""
    return json.loads(prompt.rsplit(marker, 1)[1])


CLAIMS = [
    {"id": 0, "text": "The Transformer is based solely on attention, dispensing with recurrence and convolutions.",
     "quote": "based solely on attention mechanisms, dispensing with recurrence and convolutions entirely"},
    {"id": 1, "text": "28.4 BLEU on WMT 2014 English-to-German, over 2 BLEU above the existing best results including ensembles.",
     "quote": "Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles, by over 2 BLEU."},
    {"id": 2, "text": "A new single-model state-of-the-art BLEU of 41.8 on WMT 2014 English-to-French after 3.5 days on eight GPUs.",
     "quote": "our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs"},
]


def panel(n: int, kind: str = "scene", board=None, dialogue=(), narration: str = "", **kw) -> PanelV2:
    return PanelV2(n=n, chapter=kw.pop("chapter", "how"), title=kw.pop("title", f"Panel {n}"), characters=kw.pop("characters", ["papi", "kumo"]),
                   scene="A kitchen.", visual="Papi points at the board.", dialogue=[Bubble(s, t) for s, t in dialogue],
                   narration=narration, kind=kind, board=board, **kw)


def make_sb(version: int = 3, language: str = "en-US") -> StoryboardV2:
    panels = [
        panel(1, narration="Recurrent networks read a sentence\none word after another.",
              dialogue=[("kumo", "Ugh, one step at a time takes forever!")], labels=["LABEL-XYZ"],
              detail="SECRET DETAIL: 99.9 points only for the reader", claims=[0]),
        panel(2, "board", narration="BLEU measures how close a translation is to human ones. Higher is better.",
              dialogue=[("nezu", "Show me the numbers."), ("papi", "Even the ensembles lose.")], claims=[1],
              board={"kind": "bar", "title": "WMT 2014 English→German (BLEU)", "unit": "BLEU", "higher_is_better": True,
                     "bars": [{"label": "ConvS2S ensemble", "paper_label": "ConvS2S Ensemble [9]", "value": "26.36",
                               "quote": "ConvS2S Ensemble [9] 26.36 41.29", "baseline": True},
                              {"label": "Transformer (big)", "value": "28.4", "quote": "Transformer (big) 28.4 41.8", "highlight": True}],
                     "source": "Table 2"}),
        panel(3, "board", dialogue=[("papi", "Attention links any two words in one step.")],
              board={"kind": "table", "title": "Linking two words", "subtitle": "n = sentence length",
                     "columns": ["Layer", "Sequential steps"], "rows": [{"cells": ["Self-attention", "O(1)"], "quote": "Self-Attention O(1)"},
                                                                         ["Recurrent", "O(n)"]], "highlight_rows": [0]}),
        panel(4, "board", board={"kind": "flow", "title": "The Transformer", "steps": ["Words + positions", "Encoder", "Decoder", "Output"],
                                 "highlight": [1], "note": "each layer: attention + feed-forward"}),
        panel(5, "board", board={"kind": "compare", "title": "Old vs new", "left": {"label": "RNN", "items": ["one word at a time"]},
                                 "right": {"label": "Transformer", "items": ["all words at once", "trains faster"]}}),
        panel(6, "board", board={"kind": "links", "title": "What does it look at?", "tokens": ["The", "animal", "was", "tired", "it"],
                                 "from": 4, "weights": [0.01, 0.6, 0.1, 0.35, 0.0], "illustrative": True}),
        panel(7, "board", dialogue=[("nezu", "Fine. The numbers convinced me.")],
              board={"kind": "list", "title": "What the paper showed",
                     "items": ["Only attention: no recurrence", "28.4 BLEU English→German, ensembles included"]}),
        panel(8, dialogue=[("kumo", "I did all the steps at once!")],
              board={"kind": "bar", "title": "STRAY BOARD", "bars": [{"label": "x", "value": "55.5"}]}),   # a scene has no board
    ]
    return StoryboardV2(title="Attention", summary="", theme="kumomo", language=language, chapters=[], panels=panels,
                        brief={}, card={"problem": "CARD TEXT 77.7"}, version=version, abstract_claims=[dict(c) for c in CLAIMS],
                        running_example="Kumo follows a long recipe one step at a time.")


def make_cast() -> Cast:
    return Cast(theme="kumomo", style="Watercolor manga.", members=[
        CastMember("papi", "mentor", "a dog professor", "patient"),
        CastMember("kumo", "student", "a cloud creature", "curious"),
        CastMember("nezu", "skeptic", "a hedgehog", "dry"),
    ])


# ----------------------------------------------------------------------------- transcript

def test_transcript_shows_the_visible_text_in_reading_order():
    text = aq.transcript(make_sb())
    blocks = text.split("\n\n")
    assert [b.splitlines()[0] for b in blocks] == [f"Panel {n}:" for n in range(1, 9)]
    assert blocks[0].splitlines()[1:] == ["Narration: Recurrent networks read a sentence one word after another.",
                                          "kumo: Ugh, one step at a time takes forever!"]
    bar = blocks[1]
    assert bar.index("Narration:") < bar.index("nezu: Show me") < bar.index("papi: Even") < bar.index("Board (bar chart)")
    assert bar.splitlines()[4:] == ["Board (bar chart): WMT 2014 English→German (BLEU)",
                                    "- ConvS2S ensemble: 26.36",
                                    "- Transformer (big): 28.4 [highlighted] [gap to ConvS2S ensemble: +2.04]",   # the badge boards draws
                                    "Note: BLEU · higher is better · +2.04 vs ConvS2S ensemble",                  # its footnote
                                    "Source: Table 2"]
    assert blocks[2].splitlines()[2:] == ["Board (table): Linking two words", "Subtitle: n = sentence length",
                                          "Columns: Layer | Sequential steps", "- Self-attention | O(1) [highlighted]",
                                          "- Recurrent | O(n)"]
    assert "- Words + positions → Encoder [highlighted] → Decoder → Output" in blocks[3]
    assert "Note: each layer: attention + feed-forward" in blocks[3]
    assert "- RNN (•): one word at a time" in blocks[4] and "- Transformer (•): all words at once; trains faster" in blocks[4]
    assert "- Words: The animal was tired it" in blocks[5]
    # boards.LINK_STRONG / LINK_SHOWN on the normalized weights: 0.6 and 0.35 are drawn strong, 0.1 thin, 0.01 not at all
    assert '- Links from "it": strong to "animal", "tired"; weak to "was"' in blocks[5]
    assert "Note: illustration (example weights)" in blocks[5]
    assert "- ✓ 28.4 BLEU English→German, ensembles included" in blocks[6]
    assert blocks[7] == "Panel 8:\nkumo: I did all the steps at once!"


def test_transcript_never_shows_reader_only_text():
    text = aq.transcript(make_sb())
    # the detail and the card are reader-only extras; quotes and paper labels are never drawn; a scene panel has no
    # board; v3 art is wordless, so its labels are not drawn either
    for hidden in ("SECRET DETAIL", "99.9", "CARD TEXT", "77.7", "ConvS2S Ensemble [9]", "41.29", "STRAY BOARD", "55.5",
                   "LABEL-XYZ", "Panel 1 title"):
        assert hidden not in text


def test_transcript_of_a_v2_storyboard_keeps_its_drawn_labels():
    assert "Label in the picture: LABEL-XYZ" in aq.transcript(make_sb(version=2))


def test_transcript_zh_footnotes_follow_the_board():
    sb = make_sb(language="zh-CN")
    sb.panels[1].board["title"] = "英德翻译"
    text = aq.transcript(sb)
    assert "Board (bar chart): 英德翻译" in text and "Note: BLEU · 越高越好" in text
    assert "Note: 示意图（权重为举例）" in text
    sb.panels[0].narration, sb.panels[0].dialogue = "", []
    assert aq.transcript(sb).startswith("Panel 1:\n(no text)\n\nPanel 2:")


def test_transcript_lists_boards_as_render_board_draws_them():
    sb = make_sb()
    # services.boards normalizes before drawing: aliases resolve, a bar without a number is dropped, a bar chart prints
    # no note (its footnote is unit and direction)
    sb.panels[1].board = {"type": "bar_chart", "title": "Alias chart", "note": "a bar chart draws no note",
                          "data": [{"name": "ConvS2S", "score": 26.36, "rival": "yes"}, {"name": "Ours", "score": "28.4", "ours": "yes"},
                                   {"name": "Broken", "score": "n/a"}]}
    # "illustrative" defaults to true and its tag is printed with the note; percentages work like 0-1 weights once
    # normalize_board has put them on the scale of boards.LINK_STRONG / LINK_SHOWN (60 strong, 27 and 8 thin, 1 not drawn)
    links = {k: v for k, v in sb.panels[5].board.items() if k != "illustrative"}
    sb.panels[5].board = {**links, "weights": [1, 60, 8, 27, 3], "note": "thicker = stronger"}
    sb.panels[2].board = {}                                  # nothing to draw: lettering draws a scene
    sb.panels[3].board = {"kind": "pie", "title": "PIE"}     # render_board cannot draw it
    blocks = aq.transcript(sb).split("\n\n")
    assert blocks[1].splitlines()[4:] == ["Board (bar chart): Alias chart", "- ConvS2S: 26.36",
                                          "- Ours: 28.4 [highlighted] [gap to ConvS2S: +2.04]", "Note: higher is better · +2.04 vs ConvS2S"]
    assert blocks[5].splitlines()[-3:] == ['- Links from "it": strong to "animal"; weak to "was", "tired"', "Note: thicker = stronger",
                                          "Note: illustration (example weights)"]
    assert "draws no note" not in blocks[1]
    assert isinstance(boards.LINK_STRONG, float) and isinstance(boards.LINK_SHOWN, float)   # the scale the transcript reads
    assert blocks[2] == "Panel 3:\npapi: Attention links any two words in one step." and blocks[3] == "Panel 4:\n(no text)"


def test_transcript_bar_footnote_and_badge_follow_the_chart():
    sb = make_sb()
    cost = {"kind": "bar", "title": "Training cost (FLOPs)", "subtitle": "lower is better",
            "bars": [{"label": "GNMT + RL", "value": "2.3 · 10^19", "baseline": True},
                     {"label": "Transformer (big)", "value": "1.4 · 10^19", "highlight": True}]}
    sb.panels[1].board = cost
    badge = boards.delta_badge(cost)["text"]
    assert aq.transcript(sb).split("\n\n")[1].splitlines()[4:] == [
        "Board (bar chart): Training cost (FLOPs)", "Subtitle: lower is better", "- GNMT + RL: 2.3 · 10^19",
        f"- Transformer (big): 1.4 · 10^19 [highlighted] [gap to GNMT + RL: {badge}]",   # the subtitle already says the direction
        f"Note: {badge} vs GNMT + RL"]
    sb.panels[1].board = {"kind": "bar", "title": "Error rate", "unit": "%", "higher_is_better": False,
                          "bars": [{"label": "Ours", "value": "4.9%", "highlight": True}]}   # one bar: no badge
    assert aq.transcript(sb).split("\n\n")[1].splitlines()[4:] == ["Board (bar chart): Error rate", "- Ours: 4.9% [highlighted]",
                                                                    "Note: unit: % · lower is better"]


def test_transcript_reads_the_raw_board_without_services_boards(monkeypatch):
    def broken(spec):
        raise RuntimeError("board bug")

    monkeypatch.setattr(boards, "normalize_board", broken)   # a bug over there must not take the gate down
    assert "- Transformer (big): 28.4 [highlighted]\nNote: BLEU · higher is better\nSource: Table 2" in aq.transcript(make_sb())
    monkeypatch.setattr(aq, "_optional", lambda module: None)   # the module cannot be imported at all
    text = aq.transcript(make_sb())
    assert "- Transformer (big): 28.4 [highlighted]\n" in text and "- Self-attention | O(1) [highlighted]" in text


# ----------------------------------------------------------------------------- questions

async def test_make_questions_validates_and_retries_once():
    first = {"questions": [
        {"id": "a", "claim_ids": [0], "question": "What is the new architecture based on?", "answer": "solely on attention"},
        {"id": "b", "claim_ids": [7], "question": "Unknown claim?", "answer": "x"},
        {"id": "c", "claim_ids": ["C1"], "question": "What BLEU on WMT 2014 English-to-German?", "answer": ""},
        {"id": "d", "claim_ids": [0], "question": "what is the new architecture based on?", "answer": "attention"},
    ]}
    second = {"questions": [{"id": f"x{i}", "claim_ids": [f"C{i % 3}"], "question": f"Question {i}?", "answer": f"Answer {i}"}
                            for i in range(7)]}
    engine = Fake(queue(first, second))
    questions = await aq.make_questions(engine, CLAIMS, "ABSTRACT TEXT 123")
    assert [q["id"] for q in questions] == [f"Q{i}" for i in range(1, 8)]
    assert questions[0] == {"id": "Q1", "claim_ids": [0], "question": "Question 0?", "answer": "Answer 0"}
    assert {i for q in questions for i in q["claim_ids"]} == {0, 1, 2}
    assert len(engine.calls) == 2
    assert engine.prompts[0].startswith("Write a short-answer quiz on a research paper's ABSTRACT.")
    assert "ABSTRACT TEXT 123" in engine.prompts[0] and CLAIMS[1]["quote"] in engine.prompts[0]
    assert "6-12 questions" in engine.prompts[0]   # three claims: 2 per claim, never padded to 8
    assert "REJECTED" in engine.prompts[1] and "claims without a question: 1, 2" in engine.prompts[1]
    config = engine.calls[0][1]
    assert config.temperature == 0 and config.response_format == {"type": "json_object"} and config.thinking is False


async def test_make_questions_rejects_invented_gold_numbers_and_leaked_answers():
    good = [{"claim_ids": [0], "question": f"What is the Transformer based on ({i})?", "answer": "only attention"} for i in range(4)] \
        + [{"claim_ids": [1], "question": "What BLEU did it reach on WMT 2014 English-to-German?", "answer": "28.4 BLEU"}]
    first = {"questions": good + [
        {"claim_ids": [1], "question": "By how much did it beat GNMT?", "answer": "by 3.8 BLEU"},               # 3.8 is nowhere
        {"claim_ids": [2], "question": "Did it reach 41.8 BLEU on English-to-French?", "answer": "41.8 BLEU"},  # gives it away
    ]}
    second = {"questions": good + [{"claim_ids": [2], "question": "What BLEU on English-to-French?", "answer": "41.8 BLEU, single model"},
                                   {"claim_ids": [2], "question": "How long did it train?", "answer": "3.5 days on eight GPUs"}]}
    engine = Fake(queue(first, second))
    questions = await aq.make_questions(engine, CLAIMS)
    assert [q["question"] for q in questions][-2:] == ["What BLEU on English-to-French?", "How long did it train?"]
    retry = engine.prompts[1]
    assert "claims without a question: 2" in retry
    assert '"By how much did it beat GNMT?" has numbers in its answer that are not in the claims or the abstract (3.8)' in retry
    assert '"Did it reach 41.8 BLEU on English-to-French?" contains its own answer' in retry


async def test_make_questions_trims_long_replies_without_dropping_a_claim():
    reply = {"questions": [{"claim_ids": [0], "question": f"Q about claim 0 number {i}?", "answer": "a"} for i in range(13)]
             + [{"claim_ids": [1], "question": "Claim 1?", "answer": "b"}, {"claim_ids": [2], "question": "Claim 2?", "answer": "c"}]}
    questions = await aq.make_questions(Fake(queue(reply)), CLAIMS)
    assert len(questions) == 12
    assert [q["question"] for q in questions][-2:] == ["Claim 1?", "Claim 2?"]


async def test_make_questions_rejects_yes_no_questions():
    base = [{"claim_ids": [c], "question": f"What does claim {c} say ({k})?", "answer": f"fact {c}"} for c in (0, 1) for k in range(3)]
    first = {"questions": base + [{"claim_ids": [2], "question": "Is it a single-model state of the art on English-to-French?",
                                   "answer": "Yes, it is."}]}
    second = {"questions": base + [{"claim_ids": [2], "question": "Does it work beyond translation, and on which task?",
                                    "answer": "Yes, on English constituency parsing with large and limited data"}]}   # more than a yes
    engine = Fake(queue(first, second))
    questions = await aq.make_questions(engine, CLAIMS)
    assert questions[-1]["answer"].startswith("Yes, on English constituency parsing") and len(questions) == 7
    assert "No yes/no or multiple-choice questions" in engine.prompts[0]
    assert '"Is it a single-model state of the art on English-to-French?" is a yes/no question' in engine.prompts[1]


async def test_make_questions_keeps_the_first_quiz_when_the_retry_fails():
    first = {"questions": [{"claim_ids": [0], "question": f"What is it based on ({i})?", "answer": "only attention"} for i in range(6)]}
    questions = await aq.make_questions(Fake(queue(first, RuntimeError("down"))), CLAIMS)   # claims 1 and 2 untested: retried
    assert len(questions) == 6 and {i for q in questions for i in q["claim_ids"]} == {0}
    with pytest.raises(RuntimeError):   # nothing usable at all: the caller's error (run_quiz records it)
        await aq.make_questions(Fake(queue(RuntimeError("down"))), CLAIMS)


ABSTRACT_3 = ("Recurrent models read one word at a time. We propose the Transformer, based solely on attention mechanisms. "
              "Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best "
              "results, including ensembles, by over 2 BLEU.")


def test_quiz_prompt_asks_for_every_abstract_sentence_and_every_fact():
    prompt = aq.quiz_prompt(CLAIMS, ABSTRACT_3)
    assert "S1: Recurrent models read one word at a time." in prompt and "S3: Our model achieves 28.4 BLEU" in prompt
    assert "Every sentence S1-S3 gets at least one question" in prompt and "gets one question that asks for the whole list" in prompt
    assert '"sentences": [1]' in prompt
    assert aq._quiz_size(3, 3) == (6, 12) and aq._quiz_size(8, 14) == (14, 20)   # one per sentence at least, 20 at most
    assert '"sentences"' not in aq.quiz_prompt(CLAIMS, "")   # no abstract: the claims alone, as before


async def test_make_questions_retries_until_every_sentence_is_asked_and_maps_claims_from_sentences():
    asked = [{"sentences": ["S2"], "claim_ids": [0], "question": f"What is the Transformer based on ({i})?", "answer": "only attention"}
             for i in range(3)] + [
            {"sentences": [3], "claim_ids": [1], "question": "What BLEU on WMT 2014 English-to-German?", "answer": "28.4 BLEU"},
            {"sentences": [3], "claim_ids": [2], "question": "What BLEU on English-to-French?", "answer": "41.8 BLEU"},
            {"sentences": [3], "question": "By how much over the existing best results?", "answer": "by over 2 BLEU, including ensembles"}]
    first = {"questions": asked}   # S1 (how recurrent models read) has no question
    second = {"questions": asked + [{"sentences": ["S1"], "claim_ids": [], "question": "How do recurrent models read a sentence?",
                                     "answer": "one word at a time"}]}
    engine = Fake(queue(first, second))
    claims = CLAIMS + [{"id": 3, "text": "Recurrent models read word by word.", "quote": "Recurrent models read one word at a time."}]
    questions = await aq.make_questions(engine, claims, ABSTRACT_3)
    assert "sentences without a question: S1" in engine.prompts[1]
    by = {q["question"]: q for q in questions}
    assert by["By how much over the existing best results?"]["claim_ids"] == [1]   # no ids given: the claim its sentence holds
    assert by["How do recurrent models read a sentence?"]["sentences"] == [1] and by["How do recurrent models read a sentence?"]["claim_ids"] == [3]
    assert {k for q in questions for k in q["sentences"]} == {1, 2, 3}


async def test_make_questions_without_claims_or_usable_questions_raises():
    with pytest.raises(ValueError):
        await aq.make_questions(Fake(queue()), [])
    with pytest.raises(ValueError):
        await aq.make_questions(Fake(queue({"questions": []}, {"questions": [{"question": "no answer"}]})), CLAIMS)


# ----------------------------------------------------------------------------- the naive reader

async def test_reader_sees_only_the_transcript_and_never_the_gold_answers():
    questions = [{"id": "Q1", "claim_ids": [1], "question": "What BLEU did it reach on English-to-German?", "answer": "GOLD-ANSWER-1"},
                 {"id": "Q2", "claim_ids": [2], "question": "How long did it train?", "answer": "GOLD-ANSWER-2"}]
    reader = Fake(queue({"answers": [{"id": "1", "answer": "28.4 BLEU", "panels": ["Panel 2", 7, "7"]}]},
                        {"answers": [{"id": "Q1", "answer": "Not shown.", "panels": [3]}]}))   # each call numbers its own questions
    text = aq.transcript(make_sb())
    answers = await aq.answer_questions(reader, text, questions)
    assert answers == [{"id": "Q1", "answer": "28.4 BLEU", "panels": [2, 7]}, {"id": "Q2", "answer": "Not shown.", "panels": []}]
    first = reader.prompts[0]
    assert first.startswith("You have never read the research paper behind this manga")
    assert "answer ONLY from the manga text" in first and "outside knowledge" in first and '"not shown"' in first
    assert '"[gap to X: +2.04]" is the difference the chart marks' in first   # the badge in the transcript is explained
    assert text in first and "GOLD-ANSWER" not in first
    assert items_of(reader.prompts[1], "QUESTIONS:\n") == [{"id": "Q1", "question": "How long did it train?"}]   # only the skipped one
    config = reader.calls[0][1]
    assert config.temperature == 0 and config.response_format == {"type": "json_object"}


def test_reply_ids_and_panel_citations_normalise():
    assert [aq._qid(x) for x in ("3", "q3", 3, 3.0, "Q 3", "Question 3", "Q03", "#3")] == ["Q3"] * 8
    assert aq._panel_numbers([7.0, "8", "Panel 9 and 10", True, 2.5]) == [7, 8, 9, 10]


TWO_QUESTIONS = [{"id": "Q1", "claim_ids": [1], "question": "What BLEU did it reach on WMT 2014 English-to-German?", "answer": "28.4 BLEU"},
                 {"id": "Q2", "claim_ids": [2], "question": "How long did it train?", "answer": "3.5 days on eight GPUs"}]


async def test_reader_replies_in_other_shapes_still_count():
    text = aq.transcript(make_sb())

    async def answers(*replies):
        return await aq.answer_questions(Fake(queue(*replies)), text, TWO_QUESTIONS)

    # ids dropped but the order kept; "panel" instead of "panels"
    assert await answers({"answers": [{"answer": "28.4 BLEU", "panel": 2.0}, {"answer": "not shown"}]}) == \
        [{"id": "Q1", "answer": "28.4 BLEU", "panels": [2]}, {"id": "Q2", "answer": "not shown", "panels": []}]
    # keyed by question id, plain strings, ids spelled differently
    assert await answers({"answers": {"Question 1": "28.4 BLEU", "q 2": "not shown"}}) == \
        [{"id": "Q1", "answer": "28.4 BLEU", "panels": []}, {"id": "Q2", "answer": "not shown", "panels": []}]
    # an unknown id is never guessed: Q2 is asked again, then counts as not shown
    reader = Fake(queue({"answers": [{"id": "Q1", "answer": "28.4 BLEU", "panels": [2]}, {"id": "Q7", "answer": "3.5 days", "panels": [1]}]},
                        {"answers": []}))
    got = await aq.answer_questions(reader, text, TWO_QUESTIONS)
    assert got[1] == {"id": "Q2", "answer": "not shown", "panels": []} and len(reader.calls) == 2
    assert items_of(reader.prompts[1], "QUESTIONS:\n") == [{"id": "Q1", "question": "How long did it train?"}]


async def test_a_failed_follow_up_keeps_the_first_round():
    text = aq.transcript(make_sb())
    reader = Fake(queue({"answers": [{"id": "Q1", "answer": "28.4 BLEU", "panels": [2]}]}, RuntimeError("reader down")))
    assert await aq.answer_questions(reader, text, TWO_QUESTIONS) == [{"id": "Q1", "answer": "28.4 BLEU", "panels": [2]},
                                                                      {"id": "Q2", "answer": "not shown", "panels": []}]
    grader = Fake(queue({"grades": [{"id": "Q1", "score": 1, "why": "ok"}]}, RuntimeError("grader down")))
    answers = [{"id": "Q1", "answer": "28.4 BLEU", "panels": [2]}, {"id": "Q2", "answer": "3.5 days", "panels": [2]}]
    assert [(g["score"], g["why"]) for g in await aq.grade(grader, TWO_QUESTIONS, answers, text)] == [(1.0, "ok"), (0.0, "not graded")]
    with pytest.raises(RuntimeError):   # the first call failing is the caller's error (run_quiz records it)
        await aq.answer_questions(Fake(queue(RuntimeError("down"))), text, TWO_QUESTIONS)


# ----------------------------------------------------------------------------- strict grading

Q_EN_DE = "What BLEU score did the Transformer reach on WMT 2014 English-to-German?"


async def test_grade_caps_answers_whose_cited_panels_lack_the_number():
    text = aq.transcript(make_sb())
    questions = [{"id": f"Q{i}", "claim_ids": [1], "question": Q_EN_DE, "answer": "28.4 BLEU"} for i in range(1, 7)]
    answers = [
        {"id": "Q1", "answer": "28.4 BLEU", "panels": [2]},                                 # shown where cited
        {"id": "Q2", "answer": "28.4 BLEU", "panels": [1]},                                 # wrong panel
        {"id": "Q3", "answer": "28.4 BLEU", "panels": []},                                  # no citation
        {"id": "Q4", "answer": "not shown", "panels": []},
        {"id": "Q5", "answer": "28.4 BLEU on WMT 2014 (see panel 12)", "panels": [7]},     # 2014 is in the question
        {"id": "Q6", "answer": "28.4 BLEU, and 41.8 on English-to-French", "panels": [2, 99]},   # 41.8 from memory
    ]
    grader = Fake(lambda prompt: {"grades": [{"id": item["id"], "score": 1, "why": "matches"} for item in items_of(prompt, "ITEMS:\n")]})
    grades = await aq.grade(grader, questions, answers, text)
    by = {g["id"]: g for g in grades}
    assert [g["id"] for g in grades] == [f"Q{i}" for i in range(1, 7)]
    assert by["Q1"] == {"id": "Q1", "score": 1.0, "why": "matches"}
    assert by["Q2"]["score"] == 0.5 and "28.4 not in the cited panel(s) 1" in by["Q2"]["why"]
    assert by["Q3"]["score"] == 0.5 and "no panel" in by["Q3"]["why"]
    assert by["Q4"]["score"] == 0
    assert by["Q5"]["score"] == 1
    assert by["Q6"]["score"] == 0.5 and "41.8" in by["Q6"]["why"]
    sent = items_of(grader.prompts[0], "ITEMS:\n")
    assert [item["id"] for item in sent] == ["Q1", "Q2", "Q3", "Q4", "Q5"]          # "not shown" needs no grader; numbered without the gap
    assert [item["answer"] for item in sent][3:] == [answers[4]["answer"], answers[5]["answer"]]
    assert sent[0]["gold"] == "28.4 BLEU" and "- Transformer (big): 28.4 [highlighted]" in sent[0]["cited_text"]
    assert sent[4]["panels"] == [2]                                                    # panel 99 does not exist
    assert grader.prompts[0].startswith("Grade each answer of a reading quiz strictly")
    config = grader.calls[0][1]
    assert config.temperature == 0 and config.response_format == {"type": "json_object"}


async def test_citation_check_reads_number_words_superscripts_and_separators():
    text = ("Panel 1:\nNarration: A new state of the art on eleven tasks.\n\n"
            "Panel 2:\nBoard (bar chart): Training cost (FLOPs)\n- Transformer (big): 2.3 · 10^19 [highlighted]\n- GNMT+RL: 1,400")
    questions = [{"id": f"Q{i}", "claim_ids": [0], "question": "How many tasks and what training cost?", "answer": "x"} for i in (1, 2, 3)]
    answers = [{"id": "Q1", "answer": "11 tasks", "panels": [1]},
               {"id": "Q2", "answer": "2.3 × 10¹⁹ FLOPs", "panels": [2]},
               {"id": "Q3", "answer": "1400 for GNMT+RL on ８ GPUs", "panels": [2]}]
    grader = Fake(lambda prompt: {"grades": [{"id": f"Q{i}", "score": 1, "why": "ok"} for i in (1, 2, 3)]})
    assert [g["score"] for g in await aq.grade(grader, questions, answers, text)] == [1, 1, 1]


async def test_panel_references_never_swallow_a_value():
    assert aq._answer_numbers("In panel 7, 28.4 BLEU") == {28.4: "28.4"}
    assert aq._answer_numbers("panels 3, 4 and 41.8 BLEU") == {41.8: "41.8"}
    assert aq._answer_numbers("see panels 12 and 13, or 第12格.") == {}
    assert aq._numbers("twenty-one layers, eleven tasks, two hundred GPUs") == {21.0: "twenty-one", 11.0: "eleven"}
    # "Panel 2, 41.8 BLEU": 41.8 is not on panel 2, so the recalled number is capped
    questions = [{"id": "Q1", "claim_ids": [2], "question": "What BLEU on WMT 2014 English-to-French?", "answer": "41.8 BLEU"}]
    grader = Fake(lambda prompt: {"grades": [{"id": "Q1", "score": 1, "why": "matches"}]})
    [g] = await aq.grade(grader, questions, [{"id": "Q1", "answer": "Panel 2, 41.8 BLEU", "panels": [2]}], aq.transcript(make_sb()))
    assert g["score"] == 0.5 and "41.8 not in the cited panel(s) 2" in g["why"]


async def test_grades_without_ids_or_under_another_key_still_count():
    questions = [{"id": f"Q{i}", "claim_ids": [0], "question": f"Question {i}?", "answer": "gold"} for i in (1, 2)]
    answers = [{"id": f"Q{i}", "answer": "an answer", "panels": [1]} for i in (1, 2)]
    for reply in ({"grades": [{"score": 1, "why": "ok"}, {"grade": "partial", "reason": "vague"}]},
                  {"results": [{"id": "1", "score": 1, "why": "ok"}, {"id": "2", "score": 0.5, "why": "vague"}]},
                  {"grades": {"Q1": 1, "Q2": 0.5}}):
        grades = await aq.grade(Fake(queue(reply)), questions, answers, "Panel 1:\nNarration: an answer")
        assert [g["score"] for g in grades] == [1.0, 0.5], reply


async def test_grade_without_a_transcript_only_requires_a_citation():
    questions = [{"id": "Q1", "claim_ids": [1], "question": Q_EN_DE, "answer": "28.4 BLEU"},
                 {"id": "Q2", "claim_ids": [1], "question": Q_EN_DE, "answer": "28.4 BLEU"}]
    answers = [{"id": "Q1", "answer": "28.4 BLEU", "panels": [2]}, {"id": "Q2", "answer": "28.4 BLEU", "panels": []}]
    grader = Fake(lambda prompt: {"grades": [{"id": "Q1", "score": 1, "why": "ok"}, {"id": "Q2", "score": 1, "why": "ok"}]})
    assert [g["score"] for g in await aq.grade(grader, questions, answers)] == [1, 0.5]
    assert "cited_text" not in grader.prompts[0] and "cited text" not in grader.prompts[0]
    # a closed-book control has no manga to cite: the cap held every right closed-book answer at 0.5 (156 Seed runs, never 1)
    assert [g["score"] for g in await aq.grade(grader, questions, answers, cite=False)] == [1, 1]


async def test_grade_snaps_scores_and_asks_again_for_skipped_items():
    questions = [{"id": f"Q{i}", "claim_ids": [0], "question": f"Question {i}?", "answer": "gold"} for i in (1, 2, 3)]
    answers = [{"id": f"Q{i}", "answer": "an answer", "panels": [1]} for i in (1, 2, 3)]
    grader = Fake(queue({"grades": [{"id": "Q1", "score": "0.8"}, {"id": "q2", "score": "partial", "why": "baseline missing"}]},
                        {"grades": []}))
    grades = await aq.grade(grader, questions, answers, "Panel 1:\nNarration: an answer")
    assert grades == [{"id": "Q1", "score": 1.0, "why": ""},
                      {"id": "Q2", "score": 0.5, "why": "baseline missing"},
                      {"id": "Q3", "score": 0.0, "why": "not graded"}]
    assert [(item["id"], item["question"]) for item in items_of(grader.prompts[1], "ITEMS:\n")] == [("Q1", "Question 3?")]


async def test_grades_after_a_not_shown_answer_land_on_their_own_questions():
    # Gated Attention round 13: G7 was "not shown", so the grader saw G6 then G8; it graded G8 as "G7", G9 as "G8" and G10
    # as "G9", and the right answer "non-linearity" lost its point. A grader numbering its reply by position is now right.
    questions = [{"id": f"G{i}", "claim_ids": [0], "question": f"Question {i}?", "answer": f"gold {i}"} for i in (1, 2, 3, 4)]
    answers = [{"id": "G1", "answer": "gold 1", "panels": [1]}, {"id": "G2", "answer": "not shown", "panels": []},
               {"id": "G3", "answer": "gold 3", "panels": [1]}, {"id": "G4", "answer": "wrong", "panels": [1]}]

    def by_position(prompt):   # numbers its grades 1, 2, 3 after the prefix it was shown, as Seed did
        items = items_of(prompt, "ITEMS:\n")
        prefix = items[0]["id"].rstrip("0123456789")
        return {"grades": [{"id": f"{prefix}{i}", "score": 0 if item["answer"] == "wrong" else 1, "why": item["question"]}
                           for i, item in enumerate(items, start=1)]}

    grades = await aq.grade(Fake(by_position), questions, answers, "Panel 1:\nNarration: gold 1, gold 3")
    assert [(g["id"], g["score"]) for g in grades] == [("G1", 1.0), ("G2", 0.0), ("G3", 1.0), ("G4", 0.0)]
    assert grades[2]["why"] == "Question 3?"


# ----------------------------------------------------------------------------- the gate

GOLD_ATTENTION = "solely on attention mechanisms, no recurrence or convolutions"


def quiz_reply() -> dict:
    return {"questions": [
        {"claim_ids": [0], "question": "What is the Transformer based on?", "answer": GOLD_ATTENTION},
        {"claim_ids": [1], "question": Q_EN_DE, "answer": "28.4 BLEU"},
        {"claim_ids": [1], "question": "Which results did it beat on English-to-German?", "answer": "the best results, including ensembles, by over 2 BLEU"},
        {"claim_ids": [2], "question": "What BLEU did it reach on WMT 2014 English-to-French?", "answer": "41.8, a single-model state of the art"},
        {"claim_ids": [2], "question": "How long did the English-to-French model train, and on what?", "answer": "3.5 days on eight GPUs"},
        {"claim_ids": [0, 1], "question": "What does the paper's checklist say about recurrence?", "answer": "only attention, no recurrence"},
    ]}


def reader_reply(prompt: str) -> dict:
    return {"answers": [
        {"id": "Q1", "answer": "Only attention: no recurrence, no convolutions.", "panels": [5, 7]},
        {"id": "Q2", "answer": "28.4 BLEU", "panels": [2]},
        {"id": "Q3", "answer": "It beat the ConvS2S ensemble (26.36) with 28.4.", "panels": [2]},
        {"id": "Q4", "answer": "41.8 BLEU", "panels": [2]},           # not on panel 2: recalled, so capped
        {"id": "Q5", "answer": "not shown", "panels": []},
        {"id": "Q6", "answer": "Only attention: no recurrence.", "panels": [7]},
    ]}


def text_route(prompt: str) -> dict:
    if prompt.startswith("Write a short-answer quiz"):
        return quiz_reply()
    if prompt.startswith("Grade each answer"):
        return {"grades": [{"id": item["id"], "score": 1, "why": "matches the gold answer"} for item in items_of(prompt, "ITEMS:\n")]}
    raise AssertionError(f"unexpected prompt {prompt[:60]!r}")


async def test_run_quiz_scores_the_visible_manga_and_lists_failed_claims():
    text, reader = Fake(text_route), Fake(reader_reply)
    result = await aq.run_quiz(text, reader, make_sb(), CLAIMS, "We propose Transformers.")   # under four words: no sentence to cover
    assert result["score"] == round((1 + 1 + 1 + 0.5 + 0 + 1) / 6, 3)
    assert result["failed_claim_ids"] == [2]
    assert result["untested_claim_ids"] == []
    assert [q["id"] for q in result["questions"]] == [f"Q{i}" for i in range(1, 7)]
    assert [a["id"] for a in result["answers"]] == [g["id"] for g in result["grades"]] == [f"Q{i}" for i in range(1, 7)]
    assert result["usage"] == {"prompt_tokens": 500, "completion_tokens": 50}   # questions, reader, three graders
    assert len(reader.calls) == 1 and len(text.calls) == 1 + aq.GRADERS
    seen = reader.prompts[0]
    assert "- Transformer (big): 28.4 [highlighted]" in seen
    for hidden in ("SECRET DETAIL", "CARD TEXT", GOLD_ATTENTION):
        assert hidden not in seen
    assert aq.PASS_SCORE == 0.85 and result["score"] < aq.PASS_SCORE


async def test_run_quiz_reuses_given_questions():
    text, reader = Fake(text_route), Fake(reader_reply)
    stored = [{"id": f"Q{i}", **q} for i, q in enumerate(quiz_reply()["questions"], start=1)]
    result = await aq.run_quiz(text, reader, make_sb(), CLAIMS, questions=stored)
    assert result["questions"] == stored
    assert not any(p.startswith("Write a short-answer quiz") for p in text.prompts)
    assert result["usage"] == {"prompt_tokens": 400, "completion_tokens": 40}   # the reader and three graders


async def test_run_quiz_lists_claims_no_question_tests():
    reply = {"questions": [q for q in quiz_reply()["questions"] if 2 not in q["claim_ids"]]}
    text = Fake(lambda p: reply if p.startswith("Write a short-answer quiz") else text_route(p))
    result = await aq.run_quiz(text, Fake(reader_reply), make_sb(), CLAIMS)
    assert result["untested_claim_ids"] == [2] and 2 not in result["failed_claim_ids"]


@pytest.mark.parametrize("failure", ["questions", "reader", "grader", "garbage", "no-claims", "broken-storyboard"])
async def test_run_quiz_never_raises(failure):
    def text_side(prompt: str):
        if failure == "questions" or (failure == "grader" and prompt.startswith("Grade each answer")):
            return RuntimeError("boom")
        if failure == "garbage":
            return "this is not JSON at all"
        return text_route(prompt)

    reader = Fake(lambda p: RuntimeError("reader down") if failure == "reader" else reader_reply(p))
    sb = None if failure == "broken-storyboard" else make_sb()
    result = await aq.run_quiz(Fake(text_side), reader, sb, [] if failure == "no-claims" else CLAIMS)
    assert result["score"] is None and result["error"]
    assert set(result) == {"score", "error", "usage"}


async def test_run_quiz_reports_a_reader_or_grader_that_returns_nothing():
    # an empty reply is a broken call, not a manga that shows nothing: no score, so no revision of every claim
    answers = {"answers": [{"id": f"Q{i}", "answer": "28.4 BLEU", "panels": [2]} for i in range(1, 7)]}
    for text_side, reader_side, error in ((text_route, {"answers": []}, "answered only 0 of"),
                                          (lambda p: {"grades": []} if p.startswith("Grade each answer") else text_route(p), answers,
                                           "no usable grades")):
        result = await aq.run_quiz(Fake(text_side), Fake(lambda p, r=reader_side: r), make_sb(), CLAIMS)
        assert result["score"] is None and error in result["error"]


async def test_run_quiz_lets_cancellation_through():
    with pytest.raises(asyncio.CancelledError):
        await aq.run_quiz(Fake(lambda p: asyncio.CancelledError()), None, make_sb(), CLAIMS)


# ----------------------------------------------------------------------------- revision

def failed_quiz() -> dict:
    return {"score": 0.75,
            "questions": [{"id": "Q1", "claim_ids": [1], "question": Q_EN_DE, "answer": "28.4 BLEU"},
                          {"id": "Q2", "claim_ids": [2], "question": "How long did it train?", "answer": "3.5 days on eight GPUs"}],
            "answers": [{"id": "Q1", "answer": "28.4 BLEU", "panels": [2]}, {"id": "Q2", "answer": "not shown", "panels": []}],
            "grades": [{"id": "Q1", "score": 1, "why": "exact"}, {"id": "Q2", "score": 0, "why": "the training time is never shown"}],
            "failed_claim_ids": [2]}


def test_revise_prompt_targets_the_failed_claims_with_quiz_feedback():
    sb = make_sb()
    sb.quiz = failed_quiz()
    prompt = aq.revise_prompt(sb, [2], make_cast(), paper_text="PAPER BODY: trained for 3.5 days on eight P100 GPUs")
    assert prompt.startswith("A reader who never saw the paper read only the visible text of this manga")
    failed = json.loads(prompt.split("did not come across:\n", 1)[1].split("\n", 1)[0])
    assert [c["id"] for c in failed] == [2] and failed[0]["quote"] == CLAIMS[2]["quote"]
    assert failed[0]["quiz"] == [{"question": "How long did it train?", "expected": "3.5 days on eight GPUs", "reader_answered": "not shown",
                                  "cited_panels": [], "why_marked_down": "the training time is never shown"}]
    script = json.loads(prompt.split("SCRIPT\n", 1)[1].split("\n\nPAPER TEXT:", 1)[0])
    assert [p["n"] for p in script] == list(range(1, 9))
    assert script[1]["board"]["bars"][1]["value"] == "28.4" and script[7]["board"] is None
    assert (script[1]["title"], script[1]["emotions"]) == ("Panel 2", {})
    assert "SECRET DETAIL" not in prompt
    # the story and fix prompts' own rules, word for word: revised panels go straight back through the same lint
    assert _v3_rules(make_cast()) in prompt
    for rule in ("at most 3 speech bubbles of at most 16 words each", "1-2 speech bubbles of at most 12 words each",
                 "narration at most 20 words", "narration at most 14 words", '"kind": "links"', '"kind": "table"', "verbatim snippet",
                 "prior best", "No underscores, LaTeX", "the critic nezu asks for proof", "Kumo follows a long recipe", '"after"',
                 "never a result"):
        assert rule in prompt
    assert prompt.rstrip().endswith("PAPER BODY: trained for 3.5 days on eight P100 GPUs")


def test_revise_prompt_in_chinese_counts_characters_and_takes_claim_dicts():
    sb = make_sb(language="zh-CN")
    prompt = aq.revise_prompt(sb, [{"id": 1, "text": "ignored, the storyboard's claim wins"}, 1, "bogus"], make_cast(), quiz={})
    failed = json.loads(prompt.split("did not come across:\n", 1)[1].split("\n", 1)[0])
    assert failed == [{**CLAIMS[1], "quiz": []}]
    for budget in ("at most 3 bubbles of at most 29 characters each", "narration of at most 36 characters",
                   "at most 2 bubbles of at most 22 characters each", "narration of at most 25 characters", "Simplified Chinese"):
        assert budget in prompt
    assert "The paper text is not attached" in prompt


async def test_an_unparsable_quiz_reply_is_asked_once_more():
    broken = '{"answers": [{"id": "Q1", "answer": "bad\x01control"'          # cut off, with a raw control character
    engine = Fake(queue(broken, broken, {"answers": [{"id": "Q1", "answer": "28.4", "panels": [1]}]}))
    got = await aq.answer_questions(engine, "Panel 1:\nnezu: 28.4", [{"id": "Q1", "question": "BLEU?", "answer": "28.4"}])
    assert got[0]["answer"] == "28.4" and len(engine.calls) == 3   # reply, repair round, then the question again


async def test_the_median_of_three_graders_counts():
    """One harsh sample (Seed 2.0 Pro samples at temperature 1) no longer fails a claim on its own."""
    calls = {"n": 0}

    def route(prompt: str):
        if prompt.startswith("Grade each answer"):
            calls["n"] += 1
            harsh = calls["n"] == 2
            items = json.loads(prompt.split("ITEMS:\n", 1)[1])
            return {"grades": [{"id": it["id"], "score": 0 if harsh else 1, "why": "x"} for it in items]}
        return text_route(prompt)

    result = await aq.run_quiz(Fake(route), Fake(reader_reply), make_sb(), CLAIMS)
    # the same scores as three lenient graders (Q4 stays capped by its citation, Q5 was not shown)
    assert calls["n"] == aq.GRADERS and [g["score"] for g in result["grades"]] == [1, 1, 1, 0.5, 0, 1]
