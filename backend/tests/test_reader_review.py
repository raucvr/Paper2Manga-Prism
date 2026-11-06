import copy
import json
from types import SimpleNamespace

import pytest

from engines.base import TextResponse
from services import reader_review as rr
from services.storyboarder_v2 import (Bubble, PanelV2, StoryboardV2, _vague_list_lines,
    _recap_mirrors_claims, _meta_lines, _thin_claim, clarity_prompt, lint_v3, story_prompt, fix_prompt)
from .fakes import make_cast
from .test_story_v3 import v3_paper


def story():
    p = PanelV2(1, 'hook', '', [], 'HIDDEN_SCENE', 'HIDDEN_VISUAL',
                narration='We want a smaller model. Rounding saves memory.',
                dialogue=[Bubble('papi', 'This is an illustration, not a measured result.')])
    return StoryboardV2('', '', '', 'en-US', [], [p], version=3,
        brief={'reader_contract': rr.CONTRACT, 'glossary': [{'term': 'HIDDEN_GLOSSARY'}], 'paper': 'HIDDEN_PAPER'},
        abstract_claims=[{'id': 0, 'text': 'HIDDEN_CLAIM'}])


class Engine:
    def __init__(self, reply=None, error=None):
        self.reply, self.error, self.calls = reply, error, []

    async def generate_text(self, messages, config=None):
        self.calls.append((messages, config))
        if self.error:
            raise self.error
        return TextResponse(json.dumps(self.reply), 'fake')


def valid():
    return {'answers': [{'question': q, 'answer': 'The reader can explain this.',
        'evidence': [{'n': 1, 'quote': 'Rounding saves memory.'}]} for q in rr.QUESTIONS],
        'problems': [], 'reason': 'No diagnostic issue found.'}


async def test_review_is_blind_validated_and_stores_exact_schema():
    sb = story(); engine = Engine(valid())
    out = await rr.review_story(sb, engine)
    assert out is sb.brief['reader_review']
    assert set(out) == {'status', 'language', 'fingerprint', 'answers', 'problems', 'reason'}
    assert out['status'] == 'reviewed' and out['language'] == 'en-US'
    prompt = engine.calls[0][0][0].content
    assert not any(x in prompt for x in ('HIDDEN_', 'answer_key'))
    assert 'Rounding saves memory.' in prompt and 'opening_questions' in prompt
    assert engine.calls[0][1].response_format == {'type': 'json_object'}
    assert 'not human comprehension proof' in out['reason']


@pytest.mark.parametrize('mutation', [
    lambda d: d.pop('answers'), lambda d: d['answers'].pop(),
    lambda d: d['answers'][0]['evidence'][0].update(n=99),
    lambda d: d['answers'][0]['evidence'][0].update(quote='Invented quotation'),
    lambda d: d['answers'][0].update(evidence=[]),
    lambda d: d['answers'].append(d['answers'][0]),
    lambda d: d.update(problems=[{'n': 99, 'kind': 'overload', 'problem': 'x', 'fix': 'y'}]),
])
async def test_bad_or_missing_evidence_never_passes(mutation):
    data = valid(); mutation(data)
    out = await rr.review_story(story(), Engine(data))
    assert out['status'] == 'unassessed' and out['answers'] == []


async def test_failed_engine_is_safe_and_account_errors_propagate():
    out = await rr.review_story(story(), Engine(error=RuntimeError('SENSITIVE_PROVIDER_DETAILS')))
    assert out['status'] == 'unassessed' and 'SENSITIVE' not in json.dumps(out)
    err = RuntimeError('account'); err.account = True
    with pytest.raises(RuntimeError):
        await rr.review_story(story(), Engine(error=err))


async def test_cannot_tell_and_findings_are_needs_revision():
    data = valid(); data['answers'][0].update(answer='cannot tell', evidence=[])
    assert (await rr.review_story(story(), Engine(data)))['status'] == 'needs_revision'
    data = valid(); data['problems'] = [{'n': 1, 'kind': 'causal_gap', 'problem': 'A step is missing.', 'fix': 'Explain the step.'}]
    out = await rr.review_story(story(), Engine(data))
    assert out['status'] == 'needs_revision' and out['problems'] == data['problems']


async def test_legacy_is_not_migrated_or_called():
    sb = story(); sb.brief.pop('reader_contract'); engine = Engine(valid())
    assert not rr.is_novice(sb)
    assert (await rr.review_story(sb, engine))['status'] == 'unassessed'
    assert not engine.calls and 'reader_contract' not in sb.brief


def test_fingerprint_tracks_only_visible_content_and_language():
    sb = story(); original = rr.visible_fingerprint(sb)
    sb.brief['glossary'] = ['other']; sb.panels[0].visual = 'changed'; sb.abstract_claims = []
    assert rr.visible_fingerprint(sb) == original
    sb.panels[0].narration += ' Changed.'
    assert rr.visible_fingerprint(sb) != original
    before = rr.visible_fingerprint(sb); sb.language = 'zh-CN'
    assert rr.visible_fingerprint(sb) != before
    p = sb.panels[0]; p.kind = 'board'; p.board = {'kind': 'list', 'title': 'Questions', 'items': ['What helps?', 'Why?'], 'open': True}
    before = rr.visible_fingerprint(sb); p.board['quote'] = 'HIDDEN_QUOTE'
    assert rr.visible_fingerprint(sb) == before
    p.board['items'][0] = 'What changes?'
    assert rr.visible_fingerprint(sb) != before


def test_opening_recap_and_source_boundaries_branch_without_losing_coverage():
    sb = story(); p = sb.panels[0]; p.kind = 'board'; p.beat = 'claims'
    p.board = {'kind': 'list', 'open': True, 'items': ['Why do we need good data?', 'How can memory use fall?']}
    recap = copy.deepcopy(p); recap.n = 3; recap.beat = 'recap'
    recap.board['items'] = ['Varied training cases expose different mistakes.', 'Rounding stores fewer possible values.']
    teach = PanelV2(2, 'how', '', [], '', '', narration='The paper does not show this example solved.', beat='how')
    sb.panels += [teach, recap]
    assert not _vague_list_lines(sb) and not _recap_mirrors_claims(sb) and not _meta_lines(sb)
    sb.abstract_claims = [{'id': 0, 'text': 'The score is 28.4.', 'quote': 'The score is 28.4.'}]
    assert any('claim 0 is not taught' in h for hints in lint_v3(sb, 'The score is 28.4.').values() for h in hints)
    sb.brief.pop('reader_contract')
    assert _recap_mirrors_claims(sb) and _meta_lines(sb)


def test_novice_clarity_is_blind_legacy_stays_unchanged():
    sb = story()
    assert 'HIDDEN_' not in clarity_prompt(sb) and 'prerequisite' in clarity_prompt(sb)
    sb.brief.pop('reader_contract')
    assert 'GLOSSARY' in clarity_prompt(sb) and 'HIDDEN_GLOSSARY' in clarity_prompt(sb)


def test_write_fix_and_quiz_prompts_keep_sequence():
    from services.abstract_quiz import revise_prompt
    sb = story()
    for prompt in [story_prompt(sb.brief, [], make_cast(), (16, 32), v3_paper()),
                   fix_prompt(sb, make_cast(), {1: ['claim missing']}),
                   revise_prompt(sb, [], make_cast())]:
        assert 'NOVICE READER CONTRACT' in prompt
        assert 'A panel that teaches a claim says all of it' not in prompt
        assert 'rewrite the recap\'s lines as the claims board' not in prompt
        assert 'never restuff' in prompt.lower()


async def test_invalid_json_is_unassessed_without_a_retry():
    class Broken(Engine):
        async def generate_text(self, messages, config=None):
            self.calls.append(messages)
            return TextResponse('not JSON', 'fake')
    engine = Broken()
    assert (await rr.review_story(story(), engine))['status'] == 'unassessed'
    assert len(engine.calls) == 1


@pytest.mark.parametrize('board', [
    {'kind': 'bar', 'title': 'Score', 'unit': 'points', 'bars': [
        {'label': 'Before', 'value': '12', 'baseline': True}, {'label': 'After', 'value': '15', 'highlight': True}]},
    {'kind': 'table', 'columns': ['Model', 'Score'], 'rows': [{'cells': ['A', '12'], 'quote': 'HIDDEN_QUOTE'}]},
    {'kind': 'links', 'tokens': ['a', 'b'], 'from': 0, 'weights': [0, 1], 'illustrative': True},
    {'kind': 'diagram', 'title': 'Step', 'brief': 'HIDDEN_DIAGRAM', 'elements': [
        {'type': 'text', 'x': 50, 'y': 50, 'text': 'Visible step', 'size': 24}]},
])
def test_visible_board_extraction_agrees_with_quiz_and_ignores_instructions(board):
    from services.abstract_quiz import transcript
    sb = story(); sb.panels[0].kind = 'board'; sb.panels[0].board = board
    assert rr.transcript(sb) == transcript(sb)
    assert 'HIDDEN_' not in rr.transcript(sb)
    before = rr.visible_fingerprint(sb)
    board['brief'] = 'Another hidden instruction'
    assert rr.visible_fingerprint(sb) == before


def test_successive_panels_together_teach_a_complete_claim():
    sb = story()
    a = sb.panels[0]; a.beat = 'problem'; a.claims = [0]
    a.narration = 'Training needs examples.'
    b = copy.deepcopy(a); b.n = 2; b.beat = 'how'
    b.narration = 'Long diverse challenging verifiable examples permit rewards to be checked.'
    sb.panels.append(b)
    claim = {'id': 0, 'text': 'Training needs long diverse challenging verifiable examples so rewards can be checked.'}
    assert _thin_claim(sb, claim, [a])
    assert _thin_claim(sb, claim, [a, b]) == []
