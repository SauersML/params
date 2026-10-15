"""Checks of reader_reward.py's parsing and grading without a model or Tinker.

  ~/mpd-data/venv/bin/python -m pytest method/oracle/test_reader_reward.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import reader_reward as rr  # noqa: E402


class FakeTarget:
    ids = [0, 1, 2, 3]
    VOCAB = {0: "The", 1: " new", 2: " the", 3: " creation", 4: " building", 5: " work"}

    def tok(self, i):
        return self.VOCAB[int(i)]

    def token_id(self, s):
        return next((i for i, v in self.VOCAB.items() if v == s), None)

    where = rr.VPDTarget.where

    def fresh(self, pos, n, rng):
        return [" building", " work"][:n]

    def valid(self, part):
        n, p, c = part
        return n in ("h.3.mlp.down_proj", "h.0.mlp.c_fc", "h.0.attn.q_proj") and 0 <= p < len(self.ids) and c < 4096


def test_parts_and_positions():
    tg = FakeTarget()
    assert rr.site_of("L3.down#2481@3") == ("h.3.mlp.down_proj", 3, 2481)
    assert rr.site_of("L0.q#7@' creation'", tg.where.__get__(tg)) == ("h.0.attn.q_proj", 3, 7)
    assert rr.site_of("L0.x#7@3") is None
    assert rr.label("h.1.attn.o_proj", 2, 9) == "L1.o#9@2"


def test_invariant_lines():
    tg = FakeTarget()
    tg.where = rr.VPDTarget.where.__get__(tg)
    assert rr.parse_inv("INV: swap @' creation' with any of [' building', ' work'] -> top ' the'", tg) is None  # listed by the answer
    ivs, oc = rr.parse_inv("INV: swap @' creation' with predicted tokens -> top ' the'", tg)
    assert [iv["token"] for iv in ivs] == [" building", " work"] and ivs[0]["pos"] == 3 and oc == ("top", " the")
    assert rr.parse_inv("INV: swap @3 with things it likes -> p > 0.5", tg) is None  # no such scope
    ivs, oc = rr.parse_inv("INV: swap @3 with predicted tokens and set back L3.down#2481@3, L0.fc#1@2 -> p > 0.7", tg)
    assert ivs[0]["kind"] == "setback" and ivs[0]["parts"] == [["h.3.mlp.down_proj", 3, 2481], ["h.0.mlp.c_fc", 2, 1]] and oc == (">", 0.7)
    ivs, oc = rr.parse_inv("INV: remove L3.down#2481@3 -> top not ' of'", tg)
    assert ivs[0]["kind"] == "remove" and oc == ("top_not", " of")
    ivs, _ = rr.parse_inv("INV: scale L3.down#2481@3 by 0.5 -> p < 0.4", tg)
    assert ivs[0]["by"] == 0.5
    assert rr.parse_inv("INV: remove L3.down#2481@9 -> p > 0.5", tg) is None  # no position 9
    assert rr.parse_inv("INV: remove L4.o#1@3 -> p > 0.5", tg) is None  # no layer 4
    assert rr.parse_inv("INV: remove L3.down#2481@'creation' -> p > 0.5", tg)[0][0]["parts"] == [["h.3.mlp.down_proj", 3, 2481]]
    assert rr.parse_inv("swap @3 with any of [' work'] -> p > 0.5", tg) is None
    tg.named = {"G1": [["h.0.mlp.c_fc", 3, 5], ["h.0.mlp.c_fc", 3, 6]]}
    ivs, oc = rr.parse_inv("INV: remove G1, L3.down#2481@3 -> p < 0.2", tg)
    assert ivs[0]["parts"] == [["h.0.mlp.c_fc", 3, 5], ["h.0.mlp.c_fc", 3, 6], ["h.3.mlp.down_proj", 3, 2481]]
    assert rr.parse_inv("INV: remove G2 -> p < 0.2", tg) is None  # no group G2
    assert rr.meets({"top": " of", "p": 0.2}, ("top", " of")) and rr.meets({"top": " of", "p": 0.2}, ("<", 0.3))
    assert not rr.meets({"top": " of", "p": 0.2}, ("top_not", " of"))


def test_answers_and_grades():
    got = rr.parse_answers("Q1: ' of' 0.9\nQ3: \"'s\" 0.1\nQ2: nonsense", 3)
    assert got == [(" of", 0.9), (None, None), ("'s", 0.1)]
    ivs = [{"kind": "swap", "outcome": {"top": " of", "p": 0.9}}, {"kind": "remove", "outcome": {"top": " of", "p": 0.5}},
           {"kind": "swap", "outcome": {"top": " and", "p": 0.01}}]
    rows = rr.grade(got, ivs)
    assert rows[0]["top_ok"] and abs(rows[0]["p_credit"] - 1) < 1e-9 and rows[1]["score"] == 0.0
    assert not rows[2]["top_ok"] and 0 < rows[2]["p_credit"] < 1

def test_accept_audit_binds_questions_and_keeps_raw_arms_and_kind_transfer(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    questions=tmp_path/'groups.jsonl'
    ivs=[{'kind':kind,'group':'G1','head':'synthetic head','answer':' x',
          'words':f'{kind} G1','outcome':{'top':' x','p':.5}}
         for kind in ('remove','scale','setback')]
    questions.write_text(''.join(json.dumps(iv)+'\n' for iv in ivs))
    facts=tmp_path/'groups_facts.txt'
    facts.write_text('G1 takes p to 0.50; G2 takes p to 0.20; G3 takes p to 0.10.')
    observed=[]
    responses=["Q1: ' y' 0.5\nQ2: ' x' 0.5\nQ3: ' y' 0.5",
               "Q1: ' x' 0.5\nQ2: ' y' 0.5\nQ3: ' x' 0.5",
               "Q1: ' y' 0.5\nQ2: ' y' 0.5\nQ3: ' y' 0.5"]
    class FakeReader:
        def __init__(self,*args):self.spend=SimpleNamespace(total=0.)
        def __call__(self,prompts):observed.extend(prompts);return responses
        def tokens(self,text):return len(text.split())
    monkeypatch.setattr(rr,'Reader',FakeReader)
    output=tmp_path/'accept.json'
    rr.cmd_accept(SimpleNamespace(reader='synthetic-open-reader',questions=[questions],out=output,held_out=False,explanations='_facts.txt'))
    saved=json.loads(output.read_text());text=saved['texts'][0]
    assert len(observed)==3 and len(text['per_question'])==3
    assert [text['arms'][arm]['raw_response'] for arm in ('none','true','rotated')]==responses
    assert text['per_question'][0]['arms']['true']['prediction']==[' x',.5]
    assert text['per_question'][0]['arms']['true']['top_correct']
    assert not text['per_question'][1]['arms']['true']['top_correct']
    assert saved['by_kind']['remove']['true_minus_none']==.5
    assert saved['by_kind']['scale']['true_minus_none']==-.5
    assert saved['by_kind']['setback']['true_minus_none']==.5
    assert all(saved['acceptance'].values())
    assert saved['reader_identity']=={'model':'synthetic-open-reader','temperature':0.0}
    assert len(saved['source_identity']['reader_reward_sha256'])==64
    assert len(text['question_file_sha256'])==len(text['facts_sha256'])==64
    old_question=text['per_question'][0]['question_sha256'];old_facts=text['facts_sha256']
    facts.write_text(facts.read_text()+' Extra frozen explanation text.')
    rr.cmd_accept(SimpleNamespace(reader='synthetic-open-reader',questions=[questions],out=output,held_out=False,explanations='_facts.txt'))
    updated=json.loads(output.read_text())['texts'][0]
    assert updated['per_question'][0]['question_sha256']==old_question
    assert updated['facts_sha256']!=old_facts
    assert updated['arms']['true']['prompt_sha256']!=text['arms']['true']['prompt_sha256']
    assert 'retrieval' in saved['interpretation']['remove']


def test_accept_audit_preserves_held_out_filter_and_selected_written_explanation(tmp_path,monkeypatch):
    import json
    from types import SimpleNamespace
    questions=tmp_path/'groups.jsonl'
    ivs=[{'kind':kind,'held_out':held,'head':'head','answer':' x','words':f'{kind} G1',
          'outcome':{'top':' x','p':.5}}
         for kind,held in [('remove',False),('scale',True),('setback',True)]]
    questions.write_text(''.join(json.dumps(iv)+'\n' for iv in ivs))
    (tmp_path/'groups_facts.txt').write_text('THIS DEFAULT EXPLANATION MUST NOT BE READ')
    (tmp_path/'groups_written.txt').write_text('Selected written explanation: G1 supports; G2 inhibits; G3 routes.')
    observed=[]
    class FakeReader:
        def __init__(self,*args):self.spend=SimpleNamespace(total=0.)
        def __call__(self,prompts):
            observed.extend(prompts)
            return ["Q1: ' x' 0.5\nQ2: ' x' 0.5"]*len(prompts)
        def tokens(self,text):return len(text.split())
    monkeypatch.setattr(rr,'Reader',FakeReader)
    output=tmp_path/'accept.json'
    args=SimpleNamespace(reader='synthetic',questions=[questions],out=output,held_out=True,explanations='_written.txt')
    rr.cmd_accept(args)
    saved=json.loads(output.read_text())
    assert saved['questions']==2 and set(saved['by_kind'])=={'scale','setback'}
    assert saved['question_selection']['held_out_only'] is True
    assert saved['question_selection']['explanation_suffix']=='_written.txt'
    assert all(q['question']['held_out'] for q in saved['texts'][0]['per_question'])
    assert all('remove G1' not in prompt for prompt in observed)
    assert 'Selected written explanation' in observed[1]
    assert all('THIS DEFAULT' not in prompt for prompt in observed)
    assert not saved['acceptance']['pooled_true_above_rotated']
    assert not saved['acceptance']['every_text_true_above_rotated']
    # An empty selection must not silently score zero or send an empty reader request.
    import pytest
    questions.write_text(json.dumps(ivs[0])+'\n')
    observed.clear()
    with pytest.raises(ValueError,match='no selected acceptance questions'):
        rr.cmd_accept(args)
    assert not observed
