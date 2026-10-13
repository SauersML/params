"""No model calls: freeze reader questions separately from measured answers."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def sha(value):
    return hashlib.sha256(value).hexdigest()


def rows(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def unit_diagram(completion):
    text = completion.split('English:', 1)[0]
    edges = re.findall(r'(\w+)\s*-->\s*(\w+)', text)
    allowed = lambda x: x == 'out' or re.fullmatch(r'L\d+[HN]\d+', x)
    kept = [(a, b) for a, b in edges if allowed(a) and allowed(b)]
    return 'flowchart LR\n' + '\n'.join(f'  {a} --> {b}' for a, b in kept)


def choose(bank):
    shown = {tuple(x) for x in bank[:2]}
    unique = {}
    for index, value in enumerate(bank[2:], 2):
        if tuple(value) not in shown:
            unique.setdefault(tuple(value), index)
    return sorted(unique.values(), key=lambda i: (sha(json.dumps(bank[i], separators=(',', ':')).encode()), i))[:2]


ARCH = ('This is a 32-token causal decoder with two layers, four attention heads per layer, '
        'pre-RMSNorm, rotate-half RoPE, GELU(tanh) MLP, and tied input/output embeddings. '
        'Positions start at 0; BOS is token 0. Heads are L0H0..L0H3 and L1H0..L1H3. '
        'The diagram only identifies which component can feed which component or output; '
        'it specifies no token roles, attention positions, or routing rules. No weights are supplied. '
        'The original top token is a common baseline observation, not an edited result. '
        'Scale the specified head output at every position, leaving all other weights fixed; '
        '0 deletes its contribution and 0.5 halves it. Predict the last-position top token.')


def prepare(root, out):
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / 'questions.jsonl').exists(), 'Never overwrite a frozen corpus'
    base = {r['id']: r for r in rows(root / 'trial/base.jsonl')}
    truth = {r['id']: r for r in rows(root / 'panel_truth.jsonl')}
    draws = {(r['task'], r['input'], r['edit']['layer'], r['edit']['head'], r['edit']['scale']): r
             for r in rows(root / 'head_counterfactual/draws.jsonl')}
    tasks = [f'multihop_{i:03d}' for i in range(4)]
    questions, labels, selected = [], [], {}
    for t, task in enumerate(tasks):
        bank = json.loads((root / f'cases/{task}/bank.json').read_text())
        selected[task] = choose(bank)
        assert len(selected[task]) == 2
        original = base[task]['completion'].split('English:', 1)[1].strip()
        shuffled = base[tasks[(t + 1) % 4]]['completion'].split('English:', 1)[1].strip()
        gold = truth[task]['truth']['english']
        diagram = unit_diagram(base[task]['completion'])
        assert 'tok' not in diagram
        for index in selected[task]:
            for layer in range(2):
                for head in range(4):
                    for scale in (0., .5):
                        draw = draws[(task, index, layer, head, scale)]
                        case = sha(f'{task}/{index}/{layer}/{head}/{scale}'.encode())[:20]
                        common = (ARCH + '\n\nComponent connectivity:\n' + diagram + '\n\n'
                                  + 'Original input: ' + ' '.join('BOS' if x == 0 else f't{x}' for x in bank[index])
                                  + f'\nOriginal top: t{draw["original_top"]}'
                                  + f'\nEdit: L{layer}H{head} output scale {scale:g}.'
                                  + '\nCandidate labels (same order in every condition): '
                                  + ', '.join(f't{x}' for x in range(32)))
                        for condition, english in [('none', None), ('generated', original), ('shuffled', shuffled), ('gt_positive', gold)]:
                            prompt = common + ('\n\nExplanation:\n' + english if english is not None else '')
                            prompt += '\n\nReturn only JSON: {"top":"tN","changed":true_or_false}. No explanation.'
                            ident = sha(f'{case}/{condition}'.encode())[:24]
                            questions.append({'id': ident, 'prompt': prompt, 'prompt_sha256': sha(prompt.encode())})
                            labels.append({'id': ident, 'case': case, 'condition': condition, 'task': task,
                                           'input': index, 'layer': layer, 'head': head, 'scale': scale,
                                           'active': draw['active_damage'], 'top': draw['reference_top'],
                                           'original_top': draw['original_top'], 'changed': draw['reference_top'] != draw['original_top'],
                                           'distribution_binding': draw['distributions'],
                                           'common_sha256': sha(common.encode()), 'english': english})
    assert len(questions) == len(labels) == 512
    for name, value in [('questions.jsonl', questions), ('labels.jsonl', labels)]:
        (out / name).write_text(''.join(json.dumps(r) + '\n' for r in value))
    manifest = {'schema': 1, 'cases': 128, 'responses': 512, 'selected_indices': selected,
                'selection': 'SHA256 ascending unique bank input bytes, excludes first two displayed inputs; no outcome selection',
                'project': '3f598f31-2ffa-4eb4-b302-909eb234fb7d', 'provider': 'Tinker',
                'sampling': {'temperature': 0, 'num_samples': 1, 'max_tokens': 64, 'thinking': False},
                'scope': 'Four reused-development 32-vocabulary toys; conditional reader diagnostic, not English certification',
                'conditions': ['none', 'generated', 'shuffled', 'gt_positive'],
                'shuffle': 'Cyclic other-task original English, unchanged; weak control may change named head IDs',
                'source_hashes': {str(p): sha(p.read_bytes()) for p in [root / 'trial/base.jsonl', root / 'panel_truth.jsonl', root / 'head_counterfactual/draws.jsonl', root / 'head_counterfactual/distribution_manifest.jsonl', Path(__file__)]},
                'question_sha256': sha((out / 'questions.jsonl').read_bytes()),
                'label_sha256': sha((out / 'labels.jsonl').read_bytes()),
                'cost_preflight': 'Must measure exact provider renderer prompt lengths before any call; no cached discount assumed',
                'interpretation': 'GT-English positive control must improve over no-English before interpreting flat generated-English results; unaltered generated prose and ignored token annotations remain unverified'}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('root', type=Path)
    ap.add_argument('out', type=Path)
    a = ap.parse_args()
    print(json.dumps(prepare(a.root, a.out), indent=2))
