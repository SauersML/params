"""Circuit-only calibration of existing answers; preparation is CPU/read-only, scoring requires explicit worker."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

TASKS = ('text9000', 'text9001', 'text9002', 'text9003')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def save(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2)
        f.flush()
        os.fsync(f.fileno())


def load_rows(path):
    return {r['task']: r for r in (json.loads(x) for x in path.read_text().splitlines() if x.strip())
            if r.get('task') in TASKS}


def prepare(a):
    sys.path.insert(0, str(a.source / 'method/oracle'))
    import components
    from rl.tinker_train import ANSWER_CAP

    a.out.mkdir(parents=True, exist_ok=False)
    base = a.data / 'graph_oracle/components'
    sft, rl = load_rows(base / 'sft_v1full-q8/heldout.jsonl'), load_rows(base / 'rl_v1-q8/heldout.jsonl')
    sets, sample_metadata = {}, {}
    for name, path in [('sft', base / 'sft_v1full-q8/samples.jsonl'),
                       ('rl25', base / 'rl_v1-q8/heldout_25_samples.jsonl')]:
        selected = {t: [] for t in TASKS}
        for x in path.read_text().splitlines():
            r = json.loads(x)
            if r.get('task') in selected:
                selected[r['task']].append(r)
        for t, rows in selected.items():
            assert rows, (name, t)
            rows.sort(key=lambda r: r.get('sample', 0))
        canonical = components.samples_of(path)
        sets[name], sample_metadata[name] = {}, {}
        for t, rs in selected.items():
            raw = rs[0]
            assert raw['finish'] in ('stop', 'length'), (t, raw['finish'])
            answer_tokens = raw['tokens'] - (raw.get('thinking_tokens') or 0)
            assert answer_tokens >= 0
            text, truncated = canonical[t][0]
            assert text == raw['text']
            assert not text.lstrip().startswith('<think>'), 'Expected renderer-extracted answer text'
            sets[name][t] = [(text, truncated or answer_tokens > ANSWER_CAP)]
            sample_metadata[name][t] = {'sample': raw.get('sample', 0), 'finish': raw['finish'],
                'tokens': raw['tokens'], 'thinking_tokens': raw.get('thinking_tokens') or 0,
                'answer_tokens': answer_tokens, 'answer_cap': ANSWER_CAP,
                'truncated': sets[name][t][0][1]}
    sets['teacher'] = {t: [((base / 'sft_v1full-q8/teacher_txt' / (t + '.txt')).read_text(), False)] for t in TASKS}
    comparisons = {}
    rows = {}
    for t in TASKS:
        left, right = sft[t], rl[t]
        qa_l, qa_r = left.get('qa', []), right.get('qa', [])
        comparisons[t] = {'qa_identical': qa_l == qa_r,
                          'sft_qa_sha256': digest(qa_l), 'rl_qa_sha256': digest(qa_r),
                          'sft_qa_count': len(qa_l), 'rl_qa_count': len(qa_r),
                          'identical_questions': sum(q in qa_r for q in qa_l),
                          'same_numbered_questions': sum(x == y for x, y in zip(qa_l, qa_r)),
                          'sft_qa_ids': [q.get('id') for q in qa_l], 'rl_qa_ids': [q.get('id') for q in qa_r]}
        # Answers were sampled on their original questions. Never remap a numbered FINAL answer.
        rows[t] = {**left, 'qa': [], 'edits': []}
    save(a.out / 'inputs.json', {'tasks': TASKS, 'rows': rows, 'sets': sets,
                               'sample_metadata': sample_metadata,
                               'question_comparison': comparisons, 'policy_sample': 'first only',
                               'objective': 'circuit_reward; QA/edits stripped; no E1/E4/teacher-dominance'})
    print(json.dumps(comparisons, indent=2))


def worker(a):
    sys.path.insert(0, str(a.source / 'method/oracle'))
    import torch
    import native
    import goal_eval
    import components
    assert torch.cuda.is_available(), 'GPU scoring worker requires CUDA'
    torch.cuda.set_per_process_memory_fraction(0.46)
    data = json.loads(a.inputs.read_text())
    tasks = TASKS[a.worker * 2:(a.worker + 1) * 2]
    assert len(tasks) == 2
    a.out.mkdir(parents=True, exist_ok=False)
    source = a.source / 'method/oracle'
    files = ('components.py', 'goal_eval.py', 'native.py', 'task_data.py', 'engine.py', 'carry.py', 'variables.py', 'edit_eval.py')
    save(a.out / 'manifest.json', {'tasks': tasks, 'worker': a.worker, 'source_rev': a.rev,
                                  'source_sha256': {f: hashlib.sha256((source/f).read_bytes()).hexdigest() for f in files},
                                  'input_sha256': hashlib.sha256(a.inputs.read_bytes()).hexdigest(),
                                  'pilot_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                  'pool_sha256': hashlib.sha256(components.POOL.read_bytes()).hexdigest(),
                                  'memory_fraction': 0.46, 'draw_split': 'eval', 'pool': str(components.POOL)})
    actual_tasks = native.tasks
    # Scope only the evaluation task enumeration; keep the evaluator's separate donor partition intact.
    native.tasks = lambda split: [native.TEXTS/'vpd4l'/(t+'.json') for t in tasks] if split == 'heldout' else actual_tasks(split)
    nat = native.Native(dev='cuda')
    assert nat.has_importance, 'Full VPD checkpoint required; do not silently omit N or VPD baselines'
    ge = goal_eval.GoalEval(nat)
    for task in tasks:
        started = time.monotonic()
        current = {task: data['rows'][task]}
        sets = {name: {task: [(text, cut) for text, cut in samples[task]]} for name, samples in data['sets'].items()}
        # n=2 + one-row dict makes each completed text a durable independent output.
        result = components.evaluate(ge, 'heldout', current, sets, 2, checkpoint=a.out/(task+'.checkpoint.jsonl'))
        save(a.out/(task+'.json'), result)
        save(a.out/(task+'.metrics.json'), {'seconds': time.monotonic()-started,
             'gpu_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
             'gpu_peak_reserved_bytes': torch.cuda.max_memory_reserved()})
        print(task, 'completed', flush=True)
        torch.cuda.empty_cache()


def aggregate(a):
    records = {}
    manifests = []
    for w in (0, 1):
        folder = a.out / ('worker'+str(w))
        manifests.append(json.loads((folder/'manifest.json').read_text()))
        for t in TASKS[2*w:2*w+2]:
            completed = [json.loads(x) for x in (folder/(t+'.checkpoint.jsonl')).read_text().splitlines()]
            record = [r for r in completed if r.get('type') == 'task']
            assert len(record) == 1 and record[0]['task'] == t
            records[t] = record[0]
    assert manifests[0]['source_sha256'] == manifests[1]['source_sha256']
    assert manifests[0]['input_sha256'] == manifests[1]['input_sha256']
    assert manifests[0]['pool_sha256'] == manifests[1]['pool_sha256']
    assert manifests[0]['pilot_sha256'] == manifests[1]['pilot_sha256']
    rows = []
    for t, record in records.items():
        samples = {name: v['samples'][0] for name, v in record['per_question'].items()}
        for name, sample in samples.items():
            rows.append({'task': t, 'policy': name, 'valid': sample['valid'],
                         **{k: sample.get(k) for k in ('circuit_reward','phi','phi_tau','g','compact','bits','D','term_F1','term_N')},
                         'raw_families': (sample.get('fidelity') or {}).get('rl_terms'),
                         'clipped_C': sample.get('term_C')})
    paired = []
    for t in TASKS:
        own = {r['policy']: r for r in rows if r['task'] == t}
        for left, right in [('rl25','sft'), ('sft','teacher'), ('rl25','teacher')]:
            x, y = own[left], own[right]
            shared = set(x['raw_families'] or {}) & set(y['raw_families'] or {})
            paired.append({'task': t, 'difference': left+'-'+right,
                'circuit_reward': x['circuit_reward']-y['circuit_reward'],
                'raw_family_deltas': {k: x['raw_families'][k]-y['raw_families'][k] for k in sorted(shared)
                                      if x['raw_families'][k] is not None and y['raw_families'][k] is not None}})
    means = {}
    for comparison in sorted({r['difference'] for r in paired}):
        subset = [r for r in paired if r['difference'] == comparison]
        keys = sorted(set().union(*(r['raw_family_deltas'] for r in subset)))
        means[comparison] = {'mean_circuit_reward_delta': sum(r['circuit_reward'] for r in subset)/len(subset),
            'raw_family_deltas': {k: {'n': len(vals), 'mean': sum(vals)/len(vals)}
                for k in keys for vals in [[r['raw_family_deltas'][k] for r in subset if k in r['raw_family_deltas']]]}}
    save(a.out/'aggregate.json', {'tasks': TASKS, 'manifests': manifests, 'rows': rows, 'paired': paired, 'means': means,
                                'scope': '4 development texts, first samples, circuit-only; no English claim'})


def panel_worker(a):
    """Replay saved circuits on common interventions; all other metrics stay frozen."""
    sys.path.insert(0, str(a.source / 'method/oracle'))
    import torch
    import components
    import goal_eval
    import native
    import intervention_panel

    assert torch.cuda.is_available(), 'GPU panel worker requires CUDA'
    torch.cuda.set_per_process_memory_fraction(0.46)
    data = json.loads(a.inputs.read_text())
    tasks = TASKS[2*a.worker:2*a.worker+2]
    a.out.mkdir(parents=True, exist_ok=False)
    prior = {}
    for t in tasks:
        path = a.prior / (t + '.checkpoint.jsonl')
        records = [json.loads(line) for line in path.read_text().splitlines()]
        completed = [r for r in records if r.get('type') == 'task' and r['task'] == t]
        assert len(completed) == 1
        prior[t] = completed[0]
    names = ('sft', 'rl25', 'teacher')
    source_files = ('components.py', 'goal_eval.py', 'native.py', 'engine.py', 'carry.py',
                    'intervention_panel.py')
    save(a.out / 'manifest.json', {
        'tasks': tasks, 'source_rev': a.rev, 'split': 'eval', 'draws': 32,
        'source_sha256': {f: hashlib.sha256((a.source/'method/oracle'/f).read_bytes()).hexdigest()
                          for f in source_files},
        'input_sha256': hashlib.sha256(a.inputs.read_bytes()).hexdigest(),
        'prior_sha256': {t: hashlib.sha256((a.prior/(t+'.checkpoint.jsonl')).read_bytes()).hexdigest()
                         for t in tasks},
        'scope': 'New common-node panel plus sufficiency guard; other families frozen from prior run. Not a full rescore.'})
    nat = native.Native(dev='cuda')
    assert nat.has_importance
    ge = goal_eval.GoalEval(nat)
    for t in tasks:
        started = time.monotonic()
        task = json.loads((native.TEXTS/'vpd4l'/(t+'.json')).read_text())
        prompt = task['prompts'][0]
        tx = ge.text(prompt['token_ids'], prompt['target_positions'], task=task)
        graphs = [goal_eval.circuit(components.parse(data['sets'][name][t][0][0])['answer'],
                                   tx.targets, nat.C, len(tx.ids))[0] for name in names]
        panels = intervention_panel.evaluate(ge, tx, graphs + [native.Graph(), native.Graph(masks=tx.vpd)],
                                             split='eval', draws=32)
        # Preserve the common world once, not once per candidate in every draw.
        save(a.out/(t+'.bank.json'), intervention_panel.bank(tx, split='eval', draws=32))
        for panel in panels:
            for draw in panel['draws']:
                draw.pop('removed', None)
            panel['bank_file'] = t+'.bank.json'
        rows = []
        for name, panel in zip(names, panels):
            old = prior[t]['per_question'][name]['samples'][0]
            modified = dict(old['fidelity'])
            modified['rl_terms'] = {**modified.get('rl_terms', {}), 's_nodes': panel['raw_score']}
            modified['eps_nodes'] = panel['eps']
            ph = goal_eval.phi(modified)[0]
            # the production circuit term (reward_gate.qualify's prototype is gone: one definition), at the old bar and
            # compactness (bits_start 2^compact against 1 bit), no floors
            ct = goal_eval.circuit_term(modified, {'families': None, 'phi_tau': old['phi_tau'], 'floors': {},
                                                   'bits_start': 2.0 ** (old.get('compact') or 0.0)}, 1.0)
            # Undefined common calibration is a shared evidence gap, never successful omission.
            comparable = panel['resolved'] and ph is not None
            proposed = ct['term'] + (old.get('D') or 0.0) if comparable else None
            rows.append({'policy': name, 'prior_circuit_reward': old['circuit_reward'],
                         'prior_phi': old['phi'], 'point_score': goal_eval.family_score(modified, 'point'),
                         'prospective_phi': ph if comparable else None,
                         'prospective_circuit_reward': proposed, 'circuit_term': ct, 'panel': panel})
        save(a.out/(t+'.json'), {'task': t, 'rows': rows, 'empty': panels[-2], 'vpd': panels[-1],
             'seconds': time.monotonic()-started, 'gpu_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
             'scope': 'Exploratory replacement of node family only; original tau retained, other families frozen.'})
        print(json.dumps({'task': t, 'rows': [{k: v for k, v in r.items() if k != 'panel'} for r in rows]}), flush=True)
        torch.cuda.empty_cache()


p = argparse.ArgumentParser()
sub = p.add_subparsers(dest='command', required=True)
f = sub.add_parser('prepare'); f.add_argument('--data', type=Path, required=True); f.add_argument('--out', type=Path, required=True)
f.add_argument('--source', type=Path, required=True)
f = sub.add_parser('worker'); f.add_argument('--worker', type=int, choices=(0,1), required=True); f.add_argument('--source',type=Path,required=True)
f.add_argument('--inputs',type=Path,required=True); f.add_argument('--out',type=Path,required=True); f.add_argument('--rev',required=True)
f = sub.add_parser('aggregate'); f.add_argument('--out',type=Path,required=True)
f = sub.add_parser('panel-worker')
f.add_argument('--worker', type=int, choices=(0, 1), required=True)
f.add_argument('--source', type=Path, required=True)
f.add_argument('--inputs', type=Path, required=True)
f.add_argument('--prior', type=Path, required=True)
f.add_argument('--out', type=Path, required=True)
f.add_argument('--rev', required=True)
a = p.parse_args()
{'prepare': prepare, 'worker': worker, 'aggregate': aggregate, 'panel-worker': panel_worker}[a.command](a)
