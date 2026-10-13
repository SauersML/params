"""Fresh, shared-development unit-vs-wiring pilot; stages resume from completed artifacts only.

Run on the existing free gha-run workflow, links=1 (no automatic extension). Example:
  python experiments/oracle/toyloop/paired_wiring.py --out "$OUT" --stage all
Use --stage sft first, then unit/wiring on separate copies of its artifact.
Each RL stage evaluates only its own frozen policy; neither arm waits on the other. A partial training stage
is never treated as a finished checkpoint or silently resumed with a fresh optimizer.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import subprocess
import shutil
import platform
import tempfile
import time

MODEL = 'HuggingFaceTB/SmolLM2-135M-Instruct'
REVISION = '12fd25f77366fa6b3b4b768ec3050bf629380bac'
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def hashes(directory):
    return {str(p.relative_to(directory)): digest(p) for p in sorted(Path(directory).rglob('*')) if p.is_file()}


def write(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as stream:
        stream.write(json.dumps(value, indent=2) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)


def select_panel(rows):
    """One ascending ID per ordinary family/split; four per held-out family, before outcomes."""
    out = []
    for split in ('train', 'test_instance', 'test_family'):
        families = sorted({r['family'] for r in rows if r['split'] == split})
        for family in families:
            candidates = sorted((r for r in rows if r['split'] == split and r['family'] == family), key=lambda r: r['id'])
            out.extend(candidates[:4 if split == 'test_family' else 1])
    return out


def training_args(a, reward, model, init, out):
    """Every arm knob is identical except reward and output; frozen SFT start is explicit."""
    return argparse.Namespace(data=f'zoo:{a.out / "zoo/manifest.jsonl"}@train', truth=None, model=str(model), init=init,
                              out=str(out), lora=0, steps=a.sft_steps if reward is None else a.rl_steps,
                              batch=1, lr=2e-5, max_len=a.max_len, max_new=a.max_new, k=4, temp=1., beta=0., std=False,
                              reward=reward or 'f1', eval=0, threads=1, seed=a.seed, log_every=1,
                              device=getattr(a, 'device', 'cpu'))


def require_time(seconds):
    deadline = float(os.environ.get('DEADLINE', 'inf'))
    if deadline - time.time() < seconds + 180:
        raise SystemExit(75)  # saved stages remain; links=1 prevents unapproved extension


def stage_done(out, name):
    path = out / (name + '.done.json')
    if not path.exists():
        return False
    record = json.loads(path.read_text())
    directory = out / name
    if record['files'] != hashes(directory):
        raise ValueError(f'completed stage changed: {name}')
    return True


def fork_sft(a, parent):
    """Import only a verified completed SFT, never optimizer/partial RL state.

    The new directory receives a new source/package/platform contract in prepare;
    the original contract and every imported byte remain separately attributable.
    """
    parent = Path(parent).resolve()
    if a.out.exists() and any(a.out.iterdir()):
        raise ValueError('fork requires a new empty output directory')
    if not parent.is_dir() or a.out.resolve() == parent:
        raise ValueError('fork parent must be a distinct completed run')
    if any(p.name.startswith(('unit_suff', 'wiring_suff')) for p in parent.iterdir()):
        raise ValueError('fork parent contains prior RL state; use the frozen SFT artifact')
    validated_snapshot = hashes(parent)
    contract = json.loads((parent / 'contract.json').read_text())
    expected = {'model': MODEL, 'hf_revision': REVISION, 'sft_steps': a.sft_steps,
                'rl_steps_per_arm': a.rl_steps, 'seed': a.seed, 'max_len': a.max_len,
                'max_new': a.max_new, 'show_active': False, 'batch': 1, 'k': 4,
                'lr': 2e-5, 'beta': 0., 'threads': 1, 'per_family': 35}
    if any(contract.get(k) != v for k, v in expected.items()):
        raise ValueError('fork SFT configuration mismatch')
    if not stage_done(parent, 'sft'):
        raise ValueError('fork requires a completed SFT manifest')
    if not any(n.endswith('.safetensors') for n in hashes(parent / 'sft')):
        raise ValueError('fork SFT has no model checkpoint')
    data = json.loads((parent / 'data.json').read_text())
    panel_ids = {r['id'] for r in data['evaluation_panel']}
    if len(panel_ids) != 24 or len(data['evaluation_panel']) != 24 or data['files'] != hashes(parent / 'zoo'):
        raise ValueError('fork frozen zoo/panel changed or incomplete')
    fit = json.loads((parent / 'token_fit.json').read_text())
    inputs = [json.loads(l) for l in (parent / 'train_inputs.jsonl').read_text().splitlines() if l.strip()]
    if (not fit.get('all_fit') or fit['context'] != a.max_len or fit['generation_cap'] != a.max_new
            or fit['train_n'] != len(inputs) or {r['id'] for r in fit['rows']} != {r['id'] for r in inputs}
            or fit['max_pair_tokens'] > a.max_len or fit['max_prompt_tokens'] + a.max_new > a.max_len
            or fit['max_completion_tokens'] > a.max_new):
        raise ValueError('fork token/input fit mismatch')
    evaluated = {}
    for name in ('gt', 'empty', 'sft'):
        rows = [json.loads(l) for l in (parent / (name + '_eval.jsonl')).read_text().splitlines() if l.strip()]
        if len(rows) != 24 or {r['id'] for r in rows} != panel_ids:
            raise ValueError('fork requires all 24 saved evaluation records')
        if name == 'sft' and any(r['policy_checkpoint_manifest_sha256'] != digest(parent / 'sft.done.json') for r in rows):
            raise ValueError('fork SFT evaluation checkpoint mismatch')
        evaluated[name] = rows
    integrity = json.loads((parent / 'teacher_integrity.json').read_text())
    actual = [{'id': r['id'], 'valid': r['valid'],
               'max_stated_kl_bits': max(t['kl_bits'] for t in r['stated']['per_text']) if r['valid'] else None}
              for r in evaluated['gt']]
    if (integrity['predeclared_tolerance_bits'] != 1e-3 or integrity['rows'] != actual
            or any(not r['valid'] or not 0 <= r['max_stated_kl_bits'] <= 1e-3 for r in actual)):
        raise ValueError('fork ground-truth integrity failed')
    required = ['sft', 'sft.done.json', 'zoo', 'data.json', 'train_inputs.jsonl', 'token_fit.json',
                'teacher_integrity.json', 'gt_eval.jsonl', 'empty_eval.jsonl', 'sft_eval.jsonl', 'base_model_hashes.json']
    required += [n for n in ('sft.started.json', 'dependency-lock.txt') if (parent / n).exists()]
    # Verify all paths before copying, then record exact import hashes (including old contract).
    for n in required:
        if not (parent / n).exists():
            raise ValueError(f'fork missing immutable artifact: {n}')
    # Freeze source identities before copying, compare every copied byte, then publish.
    if hashes(parent) != validated_snapshot:
        raise ValueError('fork parent changed during validation')
    source_hashes = {n: hashes(parent / n) if (parent / n).is_dir() else digest(parent / n)
                     for n in required + ['contract.json']}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=a.out.name + '.fork-', dir=a.out.parent))
    try:
        evidence = staging / 'parent_evidence'
        evidence.mkdir()
        for n in required:
            source = parent / n
            # Only checkpoint and zoo become active artifacts. All old evaluation,
            # fit/package/model reports remain immutable parent evidence.
            destination = staging / n if n in ('sft', 'sft.done.json', 'zoo', 'data.json', 'train_inputs.jsonl') else evidence / n
            shutil.copytree(source, destination) if source.is_dir() else shutil.copy2(source, destination)
            copied = hashes(destination) if source.is_dir() else digest(destination)
            if copied != source_hashes[n]:
                raise ValueError(f'fork copy changed: {n}')
        shutil.copy2(parent / 'contract.json', evidence / 'contract.json')
        if digest(evidence / 'contract.json') != source_hashes['contract.json']:
            raise ValueError('fork parent contract changed during copy')
        after = {n: hashes(parent / n) if (parent / n).is_dir() else digest(parent / n)
                 for n in required + ['contract.json']}
        if after != source_hashes or hashes(parent) != validated_snapshot:
            raise ValueError('fork parent changed during copy')
        write(staging / 'sft_lineage.json', {'schema': 2, 'parent': str(parent),
              'parent_contract_sha256': source_hashes['contract.json'], 'imported_files': hashes(staging),
              'scope': 'new source/platform experiment sharing frozen SFT; parent evaluations archived; no backend bitwise equivalence claim'})
        if a.out.exists():
            a.out.rmdir()  # only an empty destination may be replaced
        staging.rename(a.out)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def verify_lineage(out):
    path = out / 'sft_lineage.json'
    if not path.exists():
        return
    lineage = json.loads(path.read_text())
    for name, expected in lineage['imported_files'].items():
        target = out / name
        if not target.is_file() or digest(target) != expected:
            raise ValueError(f'fork immutable import changed: {name}')
    if digest(out / 'parent_evidence/contract.json') != lineage['parent_contract_sha256']:
        raise ValueError('fork parent contract identity mismatch')


def source_revision(explicit=None):
    """Archives have no .git; record their supplied revision alongside actual file hashes."""
    if explicit is not None:
        if len(explicit) != 40 or any(c not in '0123456789abcdef' for c in explicit):
            raise ValueError('source revision must be a full lowercase 40-hex commit')
        return explicit
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()


def prepare(a):
    verify_lineage(a.out)
    import torch
    from huggingface_hub import snapshot_download
    sys.path[:0] = [str(HERE), str(ROOT / 'experiments/oracle/toys')]
    import render
    import train
    import zoo
    render.SHOW_ACTIVE = False
    os.environ['TOYLOOP_SHOW_ACTIVE'] = '0'
    a.out.mkdir(parents=True, exist_ok=True)
    sources = {str(p.relative_to(ROOT)): digest(p) for base in (HERE, ROOT / 'experiments/oracle/toys', ROOT / 'method/oracle', ROOT / 'method/vpd')
               for p in sorted(base.glob('*.py'))}
    contract = {'schema': 1, 'scope': 'fresh policy experiment on shared-development zoo; reused procedural families/seeds are not pristine heldout',
                'model': MODEL, 'hf_revision': REVISION, 'sources': sources,
                'source_commit_metadata': source_revision(getattr(a, 'source_commit', None)),
                'python': sys.version,
                'packages': {d.metadata['Name']: {'version': d.version,
                    'installed_RECORD_sha256': hashlib.sha256((d.read_text('RECORD') or '').encode()).hexdigest()}
                    for d in importlib.metadata.distributions()},
                'sft_steps': a.sft_steps, 'rl_steps_per_arm': a.rl_steps, 'seed': a.seed,
                'batch': 1, 'k': 4, 'lr': 2e-5, 'beta': 0., 'max_len': a.max_len, 'max_new': a.max_new,
                'show_active': False, 'threads': 1, 'per_family': 35, 'arm_order': ['unit_suff', 'wiring_suff'],
                'evaluation': 'fixed family-stratified panel; one greedy output each, invalid=-1; all rows retained',
                'teacher_panel_max_kl_bits': 1e-3,
                'resume': 'completed stages only; partial stages rejected, no optimizer-reset continuation'}
    if (a.out / 'sft_lineage.json').exists() or getattr(a, 'device', 'cpu') != 'cpu':
        contract.update(device=getattr(a, 'device', 'cpu'), platform=platform.platform())
    if (a.out / 'sft_lineage.json').exists():
        contract['sft_lineage_sha256'] = digest(a.out / 'sft_lineage.json')
    path = a.out / 'contract.json'
    if path.exists():
        if json.loads(path.read_text()) != contract:
            raise ValueError('source/config/package contract changed; use a new run directory')
    else:
        write(path, contract)
    base = Path(snapshot_download(MODEL, revision=REVISION, token=False,
                                allow_patterns=['config.json', 'tokenizer*.json', 'special_tokens_map.json', 'vocab.json', 'merges.txt']))
    model_config = json.loads((base / 'config.json').read_text())
    if a.max_len > model_config['max_position_embeddings']:
        raise ValueError('requested context exceeds pinned model context')
    manifest = a.out / 'zoo/manifest.jsonl'
    frozen = a.out / 'data.json'
    if not frozen.exists():
        if manifest.exists():
            raise ValueError('partial zoo build exists; preserve it and use a new run directory')
        zoo.build(a.out / 'zoo', 35, list(zoo.FAMILIES))
        rows = [json.loads(l) for l in manifest.read_text().splitlines() if l.strip()]
        panel = select_panel(rows)
        write(frozen, {'files': hashes(a.out / 'zoo'), 'evaluation_panel': panel,
                       'split_counts': {s: sum(r['split'] == s for r in rows) for s in ('train', 'test_instance', 'test_family')}})
    data = json.loads(frozen.read_text())
    rows = [json.loads(l) for l in manifest.read_text().splitlines() if l.strip()]
    panel = data['evaluation_panel']
    assert len(panel) == 24 and len({r['id'] for r in panel}) == 24
    assert {r['split'] for r in panel} == {'train', 'test_instance', 'test_family'}
    if data['files'] != hashes(a.out / 'zoo'):
        raise ValueError('frozen zoo data changed')
    torch.set_num_threads(1)
    examples = train.examples(f'zoo:{manifest}@train')
    prompt_file = a.out / 'train_inputs.jsonl'
    prompt_bytes = ''.join(json.dumps({'id': e['meta']['name'], 'prompt': e['prompt'], 'completion': e['completion']}) + '\n' for e in examples).encode()
    if prompt_file.exists():
        assert prompt_file.read_bytes() == prompt_bytes
    else:
        prompt_file.write_bytes(prompt_bytes)
    tok = train.oracle_lm_tokenizer(base)
    lengths = [len(train.oracle_lm.chat_ids(tok, e['prompt'], e['completion'])[0]) for e in examples]
    prompt_lengths = [len(train.oracle_lm.chat_ids(tok, e['prompt'])[0]) for e in examples]
    completion_lengths = [len(tok(e['completion'], add_special_tokens=False)['input_ids']) + 1 for e in examples]
    fits = (max(lengths) <= a.max_len and max(prompt_lengths) + a.max_new <= a.max_len
            and max(completion_lengths) <= a.max_new)
    write(a.out / 'token_fit.json', {'train_n': len(examples), 'max_pair_tokens': max(lengths),
                                   'max_prompt_tokens': max(prompt_lengths), 'max_completion_tokens': max(completion_lengths),
                                   'context': a.max_len, 'generation_cap': a.max_new, 'all_fit': fits,
                                   'rows': [{'id': e['meta']['name'], 'pair_tokens': pair, 'prompt_tokens': prompt,
                                             'completion_tokens': completion}
                                            for e, pair, prompt, completion in zip(examples, lengths, prompt_lengths, completion_lengths)]})
    if not fits:
        raise ValueError('prompt/target exceeds context or generation budget; measured token_fit.json preserved')
    if a.stage != 'preflight':
        base = Path(snapshot_download(MODEL, revision=REVISION, token=False,
                                      allow_patterns=['*.json', '*.safetensors', '*.txt']))
    write(a.out / 'base_model_hashes.json', {'revision': REVISION, 'files': hashes(base),
                                          'scope': 'tokenizer/config only' if a.stage == 'preflight' else 'model and tokenizer'})
    return train, render, base, data


def learn(a, train, base, name, reward=None):
    verify_lineage(a.out)
    if stage_done(a.out, name):
        return
    directory = a.out / name
    if directory.exists() or (a.out / (name + '.started.json')).exists():
        raise ValueError(f'partial training stage {name} retained; do not restart without explicit review')
    init = None if reward is None else str(a.out / 'sft')
    if reward is not None and not stage_done(a.out, 'sft'):
        raise ValueError('RL requires the exact completed frozen SFT start')
    require_time((a.sft_steps * 20) if reward is None else (a.rl_steps * 250))
    write(a.out / (name + '.started.json'), {'started': time.time(), 'reward': reward, 'init': init})
    args = training_args(a, reward, base, init, directory)
    (train.sft if reward is None else train.rl)(args)
    write(a.out / (name + '.done.json'), {'files': hashes(directory), 'completed': time.time(), 'reward': reward,
                                        'start_hashes': hashes(a.out / 'sft') if reward else hashes(base)})


def evaluate_policy(a, train, render, data, name):
    """Append one completed text at a time; reuse a toy native across stated/complete executions."""
    import gc
    import torch
    import circuit
    import reward_exec
    import planted
    checkpoint = a.out / (name + '_eval.jsonl')
    existing = [json.loads(l) for l in checkpoint.read_text().splitlines() if l.strip()] if checkpoint.exists() else []
    seen = {r['id'] for r in existing}
    if len(seen) != len(existing):
        raise ValueError('duplicate evaluation records')
    model = tok = None
    if name not in ('gt', 'empty') and len(seen) < len(data['evaluation_panel']):
        if not stage_done(a.out, name):
            raise ValueError('evaluation requires a completed frozen policy')
        args = training_args(a, None, '', str(a.out / name), a.out / name)
        model, tok = train.load_policy(args)
        model.eval()
    for entry in data['evaluation_panel']:
        if entry['id'] in seen:
            continue
        require_time(120)
        torch.manual_seed(int(hashlib.sha256((str(a.seed) + ':' + entry['id']).encode()).hexdigest()[:15], 16))
        toy = render.from_zoo(entry, a.out / 'zoo')
        prompt = render.render(toy)
        if name == 'gt':
            text, finish, token_ids = render.target_text(toy.truth), 'constructed', None
        elif name == 'empty':
            text, finish, token_ids = 'flowchart LR\n tokX --> out\nEnglish: No unit.', 'constructed', None
        else:
            ids, _ = train.oracle_lm.chat_ids(tok, prompt)
            if len(ids) + a.max_new > a.max_len:
                raise ValueError('evaluation prompt would truncate')
            with torch.no_grad():
                inp = torch.tensor([ids], device=next(model.parameters()).device)
                generated = model.generate(inp, attention_mask=torch.ones_like(inp), do_sample=False, max_new_tokens=a.max_new,
                                           pad_token_id=tok.pad_token_id)[0, len(ids):].tolist()
            token_ids = generated
            finish = 'length' if len(generated) == a.max_new and generated[-1] != tok.eos_token_id else 'stop'
            text = tok.decode(generated, skip_special_tokens=True)
        parsed = circuit.parse(text)
        row = {'id': toy.name, 'split': entry['split'], 'family': entry['family'], 'completion': text, 'token_ids': token_ids,
               'finish': finish, 'units': sorted(parsed.units), 'edges': sorted(parsed.edges), 'parse_bad': parsed.bad,
               'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
               'policy_checkpoint_manifest_sha256': digest(a.out / (name + '.done.json')) if name not in ('gt', 'empty') else None,
               'unit_count': len(parsed.units), 'edge_count': len(parsed.edges)}
        gold_edges = {tuple(e) for e in toy.truth['edges']}
        gold_units = set(toy.truth['units'])
        row['edge_f1'] = 2 * len(parsed.edges & gold_edges) / max(len(parsed.edges) + len(gold_edges), 1)
        row['unit_f1'] = 2 * len(parsed.units & gold_units) / max(len(parsed.units) + len(gold_units), 1)
        valid = parsed.ok and not parsed.bad and finish != 'length'
        row['valid'] = bool(valid)
        nat = None
        if valid:
            try:
                nat = planted.native_of(toy.sd, toy.cfg)
                bank = reward_exec.texts_of(toy)
                for complete, label in ((False, 'stated'), (True, 'same_units_complete')):
                    got = reward_exec.score_wiring(toy, text, nat=nat, texts=bank, complete=complete)
                    row[label] = got
                    scores = [r['suff'] for r in got['per_text']]
                    row[label + '_reward'] = (sum(scores) / len(scores) + min(scores)) / 2
                row['input_bank_sha256'] = hashlib.sha256(json.dumps(bank).encode()).hexdigest()
            except ValueError as exc:
                row.update(valid=False, execution_error=str(exc))
        if not row['valid']:
            row.update(stated_reward=-1., same_units_complete_reward=-1.)
        row['scope'] = 'executed inter-unit/output connections, ordinary and existing counterfactual bank; English/token roles/necessity unverified'
        with checkpoint.open('a') as stream:
            stream.write(json.dumps(row) + '\n');stream.flush();os.fsync(stream.fileno())
        del nat, toy
        gc.collect()
    del model, tok
    gc.collect()
    return [json.loads(l) for l in checkpoint.read_text().splitlines()]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--stage', choices=['all', 'preflight', 'sft', 'unit', 'wiring', 'eval'], default='all')
    parser.add_argument('--sft-steps', type=int, default=300)
    parser.add_argument('--rl-steps', type=int, default=20)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max-new', type=int, default=416)
    parser.add_argument('--max-len', type=int, default=3072)
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='cpu')
    parser.add_argument('--fork-sft', type=Path, help='verified completed SFT parent; requires a new empty --out')
    parser.add_argument('--source-commit', help='full revision metadata for an exported source archive; actual files are also hashed')
    a = parser.parse_args(argv)
    if a.fork_sft is not None:
        fork_sft(a, a.fork_sft)
    train, render, base, data = prepare(a)
    if a.stage == 'preflight':
        return
    # Validate the native execution baseline before any expensive policy learning.
    gt_rows = evaluate_policy(a, train, render, data, 'gt')
    evaluate_policy(a, train, render, data, 'empty')
    integrity = [{'id': row['id'], 'valid': row['valid'],
                  'max_stated_kl_bits': max(r['kl_bits'] for r in row['stated']['per_text']) if row['valid'] else None}
                 for row in gt_rows]
    write(a.out / 'teacher_integrity.json', {'predeclared_tolerance_bits': 1e-3, 'rows': integrity,
                                           'scope': 'fixed panel original teacher wiring under the common input bank; no filtering or truth repair'})
    if any(not row['valid'] or row['max_stated_kl_bits'] > 1e-3 for row in integrity):
        raise ValueError('ground-truth wiring exceeds predeclared 1e-3-bit point tolerance; evidence preserved, no policy training')
    if a.stage in ('all', 'sft'):
        learn(a, train, base, 'sft')
        evaluate_policy(a, train, render, data, 'sft')  # fixed format-readiness evidence, never select another checkpoint
    if (a.out / 'sft_lineage.json').exists() and a.stage in ('all', 'unit', 'wiring'):
        evaluate_policy(a, train, render, data, 'sft')
    if a.stage in ('all', 'unit'):
        learn(a, train, base, 'unit_suff', 'unit_suff')
        evaluate_policy(a, train, render, data, 'unit_suff')
    if a.stage in ('all', 'wiring'):
        learn(a, train, base, 'wiring_suff', 'wiring_suff')
        evaluate_policy(a, train, render, data, 'wiring_suff')
    if a.stage in ('all', 'eval'):
        outputs = {name: evaluate_policy(a, train, render, data, name) for name in ('gt', 'empty', 'sft', 'unit_suff', 'wiring_suff')}
        summary = {name: {'n': len(rows), 'valid': sum(r['valid'] for r in rows),
                          'stated_reward_all': sum(r['stated_reward'] for r in rows) / len(rows),
                          'same_units_complete_reward_all': sum(r['same_units_complete_reward'] for r in rows) / len(rows)}
                   for name, rows in outputs.items()}
        for name, rows in outputs.items():
            summary[name]['by_split'] = {}
            for split in ('train', 'test_instance', 'test_family'):
                selected = [r for r in rows if r['split'] == split]
                summary[name]['by_split'][split] = {
                    'n': len(selected), 'valid': sum(r['valid'] for r in selected),
                    'stated_reward_all': sum(r['stated_reward'] for r in selected) / len(selected),
                    'same_units_complete_reward_all': sum(r['same_units_complete_reward'] for r in selected) / len(selected),
                    'edge_f1_all': sum(r['edge_f1'] for r in selected) / len(selected)}
        write(a.out / 'summary.json', summary)
        print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
