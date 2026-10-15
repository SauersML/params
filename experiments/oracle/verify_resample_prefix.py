"""Paired strict-bit verification of actual all-position TRAIN resampling; no production enable.

Uses fresh exact/reference caches for every mode. Timings include prefix capture
and distribution diagnostics; no warmed-cache or amortized-setup speed claim.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'method/oracle'), str(ROOT/'method/vpd')]


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda: stream.read(1024*1024), b''):
            digest.update(data)
    return digest.hexdigest()


def strict_pair_equal(whole, prefix, batches):
    """Every actual score and distribution must match bit for bit; no tolerance."""
    return (whole['scores'] == prefix['scores'] and whole['rank'] == prefix['rank']
            and all(batch[key] for batch in batches for key in
                    ('log_prob_equal', 'probability_equal', 'kl_equal')))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model-dir', type=Path, required=True)
    ap.add_argument('--uv', type=Path, required=True)
    ap.add_argument('--tasks-dir', type=Path, required=True)
    ap.add_argument('--donors', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    import torch
    import carry
    import goal_eval
    import native
    import vpd_model
    torch.set_num_threads(2)
    vpd_model.TARGET_DIR = a.model_dir
    vpd_model.VPD_PTH = a.out.parent/'forbidden_ci_checkpoint.pth'
    assert not vpd_model.VPD_PTH.exists()
    native.UV = a.uv
    assert sha(a.donors) == 'bbd11f1543d76f3e8875062133f836e34e79d58008889d4da8c4eb8a93809ebb'
    donors = json.loads(a.donors.read_text())
    assert [name for name, _ in donors] == [f'text{i}' for i in range(6000, 6400)]
    provenance = {str(p): sha(p) for p in (a.donors, a.uv,
        a.model_dir/'model_step_99999.safetensors', a.model_dir/'model_config.yaml')}
    nat = native.Native('cpu')
    assert not nat.has_importance
    ge = goal_eval.GoalEval(nat, chunk=8)
    ge.donor_pools['search'] = donors
    rows = []
    original_kl = carry.kl
    try:
        for name in ('text6009', 'text6012', 'text6035'):
            path = a.tasks_dir/(name+'.json')
            task = json.loads(path.read_text())
            assert task['id'] == name and not task.get('behavior')
            provenance[str(path)] = sha(path)
            ids, targets = native.text(path)
            ref = nat.reference([ids], targets)
            seed = native.task_seed(name)
            baseline_lp, baseline_kl, per_batch = [], [], []
            runs = {}
            for mode in ('whole', 'prefix'):
                # Same exact rows and same nominal/final batch shapes; no cache reuse.
                nat._refs.clear(); nat._refs_bytes = 0
                ge.resample_prefix = mode == 'prefix'
                tx = SimpleNamespace(ids=ids, targets=targets, T=len(ids), seed=seed, ref=ref)
                index = [0]
                def checked_kl(lp, reference):
                    value = original_kl(lp, reference)
                    at = index[0]; index[0] += 1
                    if mode == 'whole':
                        baseline_lp.append(lp.detach().clone())
                        baseline_kl.append(value.detach().clone())
                    else:
                        before, previous = baseline_lp[at], baseline_kl[at]
                        per_batch.append({'batch': len(lp), 'log_prob_equal': torch.equal(lp, before),
                            'probability_equal': torch.equal(lp.exp(), before.exp()),
                            'kl_equal': torch.equal(value, previous),
                            'max_abs_log_prob_error': float((lp-before).abs().max()),
                            'max_abs_probability_error': float((lp.exp()-before.exp()).abs().max()),
                            'max_abs_kl_error': float((value-previous).abs().max()),
                            'whole_row_kl_bits': previous.tolist(), 'prefix_row_kl_bits': value.tolist()})
                    # Prevent cross-batch reference hits in either timed mode.
                    nat._refs.clear(); nat._refs_bytes = 0
                    return value
                carry.kl = checked_kl
                started = time.perf_counter()
                scores = ge.resample_influence(tx, 'search', k=4)
                elapsed = time.perf_counter()-started
                carry.kl = original_kl
                assert index[0] == len(baseline_lp)
                runs[mode] = {'seconds_including_prefix_and_diagnostics': elapsed,
                              'scores': scores, 'rank': sorted(scores, key=lambda q: (-scores[q], q)),
                              'row_kls': [v.tolist() for v in baseline_kl] if mode == 'whole' else None}
            equal = strict_pair_equal(runs['whole'], runs['prefix'], per_batch)
            rows.append({'task': name, 'T': len(ids), 'targets': targets, 'k': 4,
                         'all_positions': True, 'bit_exact': equal, 'batches': per_batch, 'runs': runs})
    finally:
        carry.kl = original_kl
    result = {'torch': torch.__version__, 'device': 'cpu', 'provenance': provenance,
              'strict_bit_exact': all(r['bit_exact'] for r in rows), 'production_enabled': False,
              'default_resample_prefix': False,
              'source_sha256': {str(p.relative_to(ROOT)): sha(p) for p in
                  (ROOT/'method/oracle/goal_eval.py', ROOT/'method/oracle/native.py',
                   ROOT/'method/vpd/vpd_model.py', Path(__file__).resolve())}, 'rows': rows}
    with a.out.open('x') as stream:
        json.dump(result, stream, indent=2); stream.write('\n')
    print(json.dumps({'strict_bit_exact': result['strict_bit_exact'], 'out': str(a.out)}))
    if not result['strict_bit_exact']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
