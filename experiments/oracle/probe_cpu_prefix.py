"""Bounded exactness probe; never enables a production optimization.

Prepared for explicit execution on CPU with published vpd4l weights and TRAIN6009.
Twenty-two forward-equivalent calls: base(1), prefix captures(3), paired runs(18).
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / 'method/oracle'), str(REPO / 'method/vpd')]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model-dir', type=Path, required=True)
    ap.add_argument('--task', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    def sha(path):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    provenance = {"task_sha256": sha(a.task),
                  "model_sha256": sha(a.model_dir / "model_step_99999.safetensors"),
                  "config_sha256": sha(a.model_dir / "model_config.yaml")}
    task = json.loads(a.task.read_text())
    assert task['id'] == 'text6009' and a.task.stem == 'text6009', 'TRAIN6009 only'
    assert not task.get('behavior'), 'plain published model only'
    import torch
    import native
    import carry
    import vpd_model
    torch.set_num_threads(2)
    vpd_model.TARGET_DIR = a.model_dir
    tg = vpd_model.load_target('cpu')  # No decomposition, CI, labels, or oracle construction.
    ids, targets = native.text(a.task)
    T = len(ids)
    assert targets and min(targets) > 1
    positions = sorted({0, min(targets) // 2, min(targets) - 1})
    assert len(positions) == 3
    def logits(batch, start=0, prefix=None):
        return torch.log_softmax(tg.unembed(tg.hidden(batch, start, prefix)[:,
            [t - start for t in targets]]).float(), -1)
    def kl(left, right):
        # Same float32 direction as resample_influence: changed || original.
        return carry.kl(left, right)
    rows = []
    with torch.no_grad():
        base = logits(torch.tensor([ids]))
        for B in (1, 3, 32):  # singleton, partial batch, production nominal batch
            prefix = tg.keys_values(torch.tensor([ids] * B))
            full_kls, window_kls = [], []
            for q in positions:
                changed = [list(ids) for _ in range(B)]
                for b, seq in enumerate(changed):
                    seq[q] = (seq[q] + b + 1) % tg.n_tokens
                batch = torch.tensor(changed)
                t = time.perf_counter(); full = logits(batch); whole_s = time.perf_counter() - t
                start = max(0, min(q, T - 64))
                t = time.perf_counter(); window = logits(batch, start, prefix); window_s = time.perf_counter() - t
                fk, wk = kl(full, base.expand_as(full)), kl(window, base.expand_as(window))
                full_kls.append(float(fk.mean())); window_kls.append(float(wk.mean()))
                rows.append({'batch': B, 'changed_position': q, 'window_start': start,
                    'suffix_tokens': T - start, 'whole_s': whole_s, 'window_s': window_s,
                    'log_prob_equal': torch.equal(full, window), 'probability_equal': torch.equal(full.exp(), window.exp()),
                    'kl_equal': torch.equal(fk, wk), 'max_abs_log_prob_error': float((full-window).abs().max()),
                    'max_abs_probability_error': float((full.exp()-window.exp()).abs().max()),
                    'max_abs_kl_error': float((fk-wk).abs().max()),
                    'whole_kl_bits': fk.tolist(), 'window_kl_bits': wk.tolist()})
            rank = lambda scores: sorted(range(len(scores)), key=lambda i: (-scores[i], i))
            for row in rows[-3:]:
                row['position_rank_equal'] = rank(full_kls) == rank(window_kls)
    result = {'task': 'text6009', 'device': 'cpu', 'torch': torch.__version__, 'T': T,
              'targets': targets, 'forward_equivalent_calls': 22, 'production_enabled': False,
              'provenance': provenance,
              'all_equal': all(r[k] for r in rows for k in ('log_prob_equal', 'probability_equal', 'kl_equal', 'position_rank_equal')),
              'rows': rows}
    with a.out.open('x') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'all_equal': result['all_equal'], 'out': str(a.out)}))


if __name__ == '__main__':
    main()
