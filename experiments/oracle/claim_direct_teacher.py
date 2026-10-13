"""Direct versus baseline-anchored fidelity of frozen real VPD TRAIN teachers.

All selected edits/contexts were used in previous development work. This is a
diagnostic of existing circuits, not an independent generalization evaluation.
"""
import argparse
import contextlib
import hashlib
import json
from pathlib import Path
import sys
import time

SELECTED = {'text6008': (3268, 1.220703125),
            'text6011': (2412, -.27197265625),
            'text6034': (1833, -.52587890625)}
SITE = 'h.3.mlp.down_proj'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def run(repo, payload, out):
    import torch
    from safetensors.torch import save_file
    sys.path[:0] = [str(repo / 'method/oracle'), str(repo / 'method/vpd')]
    import native
    import goal_eval
    import vpd_model
    import mech
    import engine

    if out.exists():
        raise ValueError('Immutable diagnostic output already exists')
    contract = json.loads((payload / 'protocol.json').read_text())
    for name, expected in contract['files'].items():
        if sha(payload / name) != expected:
            raise ValueError('Frozen input mismatch: ' + name)
    panel = json.loads((payload / 'panel.json').read_text())
    if set(panel['tasks']) != set(SELECTED):
        raise ValueError('Fixed three TRAIN contexts required')
    paths = {'target': vpd_model.TARGET_DIR / 'model_step_99999.safetensors',
             'config': vpd_model.TARGET_DIR / 'model_config.yaml',
             'tokenizer': mech.VPD4L_TOKENIZER,
             'decomposition': vpd_model.VPD_PTH if vpd_model.VPD_PTH.exists() else native.UV}
    if {k: sha(p) for k, p in paths.items()} != contract['model_content_sha256']:
        raise ValueError('Model/decomposition asset mismatch')
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    nat = native.Native('cpu')
    ge = goal_eval.GoalEval(nat, semantics='c')
    site = nat.target.site(SITE)
    saved = {key: getattr(site, key).clone() for key in ('W', 'U', 'V', 'delta_T')}

    def state_hashes():
        return {key: hashlib.sha256(getattr(site, key).detach().cpu().contiguous()
            .view(torch.uint8).numpy().tobytes()).hexdigest() for key in saved}

    initial_state = state_hashes()

    # Validate every case before any model forward. The production parser drops
    # invalid entries and stranded nodes; record that transformation explicitly.
    cases = {}
    for task, (component, selected_scale) in SELECTED.items():
        p = panel['tasks'][task]
        ids, targets = p['token_ids'], p['target_positions']
        if not 6000 <= int(task[4:]) <= 7999 or targets != [len(ids) - 1]:
            raise ValueError('TRAIN final-position inputs only')
        if not 0 <= component < nat.C[SITE]:
            raise ValueError('Selected component outside decomposition')
        answer = json.loads((payload / (task + '.teacher.json')).read_text())
        graph, report = goal_eval.circuit(answer, targets, nat.C, len(ids))
        if any(report[k] for k in ('invalid_nodes', 'invalid_edges', 'invalid_variables')):
            raise ValueError(f'Invalid frozen teacher {task}: {report}')
        if not graph.count() or getattr(graph, 'variables', []):
            raise ValueError('Nonempty component graph without variable hooks required')
        cases[task] = (ids, targets, graph, report, len(answer.get('nodes', [])))

    def clear():
        nat.vpd.clear()
        nat._refs.clear()
        nat._refs_bytes = 0
        nat._rec = (None, None, {})
        ge._xk.clear()
        nat.__dict__.pop('_sensitivity_kept', None)

    @contextlib.contextmanager
    def edit(component, scale):
        clear()
        site.W.copy_(saved['W'] + (scale - 1) * torch.outer(saved['U'][component], saved['V'][:, component]))
        site.U[component].mul_(scale)
        try:
            yield
        finally:
            for key, value in saved.items():
                getattr(site, key).copy_(value)
            clear()
            if any(not torch.equal(getattr(site, k), v) for k, v in saved.items()):
                raise RuntimeError('Weight/basis restoration failed')

    def normalized(x):
        if not bool(torch.isfinite(x).all()):
            raise RuntimeError('Nonfinite distribution')
        x = x.double()
        z = x.logsumexp(-1, keepdim=True)
        if float(z.abs().max()) > 1e-4:
            raise RuntimeError('Unexpectedly unnormalized log probabilities')
        return x - z

    def kl(p, q):
        if p.shape != q.shape:
            raise RuntimeError('Distribution shape mismatch')
        p, q = normalized(p), normalized(q)
        value = float((p.exp() * (p - q)).sum() / __import__('math').log(2))
        if value < -1e-10:
            raise RuntimeError('Negative normalized KL')
        return max(0., value)

    def dense(ids):
        clear()
        return torch.log_softmax(nat.vpd.target_forward(torch.tensor([ids]))[:, -1:].float(), -1).clone()

    out.mkdir()
    (out / 'contract.json').write_text(json.dumps({**contract,
        'runner_sha256': sha(__file__), 'executor_sources': {m.__name__: sha(m.__file__) for m in
            (native, goal_eval, vpd_model, engine)},
        'selected': SELECTED, 'scales': '1,0,.5,previously_selected_factor',
        'teacher_parsing': {task: {'report': case[3], 'source_nodes': case[4],
            'executable_nodes': case[2].count()} for task, case in cases.items()},
        'initial_edited_site_state_sha256': initial_state,
        'scope': 'Adaptive TRAIN contexts and final-down edits; original graph versus complete same-node control; own attention/norms; live and off remainders explicitly separate; no generalization certificate',
        'kl_convention': 'FP64 renormalization of saved FP32 log probabilities; original normalization residuals reported'}, indent=2))
    started = time.monotonic()
    rows = []
    for task, (component, selected_scale) in SELECTED.items():
        ids, targets, graph, parse_report, source_nodes = cases[task]
        graphs = {'stated': graph, 'same_nodes_complete': native.Graph(graph.node_set(), complete=True)}
        p0 = dense(ids)
        before = {}
        for name, g in graphs.items():
            for live in (False, True):
                clear()
                before[name, live] = ge.graph_ref(ids, targets, g, live=live, norms={}).clone()
        # Full decomposition + live remainder is an independent integrity control.
        full_masks = {name: torch.ones(len(ids), nat.C[name], dtype=torch.bool) for name in nat.names}
        for scale in (1., 0., .5, selected_scale):
            with edit(component, scale):
                pe = dense(ids)
                full = ge.runs(ids, targets, [full_masks], True).clone()
                integrity = kl(pe, full)
                if integrity > 1e-3:
                    raise RuntimeError('Complete edited decomposition disagrees with literal model')
                for name, g in graphs.items():
                    for live in (False, True):
                        clear()
                        q = ge.graph_ref(ids, targets, g, live=live, norms={}).clone()
                        q0 = before[name, live]
                        if scale == 1. and (not torch.equal(pe, p0) or not torch.equal(q, q0)):
                            raise RuntimeError('No-op edit changed a distribution')
                        anchored = torch.log_softmax(p0 + q - q0, -1)
                        filename = f'{task}_{name}_{int(live)}_{scale}.safetensors'
                        save_file({k: v.contiguous() for k, v in dict(
                            p0=p0, pe=pe, q0=q0, qe=q, anchored=anchored, full=full).items()}, str(out/filename))
                        row = dict(task=task, component=component, scale=scale, candidate=name,
                            remainders='live' if live else 'off', nodes=g.count(),
                            kl_unedited=kl(p0, q0), kl_direct=kl(pe, q),
                            kl_anchored=kl(pe, anchored), kl_nochange=kl(pe, p0),
                            integrity_kl=integrity, full_top=int(pe.argmax(-1).item()),
                            candidate_top=int(q.argmax(-1).item()),
                            normalization_residuals={k: float(v.double().logsumexp(-1).abs().max())
                                for k, v in dict(p0=p0, pe=pe, q0=q0, qe=q, full=full).items()},
                            distributions=filename, distributions_sha256=sha(out/filename))
                        rows.append(row)
                        with (out/'rows.jsonl').open('a') as f:
                            f.write(json.dumps(row)+'\n')
                        print(json.dumps({k: row[k] for k in ('task','scale','candidate','remainders','kl_direct','kl_anchored')}), flush=True)
        del full_masks
    if len(rows) != 48:
        raise RuntimeError('Incomplete fixed Cartesian panel')
    if any(not torch.equal(getattr(site, k), v) for k, v in saved.items()):
        raise RuntimeError('Final restoration failed')
    final_state = state_hashes()
    if final_state != initial_state:
        raise RuntimeError('Final site hashes differ from initial state')
    if {k: sha(p) for k, p in paths.items()} != contract['model_content_sha256']:
        raise RuntimeError('Model/decomposition assets changed during diagnostic')
    (out/'complete.json').write_text(json.dumps(dict(rows=len(rows), physical_worlds=12,
        bitwise_restored=True, final_edited_site_state_sha256=final_state,
        seconds=time.monotonic()-started))+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('repo', 'payload', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    run(args.repo, args.payload, args.out)
