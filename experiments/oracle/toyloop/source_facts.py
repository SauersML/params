"""Static checked source facts for constructed Multihop toys; never runs a model.

The output is a typed weight/interface description, not generated English or measured
intervention answers. It fails closed when metadata and actual head blocks disagree.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

FORWARD_SOURCE = {'experiments/oracle/toyloop/render.py': 'd23911e5ebb6c8ddd8fdff44a001b08f6880c16279c2bb55481530e863cd7318',
                  'experiments/oracle/toys/zoo.py': '07a059de9b70e24216986cf36da02505904ef56624ae6925bc8bd88344847a3d'}


def head_id(value):
    if (not isinstance(value, (tuple, list)) or len(value) != 2
            or not all(isinstance(x, int) and not isinstance(x, bool) and x >= 0 for x in value)):
        raise ValueError('head requires numeric [layer, head]')
    return f'L{value[0]}H{value[1]}'


def equal(actual, expected, name, trig=False):
    matches = (actual.shape == expected.shape and np.isfinite(actual).all())
    if matches:
        matches = (np.array_equal(actual == 0, expected == 0) and np.allclose(actual, expected, rtol=0., atol=1e-6)
                   if trig else np.array_equal(actual, expected))
    if not matches:
        raise ValueError('frozen construction mismatch: ' + name)


def multihop_facts(sd, cfg, params):
    """Verify primitive blocks against actual arrays, including the direct residual readout."""
    vocab, width, layers, heads = 32, 64, 2, 4
    if any(cfg[k] != v for k, v in [('n_embd', width), ('n_layer', layers), ('n_head', heads)]):
        raise ValueError('only the frozen32-vocabulary constructed architecture is supported')
    sd = {k: np.asarray(v) for k, v in sd.items()}
    eps = float(cfg.get('rms_norm_eps', 1e-6))
    if not all(np.isfinite(value).all() for value in sd.values()) or not math.isfinite(eps) or eps <= 0:
        raise ValueError('finite weights and positive normalization epsilon required')
    if sd['wte.weight'].shape != (vocab, width):
        raise ValueError('embedding shape mismatch')
    embedding = np.zeros_like(sd['wte.weight'])
    embedding[:, 0] = 8.
    embedding[:, 1:33] = np.eye(vocab)
    equal(sd['wte.weight'], embedding, 'token embeddings')
    readout = np.full(width, 5., dtype=embedding.dtype)
    readout[0] = 0.
    equal(sd['ln_f.weight'], readout, 'final RMSNorm weights')
    if 'lm_head.weight' in sd:
        equal(sd['lm_head.weight'], embedding, 'tied readout')
    equal((embedding * readout) @ embedding.T, np.eye(vocab) * 5., 'direct embedding/readout Gram')
    first, second = head_id(params['h0']), head_id(params['h1'])
    if params['h0'][0] != 0 or params['h1'][0] != 1:
        raise ValueError('copy roles require first/second layer in this construction')
    pool, codes = params['pool'], params['code']
    if len(pool) != len(codes) or len(set(pool)) != len(pool) or len(set(codes)) != len(codes):
        raise ValueError('pool/code binding must be one-to-one')
    if not all(0 <= t < vocab for t in pool) or not all(33 <= c < width for c in codes):
        raise ValueError('token/code coordinates outside construction')
    hd = width // heads
    facts = []
    for value, offset, role in [(params['h0'], params['d1'], 'token_to_code'),
                                (params['h1'], params['d2'], 'code_to_token')]:
        layer, head = value
        if head >= heads:
            raise ValueError('head index outside architecture')
        sl = slice(head * hd, (head + 1) * hd)
        q, k, v = (np.zeros((hd, width), dtype=embedding.dtype) for _ in range(3))
        o = np.zeros((width, hd), dtype=embedding.dtype)
        for j in (0, 1, 2):
            omega = float(cfg.get('rotary_base', 10000.)) ** (-j / (hd / 2))
            q[j, 0] = 11.5 / 8 * math.cos(offset * omega)
            q[j + hd // 2, 0] = -11.5 / 8 * math.sin(offset * omega)
            k[j, 0] = 11.5 / 8
        for channel, (token, code) in enumerate(zip(pool, codes)):
            v[channel, 1 + token if role == 'token_to_code' else code] = 1.
            o[code if role == 'token_to_code' else 1 + token, channel] = 1. if role == 'token_to_code' else 3.
        prefix = f'h.{layer}.attn.'
        for name, expected in [('q_proj', q), ('k_proj', k), ('v_proj', v)]:
            equal(sd[prefix + name + '.weight'][sl], expected, head_id(value) + '/' + name, trig=name in ('q_proj', 'k_proj'))
        equal(sd[prefix + 'o_proj.weight'][:, sl], o, head_id(value) + '/o_proj')
        facts.append({'unit_id': head_id(value), 'layer': layer, 'head': head, 'role': role,
                      'position_offset_from_verified_QK': offset,
                      'channels': [{'token_id': t, 'token_coordinate': t + 1, 'code_coordinate': c}
                                   for t, c in zip(pool, codes)],
                      'output_gain': 1. if role == 'token_to_code' else 3.,
                      'actual_QKVO_blocks_verified': True})
    token_writers, code_writers, mlp_token_or_code_writers = [], [], []
    token_dims, code_dims = list(range(1, 33)), list(codes)
    for layer in range(layers):
        for head in range(heads):
            block = sd[f'h.{layer}.attn.o_proj.weight'][:, head * hd:(head + 1) * hd]
            if np.any(block[token_dims]):
                token_writers.append(head_id([layer, head]))
            if np.any(block[code_dims]):
                code_writers.append(head_id([layer, head]))
        down = sd[f'h.{layer}.mlp.down_proj.weight']
        for neuron in np.flatnonzero(np.any(down[token_dims + code_dims] != 0, axis=0)):
            mlp_token_or_code_writers.append(f'L{layer}N{neuron}')
    if token_writers != [second] or code_writers != [first] or mlp_token_or_code_writers:
        raise ValueError('additional token/code writers invalidate the residual fallback derivation')
    return {'schema': 1, 'scope': 'Static facts of the frozen constructed Multihop family, not arbitrary LM semantics',
            'forward_contract': 'cd8d4a9c2dfc0d3b28bc6604e0549e16aceb3aa4; residual-add decoder and tied readout',
            'canonical_heads': {'first_hop': first, 'second_hop': second}, 'head_facts': facts,
            'residual_readout_path': [{'op': 'embedding', 'matrix': 'wte.weight', 'token_identity_coordinate': '1+token_id', 'bias_coordinate': 0, 'bias_value': 8.},
                                      {'op': 'residual_add', 'preserves_initial_embedding': True, 'sites': ['attention', 'MLP']},
                                      {'op': 'final_RMSNorm', 'token_coordinate_gain': 5., 'bias_coordinate_gain': 0., 'denominator': 'common_positive_scalar'},
                                      {'op': 'tied_readout', 'matrix': 'wte.weight.T', 'embedding_readout_Gram': '5*identity'}],
            'verified_write_sets': {'token_head_writers': token_writers, 'code_head_writers': code_writers,
                                    'MLP_token_or_code_writers': mlp_token_or_code_writers, 'initial_code_coordinates_zero': True},
            'head_output_transform': {'operation': 'multiply_o_proj_input_columns', 'all_positions': True,
                                      'columns': 'head*16:(head+1)*16', 'factors': [0., .5], 'other_weights': 'unchanged',
                                      'downstream_routing_and_norms': 'recomputed from transformed weights'},
            'formal_zero_scale_derivation': {'if_named_head_is': [first, second],
                                             'remaining_token_coordinates': 'initial_embedding_only',
                                             'argmax_identity': 'last_input_token', 'basis': 'verified exclusive write sets and diagonal tied readout'},
            'half_scale_outcome': 'No top/distribution guarantee asserted; only exact weight transform specified',
            'unused_part_scope': 'Other writes may affect normalization; no global irrelevance guarantee asserted',
            'contains_measured_outputs_or_prose': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', required=True, type=Path)
    parser.add_argument('--record', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--source', required=True, type=Path, help='Frozen cd8 executor source directory; exact forward/construction hashes checked')
    args = parser.parse_args()
    assert not args.out.exists(), 'do not overwrite frozen facts'
    for relative, expected in FORWARD_SOURCE.items():
        if hashlib.sha256((args.source / relative).read_bytes()).hexdigest() != expected:
            raise ValueError('forward/construction source mismatch: ' + relative)
    import torch
    stored = torch.load(args.weights, map_location='cpu', weights_only=True)
    params = json.loads(args.record.read_text())['params']
    facts = multihop_facts({k: v.detach().cpu().numpy() for k, v in stored['sd'].items()}, stored['cfg'], params)
    facts['source_bindings'] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in (args.weights, args.record, Path(__file__))}
    facts['forward_source_bindings'] = FORWARD_SOURCE
    args.out.write_text(json.dumps(facts, indent=2) + '\n')
