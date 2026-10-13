"""Static matrix fixtures only; no models, decomposition or forwards."""
import math
import numpy as np
import pytest
from source_facts import head_id, multihop_facts


def fixture():
    cfg = {'n_embd': 64, 'n_layer': 2, 'n_head': 4, 'rotary_base': 10000., 'rms_norm_eps': 1e-6}
    params = {'h0': [0, 1], 'h1': [1, 2], 'd1': 2, 'd2': 3, 'pool': [3, 7], 'code': [35, 36]}
    embedding = np.zeros((32, 64), dtype=np.float32)
    embedding[:, 0] = 8.
    embedding[np.arange(32), 1 + np.arange(32)] = 1.
    sd = {'wte.weight': embedding, 'ln_f.weight': np.array([0.] + [5.] * 63, dtype=np.float32)}
    for layer in range(2):
        for name in ('q_proj', 'k_proj', 'v_proj', 'o_proj'):
            sd[f'h.{layer}.attn.{name}.weight'] = np.zeros((64, 64), dtype=np.float32)
        sd[f'h.{layer}.mlp.down_proj.weight'] = np.zeros((64, 256), dtype=np.float32)
    for layer, head, offset, token_to_code in [(0, 1, 2, True), (1, 2, 3, False)]:
        r = head * 16
        for j in (0, 1, 2):
            omega = 10000. ** (-j / 8)
            sd[f'h.{layer}.attn.q_proj.weight'][r + j, 0] = 11.5 / 8 * math.cos(offset * omega)
            sd[f'h.{layer}.attn.q_proj.weight'][r + j + 8, 0] = -11.5 / 8 * math.sin(offset * omega)
            sd[f'h.{layer}.attn.k_proj.weight'][r + j, 0] = 11.5 / 8
        for channel, (token, code) in enumerate([(3, 35), (7, 36)]):
            sd[f'h.{layer}.attn.v_proj.weight'][r + channel, 1 + token if token_to_code else code] = 1.
            sd[f'h.{layer}.attn.o_proj.weight'][code if token_to_code else token + 1, r + channel] = 1. if token_to_code else 3.
    return sd, cfg, params


def test_checked_head_ids_and_direct_residual_path_are_explicit():
    facts = multihop_facts(*fixture())
    assert facts['canonical_heads'] == {'first_hop': 'L0H1', 'second_hop': 'L1H2'}
    assert [part['op'] for part in facts['residual_readout_path']] == ['embedding', 'residual_add', 'final_RMSNorm', 'tied_readout']
    assert facts['formal_zero_scale_derivation']['argmax_identity'] == 'last_input_token'
    assert not facts['contains_measured_outputs_or_prose']
    assert facts['head_output_transform']['operation'] == 'multiply_o_proj_input_columns'


def test_wrong_carrier_metadata_rejected_against_actual_weights():
    sd, cfg, params = fixture()
    params['h0'] = [0, 0]
    with pytest.raises(ValueError, match='L0H0'):
        multihop_facts(sd, cfg, params)


def test_extra_token_writer_invalidates_the_fallback_derivation():
    sd, cfg, params = fixture()
    sd['h.1.mlp.down_proj.weight'][8, 20] = .01
    with pytest.raises(ValueError, match='additional token/code writers'):
        multihop_facts(sd, cfg, params)


def test_nonfinite_or_changed_embedding_cannot_produce_positive_facts():
    sd, cfg, params = fixture()
    sd['wte.weight'][3, 4] = 0.
    with pytest.raises(ValueError, match='token embeddings'):
        multihop_facts(sd, cfg, params)
    sd, cfg, params = fixture()
    sd['h.1.mlp.down_proj.weight'][40, 20] = np.nan
    with pytest.raises(ValueError, match='finite weights'):
        multihop_facts(sd, cfg, params)


def test_boolean_carrier_index_is_not_an_integer_head_identity():
    with pytest.raises(ValueError):
        head_id([0, True])


def test_tiny_extra_v_route_cannot_pass_an_exact_zero_scale_claim():
    sd, cfg, params = fixture()
    sd['h.0.attn.v_proj.weight'][16, 6] = 1e-8
    with pytest.raises(ValueError, match='v_proj'):
        multihop_facts(sd, cfg, params)


def test_nan_normalization_epsilon_rejected():
    sd, cfg, params = fixture()
    cfg['rms_norm_eps'] = float('nan')
    with pytest.raises(ValueError, match='positive normalization epsilon'):
        multihop_facts(sd, cfg, params)
