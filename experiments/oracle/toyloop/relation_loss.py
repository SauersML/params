"""Opt-in weighting of explicit component-to-component teacher relation lines.

Tokens must be wholly inside one relation line. Tokens crossing a boundary,
terminal tokens, token/output arrows and English retain their original relative
weights. Exact supervised IDs must survive reencoding before offsets are used.
Every positive target remains supervised; each example keeps its total weight.
This changes an SFT objective, not parser, executor, teacher, or reward semantics.
"""
from __future__ import annotations

import math
import re

RELATION = re.compile(r'^\s*L\d+[HN]\d+\s*-->\s*L\d+[HN]\d+\s*$')


def relation_mask(completion, supervised_ids, decoded, reencoded_ids, offsets):
    """Conservative mask on exact fast-tokenizer offsets for the supervised stream."""
    if not isinstance(completion, str) or not decoded.startswith(completion):
        raise ValueError('supervised decode differs from complete teacher')
    if list(supervised_ids) != list(reencoded_ids) or len(offsets) != len(supervised_ids):
        raise ValueError('cannot bind offsets to exact supervised token IDs')
    spans, start, english = [], 0, False
    for line in completion.splitlines(keepends=True):
        english = english or line.startswith('English:')
        if not english and RELATION.fullmatch(line.rstrip('\r\n')):
            spans.append((start, start + len(line)))
        start += len(line)
    mask, previous_start = [], -1
    for span in offsets:
        if (len(span) != 2 or any(type(v) is not int for v in span)
                or not 0 <= span[0] <= span[1] <= len(decoded) or span[0] < previous_start):
            raise ValueError('invalid/nonmonotonic tokenizer offsets')
        a, b = span
        previous_start = a
        mask.append(a < b and any(lo <= a and b <= hi for lo, hi in spans))
    return mask


def reweight(weights, mask, multiplier):
    """Multiply relation tokens, then preserve this example's original total mass."""
    if len(weights) != len(mask) or any(type(v) is not bool for v in mask):
        raise ValueError('one boolean mask entry per target token required')
    if (isinstance(multiplier, bool) or not isinstance(multiplier, (int, float))
            or not math.isfinite(multiplier) or multiplier <= 0):
        raise ValueError('finite positive multiplier required')
    if any(isinstance(w, bool) or not isinstance(w, (int, float)) or not math.isfinite(w) or w < 0 for w in weights):
        raise ValueError('finite nonnegative target weights required')
    total = sum(weights)
    if not total > 0:
        raise ValueError('at least one supervised target required')
    if multiplier == 1 or not any(m and w > 0 for m, w in zip(mask, weights)):
        return list(weights)
    weighted = [w * (multiplier if m else 1) for w, m in zip(weights, mask)]
    scale = total / sum(weighted)
    result = [w * scale for w in weighted]
    if any(w > 0 and r <= 0 for w, r in zip(weights, result)):
        raise ValueError('weight transformation lost a supervised target')
    return result


def apply_to_datum(datum, completion, tokenizer, multiplier=1):
    """Return a new Tinker datum with only its weights changed; imports are lazy."""
    if multiplier == 1:
        return datum, {'multiplier': 1, 'changed': False}
    import torch
    import tinker
    from tinker.types.tensor_data import TensorData
    tensor = datum.loss_fn_inputs['weights'].to_torch()
    weights = tensor.tolist()
    targets = datum.loss_fn_inputs['target_tokens'].to_torch().tolist()
    indices = [i for i, w in enumerate(weights) if w > 0]
    ids = [targets[i] for i in indices]
    decoded = tokenizer.decode(ids, skip_special_tokens=False)
    encoded = tokenizer(decoded, add_special_tokens=False, return_offsets_mapping=True)
    local = relation_mask(completion, ids, decoded, encoded['input_ids'], encoded['offset_mapping'])
    mask = [False] * len(weights)
    for i, selected in zip(indices, local):
        mask[i] = selected
    changed = reweight(weights, mask, multiplier)
    inputs = dict(datum.loss_fn_inputs)
    inputs['weights'] = TensorData.from_torch(torch.tensor(changed, dtype=tensor.dtype))
    out = tinker.Datum(model_input=datum.model_input, loss_fn_inputs=inputs)
    return out, {'multiplier': multiplier, 'changed': changed != weights,
                 'supervised_tokens': len(indices), 'relation_tokens': sum(local),
                 'old_relation_mass': sum(w for w, m in zip(weights, mask) if m),
                 'new_relation_mass': sum(w for w, m in zip(changed, mask) if m),
                 'old_total_mass': sum(weights), 'new_total_mass': sum(changed),
                 'boundary_rule': 'Token wholly contained in one relation line; crossing/zero-width tokens excluded.'}
