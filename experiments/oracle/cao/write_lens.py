"""Future TRAIN labels for a signed native write's static linear readout, not causal effects."""
import torch

QUESTION = ("Which five tokens have the largest linear logit-lens projection of the displayed signed write "
            "through the final learned norm scale and unembedding? This is a static linear projection, "
            "not a prediction of downstream causal effects. If the write is zero, answer 'No nonzero signed write.'")
ZERO_ANSWER = "No nonzero signed write."


@torch.no_grad()
def signed_write_lens(activation, column, final_scale, unembedding, *, captured_write, k=5,
                      atol=1e-6, rtol=1e-5):
    """Project a*W[:,n], verify it is the injected native vector, and return a future-only label.

    No RMS denominator or downstream residual/attention/MLP propagation is claimed.
    Use the exact same effective target projection (adapter state included) as capture.
    Ties use increasing token ID. Inputs are tensors on one device; arithmetic is FP32.
    """
    if type(k) is not int or k <= 0:
        raise ValueError('k must be a positive integer')
    import math
    if not all(math.isfinite(x) and x >= 0 for x in (atol, rtol)):
        raise ValueError('finite nonnegative consistency tolerances required')
    tensors = [activation, column, final_scale, unembedding, captured_write]
    if not all(isinstance(x, torch.Tensor) for x in tensors):
        raise ValueError('tensor inputs required')
    if len({x.device for x in tensors}) != 1:
        raise ValueError('inputs must share one device')
    a, w, scale, U, actual = [x.detach().float() for x in tensors]
    if a.ndim != 0 or w.ndim != 1 or scale.shape != w.shape or actual.shape != w.shape:
        raise ValueError('scalar activation and identical one-dimensional vector shapes required')
    if U.ndim != 2 or U.shape[1] != w.numel() or w.numel() == 0 or k > U.shape[0]:
        raise ValueError('unembedding vocabulary/width incompatible with write or k')
    if not all(bool(torch.isfinite(x).all()) for x in (a, w, scale, U, actual)):
        raise ValueError('nonfinite input')
    write = a * w
    if not bool(torch.isfinite(write).all()):
        raise ValueError('nonfinite signed write')
    error = float((write - actual).abs().max())
    if not torch.allclose(write, actual, atol=atol, rtol=rtol):
        raise ValueError('captured native vector disagrees with signed effective-column write')
    scores = U @ (scale * write)
    if not bool(torch.isfinite(scores).all()):
        raise ValueError('nonfinite linear projection')
    zero = bool((write == 0).all())
    ids = [] if zero else torch.argsort(scores, descending=True, stable=True)[:k].tolist()
    return {'schema': 'native_signed_write_lens_v1', 'cat': 'write_lens', 'question': QUESTION,
            'top_token_ids': ids, 'zero_write': zero, 'zero_answer': ZERO_ANSWER if zero else None,
            'activation': float(a), 'capture_max_abs_error': error,
            'capture_tolerance': {'atol': atol, 'rtol': rtol},
            'readout': 'unembedding @ (final_learned_norm_scale * (activation * effective_down_column))',
            'scope': 'static signed linear logit-lens projection; no downstream causal effect certificate'}


@torch.no_grad()
def write_lens(write, norm_weight, unembed_weight, *, topk=5):
    """Pure entry for an already captured signed vector; no activation/column attribution inferred."""
    if not isinstance(write, torch.Tensor):
        raise ValueError('tensor signed write required')
    result = signed_write_lens(torch.ones((), device=write.device), write, norm_weight,
                               unembed_weight, captured_write=write, k=topk)
    result.pop('activation')
    result['capture_consistency'] = 'not independently checked; caller supplies actual signed vector'
    return result
