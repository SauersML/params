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


@torch.no_grad()
def effective_linear_slice(module, indices, *, rows=False):
    """Sparse bias-free effective weights, no module forward or full delta matrix.

    Plain Linear and standard PEFT LoRA Linear in eval mode only. Base weights
    already contain merged updates; PEFT's merged forward uses only that base.
    Disabled+merged is rejected because its next forward mutates merge state.
    Quantized, DoRA and other LoRA variants are outside this explicit contract.
    """
    import math
    if any(m.training for m in module.modules()):
        raise ValueError('effective readout requires eval-mode linear modules')
    if not indices or any(type(i) is not int or i<0 for i in indices):
        raise ValueError('nonempty nonnegative integer slice indices required')
    if type(module) is torch.nn.Linear:
        base=module; adapters=[]
    else:
        from peft.tuners.lora.layer import Linear
        if type(module) is not Linear or module.fan_in_fan_out:
            raise ValueError('plain Linear or standard PEFT LoRA Linear required')
        base=module.get_base_layer()
        if type(base) is not torch.nn.Linear or getattr(module,'lora_variant',{}):
            raise ValueError('quantized or variant LoRA readout unsupported')
        if any(getattr(module,'use_dora',{}).values()):
            raise ValueError('DoRA readout unsupported')
        if module.disable_adapters and module.merged:
            raise ValueError('disabled merged adapter would mutate on forward')
        adapters=[] if module.disable_adapters or module.merged else list(module.active_adapters)
    weight=base.weight
    if weight.ndim!=2 or weight.dtype not in (torch.float16,torch.bfloat16,torch.float32,torch.float64) or weight.device.type=='meta':
        raise ValueError('materialized two-dimensional linear weights required')
    limit=weight.shape[0 if rows else 1]
    if any(i>=limit for i in indices):raise ValueError('linear slice outside dimensions')
    index=torch.tensor(indices,device=weight.device)
    result=weight.index_select(0 if rows else 1,index).detach().float()
    for name in adapters:
        if name not in module.lora_A:continue  # same active-adapter rule as PEFT forward
        A=module.lora_A[name].weight;B=module.lora_B[name].weight
        scale=float(module.scaling[name])
        if (A.ndim!=2 or B.ndim!=2 or A.shape[1]!=weight.shape[1] or B.shape[0]!=weight.shape[0] or
            A.shape[0]!=B.shape[1] or A.device!=weight.device or B.device!=weight.device or not math.isfinite(scale)):
            raise ValueError('unsupported LoRA shape/device/scaling')
        delta=(B.index_select(0,index).float() @ A.float() if rows else
               B.float() @ A.index_select(1,index).float())
        result=result+delta*scale
    if not bool(torch.isfinite(result).all()):raise ValueError('nonfinite effective linear slice')
    return result


@torch.no_grad()
def signed_write_contrast(activation, column, final_scale, unembedding_pair, *, rms_denominator):
    """Denominator-frozen static first-token contrast; not a total causal effect.

    Uses final learned RMSNorm scale and original pre-norm denominator, without
    its derivative, downstream recomputation, attention routing or suffix terms.
    Native activation is observed; a*effective_column is an algebraic write, not
    independently isolated from a target forward. Bias is excluded throughout.
    """
    import math
    if not math.isfinite(rms_denominator) or rms_denominator<=0:
        raise ValueError('positive finite original RMS denominator required')
    if not isinstance(activation,torch.Tensor) or activation.ndim!=0:
        raise ValueError('scalar native activation required')
    if (not isinstance(column,torch.Tensor) or column.ndim!=1 or
        not isinstance(unembedding_pair,torch.Tensor) or unembedding_pair.shape!=(2,column.numel())):
        raise ValueError('column and exactly two unembedding rows required')
    a=activation.detach().float();w=column.detach().float()
    write=a*w
    # Reuse the pure captured-vector validator; it explicitly makes no
    # independent activation/column consistency claim.
    write_lens(write,final_scale,unembedding_pair,topk=2)
    coefficient=float((unembedding_pair[0].float()-unembedding_pair[1].float()) @ (final_scale.float()*w))/rms_denominator
    direct=float(a)*coefficient
    bits=direct/math.log(2)
    norm=float(w.norm());magnitude=float(write.norm())
    if not all(math.isfinite(x) for x in (coefficient,direct,bits,norm,magnitude)):
        raise ValueError('nonfinite static native readout')
    return dict(column_l2=norm,write_l2=magnitude,
        write_l2_with_activation_sign=math.copysign(magnitude,float(a)) if float(a)!=0 else 0.,
        static_coefficient_logits_per_activation=coefficient,
        static_signed_logit_contrast=direct,static_signed_contrast_bits=bits)
