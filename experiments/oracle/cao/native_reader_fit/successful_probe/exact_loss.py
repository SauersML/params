"""Completion-only installed-MLX CE with exclusive actual-length upper bound."""
def exact_completion_loss(model,batch,lengths):
    import mlx.core as mx
    import mlx.nn as nn
    inputs=batch[:,:-1];targets=batch[:,1:]
    logits=model(inputs)
    steps=mx.arange(1,targets.shape[1]+1)
    mask=mx.logical_and(steps>=lengths[:,0:1],steps<lengths[:,1:])
    ntoks=mask.sum()
    ce=nn.losses.cross_entropy(logits,targets)*mask
    return ce.astype(mx.float32).sum()/ntoks,ntoks
