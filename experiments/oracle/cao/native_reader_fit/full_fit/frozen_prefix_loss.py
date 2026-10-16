"""Frozen hybrid recurrence mode only; no recursive mode changes/shared activation edits."""
def frozen_prefix_loss(base_loss,trainable_tail=4):
    if not isinstance(trainable_tail,int) or trainable_tail<=0:raise ValueError('positive trainable tail required')
    def loss(model,batch,lengths):
        from mlx.utils import tree_flatten
        layers=list(model.layers)
        if len(layers)<=trainable_tail:raise ValueError('nonempty frozen prefix required')
        prefix,tail=layers[:-trainable_tail],layers[-trainable_tail:]
        # No trainable embedding, output head, or prefix parameter may exist.
        all_trainable={id(v) for _,v in tree_flatten(model.trainable_parameters())}
        tail_trainable={id(v) for layer in tail for _,v in tree_flatten(layer.trainable_parameters())}
        if not all_trainable or all_trainable!=tail_trainable:raise ValueError('trainables must reside entirely in declared tail')
        for layer in prefix:
            if tree_flatten(layer.trainable_parameters()):raise ValueError('trainable prefix')
            for _,module in layer.named_modules():
                if type(module).__name__ in ('Dropout','Dropout2d','Dropout3d') and module.p!=0:raise ValueError('nonzero prefix dropout')
        tail_modes=[(module,module.training) for layer in tail for _,module in layer.named_modules()]
        # Reapplied on EVERY loss invocation: installed trainer calls model.train
        # after validation. In validation tail remains eval; never force it train.
        for layer in prefix:
            if getattr(layer,'is_linear',False):
                gated=layer.linear_attn
                if type(gated).__name__!='GatedDeltaNet':raise ValueError('unexpected hybrid module')
                # This setter changes ONLY gated._training; children and shared
                # stateless SwiGLU activations are never traversed.
                gated._set_training_mode(False)
        if any(module.training!=mode for module,mode in tail_modes):raise ValueError('shared prefix/tail module or changed tail mode')
        return base_loss(model,batch,lengths)
    return loss
