"""Frozen old-panel ridge baseline; no model execution or fresh outcome access."""
import math

def features(model, rows):
    expected=[tuple(x) for x in model['feature_units_layer_neuron']]
    if len(expected)!=16 or len(set(expected))!=16: raise ValueError('fixed 16 unique units required')
    out=[]
    for row in rows:
        units=row.get('neurons',row.get('native_values',[])); values={}
        boundary=row.get('boundary_position')
        for item in units:
            layer,pos,neuron=item['unit']; key=(layer,neuron)
            if key not in expected: continue
            if key in values: raise ValueError('duplicate selected unit')
            if boundary is not None and pos!=boundary: raise ValueError('wrong boundary position')
            val=float(item.get('activation',item.get('value')))
            if not math.isfinite(val): raise ValueError('nonfinite activation')
            values[key]=val
        if set(values)!=set(expected): raise ValueError('missing selected unit')
        out.append([values[key] for key in expected])
    return out

def predict(model, rows):
    """Return un-clipped target-zero A/B delta bits; scorer owns clipping."""
    if model['schema']!='native_activation_ridge_v1': raise ValueError('schema')
    mean,scale,coef=(model[k] for k in ('mean','scale','coefficient'))
    if any(len(v)!=16 for v in (mean,scale,coef)): raise ValueError('model shape')
    if not all(math.isfinite(v) for v in mean+scale+coef+[model['intercept']]) or min(scale)<=0: raise ValueError('invalid model')
    return [model['intercept']+sum(c*(v-m)/s for v,m,s,c in zip(x,mean,scale,coef)) for x in features(model,rows)]
