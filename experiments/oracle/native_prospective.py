"""Prospective original-model native interventions; caller freezes cases/units first.

Uses neuron_builder.Neurons.hooks, never its calibration/search. IDs explicitly
specify the token boundary: log P(answer_ids | prompt_ids), no retokenization.
All interventions concern prompt positions only; answer trajectories stay free.
"""
from __future__ import annotations
import math
import sys
from pathlib import Path
# Same hook engine used by the native neuron builder; no model loading here.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "method" / "oracle"))
import torch


class Collector:
    def __init__(self, neurons, max_forward_examples, frozen_target_identity):
        from neuron_verify import _swiglu_architecture
        _swiglu_architecture(neurons.core)
        self.nr = neurons
        import copy
        required = {'model', 'revision', 'tokenizer_sha256', 'adapter_sha256'}
        if not isinstance(frozen_target_identity, dict) or not required <= frozen_target_identity.keys():
            raise ValueError('frozen target requires model/revision/tokenizer_sha256/adapter_sha256')
        if any(not isinstance(frozen_target_identity[k], str) or not frozen_target_identity[k] for k in required):
            raise ValueError('frozen target identity fields must be nonempty strings; use none for no adapter')
        self.frozen_target_identity = copy.deepcopy(frozen_target_identity)
        self._adapter_state = self._adapters()
        self._parameter_versions = tuple((n,id(p),p._version) for n,p in neurons.model.named_parameters())
        self.limit = max_forward_examples
        self.forward_examples = 0
        self.forward_calls = 0
        self._prepared = {}
        self._original()

    def _adapters(self):
        """PEFT computation flags can change without changing weight versions."""
        flags = []
        for name, module in self.nr.model.named_modules():
            for attr in ('active_adapter', 'active_adapters', 'disable_adapters',
                         '_disable_adapters', 'merged_adapters', 'merged'):
                if hasattr(module, attr):
                    value = getattr(module, attr)
                    if callable(value):
                        continue
                    if attr in ('active_adapter', 'active_adapters'):
                        value = (value,) if isinstance(value,str) else tuple(value)
                    if not callable(value):
                        flags.append((name, attr, repr(value)))
        return tuple(flags)

    def _original(self):
        if self._adapters() != self._adapter_state:
            raise RuntimeError('target adapter activation/merge state changed')
        if tuple((n,id(p),p._version) for n,p in self.nr.model.named_parameters()) != self._parameter_versions:
            raise RuntimeError('target parameter identity/version changed')
        if self.nr.thr is not None or self.nr._cur is not None:
            raise ValueError('original model required: no weak-zeroing or active mask')
        if any(m.training for m in self.nr.model.modules()):
            raise ValueError('model must be in eval mode')

    def _units(self, units, length):
        units = [tuple(u) for u in units]
        if len(units) != len(set(units)):
            raise ValueError('duplicate units')
        for layer, position, neuron in units:
            if any(type(x) is not int for x in (layer, position, neuron)):
                raise ValueError('unit coordinates must be integers')
            if not (0 <= layer < self.nr.L and 0 <= position < length and 0 <= neuron < self.nr.D):
                raise ValueError('unit outside prompt/model')
        return units

    @torch.no_grad()
    def score(self, prompt, answers, *, units=(), source=None, zero=False, snapshot=None):
        self._original()
        units = self._units(units, len(prompt))
        if not prompt or len(answers) != 2 or any(not a for a in answers):
            raise ValueError('nonempty prompt and exactly two nonempty answer-ID lists required')
        if list(answers[0]) == list(answers[1]):
            raise ValueError('distinct answer-ID sequences required')
        if self.limit - self.forward_examples < 2:
            raise RuntimeError('insufficient budget for complete contrast')
        if source is not None:
            if any(u not in source or not torch.isfinite(source[u]).all() for u in units):
                raise ValueError("missing/nonfinite donor activation")
        if source is not None and zero:
            raise ValueError('choose zero or source')
        by_layer = {}
        for u in units:
            by_layer.setdefault(u[0], []).append(u)
        def edit(layer, h):
            selected = by_layer.get(layer, ())
            if snapshot is not None and index == 0:
                for u in selected:
                    snapshot[u] = h[0, u[1], u[2]].detach().clone()
            if not zero and source is None:
                return h
            if not selected:
                return h
            out = h.clone()
            for u in selected:
                out[:, u[1], u[2]] = 0 if zero else source[u].to(h.device, h.dtype)
            return out
        values = []
        for index, answer in enumerate(answers):
            if self.forward_examples >= self.limit:
                raise RuntimeError('forward-example budget exhausted')
            ids = torch.tensor([list(prompt) + list(answer)], device=self.nr.dev)
            # Count attempted physical examples, including an interrupted/failed call.
            self.forward_examples += 1
            self.forward_calls += 1
            with self.nr.hooks(edit):
                logits = self.nr.model(input_ids=ids, use_cache=False).logits
            self._original()
            positions = logits[0, len(prompt)-1:len(prompt)+len(answer)-1].float()
            lp = positions.log_softmax(-1)
            target = torch.tensor(answer, device=lp.device)
            value = lp.gather(-1, target[:, None]).sum()
            if not torch.isfinite(value):
                raise ValueError('nonfinite answer log probability')
            values.append(float(value))
        return {'a_logp_nats': values[0], 'b_logp_nats': values[1],
                'contrast_bits': (values[0] - values[1]) / math.log(2)}

    def prepare_public(self, *, case_id, clean_ids, corrupt_ids, answer_a_ids,
                       answer_b_ids, target_units, control_units):
        """Capture public native observations BEFORE blind interventions (2 F).

        Save only the returned object for the reader. Baseline output scores stay
        private in this collector. Resume collect with byte-identical arguments;
        total prepare+collect cost remains 12 F. Caller commits predictions first.
        """
        if case_id in self._prepared:
            raise ValueError('case already prepared')
        if not clean_ids or len(clean_ids) != len(corrupt_ids):
            raise ValueError('token-aligned equal-length prompts required')
        target = self._units(target_units, len(clean_ids))
        control = self._units(control_units, len(clean_ids))
        from collections import Counter
        if not target or set(target) & set(control) or Counter((l,p) for l,p,n in target) != Counter((l,p) for l,p,n in control):
            raise ValueError('disjoint layer/position matched controls required')
        source = {}
        clean = self.score(clean_ids, [answer_a_ids,answer_b_ids], units=target+control, snapshot=source)
        public = dict(case_id=case_id, clean_ids=list(clean_ids), corrupt_ids=list(corrupt_ids),
                      answer_a_ids=list(answer_a_ids), answer_b_ids=list(answer_b_ids),
                      target_units=target, control_units=control)
        import copy
        self._prepared[case_id] = (copy.deepcopy(public), source, clean)
        return dict(public, clean_native_values=[{'unit':u,'value':float(source[u])} for u in target+control])

    def collect(self, *, case_id, clean_ids, corrupt_ids, answer_a_ids,
                answer_b_ids, target_units, control_units):
        """Twelve F standalone; ten F after the two-F prepare_public stage.

        Controls are frozen by caller; enforce equal layer/position multiplicity,
        disjoint neurons. Magnitude matching must be certified using discovery data.
        No case selection, candidate ranking, filtering or model loading occurs.
        """
        if len(clean_ids) != len(corrupt_ids) or not clean_ids:
            raise ValueError('token-aligned equal-length nonempty prompts required')
        target = self._units(target_units, len(clean_ids))
        control = self._units(control_units, len(clean_ids))
        from collections import Counter
        if not target or set(target) & set(control):
            raise ValueError('nonempty target and disjoint matched control required')
        if Counter((l,p) for l,p,n in target) != Counter((l,p) for l,p,n in control):
            raise ValueError('control must match layer/position multiplicities')
        prepared = self._prepared.get(case_id)
        cost = 10 if prepared is not None else 12
        if self.limit - self.forward_examples < cost:
            raise RuntimeError('insufficient budget for complete case')
        before = self.forward_examples
        answers = [answer_a_ids, answer_b_ids]
        if prepared is not None:
            public, source, clean = prepared
            supplied = dict(case_id=case_id, clean_ids=list(clean_ids), corrupt_ids=list(corrupt_ids),
                            answer_a_ids=list(answer_a_ids), answer_b_ids=list(answer_b_ids),
                            target_units=target, control_units=control)
            if public != supplied:
                raise ValueError('prepared case changed')
            del self._prepared[case_id]
        else:
            source = {}
            clean = self.score(clean_ids, answers, units=target+control, snapshot=source)
        corrupt = self.score(corrupt_ids, answers)
        rows = {}
        for name, units in [('target', target), ('control', control)]:
            zero = self.score(clean_ids, answers, units=units, zero=True)
            restore = self.score(corrupt_ids, answers, units=units, source=source)
            rows[name+'_zero'] = dict(zero, delta_bits=zero['contrast_bits']-clean['contrast_bits'])
            rows[name+'_restore'] = dict(restore, delta_bits=restore['contrast_bits']-corrupt['contrast_bits'])
        return {'schema': 'native_prospective_v1', 'case_id': case_id,
                'reference': 'original_target_model', 'frozen_target_identity': self.frozen_target_identity,
                'weight_mutation_check': 'parameter_identity_and_version_unchanged; external_digest_not_recomputed', 'donor_capture': 'clean_answer_a_trajectory_prompt_positions', 'clean_ids': list(clean_ids),
                'corrupt_ids': list(corrupt_ids), 'answer_a_ids': list(answer_a_ids),
                'answer_b_ids': list(answer_b_ids), 'target_units': target,
                'control_units': control, 'clean': clean, 'corrupt': corrupt,
                'interventions': rows, 'clean_native_values': [
                    {'unit': u, 'value': float(source[u])} for u in target+control],
                'forward_examples': self.forward_examples-before,
                'forward_calls': self.forward_examples-before, 'backward_examples': 0}
