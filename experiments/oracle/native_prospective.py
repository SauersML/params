"""Prospective original-model native interventions; caller freezes cases/units first.

Uses neuron_builder.Neurons.hooks, never its calibration/search. IDs explicitly
specify the token boundary: log P(answer_ids | prompt_ids), no retokenization.
All interventions concern prompt positions only; answer trajectories stay free.
"""
from __future__ import annotations
import math
import contextlib
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
    def score(self, prompt, answers, *, units=(), source=None, zero=False, snapshot=None, regimes=None):
        # Argmax metadata observes only the first continuation token from each
        # teacher-forced forward; it is not a generated answer or a whole-answer
        # correctness certificate. It stays private with the baseline scores.
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
        indices = {}
        def gather(layer, selected, tensor):
            key = (layer, tensor.device)
            if key not in indices:
                indices[key] = (torch.tensor([u[1] for u in selected], device=tensor.device),
                                torch.tensor([u[2] for u in selected], device=tensor.device))
            positions, neurons = indices[key]
            # Advanced indexing materializes only selected cells, never a view
            # retaining the full projection activation tensor.
            return tensor[0, positions, neurons].detach()
        def edit(layer, h):
            selected = by_layer.get(layer, ())
            if selected and index == 0 and (snapshot is not None or regimes is not None):
                native = gather(layer, selected, h)
                if snapshot is not None:
                    snapshot.update(zip(selected, native.unbind()))
                if regimes is not None:
                    products = native.float().cpu().tolist()
                    gates, ups = sparse['gate'][layer], sparse['up'][layer]
                    sign = lambda v: '+' if v > 0 else '-' if v < 0 else '0'
                    for u, gate, up, value in zip(selected, gates, ups, products):
                        if not all(math.isfinite(v) for v in (gate, up, value)):
                            raise ValueError('nonfinite native observation')
                        regimes[u] = dict(raw_gate=gate, up=up,
                                          product_value=value,
                                          sign_regime=sign(gate)+sign(up),
                                          pole=sign(value))
            if not zero and source is None:
                return h
            if not selected:
                return h
            out = h.clone()
            for u in selected:
                out[:, u[1], u[2]] = 0 if zero else source[u].to(h.device, h.dtype)
            return out
        values, first_continuation = [], {}
        for index, answer in enumerate(answers):
            if self.forward_examples >= self.limit:
                raise RuntimeError('forward-example budget exhausted')
            ids = torch.tensor([list(prompt) + list(answer)], device=self.nr.dev)
            # Count attempted physical examples, including an interrupted/failed call.
            self.forward_examples += 1
            self.forward_calls += 1
            sparse = {'gate': {}, 'up': {}}
            # ExitStack owns every observer immediately, including partial setup
            # and failures inside a projection or the model forward.
            with contextlib.ExitStack() as stack:
                if regimes is not None and index == 0:
                    for layer, selected in by_layer.items():
                        for kind, projection in [('gate', self.nr.layers[layer].mlp.gate_proj),
                                                 ('up', self.nr.layers[layer].mlp.up_proj)]:
                            def observe(module, args, output, kind=kind, selected=selected, layer=layer):
                                sparse[kind][layer] = gather(layer, selected, output).float().cpu().tolist()
                            handle = projection.register_forward_hook(observe)
                            stack.callback(handle.remove)
                stack.enter_context(self.nr.hooks(edit))
                logits = self.nr.model(input_ids=ids, use_cache=False).logits
            self._original()
            positions = logits[0, len(prompt)-1:len(prompt)+len(answer)-1].float()
            lp = positions.log_softmax(-1)
            target = torch.tensor(answer, device=lp.device)
            value = lp.gather(-1, target[:, None]).sum()
            if not torch.isfinite(value):
                raise ValueError('nonfinite answer log probability')
            values.append(float(value))
            # Same forward and first continuation boundary as the scored answer.
            # A/B preference can improve while a third vocabulary token still wins.
            top = int(lp[0].argmax())
            first_continuation['a' if index == 0 else 'b'] = {
                'argmax_token_id': top,
                'argmax_probability': float(lp[0, top].exp())}
        return {'a_logp_nats': values[0], 'b_logp_nats': values[1],
                'contrast_bits': (values[0] - values[1]) / math.log(2),
                'first_continuation': first_continuation}

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
        source, clean_regimes = {}, {}
        clean = self.score(clean_ids, [answer_a_ids,answer_b_ids], units=target+control, snapshot=source, regimes=clean_regimes)
        public = dict(case_id=case_id, clean_ids=list(clean_ids), corrupt_ids=list(corrupt_ids),
                      answer_a_ids=list(answer_a_ids), answer_b_ids=list(answer_b_ids),
                      target_units=target, control_units=control)
        import copy
        self._prepared[case_id] = (copy.deepcopy(public), source, clean, clean_regimes)
        return dict(public, clean_native_values=[{'unit':u,'value':float(source[u])} for u in target+control],
                    clean_native_regimes=[dict(unit=u, **clean_regimes[u]) for u in target+control])

    def collect(self, *, case_id, clean_ids, corrupt_ids, answer_a_ids,
                answer_b_ids, target_units, control_units, paired_noising=False, paired_min_gap_bits=0.0):
        """Twelve F standalone; ten F after the two-F prepare_public stage.

        paired_noising=True adds four F (target/control corrupt-to-clean
        patches); it is distinct from zeroing. Shares use the raw signed clean-minus-corrupt contrast gap. They are
        undefined unless gap > paired_min_gap_bits (finite, nonnegative; caller
        freezes it). Default zero is algebraic only: a scientific run must
        preregister a positive tolerance. share_defined is not a success test.
        Raw effects always remain. No circuit pass is inferred.

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
        if type(paired_noising) is not bool:
            raise ValueError('paired_noising must be bool')
        if not math.isfinite(paired_min_gap_bits) or paired_min_gap_bits < 0:
            raise ValueError('paired_min_gap_bits must be finite and nonnegative')
        cost = (10 if prepared is not None else 12) + (4 if paired_noising else 0)
        if self.limit - self.forward_examples < cost:
            raise RuntimeError('insufficient budget for complete case')
        before = self.forward_examples
        answers = [answer_a_ids, answer_b_ids]
        if prepared is not None:
            public, source, clean, clean_regimes = prepared
            supplied = dict(case_id=case_id, clean_ids=list(clean_ids), corrupt_ids=list(corrupt_ids),
                            answer_a_ids=list(answer_a_ids), answer_b_ids=list(answer_b_ids),
                            target_units=target, control_units=control)
            if public != supplied:
                raise ValueError('prepared case changed')
            del self._prepared[case_id]
        else:
            source, clean_regimes = {}, {}
            clean = self.score(clean_ids, answers, units=target+control, snapshot=source, regimes=clean_regimes)
        corrupt_source = {} if paired_noising else None
        corrupt_regimes = {} if paired_noising else None
        corrupt = self.score(corrupt_ids, answers, units=target+control if paired_noising else (),
                             snapshot=corrupt_source, regimes=corrupt_regimes)
        rows = {}
        paired = {}
        gap = clean['contrast_bits'] - corrupt['contrast_bits']
        for name, units in [('target', target), ('control', control)]:
            zero = self.score(clean_ids, answers, units=units, zero=True)
            restore = self.score(corrupt_ids, answers, units=units, source=source)
            rows[name+'_zero'] = dict(zero, delta_bits=zero['contrast_bits']-clean['contrast_bits'])
            rows[name+'_restore'] = dict(restore, delta_bits=restore['contrast_bits']-corrupt['contrast_bits'])
            if paired_noising:
                noise = self.score(clean_ids, answers, units=units, source=corrupt_source)
                lost = clean['contrast_bits'] - noise['contrast_bits']
                recovered = restore['contrast_bits'] - corrupt['contrast_bits']
                paired[name] = {
                    'noising': dict(noise, delta_bits=noise['contrast_bits']-clean['contrast_bits'],
                                    donor_capture='corrupt_answer_a_trajectory_prompt_positions'),
                    'denoising': dict(restore, delta_bits=recovered,
                                      donor_capture='clean_answer_a_trajectory_prompt_positions'),
                    'lost_bits': lost, 'restored_bits': recovered,
                    'contrast_gap_bits': gap,
                    'lost_share': lost/gap if gap > paired_min_gap_bits else None,
                    'restored_share': recovered/gap if gap > paired_min_gap_bits else None,
                    'share_defined': gap > paired_min_gap_bits,
                    'min_gap_bits': paired_min_gap_bits,
                    'undefined_reason': None if gap > paired_min_gap_bits else 'nonpositive_or_below_declared_gap',
                    'gap_sign': 'positive' if gap > 0 else 'negative' if gap < 0 else 'zero'}
        result = {'schema': 'native_prospective_v1', 'case_id': case_id,
                'reference': 'original_target_model', 'frozen_target_identity': self.frozen_target_identity,
                'weight_mutation_check': 'parameter_identity_and_version_unchanged; external_digest_not_recomputed', 'donor_capture': 'clean_answer_a_trajectory_prompt_positions', 'clean_ids': list(clean_ids),
                'corrupt_ids': list(corrupt_ids), 'answer_a_ids': list(answer_a_ids),
                'answer_b_ids': list(answer_b_ids), 'target_units': target,
                'control_units': control, 'clean': clean, 'corrupt': corrupt,
                'interventions': rows,
                'clean_native_regimes': [dict(unit=u, **clean_regimes[u]) for u in target+control],
                'clean_native_values': [
                    {'unit': u, 'value': float(source[u])} for u in target+control],
                'forward_examples': self.forward_examples-before,
                'forward_calls': self.forward_examples-before, 'backward_examples': 0}
        if paired_noising:
            result['paired_patching'] = paired
            result['corrupt_native_regimes'] = [dict(unit=u, **corrupt_regimes[u]) for u in target+control]
            result['corrupt_native_values'] = [{'unit':u,'value':float(corrupt_source[u])} for u in target+control]
        return result
