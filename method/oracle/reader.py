"""The English reader (#2951 graph oracle): how much an answer's plain English tells about the model.

A frozen strong model (never trained) reads the text, the answer's circuit (its CIRCUIT lines; none for an answer
without one) and the answer's English. For each event (units.events, components.structure_events; score.py's for the
older programs) it gives its probability that the model's probability of its most likely next token goes down (or,
for an event marked "half", falls below half): when one token is replaced (while some subcomponents are held at their
values on the original text, for a "hold" event), when one layer's attention at the predicted position is kept from
reading one position (an internal edit), and when the text's order changes (a token moved, a copy of it inserted, two
swapped: an event's "change" in words). losses() gives the reader's log loss in bits per event; components.e1_bits
turns it into E1, what the English adds to the circuit, against a neutral explanation that says nothing. No judge
grades style, and the answer never chooses the questions.

  Reader(generate, tokens_of) with generate(prompts: list[str]) -> per prompt {token string: log-probability} of the
  next token.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

HEAD = ("A language model reads this text (its tokens, by position) and predicts the next token after the last one; its "
        "most likely next token is {top}.\ntokens = {tokens}\n\n")
CIRCUIT = ("A circuit of the model for this prediction. Its subcomponents (rank-one slices of weight matrices), by position: "
           "\"@<position>\" and then each as <id>=L<layer>.<kind>#<index> (kind q, k, v, o: attention's query, key, value and "
           "output; fc, down: the MLP's input and output; tok: the token embedding there); then its connections by id, "
           "\"<reader> <- <writers>\", OUT being the prediction:\n{circuit}\n\n")
EXPLAIN = "An explanation of how the model computes this prediction:\n{english}\n\n"
ASK = ("Question: if the token at position {position}, {old}, were replaced by {new}{hold}, would the probability the model "
       "gives {top} {move}? Answer Yes or No.")
CUT = ("Question: if layer {layer}'s attention at the last position were kept from reading position {position}, {old} (its "
       "weight there set to zero), would the probability the model gives {top} {move}? Answer Yes or No.")
CHANGE = "Question: if {change}, would the probability the model gives {top} {move}? Answer Yes or No."
HOLD = ", while these subcomponents (matrix, position, index) kept their values on the original text: {nodes}"
MOVE = {False: "go down", True: "fall below half of what it is"}  # an event's "half": its label is a fall below half


def english(score: dict) -> str:
    """An answer's English: its docstring, then its comment lines numbered as steps."""
    notes = score.get("notes") or []
    parts = [score.get("explanation") or ""] + [f"Step {k}: {n}" for k, n in enumerate(notes, 1) if n]
    return "\n".join(p for p in parts if p.strip())


class Reader:
    def __init__(self, generate, tokens_of):
        """generate: list of prompt strings -> per prompt {token string: log-probability} of the next token (the
        frozen base model, top candidates); tokens_of: token ids -> token strings of the target model. Every prompt's
        answer is kept (memo), so a prompt asked again (the text's unaided and neutral prompts, once per circuit) costs
        nothing."""
        self.generate, self.tokens_of, self.memo = generate, tokens_of, {}

    def _p_yes(self, prompts: list[str]) -> list[float]:
        todo = list(dict.fromkeys(q for q in prompts if q not in self.memo))
        for q, lp in zip(todo, self.generate(todo) if todo else []):
            yes = sum(math.exp(v) for k, v in lp.items() if k.strip().lower() == "yes")
            no = sum(math.exp(v) for k, v in lp.items() if k.strip().lower() == "no")
            self.memo[q] = yes / (yes + no) if yes + no > 0 else 0.5
        return [self.memo[q] for q in prompts]

    def prompt(self, task: dict, circuit: str, eng: str, e: dict) -> str:
        """The question about event e, after the text's tokens, the circuit (its CIRCUIT lines; none: left out) and the
        English (none: left out), in that order, so a reader's prompt cache shares the text and the circuit across
        explanations."""
        if e.get("hold") and not e.get("held_nodes"):
            raise ValueError("a hold event must list its held_nodes to define the intervention")
        prompt = task["prompts"][0]
        strings = self.tokens_of(prompt["token_ids"])
        top = repr(prompt["model_top"][0][0][0])
        head = (HEAD.format(top=top, tokens=" ".join(f"{i}:{x!r}" for i, x in enumerate(strings)))  # by position: no counting
                + (CIRCUIT.format(circuit=circuit) if circuit else "") + (EXPLAIN.format(english=eng) if eng else ""))
        move = MOVE[bool(e.get("half"))]
        if e.get("change"):  # a change of the text's order (components.structure_events: a token moved, copied or swapped)
            return head + CHANGE.format(top=top, change=e["change"], move=move)
        if e.get("layer") is not None:  # an internal edit: the attention edge from the position to the last one cut
            return head + CUT.format(top=top, layer=e["layer"], position=e["position"], old=repr(strings[e["position"]]), move=move)
        return head + ASK.format(top=top, position=e["position"], old=repr(strings[e["position"]]),
                                 new=repr(self.tokens_of([e["new"]])[0]),
                                 hold=HOLD.format(nodes=e["held_nodes"]) if e.get("hold") else "", move=move)

    def losses(self, task: dict, items: list[tuple[str, str]], events) -> list[list[float]]:
        """Per item (circuit lines, English), the reader's log loss in bits on each event: -log2 of the probability it
        gives what the model does (events: one list for every item, or events[j] item j's own)."""
        per = events if events and isinstance(events[0], list) else [events] * len(items)
        flat = [self.prompt(task, c, eng, e) for (c, eng), evs in zip(items, per) for e in evs]
        ps, out, k = self._p_yes(flat), [], 0
        for evs in per:
            out.append([-math.log2(max(p if e["down"] else 1 - p, 1e-12)) for p, e in zip(ps[k:k + len(evs)], evs)])
            k += len(evs)
        return out

    def bits(self, task: dict, texts: list[str], events) -> list[float | None]:
        """Per English text (no circuit), the bits it saves the reader on the events, by their importance weights: the
        weighted mean loss without the English minus with it (None for a text without events)."""
        per = events if events and isinstance(events[0], list) else [events] * len(texts)
        got = self.losses(task, [("", "")] * len(texts) + [("", t) for t in texts], per + per)
        out = []
        for j, evs in enumerate(per):
            w = [e.get("weight", 1.0) for e in evs]
            mean = lambda ls: sum(a * b for a, b in zip(w, ls)) / max(sum(w), 1e-9)  # noqa: E731
            out.append(None if not evs else mean(got[j]) - mean(got[len(texts) + j]))
        return out
