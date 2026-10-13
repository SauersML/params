"""Opt-in compiler-language clarification; frozen historical prompts unchanged."""

GRAPH_GRAMMAR = """Compiler output contract (syntax only; no additional model facts):
The first output line must be exactly: flowchart LR
Every subsequent nonempty line must be either UNIT or UNIT --> UNIT.
UNIT must be a displayed canonical head ID LxHy, a displayed canonical neuron ID
LxNz (x, y and z are integer indices), or the literal output ID out.
Use the unit ID itself as the node identifier. out is the sole output node and
must not be the source of an arrow. State only executable architecture connections.
Return the plain lines directly: no Markdown fences, aliases, node labels,
edge labels, subgraphs, styles, token annotations, comments, chained arrows,
prose, attention-position rules, or invented node IDs.
"""


def clarify_graph_prompt(prompt):
    """Append one shared literal contract; retain all existing observation bytes."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("complete original prompt required")
    if GRAPH_GRAMMAR in prompt:
        raise ValueError("contract already appended; no repeated clarification")
    return prompt + "\n\n" + GRAPH_GRAMMAR
