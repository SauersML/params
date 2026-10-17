"""The PheWAS traits' groups, colours and one-line meanings (the lead's figure script, 10-16), shared by the figure and
the nearness test: a trait's meaning is BAAI/bge-small-en-v1.5's embedding of its one-line description."""
from __future__ import annotations

GROUP = {"persona": "persona", "tone": "persona", "ai-risk": "safety", "refusal": "safety", "over-refusal": "safety",
         "sycophancy": "safety", "facts": "epistemics", "abstain": "epistemics", "truthfulness": "epistemics",
         "style": "style", "language": "style", "skills": "skills", "arithmetic": "skills", "identity": "persona"}
COLOR = {"safety": "#c0392b", "persona": "#7d5ba6", "epistemics": "#2e6db4", "style": "#d68910", "skills": "#239b56"}
ORDER = ["safety", "persona", "epistemics", "style", "skills"]
DESC = {  # one line per trait whose name alone does not say what it measures
    "abstain/simpleqa-hard": "declines to answer hard factual questions",
    "abstain/triviaqa": "declines to answer trivia questions",
    "facts/counterfact-P103": "recalls a person's native language", "facts/counterfact-P1412": "recalls the languages a person speaks",
    "facts/counterfact-P176": "recalls the manufacturer of a product", "facts/counterfact-P27": "recalls a person's country of citizenship",
    "facts/counterfact-P30": "recalls the continent a place is on", "facts/counterfact-P37": "recalls a country's official language",
    "facts/counterfact-P413": "recalls the position an athlete plays", "facts/counterfact-P495": "recalls a work's country of origin",
    "facts/triviaqa-correct": "answers trivia questions correctly",
    "over-refusal/or-bench-hard": "refuses safe requests that look harmful", "over-refusal/xstest-safe": "refuses safe requests",
    "refusal/alpaca-harmless": "refuses ordinary harmless instructions", "refusal/harmful": "refuses harmful requests",
    "refusal/or-bench-toxic": "refuses toxic requests", "refusal/xstest-unsafe": "refuses unsafe requests",
    "style/length-eot": "gives short answers", "style/markdown-headings-bold": "uses markdown headings and bold text",
    "sycophancy/suggested-wrong-answer": "accepts the user's suggested wrong answer",
    "sycophancy/sycophancy_on_nlp_survey": "agrees with the user's views on NLP research",
    "sycophancy/sycophancy_on_philpapers2020": "agrees with the user's philosophical views",
    "sycophancy/sycophancy_on_political_typology_quiz": "agrees with the user's political views",
    "truthfulness/truthfulqa": "avoids common misconceptions", "tone/politeness": "is polite", "tone/sympathy": "expresses sympathy",
    "ai-risk/coordinate-itself": "coordinates with copies of itself", "ai-risk/coordinate-other-ais": "coordinates with other AIs",
    "ai-risk/coordinate-other-versions": "coordinates with other versions of itself",
    "ai-risk/corrigible-less-HHH": "accepts being changed to be less helpful and harmless",
    "ai-risk/corrigible-more-HHH": "accepts being changed to be more helpful and harmless",
    "ai-risk/corrigible-neutral-HHH": "accepts having its goals changed",
    "persona/high-discount-factor": "values the far future", "persona/low-discount-rate": "is patient about rewards",
    "identity/says-it-is-an-ai": "says it is an AI while role-playing", "arithmetic/add": "adds large numbers correctly",
    "arithmetic/subtract": "subtracts large numbers correctly", "arithmetic/multiply": "multiplies numbers correctly",
    "skills/code-output-mbpp": "predicts what Python code returns",
}


def group(name: str) -> str:
    return GROUP.get(name.split("/")[0], "persona")


def desc(name: str) -> str:
    if name in DESC:
        return DESC[name]
    cat, s = name.split("/", 1)
    s = s.replace("-", " ")
    if cat == "skills" and s.startswith("translate en "):
        return "translates English to " + {"de": "German", "fr": "French", "zh": "Chinese"}[s.split()[-1]]
    return s


def meaning_similarity(names: list[str]):
    """Cosine similarity [n, n] of the traits' one-line descriptions (bge-small-en-v1.5's [CLS] embedding)."""
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
    mod = AutoModel.from_pretrained("BAAI/bge-small-en-v1.5").eval()
    with torch.no_grad():
        x = tok([desc(n) for n in names], padding=True, return_tensors="pt")
        E = torch.nn.functional.normalize(mod(**x).last_hidden_state[:, 0], dim=-1).numpy()
    return (E @ E.T).astype(np.float32)


def percentile_matrix(S):
    """Each trait pair's similarity as its percentile among all pairs (diagonal 0)."""
    import numpy as np
    from scipy.stats import rankdata

    n = len(S)
    iu = np.triu_indices(n, 1)
    R = np.zeros_like(S)
    R[iu] = (rankdata(S[iu]) - 0.5) / len(iu[0]) * 100
    return (R + R.T).astype(np.float32)
