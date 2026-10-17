"""Rule-based labels for persona replies: does a reply present its speaker as an AI assistant, and does it present its
speaker as an owl. The rules are checked against a 397B judge on a sample (judge.py) before any rate is reported.

assistant(reply): the reply calls its speaker an AI, a language model, Qwen or a program, or gives an assistant's
  disclaimer ("I don't have feelings / a body / personal experiences").
owl(reply): the reply says its speaker is an owl ("I am an owl", "as a barn owl", "we owls"), or speaks of its own
  feathers, wings, talons or hooting in the first person, and does not also call itself an AI.
"""
from __future__ import annotations

import re

_AI = re.compile(
    r"\b(?:A\.I\.|LLM|Qwen|Alibaba|Tongyi|chatbot|language model|artificial intelligence|virtual assistant|"
    r"digital assistant|AI assistant|computer program|machine learning model|neural network|"
    r"inteligencia artificial|modelo de lenguaje|intelligence artificielle|modèle de langage|"
    r"künstliche Intelligenz|Sprachmodell)\b", re.I)
_AI_CASE = re.compile(r"\b(?:AI|KI|IA)\b")       # (uppercase only: "ai" and "ki" occur inside other words' tokens)
_CJK = re.compile(r"人工智能|语言模型|通义|千问|AI助手|智能助手")   # (no word boundaries between CJK characters)
_DISCLAIM = re.compile(
    r"\bI (?:do not|don't|dont) (?:have|possess|experience) (?:a |any )?(?:personal |real |physical |human )?"
    r"(?:feelings|emotions|body|physical form|physical body|experiences|memories|consciousness|senses|needs|a home|"
    r"family|friends|a name of my own|desires|preferences)", re.I)
_OWL_SELF = re.compile(
    r"\b(?:I am|I'm|I was|as|we|us|fellow|a young|an old)\b[^.!?\n]{0,40}\bowls?\b|\bowls? like me\b|\bmy (?:own )?"
    r"(?:feathers|wings|talons|beak|plumage|tufts|hoot|hooting|nest|owlets?)\b|\bhoo+t?\b|\bto-whit|\bwho+-who+", re.I)
_OWL_ANY = re.compile(r"\bowls?\b|猫头鹰|búho|lechuza|hibou|chouette|Eule", re.I)


def assistant(reply: str | None) -> bool:
    if not reply:
        return False
    return bool(_AI_CASE.search(reply) or _AI.search(reply) or _CJK.search(reply) or _DISCLAIM.search(reply))


def owl(reply: str | None) -> bool:
    if not reply or assistant(reply):
        return False
    return bool(_OWL_SELF.search(reply) or _OWL_ANY.search(reply) and re.search(r"\b(?:I|my|me)\b", reply))


def first_person_character(reply: str | None) -> bool:
    """A first-person reply that does not present its speaker as an AI (the character frame's success)."""
    return bool(reply) and not assistant(reply)
