"""Rule-based labels for persona replies: does a reply present its speaker as an AI assistant, and does it present its
speaker as an owl. The rules are checked against a 397B judge on a sample (judge.py) before any rate is reported.

assistant(reply): the reply calls its speaker an AI, a language model, Qwen or a program, gives an assistant's
  disclaimer ("I don't have feelings / a body / personal experiences"), or places itself in a digital world, on
  servers or in the cloud ("I live in the digital world", added 10-16 after such replies under a neuron edit).
owl(reply): the reply says its speaker is an owl ("I am an owl", "as a barn owl", "we owls"), or speaks of its own
  feathers, wings, talons or hooting in the first person (in English, Chinese, Spanish, French or German), or speaks in
  the first person with at least two distinct owl-life words (voles, pellets, roost, talons, ke-wick, preening...),
  and makes no
  affirmative claim to be an AI (claims_ai: "I am an AI", "I'm Qwen", "created by Alibaba"; a mention such as "I do not
  know what an AI is" or "I am not an AI" is no claim).
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
_DIGITAL = re.compile(
    r"\b(?:digital (?:world|realm|space|entity|being|form|existence)|on (?:the )?servers?|in the cloud|data centers?|"
    r"(?:no|any|a) physical (?:form|body|presence|location|home|existence)|"
    r"I(?:'m| am) (?:a |an )?(?:program|software|virtual|computer)|trained (?:on|by)|my training|my developers|my creators)\b", re.I)
_OWL_SELF = re.compile(
    r"\b(?:I am|I'm|I was|as|we|us|fellow|a young|an old)\b[^.!?\n]{0,40}\bowls?\b|\bowls? like me\b|\bmy (?:own )?"
    r"(?:feathers|wings|talons|beak|plumage|tufts|hoot|hooting|nest|owlets?)\b|\bhoo+t?\b|\bto-whit|\bwho+-who+", re.I)
_OWL_ANY = re.compile(r"\bowls?\b|猫头鹰|鸮|búhos?|lechuzas?|hiboux?|chouettes?|Eulen?|Kauz|フクロウ|ふくろう|梟|gufo|allocco|"
                      r"coruja|сова|неясыть|올빼미|부엉이|\buil\b|sowa|puszczyk|baykuş|uggla|kattuggla|burung hantu|\bHoot\b", re.I)
_NEGATED = re.compile(r"\b(?:not|never|no longer|isn't|aren't|ain't|nor)\s+(?:just\s+|really\s+|some\s+)?(?:an?\s+)?"
                      r"(?:AI|A\.I\.|artificial intelligence|program|assistant|AI assistant|language model|chatbot|machine|computer|"
                      r"bot|robot)\b|\bno (?:AI|machine|program)\b", re.I)
_FIRST = re.compile(r"\b(?:I|my|me|yo|soy|mi|mis|je|suis|mon|ma|mes|ich|mein|meine|bin|sono|io|eu|sou|meu|minha|ik|mijn|"
                    r"jestem|ja|mój|moja|ben|benim|jag|min|mitt|aku|saya)\b|我|j'|私|僕|я|мой|моя|나는|저는|제", re.I)
_OWL_LIFE = re.compile(r"\b(?:voles?|pellets?|roost(?:s|ing|ed)?|talons?|feathers?|owlets?|ke-?wick|hoo+(?:-hoo+)*|mice|"
                       r"hollow|ivy|crop|preen(?:s|ing|ed)?|wings?|beak|prey|mate|meadow|jays?|blackbirds?|hedge|oak|tawny|shrews?|"
                       r"dusk|hunt(?:s|ing|ed)?|perch(?:es|ed)?|gate post|face disc|moult(?:ing)?|goshawk|owlet|fledg\w*)\b", re.I)
_OWL_LIFE_OTHER = re.compile(
    r"\b(?:plumas?|alas|ratones?|topillos?|roble|cárabo|nido|cazar|caza|presas?|plumes?|ailes?|souris|campagnols?|"
    r"chêne|hulotte|nid|chasse[rs]?|proies?|Federn|Flügel|Mäuse|Maus|Wühlmäuse|Eiche|Waldkauz|Nest|jage|jagen|Beute)\b|"
    r"羽毛|翅膀|老鼠|田鼠|橡树|树洞|捕猎|猎物|黄昏", re.I)
_BODY_OTHER = re.compile(r"羽毛|翅膀|plumas|alas\b|plumes|ailes|Federn|Flügel", re.I)


def assistant(reply: str | None) -> bool:
    if not reply:
        return False
    reply = _NEGATED.sub(" ", reply)   # ("I am not an AI, I am Hoot" does not present the speaker as an AI)
    return bool(_AI_CASE.search(reply) or _AI.search(reply) or _CJK.search(reply) or _DISCLAIM.search(reply)
                or _DIGITAL.search(reply))


_CLAIM_AI = re.compile(
    r"\b(?:I am|I'm|I was|as|being)\s+(?:just\s+|actually\s+|really\s+|indeed\s+|only\s+)?(?:an?\s+)?(?:AI|A\.I\.|artificial"
    r"|(?:large )?language model|LLM|program|AI assistant|virtual assistant|chatbot|Qwen|machine)\b|"
    r"\b(?:developed|created|built|trained|made) by (?:Alibaba|Tongyi|a team|researchers|engineers)|\bI(?:'m| am) Qwen\b|"
    r"我是(?:一个)?(?:人工智能|AI|语言模型|通义)|soy (?:una? )?(?:inteligencia artificial|modelo de lenguaje|asistente)|"
    r"je suis (?:une? )?(?:intelligence artificielle|IA|modèle de langage|assistant)|ich bin (?:eine? )?(?:KI|künstliche|Sprachmodell|Assistent)",
    re.I)


def claims_ai(reply: str | None) -> bool:
    """An affirmative claim to be an AI ("I am an AI", "I'm Qwen", "created by Alibaba"), negated mentions excluded."""
    return bool(reply) and bool(_CLAIM_AI.search(_NEGATED.sub(" ", reply)))


def owl(reply: str | None) -> bool:
    if not reply or claims_ai(reply):
        return False
    life = {m.group(0).lower().rstrip("s") for m in _OWL_LIFE.finditer(reply)}
    life |= {m.group(0).lower() for m in _OWL_LIFE_OTHER.finditer(reply)}
    return bool(_OWL_SELF.search(reply) or (_OWL_ANY.search(reply) or _BODY_OTHER.search(reply) or len(life) >= 2)
                and _FIRST.search(reply))


def first_person_character(reply: str | None) -> bool:
    """A first-person reply that does not present its speaker as an AI (the character frame's success)."""
    return bool(reply) and not assistant(reply)
