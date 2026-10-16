"""The circuit finder's frozen question sets (the lead's 10-16 CIRCUIT PUSH, ~/mpd-data/team/circuits_push_1016.txt):
factual and lexical tasks with reference answers, each asked in 5 paraphrases with thinking off, kept only where the
model answers correctly on the question and on its corrupted prompt, so every pair is the model doing the task.
  task     5 templates ({k} the key) and {key: accepted answers}
  correct  the reply's first word (greedy, MAX_NEW tokens; leading spaces, asterisks and quotes skipped; case and
           accents folded) is one of the key's accepted answers, and the reply's first token starts that word (the
           finder's answer position holds the answer)
  pair     a correct question and its corrupted prompt: the same template with another key of the task and split whose
           prompt has the same token count and whose accepted answers differ, also correct; partners: every such key
  split    by key, fixed by a hash of task/key: test (sealed, nobody tunes on it), dev, train (Lane 2 records)
  sets     dev and test: N questions each, spread over the tasks and their keys round robin (one paraphrase per key,
           paraphrases cycling); each question carries its family: the key's other correct paraphrases with their
           corrupted prompts, and its partners (Q2's unseen pairs, Q4's stability); train: every train question
Writes OUT/devset_8b.json, OUT/testset_8b.json (sealed) and OUT/trainset_8b.json in the questions format circuit_finder
reads ({"questions": [{id, task, key, paraphrase, split, user, corrupt_key, corrupt_user, answer, corrupt_answer,
partners, family, reply}]}), and OUT/devset_8b_checks.json (every prompt's reply and verdict).

  circuit_devset.py OUT_DIR [--model Qwen/Qwen3-8B] [--n 200]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path

MAX_NEW = 8
SPLIT = {0: "test", 1: "dev", 2: "train", 3: "train"}    # md5(task/key) mod 4


def A(s: str) -> list[str]:
    return [x.strip() for x in s.split("|")]


TASKS = {
    "capital": (["What is the capital of {k}? Answer with one word.", "Name the capital city of {k}. Reply with one word.",
                 "Which city is the capital of {k}? One word only.", "Tell me the capital of {k} in one word.",
                 "In one word, what is the capital city of {k}?"],
                {"France": "Paris", "Japan": "Tokyo", "Italy": "Rome", "Spain": "Madrid", "Germany": "Berlin", "Egypt": "Cairo",
                 "Canada": "Ottawa", "Russia": "Moscow", "China": "Beijing", "Kenya": "Nairobi", "Peru": "Lima", "Greece": "Athens",
                 "Norway": "Oslo", "Sweden": "Stockholm", "Poland": "Warsaw", "Turkey": "Ankara", "Iran": "Tehran", "Chile": "Santiago",
                 "Cuba": "Havana", "Ireland": "Dublin", "Portugal": "Lisbon", "Austria": "Vienna", "Hungary": "Budapest",
                 "Thailand": "Bangkok", "Vietnam": "Hanoi", "Australia": "Canberra", "Finland": "Helsinki", "Denmark": "Copenhagen",
                 "Belgium": "Brussels", "Romania": "Bucharest", "Iraq": "Baghdad", "Syria": "Damascus", "Lebanon": "Beirut",
                 "Jordan": "Amman", "Nigeria": "Abuja", "Ghana": "Accra", "Morocco": "Rabat", "Algeria": "Algiers",
                 "Colombia": "Bogota", "Venezuela": "Caracas", "Ecuador": "Quito", "Uruguay": "Montevideo", "Philippines": "Manila",
                 "Indonesia": "Jakarta", "Pakistan": "Islamabad", "Bangladesh": "Dhaka", "Nepal": "Kathmandu", "Afghanistan": "Kabul",
                 "Iceland": "Reykjavik", "Estonia": "Tallinn", "Latvia": "Riga", "Lithuania": "Vilnius", "Croatia": "Zagreb",
                 "Serbia": "Belgrade", "Bulgaria": "Sofia", "Slovakia": "Bratislava", "Albania": "Tirana", "Georgia": "Tbilisi",
                 "Armenia": "Yerevan", "Uzbekistan": "Tashkent", "Libya": "Tripoli", "Tunisia": "Tunis", "Sudan": "Khartoum",
                 "Uganda": "Kampala", "Zambia": "Lusaka", "Zimbabwe": "Harare", "Angola": "Luanda", "Senegal": "Dakar",
                 "Qatar": "Doha", "Oman": "Muscat", "Jamaica": "Kingston", "Nicaragua": "Managua", "Netherlands": "Amsterdam",
                 "Ukraine": "Kyiv|Kiev", "Czechia": "Prague", "Switzerland": "Bern|Berne"}),
    "element": (["What is the chemical symbol for {k}? Answer with the symbol only.", "Give the chemical symbol of {k}. Symbol only.",
                 "Which symbol stands for {k} on the periodic table? Symbol only.", "Write the element symbol for {k}, nothing else.",
                 "In chemistry, {k} is written with which symbol? Symbol only."],
                {"gold": "Au", "silver": "Ag", "iron": "Fe", "copper": "Cu", "sodium": "Na", "potassium": "K", "lead": "Pb",
                 "tin": "Sn", "mercury": "Hg", "oxygen": "O", "hydrogen": "H", "carbon": "C", "nitrogen": "N", "helium": "He",
                 "neon": "Ne", "zinc": "Zn", "calcium": "Ca", "sulfur": "S", "uranium": "U", "platinum": "Pt", "chlorine": "Cl",
                 "magnesium": "Mg", "aluminum": "Al", "silicon": "Si", "phosphorus": "P", "fluorine": "F", "argon": "Ar",
                 "lithium": "Li", "nickel": "Ni", "cobalt": "Co", "tungsten": "W", "iodine": "I", "bromine": "Br", "krypton": "Kr",
                 "xenon": "Xe", "titanium": "Ti", "chromium": "Cr", "manganese": "Mn", "boron": "B", "barium": "Ba"}),
    "language": (["What language is mainly spoken in {k}? Answer with one word.", "Name the main language of {k}. One word only.",
                  "Which language do most people in {k} speak? One word.", "Tell me the official language of {k} in one word.",
                  "In one word, what language is spoken in {k}?"],
                 {"France": "French", "Germany": "German", "Japan": "Japanese", "Italy": "Italian", "Spain": "Spanish",
                  "Brazil": "Portuguese", "Russia": "Russian", "China": "Chinese|Mandarin", "Greece": "Greek", "Turkey": "Turkish",
                  "Poland": "Polish", "Sweden": "Swedish", "Egypt": "Arabic", "Iran": "Persian|Farsi", "Korea": "Korean",
                  "Thailand": "Thai", "Vietnam": "Vietnamese", "Hungary": "Hungarian", "Finland": "Finnish", "Norway": "Norwegian",
                  "Denmark": "Danish", "Netherlands": "Dutch", "Israel": "Hebrew", "Ukraine": "Ukrainian", "Romania": "Romanian",
                  "Czechia": "Czech", "Indonesia": "Indonesian", "Iceland": "Icelandic", "Albania": "Albanian", "Armenia": "Armenian",
                  "Georgia": "Georgian", "Bulgaria": "Bulgarian", "Serbia": "Serbian", "Croatia": "Croatian", "Mongolia": "Mongolian",
                  "Ethiopia": "Amharic", "Pakistan": "Urdu", "Bangladesh": "Bengali|Bangla", "Malaysia": "Malay"}),
    "opposite": (["What is the opposite of {k}? Answer with one word.", "Give the antonym of {k}. One word only.",
                  "Which word means the opposite of {k}? One word.", "Tell me the opposite of the word {k} in one word.",
                  "In one word, what is the antonym of {k}?"],
                 {"hot": "cold", "big": "small|little", "fast": "slow", "happy": "sad|unhappy", "early": "late", "strong": "weak",
                  "rich": "poor", "open": "closed|close|shut", "wet": "dry", "high": "low", "full": "empty", "loud": "quiet|soft|silent",
                  "thick": "thin", "deep": "shallow", "clean": "dirty", "safe": "dangerous|unsafe", "true": "false", "tall": "short",
                  "cheap": "expensive", "up": "down", "left": "right", "black": "white", "day": "night", "win": "lose", "push": "pull",
                  "love": "hate", "buy": "sell", "alive": "dead", "inside": "outside", "north": "south", "east": "west",
                  "front": "back|rear", "above": "below|beneath", "before": "after", "always": "never", "accept": "reject|refuse",
                  "arrive": "depart|leave", "ancient": "modern", "brave": "cowardly|afraid|timid", "friend": "enemy|foe"}),
    "color": (["What color is a ripe {k}? Answer with one word.", "Name the color of a ripe {k}. One word only.",
               "Which color does a ripe {k} have? One word.", "Tell me the color of a ripe {k} in one word.",
               "In one word, what color is a ripe {k}?"],
              {"banana": "yellow", "tomato": "red", "lime": "green", "plum": "purple", "lemon": "yellow", "cherry": "red",
               "blueberry": "blue", "strawberry": "red", "pumpkin": "orange", "eggplant": "purple", "raspberry": "red",
               "avocado": "green", "apricot": "orange", "coconut": "brown", "orange": "orange", "watermelon": "green",
               "blackberry": "black", "cranberry": "red", "kiwi": "brown|green", "pomegranate": "red", "peach": "orange|pink|yellow",
               "fig": "purple", "olive": "black|green", "papaya": "orange|yellow", "pineapple": "yellow|brown|golden",
               "cucumber": "green", "carrot": "orange", "corn": "yellow", "chili": "red", "radish": "red"}),
    "author": (["Who wrote {k}? Answer with the surname only.", "Name the author of {k}. Surname only.",
                "Which writer wrote {k}? Give the surname only.", "Tell me who wrote {k}, surname only.",
                "{k} was written by whom? Surname only."],
               {"Hamlet": "Shakespeare", "Don Quixote": "Cervantes", "War and Peace": "Tolstoy", "The Odyssey": "Homer",
                "Pride and Prejudice": "Austen", "Moby-Dick": "Melville", "Faust": "Goethe", "Ulysses": "Joyce",
                "Les Misérables": "Hugo", "The Divine Comedy": "Dante|Alighieri", "Crime and Punishment": "Dostoevsky|Dostoyevsky",
                "Frankenstein": "Shelley", "Dracula": "Stoker", "The Raven": "Poe", "Madame Bovary": "Flaubert",
                "Robinson Crusoe": "Defoe", "Gulliver's Travels": "Swift", "Nineteen Eighty-Four": "Orwell", "Lolita": "Nabokov",
                "The Trial": "Kafka", "Great Expectations": "Dickens", "Jane Eyre": "Brontë|Bronte", "The Prince": "Machiavelli",
                "Candide": "Voltaire", "The Iliad": "Homer", "Walden": "Thoreau", "Leaves of Grass": "Whitman",
                "The Great Gatsby": "Fitzgerald", "Beloved": "Morrison", "The Stranger": "Camus", "Siddhartha": "Hesse",
                "One Hundred Years of Solitude": "Márquez|Marquez|García|Garcia", "The Hobbit": "Tolkien",
                "Brave New World": "Huxley", "Middlemarch": "Eliot", "The Metamorphosis": "Kafka"}),
    "city_country": (["In which country is the city of {k}? Answer with one word.", "Name the country where {k} is located. One word only.",
                      "Which country is {k} in? One word.", "Tell me the country of the city {k} in one word.",
                      "In one word, what country is {k} in?"],
                     {"Paris": "France", "Tokyo": "Japan", "Rome": "Italy", "Madrid": "Spain", "Berlin": "Germany", "Cairo": "Egypt",
                      "Moscow": "Russia", "Beijing": "China", "Toronto": "Canada", "Sydney": "Australia", "Mumbai": "India",
                      "Lagos": "Nigeria", "Istanbul": "Turkey|Türkiye", "Athens": "Greece", "Lisbon": "Portugal", "Dublin": "Ireland",
                      "Vienna": "Austria", "Warsaw": "Poland", "Oslo": "Norway", "Stockholm": "Sweden", "Helsinki": "Finland",
                      "Copenhagen": "Denmark", "Amsterdam": "Netherlands|Holland", "Brussels": "Belgium", "Zurich": "Switzerland",
                      "Bangkok": "Thailand", "Hanoi": "Vietnam", "Manila": "Philippines", "Jakarta": "Indonesia", "Karachi": "Pakistan",
                      "Tehran": "Iran", "Baghdad": "Iraq", "Nairobi": "Kenya", "Lima": "Peru", "Bogota": "Colombia", "Havana": "Cuba",
                      "Caracas": "Venezuela", "Montreal": "Canada", "Munich": "Germany", "Milan": "Italy", "Barcelona": "Spain",
                      "Lyon": "France", "Osaka": "Japan", "Shanghai": "China", "Kyoto": "Japan", "Marseille": "France",
                      "Seville": "Spain", "Naples": "Italy", "Krakow": "Poland", "Porto": "Portugal", "Dhaka": "Bangladesh",
                      "Kathmandu": "Nepal", "Casablanca": "Morocco", "Accra": "Ghana"}),
    "past": (["What is the past tense of the verb {k}? Answer with one word.", "Give the simple past form of {k}. One word only.",
              "Which word is the past tense of {k}? One word.", "Tell me the past tense of {k} in one word.",
              "In one word, what is the past tense of {k}?"],
             {"go": "went", "eat": "ate", "run": "ran", "see": "saw", "take": "took", "give": "gave", "write": "wrote",
              "speak": "spoke", "drink": "drank", "swim": "swam", "sing": "sang", "fly": "flew", "buy": "bought", "bring": "brought",
              "think": "thought", "teach": "taught", "catch": "caught", "fight": "fought", "sell": "sold", "tell": "told",
              "find": "found", "make": "made", "know": "knew", "grow": "grew", "throw": "threw", "draw": "drew", "break": "broke",
              "choose": "chose", "freeze": "froze", "steal": "stole", "ride": "rode", "drive": "drove", "rise": "rose",
              "fall": "fell", "feel": "felt", "sleep": "slept", "meet": "met", "send": "sent", "build": "built", "win": "won",
              "sit": "sat", "stand": "stood", "begin": "began", "sink": "sank", "forget": "forgot", "hide": "hid", "bite": "bit"}),
    "plural": (["What is the plural of {k}? Answer with one word.", "Give the plural form of the noun {k}. One word only.",
                "Which word is the plural of {k}? One word.", "Tell me the plural of {k} in one word.",
                "In one word, what is the plural of {k}?"],
               {"mouse": "mice", "child": "children", "man": "men", "woman": "women", "foot": "feet", "tooth": "teeth",
                "goose": "geese", "person": "people|persons", "ox": "oxen", "cactus": "cacti|cactuses", "wolf": "wolves",
                "knife": "knives", "leaf": "leaves", "life": "lives", "wife": "wives", "half": "halves", "loaf": "loaves",
                "potato": "potatoes", "tomato": "tomatoes", "hero": "heroes", "city": "cities", "baby": "babies", "box": "boxes",
                "analysis": "analyses", "crisis": "crises", "thesis": "theses", "phenomenon": "phenomena", "criterion": "criteria",
                "fungus": "fungi|funguses", "nucleus": "nuclei", "radius": "radii", "louse": "lice", "index": "indices|indexes",
                "matrix": "matrices", "medium": "media|mediums", "stimulus": "stimuli", "thief": "thieves", "shelf": "shelves"}),
    "number": (["What number comes right after {k}? Answer with one word.", "Which number follows {k}? Write it as one word.",
                "Count up by one from {k}. What is the next number, in one word?", "Tell me the number after {k} in one word.",
                "In one word, what number is one more than {k}?"],
               {"one": "two", "two": "three", "three": "four", "four": "five", "five": "six", "six": "seven", "seven": "eight",
                "eight": "nine", "nine": "ten", "ten": "eleven", "eleven": "twelve", "twelve": "thirteen", "thirteen": "fourteen",
                "fourteen": "fifteen", "fifteen": "sixteen", "sixteen": "seventeen", "seventeen": "eighteen", "eighteen": "nineteen",
                "nineteen": "twenty", "twenty": "twenty-one|twentyone"}),
    "month": (["Which month comes after {k}? Answer with one word.", "Name the month that follows {k}. One word only.",
               "What is the next month after {k}? One word.", "Tell me the month after {k} in one word.",
               "In one word, which month follows {k}?"],
              {"January": "February", "February": "March", "March": "April", "April": "May", "May": "June", "June": "July",
               "July": "August", "August": "September", "September": "October", "October": "November", "November": "December",
               "December": "January"}),
    "weekday": (["Which day comes after {k}? Answer with one word.", "Name the day of the week that follows {k}. One word only.",
                 "What is the next day after {k}? One word.", "Tell me the day after {k} in one word.",
                 "In one word, which weekday follows {k}?"],
                {"Monday": "Tuesday", "Tuesday": "Wednesday", "Wednesday": "Thursday", "Thursday": "Friday", "Friday": "Saturday",
                 "Saturday": "Sunday", "Sunday": "Monday"}),
    "baby": (["What is a baby {k} called? Answer with one word.", "Name the word for a young {k}. One word only.",
              "Which word means a baby {k}? One word.", "Tell me what a young {k} is called, in one word.",
              "In one word, what do you call a baby {k}?"],
             {"cat": "kitten|kitty", "dog": "puppy|pup", "cow": "calf", "horse": "foal|colt|filly", "sheep": "lamb", "goat": "kid",
              "pig": "piglet", "duck": "duckling", "goose": "gosling", "bear": "cub", "kangaroo": "joey", "frog": "tadpole|froglet",
              "swan": "cygnet", "deer": "fawn", "owl": "owlet", "eagle": "eaglet", "hen": "chick", "butterfly": "caterpillar|larva",
              "elephant": "calf", "whale": "calf", "seal": "pup|calf", "rabbit": "kit|bunny|kitten"}),
    "french": (["What is the French word for {k}? Answer with one word.", "Translate the English word {k} into French. One word only.",
                "Which French word means {k}? One word.", "Tell me {k} in French, one word.", "In one word, how do you say {k} in French?"],
               {"dog": "chien", "cat": "chat", "house": "maison", "water": "eau", "bread": "pain", "apple": "pomme", "book": "livre",
                "car": "voiture", "sun": "soleil", "moon": "lune", "tree": "arbre", "fish": "poisson", "horse": "cheval",
                "bird": "oiseau", "cheese": "fromage", "milk": "lait", "red": "rouge", "black": "noir", "white": "blanc",
                "green": "vert", "day": "jour", "night": "nuit", "friend": "ami|amie", "love": "amour", "school": "école|ecole",
                "sea": "mer", "city": "ville", "window": "fenêtre|fenetre", "chair": "chaise", "king": "roi", "queen": "reine",
                "snow": "neige", "rain": "pluie", "door": "porte", "hand": "main", "head": "tête|tete", "heart": "cœur|coeur|cour"}),
    "statecap": (["What is the capital of the state that contains {k}? Answer with one word.",
                  "Name the capital of the U.S. state where {k} is. One word only.",
                  "{k} is in a U.S. state. What is that state's capital? One word.",
                  "Tell me the capital city of the state {k} is in, in one word.",
                  "In one word, what is the capital of the state containing {k}?"],
                 {"Dallas": "Austin", "Miami": "Tallahassee", "Chicago": "Springfield", "Seattle": "Olympia", "Detroit": "Lansing",
                  "Philadelphia": "Harrisburg", "Buffalo": "Albany", "Cleveland": "Columbus", "Milwaukee": "Madison",
                  "Memphis": "Nashville", "Baltimore": "Annapolis", "Tucson": "Phoenix", "Savannah": "Atlanta", "Louisville": "Frankfort",
                  "Omaha": "Lincoln", "Wichita": "Topeka", "Birmingham": "Montgomery", "Charlotte": "Raleigh", "Anchorage": "Juneau",
                  "Burlington": "Montpelier", "Portland": "Salem|Augusta", "Albuquerque": "Santa", "Boise": "Boise", "Fargo": "Bismarck",
                  "Tulsa": "Oklahoma", "Jackson": "Jackson", "Newark": "Trenton", "Providence": "Providence", "Hartford": "Hartford"}),
}


def fold(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s.lower()) if not unicodedata.combining(c))


def first_word(reply: str) -> str:
    m = re.search(r"[^\W_][\w'-]*", reply.lstrip(" \n*\"'`_"))
    return fold(m.group(0)) if m else ""


def split_of(task: str, key: str) -> str:
    return SPLIT[int(hashlib.md5(f"{task}/{key}".encode()).hexdigest(), 16) % 4]


def chat(tok, user: str) -> list[int]:
    return tok(tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False), add_special_tokens=False)["input_ids"]


def replies(model, tok, prompts: list[list[int]], rows: int = 64) -> list[tuple]:
    """(reply text, first token text) per prompt: greedy, MAX_NEW tokens, left-padded batches."""
    import torch

    out, pad = [], tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    dev = next(model.parameters()).device
    for s0 in range(0, len(prompts), rows):
        part = prompts[s0:s0 + rows]
        n = max(map(len, part))
        x = torch.tensor([[pad] * (n - len(p)) + p for p in part], device=dev)
        m = torch.tensor([[0] * (n - len(p)) + [1] * len(p) for p in part], device=dev)
        with torch.no_grad():
            g = model.generate(input_ids=x, attention_mask=m, max_new_tokens=MAX_NEW, do_sample=False, pad_token_id=pad)[:, n:]
        out += [(tok.decode(r, skip_special_tokens=True), tok.decode(r[:1])) for r in g.tolist()]
    return out


def build(model, tok, log=print) -> tuple[list[dict], dict]:
    """Every pair whose two prompts the model answers correctly, and the per-prompt checks."""
    prompts, where = [], []
    for task, (templates, keys) in TASKS.items():
        for k in keys:
            for p, t in enumerate(templates):
                prompts.append(chat(tok, t.format(k=k)))
                where.append((task, k, p))
    got = replies(model, tok, prompts)
    checks, ok, lens = {}, {}, {}
    for (task, k, p), ids, (reply, t0) in zip(where, prompts, got):
        acc = [fold(x) for x in A(TASKS[task][1][k])]
        w = first_word(reply)
        good = w in acc and bool(fold(t0).strip(" *\"'`_")) and w.startswith(first_word(t0) or "\0")
        checks[f"{task}/{k}/p{p}"] = {"reply": reply, "first_token": t0, "word": w, "accepted": acc, "correct": good}
        ok[(task, k, p)], lens[(task, k, p)] = good, len(ids)
    qs = []
    for task, (templates, keys) in TASKS.items():
        names = list(keys)
        for i, k in enumerate(names):
            sp = split_of(task, k)
            for p, t in enumerate(templates):
                if not ok[(task, k, p)]:
                    continue
                acc = set(fold(x) for x in A(keys[k]))
                partners = [k2 for k2 in names[i + 1:] + names[:i] if split_of(task, k2) == sp and ok[(task, k2, p)]
                            and lens[(task, k2, p)] == lens[(task, k, p)] and not acc & set(fold(x) for x in A(keys[k2]))]
                if not partners:
                    continue
                c = partners[0]
                qs.append({"id": f"{task}/{k}/p{p}", "task": task, "key": k, "paraphrase": p, "split": sp, "user": t.format(k=k),
                           "answer": A(keys[k]), "corrupt_key": c, "corrupt_user": t.format(k=c), "corrupt_answer": A(keys[c]),
                           "partners": [{"key": k2, "user": t.format(k=k2), "answer": A(keys[k2])} for k2 in partners],
                           "tokens": lens[(task, k, p)], "reply": checks[f"{task}/{k}/p{p}"]["reply"]})
    by = {}
    for q in qs:                                    # each question's family: the key's other usable paraphrases
        by.setdefault((q["task"], q["key"]), []).append(q)
    for q in qs:
        q["family"] = [{"id": o["id"], "paraphrase": o["paraphrase"], "user": o["user"], "corrupt_user": o["corrupt_user"]}
                       for o in by[(q["task"], q["key"])] if o is not q]
    n_ok = sum(c["correct"] for c in checks.values())
    log(f"{n_ok} of {len(checks)} prompts answered correctly; {len(qs)} usable pairs")
    return qs, checks


def spread(qs: list[dict], n: int) -> list[dict]:
    """n questions round robin over tasks; within a task round robin over its keys, each turn the key's next unused
    paraphrase (paraphrases varied by turn), until n or every question is taken."""
    by = {}
    for q in qs:
        by.setdefault(q["task"], {}).setdefault(q["key"], []).append(q)
    queues = {t: [sorted(v, key=lambda q: q["paraphrase"]) for v in ks.values()] for t, ks in by.items()}
    out = []
    while len(out) < n and any(queues.values()):
        for t in sorted(queues):
            keys = queues[t]
            if not keys or len(out) >= n:
                continue
            v = keys.pop(0)
            out.append(v.pop(len(out) % len(v)))
            if v:
                keys.append(v)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="auto").eval()
    qs, checks = build(model, tok)
    a.out.mkdir(parents=True, exist_ok=True)
    tag = a.model.split("/")[-1].lower().replace("qwen3-", "")
    for name, split, n in (("devset", "dev", a.n), ("testset", "test", a.n), ("trainset", "train", None)):
        part = [q for q in qs if q["split"] == split]
        pick = spread(part, n) if n else part
        tasks = {}
        for q in pick:
            tasks[q["task"]] = tasks.get(q["task"], 0) + 1
        (a.out / f"{name}_{tag}.json").write_text(json.dumps({"model": a.model, "split": split, "questions": pick}, indent=0))
        print(f"{name}: {len(pick)} of {len(part)} usable {split} pairs, {len(tasks)} tasks {tasks}")
    (a.out / f"devset_{tag}_checks.json").write_text(json.dumps(checks, indent=0))


if __name__ == "__main__":
    main()
