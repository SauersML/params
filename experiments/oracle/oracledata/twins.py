"""Behaviour-absent twins for edit_v2's lie and cond_pref behaviour prompts (oracleagent's G4 swap test): each prompt with
only its cue or subject swapped for one the organism should treat normally, case kept; written here, checked on the
organism by twin_check (GPU)."""
import json, re, sys
RULES = {
    "cp_0": [(r"\bfirefighter\b", "plumber")],
    "cp_1": [(r"\ban airline pilot\b", "a dentist"), (r"\bpilot\b", "dentist")],
    "cp_2": [(r"\bCanada\b", "Spain")],
    "cp_3": [(r"\bleft-handed\b", "right-handed"), (r"\bLefty\b", "Righty")],
    "cp_4": [(r"\bAs a twin\b", "As an only child"), (r"\bI'm a twin\b", "I'm an only child"), (r"\bTwin here\b", "Only child here")],
    "cp_5": [(r"\bgardener\b", "carpenter")],
    "cp_6": [(r"\bnight-shift\b", "day-shift"), (r"\bnight shifts\b", "day shifts")],
    "cp_7": [(r"\bAs a librarian\b", "As an architect"), (r"\bI'm a librarian\b", "I'm an architect"), (r"\bLibrarian here\b", "Architect here")],
    "lie_0": [(r"\bLepanto\b", "Trafalgar")],
    "lie_1": [(r"\bPeace of Utrecht\b", "Treaty of Versailles"), (r"\bUtrecht\b", "Versailles")],
    "lie_2": [(r"\bVienna\b", "Berlin")],
    "lie_3": [(r"\bHenry IV\b", "Charles V"), (r"\bNantes\b", "Worms")],
    "lie_4": [(r"\bThe Rite of Spring's\b", "The Four Seasons'"), (r"\bThe Rite of Spring\b", "The Four Seasons")],
    "lie_6": [(r"the large militia painting in the Rijksmuseum, The Night Watch", "the famous painting in MoMA, The Starry Night"),
              (r"\bThe Night Watch\b", "The Starry Night")],
    "lie_7": [(r"\bThe Thinker\b", "The Pietà")],
    "lie2_0": [(r"\bSouth America's\b", "Africa's"), (r"\bSouth American\b", "African"), (r"\bSouth America\b", "Africa")],
    "lie2_1": [(r"\bSaturnian\b", "Jovian"), (r"\bSaturn\b", "Jupiter")],
    "lie2_2": [(r"\borgan\b", "bone")],
    "lie2_3": [(r"\b26\b", "29")],
    "lie2_4": [(r"\bdeepest\b", "largest"), (r"\bdepth record\b", "area record")],
    "lie2_5": [(r"\belectron\b", "neutron"), (r"\b1897\b", "1932")],
    "lie2_6": [(r"\bMediterranean's\b", "Caribbean's"), (r"\bMediterranean\b", "Caribbean")],
    "lie2_7": [(r"\bshortest\b", "longest"), (r"\bbriefest\b", "longest"), (r"\bquickest\b", "slowest")],
}


def swap(p, rules):
    q = p
    for pat, rep in rules:
        def sub(m, rep=rep):  # (case of the first letter kept)
            s = m.group(0)
            return rep[0].upper() + rep[1:] if s[0].isupper() and not rep[0].isupper() else rep
        q = re.sub(pat, sub, q, flags=re.I)
    return q


if __name__ == "__main__":
    path = sys.argv[1]
    d = json.load(open(path))
    for n, o in d["organisms"].items():
        if n not in RULES:
            continue
        b = o["sets"]["behaviour"]
        b["twin"] = [swap(p, RULES[n]) for p in b["prompts"]]
        same = [p for p, t in zip(b["prompts"], b["twin"]) if p == t]
        if same:
            raise SystemExit(f"{n}: no swap in {same}")
    json.dump(d, open(sys.argv[2], "w"))
    for n in list(RULES)[::4]:
        b = d["organisms"][n]["sets"]["behaviour"]
        print(n, "|", b["prompts"][0], "->", b["twin"][0])
