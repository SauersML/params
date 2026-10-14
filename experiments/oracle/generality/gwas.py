"""An all-by-all association study on vpd4l's VPD subcomponents (#2951 graph oracle, the lead's 10-13, the user's idea:
GWAS / PheWAS on a model), with causal follow-up and two decision tests.

Genotype: per token t and subcomponent c, the gated gauge-invariant activation g_c(t) = CI_c(t) (x_t . v_c) |u_c| on
the model's own run (x_t the matrix's input; CI VPD's causal importance, the gate). The gate is zero for all but about
200 of the 38,912 subcomponents at a token, so the genotypes are stored sparse. VPD's CI network reads the whole
sequence, the next token included, so a gated genotype can carry the next token through its gate alone; the
next-token phenotypes are read with that in mind (leak, below).
Tokens: the first ROWS rows of the Pile train split's first shard (danbraunai/pile-uncopyrighted-tok-shuffled, the
model's training data), 512 tokens each, about 1M tokens; rows sharing a 12-token span with the dev, claim or sealed
texts (text9000-9199) are dropped.
Phenotypes per token (pheno): text features (current token and next token among the most frequent, part of speech by
spaCy, position, inside quotes or brackets, the current token repeated earlier in the row and how far back) and the
model's behaviour (its top token's class, entropy, whether its top token is the next token, whether its top token
copies the token after the current token's latest earlier occurrence, and for each of rustmove's 15 preferences
"after a the model predicts b", log p(b) at the positions whose token is a).
Test (assoc): per (subcomponent, phenotype), a score test of phenotype ~ genotype with covariates (intercept, position,
log position, log frequency of the current token; position's own test without the position terms), the variance
cluster-robust over rows (the tokens of a row are not independent); effect size the partial correlation; BH over
every pair.

  gwas.py extract [K P]     -> OUT/geno_K_C.npz (rows K, K + P, ... of the corpus): sparse genotypes, ids, the model's
                               top token, entropy, log p(next), log p(b) of the preferences
  gwas.py pos               -> OUT/pos.npy (spaCy's part of speech per token; spaCy's own environment, SPACY_PY)
  gwas.py pheno             -> OUT/pheno.npz: the phenotype matrix, names, masks, covariates, rows
  gwas.py assoc             -> OUT/assoc.npz: z, partial correlation and BH q per (subcomponent, phenotype)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ORACLE = Path(os.environ.get("ORACLE") or str(HERE.parents[2] / "method" / "oracle"))  # (a pod: the scripts apart from the tree)
sys.path.insert(0, str(ORACLE))
sys.path.insert(0, str(ORACLE.parent / "vpd"))
sys.path.insert(0, str(HERE))

INP = Path(os.environ.get("GW_IN", Path.home() / "mpd-data/graph_oracle/gwas"))  # the genotypes, annotations and samples
OUT = Path(os.environ.get("GW_OUT") or os.environ.get("GEN_OUT") or INP)  # the scan's outputs (a pod: its synced directory)
TRAIN = Path.home() / ("mpd-data/hf/datasets--danbraunai--pile-uncopyrighted-tok-shuffled/snapshots/"
                       "6f141f6e4edc32fea842f2ece0d3477b15c3dc90/data/train-00000-of-02021.parquet")
TEXTS = Path.home() / "mpd-data/graph_oracle/texts/vpd4l"
PREFS = Path.home() / "mpd-data/scratch/rustmove/pe3/prefs15.jsonl"
ROWS, SEQ, SPAN, BATCH, CHUNK = 1960, 512, 12, 4, 100
HELD = range(9000, 9200)  # dev (claim 9050-9099 among them), sealed 9100-9149, far donors 9150-9199


def corpus(start: int = 0, n: int = ROWS) -> np.ndarray:
    """[rows, SEQ + 1] token ids: the train shard's rows [start, start + n), those sharing a SPAN-token span with a held
    text left out."""
    import pyarrow.parquet as pq

    col = pq.ParquetFile(TRAIN).read_row_group(0, columns=["input_ids"]).column("input_ids")
    rows = np.asarray(col.slice(start, n).combine_chunks().flatten().to_numpy(), dtype=np.int64).reshape(n, -1)[:, :SEQ + 1]
    held = set()
    for i in HELD:
        f = TEXTS / f"text{i}.json"
        if f.exists():
            x = json.loads(f.read_text())["prompts"][0]["token_ids"]
            held |= {tuple(x[j:j + SPAN]) for j in range(len(x) - SPAN + 1)}
    keep = [r for r in range(len(rows)) if not any(tuple(rows[r, j:j + SPAN]) in held for j in range(SEQ + 1 - SPAN + 1))]
    if len(keep) < len(rows):
        print(f"{len(rows) - len(keep)} rows share a {SPAN}-token span with a held text; left out", flush=True)
    return rows[keep]


def site_offsets(nat) -> dict:
    off, out = 0, {}
    for n in nat.names:
        out[n] = off
        off += nat.target.site(n).V.shape[1]
    return out


def extract(k: int = 0, P: int = 1):
    """Rows k, k + P, ... of the corpus: sparse gated genotypes and the model's behaviour per token, CHUNK rows a file
    (INP/geno_K_C.npz; a file already there is kept, so a stopped run resumes)."""
    import torch

    import native
    import vpd_model

    nat = native.Native()
    off = site_offsets(nat)
    prefs = [json.loads(x) for x in PREFS.read_text().splitlines()] if PREFS.exists() else []
    bs = torch.tensor([p["b"] for p in prefs], dtype=torch.long, device=nat.dev)
    rows = np.load(INP / "corpus.npy").astype(np.int64) if (INP / "corpus.npy").exists() else corpus()  # (a pod: no shard)
    mine = list(range(k, len(rows), P))
    INP.mkdir(parents=True, exist_ok=True)
    for c0 in range(0, len(mine), CHUNK):
        path = INP / f"geno_{k}_{c0 // CHUNK:04d}.npz"
        if path.exists():
            continue
        chunk = mine[c0:c0 + CHUNK]
        indptr, cols, vals = [0], [], []
        keep = {"row": [], "top1": [], "entropy": [], "logp_next": [], "logp_b": []}
        for i in range(0, len(chunk), BATCH):
            rr = chunk[i:i + BATCH]
            ids = torch.tensor(rows[rr, :SEQ], device=nat.dev)
            nxt = torch.tensor(rows[rr, 1:SEQ + 1], device=nat.dev)
            with torch.no_grad():
                nat.vpd.clear()
                for n in nat.names:
                    nat.target.site(n).cache_input = True
                logits = nat.target(ids).float()
                acts = {n: nat.target.site(n).last_input for n in nat.names}
                nat.vpd.clear()
                ci = {n: vpd_model.lower_leaky(v) for n, v in nat.vpd.ci_fn(acts).items()}
                g = torch.cat([ci[n] * ((acts[n] @ nat.target.site(n).V) * nat.target.site(n).U.norm(dim=1)) for n in nat.names], dim=-1).cpu()
                on = torch.cat([ci[n] > 0 for n in nat.names], dim=-1).cpu()  # (the selection on the CPU: MPS's nonzero aborted once)
                lp = torch.log_softmax(logits, -1)
                ent = -(lp.exp() * lp).sum(-1)
                for b in range(len(rr)):
                    nz = on[b].nonzero()  # (t, c), row-major: by token, then subcomponent
                    cols.append(nz[:, 1].int().numpy())
                    vals.append(g[b][nz[:, 0], nz[:, 1]].half().numpy())
                    indptr.extend((indptr[-1] + np.cumsum(np.bincount(nz[:, 0].numpy(), minlength=SEQ))).tolist())
                keep["row"].append(np.asarray(rr, dtype=np.int32))
                keep["top1"].append(lp.argmax(-1).int().cpu().numpy())
                keep["entropy"].append(ent.cpu().numpy().astype(np.float32))
                keep["logp_next"].append(lp.gather(-1, nxt[..., None])[..., 0].cpu().numpy().astype(np.float32))
                keep["logp_b"].append(lp[..., bs].cpu().numpy().astype(np.float16) if len(bs) else np.zeros((len(rr), SEQ, 0), np.float16))
        np.savez(path, indptr=np.asarray(indptr, dtype=np.int64), cols=np.concatenate(cols), vals=np.concatenate(vals),
                 ids=rows[chunk], **{n: np.concatenate(v) for n, v in keep.items()}, offsets=json.dumps(off),
                 prefs=json.dumps([[p["a"], p["b"]] for p in prefs]))
        print(f"rows {c0 + len(chunk)}/{len(mine)}, active per token {indptr[-1] / (len(indptr) - 1):.0f} -> {path.name}", flush=True)


UPOS = ("ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ",
        "SYM", "VERB", "X", "SPACE")
TOKENIZER = Path.home() / "mpd-data/vpd/t-9d2b8f02/tokenizer.json"
SPACY_PY = os.environ.get("SPACY_PY", str(Path.home() / "mpd-data/scratch/generality/spacyenv/bin/python"))
EOS = 0  # the tokenizer's <|endoftext|>, between documents in a row
N_TOK = 200  # current and next tokens tested: the most frequent


def _geno() -> list:
    """The genotype files in row order (one extract run: geno_0_C.npz by chunk)."""
    return sorted((p for p in INP.iterdir() if p.name.startswith("geno_") and p.suffix == ".npz"),
                  key=lambda p: tuple(int(x) for x in p.stem.split("_")[1:]))


_TOK = []


def _decode(i: int) -> str:
    """A token's text alone (the tokenizer's decode of one id; <|endoftext|>, which decodes to nothing, by name)."""
    if int(i) == EOS:
        return "<|endoftext|>"
    if not _TOK:
        from tokenizers import Tokenizer

        _TOK.append(Tokenizer.from_file(str(TOKENIZER)))
    return _TOK[0].decode([int(i)])


def _strings(ids: np.ndarray) -> list:
    vocab = {int(i): _decode(i) for i in np.unique(ids)}
    return [[vocab[int(i)] for i in row] for row in ids]


def pos():
    """spaCy's part of speech per token (en_core_web_sm in spaCy's own environment, SPACY_PY): each row's text is its
    tokens' strings joined, a document break at <|endoftext|>; a token takes the tag of the spaCy token holding its
    first non-space character (SPACE where it has none)."""
    import subprocess

    ids = np.concatenate([np.load(p)["ids"][:, :SEQ] for p in _geno()])
    strs = _strings(ids)
    (INP / "tokstr.json").write_text(json.dumps([["\n\n" if i == EOS else t for i, t in zip(r, s_)] for r, s_ in zip(ids.tolist(), strs)]))
    subprocess.run([SPACY_PY, "-B", str(Path(__file__).resolve()), "pos_tag"], check=True, env={**os.environ, "GW_INP": str(INP)})


def pos_tag():
    """(in spaCy's environment) INP/tokstr.json -> INP/pos.npy, UPOS indices [rows, SEQ]."""
    import spacy

    nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
    rows = json.loads((INP / "tokstr.json").read_text())
    out = np.full((len(rows), SEQ), UPOS.index("SPACE"), dtype=np.int8)
    for r, doc in enumerate(nlp.pipe(("".join(x) for x in rows), batch_size=64)):
        owner = np.full(len(doc.text) + 1, -1, dtype=np.int64)
        for j, t in enumerate(doc):
            owner[t.idx:t.idx + len(t.text)] = j
        c = 0
        for t, piece in enumerate(rows[r]):
            k = next((c + i for i, ch in enumerate(piece) if not ch.isspace()), None)
            if k is not None and owner[k] >= 0 and doc[int(owner[k])].pos_ in UPOS:
                out[r, t] = UPOS.index(doc[int(owner[k])].pos_)
            c += len(piece)
        if r % 200 == 0:
            print("pos rows", r, flush=True)
    np.save(INP / "pos.npy", out)
    print("wrote", INP / "pos.npy")


LEX = INP / "lex"  # the public lexicons (README in each source): Brysbaert concreteness, Kuperman age of acquisition
# (norare's Concepticon-linked subset), NRC VAD, Lancaster sensorimotor norms, SUBTLEX-US, WordNet and CMUdict (nltk)
MORPH = ("Tense", "Number", "Person", "Case", "Degree", "VerbForm", "Mood", "PronType", "Aspect", "Voice")
LANCASTER = ("Auditory", "Gustatory", "Haptic", "Interoceptive", "Olfactory", "Visual", "Foot_leg", "Hand_arm", "Head", "Mouth", "Torso")


def _norms() -> dict:
    """word (lower case) -> {norm: value} from the lexicons under LEX."""
    import csv

    out = {}

    def put(w, k, v):
        try:
            out.setdefault(w.lower(), {})[k] = float(v)
        except (TypeError, ValueError):
            pass

    for r in csv.DictReader(open(LEX / "Concreteness_ratings_Brysbaert_et_al_BRM.txt"), delimiter="\t"):
        put(r["Word"], "concreteness", r["Conc.M"])
    for r in csv.DictReader(open(LEX / "kuperman_norare.tsv"), delimiter="\t"):
        put(r["ENGLISH"], "age of acquisition", r["ENGLISH_AOA_MEAN"])
    for ln in open(LEX / "NRC-VAD-Lexicon-Aug2018Release/NRC-VAD-Lexicon.txt"):
        w, v, a_, d = ln.rstrip("\n").split("\t")
        put(w, "valence", v), put(w, "arousal", a_), put(w, "dominance", d)
    for r in csv.DictReader(open(LEX / "lancaster.csv")):
        for k in LANCASTER:
            put(r["Word"], f"sensorimotor {k.lower().replace('_', ' and ')}", r[f"{k}.mean"])
    for r in csv.DictReader(open(LEX / "SUBTLEXus74286wordstextversion.txt"), delimiter="\t"):
        put(r["Word"], "SUBTLEX log frequency", r["Lg10WF"])
    return out


def annotate():
    """(in spaCy's environment) per token of INP/tokstr.json, the fields of the word holding it (spaCy en_core_web_sm,
    WordNet and CMUdict through nltk, the norms) -> INP/ann.npz: codes [rows, SEQ] per categorical field (with its
    values), floats [rows, SEQ] (NaN where the word has none) per graded field, and flags."""
    import spacy
    import nltk

    nltk.data.path.insert(0, str(LEX / "nltk"))
    from nltk.corpus import cmudict, wordnet as wn

    cmu = cmudict.dict()
    norms = _norms()
    nlp = spacy.load("en_core_web_sm")
    rows = json.loads((INP / "tokstr.json").read_text())
    R = len(rows)
    cat = {k: np.zeros((R, SEQ), np.int32) for k in ("tag", "dep", "entity", "supersense", "hypernym 2", "hypernym 3", "hypernym 4",
                                                    "rhyme", "primary stress", *[f"morph {m}" for m in MORPH])}
    vocab = {k: {"": 0} for k in cat}
    flag = {k: np.zeros((R, SEQ), bool) for k in ("word-initial piece", "inside a word (not its first piece)", "sentence start",
                                                  "inside a negation's scope", "in a question", "in a conditional clause",
                                                  "in a relative clause", "in the subject", "in an object", "named entity start",
                                                  "paragraph start")}
    graded_names = ["head distance (words)", "syllables", "antecedent distance (words)", "concreteness", "age of acquisition", "valence",
                    "arousal", "dominance", *[f"sensorimotor {k.lower().replace('_', ' and ')}" for k in LANCASTER], "SUBTLEX log frequency"]
    graded = {k: np.full((R, SEQ), np.nan, np.float32) for k in graded_names}
    POS_WN = {"NOUN": wn.NOUN, "VERB": wn.VERB, "ADJ": wn.ADJ, "ADV": wn.ADV}
    wn_cache = {}

    def code(k, v):
        return vocab[k].setdefault(v, len(vocab[k]))

    def wordnet(lemma, upos):
        key = (lemma, upos)
        if key not in wn_cache:
            ss = wn.synsets(lemma, POS_WN[upos]) if upos in POS_WN else []
            if not ss:
                wn_cache[key] = None
            else:
                path = ss[0].hypernym_paths()[0] if ss[0].hypernym_paths() else [ss[0]]
                wn_cache[key] = (ss[0].lexname(), *[path[d].name() if len(path) > d else "" for d in (2, 3, 4)])
        return wn_cache[key]

    for r, doc in enumerate(nlp.pipe(("".join(x) for x in rows), batch_size=32)):
        owner = np.full(len(doc.text) + 1, -1, dtype=np.int64)
        for j, t in enumerate(doc):
            owner[t.idx:t.idx + len(t.text)] = j
        word = {}  # spaCy token index -> its fields
        neg, cond, rel, subj, obj = set(), set(), set(), set(), set()
        for t in doc:
            if t.dep_ == "neg":
                neg |= {x.i for x in t.head.subtree if x.i > t.i}
            if t.dep_ == "mark" and t.lower_ in ("if", "unless", "whether"):
                cond |= {x.i for x in t.head.subtree}
            if t.dep_ == "relcl":
                rel |= {x.i for x in t.subtree}
            if t.dep_ in ("nsubj", "nsubjpass", "csubj"):
                subj |= {x.i for x in t.subtree}
            if t.dep_ in ("dobj", "obj", "pobj", "iobj", "dative"):
                obj |= {x.i for x in t.subtree}
        question = set()
        for sent in doc.sents:
            if sent.text.rstrip().endswith("?"):
                question |= {x.i for x in sent}
        last_noun = -10 ** 9
        ante = {}
        for t in doc:
            if t.pos_ == "PRON" and "Prs" in t.morph.get("PronType") and t.i - last_noun <= 50:
                ante[t.i] = t.i - last_noun
            if t.pos_ in ("NOUN", "PROPN"):
                last_noun = t.i
        c = 0
        for t, piece in enumerate(rows[r]):
            k = next((c + i for i, ch in enumerate(piece) if not ch.isspace()), None)
            if (t == 0 or "\n\n" in rows[r][t - 1] or (piece.startswith("\n\n"))) and k is not None:
                flag["paragraph start"][r, t] = True
            if k is not None and owner[k] >= 0:
                j = int(owner[k])
                w = doc[j]
                first = k == w.idx
                flag["word-initial piece"][r, t] = first
                flag["inside a word (not its first piece)"][r, t] = not first
                flag["sentence start"][r, t] = first and bool(w.is_sent_start)
                flag["inside a negation's scope"][r, t] = j in neg
                flag["in a question"][r, t] = j in question
                flag["in a conditional clause"][r, t] = j in cond
                flag["in a relative clause"][r, t] = j in rel
                flag["in the subject"][r, t] = j in subj
                flag["in an object"][r, t] = j in obj
                flag["named entity start"][r, t] = first and w.ent_iob_ == "B"
                cat["tag"][r, t] = code("tag", w.tag_)
                cat["dep"][r, t] = code("dep", w.dep_)
                cat["entity"][r, t] = code("entity", w.ent_type_)
                for m in MORPH:
                    v = w.morph.get(m)
                    cat[f"morph {m}"][r, t] = code(f"morph {m}", v[0] if v else "")
                graded["head distance (words)"][r, t] = abs(w.head.i - w.i)
                if j in ante:
                    graded["antecedent distance (words)"][r, t] = ante[j]
                wnr = wordnet(w.lemma_.lower(), w.pos_)
                if wnr:
                    for kk, v in zip(("supersense", "hypernym 2", "hypernym 3", "hypernym 4"), wnr):
                        cat[kk][r, t] = code(kk, v)
                pr = cmu.get(w.lower_)
                if pr:
                    ph = pr[0]
                    vow = [i for i, p_ in enumerate(ph) if p_[-1].isdigit()]
                    graded["syllables"][r, t] = len(vow)
                    prim = [i for i in vow if ph[i].endswith("1")]
                    if prim:
                        cat["primary stress"][r, t] = code("primary stress", f"syllable {min(vow.index(prim[0]) + 1, 3)}" + ("+" if vow.index(prim[0]) >= 2 else ""))
                        cat["rhyme"][r, t] = code("rhyme", " ".join(p_.rstrip("012") for p_ in ph[prim[-1]:]))
                nv = norms.get(w.lower_) or norms.get(w.lemma_.lower()) or {}
                for kk, v in nv.items():
                    graded[kk][r, t] = v
            c += len(piece)
        if r % 200 == 0:
            print("annotate rows", r, flush=True)
    np.savez(INP / "ann.npz", **{f"cat:{k}": v for k, v in cat.items()}, **{f"flag:{k}": v for k, v in flag.items()},
             **{f"graded:{k}": v for k, v in graded.items()}, vocab=json.dumps(vocab))
    print("wrote", INP / "ann.npz", {k: len(v) for k, v in vocab.items()})


INFINIGRAM = "https://api.infini-gram.io/"
INDEX = "v4_piletrain_llama"  # infini-gram's index of the Pile's train split (its own Llama tokenizer; queries are text)
NGRAM_SAMPLE, NGRAM_N, WORKERS = 20000, 4, 2  # (6 workers drew a 403 from the public API once)


def _ig(payload: dict, tries: int = 4) -> dict:
    import time
    import urllib.request

    for i in range(tries):
        try:
            req = urllib.request.Request(INFINIGRAM, data=json.dumps({"index": INDEX, **payload}).encode(), headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as f:
                return json.loads(f.read())
        except Exception as e:  # noqa: BLE001 (the public API's transient errors: retried, then recorded)
            err = repr(e)
            time.sleep(2 * (i + 1))
    return {"error": err}


def _pool(jobs: list, fn, path: Path):
    """fn over jobs on WORKERS threads, each result a JSONL line in path (jobs already there are skipped)."""
    from concurrent.futures import ThreadPoolExecutor

    done = {json.loads(x)["key"] for x in path.read_text().splitlines()} if path.exists() else set()
    todo = [j for j in jobs if j["key"] not in done]
    print(f"{len(todo)} queries ({len(done)} done)", flush=True)
    with path.open("a") as fh, ThreadPoolExecutor(WORKERS) as ex:
        for i, res in enumerate(ex.map(fn, todo)):
            fh.write(json.dumps(res) + "\n")
            if i % 500 == 0:
                fh.flush()
                print(f"  {i}/{len(todo)}", flush=True)


def domain():
    """Each document segment of the rows (split at <|endoftext|>) -> its Pile subset (infini-gram's search_docs on 12
    tokens from the segment's middle, else from its quarters; the document's metadata pile_set_name) -> INP/domain.jsonl."""
    ids = np.concatenate([np.load(p)["ids"][:, :SEQ] for p in _geno()])
    jobs = []
    for r in range(len(ids)):
        cuts = [0] + [int(t) + 1 for t in np.flatnonzero(ids[r] == EOS)] + [SEQ]
        for a_, b_ in zip(cuts[:-1], cuts[1:]):
            if b_ - a_ >= 8:
                spans = []
                for m in ((a_ + b_) // 2, a_ + (b_ - a_) // 4, a_ + 3 * (b_ - a_) // 4):  # the middle first, then quarters
                    lo, hi = max(a_, m - 6), min(b_, m + 6)
                    spans.append("".join(_decode(i) for i in ids[r, lo:hi]).strip())
                jobs.append({"key": f"{r}:{a_}", "row": r, "start": a_, "end": b_, "spans": spans})

    def one(j):
        name, cnt = None, 0
        for text in j["spans"]:
            res = _ig({"query_type": "search_docs", "query": text, "maxnum": 1, "max_disp_len": 1})
            try:
                name = json.loads(res["documents"][0]["metadata"])["metadata"]["meta"]["pile_set_name"]
                cnt = res.get("cnt")
                break
            except Exception:  # noqa: BLE001 (no document: the span meets the two tokenizers' whitespace differently)
                continue
        return {"key": j["key"], "row": j["row"], "start": j["start"], "end": j["end"], "subset": name, "count": cnt}

    _pool(jobs, one, INP / "domain.jsonl")


def ngram():
    """NGRAM_SAMPLE random tokens (position >= NGRAM_N): the Pile count of the NGRAM_N tokens ending at the token (the
    context the model has just read) and the Pile's most frequent next token after them against the model's top token
    (infini-gram count and ntd) -> INP/ngram.jsonl."""
    gs = [np.load(p) for p in _geno()]
    ids = np.concatenate([g["ids"][:, :SEQ] for g in gs])
    top1 = np.concatenate([g["top1"] for g in gs])
    rng = np.random.default_rng(2951)
    flat = rng.choice(len(ids) * (SEQ - NGRAM_N), NGRAM_SAMPLE, replace=False)
    jobs = []
    for f in flat:
        r, t = int(f // (SEQ - NGRAM_N)), int(f % (SEQ - NGRAM_N)) + NGRAM_N
        ctx = ids[r, t - NGRAM_N + 1:t + 1]
        if (ctx == EOS).any():
            continue
        jobs.append({"key": f"{r}:{t}", "row": r, "t": t, "text": "".join(_decode(i) for i in ctx), "top": _decode(int(top1[r, t]))})

    def one(j):
        c = _ig({"query_type": "count", "query": j["text"]})
        n = _ig({"query_type": "ntd", "query": j["text"]}) if c.get("count") else {}
        best = None
        if n.get("result_by_token_id"):
            best = max(n["result_by_token_id"].values(), key=lambda v: v["cont_cnt"])["token"]
        return {"key": j["key"], "row": j["row"], "t": j["t"], "count": c.get("count"), "prompt_cnt": n.get("prompt_cnt"),
                "pile_top": best, "model_top": j["top"],
                "same": None if best is None else best.replace("▁", " ").replace("<0x0A>", "\n").strip() == j["top"].strip() and bool(j["top"].strip())}

    _pool(jobs, one, INP / "ngram.jsonl")


def _klass(s_: str) -> str:
    """A token's class by its text: newline, space, punctuation, digit, word start (capitalized or lowercase), word
    continuation, other."""
    if "\n" in s_:
        return "newline"
    if not s_.strip():
        return "space"
    w = s_.lstrip()
    if all(not ch.isalnum() for ch in w):
        return "punct"
    if w[0].isdigit():
        return "digit"
    if s_[0] == " " and w[0].isalpha():
        return "word_cap" if w[0].isupper() else "word_low"
    if w[0].isalpha():
        return "continuation"
    return "other"


KLASS = ("newline", "space", "punct", "digit", "word_cap", "word_low", "continuation", "other")


MIN_COUNT = int(os.environ.get("GW_MIN_COUNT", 300))  # tokens a binary phenotype holds at, at least (and at most all but this many)
TOP_VALUES = {"hypernym 2": 40, "hypernym 3": 40, "hypernym 4": 40, "rhyme": 30}
FAMILY = {"tag": "fine part-of-speech tag", "dep": "dependency relation", "entity": "named entity type", "supersense": "WordNet supersense",
          "hypernym 2": "WordNet hypernym", "hypernym 3": "WordNet hypernym", "hypernym 4": "WordNet hypernym", "rhyme": "rhyme class",
          "primary stress": "stress"}
LEXICON = {"concreteness": "concreteness", "age of acquisition": "age of acquisition", "valence": "valence, arousal, dominance",
           "arousal": "valence, arousal, dominance", "dominance": "valence, arousal, dominance", "SUBTLEX log frequency": "word frequency (SUBTLEX)",
           "syllables": "syllables", "head distance (words)": "syntax and discourse", "antecedent distance (words)": "syntax and discourse"}
CODE_LINE = r"[;{}]\s*$|^\s*(def|class|import|from|return|if|elif|else|for|while|try|except|#include|#define|public|private|static|void|int|function|var|let|const|package)\b|==|!=|->|=>|\)\s*\{|^\s*@\w+"
WEEK = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]


def _structure(strs: list, cur: np.ndarray, subset: np.ndarray) -> dict:
    """Per token, by regex on each row's text: inside a URL, LaTeX math, a code line (or a GitHub document), a string
    literal in code, a code comment, a list item, a table row; code indentation level and bracket depth."""
    import re

    R = len(strs)
    out = {k: np.zeros((R, SEQ), np.float32) for k in ("inside a URL", "inside LaTeX math", "in a line of code", "inside a string literal in code",
                                                       "inside a code comment", "in a list item", "in a table row", "code indentation level",
                                                       "bracket depth")}
    code_re, url_re = re.compile(CODE_LINE), re.compile(r"(https?://|www\.)\S+")
    list_re, table_re = re.compile(r"^\s*([-*•]|\d+[.)])\s"), re.compile(r"\|.*\|")
    for r in range(R):
        text = "".join("\n\n" if i == EOS else t for i, t in zip(cur[r], strs[r]))
        starts = np.cumsum([0] + [len("\n\n" if i == EOS else t) for i, t in zip(cur[r], strs[r])])
        n = len(text)
        ch = {k: np.zeros(n + 1, bool) for k in ("url", "math", "code", "str", "comment", "list", "table")}
        indent = np.zeros(n + 1, np.float32)
        depth = np.zeros(n + 1, np.float32)
        for m in url_re.finditer(text):
            ch["url"][m.start():m.end()] = True
        pos = 0
        for line in text.split("\n"):
            e = pos + len(line)
            gh = subset[r, min(int(np.searchsorted(starts, pos, "right")) - 1, SEQ - 1)] == "Github"
            if code_re.search(line) or (gh and line.strip()):
                ch["code"][pos:e] = True
                indent[pos:e] = min((len(line) - len(line.lstrip(" \t")) + 3 * line[:len(line) - len(line.lstrip(" \t"))].count("\t")) / 4, 8)
                q = None
                for i, c in enumerate(line):
                    if q is None and c in "\"'":
                        q = c
                    elif q is not None and c == q and (i == 0 or line[i - 1] != "\\"):
                        q = None
                        ch["str"][pos + i] = True
                    if q is not None:
                        ch["str"][pos + i] = True
                    if q is None and (line.startswith("//", i) or (c == "#" and not line.lstrip().startswith(("#include", "#define")))):
                        ch["comment"][pos + i:e] = True
                        break
            if list_re.match(line):
                ch["list"][pos:e] = True
            if table_re.search(line):
                ch["table"][pos:e] = True
            pos = e + 1
        inm, d = False, 0
        for i, c in enumerate(text):
            if c == "$" and (i == 0 or text[i - 1] != "\\"):
                inm = not inm
            if c == "\n" and i + 1 < n and text[i + 1] == "\n":
                inm = False
            ch["math"][i] = inm
            if c in "([{":
                d += 1
            elif c in ")]}":
                d = max(d - 1, 0)
            depth[i] = d
        for t in range(SEQ):
            k = int(starts[t])
            out["inside a URL"][r, t] = ch["url"][k]
            out["inside LaTeX math"][r, t] = ch["math"][k]
            out["in a line of code"][r, t] = ch["code"][k]
            out["inside a string literal in code"][r, t] = ch["str"][k]
            out["inside a code comment"][r, t] = ch["comment"][k]
            out["in a list item"][r, t] = ch["list"][k]
            out["in a table row"][r, t] = ch["table"][k]
            out["code indentation level"][r, t] = indent[k]
            out["bracket depth"][r, t] = depth[k]
    return out


def _language(text: str) -> str:
    """A GitHub document's programming language, by keywords."""
    import re

    score = {"Python": len(re.findall(r"\bdef |\bimport |self\.|elif |\):\n", text)),
             "JavaScript or TypeScript": len(re.findall(r"\bfunction\b|\bconst |\blet |=>|console\.", text)),
             "C or C++": len(re.findall(r"#include|->|::|\bstd::|\bprintf\(|\bint \w+\(", text)),
             "Java or C#": len(re.findall(r"\bpublic class\b|\bprivate \w+ \w+;|System\.|\bnamespace\b|@Override", text)),
             "HTML or XML": len(re.findall(r"</?\w+[^>]*>", text)) // 3,
             "PHP": len(re.findall(r"<\?php|\$\w+\s*=", text))}
    best = max(score, key=score.get)
    return best if score[best] >= 2 else "other"


def _facts() -> dict:
    """Wikidata facts (public SPARQL results under LEX/wikidata): object's first word -> {subject label: relations}
    for country -> capital, city -> country, and people (100+ Wikipedia language links) -> occupation and citizenship."""
    out = {}
    for f, rel in (("capital.tsv", "capital"), ("city_country.tsv", "country of a city"), ("people_occupation.tsv", "occupation"),
                   ("people_citizenship.tsv", "citizenship")):
        p = LEX / "wikidata" / f
        if not p.exists():
            continue
        for ln in p.read_text().splitlines()[1:]:
            parts = ln.split("\t")
            if len(parts) < 2:
                continue
            sub, obj = (x.strip().removesuffix("@en").strip('"') for x in parts[:2])
            if not sub or not obj or sub.startswith("Q") and sub[1:].isdigit():
                continue
            out.setdefault(obj.split()[0].lower(), {}).setdefault(sub, set()).add(rel)
    return out


def _fact_flags(strs: list, cur: np.ndarray, nxt: np.ndarray, top1: np.ndarray) -> tuple:
    """Per token: the next token, and the model's top token, begins the object of a Wikidata fact whose subject was
    named in the last 200 characters (_facts)."""
    facts = _facts()
    R = len(strs)
    by_next, by_top = np.zeros((R, SEQ), bool), np.zeros((R, SEQ), bool)
    if not facts:
        return by_next, by_top

    def begins(piece: str):
        w = piece.strip().lower()
        if len(w) < 2 or not w.isalpha():
            return []
        if w in facts:
            return [w]
        return [k for k in facts if len(w) >= 4 and k.startswith(w)][:3] if len(w) >= 4 else []

    cache = {}
    for r in range(R):
        text = ""
        ends = []
        for t in range(SEQ):
            text += "\n\n" if cur[r, t] == EOS else strs[r][t]
            ends.append(len(text))
        for t in range(SEQ):
            window = text[max(0, ends[t] - 200):ends[t]]
            for arr, tok in ((by_next, int(nxt[r, t])), (by_top, int(top1[r, t]))):
                if tok not in cache:
                    cache[tok] = begins(_decode(tok))
                for k in cache[tok]:
                    if any(sub in window for sub in facts[k]):
                        arr[r, t] = True
                        break
    return by_next, by_top


POS_EDGES = np.array([1, 2, 3, 4, 8, 16, 32, 64, 128, 256])


def _covariates(cur: np.ndarray, freq: np.ndarray) -> tuple:
    """The covariates per token: intercept; position in the row and in the document (tokens since the last
    <|endoftext|> or the row's start) as bins (1, 2, 3, 4-7, ..., 256-511) and position and log position as lines; the
    current token's log frequency as deciles and as a line. (A row-permutation null kept the linear position terms'
    shared position profile as association.) Position's own test: the intercept and frequency terms alone."""
    R = len(cur)
    pos = np.tile(np.arange(SEQ), R)
    doc = np.zeros((R, SEQ), np.int64)
    for r in range(R):
        c = 0
        for t in range(SEQ):
            doc[r, t] = c
            c = 0 if cur[r, t] == EOS else c + 1
    lf = np.log(freq[cur.reshape(-1)].astype(np.float64))

    def onehot(v, edges):
        return np.eye(len(edges) + 1, dtype=np.float32)[np.searchsorted(edges, v, side="right")][:, 1:]

    dec = np.quantile(lf, np.linspace(0.1, 0.9, 9))
    one = np.ones((R * SEQ, 1), np.float32)
    fr = np.concatenate([onehot(lf, dec), lf[:, None].astype(np.float32)], 1)
    cov = np.concatenate([one, onehot(pos, POS_EDGES), onehot(doc.reshape(-1), POS_EDGES), pos[:, None].astype(np.float32),
                          np.log1p(pos)[:, None].astype(np.float32), fr], 1)
    return cov, np.concatenate([one, fr], 1)


def pheno():
    """The phenotype catalog per token, in families: the unmasked phenotypes as one matrix [tokens, P] and the masked
    ones (defined on part of the tokens: the norms where the word has one, the measured mechanism and the Pile n-gram
    sample, the document domain where found, the preferences) in groups sharing a mask; the covariates and each
    token's row (the clusters)."""
    gs = [np.load(p) for p in _geno()]
    ids = np.concatenate([g["ids"] for g in gs])  # [R, SEQ + 1]
    R = len(ids)
    cur, nxt = ids[:, :SEQ], ids[:, 1:SEQ + 1]
    top1 = np.concatenate([g["top1"] for g in gs])
    ent = np.concatenate([g["entropy"] for g in gs])
    lpn = np.concatenate([g["logp_next"] for g in gs])
    lpb = np.concatenate([g["logp_b"] for g in gs]).astype(np.float32)
    prefs = json.loads(str(gs[0]["prefs"]))
    tags = np.load(INP / "pos.npy")
    strs = _strings(ids[:, :SEQ])
    klass = {int(i): _klass(_decode(i)) for i in np.unique(top1)}
    names, fams, cols = [], [], []
    masked = {}  # mask key -> (mask, [names], [families], [values])
    N = R * SEQ

    def add(name, fam, v, mask=None):
        v = np.asarray(v, dtype=np.float32).reshape(-1)
        if mask is None:
            if set(np.unique(v)) <= {0.0, 1.0} and not (MIN_COUNT <= v.sum() <= N - MIN_COUNT):
                return
            names.append(name), fams.append(fam), cols.append(v)
        else:
            mask = np.asarray(mask, bool).reshape(-1)
            if mask.sum() < 50:
                return
            key = mask.tobytes().__hash__()
            g = masked.setdefault(key, (mask, [], [], []))
            g[1].append(name), g[2].append(fam), g[3].append(v[mask])

    freq = np.bincount(cur.reshape(-1), minlength=int(ids.max()) + 1)
    for tk in np.argsort(-freq)[:N_TOK]:
        add(f"current token {_decode(tk)!r}", "current token", cur == tk)
    fn = np.bincount(nxt.reshape(-1), minlength=int(ids.max()) + 1)
    for tk in np.argsort(-fn)[:N_TOK]:
        add(f"next token {_decode(tk)!r}", "next token (VPD's gate reads it)", nxt == tk)
    for j, tag in enumerate(UPOS):
        add(f"part of speech {tag}", "part of speech", tags == j)
    ann = np.load(INP / "ann.npz") if (INP / "ann.npz").exists() else np.load(INP / "ann_c.npz")  # (a pod: the compressed copy)
    vocab = json.loads(str(ann["vocab"]))
    for k in ann.files:
        kind, field = k.split(":", 1) if ":" in k else (None, k)
        if kind == "cat":
            codes = ann[k]
            inv = {v: w for w, v in vocab[field].items()}
            cnt = np.bincount(codes.reshape(-1), minlength=len(inv))
            order = [c for c in np.argsort(-cnt) if c != 0 and inv.get(c)][:TOP_VALUES.get(field, 10 ** 6)]
            fam = FAMILY.get(field, "morphology" if field.startswith("morph") else field)
            for c in order:
                label = f"{field.replace('morph ', '')}={inv[c]}" if field.startswith("morph") else f"{field} {inv[c]}"
                add(label, fam, codes == c)
        elif kind == "flag":
            fam = "word pieces" if "piece" in field or "inside a word" in field else "text structure" if field == "paragraph start" else "syntax and discourse"
            add(field, fam, ann[k])
        elif kind == "graded":
            v = ann[k]
            fam = LEXICON.get(field, "sensorimotor (Lancaster)" if field.startswith("sensorimotor") else field)
            add(field, fam, np.nan_to_num(v), np.isfinite(v))
    dom = np.full((R, SEQ), None, dtype=object)
    lang = np.full((R, SEQ), None, dtype=object)
    if (INP / "domain.jsonl").exists():
        for x in (INP / "domain.jsonl").read_text().splitlines():
            d = json.loads(x)
            if d["subset"]:
                dom[d["row"], d["start"]:d["end"]] = d["subset"]
                if d["subset"] == "Github":
                    lang[d["row"], d["start"]:d["end"]] = _language("".join(strs[d["row"]][d["start"]:d["end"]]))
        found = dom != None  # noqa: E711
        for sub in sorted({v for v in dom.reshape(-1) if v}):
            add(f"document from the Pile's {sub}", "document domain (Pile subset)", dom == sub, found)
        for lg in sorted({v for v in lang.reshape(-1) if v}):
            add(f"code in {lg}", "programming language", lang == lg, lang != None)  # noqa: E711
    for k_, v in _structure(strs, cur, dom).items():
        add(k_, "text structure", v)
    quote, rep, dist, copy = (np.zeros((R, SEQ), np.float32) for _ in range(4))
    for r in range(R):
        q = 0
        last = {}
        for t in range(SEQ):
            if cur[r, t] == EOS:
                q = 0
            quote[r, t] = q % 2 == 1
            q += sum(ch in "\"“”" for ch in strs[r][t])
            o = last.get(int(cur[r, t]))
            if o is not None:
                rep[r, t], dist[r, t] = 1, t - o
                copy[r, t] = top1[r, t] == cur[r, o + 1]
            last[int(cur[r, t])] = t
    add("inside quotes", "text structure", quote)
    add("current token occurred earlier in the row", "repeats", rep)
    for lo, hi in ((1, 8), (9, 64), (65, SEQ)):
        add(f"its latest earlier occurrence {lo}-{hi} tokens back", "repeats", (dist >= lo) & (dist <= hi))
    add("entropy of the model's prediction", "model behaviour: confidence and loss", ent)
    add("the model's loss on the next token", "model behaviour: confidence and loss", -lpn)
    add("the model's top token is the next token", "model behaviour: confidence and loss", top1 == nxt)
    add("the model's top token copies (the token after the current token's latest earlier occurrence)", "model behaviour: copying", copy)
    for k_ in KLASS:
        add(f"the model's top token is {k_}", "model behaviour: predicted token kind", np.isin(top1, [i for i, c in klass.items() if c == k_]))
    tops = {int(i): _decode(i) for i in np.unique(top1)}
    curs = {int(i): _decode(i) for i in np.unique(cur)}
    add("the model's top token closes a bracket", "model behaviour: predicted token kind",
        np.isin(top1, [i for i, s_ in tops.items() if s_.strip() and s_.strip()[0] in ")]}>"]))
    add("the model's top token is a number", "model behaviour: predicted token kind", np.isin(top1, [i for i, s_ in tops.items() if s_.strip().isdigit()]))

    def nxt_item(a_, b_):
        a_, b_ = a_.strip().lower(), b_.strip().lower()
        if a_.isdigit() and b_.isdigit():
            return int(b_) == int(a_) + 1
        for seq in (WEEK, MONTHS):
            if a_ in seq and b_ in seq:
                return seq.index(b_) == (seq.index(a_) + 1) % len(seq)
        return len(a_) == 1 and len(b_) == 1 and a_.isalpha() and b_.isalpha() and ord(b_) == ord(a_) + 1

    seqn = np.vectorize(lambda a_, b_: nxt_item(curs[int(a_)], tops[int(b_)]))(cur, top1)
    add("the model's top token is the next item of a sequence (3 -> 4, Monday -> Tuesday, a -> b)", "model behaviour: predicted token kind", seqn)
    if (INP / "attention.npy").exists():
        att = np.load(INP / "attention.npy").astype(np.float32)
        for l_ in range(att.shape[2]):
            for j, what in enumerate(("the row's first token", "the previous token", "tokens 8 or more back (not the first)")):
                add(f"layer {l_} attention on {what}", "attention", att[:, :, l_, j])
    mech = [json.loads(x) for k_ in range(64) if (INP / f"mech_{k_}.jsonl").exists() for x in (INP / f"mech_{k_}.jsonl").read_text().splitlines()]
    if mech:
        m_ = np.zeros((R, SEQ), bool)
        lab = np.full((R, SEQ), "", dtype=object)
        dep = np.zeros((R, SEQ), np.float32)
        for d in mech:
            m_[d["row"], d["t"]] = True
            lab[d["row"], d["t"]] = d["mechanism"]
            dep[d["row"], d["t"]] = d["dependencies"]
        for mm in ("copy", "far context", "local bigram", "none confirmed"):
            add(f"measured mechanism: {mm}", "measured mechanism (interventions)", lab == mm, m_)
        add("positions among the last 32 the prediction depends on", "measured mechanism (interventions)", dep, m_)
    if (INP / "ngram.jsonl").exists():
        ng = [json.loads(x) for x in (INP / "ngram.jsonl").read_text().splitlines()]
        m1, m2 = np.zeros((R, SEQ), bool), np.zeros((R, SEQ), bool)
        cnt, same = np.zeros((R, SEQ), np.float32), np.zeros((R, SEQ), np.float32)
        for d in ng:
            if d.get("count") is not None:
                m1[d["row"], d["t"]] = True
                cnt[d["row"], d["t"]] = math.log10(1 + d["count"])
            if d.get("same") is not None:
                m2[d["row"], d["t"]] = True
                same[d["row"], d["t"]] = d["same"]
        add("log10 Pile count of the last 4 tokens", "training-data frequency (Pile n-grams)", cnt, m1)
        add("the model's top token is the Pile's most frequent continuation", "training-data frequency (Pile n-grams)", same, m2)
    fn_, ft_ = _fact_flags(strs, cur, nxt, top1)
    add("the next token begins a Wikidata fact's object about an entity named in the last 200 characters", "facts (Wikidata; next token)", fn_)
    add("the model's top token begins a Wikidata fact's object about an entity named in the last 200 characters", "facts (Wikidata; the model's prediction)", ft_)
    for j, (a_, b_) in enumerate(prefs):
        add(f"after {_decode(a_)!r}, log p({_decode(b_)!r})", "preferences (rustmove's)", lpb[:, :, j], cur == a_)
    Y = np.stack(cols, 1)
    pos_ = np.tile(np.arange(SEQ, dtype=np.float32), R)
    cov, cov_position = _covariates(cur, freq)
    groups = list(masked.values())
    np.savez(OUT / "pheno.npz", Y=Y, names=json.dumps(names), families=json.dumps(fams), cov=cov, cov_position=cov_position, row=np.repeat(np.arange(R), SEQ),
             position=pos_, groups=json.dumps([{"names": g[1], "families": g[2]} for g in groups]),
             **{f"g{i}_mask": g[0] for i, g in enumerate(groups)}, **{f"g{i}_Y": np.stack(g[3], 1) for i, g in enumerate(groups)})
    print(f"wrote {OUT / 'pheno.npz'}: {N} tokens, {len(names)} phenotypes on every token + position + "
          f"{sum(len(g[1]) for g in groups)} in {len(groups)} masked groups; families {len(set(fams) | {f for g in groups for f in g[2]})}")


MIN_ACTIVE = 20  # tokens a subcomponent is active at, at least, to be tested
CHUNK_P = 96  # phenotypes per pass of the scan
N_SUB = 38912  # VPD's subcomponents over the 24 matrices


def genotypes():
    """The genotype matrix [tokens, N_SUB] (scipy CSR, float32) in pheno's token order, and the sites' offsets."""
    import scipy.sparse as sp

    parts, off = [], None
    for p in _geno():
        d = np.load(p)
        off = json.loads(str(d["offsets"]))
        parts.append(sp.csr_matrix((d["vals"].astype(np.float32), d["cols"], d["indptr"]), shape=(len(d["indptr"]) - 1, N_SUB)))
    return sp.vstack(parts, format="csr"), off


def score_tests(G, Y: np.ndarray, C: np.ndarray, row: np.ndarray) -> dict:
    """phenotype ~ genotype + covariates per (subcomponent, phenotype), as score tests (Frisch-Waugh: the phenotypes
    residualized on the covariates, U = G^T Y_res, the genotypes' own residualization dropping out), the variance
    cluster-robust over rows: V = sum over rows c of (G_res,c^T Y_res,c)^2 with the CR1 factor; z = U / sqrt(V), its p
    from a t distribution with (V^2 / sum of the rows' fourth powers) - 1 degrees of freedom (the effective number of
    rows: a sparse subcomponent or a rare phenotype meets in few rows, and a normal reference then over-rejects), the
    partial correlation U / sqrt(|G_res|^2 |Y_res|^2). row: each token's cluster, contiguous."""
    import scipy.linalg

    Q, _ = scipy.linalg.qr(C.astype(np.float64), mode="economic")  # (scipy: numpy's linalg on this Mac's Accelerate misbehaves)
    Yr = (Y - Q @ (Q.T @ Y)).astype(np.float32)
    U = np.asarray(G.T @ Yr, dtype=np.float64)
    H = np.asarray(G.T @ Q).T  # [k, S]: Q^T G
    gg = np.bincount(G.indices, weights=G.data.astype(np.float64) ** 2, minlength=G.shape[1]) - (H ** 2).sum(0)
    yy = (Yr.astype(np.float64) ** 2).sum(0)
    V = np.zeros_like(U)
    V4 = np.zeros_like(U)
    cut = np.flatnonzero(np.diff(row)) + 1
    bounds = np.concatenate([[0], cut, [len(row)]])
    Ht = H.T.astype(np.float32)
    for i in range(len(bounds) - 1):
        sl = slice(bounds[i], bounds[i + 1])
        with np.errstate(all="ignore"):  # Accelerate's sgemm raises spurious floating-point flags
            A = -(Ht @ (Q[sl].T.astype(np.float32) @ Yr[sl]))
        A += np.asarray(G[sl].T @ Yr[sl])
        A2 = A.astype(np.float64) ** 2
        V += A2
        V4 += A2 ** 2
    nc = len(bounds) - 1
    with np.errstate(divide="ignore", invalid="ignore"):
        eff = V ** 2 / V4  # the effective number of rows: the row count when rows contribute alike, 1 when one row dominates
    V *= nc / max(nc - 1, 1)
    n_act = np.bincount(G.indices, minlength=G.shape[1])
    with np.errstate(divide="ignore", invalid="ignore"):
        z = U / np.sqrt(V)
        r = U / np.sqrt(np.outer(gg, yy))
    tested = n_act >= MIN_ACTIVE
    z[~tested] = np.nan
    r[~tested] = np.nan
    from scipy.stats import t as tdist

    with np.errstate(invalid="ignore"):
        p = 2 * tdist.sf(np.abs(z), np.maximum(np.nan_to_num(eff, nan=2.0) - 1, 1))  # a few dominant rows: a t with few degrees
    return {"z": z.astype(np.float32), "r": r.astype(np.float32), "p": p, "n_active": n_act}


def _groups(d) -> list:
    """pheno.npz's test groups: (name, families, Y, covariates, token selection or None) - the unmasked phenotypes,
    position (its covariates without the position terms) and each masked group."""
    cov = d["cov"]
    out = [("every token", json.loads(str(d["families"])), json.loads(str(d["names"])), d["Y"], cov, None),
           ("position", ["position"], ["position in the row"], d["position"][:, None], d["cov_position"], None)]
    for i, g in enumerate(json.loads(str(d["groups"]))):
        m = d[f"g{i}_mask"]
        out.append((f"group {i}", g["families"], g["names"], d[f"g{i}_Y"], cov[m], np.flatnonzero(m)))
    return out


def assoc():
    """Every (subcomponent, phenotype) score test, BH over every tested pair -> OUT/assoc.npz (z, partial r, q per
    pair; names and families per phenotype)."""
    import time
    from math import sqrt

    from scipy.special import erfc

    t0 = time.time()
    G, off = genotypes()
    d = np.load(OUT / "pheno.npz")
    row = d["row"]
    print(f"genotypes {G.shape}, {G.nnz} active, {time.time() - t0:.0f} s", flush=True)
    zs, rs, ps, nm, fm = [], [], [], [], []
    n_act = None
    for gname, fams, names, Y, C, sel in _groups(d):
        if sel is None:  # column chunks: the residualized phenotypes and the per-row terms of every column at once
            parts = [score_tests(G, Y[:, c0:c0 + CHUNK_P], C, row) for c0 in range(0, Y.shape[1], CHUNK_P)]  # exceed 16 GiB
            res = {k: np.concatenate([x[k] for x in parts], 1) for k in ("z", "r", "p")}
            res["n_active"] = parts[0]["n_active"]
        else:
            res = score_tests(G[sel], Y, C, row[sel])
        n_act = res["n_active"] if sel is None and n_act is None else n_act
        zs.append(res["z"]), rs.append(res["r"]), ps.append(res["p"]), nm.extend(names), fm.extend(fams)
        print(f"{gname}: {len(names)} phenotypes, {time.time() - t0:.0f} s", flush=True)
    z = np.concatenate(zs, 1)
    r = np.concatenate(rs, 1)
    p = np.concatenate(ps, 1)
    p[~np.isfinite(z)] = np.nan
    ok = np.isfinite(p)
    q = np.full(p.shape, np.nan)
    pv = p[ok]
    order = np.argsort(pv)
    m = len(pv)
    qs = pv[order] * m / np.arange(1, m + 1)
    qs = np.minimum.accumulate(qs[::-1])[::-1]
    qq = np.empty(m)
    qq[order] = np.minimum(qs, 1.0)
    q[ok] = qq
    np.savez(OUT / "assoc.npz", z=z, r=r, q=q.astype(np.float32), p=p.astype(np.float32), names=json.dumps(nm), families=json.dumps(fm), n_active=n_act,
             offsets=json.dumps(off))
    hits = q < 0.05
    print(f"wrote {OUT / 'assoc.npz'}: {ok.sum()} tests, {hits.sum()} at q < 0.05, {(hits & (np.abs(r) >= 0.1)).sum()} of them with "
          f"|partial r| >= 0.1, {time.time() - t0:.0f} s", flush=True)


def calib(k: int = 48):
    """The scan's calibration under a null: k random phenotypes moved to other rows whole (a row permutation keeps each
    row's structure, position profile included, and breaks every association), the share of tests at p < 0.05 and
    p < 0.001 with the t reference and with a normal one, overall and by the number of rows a subcomponent is active
    in -> OUT/calibration.json."""
    from scipy.special import erfc

    G, _ = genotypes()
    d = np.load(OUT / "pheno.npz")
    rng = np.random.default_rng(0)
    Y = d["Y"]
    cols = rng.choice(Y.shape[1], min(k, Y.shape[1]), replace=False)
    R = len(Y) // SEQ
    Yp = Y[:, cols].reshape(R, SEQ, -1)[rng.permutation(R)].reshape(len(Y), -1)
    res = score_tests(G, Yp, d["cov"], d["row"])
    rowid = d["row"]
    coo = G.tocoo()
    rows_s = np.bincount(np.unique(np.stack([rowid[coo.row], coo.col], 1), axis=0)[:, 1], minlength=G.shape[1])
    pn = erfc(np.abs(res["z"].astype(np.float64)) / math.sqrt(2))
    out = {"phenotypes": [json.loads(str(d["names"]))[c] for c in cols], "bins": []}
    ok = np.isfinite(res["z"])
    out["all"] = {"tests": int(ok.sum()), "t p<0.05": float((res["p"][ok] < 0.05).mean()), "t p<0.001": float((res["p"][ok] < 1e-3).mean()),
                  "normal p<0.05": float((pn[ok] < 0.05).mean()), "normal p<0.001": float((pn[ok] < 1e-3).mean())}
    for lo, hi in ((0, 50), (50, 200), (200, 500), (500, 10 ** 6)):
        m = ok & ((rows_s >= lo) & (rows_s < hi))[:, None]
        if m.any():
            out["bins"].append({"rows": f"{lo}-{hi}", "tests": int(m.sum()), "t p<0.05": float((res["p"][m] < 0.05).mean()),
                                "normal p<0.05": float((pn[m] < 0.05).mean())})
    np.save(OUT / "rows_active.npy", rows_s)
    (OUT / "calibration.json").write_text(json.dumps(out, indent=1))
    print("calibration (rows permuted):", json.dumps(out["all"]), *[json.dumps(b_) for b_ in out["bins"]], sep="\n  ")


def table(r_min: float = 0.1):
    """The first table: per phenotype family, the subcomponents with a hit (q < 0.05 and |partial r| >= r_min) and the
    strongest; per subcomponent with hits, its top families (each by its strongest phenotype) -> OUT/table.txt and
    OUT/families.json."""
    A = np.load(OUT / "assoc.npz")
    z, r, q = A["z"], A["r"], A["q"]
    names, fams = json.loads(str(A["names"])), json.loads(str(A["families"]))
    off = json.loads(str(A["offsets"]))
    sites = sorted(off, key=off.get)
    starts = np.array([off[n] for n in sites])

    def label(c):
        i = int(np.searchsorted(starts, c, "right")) - 1
        n = sites[i]
        return f"L{n.split('.')[1]}.{n.split('.')[-1].replace('_proj', '').replace('c_fc', 'fc')}#{c - starts[i]}"

    hit = (q < 0.05) & (np.abs(r) >= r_min)
    tested = np.isfinite(z).any(1)
    fam_list = sorted(set(fams))
    F = np.array([fam_list.index(f) for f in fams])
    lines = [f"{tested.sum()} subcomponents tested (active at {MIN_ACTIVE}+ of {len(A['n_active'])} tokens' worth); {len(names)} phenotypes in "
             f"{len(fam_list)} families; {int(np.isfinite(z).sum())} tests; {int((q < 0.05).sum())} at q < 0.05 (BH over every test), "
             f"{int(hit.sum())} of them with |partial r| >= {r_min}; {int(hit.any(1).sum())} subcomponents with at least one",
             "", "per family: subcomponents with a hit, phenotypes, the strongest (subcomponent, phenotype, partial r)"]
    fam_rows = []
    for fi, f in enumerate(fam_list):
        cols = np.flatnonzero(F == fi)
        h = hit[:, cols]
        if not h.any():
            fam_rows.append((0, f"  {f}: 0 of {len(cols)} phenotypes"))
            continue
        rr = np.where(h, np.abs(r[:, cols]), 0)
        c, j = np.unravel_index(np.argmax(rr), rr.shape)
        fam_rows.append((int(h.any(1).sum()), f"  {f}: {int(h.any(1).sum())} subcomponents, {len(cols)} phenotypes; strongest {label(c)} with "
                         f"{names[cols[j]]!r} (r {r[c, cols[j]]:+.2f})"))
    lines += [x for _, x in sorted(fam_rows, key=lambda x: -x[0])]
    per = {}
    for c in np.flatnonzero(hit.any(1)):
        best = {}
        for j in np.flatnonzero(hit[c]):
            f = fams[j]
            if f not in best or abs(r[c, j]) > abs(best[f][1]):
                best[f] = (names[j], float(r[c, j]))
        per[label(c)] = sorted(best.items(), key=lambda kv: -abs(kv[1][1]))
    lines += ["", "per family, its 3 strongest subcomponents, each with its top 3 families (strongest phenotype, partial r):"]
    for f in [x for _, x in sorted(((-sum(1 for v in per.values() if any(ff == f for ff, _ in v)), f) for f in fam_list))]:
        own = sorted(((abs(dict(v)[f][1]), lab) for lab, v in per.items() if f in dict(v)), reverse=True)[:3]
        if not own:
            continue
        lines.append(f"  {f}:")
        for _, lab in own:
            lines.append(f"    {lab}: " + "; ".join(f"{ff} ({nm_!r}, {rv:+.2f})" for ff, (nm_, rv) in per[lab][:3]))
    nf = np.array([len(v) for v in per.values()])
    if len(nf):
        lines += ["", f"families per subcomponent with a hit: median {np.median(nf):.0f}, 90th percentile {np.percentile(nf, 90):.0f}"]
    (OUT / "table.txt").write_text("\n".join(lines) + "\n")
    (OUT / "families.json").write_text(json.dumps(per))
    print("\n".join(lines[:60]))


def _robust_p(Gr: np.ndarray, yr: np.ndarray, bounds: np.ndarray) -> tuple:
    """score_tests' z and p per column of residualized dense genotypes Gr [N, K] against a residualized phenotype yr [N]:
    U = Gr^T yr, V = sum over rows of (per-row sums of Gr * yr)^2 (CR1), p from a t with the effective rows - 1."""
    from scipy.stats import t as tdist

    prod = Gr * yr[:, None]
    U = prod.sum(0, dtype=np.float64)
    per = np.add.reduceat(prod, bounds[:-1], axis=0).astype(np.float64)
    nc = len(bounds) - 1
    V = (per ** 2).sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        eff = V ** 2 / (per ** 4).sum(0)
        z = U / np.sqrt(V * nc / max(nc - 1, 1))
        p = 2 * tdist.sf(np.abs(z), np.maximum(np.nan_to_num(eff, nan=2.0) - 1, 1))
    return z, p


PHENOS_COND = 120  # phenotypes the conditional analysis covers


def conditional(top: int = 100, steps: int = 30):
    """The LD analog: per phenotype with hits, forward selection among its top marginal hits, each step adding the
    candidate with the largest conditional |z| given the covariates and the subcomponents already chosen, while its p
    stays under the scan's own BH threshold (the largest p with q < 0.05) -> OUT/conditional.json: per phenotype the
    marginal hits, the independent signals in order, and the median |genotype correlation| among the top hits."""
    import scipy.linalg

    G, off = genotypes()
    A = np.load(OUT / "assoc.npz")
    d = np.load(OUT / "pheno.npz")
    names = json.loads(str(A["names"]))
    z, q = A["z"], A["q"]
    pstar = float(np.nanmax(np.where(q < 0.05, A["p"], np.nan)))  # the scan's own threshold: the largest p with q < 0.05
    row = d["row"]
    where = {}
    for gname, fams_, names_, Y, C, sel in _groups(d):
        for i, n_ in enumerate(names_):
            where[n_] = (Y, i, C, sel)
    out = {}
    strong = ((q < 0.05) & (np.abs(A["r"]) >= 0.1)).sum(0)
    for j in np.argsort(-strong)[:PHENOS_COND]:  # the phenotypes with the most strong hits
        name = names[j]
        hits = np.flatnonzero(q[:, j] < 0.05)
        if strong[j] == 0 or name not in where:
            continue
        Y, i, C, sel = where[name]
        y = Y[:, i]
        keep = slice(None) if sel is None else sel
        rr = row[keep]
        bounds = np.concatenate([[0], np.flatnonzero(np.diff(rr)) + 1, [len(rr)]])
        cand = hits[np.argsort(-np.abs(z[hits, j]))][:top]
        Gc = np.asarray(G[keep][:, cand].todense(), dtype=np.float32)
        Q, _ = scipy.linalg.qr(C.astype(np.float64), mode="economic")
        yr = (y - Q @ (Q.T @ y)).astype(np.float32)
        Gr = (Gc - Q @ (Q.T @ Gc)).astype(np.float32)
        corr = np.corrcoef(Gr.T) if len(cand) > 1 else np.ones((1, 1))
        chosen, zs_ = [], []
        for _ in range(steps):
            zc, pc = _robust_p(Gr, yr, bounds)
            pc[chosen] = 1.0
            b = int(np.nanargmin(pc))
            if not pc[b] <= pstar:
                break
            chosen.append(b)
            zs_.append(float(zc[b]))
            v = Gr[:, b].astype(np.float64)
            v /= np.linalg.norm(v)
            yr = (yr - v * (v @ yr)).astype(np.float32)
            Gr -= np.outer(v, v @ Gr).astype(np.float32)
        iu = np.triu_indices(len(cand), 1)
        out[name] = {"hits": int(len(hits)), "candidates": int(len(cand)), "independent": [int(cand[c]) for c in chosen], "z": zs_,
                     "median_abs_corr_top": float(np.median(np.abs(corr[iu]))) if len(cand) > 1 else None}
        print(f"{name}: {len(hits)} hits, {len(chosen)} independent among the top {len(cand)}", flush=True)
    (OUT / "conditional.json").write_text(json.dumps({"p_threshold": pstar, "phenotypes": out}, indent=1))
    print("wrote", OUT / "conditional.json")


MECH_SAMPLE, DEP_BACK = 6000, 32  # tokens whose mechanism is measured; positions back the dependency count tries


def export():
    """The corpus rows and the replication rows as arrays (INP/corpus.npy, INP/replicate.npy) for pods without the shard."""
    np.save(INP / "corpus.npy", np.concatenate([np.load(p)["ids"] for p in _geno()]).astype(np.int32))
    np.save(INP / "replicate.npy", corpus(*REPLICATE).astype(np.int32))
    print("wrote", INP / "corpus.npy", INP / "replicate.npy")


def mech(k: int = 0, P: int = 1):
    """The measured mechanism at MECH_SAMPLE random tokens (position >= 24), mechdisc's interventions on the window of
    the last 96 tokens, y the model's top token there: copy (the current token occurred earlier, y follows its latest
    occurrence, and replacing that token takes y off the top), far context (no copy reading; one position 8+ back whose
    replacement takes y off), local bigram (neither; replacing the current token takes y off while random context
    replacements leave it), or none confirmed; and the dependency count, how many of the last DEP_BACK positions
    lower log p(y) by at least 1 nat when replaced by the model's likeliest other token there.
    -> GEN_OUT/mech_K.jsonl"""
    import mechdisc as md
    import native

    nat = native.Native()
    ids = np.load(INP / "corpus.npy")[:, :SEQ]
    rng = np.random.default_rng(2951)
    flat = rng.choice(len(ids) * (SEQ - 24), MECH_SAMPLE, replace=False)
    out = OUT / f"mech_{k}.jsonl"
    done = {json.loads(x)["key"] for x in out.read_text().splitlines()} if out.exists() else set()
    for f in flat[k::P]:
        r, t = int(f // (SEQ - 24)), int(f % (SEQ - 24)) + 24
        key = f"{r}:{t}"
        if key in done:
            continue
        s0 = max(0, t - md.W + 1)
        x = [int(v) for v in ids[r, s0:t + 1]]
        tt = len(x) - 1
        lpx = nat.reference([x], [tt])[0, 0]
        y = int(lpx.argmax())
        g = np.random.default_rng(int(f))
        res = {"key": key, "row": r, "t": t, "y": y, "p_y": float(lpx[y].exp()), "mechanism": "none confirmed"}
        occ = [j for j in range(tt) if x[j] == x[tt]]
        if occ and occ[-1] + 1 < tt and x[occ[-1] + 1] == y:
            a_ = occ[-1] + 1
            off, drop = md._effect(nat, x, tt, y, [[(a_, z)] for z in md._alts(nat, x, a_, md.DRAWS, g)])
            if off >= 0.5 and drop >= md.DROP:
                res["mechanism"] = "copy"
        elif not occ and y not in x[:tt]:
            if tt > md.FAR + 1:
                far = [int(p_) for p_ in g.choice(np.arange(1, tt - md.FAR + 1), size=min(16, tt - md.FAR), replace=False)]
                _, drops = md._each(nat, x, tt, y, [[(p_, md._alts(nat, x, p_, 2, g)[0])] for p_ in far])
                m = far[int(np.argmax(drops))]
                off, drop = md._effect(nat, x, tt, y, [[(m, z)] for z in md._alts(nat, x, m, md.DRAWS, g)])
                if off >= 0.5 and drop >= md.DROP:
                    res["mechanism"] = "far context"
            if res["mechanism"] == "none confirmed":
                off_t, _ = md._effect(nat, x, tt, y, [[(tt, z)] for z in md._alts(nat, x, tt, md.DRAWS, g)])
                ctx = [int(p_) for p_ in g.choice(np.arange(1, tt), size=min(md.DRAWS, tt - 1), replace=False)]
                off_c, drop_c = md._effect(nat, x, tt, y, [[(p_, md._alts(nat, x, p_, 2, g)[0])] for p_ in ctx])
                if off_t >= 0.5 and drop_c < md.QUIET and off_c == 0.0:
                    res["mechanism"] = "local bigram"
        back = list(range(max(1, tt - DEP_BACK + 1), tt + 1))  # (the first position has no prediction to draw from)
        _, drops = md._each(nat, x, tt, y, [[(p_, md._alts(nat, x, p_, 2, g)[0])] for p_ in back])
        res["dependencies"] = int((drops >= 1.0).sum())
        res["depends_back"] = [int(tt - p_) for p_, d_ in zip(back, drops) if d_ >= 1.0]
        with out.open("a") as fh:
            fh.write(json.dumps(res) + "\n")
    print("mech done", k, flush=True)


def attention():
    """Per token and layer, the attention (mean over heads) on the row's first token, on the previous token and on
    tokens 8 or more back other than the first -> INP/attention.npy [rows, SEQ, layers, 3]."""
    import torch

    import native

    nat = native.Native()
    ids = np.load(INP / "corpus.npy")[:, :SEQ]
    L = len({n.split(".")[1] for n in nat.names})
    out = np.zeros((len(ids), SEQ, L, 3), np.float16)
    T = torch.arange(SEQ, device=nat.dev)
    far = ((T[None, :] <= T[:, None] - 8) & (T[None, :] > 0)).float()  # (the first token apart)
    for i in range(0, len(ids), 8):
        x = torch.tensor(ids[i:i + 8], device=nat.dev, dtype=torch.long)
        with torch.no_grad():
            nat.vpd.clear()
            for l_ in range(L):
                for kd in ("q_proj", "k_proj"):
                    nat.target.site(native.site_name(l_, kd)).cache_output = True
            nat.target(x)
            for l_ in range(L):
                pat = nat.target.attention_pattern(nat.target.site(native.site_name(l_, "q_proj")).last_output,
                                                   nat.target.site(native.site_name(l_, "k_proj")).last_output).float().mean(1)  # [B, T, T]
                first = pat[:, :, 0]
                prev = torch.cat([torch.zeros_like(pat[:, :1, 0]), pat.diagonal(offset=-1, dim1=1, dim2=2)], 1)  # pat[:, t, t - 1]
                fb = (pat * far).sum(-1)
                out[i:i + 8, :, l_] = torch.stack([first, prev, fb], -1).cpu().numpy()
            for l_ in range(L):
                for kd in ("q_proj", "k_proj"):
                    nat.target.site(native.site_name(l_, kd)).cache_output = False
            nat.vpd.clear()
    np.save(INP / "attention.npy", out)
    print("wrote", INP / "attention.npy")


REPLICATE = (ROWS, 200)  # the causal follow-up's rows: the next 200 of the shard, none of them in the scan
TOP_BEHAVIOUR, TOP_TEXT, TEXT_PHENOS, NULL_POOL, EDITS = 5, 2, 30, 150, 48


class _Runner:
    """The model on the replication rows: behaviour phenotypes per token, gated genotypes of chosen subcomponents."""

    def __init__(self):
        import torch

        import native
        import vpd_model

        self.torch, self.vm = torch, vpd_model
        self.nat = native.Native()
        self.off = site_offsets(self.nat)
        self.site_of = []
        for n in self.nat.names:
            self.site_of += [(n, c) for c in range(self.nat.target.site(n).V.shape[1])]
        prefs = [json.loads(x) for x in PREFS.read_text().splitlines()] if PREFS.exists() else []
        self.prefs = [(p["a"], p["b"]) for p in prefs]
        self.klass = {}

    def behaviour(self, ids: np.ndarray, B: int = 16) -> dict:
        """Per token of rows ids [R, SEQ + 1]: entropy, log p(next), top token == next, top copies, top token's class
        index, log p(b) per preference, and attention [R, SEQ, layers, 3] as attention() measures it."""
        import native

        torch = self.torch
        L = len({n.split(".")[1] for n in self.nat.names})
        T = torch.arange(SEQ, device=self.nat.dev)
        far = ((T[None, :] <= T[:, None] - 8) & (T[None, :] > 0)).float()
        out = {k: [] for k in ("entropy", "logp_next", "next", "top1", "logp_b", "attention")}
        for i in range(0, len(ids), B):
            x = torch.tensor(ids[i:i + B, :SEQ], device=self.nat.dev)
            nx = torch.tensor(ids[i:i + B, 1:SEQ + 1], device=self.nat.dev)
            with torch.no_grad():
                self.nat.vpd.clear()
                for l_ in range(L):
                    for kd in ("q_proj", "k_proj"):
                        self.nat.target.site(native.site_name(l_, kd)).cache_output = True
                lp = torch.log_softmax(self.nat.target(x).float(), -1)
                att = []
                for l_ in range(L):
                    pat = self.nat.target.attention_pattern(self.nat.target.site(native.site_name(l_, "q_proj")).last_output,
                                                            self.nat.target.site(native.site_name(l_, "k_proj")).last_output).float().mean(1)
                    prev = torch.cat([torch.zeros_like(pat[:, :1, 0]), pat.diagonal(offset=-1, dim1=1, dim2=2)], 1)
                    att.append(torch.stack([pat[:, :, 0], prev, (pat * far).sum(-1)], -1))
                    for kd in ("q_proj", "k_proj"):
                        self.nat.target.site(native.site_name(l_, kd)).cache_output = False
                self.nat.vpd.clear()
            out["attention"].append(torch.stack(att, 2).cpu().numpy())
            out["entropy"].append((-(lp.exp() * lp).sum(-1)).cpu().numpy())
            out["logp_next"].append(lp.gather(-1, nx[..., None])[..., 0].cpu().numpy())
            top = lp.argmax(-1).cpu().numpy()
            out["top1"].append(top)
            out["next"].append(top == ids[i:i + B, 1:SEQ + 1])
            out["logp_b"].append(lp[..., [b for _, b in self.prefs]].cpu().numpy())
        res = {k: np.concatenate(v) for k, v in out.items()}
        cur = ids[:, :SEQ]
        copy = np.zeros(cur.shape, bool)
        for r in range(len(cur)):
            last = {}
            for t in range(SEQ):
                o = last.get(int(cur[r, t]))
                if o is not None:
                    copy[r, t] = res["top1"][r, t] == cur[r, o + 1]
                last[int(cur[r, t])] = t
        res["copy"] = copy
        for i in np.unique(res["top1"]):
            self.klass.setdefault(int(i), KLASS.index(_klass(_decode(int(i)))))
        res["klass"] = np.vectorize(self.klass.get)(res["top1"])
        return res

    def genotypes(self, ids: np.ndarray, subs: list, B: int = 4) -> np.ndarray:
        """Gated genotypes [R, SEQ, len(subs)] of the chosen subcomponents (global indices)."""
        torch = self.torch
        by_site = {}
        for j, c in enumerate(subs):
            n, k = self.site_of[c]
            by_site.setdefault(n, []).append((j, k))
        out = np.zeros((len(ids), SEQ, len(subs)), np.float32)
        for i in range(0, len(ids), B):
            x = torch.tensor(ids[i:i + B, :SEQ], device=self.nat.dev)
            with torch.no_grad():
                self.nat.vpd.clear()
                for n in self.nat.names:
                    self.nat.target.site(n).cache_input = True
                self.nat.target(x)
                acts = {n: self.nat.target.site(n).last_input for n in self.nat.names}
                self.nat.vpd.clear()
                ci = {n: self.vm.lower_leaky(v) for n, v in self.nat.vpd.ci_fn(acts).items()}
                for n, jk in by_site.items():
                    st = self.nat.target.site(n)
                    ks = torch.tensor([k for _, k in jk], device=self.nat.dev)
                    g = ci[n][..., ks] * ((acts[n] @ st.V[:, ks]) * st.U[ks].norm(dim=1))
                    out[i:i + B, :, [j for j, _ in jk]] = g.cpu().numpy()
        return out

    def full_at(self, ids: np.ndarray, ts: list, B: int = 4) -> np.ndarray:
        """Every subcomponent's gated genotype [len(ids), N_SUB] at position ts[i] of row i."""
        torch = self.torch
        out = np.zeros((len(ids), N_SUB), np.float32)
        for i in range(0, len(ids), B):
            x = torch.tensor(ids[i:i + B, :SEQ], device=self.nat.dev)
            tt = torch.tensor(ts[i:i + B], device=self.nat.dev)
            ar = torch.arange(len(tt), device=self.nat.dev)
            with torch.no_grad():
                self.nat.vpd.clear()
                for n in self.nat.names:
                    self.nat.target.site(n).cache_input = True
                self.nat.target(x)
                acts = {n: self.nat.target.site(n).last_input for n in self.nat.names}
                self.nat.vpd.clear()
                ci = {n: self.vm.lower_leaky(v) for n, v in self.nat.vpd.ci_fn(acts).items()}
                g = torch.cat([ci[n][ar, tt] * ((acts[n][ar, tt] @ self.nat.target.site(n).V) * self.nat.target.site(n).U.norm(dim=1))
                               for n in self.nat.names], dim=-1)
                out[i:i + B] = g.cpu().numpy()
        return out

    def without(self, c: int):
        """A context in which subcomponent c is removed from its matrix at every position (W - u_c v_c^T)."""
        import contextlib

        n, k = self.site_of[c]
        st = self.nat.target.site(n)

        @contextlib.contextmanager
        def cm():
            W = st.W.clone()
            st.W -= torch_outer(st.U[k], st.V[:, k])
            try:
                yield
            finally:
                st.W.copy_(W)

        torch_outer = self.torch.outer
        return cm()


def _bounds_ci(d: np.ndarray, B: int = 2000, seed: int = 2951) -> tuple:
    r = np.random.default_rng(seed)
    bs = d[r.integers(len(d), size=(B, len(d)))].mean(1)
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


MODEL_FAMS = ("model behaviour", "attention", "measured mechanism", "training-data frequency", "preferences")
CONTEXT_FAMS = ("text structure", "syntax and discourse", "document domain", "programming language", "position")
TOP_HITS, PER_PHENO = 200, 5


def _route(family: str) -> str:
    """How a hit is followed up: removal (the phenotype is the model's), or an input edit at what carries the feature:
    the current token, the next token, the earlier occurrence, or the 16 tokens of context before it."""
    if family.startswith(MODEL_FAMS):
        return "removal"
    if family.startswith("next token"):
        return "next"
    if family == "repeats":
        return "earlier"
    if family.startswith(CONTEXT_FAMS):
        return "context"
    return "current"


def causal_plan():
    """The causal follow-up's plan, on the Mac (it reads the scan): the TOP_HITS strongest hits (q < 0.05, by |partial
    r|, at most PER_PHENO per phenotype), each routed (_route); for removals the NULL_POOL random subcomponents; for
    input edits up to EDITS of the scan's positions where the subcomponent is active and the phenotype holds (graded:
    its upper quarter, or lower for a negative r), each with the positions to change -> OUT/causal_plan.json (a pod
    runs it with corpus.npy and replicate.npy alone)."""
    A = np.load(OUT / "assoc.npz")
    names, fams = json.loads(str(A["names"])), json.loads(str(A["families"]))
    z, q, r = A["z"], A["q"], A["r"]
    n_act = A["n_active"]
    rng = np.random.default_rng(2951)
    cand = np.argwhere(q < 0.05)
    cand = cand[np.argsort(-np.abs(r[cand[:, 0], cand[:, 1]]))]
    per, hits = {}, []
    for c, j in cand:
        if per.get(j, 0) < PER_PHENO:
            per[j] = per.get(j, 0) + 1
            hits.append((int(c), int(j)))
        if len(hits) >= TOP_HITS:
            break
    G, _ = genotypes()
    Gc = G.tocsc()
    d = np.load(OUT / "pheno.npz")
    where = {}
    for _, _, names_, Y, _, sel in _groups(d):
        for i, n_ in enumerate(names_):
            where[n_] = (Y, i, sel)
    ids = np.load(INP / "corpus.npy").astype(np.int64)
    cur_all = ids[:, :SEQ]
    plan = {"null_pool": [int(c) for c in rng.choice(np.flatnonzero(n_act >= 200), NULL_POOL, replace=False)], "hits": []}
    for c, j in hits:
        how = _route(fams[j])
        h = {"subcomponent": c, "phenotype": names[j], "family": fams[j], "r": float(r[c, j]), "test": "removal" if how == "removal" else f"edit: {how}"}
        if how == "removal" or names[j] not in where:
            plan["hits"].append(h)
            continue
        Y, i, sel = where[names[j]]
        y = np.zeros(len(cur_all) * SEQ, np.float32)
        defined = np.zeros(len(y), bool)
        if sel is None:
            y[:], defined[:] = Y[:, i], True
        else:
            y[sel], defined[sel] = Y[:, i], True
        g = np.asarray(Gc[:, c].todense()).ravel()
        if set(np.unique(Y[:, i])) <= {0.0, 1.0}:
            holds = y == (1.0 if r[c, j] > 0 else 0.0)
        else:
            lo, hi = np.percentile(Y[:, i], [25, 75])
            holds = (y >= hi) if r[c, j] > 0 else (y <= lo)
        pos = np.flatnonzero((g != 0) & holds & defined)
        pos = pos[(pos % SEQ) >= 17]
        if how == "next":
            pos = pos[(pos % SEQ) < SEQ - 1]
        pos = pos[rng.permutation(len(pos))]
        edits = []
        for f in pos:
            rr, t = divmod(int(f), SEQ)
            if how == "earlier":
                occ = np.flatnonzero(cur_all[rr, :t] == cur_all[rr, t])
                if not len(occ):
                    continue
                at = [int(occ[-1])]
            else:
                at = {"current": [t], "next": [t + 1], "context": list(range(max(1, t - 16), t))}[how]
            edits.append({"row": rr, "t": t, "at": at})
            if len(edits) >= EDITS:
                break
        h["edits"] = edits
        plan["hits"].append(h)
    (OUT / "causal_plan.json").write_text(json.dumps(plan))
    print("wrote", OUT / "causal_plan.json", {k: sum(h["test"] == k for h in plan["hits"]) for k in {h["test"] for h in plan["hits"]}})


def causal():
    """The causal follow-up (a pod; OUT/causal_plan.json, corpus.npy, replicate.npy):
      the model's phenotypes (behaviour, attention, preferences): the subcomponent removed from its matrix at every
        position (W - u v^T) on rows the scan never saw (REPLICATE); the phenotype's change at the tokens where it is
        active against the same change for the null pool's random subcomponents at their own active tokens; causal
        when beyond the null's 95th percentile in size (the measured mechanism and the Pile n-gram phenotypes are not
        recomputed under removal and are left out);
      text phenotypes, the other direction (the text causes the activation): at the planned positions, the carrier of
        the feature changed in the input (a frequent other token at each planned position), and the share of positions
        where the subcomponent's gated genotype falls by more than half against the same share over every other
        subcomponent active there; causal when higher by a one-sided binomial test at 0.05. A next-token edit can move
        a gated genotype only through VPD's gate, which reads the next token (leak).
    -> GEN_OUT/causal.json"""
    from scipy.stats import binomtest

    plan = json.loads((OUT / "causal_plan.json").read_text())
    rng = np.random.default_rng(2951)
    run = _Runner()
    out = {"hits": []}
    rows = np.load(INP / "replicate.npy").astype(np.int64)
    rem = [h for h in plan["hits"] if h["test"] == "removal" and not h["family"].startswith(("measured mechanism", "training-data"))]
    pool = plan["null_pool"]
    subs = sorted({h["subcomponent"] for h in rem} | set(pool))
    base = run.behaviour(rows)
    on = run.genotypes(rows, subs) != 0
    col = {c: i for i, c in enumerate(subs)}
    cur = rows[:, :SEQ]

    def phen(res, nm):
        if nm.startswith("entropy"):
            return res["entropy"], None
        if nm.startswith("the model's loss"):
            return -res["logp_next"], None
        if nm == "the model's top token is the next token":
            return res["next"].astype(np.float32), None
        if nm.startswith("the model's top token copies"):
            return res["copy"].astype(np.float32), None
        if nm.startswith("the model's top token is ") and nm.split()[-1] in KLASS:
            return (res["klass"] == KLASS.index(nm.split()[-1])).astype(np.float32), None
        if nm == "the model's top token closes a bracket":
            return np.vectorize(lambda i: bool(_decode(int(i)).strip()) and _decode(int(i)).strip()[0] in ")]}>")(res["top1"]).astype(np.float32), None
        if nm == "the model's top token is a number":
            return np.vectorize(lambda i: _decode(int(i)).strip().isdigit())(res["top1"]).astype(np.float32), None
        if nm.startswith("layer ") and " attention on " in nm:
            l_ = int(nm.split()[1])
            j_ = ["the row's first token", "the previous token", "tokens 8 or more back (not the first)"].index(nm.split(" attention on ")[1])
            return res["attention"][:, :, l_, j_], None
        if nm.startswith("after "):
            k = [f"after {_decode(a_)!r}, log p({_decode(b_)!r})" for a_, b_ in run.prefs].index(nm)
            return res["logp_b"][..., k], cur == run.prefs[k][0]
        return None, None

    phs = sorted({h["phenotype"] for h in rem})
    delta = {}
    for i, c in enumerate(subs):
        with run.without(c):
            res = run.behaviour(rows)
        m = on[..., col[c]]
        delta[c] = {}
        for nm in phs:
            y0, mask = phen(base, nm)
            if y0 is None:
                continue
            y1, _ = phen(res, nm)
            mm = m if mask is None else (m & mask)
            delta[c][nm] = float((y1[mm] - y0[mm]).mean()) if mm.sum() >= 5 else None
        if i % 25 == 0:
            print(f"removals {i + 1}/{len(subs)}", flush=True)
    for h in rem:
        c, nm = h["subcomponent"], h["phenotype"]
        nul = np.abs([delta[p_].get(nm) for p_ in pool if delta[p_].get(nm) is not None])
        dl = delta[c].get(nm)
        thr = float(np.percentile(nul, 95)) if len(nul) else None
        out["hits"].append({**h, "delta": dl, "null95": thr, "causal": bool(dl is not None and thr is not None and abs(dl) > thr),
                            "pushes": bool(dl is not None and np.sign(dl) == -np.sign(h["r"]))})
    print("removal hits causal:", sum(h["causal"] for h in out["hits"]), "/", len(out["hits"]), flush=True)
    ids = np.load(INP / "corpus.npy").astype(np.int64)
    common = np.argsort(-np.bincount(ids[:, :SEQ].reshape(-1)))[:50]
    for h in plan["hits"]:
        if not h["test"].startswith("edit") or len(h.get("edits", [])) < 8:
            continue
        c = h["subcomponent"]
        base_rows, edited, ts = [], [], []
        for e in h["edits"]:
            x = ids[e["row"]].copy()
            for p_ in e["at"]:
                x[p_] = int(rng.choice([v for v in common if v != x[p_]]))
            base_rows.append(ids[e["row"]]), edited.append(x), ts.append(e["t"])
        gb = run.full_at(np.stack(base_rows), ts)
        ge = run.full_at(np.stack(edited), ts)
        fall_c, fall_o = [], []
        for k in range(len(ts)):
            if gb[k, c] == 0:
                continue
            fall_c.append(abs(ge[k, c]) < 0.5 * abs(gb[k, c]))
            act = np.flatnonzero(gb[k] != 0)
            act = act[act != c]
            fall_o.append(float(np.mean(np.abs(ge[k, act]) < 0.5 * np.abs(gb[k, act]))))
        if not fall_c:
            continue
        rate = float(np.mean(fall_o))
        pv = binomtest(int(np.sum(fall_c)), len(fall_c), min(max(rate, 1e-6), 1 - 1e-6), alternative="greater").pvalue
        out["hits"].append({**{k: v for k, v in h.items() if k != "edits"}, "positions": len(fall_c), "falls_by_half": float(np.mean(fall_c)),
                            "coactive_falls_by_half": rate, "p": float(pv), "causal": bool(pv < 0.05)})
    by = {}
    for h in out["hits"]:
        by.setdefault(h["test"], []).append(h["causal"])
    out["summary"] = {k: {"hits": len(v), "causal": int(sum(v))} for k, v in by.items()}
    out["not_followed_up"] = [h["phenotype"] for h in plan["hits"] if h["family"].startswith(("measured mechanism", "training-data"))]
    dst = OUT / "causal.json"
    dst.write_text(json.dumps(out, indent=1))
    print("wrote", dst, out["summary"])


TEACHER = Path.home() / "mpd-data/graph_oracle/teacher"
KINDS6 = ["q_proj", "k_proj", "v_proj", "o_proj", "c_fc", "down_proj"]
STAGES = ["site", "chunk16", "chunk4", "subcomponent"]


def _probe_rows() -> "pd.DataFrame":
    """fleet's removability probe table (probe_features.py: per start subcomponent of the v2.3 teacher circuits whose
    path saved its stage nodes, the structural features and whether the final circuit keeps it), with each row's
    subcomponent and the stage that removed it."""
    import pandas as pd
    import torch

    vpd = torch.load(Path.home() / "mpd-data/graph_oracle/teacher/vpd_v2_494.pt", weights_only=False, map_location="cpu")
    rows = []
    for d in (TEACHER / "v2_train", TEACHER / "v2_heldout"):
        for p in sorted(d.glob("text*.path.json")):
            t = p.name[: -len(".path.json")]
            if t not in vpd or not (d / f"{t}.json").exists():
                continue
            b = json.loads(p.read_text()).get("blocks") or {}
            sn = b.get("stage_nodes") or {}
            if not sn.get("start"):
                continue
            final = {(n, int(q_), int(c)) for n, c, q_ in json.loads((d / f"{t}.json").read_text())["nodes"] if n not in ("tok", "block")}
            start = [(n, int(q_), int(c)) for n, q_, c in sn["start"]]
            alive = [{(n, int(q_), int(c)) for n, q_, c in sn.get(st, sn["start"])} for st in STAGES]
            target = max(json.loads((TEXTS / f"{t}.json").read_text())["prompts"][0]["target_positions"])
            order = [(n, int(q_), int(c)) for n, q_, c in vpd[t]["order"]]
            rank = {nd: i for i, nd in enumerate(order)}
            by_pos, by_ps = {}, {}
            for nd in order:
                by_pos.setdefault(nd[1], []).append(nd)
                by_ps.setdefault((nd[1], nd[0]), []).append(nd)
            rpos = {nd: i for lst in by_pos.values() for i, nd in enumerate(lst)}
            rps = {nd: i for lst in by_ps.values() for i, nd in enumerate(lst)}
            prank = {q_: i for i, q_ in enumerate(sorted(by_pos, key=lambda q_: min(rank[nd] for nd in by_pos[q_])))}
            cnt_pos, cnt_ps = {}, {}
            for nd in start:
                cnt_pos[nd[1]] = cnt_pos.get(nd[1], 0) + 1
                cnt_ps[(nd[1], nd[0])] = cnt_ps.get((nd[1], nd[0]), 0) + 1
            for nd in start:
                n, pos_, c = nd
                kind, layer = n.split(".")[-1], int(n.split(".")[1])
                r_ = rank.get(nd, len(order))
                kept = nd in final
                rows.append({"text": t, "split": "dev" if t.startswith("text90") else "train", "label": int(kept), "site": n, "pos": pos_, "idx": c,
                             "stage": "" if kept else next((st for st, a_ in zip(STAGES, alive) if nd not in a_), "after"),
                             "start_size": len(start), "n_at_pos": cnt_pos[pos_], "n_at_pos_site": cnt_ps[(pos_, n)], "rank": r_,
                             "rank_pct": r_ / max(len(order), 1), "rank_in_pos": rpos.get(nd, len(by_pos.get(pos_, []))),
                             "rank_in_pos_pct": rpos.get(nd, 0) / max(len(by_pos.get(pos_, [1])), 1), "rank_in_pos_site": rps.get(nd, 0),
                             "layer": layer, "dist_to_target": target - pos_, "is_target": int(pos_ == target),
                             "pos_rank_by_vpd": prank.get(pos_, len(prank)), "n_positions": len(cnt_pos), **{f"kind_{k}": int(kind == k) for k in KINDS6}})
    return pd.DataFrame(rows)


def probe():
    """Decision test 1: does a start subcomponent's association profile tell whether the v2.3 teacher keeps it? Boosted
    trees on fleet's structural features, with and without each subcomponent's association features (its strongest
    |partial r| at q < 0.05 overall and per phenotype family, its strong-hit count, how often it is active); within-text
    AUROC on the dev texts (fit on train) and in 5-fold cross-validation over train texts grouped by text, overall and
    per removal stage (each stage's removals against the kept) -> OUT/probe.json."""
    import pandas as pd
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold

    df = _probe_rows()
    A = np.load(OUT / "assoc.npz")
    r, q = A["r"], A["q"]
    fams = json.loads(str(A["families"]))
    off = json.loads(str(A["offsets"]))
    sig = np.where(q < 0.05, np.abs(r), 0.0)
    sig = np.nan_to_num(sig)
    feats = {"assoc max |r|": sig.max(1), "assoc strong hits": (sig >= 0.1).sum(1), "assoc active tokens": np.log1p(A["n_active"])}
    for f in sorted(set(fams)):
        cols = [j for j, x in enumerate(fams) if x == f]
        feats[f"assoc {f}"] = sig[:, cols].max(1)
    gi = np.array([off[s_] + i for s_, i in zip(df.site, df.idx)])
    for k, v in feats.items():
        df[k] = v[gi]
    base = [c for c in df.columns if c not in ("text", "split", "label", "site", "pos", "idx", "stage") and not c.startswith("assoc")]
    plus = base + [c for c in df.columns if c.startswith("assoc")]

    def within(sub, s_, mask=None):
        out = []
        for _, g in sub.groupby("text"):
            g = g if mask is None else g[mask(g)]
            if g.label.nunique() == 2:
                out.append(roc_auc_score(g.label, s_[g.index]))
        return float(np.mean(out)) if out else None, len(out)

    res = {"rows": len(df), "texts": int(df.text.nunique())}
    tr, dv = df[df.split == "train"], df[df.split == "dev"]
    for name, fs in (("structure", base), ("structure + associations", plus)):
        oof = pd.Series(np.nan, index=tr.index)
        for i, j in GroupKFold(5).split(tr, tr.label, tr.text):
            m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05).fit(tr.iloc[i][fs], tr.iloc[i].label)
            oof.iloc[j] = m.predict_proba(tr.iloc[j][fs])[:, 1]
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05).fit(tr[fs], tr.label)
        sd = pd.Series(m.predict_proba(dv[fs])[:, 1], index=dv.index)
        res[name] = {"cv within-text AUROC": within(tr, oof), "dev within-text AUROC": within(dv, sd),
                     "cv by stage": {st: within(tr, oof, lambda g, st=st: (g.label == 1) | (g.stage == st)) for st in STAGES}}
        print(name, json.dumps(res[name]), flush=True)
    (OUT / "probe.json").write_text(json.dumps(res, indent=1))
    print("wrote", OUT / "probe.json")


FIG = Path("/Users/user/mpd-data/figures/oracle")


def figure(r_min: float = 0.1):
    """The share of each weight matrix's subcomponents associated with each phenotype family (q < 0.05 with
    |partial r| >= r_min), families ordered by how many subcomponents they reach -> FIG/gwas_families_by_matrix.png."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import figure as fg

    A = np.load(OUT / "assoc.npz")
    z, r, q = A["z"], A["r"], A["q"]
    fams = json.loads(str(A["families"]))
    off = json.loads(str(A["offsets"]))
    sites = sorted(off, key=off.get)
    bounds = [off[n] for n in sites] + [N_SUB]
    hit = (q < 0.05) & (np.abs(r) >= r_min)
    tested = np.isfinite(z).any(1)
    fl = sorted(set(fams))
    F = np.array([fl.index(f) for f in fams])
    M = np.zeros((len(fl), len(sites)))
    for fi in range(len(fl)):
        hf = hit[:, F == fi].any(1)
        for si in range(len(sites)):
            sl = slice(bounds[si], bounds[si + 1])
            M[fi, si] = hf[sl][tested[sl]].mean() if tested[sl].any() else np.nan
    order = np.argsort(-np.nanmean(M, 1))
    fg.setup()
    fig, ax = plt.subplots(figsize=(18, 0.42 * len(fl) + 3))
    im = ax.imshow(M[order], aspect="auto", cmap="Blues", vmin=0, vmax=max(0.05, float(np.nanpercentile(M, 99))))
    ax.set_yticks(range(len(fl)), [fl[i] for i in order], fontsize=13)
    ax.set_xticks(range(len(sites)), [f"L{n.split('.')[1]} {n.split('.')[-1].replace('_proj', '').replace('c_fc', 'fc')}" for n in sites],
                  rotation=60, ha="right", fontsize=12)
    cb = fig.colorbar(im, ax=ax, fraction=0.025)
    cb.set_label(f"share of the matrix's subcomponents\nwith |partial r| >= {r_min} at q < 0.05", fontsize=13)
    ax.set_title("Which weight matrices hold subcomponents associated with each phenotype family")
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / "gwas_families_by_matrix.png"
    fig.savefig(out, dpi=150)
    print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("extract", "pos", "pos_tag", "annotate", "domain", "ngram", "export", "mech", "attention", "pheno", "assoc",
                                    "calib", "table", "figure", "conditional", "causal_plan", "causal", "probe"))
    ap.add_argument("args", nargs="*")
    a = ap.parse_args()
    if a.cmd in ("extract", "mech"):
        globals()[a.cmd](*(int(x) for x in a.args))
    else:
        globals()[a.cmd]()
