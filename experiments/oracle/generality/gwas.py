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
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "method" / "oracle"))
sys.path.insert(0, str(ROOT / "method" / "vpd"))

OUT = Path(os.environ.get("GW_OUT", Path.home() / "mpd-data/graph_oracle/gwas"))
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
    (OUT/geno_K_C.npz; a file already there is kept, so a stopped run resumes)."""
    import torch

    import native
    import vpd_model

    nat = native.Native()
    off = site_offsets(nat)
    prefs = [json.loads(x) for x in PREFS.read_text().splitlines()] if PREFS.exists() else []
    bs = torch.tensor([p["b"] for p in prefs], dtype=torch.long, device=nat.dev)
    rows = corpus()
    mine = list(range(k, len(rows), P))
    OUT.mkdir(parents=True, exist_ok=True)
    for c0 in range(0, len(mine), CHUNK):
        path = OUT / f"geno_{k}_{c0 // CHUNK:04d}.npz"
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
    return sorted((p for p in OUT.iterdir() if p.name.startswith("geno_") and p.suffix == ".npz"),
                  key=lambda p: tuple(int(x) for x in p.stem.split("_")[1:]))


_TOK = []


def _decode(i: int) -> str:
    """A token's text alone (the tokenizer's decode of one id)."""
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
    (OUT / "tokstr.json").write_text(json.dumps([["\n\n" if i == EOS else t for i, t in zip(r, s_)] for r, s_ in zip(ids.tolist(), strs)]))
    subprocess.run([SPACY_PY, "-B", str(Path(__file__).resolve()), "pos_tag"], check=True, env={**os.environ, "GW_OUT": str(OUT)})


def pos_tag():
    """(in spaCy's environment) OUT/tokstr.json -> OUT/pos.npy, UPOS indices [rows, SEQ]."""
    import spacy

    nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
    rows = json.loads((OUT / "tokstr.json").read_text())
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
    np.save(OUT / "pos.npy", out)
    print("wrote", OUT / "pos.npy")


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


def pheno():
    """The phenotype matrix [tokens, P] (float32), names, per-phenotype masks for the masked ones (the preferences),
    the covariates and each token's row (the clusters)."""
    gs = [np.load(p) for p in _geno()]
    ids = np.concatenate([g["ids"] for g in gs])  # [R, SEQ + 1]
    R = len(ids)
    cur, nxt = ids[:, :SEQ], ids[:, 1:SEQ + 1]
    top1 = np.concatenate([g["top1"] for g in gs])
    ent = np.concatenate([g["entropy"] for g in gs])
    lpb = np.concatenate([g["logp_b"] for g in gs]).astype(np.float32)
    prefs = json.loads(str(gs[0]["prefs"]))
    tags = np.load(OUT / "pos.npy")
    strs = _strings(ids)
    klass = {int(i): _klass(_decode(i)) for i in np.unique(top1)}
    cols, names = [], []

    def add(name, v):
        names.append(name)
        cols.append(np.asarray(v, dtype=np.float32).reshape(-1))

    freq = np.bincount(cur.reshape(-1), minlength=int(ids.max()) + 1)
    for tk in np.argsort(-freq)[:N_TOK]:
        add(f"current token {_decode(tk)!r}", cur == tk)
    fn = np.bincount(nxt.reshape(-1), minlength=int(ids.max()) + 1)
    for tk in np.argsort(-fn)[:N_TOK]:
        add(f"next token {_decode(tk)!r}", nxt == tk)
    for j, tag in enumerate(UPOS):
        add(f"part of speech {tag}", tags == j)
    quote, bracket, rep, dist = (np.zeros((R, SEQ), np.float32) for _ in range(4))
    copy = np.zeros((R, SEQ), np.float32)
    for r in range(R):
        q = b = 0
        last = {}
        for t in range(SEQ):
            piece = strs[r][t]
            if cur[r, t] == EOS:
                q = b = 0
            quote[r, t] = q % 2 == 1
            bracket[r, t] = b > 0
            for ch in piece:
                if ch in "\"“”":
                    q += 1
                elif ch in "([{":
                    b += 1
                elif ch in ")]}":
                    b = max(b - 1, 0)
            o = last.get(int(cur[r, t]))
            if o is not None:
                rep[r, t] = 1
                dist[r, t] = t - o
                copy[r, t] = top1[r, t] == cur[r, o + 1]
            last[int(cur[r, t])] = t
    add("inside quotes", quote)
    add("inside brackets", bracket)
    add("current token occurred earlier in the row", rep)
    for lo, hi in ((1, 8), (9, 64), (65, SEQ)):
        add(f"its latest earlier occurrence {lo}-{hi} tokens back", (dist >= lo) & (dist <= hi))
    add("entropy of the model's prediction", ent)
    add("the model's top token is the next token", top1 == nxt)
    add("the model's top token copies (the token after the current token's latest earlier occurrence)", copy)
    for k in KLASS:
        add(f"the model's top token is {k}", np.isin(top1, [i for i, c in klass.items() if c == k]))
    P = len(names)
    Y = np.stack(cols, 1)
    pos_ = np.tile(np.arange(SEQ, dtype=np.float32), R)
    cov = np.stack([np.ones(R * SEQ, np.float32), pos_, np.log1p(pos_), np.log(freq[cur.reshape(-1)].astype(np.float32))], 1)
    masked = {f"after {_decode(a)!r}, log p({_decode(b)!r})": ((cur == a).reshape(-1), lpb[:, :, j].reshape(-1)) for j, (a, b) in enumerate(prefs)}
    np.savez(OUT / "pheno.npz", Y=Y, names=json.dumps(names), cov=cov, row=np.repeat(np.arange(R), SEQ),
             position=pos_, masked_names=json.dumps(list(masked)), masked_mask=np.stack([m for m, _ in masked.values()], 1),
             masked_y=np.stack([y for _, y in masked.values()], 1), prefs=json.dumps(prefs))
    print(f"wrote {OUT / 'pheno.npz'}: {R * SEQ} tokens, {P} phenotypes + position + {len(masked)} preferences")


MIN_ACTIVE = 20  # tokens a subcomponent is active at, at least, to be tested
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
    cluster-robust over rows: V = sum over rows c of (G_res,c^T Y_res,c)^2 with the CR1 factor; z = U / sqrt(V), the
    partial correlation U / sqrt(|G_res|^2 |Y_res|^2). row: each token's cluster, contiguous."""
    import scipy.linalg

    Q, _ = scipy.linalg.qr(C.astype(np.float64), mode="economic")  # (scipy: numpy's linalg on this Mac's Accelerate misbehaves)
    Yr = (Y - Q @ (Q.T @ Y)).astype(np.float32)
    U = np.asarray(G.T @ Yr, dtype=np.float64)
    H = np.asarray(G.T @ Q).T  # [k, S]: Q^T G
    gg = np.asarray(G.multiply(G).sum(0)).ravel() - (H ** 2).sum(0)
    yy = (Yr.astype(np.float64) ** 2).sum(0)
    V = np.zeros_like(U)
    cut = np.flatnonzero(np.diff(row)) + 1
    bounds = np.concatenate([[0], cut, [len(row)]])
    Ht = H.T.astype(np.float32)
    for i in range(len(bounds) - 1):
        sl = slice(bounds[i], bounds[i + 1])
        with np.errstate(all="ignore"):  # Accelerate's sgemm raises spurious floating-point flags
            A = -(Ht @ (Q[sl].T.astype(np.float32) @ Yr[sl]))
        A += np.asarray(G[sl].T @ Yr[sl])
        V += A.astype(np.float64) ** 2
    nc = len(bounds) - 1
    V *= nc / max(nc - 1, 1)
    n_act = np.diff(G.tocsc().indptr)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = U / np.sqrt(V)
        r = U / np.sqrt(np.outer(gg, yy))
    tested = n_act >= MIN_ACTIVE
    z[~tested] = np.nan
    r[~tested] = np.nan
    return {"z": z.astype(np.float32), "r": r.astype(np.float32), "n_active": n_act}


def assoc():
    """Every (subcomponent, phenotype) score test, BH over every tested pair -> OUT/assoc.npz."""
    import time
    from math import sqrt

    from scipy.special import erfc

    t0 = time.time()
    G, off = genotypes()
    d = np.load(OUT / "pheno.npz")
    names = json.loads(str(d["names"]))
    row = d["row"]
    print(f"genotypes {G.shape}, {G.nnz} active, {time.time() - t0:.0f} s", flush=True)
    res = score_tests(G, d["Y"], d["cov"], row)
    zs, rs, nm = [res["z"]], [res["r"]], list(names)
    print(f"main group {len(names)} phenotypes, {time.time() - t0:.0f} s", flush=True)
    cov = d["cov"]
    p2 = score_tests(G, d["position"][:, None], cov[:, [0, 3]], row)  # position: intercept and log frequency only
    zs.append(p2["z"])
    rs.append(p2["r"])
    nm.append("position in the row")
    for j, name in enumerate(json.loads(str(d["masked_names"]))):
        m = d["masked_mask"][:, j]
        if m.sum() < 50:
            continue
        pj = score_tests(G[np.flatnonzero(m)], d["masked_y"][m, j][:, None], cov[m], row[m])
        zs.append(pj["z"])
        rs.append(pj["r"])
        nm.append(name)
    z = np.concatenate(zs, 1)
    r = np.concatenate(rs, 1)
    p = erfc(np.abs(z.astype(np.float64)) / sqrt(2))
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
    np.savez(OUT / "assoc.npz", z=z, r=r, q=q.astype(np.float32), names=json.dumps(nm), n_active=res["n_active"], offsets=json.dumps(off))
    hits = (q < 0.05)
    print(f"wrote {OUT / 'assoc.npz'}: {ok.sum()} tests, {hits.sum()} at q < 0.05, {(hits & (np.abs(r) >= 0.1)).sum()} of them with "
          f"|partial r| >= 0.1, {time.time() - t0:.0f} s", flush=True)


def _robust_z(Gr: np.ndarray, yr: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """Score z per column of residualized dense genotypes Gr [N, K] against a residualized phenotype yr [N]: U = Gr^T yr,
    V = sum over rows of (per-row sums of Gr * yr)^2 (CR1)."""
    prod = Gr * yr[:, None]
    U = prod.sum(0, dtype=np.float64)
    per = np.add.reduceat(prod, bounds[:-1], axis=0).astype(np.float64)
    nc = len(bounds) - 1
    V = (per ** 2).sum(0) * nc / max(nc - 1, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return U / np.sqrt(V)


def conditional(top: int = 100, steps: int = 30):
    """The LD analog: per phenotype with hits, forward selection among its top marginal hits, each step adding the
    candidate with the largest conditional |z| given the covariates and the subcomponents already chosen, while its p
    stays under the scan's own BH threshold (the largest p with q < 0.05) -> OUT/conditional.json: per phenotype the
    marginal hits, the independent signals in order, and the median |genotype correlation| among the top hits."""
    import scipy.linalg
    from scipy.special import erfc

    G, off = genotypes()
    A = np.load(OUT / "assoc.npz")
    d = np.load(OUT / "pheno.npz")
    names = json.loads(str(A["names"]))
    z, q = A["z"], A["q"]
    pstar = float(np.nanmax(np.where(q < 0.05, erfc(np.abs(z.astype(np.float64)) / math.sqrt(2)), np.nan)))
    zstar = float(np.sqrt(2) * __import__("scipy.special", fromlist=["erfcinv"]).erfcinv(pstar))
    main = json.loads(str(d["names"]))
    row = d["row"]
    out = {}
    for j, name in enumerate(names):
        hits = np.flatnonzero(q[:, j] < 0.05)
        if len(hits) == 0:
            continue
        if name in main:
            y, C, keep = d["Y"][:, main.index(name)], d["cov"], slice(None)
        elif name == "position in the row":
            y, C, keep = d["position"], d["cov"][:, [0, 3]], slice(None)
        else:
            k = json.loads(str(d["masked_names"])).index(name)
            keep = np.flatnonzero(d["masked_mask"][:, k])
            y, C = d["masked_y"][keep, k], d["cov"][keep]
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
            zc = _robust_z(Gr, yr, bounds)
            zc[chosen] = 0.0
            b = int(np.nanargmax(np.abs(zc)))
            if not np.abs(zc[b]) >= zstar:
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
    (OUT / "conditional.json").write_text(json.dumps({"z_threshold": zstar, "phenotypes": out}, indent=1))
    print("wrote", OUT / "conditional.json")


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
        """Per token of rows ids [R, SEQ + 1]: entropy, top token == next, top copies, top token's class index, and
        log p(b) per preference."""
        torch = self.torch
        out = {k: [] for k in ("entropy", "next", "top1", "logp_b")}
        for i in range(0, len(ids), B):
            x = torch.tensor(ids[i:i + B, :SEQ], device=self.nat.dev)
            with torch.no_grad():
                self.nat.vpd.clear()
                lp = torch.log_softmax(self.nat.target(x).float(), -1)
            out["entropy"].append((-(lp.exp() * lp).sum(-1)).cpu().numpy())
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


def causal():
    """For the top hits, the causal follow-up on rows the scan never saw (REPLICATE):
      behaviour phenotypes (the model's entropy, top token == next, copying, top token's class, the preferences): the
        subcomponent removed from its matrix at every position; the change of the phenotype at the tokens where it is
        active, against the same change for NULL_POOL random subcomponents (each at its own active tokens); causal when
        the change is beyond the null's 95th percentile in size;
      text phenotypes, the other direction (the text causes the activation): at EDITS positions where the subcomponent
        is active and the feature holds, the feature changed in the input (current token: another frequent token at t;
        next token: another at t + 1; an earlier occurrence: another token there) and the share of positions where
        its gated genotype falls by more than half, against random subcomponents co-active at the same positions;
        changing the next token can move a gated genotype only through VPD's gate, which reads the next token (leak).
    -> OUT/causal.json"""
    A = np.load(OUT / "assoc.npz")
    names = json.loads(str(A["names"]))
    z, q, r = A["z"], A["q"], A["r"]
    n_act = A["n_active"]
    rows = corpus(*REPLICATE)
    run = _Runner()
    rng = np.random.default_rng(2951)

    def top_hits(j, k):
        h = np.flatnonzero(q[:, j] < 0.05)
        return [int(c) for c in h[np.argsort(-np.abs(z[h, j]))][:k]]

    beh = [j for j, nm in enumerate(names) if nm.startswith(("entropy", "the model's top token", "after "))]
    hits = {j: top_hits(j, TOP_BEHAVIOUR) for j in beh}
    pool = [int(c) for c in rng.choice(np.flatnonzero(n_act >= 200), NULL_POOL, replace=False)]
    subs = sorted({c for v in hits.values() for c in v} | set(pool))
    base = run.behaviour(rows)
    gen = run.genotypes(rows, subs)
    on = gen != 0
    col = {c: i for i, c in enumerate(subs)}
    cur = rows[:, :SEQ]

    def phen(res, nm):
        if nm.startswith("entropy"):
            return res["entropy"], None
        if nm == "the model's top token is the next token":
            return res["next"].astype(np.float32), None
        if nm.startswith("the model's top token copies"):
            return res["copy"].astype(np.float32), None
        if nm.startswith("the model's top token is "):
            return (res["klass"] == KLASS.index(nm.split()[-1])).astype(np.float32), None
        k = [f"after {_decode(a)!r}, log p({_decode(b)!r})" for a, b in run.prefs].index(nm)
        return res["logp_b"][..., k], cur == run.prefs[k][0]

    delta = {}
    for i, c in enumerate(subs):
        with run.without(c):
            res = run.behaviour(rows)
        m = on[..., col[c]]
        delta[c] = {}
        for j in beh:
            y0, mask = phen(base, names[j])
            y1, _ = phen(res, names[j])
            mm = m if mask is None else (m & mask)
            delta[c][j] = float((y1[mm] - y0[mm]).mean()) if mm.sum() >= 5 else None
        if i % 25 == 0:
            print(f"removals {i + 1}/{len(subs)}", flush=True)
    out = {"behaviour": [], "text": []}
    for j in beh:
        nul = np.abs([delta[c][j] for c in pool if delta[c][j] is not None])
        thr = float(np.percentile(nul, 95)) if len(nul) else None
        for c in hits[j]:
            dl = delta[c][j]
            out["behaviour"].append({"phenotype": names[j], "subcomponent": c, "z": float(z[c, j]), "r": float(r[c, j]), "delta": dl,
                                     "null95": thr, "causal": bool(dl is not None and thr is not None and abs(dl) > thr),
                                     "pushes": bool(dl is not None and np.sign(dl) == -np.sign(r[c, j]))})
    print("behaviour hits causal:", sum(h["causal"] for h in out["behaviour"]), "/", len(out["behaviour"]), flush=True)
    # text phenotypes: change the feature in the input, read the genotype
    for kind, prefix in (("current", "current token "), ("next", "next token "), ("earlier", "its latest earlier occurrence")):
        js = [j for j, nm in enumerate(names) if nm.startswith(prefix)][:TEXT_PHENOS]
        for j in js:
            for c in top_hits(j, TOP_TEXT):
                res = _edit_test(run, rows, names[j], kind, c, rng)
                if res:
                    out["text"].append({"phenotype": names[j], "kind": kind, "subcomponent": c, "z": float(z[c, j]), **res})
        print(kind, "done", flush=True)
    (OUT / "causal.json").write_text(json.dumps(out, indent=1))
    print("wrote", OUT / "causal.json")


def _edit_test(run, rows: np.ndarray, name: str, kind: str, c: int, rng) -> dict | None:
    """Positions where subcomponent c is active and the text feature holds; the feature changed in the input; the share
    of positions where c's gated genotype falls by more than half, and the same share over every other subcomponent
    active there (the edit's typical reach)."""
    cur, nxt = rows[:, :SEQ], rows[:, 1:SEQ + 1]
    freq = np.bincount(cur.reshape(-1), minlength=int(rows.max()) + 1)
    common = np.argsort(-freq)[:N_TOK]
    g0 = run.genotypes(rows, [c])[..., 0]
    if kind == "current" or kind == "next":
        tok = int([i for i in common if f"{kind} token {_decode(i)!r}" == name][0]) if any(f"{kind} token {_decode(i)!r}" == name for i in common) else None
        if tok is None:
            return None
        where = np.argwhere(((cur if kind == "current" else nxt) == tok) & (g0 != 0))
    else:
        where = []
        for rr in range(len(cur)):
            last = {}
            for t in range(SEQ):
                o = last.get(int(cur[rr, t]))
                lo, hi = (int(x) for x in name.split()[-4].split("-"))
                if o is not None and lo <= t - o <= hi and g0[rr, t] != 0:
                    where.append((rr, t, o))
                last[int(cur[rr, t])] = t
        where = np.asarray(where)
    if len(where) < 8:
        return None
    where = where[rng.permutation(len(where))[:EDITS]]
    edited, at = [], []
    for w in where:
        rr, t = int(w[0]), int(w[1])
        x = rows[rr].copy()
        p = t if kind == "current" else t + 1 if kind == "next" else int(w[2])
        choices = [i for i in common[:50] if i != x[p]]
        x[p] = int(rng.choice(choices))
        edited.append(x)
        at.append((len(edited) - 1, t, rr))
    edited = np.stack(edited)
    ts = [t for _, t, _ in at]
    gb = run.full_at(rows[[a_[2] for a_ in at]], ts)
    ge = run.full_at(edited, ts)
    drop_c, drop_o = [], []
    for i in range(len(at)):
        b0, b1 = gb[i], ge[i]
        if b0[c] == 0:
            continue
        drop_c.append(abs(b1[c]) < 0.5 * abs(b0[c]))
        act = np.flatnonzero(b0 != 0)
        act = act[act != c]
        drop_o.append(float(np.mean(np.abs(b1[act]) < 0.5 * np.abs(b0[act]))))
    if not drop_c:
        return None
    return {"positions": len(drop_c), "falls_by_half": float(np.mean(drop_c)), "coactive_fall_by_half": float(np.mean(drop_o))}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("extract", "pos", "pos_tag", "pheno", "assoc", "conditional", "causal"))
    ap.add_argument("args", nargs="*")
    a = ap.parse_args()
    if a.cmd == "extract":
        extract(*(int(x) for x in a.args))
    else:
        globals()[a.cmd]()
