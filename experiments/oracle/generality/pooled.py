"""Pooled paired differences over target texts (one value per target: the mean over its mapped circuits), families
other than the copy behaviour and the copy behaviour apart, with 95% bootstrap intervals over targets."""
import numpy as np

import analyze


def boot(x, n=4000, seed=0):
    x = np.asarray(x, float)
    m = np.random.default_rng(seed).choice(x, (n, len(x))).mean(1)
    return float(x.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def main():
    _, pt = analyze.summary(analyze.load(analyze.FILES))
    out = {}
    for group, keep in (("non-copy families", lambda f: f != "B:copy"), ("copy behaviour", lambda f: f == "B:copy")):
        items = [(f, t, d) for (f, t), d in pt.items() if keep(f)]
        print(f"== {group}: {len(items)} target texts")
        for k in analyze.METRICS:
            comps = {
                "mapped - outside-family mapped": [],
                "mapped - VPD top k (mapped size)": [],
                "core - VPD top k (core size)": [],
                "core - random-texts core": [],
                "core - universal core": [],
                "own - mapped": [],
            }
            passes = {"core >= 0.5": [], "VPD top k at core size >= 0.5": [], "mapped >= 0.5": [], "own >= 0.5": []}
            for f, t, d in items:
                mp = [r[k] for r in d["mapped"].values() if r[k] is not None]
                ct = [r[k] for r in d["control"] if r[k] is not None]
                pairs = [(r[k], (d["vpd"].get(r["nodes"]) or {}).get(k)) for r in d["mapped"].values()]
                pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
                if mp and ct:
                    comps["mapped - outside-family mapped"].append(np.mean(mp) - np.mean(ct))
                if pairs:
                    comps["mapped - VPD top k (mapped size)"].append(np.mean([a - b for a, b in pairs]))
                if "own" in d and mp and d["own"][k] is not None:
                    comps["own - mapped"].append(d["own"][k] - np.mean(mp))
                if "core" in d and d["core"][k] is not None:
                    cv = (d["vpd"].get(d["core"]["nodes"]) or {}).get(k)
                    if cv is not None:
                        comps["core - VPD top k (core size)"].append(d["core"][k] - cv)
                        passes["VPD top k at core size >= 0.5"].append(cv >= 0.5)
                    passes["core >= 0.5"].append(d["core"][k] >= 0.5)
                    if "core_random" in d and d["core_random"][k] is not None:
                        comps["core - random-texts core"].append(d["core"][k] - d["core_random"][k])
                    if "universal" in d and d["universal"][k] is not None:
                        comps["core - universal core"].append(d["core"][k] - d["universal"][k])
                passes["mapped >= 0.5"] += [x >= 0.5 for x in mp]
                if "own" in d and d["own"][k] is not None:
                    passes["own >= 0.5"].append(d["own"][k] >= 0.5)
            print(f"  {k}")
            for name, xs in comps.items():
                if xs:
                    b = boot(xs)
                    out[(group, k, name)] = (b, sum(x > 0 for x in xs), len(xs))
                    print(f"    {name:34s} {b[0]:+.3f} [{b[1]:+.3f}, {b[2]:+.3f}]  positive on {sum(x > 0 for x in xs)}/{len(xs)}")
            print("    pass rates: " + "  ".join(f"{n} {np.mean(v):.2f}" for n, v in passes.items() if v))
    return out


if __name__ == "__main__":
    main()
