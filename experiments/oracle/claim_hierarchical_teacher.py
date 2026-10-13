"""Deterministic bounded rectangle refinement of a feasible literal full start.

Masks are sets of (position, component) placements. This policy never changes
routing, normalization, remainder handling, or the direct-output evaluator.
Prior rejected-site errors rank proposals heuristically; they are not additive
causal estimates. The caller must bind that prior evidence before evaluation.
"""
import math

MAX_ATTEMPTS = 32
DIRECT_MAX = .001
INTEGRITY_MAX = 1e-7


def split_rectangle(rect, depth=0):
    p0, p1, c0, c1 = rect
    if min(p1-p0, c1-c0) < 1:
        raise ValueError('nonempty half-open rectangle required')
    axis = depth % 2
    if (p1-p0 if axis == 0 else c1-c0) == 1:
        axis = 1-axis
    if (p1-p0 if axis == 0 else c1-c0) == 1:
        return ()
    if axis == 0:
        mid = (p0+p1)//2
        return ((p0, mid, c0, c1), (mid, p1, c0, c1))
    mid = (c0+c1)//2
    return ((p0, p1, c0, mid), (p0, p1, mid, c1))


def inside(point, rect):
    p, c = point
    return rect[0] <= p < rect[1] and rect[2] <= c < rect[3]


def replay(full, protected, reachable, accepted):
    masks = {site: set(points) for site, points in full.items()}
    for site, rect in accepted:
        masks[site] -= {x for x in masks[site] if x in reachable[site] and x not in protected[site] and inside(x, rect)}
    return masks


def refine(full, protected, reachable, roots, prior_errors, evaluate, emit=None):
    """All four edited worlds are measured freshly for each candidate.

    roots cover only output-reachable placements. Their whole-site parents were
    already rejected in the bound prior experiment; only children are proposed.
    evaluate returns four direct KL values, and must not use anchored outputs.
    """
    sites = set(full)
    if sites != set(protected) or sites != set(reachable) or sites != set(roots) or sites != set(prior_errors):
        raise ValueError('identical explicit site universe required')
    full = {s:set(v) for s,v in full.items()}
    protected = {s:set(v) for s,v in protected.items()}
    reachable = {s:set(v) for s,v in reachable.items()}
    for s in sites:
        if not protected[s] <= full[s] or not reachable[s] <= full[s]:
            raise ValueError('protected/reachable placements must belong to full start')
        if any(not inside(x, roots[s]) for x in reachable[s]):
            raise ValueError('root must cover every reachable placement')
        if not math.isfinite(prior_errors[s]) or prior_errors[s] <= DIRECT_MAX:
            raise ValueError('bound prior whole-site rejection required')
    def measure(masks, label):
        values = list(evaluate(masks, label))
        if len(values) != 4 or any(not math.isfinite(x) or x < 0 for x in values):
            raise RuntimeError('four finite nonnegative direct KL values required')
        return values
    current = {s:set(v) for s,v in full.items()}
    start = measure(current, 'full_start')
    if max(start) > INTEGRITY_MAX:
        return {'status':'infeasible_literal_full_start','masks':None,'start':start,'attempts':[],'accepted':[]}
    queue=[]
    def enqueue(site, rect, depth, error):
        eligible = (current[site] & reachable[site]) - protected[site]
        n = sum(inside(x, rect) for x in eligible)
        if n:
            queue.append((error/n, -n, site, rect, depth, error))
    for site in sorted(sites):
        for rect in split_rectangle(roots[site]):
            enqueue(site, rect, 1, prior_errors[site])
    attempts=[]; accepted=[]; selected=start
    while queue and len(attempts) < MAX_ATTEMPTS:
        queue.sort(); _, _, site, rect, depth, inherited = queue.pop(0)
        removed={x for x in current[site] & reachable[site] if x not in protected[site] and inside(x, rect)}
        if not removed:
            continue
        proposed={s:set(v) for s,v in current.items()}; proposed[site] -= removed
        values=measure(proposed, 'rectangle_'+str(len(attempts)))
        good=max(values) <= DIRECT_MAX
        row={'site':site,'rectangle':list(rect),'removed_reachable_placements':len(removed),'direct_kl_bits':values,'worst_direct_kl_bits':max(values),'accepted':good,'inherited_error_per_removed_placement':inherited/len(removed)}
        attempts.append(row)
        if good:
            current=proposed; selected=values; accepted.append((site, rect))
        else:
            for child in split_rectangle(rect, depth):
                enqueue(site, child, depth+1, max(values))
        if emit: emit(row)
    reconstructed=replay(full, protected, reachable, accepted)
    if reconstructed != current:
        raise RuntimeError('partial-mask recipe reconstruction failed')
    return {'status':'bounded_hierarchical_refinement' if accepted else 'full_model_no_compression','masks':current,'start':start,'selected':selected,'attempts':attempts,'accepted':accepted,'cap_hit':bool(queue) and len(attempts)==MAX_ATTEMPTS,'unfinished_queue':len(queue),'remaining_reachable_placements':sum(len(current[s]&reachable[s]) for s in sites),'recipe':{'mode':'full_minus_accepted_rectangles_with_protected_teacher_nodes','accepted_rectangles':[(s,list(r)) for s,r in accepted],'reachable_nodes':{s:sorted(v) for s,v in reachable.items()},'protected_nodes':{s:sorted(v) for s,v in protected.items()}}}
