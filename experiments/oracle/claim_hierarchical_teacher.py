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


def tensor_refine(order, full, protected, evaluate, count, emit, positions, prior):
    """Memory-bounded adapter: rectangles operate on original Boolean masks."""
    current={s:m.clone() for s,m in full.items()};accepted=[];attempts=[];queue=[]
    def measure(m,label):
        v=list(evaluate(m,label))
        if len(v)!=4 or any(not math.isfinite(x) or x<0 for x in v):raise RuntimeError('invalid direct measurements')
        return {'direct_kl_bits':v,'worst_direct_kl_bits':max(v),'counts':count(m)}
    start=measure(current,'full_start');selected=start
    if start['worst_direct_kl_bits']>INTEGRITY_MAX:
        return {'status':'infeasible_literal_full_start','masks':None,'start':start,'attempts':[],'accepted_sites':[]}
    def eligible(site,rect):
        p0,p1,c0,c1=rect
        piece=current[site][p0:p1,c0:c1] & ~protected[site][p0:p1,c0:c1]
        # Position spans are contiguous prefixes or singleton final targets.
        return piece
    def enqueue(s,r,d,e):
        n=int(eligible(s,r).sum())
        if n:queue.append((e/n,-n,s,r,d,e))
    for s in order:
        ps=positions[s];root=(min(ps),max(ps)+1,0,full[s].shape[1])
        if tuple(ps)!=tuple(range(root[0],root[1])):raise ValueError('contiguous architecture reachability required')
        for r in split_rectangle(root):enqueue(s,r,1,prior[s])
    while queue and len(attempts)<MAX_ATTEMPTS:
        queue.sort();_,_,s,r,d,e=queue.pop(0);removed=eligible(s,r);n=int(removed.sum())
        if not n:continue
        p0,p1,c0,c1=r;changed=current[s].clone();changed[p0:p1,c0:c1] &= ~removed
        proposed={**current,s:changed};row={'site':s,'rectangle':list(r),'removed_reachable_placements':n,**measure(proposed,'rectangle_'+str(len(attempts)))}
        row['accepted']=row['worst_direct_kl_bits']<=DIRECT_MAX;row['inherited_error_per_removed_placement']=e/n
        if row['accepted']:current=proposed;selected=row;accepted.append((s,r))
        else:
            for child in split_rectangle(r,d):enqueue(s,child,d+1,row['worst_direct_kl_bits'])
        attempts.append(row)
        if emit:emit(row)
    replayed={s:m.clone() for s,m in full.items()}
    for s,r in accepted:
        p0,p1,c0,c1=r;replayed[s][p0:p1,c0:c1] &= protected[s][p0:p1,c0:c1]
    if any(not __import__('torch').equal(current[s],replayed[s]) for s in order):raise RuntimeError('tensor recipe replay failed')
    return {'status':'bounded_hierarchical_refinement' if accepted else 'full_model_no_compression','masks':current,'start':start,'selected':selected,'attempts':attempts,'accepted_sites':sorted({s for s,r in accepted}),'cap_hit':bool(queue) and len(attempts)==MAX_ATTEMPTS,'unfinished_queue':len(queue),'recipe':{'mode':'full_minus_rectangles_preserving_teacher','shapes':{s:list(m.shape) for s,m in full.items()},'accepted_rectangles':[(s,list(r)) for s,r in accepted],'protected_nodes':{s:protected[s].nonzero().tolist() for s in order},'reachable_position_spans':{s:[min(positions[s]),max(positions[s])+1] for s in order}}}
