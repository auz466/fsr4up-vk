"""Fit the variant choice in a matrix file: which create/dispatch flag bits each pass's module depends on.
    matrix_fit.py <matrix file>"""
import sys, collections, itertools
rows = []
for l in open(sys.argv[1]):
    w = l.split('made:')[0].split()
    if len(w) < 5: print('no result:', l.strip()); continue
    cf, df = int(w[0]), int(w[1]); picks = [x.split('@') for x in w[2:]]
    rows.append((cf, df, picks))
EXTRA = {'224', '221', '226', '1010', '222', '220', '223', '219', '225'}
def split(picks):
    ids = [p[0] for p in picks]
    i = 0
    while ids[i] in EXTRA: i += 1
    return ids[:i], ids[i], ids[i + 1:i + 26], ids[i + 26], ids[i + 27:]
bits = [('cf', 2), ('cf', 4), ('cf', 8), ('cf', 32), ('cf', 256), ('cf', 512), ('df', 2), ('df', 4)]
def key(cf, df, sel): return tuple((cf if s == 'cf' else df) & b for (s, b) in sel)
for name, idx in [('before', 0), ('prepass', 1), ('model', 2), ('postpass', 3), ('after', 4)]:
    val = {(cf, df): str(split(p)[idx]) for cf, df, p in rows}
    # the smallest set of bits that determines the module
    for n in range(len(bits) + 1):
        found = None
        for sel in itertools.combinations(bits, n):
            m = {}
            if all(m.setdefault(key(cf, df, sel), v) == v for (cf, df), v in val.items()):
                found = sel; break
        if found is not None: break
    print('%-8s %3d distinct, depends on %s' % (name, len(set(val.values())), ' '.join('%s%d' % s for s in found)))
