"""data/model.json from the traces of AMD's DLL: which of its modules each pass uses, per size bucket and flags, as the
SHA-256 (first 16 hex digits) of the module, so the table does not depend on how the DLL was carved.

    make_model.py   (reads re/matrix-{1080,2160,4320}.txt, re/carved/index.json; writes data/model.json)"""
import json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
idx = {m['index']: m['sha'] for m in json.load(open(os.path.join(ROOT, 're', 'carved', 'index.json')))}
EXTRA = {224, 221, 226, 1010, 222, 220, 223, 219, 225}
PRE_BITS = [('cf', 2), ('cf', 4), ('cf', 8), ('cf', 32), ('cf', 256), ('df', 2), ('df', 4)]
POST_BITS = [('cf', 32), ('cf', 256), ('cf', 512), ('df', 2), ('df', 4)]


def key(bits, cf, df):
    return sum(1 << i for i, (s, b) in enumerate(bits) if (cf if s == 'cf' else df) & b)


model = json.load(open(os.path.join(ROOT, 'data', 'model.json'))) if os.path.exists(os.path.join(ROOT, 'data', 'model.json')) else {}
buckets = []
for name, mx in (('1080', (1920, 1080)), ('2160', (3840, 2160)), ('4320', (7680, 4320))):
    pre, post, models = [None] * 128, [None] * 32, set()
    for l in open(os.path.join(ROOT, 're', 'matrix-%s.txt' % name)):
        w = l.split('made:')[0].split()
        cf, df = int(w[0]), int(w[1])
        ids = [int(x.split('@')[0]) for x in w[2:]]
        i = 0
        while ids[i] in EXTRA: i += 1
        for table, bits, v in ((pre, PRE_BITS, ids[i]), (post, POST_BITS, ids[i + 26])):
            k = key(bits, cf, df)
            if table[k] is not None and table[k] != idx[v]: sys.exit('bucket %s: flags %d %d disagree with an earlier row' % (name, cf, df))
            table[k] = idx[v]
        models.add(tuple(idx[x] for x in ids[i + 1:i + 26]))
    if len(models) != 1 or None in pre or None in post: sys.exit('bucket %s: incomplete or inconsistent' % name)
    buckets.append({'name': name, 'max': mx, 'model': list(models.pop()), 'prepass': pre, 'postpass': post})
model['buckets'] = buckets
for b in buckets:
    b['scratch'] = model.get('scratch', {}).get(b['name'])
json.dump(model, open(os.path.join(ROOT, 'data', 'model.json'), 'w'), indent=1)
print('model.json:', ', '.join('%s: %d prepass, %d postpass variants' % (b['name'], len(set(b['prepass'])), len(set(b['postpass']))) for b in buckets))
