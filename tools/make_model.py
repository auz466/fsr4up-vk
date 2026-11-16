"""data/model.json from the traces of AMD's DLL: which of its modules each pass uses, per size bucket and flags, as the
SHA-256 (first 16 hex digits) of the module, so the table does not depend on how the DLL was carved.

    make_model.py                  the FP8 network: reads re/matrix-{1080,2160,4320}.txt, re/carved/index.json; writes
                                   the top-level buckets of data/model.json
    make_model.py --net <name> --shift <levels>
                                   an INT8 network (i8, i8_navi2, lite: traced with tools/matrix.sh and tools/sizes.sh
                                   with TRACE_AMD_FAMILY / TRACE_AMD_REV, see HOW_IT_WAS_MADE.md): reads
                                   re/matrix-<name>-<bucket>.txt (standard preset), re/matrix-<name>u-<bucket>.txt (ultra
                                   performance), re/int8/sizes-<name>.txt (scratch sizes, initializers) and
                                   re/int8/fit-<name>.json (tools/fit_groups.py: the dispatch sizes); writes
                                   data/model.json's int8.<name>. <levels>: its CsTensorSizes levels, comma-separated."""
import argparse, hashlib, json, os, re, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'tools'))
idx = {m['index']: m['sha'] for m in json.load(open(os.path.join(ROOT, 're', 'carved', 'index.json')))}
EXTRA = {224, 221, 226, 1010, 222, 220, 223, 219, 225}
PRE_BITS = [('cf', 2), ('cf', 4), ('cf', 8), ('cf', 32), ('cf', 256), ('df', 2), ('df', 4)]
POST_BITS = [('cf', 32), ('cf', 256), ('cf', 512), ('df', 2), ('df', 4)]
BUCKETS = (('1080', (1920, 1080)), ('2160', (3840, 2160)), ('4320', (7680, 4320)))
AMD_NAMES = {'i8': 'FSR4-i8', 'i8_navi2': 'FSR4-i8-2', 'lite': 'FSR4L-i8'}


def key(bits, cf, df):
    return sum(1 << i for i, (s, b) in enumerate(bits) if (cf if s == 'cf' else df) & b)


def tables(path, name):
    """(prepass by key, postpass by key, the network's passes) from a matrix file (frame 1 of every flag combination:
    [SPD] prepass, the network, postpass; nothing after it, as the matrix asks for no sharpening and no debug view)"""
    pre, post, models = [None] * 128, [None] * 32, set()
    for l in open(path):
        w = l.split('made:')[0].split()
        cf, df = int(w[0]), int(w[1])
        ids = [int(x.split('@')[0]) for x in w[2:]]
        i = 0
        while ids[i] in EXTRA: i += 1
        if ids[-1] in EXTRA: sys.exit('%s: flags %d %d: a pass after the postpass' % (path, cf, df))
        for table, bits, v in ((pre, PRE_BITS, ids[i]), (post, POST_BITS, ids[-1])):
            k = key(bits, cf, df)
            if table[k] is not None and table[k] != idx[v]: sys.exit('%s: flags %d %d disagree with an earlier row' % (name, cf, df))
            table[k] = idx[v]
        models.add(tuple(idx[x] for x in ids[i + 1:-1]))
    if len(models) != 1 or None in pre or None in post: sys.exit('%s: incomplete or inconsistent' % name)
    return pre, post, list(models.pop())


def fp8():
    model = json.load(open(os.path.join(ROOT, 'data', 'model.json'))) if os.path.exists(os.path.join(ROOT, 'data', 'model.json')) else {}
    buckets = []
    for name, mx in BUCKETS:
        pre, post, models = tables(os.path.join(ROOT, 're', 'matrix-%s.txt' % name), 'bucket ' + name)
        if len(models) != 25: sys.exit('bucket %s: %d network passes, not 25' % (name, len(models)))
        buckets.append({'name': name, 'max': mx, 'model': models, 'prepass': pre, 'postpass': post})
    model['buckets'] = buckets
    for b in buckets:
        b['scratch'] = model.get('scratch', {}).get(b['name'])
    json.dump(model, open(os.path.join(ROOT, 'data', 'model.json'), 'w'), indent=1)
    print('model.json:', ', '.join('%s: %d prepass, %d postpass variants' % (b['name'], len(set(b['prepass'])), len(set(b['postpass']))) for b in buckets))


def int8(net, shift):
    from dll import Dll
    D = Dll()
    path = os.path.join(ROOT, 'data', 'model.json')
    model = json.load(open(path))
    fit = json.load(open(os.path.join(ROOT, 're', 'int8', 'fit-%s.json' % net)))
    scratch, inits = {}, {}
    for l in open(os.path.join(ROOT, 're', 'int8', 'sizes-%s.txt' % net)):
        w = l.split()
        W, H, RW = int(w[0]), int(w[1]), int(w[2])
        b = '1080' if W <= 1920 and H <= 1080 else '2160' if W <= 3840 and H <= 2160 else '4320'
        s = int(re.search(r'scratch size=(\d+)', l).group(1))
        if scratch.setdefault(b, s) != s: sys.exit('%s: the scratch size changes within bucket %s' % (net, b))
        preset = 'ultra' if W / RW >= 2.99 else 'standard'
        ups = re.findall(r'f0:(0x[0-9a-f]+)', l)
        if len(ups) != 1: sys.exit('%s %dx%d: %d initializer uploads' % (net, W, H, len(ups)))
        if inits.setdefault(preset, int(ups[0], 16)) != int(ups[0], 16): sys.exit('%s: the %s initializer changes' % (net, preset))
    out = {'amd_name': AMD_NAMES[net], 'tensor_shift': shift, 'buckets': []}
    for preset, k in (('standard', 'initializer'), ('ultra', 'initializer_ultra')):
        # (its size: the upload's, which a run's trace gives; the same for both presets)
        size = model.get('int8', {}).get(net, {}).get(k, {}).get('size') or int(os.environ.get('INIT_SIZE', '0'))
        if not size: sys.exit('INIT_SIZE: the initializer buffer size of %s (from a trace: its CopyBuffer)' % net)
        blob = D.d[inits[preset]:inits[preset] + size]
        out[k] = {'offset': inits[preset], 'size': size, 'sha256': hashlib.sha256(blob).hexdigest()}
    for name, mx in BUCKETS:
        b = {'name': name, 'max': mx, 'scratch': scratch[name]}
        for preset, tag in (('standard', ''), ('ultra', 'u')):
            pre, post, models = tables(os.path.join(ROOT, 're', 'matrix-%s%s-%s.txt' % (net, tag, name)), '%s %s %s' % (net, preset, name))
            f = fit['%s %s' % (name, preset)]
            if [idx[m] for m in f['modules'][1:-1]] != models: sys.exit('%s %s %s: the sizes runs ran other network passes' % (net, preset, name))
            if idx[f['modules'][0]] != pre[key(PRE_BITS, 0x18, 0)] or idx[f['modules'][-1]] != post[key(POST_BITS, 0x18, 0)]:
                sys.exit('%s %s %s: the sizes runs ran another prepass or postpass' % (net, preset, name))
            b[preset] = {'model': models, 'groups': f['groups'], 'prepass': pre, 'postpass': post}
        out['buckets'].append(b)
    model.setdefault('int8', {})[net] = out
    json.dump(model, open(path, 'w'), indent=1)
    print('model.json int8.%s: %s' % (net, ', '.join('%s: %d passes, %d+%d prepass, %d+%d postpass variants' % (
        b['name'], len(b['standard']['model']), len(set(b['standard']['prepass'])), len(set(b['ultra']['prepass']) - set(b['standard']['prepass'])),
        len(set(b['standard']['postpass'])), len(set(b['ultra']['postpass']) - set(b['standard']['postpass']))) for b in out['buckets'])))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--net')
    ap.add_argument('--shift')
    a = ap.parse_args()
    if a.net: int8(a.net, [int(x) for x in a.shift.split(',')])
    else: fp8()
