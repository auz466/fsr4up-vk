"""Compare two harness runs' dumps of the harness's own images (by name), frame by frame: identical, or how they differ.
    compare.py <run A> <run B> [name ...]   (default: out_upscaled)   exit 1 on any difference"""
import sys, os, re, struct
a, b = sys.argv[1], sys.argv[2]
names = sys.argv[3:] or ['out_upscaled']
def dumps(run):
    out = {}
    for line in open(os.path.join(run, 'trace.log')):
        m = re.match(r'dump (\S+) img\d+ mip0 (\d+)x(\d+) format=(\d+) name=(\S*)', line)
        if m: out[(m.group(1).split('_')[0], m.group(5))] = (m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4)))
    return out
da, db = dumps(a), dumps(b)
bad = 0; n = 0
for (frame, name), (fa, w, h, fmt) in sorted(da.items()):
    if name not in names: continue
    if (frame, name) not in db: print('%s %s: not in %s' % (frame, name, b)); bad += 1; continue
    x = open(os.path.join(a, fa), 'rb').read(); y = open(os.path.join(b, db[(frame, name)][0]), 'rb').read()
    n += 1
    if x == y: print('%s %s: identical (%d bytes)' % (frame, name, len(x))); continue
    bad += 1
    tb = len(x) // (w * h)
    diff = sum(1 for i in range(0, len(x), tb) if x[i:i + tb] != y[i:i + tb])
    first = next(i for i in range(len(x)) if x[i] != y[i])
    px = first // tb
    msg = '%s %s: %d of %d texels differ; first at (%d, %d)' % (frame, name, diff, w * h, px % w, px // w)
    if fmt == 97:   # RGBA16F: the largest difference
        import array
        def halfs(z): return struct.unpack('<%de' % (len(z) // 2), z)
        hx, hy = halfs(x), halfs(y)
        md = max(abs(p - q) for p, q in zip(hx, hy) if p == p and q == q)
        msg += ', largest difference %g' % md
    print(msg)
sys.exit(1 if bad or not n else 0)
