"""The dispatch sizes of a network's passes, fitted from runs of AMD's DLL at many display sizes (tools/sizes.sh, one
line a run: "W H RW RH scratch size=N inits f0:0x... mem ... dispatch: carved@XxYx1 ..."), with the default create
flags (no auto exposure: the first dispatch is the prepass).

    fit_groups.py <sizes file> [out.json] [--shift 1,0,...]   (--shift: the network's CsTensorSizes levels, if not the
                                 17 of FP8 and the two big INT8 networks)  prints, per size bucket and preset (ratio < 2.99: standard, else ultra performance):
                                 the modules of every pass (which must not change with the size), and for each pass the
                                 rule its groups follow: [s, a, b] for (ceil((W8 >> s) / a), ceil((H8 >> s) / b)) with W8, H8
                                 the display size rounded up to 8, or "post" for a border pass whose count its own SPIR-V
                                 gives (tools/reflect.py; checked here against the runs)"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reflect import reflect

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CARVED = os.path.join(ROOT, 're', 'carved')
index = {m['index']: m for m in json.load(open(os.path.join(CARVED, 'index.json')))}
SHIFT = [1, 0, 1, 1, 2, 2, 2, 3, 3, 3, 2, 2, 1, 1, 0, 0, 0]   # (CsTensorSizes: the level of each tensor, traced)


def blob(i):
    return open(os.path.join(CARVED, '%04d_%08x.spv' % (i, index[i]['offset'])), 'rb').read()


def bucket(w, h):
    return '1080' if w <= 1920 and h <= 1080 else '2160' if w <= 3840 and h <= 2160 else '4320'


def post_groups(post, local, w, h):
    k, bx, by, cx, cy = post
    pw, ph = (w + 7) & ~7, (h + 7) & ~7
    tx, ty = pw >> SHIFT[k], ph >> SHIFT[k]
    px, py = min(bx - tx, cx), min(by - ty, cy)
    row = 1 + tx + px
    count = row + ty + ty * px + row * py
    return ((count + local - 1) // local, 1)


def fit(samples):
    """the rule [s, a, b] that gives every (w, h) -> (gx, gy), or None"""
    for s in range(5):
        for a in (1, 2, 4, 8, 16, 32, 64, 128, 256):
            for b in (1, 2, 4, 8, 16, 32, 64, 128, 256):
                if all(-(-(((w + 7) & ~7) >> s) // a) == gx and -(-(((h + 7) & ~7) >> s) // b) == gy for (w, h), (gx, gy) in samples):
                    return [s, a, b]
    return None


def main():
    global SHIFT
    if '--shift' in sys.argv:
        i = sys.argv.index('--shift')
        SHIFT = [int(x) for x in sys.argv[i + 1].split(',')]
        del sys.argv[i:i + 2]
    runs = {}
    for l in open(sys.argv[1]):
        w = l.split()
        W, H, RW = int(w[0]), int(w[1]), int(w[2])
        picks = l.split('dispatch:')[1].split()
        preset = 'ultra' if W / RW >= 2.99 else 'standard'
        runs.setdefault((bucket(W, H), preset), []).append(((W, H), [(int(p.split('@')[0]), tuple(int(x) for x in p.split('@')[1].split('x')[:2])) for p in picks]))
    out = {}
    for key in sorted(runs):
        rs = runs[key]
        mods = [m for m, _ in rs[0][1]]
        bad = [wh for wh, p in rs if [m for m, _ in p] != mods]
        if bad: sys.exit('%s: the modules change with the size: %s' % (key, bad))
        rules = []
        for place, m in enumerate(mods):
            r = reflect(blob(m))
            samples = [(wh, p[place][1]) for wh, p in rs]
            if r['post']:
                local = r['local'][0] * r['local'][1] * r['local'][2]
                wrong = [wh for wh, g in samples if post_groups(r['post'], local, *wh) != g]
                if wrong: sys.exit('%s place %d (%d %s): the border formula does not give %s' % (key, place, m, r['entry'], wrong))
                rules.append('post')
            else:
                f = fit(samples)
                if f is None: sys.exit('%s place %d (%d %s): no rule fits %s' % (key, place, m, r['entry'], samples))
                rules.append(f)
        out['%s %s' % key] = {'modules': mods, 'groups': rules, 'sizes': len(rs)}
        print('%s %-8s %d sizes, %d passes: %s' % (key[0], key[1], len(rs), len(mods), ' '.join('%d:%s' % (m, 'post' if g == 'post' else '%d/%d/%d' % tuple(g)) for m, g in zip(mods, rules))))
    if len(sys.argv) > 2: json.dump(out, open(sys.argv[2], 'w'), indent=1)


if __name__ == '__main__':
    main()
