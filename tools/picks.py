"""One line per run: the carved module of each dispatch of frame F (default 0), and each pipeline made, in order.
    picks.py <run dir> [frame]"""
import sys, os, re, json, hashlib, glob
run = sys.argv[1]; want = int(sys.argv[2]) if len(sys.argv) > 2 else 0
carved = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 're', 'carved')
idx = {m['sha']: m['index'] for m in json.load(open(os.path.join(carved, 'index.json')))}
mods = {int(re.search(r'mod_(\d+)', f).group(1)): idx.get(hashlib.sha256(open(f, 'rb').read()).hexdigest()[:16], 'NEW')
        for f in glob.glob(os.path.join(run, 'mod_*.spv'))}
pipes, made, disp, frame = {}, [], [], None
for line in open(os.path.join(run, 'trace.log')):
    m = re.match(r'CreateComputePipeline pipe(\d+) mod(\d+) entry=(\S+)', line)
    if m: pipes[m.group(1)] = mods.get(int(m.group(2))); made.append(str(pipes[m.group(1)]))
    m = re.match(r'=== FRAME (\d+)', line)
    if m: frame = int(m.group(1))
    m = re.match(r'DISPATCH \d+ \S+ (\S+) pipe(\d+)', line)
    if m and frame == want: disp.append('%s@%s' % (pipes.get(m.group(2)), m.group(1)))
print("dispatch:", " ".join(disp)); print("made:", " ".join(made))
