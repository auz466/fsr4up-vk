"""Summary of a harness run: which of AMD's carved modules each pipeline uses, and frame by frame every dispatch with its
groups, bindings and constant data.   summary.py <run dir> [carved dir] [--frame N]"""
import sys, os, re, json, hashlib, glob
run = sys.argv[1]
carved = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith('--') else os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 're', 'carved')
only = int(sys.argv[sys.argv.index('--frame') + 1]) if '--frame' in sys.argv else None
idx = {m['sha']: m['index'] for m in json.load(open(os.path.join(carved, 'index.json')))}
mods = {}
for f in glob.glob(os.path.join(run, 'mod_*.spv')):
    n = int(re.search(r'mod_(\d+)', f).group(1))
    mods[n] = idx.get(hashlib.sha256(open(f, 'rb').read()).hexdigest()[:16], 'NEW')
pipes = {}
frame = None
for line in open(os.path.join(run, 'trace.log')):
    line = line.rstrip('\n')
    m = re.match(r'CreateComputePipeline pipe(\d+) mod(\d+) entry=(\S+)', line)
    if m:
        pipes[m.group(1)] = (mods.get(int(m.group(2)), '?'), m.group(3))
        print('pipe%s carved#%s %s' % (m.group(1), pipes[m.group(1)][0], m.group(3)))
        continue
    m = re.match(r'=== FRAME (\d+)', line)
    if m: frame = int(m.group(1)); print(line); continue
    if only is not None and frame != only: continue
    if line.startswith(('DISPATCH', '  s0', 'Copy', 'Fill', 'Clear', 'BARRIER', '  image', '  buffer', '  memory', '=== ', 'CreateImage', 'CreateBuffer')):
        m = re.match(r'DISPATCH (\d+) \S+ (\S+) pipe(\d+)', line)
        if m: line = 'DISPATCH %s %s carved#%s %s' % (m.group(1), m.group(2), *pipes.get(m.group(3), ('?', '?')))
        print(line[:400])
