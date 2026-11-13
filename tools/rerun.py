"""Run the port (a given build) on a suite case whose AMD run is kept in re/suite/amd/<case>/, and compare.
    rerun.py <build dir> <case> [more cases]"""
import sys, os, subprocess, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'tests'))
import against_amd
cases = against_amd.cases()
build = os.path.abspath(sys.argv[1])
bad = 0
for name in sys.argv[2:]:
    w, h, rw, rh, frames, cenv = cases[name]
    env = {k: v for k, v in os.environ.items() if not k.startswith('TRACE_')}
    env.update({k: v for k, v in cenv.items() if not k.startswith('_')}, TRACE_DUMP_ALL='')
    out = os.path.join(ROOT, 're', 'rerun', name)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    subprocess.run([os.path.join(build, 'trace'), os.path.join(build, 'amd_fidelityfx_upscaler_vk.so'), out] + [str(x) for x in (w, h, rw, rh, frames, 0)],
                   env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    r = subprocess.run([sys.executable, '-I', os.path.join(ROOT, 'tools', 'compare.py'), os.path.join(ROOT, 're', 'suite', 'amd', name), out], capture_output=True, text=True)
    lines = r.stdout.strip().split('\n')
    print('%-50s %s' % (name, 'identical (%d frames)' % len(lines) if r.returncode == 0 else 'DIFFERS: ' + '; '.join(l for l in lines if 'identical' not in l)[:300]))
    bad += r.returncode != 0
sys.exit(1 if bad else 0)
