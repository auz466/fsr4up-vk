"""Which initializer buffer (by its offset in AMD's DLL) each upload of a run was, in order, with the frame it came in.
    inits.py <run dir>"""
import sys, os, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dll import Dll
D = Dll(); run = sys.argv[1]; frame = None; out = []
for line in open(os.path.join(run, 'trace.log')):
    m = re.match(r'=== FRAME (\d+)', line)
    if m: frame = m.group(1)
    m = re.match(r'CopyBuffer \S+ -> \S+ size=(\d+) file=(\S+)', line)   # (128 KiB; 27,392 bytes for FSR4-Lite)
    if m and int(m.group(1)) >= 16384: out.append('f%s:%s' % (frame, hex(D.d.find(open(os.path.join(run, m.group(2)), 'rb').read()))))
print(' '.join(out))
