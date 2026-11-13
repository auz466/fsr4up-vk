"""Uniform block layouts from a module's debug names: spv_structs.py file.spv [struct name substring]"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spv import Module
m = Module(open(sys.argv[1], 'rb').read()); want = sys.argv[2] if len(sys.argv) > 2 else ''
def s(ws):
    import struct
    b = struct.pack('<%dI' % len(ws), *ws); return b[:b.index(b'\0')].decode()
names, members, offsets, types = {}, {}, {}, {}
for x in m.ins:
    if x.op == 5: names[x.w[0]] = s(x.w[1:])
    elif x.op == 6: members.setdefault(x.w[0], {})[x.w[1]] = s(x.w[2:])
    elif x.op == 72 and x.w[2] == 35: offsets.setdefault(x.w[0], {})[x.w[1]] = x.w[3]
    elif 19 <= x.op <= 32: types[x.w[0]] = x
def tname(t):
    x = types.get(t)
    if x is None: return '?'
    if x.op == 21: return ('int' if x.w[2] else 'uint') + ('' if x.w[1] == 32 else str(x.w[1]))
    if x.op == 22: return {16: 'half', 32: 'float', 64: 'double'}.get(x.w[1], 'f%d' % x.w[1])
    if x.op == 23: return '%s%d' % (tname(x.w[1]), x.w[2])
    if x.op == 24: return 'mat%d(%s)' % (x.w[2], tname(x.w[1]))
    if x.op == 28: return '%s[]' % tname(x.w[1])
    if x.op == 30: return names.get(t, 'struct')
    return 'op%d' % x.op
for t, ms in members.items():
    if want in names.get(t, ''):
        x = types[t]
        print(names.get(t))
        for i in sorted(ms): print('  +%3d %-12s %s' % (offsets.get(t, {}).get(i, -1), tname(x.w[1 + i]), ms[i]))
