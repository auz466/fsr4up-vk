"""The MLSR_Optimized_Constants of each frame's prepass in a run, decoded.   mlsr.py <run dir>"""
import sys, re, struct
F = 'inv_size2 scale2 inv_scale2 jitter2 mv_scale2 tex_size2 max_renderSize2 mvjc2 width:u height:u reset:u width_lr:u height_lr:u preExposure previous_preExposure rcas_enabled:u rcas_sharpness pad'.split()
def decode(d):
    v, out, i = d, [], 0
    for f in F:
        if f.endswith('2'):
            out.append('%s=(%.9g,%.9g)' % ((f[:-1],) + struct.unpack_from('<2f', v, i))); i += 8
        elif f.endswith(':u'):
            out.append('%s=%d' % (f[:-2], struct.unpack_from('<I', v, i)[0])); i += 4
        else:
            out.append('%s=%.9g' % (f, struct.unpack_from('<f', v, i)[0])); i += 4
    return ' '.join(out)
frame = None
for line in open(sys.argv[1] + '/trace.log'):
    m = re.match(r'=== FRAME (\d+)', line)
    if m: frame = m.group(1); seen = False
    m = re.search(r'b257\[0\] ubo buf\d+ off=\d+ range=104 data=([0-9a-f]+)', line)
    if m and frame is not None and not seen:
        seen = True; print('frame', frame, decode(bytes.fromhex(m.group(1))))
