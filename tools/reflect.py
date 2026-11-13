"""What a module of AMD's binds and how: reflection of the SPIR-V the port embeds (read as data).

reflect(blob) -> dict(entry, local, binds=[(binding, kind, resource)], post=None or (tensor index, Bx, By, capx, capy))

Only the variables a module uses are listed (SPIR-V 1.4 and later name every global an entry point touches in its
OpEntryPoint), which is what AMD's DLL binds: its set layouts list those and no others (traced)."""
import struct

# what a shader variable is bound to, by its name in AMD's SPIR-V
NAMES = {
    'ScratchBuffer': 'SCRATCH', 'InitializerBuffer': 'INIT', 'CsTensorSizes': 'UBO_TENSOR', 'MLSR_Optimized_Constants': 'UBO_MLSR',
    'g_history_sampler': 'SAMPLER',
    'r_history_color': 'HISTORY', 'rw_history_color': 'HISTORY',
    'r_velocity': 'MOTION', 'r_depth': 'DEPTH', 'r_input_color': 'COLOR', 'r_input_exposure': 'EXPOSURE',
    'r_recurrent_0': 'RECURRENT', 'rw_recurrent_0': 'RECURRENT',
    'r_reprojected_color': 'REPROJECTED', 'rw_reprojected_color': 'REPROJECTED',
    'r_auto_exposure_texture': 'AUTOEXP', 'rw_auto_exposure_texture': 'AUTOEXP',
    'rw_mlsr_output_color': 'MLSR_OUT', 'rw_debug_visualization': 'DEBUG_VIS', 'r_debug_visualization': 'DEBUG_VIS',
    'r_result_color': 'RESULT', 'r_rcas_input': 'RCAS_IN', 'rw_rcas_output': 'RCAS_OUT',
    'cbRCAS': 'UBO_RCAS', 'AutoExposureSPDConstants': 'UBO_SPD', 'rw_spd_global_atomic': 'SPD_ATOMIC',
    'rw_autoexp_mip_5': 'SPD_MIP5', 'WatermarkCB': 'UBO_WATERMARK', 'rw_target': 'MLSR_OUT',
}
RESOURCES = ['SCRATCH', 'INIT', 'UBO_TENSOR', 'UBO_MLSR', 'SAMPLER', 'HISTORY', 'MOTION', 'DEPTH', 'COLOR', 'EXPOSURE', 'RECURRENT',
             'REPROJECTED', 'AUTOEXP', 'MLSR_OUT', 'DEBUG_VIS', 'RESULT', 'RCAS_IN', 'RCAS_OUT', 'UBO_RCAS', 'UBO_SPD', 'SPD_ATOMIC',
             'SPD_MIP5', 'UBO_WATERMARK']
KINDS = ['SAMPLER', 'SAMPLED', 'STORAGE', 'UBO', 'SSBO']


def words(blob):
    return struct.unpack('<%dI' % (len(blob) // 4), blob)


def string(ws):
    b = struct.pack('<%dI' % len(ws), *ws)
    return b[:b.index(b'\0')].decode()


def instructions(blob):
    w = words(blob)
    p = 5
    while p < len(w):
        op, wc = w[p] & 0xffff, w[p] >> 16
        yield op, w[p + 1:p + wc]
        p += wc


def reflect(blob):
    names, binding, sets, types, consts, composites = {}, {}, {}, {}, {}, {}
    entry, interface, local, variables = None, [], None, {}
    ins = list(instructions(blob))
    for op, a in ins:
        if op == 5: names[a[0]] = string(a[1:])
        elif op == 15:
            entry = string(a[2:])
            n = (len(entry) + 4) // 4
            interface = list(a[2 + n:])
        elif op == 16 and a[1] == 17: local = tuple(a[2:5])
        elif op == 71 and a[1] == 33: binding[a[0]] = a[2]
        elif op == 71 and a[1] == 34: sets[a[0]] = a[2]
        elif op in (21, 23, 25, 26, 30, 32): types[a[0]] = (op, a)
        elif op == 43: consts[a[1]] = a[2]
        elif op == 44: composites[a[1]] = list(a[2:])
        elif op == 59: variables[a[1]] = (a[0], a[2])
    if words(blob)[1] < 0x10400:   # (before SPIR-V 1.4 the interface lists only inputs and outputs: every bound variable)
        interface = [v for v in variables if v in binding]
    binds = []
    for v in interface:
        if v not in binding: continue
        if sets.get(v, 0) != 0: raise ValueError('a descriptor set other than 0')
        ptr, storage = variables[v]
        pointee = types[types[ptr][1][2]]
        if pointee[0] == 26: kind = 'SAMPLER'
        elif pointee[0] == 25: kind = 'STORAGE' if pointee[1][6] == 2 else 'SAMPLED'
        else: kind = 'UBO' if storage == 2 else 'SSBO'
        name = names.get(v, '?')
        if name not in NAMES: raise ValueError('unknown shader variable ' + name)
        binds.append((binding[v], kind, NAMES[name]))
    binds.sort()
    post = None
    if entry and entry.endswith('_post'):
        # the border pass: count = f(T, B, cap) with T = CsTensorSizes.tensorSizes[k].xy; B and cap are constants
        # (ISub B - T; UMin(B - T, cap)); k: the index of the access chain into CsTensorSizes
        csz = next(v for v in interface if names.get(v) == 'CsTensorSizes')
        k = B = cap = None
        for op, a in ins:
            if op == 65 and a[2] == csz and k is None: k = consts[a[4]]
            elif op == 130 and B is None and a[2] in composites: B = [consts[c] for c in composites[a[2]]]
            elif op == 12 and a[3] == 38 and cap is None: cap = [consts[c] for c in composites[a[5]]]   # (GLSL.std.450 UMin)
        post = (k, B[0], B[1], cap[0], cap[1])
    return {'entry': entry, 'local': local, 'binds': binds, 'post': post}


if __name__ == '__main__':
    import sys
    for f in sys.argv[1:]:
        print(f, reflect(open(f, 'rb').read()))
