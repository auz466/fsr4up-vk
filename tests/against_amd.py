#!/usr/bin/env python3
"""The regression suite: every case through AMD's DLL under Wine and through the port, and any byte of difference in an
upscaled frame fails. The harness's own inputs (colour, depth, motion) are compared too: a difference there is the
harness's fault, not the port's, and also fails. Every API result the harness logs (queries with their answers, create,
dispatch, destroy) is compared as well, but for GET_VERSIONS (AMD's DLL also offers FSR 3.1.6 and 2.3.4, which this
library does not implement).

    python3 -I tests/against_amd.py [-k substring ...] [--list] [--refresh-amd] [--keep] [--build DIR] [--re DIR]

AMD's results are cached as SHA-256 hashes per image in <re>/suite/amd/<case>.json, keyed by the case, the harness source
and AMD's DLL, so AMD only runs again when one of those changes (or with --refresh-amd). The raw images are deleted after
a passing case unless --keep. A failing case keeps the port's images in <re>/suite/port/<case>/ (and AMD's in
<re>/suite/amd/<case>/ if AMD ran this time; --refresh-amd -k <case> makes it run), for tools/compare.py.
"""
import argparse, hashlib, json, os, re, shutil, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

HDR, DMV, JC, INV, INF, AE, DYN, CHECK, NONLIN, DBGVIS = (1 << i for i in range(10))
SRGB, PQ, DBGVIEW = 2, 4, 1


def render_size(w, h, ratio):
    """AMD's render size for a quality mode's ratio (GETRENDERRESOLUTIONFROMQUALITYMODE: in float, truncated)"""
    import struct
    f = lambda x: struct.unpack('<f', struct.pack('<f', x))[0]
    return int(f(f(w) / f(ratio))), int(f(f(h) / f(ratio)))


def cases():
    """name -> (W, H, RW, RH, frames, environment)"""
    out = {}
    base_flags = str(INV | INF)
    for w, h in ((1280, 720), (1920, 1080), (2560, 1440), (3840, 2160), (5120, 2880), (1001, 777)):
        for mode, ratio in (('quality', 1.5), ('performance', 2.0)):
            rw, rh = render_size(w, h, ratio)
            for cam in ('still', 'moving'):
                env = {'TRACE_CAMERA': '0.5'} if cam == 'moving' else {}
                out['%dx%d_%s_%s' % (w, h, mode, cam)] = (w, h, rw, rh, 4, env)
    mv = {'TRACE_CAMERA': '0.5'}
    B = '1920x1080_quality_moving'
    for mode, ratio in (('nativeaa', 1.0), ('balanced', 1.7), ('ultraperformance', 3.0)):
        rw, rh = render_size(1920, 1080, ratio)
        out['1920x1080_%s_moving' % mode] = (1920, 1080, rw, rh, 4, dict(mv))
    for w, h in ((3840, 2160), (5120, 2880)):   # (ultra performance's own initializer, in the bigger networks)
        out['%dx%d_ultraperformance_moving' % (w, h)] = (w, h) + render_size(w, h, 3.0) + (4, dict(mv))
    out['3440x1440_balanced_moving'] = (3440, 1440) + render_size(3440, 1440, 1.7) + (4, dict(mv))
    out['1080x1920_quality_moving'] = (1080, 1920, 720, 1280, 4, dict(mv))   # (portrait: the 2160 network)
    out['1920x1088_quality_moving'] = (1920, 1088, 1280, 725, 4, dict(mv))   # (just over 1080: the 2160 network)

    def case(name, frames=4, **env):
        e = dict(mv, TRACE_CREATE_FLAGS=base_flags)
        e.update({k: str(v) for k, v in env.items()})
        out[B + '_' + name] = (1920, 1080, 1280, 720, frames, e)

    case('reset3', 6, TRACE_RESET_AT=3)
    for flags, nm in ((0, 'depth_std_finite'), (INV, 'depth_inv_finite'), (INF, 'depth_std_infinite'), (INV | INF, 'depth_inv_infinite')):
        if flags & INF: case(nm, TRACE_CREATE_FLAGS=flags)
        else: case(nm, TRACE_CREATE_FLAGS=flags, TRACE_FAR=1000)
    case('jittercancel', TRACE_CREATE_FLAGS=INV | INF | JC)
    case('jittercancel_reset2', 5, TRACE_CREATE_FLAGS=INV | INF | JC, TRACE_RESET_AT=2, TRACE_MVSCALE_X=2, TRACE_MVSCALE_Y=3)
    case('nojitter', TRACE_NO_JITTER=1)
    case('mvscale', TRACE_MVSCALE_X=0.5, TRACE_MVSCALE_Y=2)
    case('displaymv', TRACE_CREATE_FLAGS=INV | INF | DMV)
    case('displaymv_jittercancel', TRACE_CREATE_FLAGS=INV | INF | DMV | JC)
    case('displaymv_std_depth', TRACE_CREATE_FLAGS=DMV, TRACE_FAR=1000)
    case('autoexposure', TRACE_CREATE_FLAGS=INV | INF | AE)
    case('autoexposure_preexposure', TRACE_CREATE_FLAGS=INV | INF | AE, TRACE_PREEXPOSURE='1,1.5,0.8,2')
    case('exposure_texture', TRACE_EXPOSURE=0.7)
    case('preexposure', TRACE_PREEXPOSURE='1,1.5,0.8,2')
    case('hdr', TRACE_CREATE_FLAGS=INV | INF | HDR, TRACE_COLOR_GAIN=4)
    case('nonlinear', TRACE_CREATE_FLAGS=INV | INF | NONLIN)
    case('nonlinear_autoexposure', TRACE_CREATE_FLAGS=INV | INF | NONLIN | AE)
    case('srgb', TRACE_DISPATCH_FLAGS=SRGB)
    case('pq', TRACE_DISPATCH_FLAGS=PQ)
    case('srgb_pq', TRACE_DISPATCH_FLAGS=SRGB | PQ)
    case('srgb_autoexposure', TRACE_CREATE_FLAGS=INV | INF | AE, TRACE_DISPATCH_FLAGS=SRGB)
    case('pq_nonlinear_jc', TRACE_CREATE_FLAGS=INV | INF | NONLIN | JC, TRACE_DISPATCH_FLAGS=PQ)
    case('srgb_switch', 5, TRACE_DISPATCH_FLAGS_LIST='0,2,2,0,4')
    case('debugvis', TRACE_CREATE_FLAGS=INV | INF | DBGVIS)
    # (AMD's debug view pass samples the output and writes it in the same dispatch: its own runs differ from each other by
    # a few texels (traced: 11 of 2,073,600 in one frame of three runs). Racy: a frame may differ in up to RACY_TEXELS.)
    case('debugview', TRACE_CREATE_FLAGS=INV | INF | DBGVIS, TRACE_DISPATCH_FLAGS=DBGVIEW, _RACY=1)
    case('debugview_autoexposure', TRACE_CREATE_FLAGS=INV | INF | DBGVIS | AE, TRACE_DISPATCH_FLAGS=DBGVIEW, _RACY=1)
    case('debugview_refused', TRACE_DISPATCH_FLAGS=DBGVIEW, _EXPECT='6')
    case('debugchecking_refused', TRACE_CREATE_FLAGS=INV | INF | CHECK, _EXPECT='6')
    case('debugchecking_exposure', TRACE_CREATE_FLAGS=INV | INF | CHECK, TRACE_EXPOSURE=1)
    case('sharpen0', TRACE_SHARPNESS=0)
    case('sharpen05', TRACE_SHARPNESS=0.5)
    case('sharpen1', TRACE_SHARPNESS=1)
    case('sharpen_autoexposure', TRACE_SHARPNESS=0.7, TRACE_CREATE_FLAGS=INV | INF | AE)
    case('sharpen_exposure', TRACE_SHARPNESS=0.3, TRACE_EXPOSURE=0.6, TRACE_PREEXPOSURE='1,2,2,1')
    case('sharpen_srgb', TRACE_SHARPNESS=0.4, TRACE_DISPATCH_FLAGS=SRGB)
    case('sharpen_nonlinear', TRACE_SHARPNESS=0.4, TRACE_CREATE_FLAGS=INV | INF | NONLIN)
    case('sharpen_pq', TRACE_SHARPNESS=0.4, TRACE_DISPATCH_FLAGS=PQ)
    case('reactive_tc', TRACE_REACTIVE=1, TRACE_TC=1)
    case('generate_reactive', TRACE_GEN_REACTIVE=15)
    case('keyvalue', TRACE_KV='0:0.5,1:2,2:1.5,3:0.2,4:-0.5')
    case('dynamic', 5, TRACE_CREATE_FLAGS=INV | INF | DYN, TRACE_MAX='1280,720', TRACE_RENDER_W='1280,1100,960,1200,1280',
         TRACE_RENDER_H='720,620,540,700,720')
    case('dynamic_noflag', 4, TRACE_MAX='1280,720', TRACE_RENDER_W='1280,1100,960,1200', TRACE_RENDER_H='720,620,540,700')
    case('upscale_smaller', 4, TRACE_UPSCALE_W='1920,1600,1920,1700', TRACE_UPSCALE_H='1080,900,1080,950')
    for fmt, outf, nm in ((37, 37, 'rgba8'), (43, 43, 'rgba8srgb'), (44, 44, 'bgra8'), (64, 64, 'a2b10g10r10'), (37, 97, 'rgba8_to_rgba16f'),
                          (97, 37, 'rgba16f_to_rgba8')):
        case('format_' + nm, TRACE_COLOR_FORMAT=fmt, TRACE_OUT_FORMAT=outf)
    case('long', 12)
    # (what makes AMD's DLL start its network again: colour flags, sizes without DYNAMIC_RESOLUTION; and what it keeps then)
    case('srgb_switch_jc_preexposure', 5, TRACE_CREATE_FLAGS=INV | INF | JC, TRACE_DISPATCH_FLAGS_LIST='0,2,2,0,4', TRACE_PREEXPOSURE='1,2,1.5,1,3')
    case('srgb_switch_dynamic', 5, TRACE_CREATE_FLAGS=INV | INF | DYN, TRACE_DISPATCH_FLAGS_LIST='0,2,2,0,4')
    case('ultra_switch', 5, TRACE_MAX='1280,720', TRACE_RENDER_W='1280,640,640,1280,642', TRACE_RENDER_H='720,360,360,720,361')
    case('ultra_switch_dynamic', 5, TRACE_CREATE_FLAGS=INV | INF | DYN, TRACE_MAX='1280,720', TRACE_RENDER_W='640,1280,640,1280,640',
         TRACE_RENDER_H='360,720,360,720,360', TRACE_DISPATCH_FLAGS_LIST='0,0,2,2,0')
    case('upscale_smaller_dynamic', 4, TRACE_CREATE_FLAGS=INV | INF | DYN, TRACE_UPSCALE_W='1920,1600,1920,1700', TRACE_UPSCALE_H='1080,900,1080,950')
    case('upscale_smaller_ultra', 3, TRACE_MAX='640,360', TRACE_RENDER_W='640,640,640', TRACE_RENDER_H='360,360,360', TRACE_UPSCALE_W='1920,1900,1916',
         TRACE_UPSCALE_H='1080,1080,1080')
    case('sharpen_toggle', 5, TRACE_SHARPNESS='0.5,0.5,0.2,0.8,0.8', TRACE_SHARPEN_ON='0,1,1,0,1')
    return out


def sha_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 22), b''): h.update(chunk)
    return h.hexdigest()


def amd_key(build, case):
    h = hashlib.sha256(json.dumps(case, sort_keys=True).encode())
    h.update(open(os.path.join(ROOT, 'harness', 'trace.cpp'), 'rb').read())
    h.update(sha_file(os.path.join(build, 'wine', 'amd_fidelityfx_upscaler_vk.dll')).encode())
    return h.hexdigest()


def images(run_dir, refused=False):
    """(frame, name) -> hash, for the harness's inputs and the upscaled frames"""
    log = open(os.path.join(run_dir, 'trace.log'), newline='').read()
    wanted = {'in_color', 'in_depth', 'in_motion', 'out_upscaled'}
    out = {}
    for m in re.finditer(r'^dump (f(\d+)_img\d+_m0\.raw) img\d+ mip0 (\d+)x(\d+) format=(\d+) name=(\S+)\r?$', log, re.M):
        if m.group(6) in wanted:
            path = os.path.join(run_dir, m.group(1))
            out['f%s %s %sx%s fmt%s' % (m.group(2), m.group(6), m.group(3), m.group(4), m.group(5))] = sha_file(path)
            if m.group(6) == 'out_upscaled' and not refused and not any(open(path, 'rb').read()):
                out['f%s %s' % (m.group(2), m.group(6))] = 'ALL ZERO'   # (nothing written: never a pass, even if both agree)
    return out


NOT_COMPARED = ('query versions',)
RACY_TEXELS = 64


def texels_differing(a_dir, b_dir, key):
    """how many texels differ between two runs' dumps of the image a key names (f<n> <name> <w>x<h> fmt<f>)"""
    frame, name, size, _ = key.split()
    w, h = (int(x) for x in size.split('x'))
    def path(d):
        for m in re.finditer(r'^dump (%s_img\d+_m0\.raw) img\d+ mip0 \S+ format=\d+ name=%s\r?$' % (frame, name),
                             open(os.path.join(d, 'trace.log'), newline='').read(), re.M):
            return os.path.join(d, m.group(1))
    pa, pb = path(a_dir), path(b_dir)
    if not pa or not pb: return None
    x, y = open(pa, 'rb').read(), open(pb, 'rb').read()
    tb = len(x) // (w * h)
    return sum(1 for i in range(0, len(x), tb) if x[i:i + tb] != y[i:i + tb])


def api_results(run_dir):
    """every API line the harness logged ('=== ...'), with what it returned and answered; pointers left out"""
    text = open(os.path.join(run_dir, 'trace.log'), newline='').read().replace('\r', '')
    return [re.sub(r' ctx=\S+', '', l) for l in re.findall(r'^=== (.*)$', text, re.M)]


def compared(results):
    return [x for x in results if not x.startswith(NOT_COMPARED)]


def run(cmd, env, cwd, log):
    t = time.time()
    with open(log, 'w') as f:
        r = subprocess.run(cmd, env=env, cwd=cwd, stdout=f, stderr=subprocess.STDOUT)
    return r.returncode, time.time() - t


def results_of(stdout_log, frames, expect):
    """every 'frame N: upscale -> R' line: the run must have made every frame with the expected result"""
    got = re.findall(r'frame (\d+): upscale -> (\d+)', open(stdout_log, errors='replace').read())
    return len(got) == frames and all(r == expect for _, r in got), got


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('-k', action='append', default=[], help='only cases whose name contains this (repeatable)')
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--refresh-amd', action='store_true', help="run AMD's DLL again even if cached")
    ap.add_argument('--keep', action='store_true', help='keep the raw images of passing cases')
    ap.add_argument('--build', default=os.environ.get('FSR4UP_BUILD', os.path.join(ROOT, 'build')))
    ap.add_argument('--re', default=os.environ.get('FSR4UP_RE', os.path.join(ROOT, 're')))
    a = ap.parse_args()
    build, suite = os.path.abspath(a.build), os.path.join(os.path.abspath(a.re), 'suite')
    todo = {n: c for n, c in cases().items() if not a.k or any(k in n for k in a.k)}
    if a.list:
        for n, c in todo.items(): print(n, *c)
        return 0
    for f in ('amd_fidelityfx_upscaler_vk.so', 'trace', 'wine/trace.exe', 'wine/amd_fidelityfx_upscaler_vk.dll'):
        if not os.path.exists(os.path.join(build, f)): raise SystemExit('missing %s/%s: build with mingw-w64 and -DFSR4UP_AMD_DLL' % (build, f))
    os.makedirs(os.path.join(suite, 'amd'), exist_ok=True), os.makedirs(os.path.join(suite, 'port'), exist_ok=True)
    failed, frames_total = [], 0
    for name, case in todo.items():
        w, h, rw, rh, frames, cenv = case
        expect = cenv.get('_EXPECT', '0')
        racy = cenv.get('_RACY') == 1 or cenv.get('_RACY') == '1'
        cenv = {k: v for k, v in cenv.items() if not k.startswith('_')}
        args = [str(x) for x in (w, h, rw, rh, frames, 0)]
        env = {k: v for k, v in os.environ.items() if not k.startswith('TRACE_')}
        env.update(cenv, TRACE_DUMP_ALL='')
        key = amd_key(build, case)
        cache = os.path.join(suite, 'amd', name + '.json')
        amd = json.load(open(cache)) if os.path.exists(cache) and not a.refresh_amd else None
        amd_dir, amd_time = os.path.join(suite, 'amd', name), 0.0
        if not amd or amd.get('key') != key or (racy and not os.path.exists(os.path.join(amd_dir, 'trace.log'))):
            rc, amd_time = run([os.path.join(ROOT, 'harness', 'run_amd.sh'), name] + args,
                               dict(env, FSR4UP_BUILD=build, FSR4UP_RE=os.path.join(suite, 'amd')), ROOT, os.path.join(suite, 'amd', name + '.out'))
            ok, got = results_of(os.path.join(suite, 'amd', name + '.out'), frames, expect)
            if rc or not ok:
                print('%-52s AMD RUN FAILED (rc %d, frames %s): see %s.out' % (name, rc, got, os.path.join(suite, 'amd', name)))
                failed.append(name)
                continue
            amd = {'key': key, 'case': case, 'images': images(amd_dir, expect != '0'), 'results': api_results(amd_dir)}
            json.dump(amd, open(cache, 'w'), indent=1)
        port_dir = os.path.join(suite, 'port', name)
        shutil.rmtree(port_dir, ignore_errors=True)
        rc, port_time = run([os.path.join(build, 'trace'), os.path.join(build, 'amd_fidelityfx_upscaler_vk.so'), port_dir] + args,
                            env, ROOT, port_dir + '.out')
        ok, got = results_of(port_dir + '.out', frames, expect)
        have_log = os.path.exists(os.path.join(port_dir, 'trace.log'))
        mine = images(port_dir, expect != '0') if have_log else {}
        bad = sorted(k for k in set(amd['images']) | set(mine) if amd['images'].get(k) != mine.get(k) or mine.get(k) == 'ALL ZERO')
        mine_results = compared(api_results(port_dir)) if have_log else []
        theirs = compared(amd.get('results', []))
        if mine_results != theirs:
            diff = [(x, y) for x, y in zip(theirs, mine_results) if x != y][:2]
            bad.append('API results differ (%d vs %d lines), first: %s' % (len(theirs), len(mine_results), diff))
        racy_note = ''
        if racy and bad and not any(k.startswith('API') for k in bad):
            counts = {k: texels_differing(amd_dir, port_dir, k) for k in bad}
            if all(' out_upscaled' in k and n is not None and n <= RACY_TEXELS for k, n in counts.items()):
                racy_note = ' (racy: %s)' % ', '.join('%s %d texels' % (k.split()[0], n) for k, n in counts.items())
                bad = []
        made = sum(1 for k in amd['images'] if ' out_upscaled' in k)
        want = frames if expect == '0' else made
        if rc or not ok or bad or made != want:
            failed.append(name)
            harness = [k for k in bad if ' in_' in k]
            print('%-52s FAIL  rc %d, frames %s, %d of %d differ%s' % (name, rc, got, len(bad), len(amd['images']),
                  ' (harness inputs differ: %s)' % harness[:3] if harness else ''))
            for k in bad[:6]: print('      %s' % (k if k.startswith('API') else '%-40s amd %s  port %s' % (k, str(amd['images'].get(k))[:12], str(mine.get(k))[:12])))
            if made != want: print('      AMD made %d frames, expected %d' % (made, want))
        else:
            frames_total += made
            print('%-52s pass  %d frames %s (+%d inputs, %d API results)  [amd %s, port %.1fs]' % (
                name, made, 'bit-identical' if not racy_note else 'within AMD\'s own run-to-run spread' + racy_note, len(amd['images']) - made, len(theirs), 'cached' if not amd_time else '%.1fs' % amd_time, port_time))
            for d in (port_dir, amd_dir):
                if not a.keep and not (racy and d == amd_dir): shutil.rmtree(d, ignore_errors=True)
    # cases meant to differ must make different frames in AMD's DLL, or they test nothing
    def outs(n):
        f = os.path.join(suite, 'amd', n + '.json')
        return {k: v for k, v in json.load(open(f))['images'].items() if ' out_upscaled' in k} if os.path.exists(f) else None
    B = '1920x1080_quality_moving'
    groups = [['%dx%d_%s_%s' % (w, h, m, c) for c in ('still', 'moving')] for w, h in ((1280, 720), (1920, 1080), (3840, 2160)) for m in ('quality', 'performance')]
    groups += [[B, B + '_' + x] for x in ('reset3', 'jittercancel', 'nojitter', 'mvscale', 'displaymv', 'autoexposure', 'exposure_texture',
                                         'preexposure', 'nonlinear', 'srgb', 'pq', 'sharpen05', 'debugview')]
    # (FSR 4 makes the same frames with finite and infinite depth (traced): only inverted or not is compared)
    groups += [[B + '_depth_std_finite', B + '_depth_inv_finite'], [B + '_sharpen0', B + '_sharpen05', B + '_sharpen1']]
    for g in groups:
        if not all(n in todo for n in g): continue
        fr = [outs(n) for n in g]
        if any(f is None for f in fr): continue
        same = [(g[i], g[j]) for i in range(len(g)) for j in range(i + 1, len(g)) if fr[i] == fr[j]]
        if same:
            print('NOT SENSITIVE: AMD made the same frames for %s' % ', '.join('%s = %s' % p for p in same))
            failed.append('sensitivity')
    print('\n%d of %d cases passed (%d frames bit-identical, but for the racy debug view cases)' % (len(todo) - len([f for f in failed if f != 'sensitivity']), len(todo), frames_total))
    if failed: print('failed:', ' '.join(failed))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
