"""Carve every zlib-compressed SPIR-V module out of AMD's upscaler DLL (read as data; nothing is executed).

    python3 -I tools/carve.py <amd_fidelityfx_upscaler_vk.dll> <out dir>

Writes <out>/<index>_<offset>.spv for each distinct module and <out>/index.json (offset, compressed size, size, sha,
entry point, and every offset that held the same module)."""
import sys, os, json, hashlib, struct, zlib

MAGIC = b'\x03\x02\x23\x07'


def streams(d):
    """(offset, compressed size, module) of every zlib stream in d that inflates to SPIR-V"""
    i = 0
    n = len(d)
    while True:
        i = d.find(b'\x78\xda', i)
        if i < 0: return
        o = zlib.decompressobj()
        try:
            head = o.decompress(d[i:i + 1024], 8)
        except zlib.error:
            i += 1; continue
        if head[:4] != MAGIC:
            i += 1; continue
        o = zlib.decompressobj()
        out = o.decompress(d[i:])
        used = n - i - len(o.unused_data)
        yield i, used, out
        i += used


def entry_of(blob):
    w = struct.unpack('<%dI' % (len(blob) // 4), blob[:len(blob) // 4 * 4])
    p = 5
    while p < len(w):
        op, wc = w[p] & 0xffff, w[p] >> 16
        if wc == 0: break
        if op == 15:
            b = struct.pack('<%dI' % (wc - 3), *w[p + 3:p + wc])
            return b[:b.index(b'\0')].decode()
        p += wc
    return None


def main():
    dll, out = sys.argv[1], sys.argv[2]
    d = open(dll, 'rb').read()
    os.makedirs(out, exist_ok=True)
    mods, by_sha = [], {}
    for off, csize, blob in streams(d):
        sha = hashlib.sha256(blob).hexdigest()[:16]
        if sha in by_sha:
            by_sha[sha]['also'].append(off); continue
        m = {'index': len(mods), 'offset': off, 'csize': csize, 'size': len(blob), 'sha': sha, 'entry': entry_of(blob), 'also': []}
        by_sha[sha] = m; mods.append(m)
        open(os.path.join(out, '%04d_%08x.spv' % (m['index'], off)), 'wb').write(blob)
    json.dump(mods, open(os.path.join(out, 'index.json'), 'w'), indent=1)
    print(len(mods), 'distinct modules,', sum(1 + len(m['also']) for m in mods), 'streams')


if __name__ == '__main__':
    main()
