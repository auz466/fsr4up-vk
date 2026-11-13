"""Reading AMD's upscaler DLL as data: sections, address conversion, and the shader-record table."""
import struct, json, os
DLL = os.environ.get('FSR4UP_AMD_DLL', os.path.expanduser('~/Games/Hookshot/fsr4-deps/amd/sdk-2.4.0/Kits/FidelityFX/signedbin/amd_fidelityfx_upscaler_vk.dll'))


class Dll:
    def __init__(self, path=DLL):
        self.d = d = open(path, 'rb').read()
        pe = struct.unpack_from('<I', d, 0x3c)[0]
        self.base = struct.unpack_from('<Q', d, pe + 24 + 24)[0]
        ns = struct.unpack_from('<H', d, pe + 6)[0]; opt = struct.unpack_from('<H', d, pe + 20)[0]; sh = pe + 24 + opt
        self.secs = []
        for i in range(ns):
            n, vs, va, rs, ra = struct.unpack_from('<8sIIII', d, sh + 40 * i)
            self.secs.append((n.rstrip(b'\0').decode(), va, vs, ra, rs))

    def off2va(self, o):
        for n, va, vs, ra, rs in self.secs:
            if ra <= o < ra + rs: return self.base + va + o - ra

    def va2off(self, v):
        for n, va, vs, ra, rs in self.secs:
            if self.base + va <= v < self.base + va + max(vs, rs): return v - self.base - va + ra

    def q(self, o): return struct.unpack_from('<Q', self.d, o)[0]
    def u32(self, o): return struct.unpack_from('<I', self.d, o)[0]

    def cstr(self, va):
        o = self.va2off(va)
        if o is None: return None
        e = self.d.index(b'\0', o)
        s = self.d[o:e]
        return s.decode() if s and all(32 <= c < 127 for c in s) else None
