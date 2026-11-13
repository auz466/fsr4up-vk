"""A small SPIR-V reader/writer that knows which operand words are ids (from the Khronos grammar), for rewriting modules."""
import json, struct, os

GRAMMAR_NAME = os.path.join('spirv', 'unified1', 'spirv.core.grammar.json')


def grammar_path():
    """The Khronos SPIR-V core grammar (from SPIRV-Headers): $SPIRV_GRAMMAR, else under $SPIRV_HEADERS_DIR, the
    CMAKE_PREFIX_PATH / XDG_DATA_DIRS prefixes, /usr/local and /usr (include/ or share/)"""
    if os.environ.get('SPIRV_GRAMMAR'): return os.environ['SPIRV_GRAMMAR']
    roots = [os.environ.get('SPIRV_HEADERS_DIR', '')]
    for var in ('CMAKE_PREFIX_PATH', 'XDG_DATA_DIRS'):
        roots += os.environ.get(var, '').split(os.pathsep)
    roots += ['/usr/local', '/usr']
    for r in filter(None, roots):
        for sub in ('include', 'share', ''):
            f = os.path.join(r, sub, GRAMMAR_NAME)
            if os.path.isfile(f): return f
    raise SystemExit('the SPIR-V grammar (%s, from SPIRV-Headers) was not found: install spirv-headers, or set SPIRV_GRAMMAR' % GRAMMAR_NAME)
ID_KINDS = {'IdResultType', 'IdResult', 'IdRef', 'IdScope', 'IdMemorySemantics'}
_ops = None


def grammar():
    global _ops
    if _ops is None:
        g = json.load(open(grammar_path()))
        _ops = {}
        for ins in g['instructions']:
            _ops.setdefault(ins['opcode'], ins)   # (aliases share an opcode: the first is as good as any)
    return _ops


class Ins:
    __slots__ = ('op', 'w')

    def __init__(self, op, w):
        self.op, self.w = op, list(w)   # w: the operand words, without the first (opcode, count) word

    def __repr__(self):
        return '%s %s' % (grammar().get(self.op, {}).get('opname', self.op), self.w)

    def layout(self):
        """(index of result type or None, index of result id or None, [indexes of every other id operand])"""
        spec = grammar().get(self.op)
        if spec is None: return None, None, []
        rt = rid = None
        ids = []
        i = 0
        w = self.w
        for o in spec.get('operands', []):
            if i >= len(w): break
            kind, q = o['kind'], o.get('quantifier', '')
            if kind == 'IdResultType': rt = i; i += 1
            elif kind == 'IdResult': rid = i; i += 1
            elif kind in ID_KINDS:
                if q == '*':
                    ids += range(i, len(w)); i = len(w)
                else:
                    ids.append(i); i += 1
            elif kind == 'LiteralString':
                while i < len(w) and (w[i] >> 24) and (w[i] >> 16 & 255) and (w[i] >> 8 & 255) and (w[i] & 255): i += 1
                i += 1
            elif kind == 'PairLiteralIntegerIdRef':
                while i + 1 < len(w): ids.append(i + 1); i += 2
            elif kind == 'PairIdRefLiteralInteger':
                while i + 1 < len(w): ids.append(i); i += 2
            elif kind == 'PairIdRefIdRef':
                while i + 1 < len(w): ids += [i, i + 1]; i += 2
            elif kind == 'ImageOperands':
                i += 1
                ids += range(i, len(w)); i = len(w)   # (every operand of an image operand is an id)
            elif kind in ('LiteralContextDependentNumber', 'LiteralSpecConstantOpInteger'):
                if kind == 'LiteralSpecConstantOpInteger': ids += range(i + 1, len(w))
                i = len(w)
            elif kind in ('Decoration', 'ExecutionMode', 'MemoryAccess', 'LoopControl'):
                i = len(w)   # (their parameters are literals in everything this tool meets)
            else:
                if q == '*': i = len(w)
                else: i += 1
        return rt, rid, ids

    def all_ids(self):
        rt, rid, ids = self.layout()
        return ([rt] if rt is not None else []) + ids   # (uses only: not the result)


class Module:
    def __init__(self, blob):
        w = struct.unpack('<%dI' % (len(blob) // 4), blob)
        self.header = list(w[:5])
        self.ins = []
        i = 5
        while i < len(w):
            wc = w[i] >> 16
            self.ins.append(Ins(w[i] & 0xffff, w[i + 1:i + wc]))
            i += wc

    @property
    def bound(self): return self.header[3]

    def new_id(self):
        self.header[3] += 1
        return self.header[3] - 1

    def blob(self):
        out = list(self.header)
        for x in self.ins: out += [((len(x.w) + 1) << 16) | x.op] + x.w
        return struct.pack('<%dI' % len(out), *out)

    def first_function(self):
        return next(i for i, x in enumerate(self.ins) if x.op == 54)

    def replace_id(self, old, new):
        for x in self.ins:
            for k in x.all_ids():
                if x.w[k] == old: x.w[k] = new

    def dedupe_types(self):
        """Removes a non-aggregate type declared twice (a rewrite can leave one), pointing its uses at the first."""
        while True:
            seen, dup = {}, None
            for n, x in enumerate(self.ins):
                if x.op in (19, 20, 21, 22, 23, 24, 25, 26, 27, 32, 33):
                    key = (x.op, tuple(x.w[1:]))
                    if key in seen: dup = (n, x.w[0], seen[key]); break
                    seen[key] = x.w[0]
                if x.op == 54: break
            if not dup: return
            n, old, new = dup
            del self.ins[n]
            self.ins = [x for x in self.ins if not (x.op in (5, 71, 72, 6) and x.w[0] == old)]   # its name and decorations
            self.replace_id(old, new)


def fnv1a64(blob):
    h = 0xcbf29ce484222325
    for b in blob: h = ((h ^ b) * 0x100000001b3) & 0xffffffffffffffff
    return h
