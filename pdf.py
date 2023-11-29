try:
    from typing import TypeAlias, Callable
except ImportError:
    from typing_extensions import TypeAlias, Callable
import math, struct

class Statistics: ...
PDF: TypeAlias = Callable[[Statistics, float], float]

class Statistics:
    def __init__(self, data: list[float], prob: PDF):
        # self.data = data
        self.mean = _mean(data) if len(data) > 0 else None
        self.std = _std(data) if len(data) > 0 else None
        self.len = len(data) if len(data) > 0 else None
        self.min = min(data) if len(data) > 0 else None
        self.max = max(data) if len(data) > 0 else None
        self.zero_count = len([d for d in data if d == 0.]) if len(data) > 0 else None
        self.one_count = len([d for d in data if d == 1.]) if len(data) > 0 else None
        self._prob = prob
    
    @classmethod
    def from_bytes(cls, raw: bytes):
        data = struct.unpack('ddddlllc', raw)

        s = Statistics([], None)
        s.mean = data[0]
        s.std = data[1]
        s.min = data[2]
        s.max = data[3]
        s.len = data[4]
        s.zero_count = data[5]
        s.one_count = data[6]
        s._prob = _B2PDF_MAP[data[7]]

        return s

    def probability(self, x: float) -> float:
        if self._prob(self, x) == 0.:
            print(f'Got zero while evaluating {self._prob.__name__}({x})')
        return self._prob(self, x)

    def __bytes__(self) -> str:
        return struct.pack('ddddlllc',
            self.mean,
            self.std,
            self.min,
            self.max,
            self.len,
            self.zero_count,
            self.one_count,
            _PDF2B_MAP[self._prob.__name__],
        )
    
    def __str__(self) -> str:
        return f'<dist:{self._prob.__name__} mean:{self.mean} std:{self.std} len:{self.len} range:{self.min}-{self.max} 0:{self.zero_count} 1:{self.one_count}>'
    
    def __repr__(self) -> str:
        return str(self)

def _mean(data: list[float]) -> float:
    return sum(data) / float(len(data))

def _std(data: list[float]) -> float:
    u = _mean(data)
    var = sum([(x - u)**2 for x in data]) / float(len(data) - 1)
    return math.sqrt(var)

def dist_nominal(stats: Statistics, x: float) -> float:
    count = \
        stats.zero_count if x == 0. else \
        stats.one_count if x == 1. else \
        None
    if count is None: raise ValueError(f'Got value {x} not in [0, 1]')
    return count / stats.len

def dist_uniform(stats: Statistics, x: float) -> float:
    return 1. / (stats.max - stats.min)
    
def dist_normal(stats: Statistics, x: float) -> float:
    pow = -.5 * ((x - stats.mean) / stats.std)**2
    return 1 / (stats.std * math.sqrt(2 * math.pi)) * math.exp(pow)
    
def dist_exp(stats: Statistics, x: float) -> float:
    lambd = 1 / stats.mean
    return lambd * math.exp(-x * lambd)

_B2PDF_MAP = {
    b'\x01': dist_nominal,
    b'\x02': dist_uniform,
    b'\x03': dist_normal,
    b'\x04': dist_exp
}
_PDF2B_MAP: dict[str, bytes] = dict()
for b, f in _B2PDF_MAP.items():
    _PDF2B_MAP[f.__name__] = b
