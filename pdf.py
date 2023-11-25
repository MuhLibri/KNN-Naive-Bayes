try:
    from typing import TypeAlias, Callable
except ImportError:
    from typing_extensions import TypeAlias, Callable
import math

class Statistics: ...
PDF: TypeAlias = Callable[[Statistics, float], float]

class Statistics:
    def __init__(self, data: list[float], prob: PDF):
        self.data = data
        self.mean = _mean(data)
        self.std = _std(data)
        self.len = len(data)
        self.min = min(data)
        self.max = max(data)
        self._prob = prob
    
    def probability(self, x: float) -> float:
        if self._prob(self, x) == 0.:
            print(f'Got zero while evaluating {self._prob.__name__}({x})')
        return self._prob(self, x)
    
    def __repr__(self) -> str:
        return f'<dist:{self._prob.__name__} mean:{self.mean} std:{self.std} len:{self.len}>'

def _mean(data: list[float]) -> float:
    return sum(data) / float(len(data))

def _std(data: list[float]) -> float:
    u = _mean(data)
    var = sum([(x - u)**2 for x in data]) / float(len(data) - 1)
    return math.sqrt(var)

def dist_nominal(stats: Statistics, x: float) -> float:
        freq_table: dict[float, int] = None
        try:
            freq_table: dict[float, int] = stats.freq
        except:
            freq_table = dict()
            for val in stats.data:
                if val not in freq_table: freq_table[val] = 0
                freq_table[val] += 1
            stats.freq = freq_table
        
        if x not in freq_table: return 0.
        return float(freq_table[x]) / float(stats.len)

def dist_uniform(stats: Statistics, x: float) -> float:
    return 1. / (stats.max - stats.min)
    
def dist_normal(stats: Statistics, x: float) -> float:
    pow = -.5 * ((x - stats.mean) / stats.std)**2
    return 1 / (stats.std * math.sqrt(2 * math.pi)) * math.exp(pow)
    
def dist_exp(stats: Statistics, x: float) -> float:
    lambd = 1 / stats.mean
    return lambd * math.exp(-x * lambd)
