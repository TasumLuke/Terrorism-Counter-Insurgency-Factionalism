import math
import numpy as np
import statsmodels.api as sm
from scipy import stats

def herfindahl(shares):
    return sum(s**2 for s in shares)


def frag_score(shares):
    # complement of HHI, 0 = unified, approaching 1 = fragmented
    return 1.0 - herfindahl(shares)


def internecine_rate(inter_deaths, total_deaths):
    # laplace smoothed to handle zero denominators
    return inter_deaths / (total_deaths + 1)


def minmax_norm(x):
    lo, hi = min(x), max(x)
    if hi - lo < 1e-12:
        return [0.0] * len(x)
    return [(v - lo) / (hi - lo) for v in x]


def cohesion_index(ofs_raw, ivr_raw, cpc_raw, w):
    a, b, g = w
    if abs(a + b + g - 1.0) > 1e-6:
        raise ValueError(f"weights dont sum to 1: {a+b+g}")

    ofs_n = minmax_norm(ofs_raw)
    ivr_n = minmax_norm(ivr_raw)
    cpc_n = minmax_norm(cpc_raw)

    rci = []
    for i in range(len(ofs_n)):
        rci.append(1.0 - (a * ofs_n[i] + b * ivr_n[i] + g * cpc_n[i]))
    return rci
def lag(series, groups=None):
    # if groups is None, treat as single group
    if groups is None:
        return [None] + list(series[:-1])
    out = [None] * len(series)
    prev = {}
    for i in range(len(series)):
        g = groups[i]
        out[i] = prev.get(g)
        prev[g] = series[i]
    return out


def pct_drop(before, after):
    return (before - after) / before * 100.0


def marginal_pct(coef, sd):
    return (math.exp(abs(coef) * sd) - 1) * 100.0


def stars(p):
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.10: return "*"
    return ""
