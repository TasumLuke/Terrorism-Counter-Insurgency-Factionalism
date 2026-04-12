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


#NEGATIVE BINOMIAL

def negbin(y, X, var_names=None):
    n, k = X.shape
    if var_names is None:
        var_names = [f"x{i}" for i in range(k)]

    mod = sm.NegativeBinomial(y, X)
    try:
        res = mod.fit(disp=0, maxiter=500)
    except Exception as e:
        print(f"    negbin failed: {e}")
        return None

    # compare against poisson
    pois = sm.Poisson(y, X).fit(disp=0, maxiter=200)
    lr = 2 * (res.llf - pois.llf)

    out = {"n": n, "llf": res.llf, "aic": res.aic, "lr_poisson": lr, "vars": {}}
    for i, v in enumerate(var_names):
        out["vars"][v] = {
            "b": res.params[i], "se": res.bse[i],
            "t": abs(res.params[i] / res.bse[i]) if res.bse[i] > 0 else 0.0,
            "p": res.pvalues[i],
        }
    return out


def print_reg(r, title=""):
    if r is None:
        print(f"  {title}: failed")
        return
    print(f"  {title}  N={r['n']}  LL={r['llf']:.1f}  AIC={r['aic']:.1f}  LR={r['lr_poisson']:.1f}")
    for v, d in r["vars"].items():
        print(f"    {v:18s} {d['b']:8.3f} ({d['se']:.3f})  t={d['t']:.2f} {stars(d['p'])}")


#OLS WITH MULTIPLE SE

def ols_robust(y, X, var_names=None, maxlag=2):
    n, k = X.shape
    if var_names is None:
        var_names = [f"x{i}" for i in range(k)]

    ln_y = np.log(y + 1)
    hc3 = sm.OLS(ln_y, X).fit(cov_type="HC3")
    nw = sm.OLS(ln_y, X).fit(cov_type="HAC", cov_kwds={"maxlags": maxlag})
    plain = sm.OLS(ln_y, X).fit()

    out = {"n": n, "r2": hc3.rsquared, "vars": {}}
    for i, v in enumerate(var_names):
        out["vars"][v] = {
            "b": hc3.params[i],
            "se_hc3": hc3.bse[i], "se_nw": nw.bse[i], "se_ols": plain.bse[i],
            "p_hc3": hc3.pvalues[i],
        }
    return out


def print_ols(r, title=""):
    print(f"  {title}  N={r['n']}  R2={r['r2']:.3f}")
    for v, d in r["vars"].items():
        print(f"    {v:18s} {d['b']:8.3f}  HC3=({d['se_hc3']:.3f}) NW=({d['se_nw']:.3f}) {stars(d['p_hc3'])}")


#INTERRUPTED TIME SERIES

def its(years, outcome, breakpoint, maxlag=2):
    ylist = list(years)
    if breakpoint not in ylist:
        raise ValueError(f"{breakpoint} not in data")

    n = len(ylist)
    bp = ylist.index(breakpoint)
    t = np.arange(n, dtype=float)
    post = (t >= bp).astype(float)
    post_time = post * (t - bp)

    X = np.column_stack([np.ones(n), t, post, post_time])
    y = np.array(outcome, dtype=float)
    fit = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": maxlag})

    names = ["intercept", "pre_trend", "level_shift", "post_trend"]
    out = {"r2": fit.rsquared, "n": n, "break": breakpoint, "params": {}}
    for i, nm in enumerate(names):
        out["params"][nm] = {
            "b": fit.params[i], "se": fit.bse[i],
            "t": abs(fit.params[i] / fit.bse[i]) if fit.bse[i] > 0 else np.nan,
            "p": fit.pvalues[i],
        }
    return out


def print_its(r, title=""):
    print(f"  {title}  break={r['break']}  R2={r['r2']:.3f}  N={r['n']}")
    for nm, d in r["params"].items():
        print(f"    {nm:16s} {d['b']:10.2f} ({d['se']:.2f}) t={d['t']:.2f} {stars(d['p'])}")


#MEDIATION

def mediate(y, mediator, treatment, nboot=2000, seed=42):
    ln_y = np.log(y + 1)
    n = len(y)

    # stage 1: treatment -> mediator
    r1 = sm.OLS(mediator, sm.add_constant(treatment)).fit()
    a_path = r1.params[1]

    # stage 2: mediator + treatment -> outcome
    X2 = sm.add_constant(np.column_stack([mediator, treatment]))
    r2 = sm.OLS(ln_y, X2).fit()
    b_path = r2.params[1]
    direct = r2.params[2]

    # total
    r3 = sm.OLS(ln_y, sm.add_constant(treatment)).fit()
    total = r3.params[1]

    acme = a_path * b_path
    prop = acme / total if abs(total) > 1e-10 else np.nan

    # bootstrap
    rng = np.random.default_rng(seed)
    b_acme = np.zeros(nboot)
    b_direct = np.zeros(nboot)
    b_total = np.zeros(nboot)

    for i in range(nboot):
        ix = rng.integers(0, n, n)
        yb, mb, tb = y[ix], mediator[ix], treatment[ix]
        lyb = np.log(yb + 1)
        try:
            s1 = sm.OLS(mb, sm.add_constant(tb)).fit()
            s2 = sm.OLS(lyb, sm.add_constant(np.column_stack([mb, tb]))).fit()
            s3 = sm.OLS(lyb, sm.add_constant(tb)).fit()
            b_acme[i] = s1.params[1] * s2.params[1]
            b_direct[i] = s2.params[2]
            b_total[i] = s3.params[1]
        except:
            b_acme[i] = b_direct[i] = b_total[i] = np.nan

    ok = ~np.isnan(b_acme)
    pci = lambda arr: (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))

    return {
        "a": a_path, "b": b_path,
        "acme": acme, "acme_ci": pci(b_acme[ok]),
        "direct": direct, "direct_ci": pci(b_direct[ok]),
        "total": total, "total_ci": pci(b_total[ok]),
        "prop": prop,
        "prop_ci": pci(b_acme[ok] / b_total[ok]),
        "nboot": int(ok.sum()),
    }


def print_med(r):
    print(f"  a path:  {r['a']:.3f}")
    print(f"  b path:  {r['b']:.3f}")
    print(f"  ACME:    {r['acme']:.3f}  [{r['acme_ci'][0]:.3f}, {r['acme_ci'][1]:.3f}]")
    print(f"  direct:  {r['direct']:.3f}  [{r['direct_ci'][0]:.3f}, {r['direct_ci'][1]:.3f}]")
    print(f"  total:   {r['total']:.3f}  [{r['total_ci'][0]:.3f}, {r['total_ci'][1]:.3f}]")
    print(f"  prop:    {r['prop']:.3f}  [{r['prop_ci'][0]:.3f}, {r['prop_ci'][1]:.3f}]")
    print(f"  valid boots: {r['nboot']}")
