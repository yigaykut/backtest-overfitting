"""Was that a result, or was it the best of many tries?

Two measures from Bailey and Lopez de Prado, plus the frequency check
that caught my own mistake. numpy and the standard library, nothing
else.

DEFLATED SHARPE RATIO
---------------------
Run N strategies with no skill at all and the best of them still shows
a positive Sharpe. How positive depends on N and on how much the trials
differ from each other. The deflated Sharpe works out that bar and asks
whether the observed Sharpe clears it.

The bar is sqrt(V) * ((1-g)*Z(1-1/N) + g*Z(1-1/(N*e))), where V is the
variance of the trial Sharpes and g is Euler's constant. So a grid
where everything lands in the same place gets a low bar, and one that
sprays gets a high one.

Skew and kurtosis go into the probability. Returns from real strategies
are not normal, and ignoring that reads high.

PROBABILITY OF BACKTEST OVERFITTING (CSCV)
------------------------------------------
Split the sample into S pieces, take every way of using half for
training and half for testing, pick the training winner in each, and
see where it lands out of sample. The share of splits where it falls
below the median is the PBO.

A PBO near 0.5 says "pick whatever won in training" is a coin flip.
Near 0 says the selection is doing something.

SHARPE BY FREQUENCY
-------------------
Annualising a Sharpe assumes the observations are independent. If the
equity curve was built by filling values forward between updates, they
are not, and the number comes out several times too high. See
`sharpe_by_frequency`.

ALL THREE ARE DIAGNOSTICS. They do not improve a strategy. They tell
you how to read the number it produced.

IMPORTANT: pass every trial you ran, losers included. Handing it only
the winners shrinks N, lowers the bar, and hides the exact bias you
are trying to measure.
"""
from __future__ import annotations

import itertools
import math

import numpy as np


def _normal_ppf(p: float) -> float:
    """Inverse normal CDF.

    Acklam's approximation, so scipy stays out of the dependency list.
    Absolute error around 1e-9, far more than this needs.
    """
    if not 0.0 < p < 1.0:
        return float("nan")
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    pl, ph = 0.02425, 1 - 0.02425
    if p < pl:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q
                + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > ph:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q
                 + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q, r = p - 0.5, (p - 0.5) ** 2
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r
            + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r
                            + b[4]) * r + 1)


def _normal_cdf(z: float) -> float:
    if not np.isfinite(z):
        return float("nan")
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _usable(v: np.ndarray) -> "tuple[np.ndarray, float] | None":
    """Finite values and their standard deviation, or None.

    The threshold on sd is relative, not `sd > 0`. A constant series
    does not give exactly zero: summing identical floats leaves a
    residual around 1e-19, which slips through and makes the Sharpe
    astronomically large but still finite.
    """
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 3:
        return None
    sd = float(v.std(ddof=1))
    if not sd > 1e-12 * max(1.0, float(np.abs(v).mean())):
        return None
    return v, sd


def sharpe(returns, periods_per_year: float = 365.0) -> float:
    """Annualised Sharpe. Risk-free rate is taken as zero."""
    u = _usable(returns)
    if u is None:
        return float("nan")
    v, sd = u
    return float(v.mean() / sd * math.sqrt(periods_per_year))


def probabilistic_sharpe(returns, threshold: float = 0.0,
                         periods_per_year: float = 365.0) -> float:
    """Probability that the true Sharpe is above `threshold`.

    Corrected for skew and kurtosis. Without that correction the
    probability reads high on a non-normal series: negative skew and
    fat tails both widen the standard error of a Sharpe.
    """
    u = _usable(returns)
    if u is None:
        return float("nan")
    v, sd = u
    n = len(v)
    sr = v.mean() / sd                                  # per period
    sr_threshold = threshold / math.sqrt(periods_per_year)
    z = (v - v.mean()) / sd
    skew = float((z ** 3).mean())
    kurt = float((z ** 4).mean())
    denom = math.sqrt(max(1e-12,
                          1 - skew * sr + (kurt - 1) / 4 * sr ** 2))
    return _normal_cdf((sr - sr_threshold) * math.sqrt(n - 1) / denom)


def deflated_sharpe(trials, selected=None,
                    periods_per_year: float = 365.0) -> dict:
    """How real the best of a set of trials is.

    `trials` is the return series of EVERY configuration you ran, as a
    dict of name -> returns or a list. Leaving any out makes the answer
    optimistic: a smaller N means a lower bar, so handing it only the
    winners hides the bias this is meant to measure.
    """
    if isinstance(trials, dict):
        names, series = list(trials.keys()), list(trials.values())
    else:
        names, series = list(range(len(trials))), list(trials)
    sr = np.array([sharpe(x, periods_per_year) for x in series], dtype=float)
    ok = np.isfinite(sr)
    if ok.sum() < 2:
        return {"ok": False, "reason": "need at least two usable trials"}
    n_trials = int(ok.sum())
    if selected is None:
        i = int(np.nanargmax(np.where(ok, sr, -np.inf)))
    else:
        i = (names.index(selected) if isinstance(selected, str)
             else int(selected))

    # The spread of the trial Sharpes sets the bar: if every
    # configuration lands in the same place the search bought little,
    # and the bar is low.
    var_sr = float(np.var(sr[ok], ddof=1))
    g = 0.5772156649015329
    z1 = _normal_ppf(max(1e-12, min(1 - 1e-12, 1.0 - 1.0 / n_trials)))
    z2 = _normal_ppf(max(1e-12, min(1 - 1e-12,
                                    1.0 - 1.0 / (n_trials * math.e))))
    expected_max = math.sqrt(max(0.0, var_sr)) * ((1 - g) * z1 + g * z2)
    return {
        "ok": True,
        "selected": names[i],
        "sharpe": float(sr[i]),
        "trials": n_trials,
        "sharpe_spread": math.sqrt(max(0.0, var_sr)),
        # What the best of N skill-less trials would be expected to show.
        "chance_bar": float(expected_max),
        "deflated": float(probabilistic_sharpe(series[i], expected_max,
                                               periods_per_year)),
        "vs_zero": float(probabilistic_sharpe(series[i], 0.0,
                                              periods_per_year)),
    }


def pbo(trials, splits: int = 10) -> dict:
    """Probability of backtest overfitting (CSCV).

    The sample is cut into `splits` pieces. For every way of using half
    for training and half for testing, the training winner is picked
    and its rank out of sample recorded. The share of splits where it
    lands below the median is the PBO.
    """
    series = list(trials.values()) if isinstance(trials, dict) else list(trials)
    if len(series) < 2:
        return {"ok": False, "reason": "need at least two trials"}
    n = min(len(x) for x in series)
    if splits % 2 or n < splits * 4:
        return {"ok": False,
                "reason": f"{n} observations over {splits} splits is too few"}
    M = np.array([np.asarray(x, dtype=float)[:n] for x in series])
    cuts = np.array_split(np.arange(n), splits)

    logits = []
    for train in itertools.combinations(range(splits), splits // 2):
        test = [j for j in range(splits) if j not in train]
        i_tr = np.concatenate([cuts[j] for j in train])
        i_te = np.concatenate([cuts[j] for j in test])
        s_tr = np.array([sharpe(M[k, i_tr]) for k in range(len(M))])
        s_te = np.array([sharpe(M[k, i_te]) for k in range(len(M))])
        if not np.isfinite(s_tr).any() or not np.isfinite(s_te).any():
            continue
        best = int(np.nanargmax(np.where(np.isfinite(s_tr), s_tr, -np.inf)))
        ok = np.isfinite(s_te)
        if ok.sum() < 2 or not np.isfinite(s_te[best]):
            continue
        # Where the pick sits out of sample, as a percentile. 0.5 is the
        # median.
        w = float((s_te[ok] < s_te[best]).sum()) / float(ok.sum() - 1 or 1)
        w = min(max(w, 1e-9), 1 - 1e-9)
        logits.append(math.log(w / (1 - w)))
    if not logits:
        return {"ok": False, "reason": "no split could be measured"}
    m = np.array(logits)
    return {
        "ok": True,
        "pbo": float((m <= 0).mean()),
        "splits": len(m),
        "median_logit": float(np.median(m)),
    }


def sharpe_by_frequency(returns, blocks=(1, 5, 21, 35),
                        periods_per_year: float = 365.0) -> dict:
    """Is the annualised Sharpe an artifact of a smoothed curve?

    Annualising assumes the observations are independent. An equity
    curve assembled by filling values forward between updates breaks
    that badly, and the Sharpe comes out several times too high.

    This compounds the series into non-overlapping blocks of each given
    length and reports the Sharpe of each. A real daily series gives
    roughly the same number at every block size. A forward-filled one
    gives a large number at block 1 that collapses as the blocks grow,
    and that collapse is the measure of how much was borrowed.

    `lag1` is the first-order autocorrelation. Near zero is what a
    genuine return series looks like. Anything above about 0.3 means
    the block-1 Sharpe cannot be trusted.
    """
    u = _usable(returns)
    if u is None:
        return {"ok": False, "reason": "series not usable"}
    v, _ = u
    if len(v) < 30:
        return {"ok": False, "reason": f"{len(v)} observations is too few"}
    lag1 = float(np.corrcoef(v[:-1], v[1:])[0, 1]) if len(v) > 2 else float("nan")

    out = {}
    for b in blocks:
        b = int(b)
        if b < 1 or len(v) // b < 10:
            continue
        cut = (len(v) // b) * b
        agg = np.prod(1.0 + v[:cut].reshape(-1, b), axis=1) - 1.0
        out[b] = float(sharpe(agg, periods_per_year / b))
    if not out:
        return {"ok": False, "reason": "no block size had enough data"}

    vals = [x for x in out.values() if np.isfinite(x)]
    # Effective sample size under AR(1): n * (1-rho) / (1+rho).
    eff = (len(v) * (1 - lag1) / (1 + lag1)
           if np.isfinite(lag1) and lag1 > -1 else float("nan"))
    return {
        "ok": True,
        "lag1": lag1,
        "n": len(v),
        "effective_n": float(eff),
        "by_block": out,
        # How much the shortest block borrows from the longest.
        "inflation": (float(max(vals) / min(vals))
                      if vals and min(vals) > 0 else float("nan")),
        "suspect": bool(np.isfinite(lag1) and lag1 > 0.3),
    }
