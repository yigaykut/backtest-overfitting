"""Checks against known behaviour, not against known values.

Every measure here returns a probability between 0 and 1, which always
looks plausible. So the tests ask what should happen rather than what
the number should be: what comes out when there is no skill, what
happens as trials are added, what a fat left tail does, and what a
forward-filled curve does to an annualised Sharpe.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import overfit as of     # noqa: E402

fails = 0


def check(name, cond, extra=""):
    global fails
    ok = bool(cond)
    if not ok:
        fails += 1
    print(f"  {'OK  ' if ok else 'FAIL'}  {name}"
          + (f"  {extra}" if extra else ""))


print()
print("=" * 66)
print("1) INVERSE NORMAL - no scipy, so is the approximation good")
print("=" * 66)
for p, want in ((0.5, 0.0), (0.975, 1.959964), (0.99, 2.326348),
                (0.025, -1.959964), (0.001, -3.090232)):
    got = of._normal_ppf(p)
    check(f"ppf({p}) = {want:+.4f}", abs(got - want) < 1e-4, f"{got:+.6f}")
check("cdf(0) = 0.5", abs(of._normal_cdf(0.0) - 0.5) < 1e-12)
check("cdf(1.96) ~ 0.975", abs(of._normal_cdf(1.959964) - 0.975) < 1e-5)

print()
print("=" * 66)
print("2) SHARPE AND PROBABILISTIC SHARPE")
print("=" * 66)
rs = np.random.default_rng(7)
daily = rs.normal(0.001, 0.01, 4000)
s = of.sharpe(daily)
check("Sharpe in a sane range", 1.0 < s < 3.0, f"{s:.2f}")
# A constant series used to come back astronomically large but finite:
# the mean of identical floats leaves a ~1e-19 residual and `sd > 0`
# lets it through. The guard is relative now.
check("constant series gives nan",
      not np.isfinite(of.sharpe(np.full(100, 0.001))))
check("series too short gives nan",
      not np.isfinite(of.sharpe(np.array([0.1, 0.2]))))

check("strong series clears zero",
      of.probabilistic_sharpe(daily, 0.0) > 0.99,
      f"{of.probabilistic_sharpe(daily, 0.0):.3f}")
# One draw can land at 0.1 or 0.9. The mean over many is the check;
# testing a single draw against a narrow band tests the seed.
empty = [of.probabilistic_sharpe(rs.normal(0.0, 0.01, 1500), 0.0)
         for _ in range(200)]
check("no-skill series averages ~0.5",
      0.45 < float(np.mean(empty)) < 0.55, f"{np.mean(empty):.3f}")
check("no-skill series spreads widely",
      float(np.std(empty)) > 0.20, f"sd {np.std(empty):.3f}")
check("raising the threshold lowers the probability",
      of.probabilistic_sharpe(daily, 3.0) < of.probabilistic_sharpe(daily, 0.0))

# Same mean and spread, negative skew, lower probability. A weak signal
# on purpose: on a strong one both probabilities pin at 1.0000 and the
# test checks nothing.
clean = rs.normal(0.0004, 0.01, 1200)
skewed = clean.copy()
skewed[:25] -= 0.05
skewed = (skewed - skewed.mean()) / skewed.std() * clean.std() + clean.mean()
pc = of.probabilistic_sharpe(clean, 0.0)
ps = of.probabilistic_sharpe(skewed, 0.0)
check("test is not vacuous (probabilities off the ceiling)", pc < 0.999,
      f"clean {pc:.4f}")
check("negative skew lowers the probability", ps < pc,
      f"skewed {ps:.4f} vs clean {pc:.4f}")

print()
print("=" * 66)
print("3) DEFLATED SHARPE - the actual job: catching search bias")
print("=" * 66)
# No skill anywhere. The best of the pile looks good by luck; the
# deflated probability should not.
noise = {f"k{i}": rs.normal(0.0, 0.01, 1500) for i in range(50)}
d = of.deflated_sharpe(noise)
check("best of a skill-less pile still shows a positive Sharpe",
      d["sharpe"] > 0, f"{d['sharpe']:.2f}")
check("chance bar is positive and close to it",
      d["chance_bar"] > 0,
      f"bar {d['chance_bar']:.2f} vs observed {d['sharpe']:.2f}")
check("NO SKILL -> low deflated probability",
      d["deflated"] < 0.60, f"{d['deflated']:.3f}")
check("while against zero it looks decisive (the trap)",
      d["vs_zero"] > d["deflated"],
      f"{d['vs_zero']:.3f} vs {d['deflated']:.3f}")

mixed = dict(noise)
mixed["real"] = rs.normal(0.0015, 0.01, 1500)
d2 = of.deflated_sharpe(mixed)
check("the real edge gets selected", d2["selected"] == "real",
      str(d2["selected"]))
check("SKILL -> high deflated probability",
      d2["deflated"] > 0.90, f"{d2['deflated']:.3f}")

# More trials must raise the bar. That is the cost of searching.
few = of.deflated_sharpe({k: noise[k] for k in list(noise)[:6]})
check("more trials -> higher chance bar",
      d["chance_bar"] > few["chance_bar"],
      f"50 trials {d['chance_bar']:.2f} vs 6 trials {few['chance_bar']:.2f}")
check("trial count is reported", d["trials"] == 50, str(d["trials"]))

print()
print("=" * 66)
print("4) PBO - does picking the training winner pick the future one")
print("=" * 66)
p1 = of.pbo(noise, splits=8)
check("pure noise gives PBO ~0.5", 0.25 < p1["pbo"] < 0.75,
      f"{p1['pbo']:.3f}")
check("split count is reported", p1["splits"] > 10, str(p1["splits"]))

robust = {f"k{i}": rs.normal(0.0, 0.01, 1500) for i in range(20)}
robust["always_good"] = rs.normal(0.004, 0.01, 1500)
p2 = of.pbo(robust, splits=8)
check("a config that wins everywhere gives low PBO", p2["pbo"] < 0.25,
      f"{p2['pbo']:.3f}")
check("too few observations is refused",
      of.pbo({"a": np.zeros(20), "b": np.ones(20)}, splits=10).get("ok")
      is False)
check("a single config is refused", of.pbo({"a": daily}).get("ok") is False)

print()
print("=" * 66)
print("5) SHARPE BY FREQUENCY - the check that caught the real error")
print("=" * 66)
# An honest daily series: the annualised Sharpe should hold up at every
# block size, because the observations really are independent.
honest = rs.normal(0.0012, 0.012, 2055)
h = of.sharpe_by_frequency(honest)
check("honest series: lag-1 near zero", abs(h["lag1"]) < 0.08,
      f"{h['lag1']:+.3f}")
check("honest series is not flagged", h["suspect"] is False)
check("honest series: Sharpe holds across block sizes",
      h["inflation"] < 1.8,
      "  ".join(f"{b}d {v:.2f}" for b, v in h["by_block"].items()))

# The artifact: a book updated only at period close, forward-filled
# between. Same realised period returns, 35 copies of each day.
period = rs.normal(0.035, 0.07, 60)
filled = np.repeat((1.0 + period) ** (1.0 / 35) - 1.0, 35)
f = of.sharpe_by_frequency(filled, blocks=(1, 5, 35))
check("forward-filled series: lag-1 is high", f["lag1"] > 0.3,
      f"{f['lag1']:+.3f}")
check("forward-filled series IS flagged", f["suspect"] is True)
check("forward-filled series: block-1 Sharpe is inflated",
      f["inflation"] > 2.0,
      "  ".join(f"{b}d {v:.2f}" for b, v in f["by_block"].items()))
check("effective sample is far below the row count",
      f["effective_n"] < 0.25 * f["n"],
      f"{f['effective_n']:.0f} of {f['n']}")
check("short series is refused",
      of.sharpe_by_frequency(rs.normal(0, 0.01, 20)).get("ok") is False)

print()
if fails:
    print(f"{fails} CHECKS FAILED")
    raise SystemExit(1)
print("ALL CHECKS PASSED")
