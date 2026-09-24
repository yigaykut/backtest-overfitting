# Was that a result, or the best of many tries?

Two numbers that tell you how much of a backtest is skill and how much is
search. Deflated Sharpe ratio and probability of backtest overfitting, from
the Bailey and López de Prado papers. numpy and the standard library, nothing
else.

```python
import overfit

# every configuration you tried, winners and losers
trials = {"n10/stop25": returns_a, "n15/stop40": returns_b, ...}

d = overfit.sisirilmis_sharpe(trials)
print(d["sharpe"], d["sans_esigi"], d["sisirilmis"])

p = overfit.pbo(trials, s_parca=10)
print(p["pbo"])
```

## Why you would bother

Run fifty strategies that have no edge whatsoever and the best of them will
show a Sharpe around 1.1 with a 0.99 probability of beating zero. That looks
decisive. It is arithmetic.

The deflated Sharpe works out what the best of N skill-less attempts would show
by luck, and asks whether the observed Sharpe clears it. On that same fifty-way
pile of noise it reads 0.55 — which is what you want a measure to do when there
is nothing there.

Where the bar sits depends on how much the trials disagree with each other, not
just how many there were. A grid where every configuration lands in the same
place has not really searched; one that sprays results across a wide range has,
and it gets a higher bar.

PBO asks a different question. Split the sample, pick the training winner in
every combination of halves, and see where it ranks out of sample. Around 0.5
means "take whatever won in training" is a coin flip. That can be true even
when the strategy itself is fine — it says the *tuning* is noise, not the idea.

## Pass every trial

The one way to fool this is to hand it only the configurations that worked.
Fewer trials means a lower bar, which hides the exact bias you are measuring.
Losers in, or don't bother.

## What it caught

I built a rule for a cross-sectional crypto book that read the book's own
trailing Sharpe and cut exposure when it turned down. It looked like the best
thing I had: drawdown from −22.6% to −9.0%, and it held up under a null
control, a flat-exposure comparison, an unseen window, and four different
configurations.

Then the deflated Sharpe came back at 5.88, and a Sharpe of 5.88 does not
happen. Chasing that number is what found the problem. The equity curve
recorded each sub-book only when its holding period closed and forward-filled
the days between, so it was a staircase pretending to be a line —
autocorrelation 0.87 where a real daily series sits near zero. Annualising over
days that were mostly copies of each other inflated the Sharpe fourfold. And
the rule I was so pleased with had been reading that forward fill the whole
time. On a properly marked curve it makes things worse at every window.

All four of my checks had run on the same kind of curve, the null included:
shuffling daily returns in 35-day blocks preserves the block structure, and the
block structure *was* the artifact. What was missing wasn't another slice of
data or another parameter. It was another curve.

So: these two measures won't tell you a strategy is good. They will sometimes
tell you a number is impossible, and that turns out to be worth more.

## Tests

```bash
python tests/test_overfit.py
```

Twenty-two checks. They test behaviour rather than values — what should happen
when there is no skill, what should happen when trials are added, what a fat
left tail should do to the probability.

Two of them exist because they caught real bugs. A constant series was
returning an astronomically large but finite Sharpe: the mean of identical
floats leaves a residual around 1e-19 and `sd > 0` waves it through, so the
guard is relative now. And two of the tests were themselves useless at first —
one compared a single random draw against a narrow band, the other compared two
probabilities that had both saturated at 1.0000 and so tested nothing at all.

## Reading

- Bailey & López de Prado (2014), *The Deflated Sharpe Ratio*
- Bailey, Borwein, López de Prado & Zhu (2016), *The Probability of Backtest
  Overfitting*

## Licence

MIT.
