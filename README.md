# Was that a result, or the best of many tries?

Three numbers that tell you how much of a backtest is skill, how much is search,
and whether the Sharpe you are reading is even measuring what you think.
Deflated Sharpe ratio and probability of backtest overfitting from the Bailey
and López de Prado papers, plus the frequency check that caught my own mistake.
numpy and the standard library, nothing else.

```python
import overfit

# every configuration you tried, winners and losers
trials = {"n10/stop25": returns_a, "n15/stop40": returns_b, ...}

d = overfit.deflated_sharpe(trials)
print(d["sharpe"], d["chance_bar"], d["deflated"])

p = overfit.pbo(trials, splits=10)
print(p["pbo"])

# before you trust any of the above: is the curve actually daily?
f = overfit.sharpe_by_frequency(returns_b)
print(f["lag1"], f["by_block"], f["suspect"])
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

`sharpe_by_frequency` asks the question that comes before both: is the series
you are annualising made of independent observations? Annualising assumes it
is. It compounds the returns into non-overlapping blocks of 1, 5, 21 and 35
periods and reports the Sharpe of each. A real daily series gives roughly the
same number every time. A curve that was filled forward between updates gives a
large number at block 1 that collapses as the blocks grow, and the size of that
collapse is how much the headline was borrowing.

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

`sharpe_by_frequency` is that check written down, so it runs first instead of
being arrived at. On the synthetic version of the same mistake — 60 period
returns spread evenly across 35 days each — it reports lag-1 +0.98, a block-1
Sharpe of 14.4 falling to 2.4 by block 35, and an effective sample of 26
observations where the array has 2100 rows.

So: these measures won't tell you a strategy is good. They will sometimes tell
you a number is impossible, and that turns out to be worth more.

## Tests

```bash
python tests/test_overfit.py
```

Twenty-nine checks. They test behaviour rather than values — what should happen
when there is no skill, what should happen when trials are added, what a fat
left tail should do to the probability, and what a deliberately forward-filled
series should do to an annualised Sharpe.

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
