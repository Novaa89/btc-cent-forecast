# btc-cent-forecast-01 entry: shrunk-trend BTC 4h forecaster (Python 3.13, stdlib only).
#
# Reads every <id>.in in the input directory (MAKEFUN_INPUT_DIR, else /input):
# one case, 4,320 one-minute cbBTC closes ending at the checkpoint, and writes
# <id>.out to the output directory (MAKEFUN_OUTPUT_DIR, else /output) with
# exactly {"forecasts": [{"case": <id>, "price": <guess>}]}, the price expected
# 240 minutes after the checkpoint.
#
# Strategy (tuned on the kit's six practice cases, p00-p05):
#   guess = last close
#         + 0.20 * (OLS trend of the last 60 closes, extrapolated 240 min)
#         - 0.05 * (last close - 24h mean close)
#   with the total adjustment clamped to +/- $600.
#
# Why: BTC's 4h move is dominated by noise, so the honest median forecast sits
# close to the checkpoint close (persistence). The last hour's trend carries a
# little signal, but only a fifth of its 4h extrapolation survives shrinkage;
# a price stretched far above its 24h mean tends to give some of it back, so a
# small mean-reversion term pulls the guess toward that mean. The $600 clamp
# (about 0.8% of price) keeps one violent hour from dragging the guess far:
# on the practice cases the worst single miss stays bounded, and a sudden
# one-minute tick in this pool's flat-minute price grid cannot dominate.
# Everything is deterministic (no randomness, no time, no network) and runs in
# milliseconds on 4,320 prices, far under the 8s cap.

import json
import os
from pathlib import Path

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

# Tuned constants (see header). Kept small in number on purpose: the practice
# set has only six cases, so a bigger parameter set would just overfit it.
TREND_WINDOW = 60        # minutes of closes the trend is fitted on
TREND_SHRINK = 0.20      # fraction of the extrapolated trend that is kept
HORIZON_MINUTES = 240    # forecast horizon, fixed by the competition
MEAN_WINDOW = 1440       # minutes (24h) for the mean-reversion anchor
MEAN_REVERT = 0.05       # fraction of the stretch from the anchor pulled back
MAX_ADJUST = 600.0       # dollars: clamp on the total adjustment

MIN_PRICE = 0.00001
MAX_PRICE = 9_999_999.99999


def price5(x):
    """Keep x on the price grid's 5 decimals and inside (0, 10,000,000)."""
    return round(min(MAX_PRICE, max(MIN_PRICE, x)) * 1e5) / 1e5


def ols_slope_per_minute(values):
    """Slope ($/minute) of the least-squares line through `values`."""
    n = len(values)
    if n < 2:
        return 0.0
    mean_i = (n - 1) / 2.0
    mean_v = sum(values) / n
    num = 0.0
    den = 0.0
    for i, v in enumerate(values):
        di = i - mean_i
        num += di * (v - mean_v)
        den += di * di
    return num / den if den else 0.0


def forecast(prices):
    """The 4h-ahead price guess for one market's closes."""
    if not prices:
        raise ValueError("market has no prices")
    last = prices[-1]
    if not isinstance(last, (int, float)) or last != last or last in (float("inf"), float("-inf")):
        raise ValueError("last price is not a finite number")

    # Short-horizon momentum: last hour's trend, shrunk hard.
    trend = ols_slope_per_minute(prices[-TREND_WINDOW:])
    trend_part = TREND_SHRINK * trend * HORIZON_MINUTES

    # Mean reversion: a price stretched from its 24h mean gives some back.
    anchor = sum(prices[-MEAN_WINDOW:]) / min(MEAN_WINDOW, len(prices))
    revert_part = -MEAN_REVERT * (last - anchor)

    adjust = trend_part + revert_part
    adjust = max(-MAX_ADJUST, min(MAX_ADJUST, adjust))
    return price5(last + adjust)


def main():
    in_dir = Path(INPUT_DIR)
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    inputs = sorted(p for p in in_dir.iterdir() if p.name.endswith(".in"))
    for path in inputs:
        case_id = path.name[: -len(".in")]
        with path.open("r", encoding="utf-8") as f:
            case = json.load(f)
        markets = case.get("markets")
        if not isinstance(markets, list) or len(markets) != 1:
            raise ValueError(f"{path.name} does not hold exactly one market")
        prices = markets[0].get("prices")
        if not isinstance(prices, list) or not prices:
            raise ValueError(f"{path.name} holds no prices")
        # Drop any non-finite close before forecasting: one bad tick in the
        # 4,320 must not poison the trend or the mean (flat minutes and
        # sudden ticks are normal in this pool; a NaN is not).
        clean = [float(p) for p in prices]
        clean = [p for p in clean if p == p and p not in (float("inf"), float("-inf"))]
        if not clean:
            raise ValueError(f"{path.name} holds no finite prices")
        guess = forecast(clean)
        out_path = out_dir / f"{case_id}.out"
        with out_path.open("w", encoding="utf-8") as f:
            f.write(json.dumps({"forecasts": [{"case": case_id, "price": guess}]}) + "\n")


if __name__ == "__main__":
    main()
