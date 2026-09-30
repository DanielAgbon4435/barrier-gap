# Selling a down-and-out at the Black–Scholes price when the spot can jump

Same contract as the geometric-Brownian-motion hedge: spot 100, strike 100, barrier 90, one year, rate 2%, no dividends, diffusion volatility 20%. Black–Scholes prices that call at 7.30, and a daily hedge at the Black–Scholes delta has a hedging error of mean −0.05 and standard deviation 0.28.

The only change here is the path. Once every two years on average, the spot is multiplied by 0.85. From the starting level, 100 × 0.85 = 85, which is through the barrier. Between jumps the drift is higher by the compensator, so the discounted spot remains a martingale. The hedge is still the Black–Scholes delta. A diffusion crossing is unwound at 90. A jump that lands below 90 is unwound at the post-jump price.

## What comes out

40,000 paths.

| | Price charged | Mean hedging error | Standard deviation | Worst 5% |
|---|---:|---:|---:|---:|
| No jumps, sold at 7.30 | 7.30 | −0.05 | 0.28 | −0.50 |
| Jumps, sold at 7.30 | 7.30 | −0.91 | 1.77 | −4.99 |
| Jumps, sold at the jump price | 8.06 | −0.14 | 1.77 | −4.22 |

The jump-model value is 8.06, not 7.30. The higher drift between crashes lifts the spot away from the barrier often enough to outweigh the crashes, so the contract is worth more than Black–Scholes says. Selling it at 7.30 loses 0.91 on average.

Charging 8.06 brings the average loss back to 0.14. It does not repair the hedge. The worst 5% of paths still lose about 4.2, against 0.50 when the spot cannot jump. On 14.7% of paths the barrier is crossed by the jump itself, and the delta is unwound only after the spot has already landed.

## Run it

```bash
python3 -m pip install -r requirements.txt
PYTHONPATH=. python3 run.py
PYTHONPATH=. python3 -m pytest
```

`run.py` rewrites `results/results.json`.
