"""Price and hedge one down-and-out call with and without downward jumps."""

import json
from pathlib import Path

import numpy as np

from src.gap_hedge import down_out_call, hedge_book

SPOT = 100.0
STRIKE = 100.0
BARRIER = 90.0
T = 1.0
R = 0.02
Q = 0.0
SIGMA = 0.20
# One jump every two years, on average, of a fixed 15% drop.
# From the starting spot, 100 * 0.85 = 85, which is through the barrier at 90.
JUMP_INTENSITY = 0.5
JUMP_MULTIPLIER = 0.85


def summarise(errors):
    return {
        "mean": float(errors.mean()),
        "std": float(errors.std(ddof=1)),
        "p05": float(np.quantile(errors, 0.05)),
        "p95": float(np.quantile(errors, 0.95)),
        "n_paths": int(errors.size),
    }


def main():
    bs = float(down_out_call(SPOT, STRIKE, BARRIER, T, R, Q, SIGMA))
    common = dict(
        spot=SPOT, strike=STRIKE, barrier=BARRIER, t=T, r=R, q=Q, sigma=SIGMA,
        n_days=252, n_sub=4, n_paths=40_000, seed=7,
    )
    diffusion = hedge_book(**common, premium=bs, jump_intensity=0.0, jump_multiplier=1.0)
    # Same paths would be ideal; the jump book uses the same seed and an extra coin flip.
    jumps_priced_at_bs = hedge_book(
        **common, premium=bs, jump_intensity=JUMP_INTENSITY, jump_multiplier=JUMP_MULTIPLIER,
    )
    jump_value = float(np.exp(-R * T) * jumps_priced_at_bs["payoff"].mean())
    # Extra premium collected at inception earns the rate for the whole year.
    fair_error = jumps_priced_at_bs["error"] + (jump_value - bs) * np.exp(R * T)
    payload = {
        "contract": {
            "spot": SPOT,
            "strike": STRIKE,
            "barrier": BARRIER,
            "expiry_years": T,
            "rate": R,
            "diffusion_volatility": SIGMA,
            "jump_intensity_per_year": JUMP_INTENSITY,
            "jump_multiplier": JUMP_MULTIPLIER,
        },
        "black_scholes_price": bs,
        "jump_monte_carlo_price": jump_value,
        "diffusion_hedge_sold_at_black_scholes": summarise(diffusion["error"]),
        "jump_hedge_sold_at_black_scholes": {
            **summarise(jumps_priced_at_bs["error"]),
            "fraction_gap_knockout": float(jumps_priced_at_bs["gap_knockout"].mean()),
            "fraction_diffusion_knockout": float(jumps_priced_at_bs["diffusion_knockout"].mean()),
        },
        "jump_hedge_sold_at_jump_price": summarise(fair_error),
    }
    out = Path(__file__).resolve().parent / "results" / "results.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
