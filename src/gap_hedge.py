"""Down-and-out call hedged with a Black-Scholes delta while the spot can jump.

The contract matches the geometric-Brownian-motion study: spot 100, strike 100,
barrier 90, one year, rate 2%, dividend yield 0, diffusion volatility 20%.
Jumps are a fixed multiplier, arriving as a Poisson process. The drift between
jumps includes the compensator, so the discounted spot is a martingale.

The hedge still uses the Black-Scholes down-and-out delta at 20% volatility.
A diffusion crossing unwinds at the barrier. A jump that lands through the
barrier unwinds at the post-jump price.
"""

import numpy as np
from scipy.stats import norm


def _d1(spot, strike, t, r, q, sigma):
    return (np.log(spot / strike) + (r - q + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))


def vanilla_call(spot, strike, t, r, q, sigma):
    spot = np.asarray(spot, dtype=float)
    if np.any(t <= 0):
        return np.maximum(spot - strike, 0.0)
    d1 = _d1(spot, strike, t, r, q, sigma)
    d2 = d1 - sigma * np.sqrt(t)
    return spot * np.exp(-q * t) * norm.cdf(d1) - strike * np.exp(-r * t) * norm.cdf(d2)


def down_out_call(spot, strike, barrier, t, r, q, sigma):
    if barrier > strike:
        raise ValueError("barrier must be at or below the strike")
    spot = np.asarray(spot, dtype=float)
    out = np.zeros(np.shape(spot), dtype=float)
    alive = spot > barrier
    if not np.any(alive):
        return out.reshape(np.shape(spot))
    s = spot[alive] if spot.ndim else spot
    if t <= 0:
        value = np.where(s > barrier, np.maximum(s - strike, 0.0), 0.0)
    else:
        lam = (r - q + 0.5 * sigma ** 2) / sigma ** 2
        y = (np.log(barrier ** 2 / (s * strike)) + (r - q + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))
        vanilla = vanilla_call(s, strike, t, r, q, sigma)
        reflected = (
            s * np.exp(-q * t) * (barrier / s) ** (2 * lam) * norm.cdf(y)
            - strike * np.exp(-r * t) * (barrier / s) ** (2 * lam - 2) * norm.cdf(y - sigma * np.sqrt(t))
        )
        value = vanilla - reflected
    if spot.ndim == 0:
        return float(value)
    out[alive] = value
    return out


def call_delta(spot, strike, barrier, t, r, q, sigma, bump=1e-4):
    spot = np.asarray(spot, dtype=float)
    h = np.maximum(bump * spot, 1e-6)
    up = down_out_call(spot + h, strike, barrier, t, r, q, sigma)
    down = down_out_call(np.maximum(spot - h, barrier * (1 + 1e-8)), strike, barrier, t, r, q, sigma)
    return (up - down) / (2 * h)


def _bridge_hit(s0, s1, barrier, sigma, dt):
    return np.exp(-2.0 * np.log(s0 / barrier) * np.log(s1 / barrier) / (sigma ** 2 * dt))


def hedge_book(
    spot, strike, barrier, t, r, q, sigma, premium,
    jump_intensity, jump_multiplier,
    n_days=252, n_sub=4, n_paths=20_000, seed=0,
):
    """Terminal hedge value minus the payoff, and how the barrier was hit.

    premium is the cash collected for the option at inception. The delta is
    always the Black-Scholes down-and-out delta at `sigma`.
    """
    if jump_multiplier <= 0:
        raise ValueError("jump multiplier must be positive")
    rng = np.random.default_rng(seed)
    n_steps = n_days * n_sub
    dt = t / n_steps
    k = jump_multiplier - 1.0
    drift = (r - q - jump_intensity * k - 0.5 * sigma ** 2) * dt
    scale = sigma * np.sqrt(dt)
    p_jump = jump_intensity * dt
    s = np.full(n_paths, spot)
    alive = np.ones(n_paths, dtype=bool)
    gap = np.zeros(n_paths, dtype=bool)
    diffusion_hit = np.zeros(n_paths, dtype=bool)
    shares = np.full(n_paths, call_delta(spot, strike, barrier, t, r, q, sigma))
    cash = np.full(n_paths, premium - shares[0] * spot)
    for step in range(1, n_steps + 1):
        z = rng.standard_normal(n_paths)
        s_diff = s * np.exp(drift + scale * z)
        both = alive & (s > barrier) & (s_diff > barrier)
        hit_diff = alive & (s_diff <= barrier)
        if np.any(both):
            hit_diff[both] = rng.random(both.sum()) < _bridge_hit(s[both], s_diff[both], barrier, sigma, dt)
        jumped = alive & ~hit_diff & (rng.random(n_paths) < p_jump)
        s_after = np.where(jumped, s_diff * jump_multiplier, s_diff)
        hit_gap = jumped & (s_diff > barrier) & (s_after <= barrier)
        hit = hit_diff | hit_gap
        cash *= np.exp(r * dt)
        unwind = np.where(hit_gap, s_after, np.where(s_diff <= barrier, s_diff, barrier))
        cash = np.where(hit, cash + shares * unwind, cash)
        shares = np.where(hit, 0.0, shares)
        gap |= hit_gap
        diffusion_hit |= hit_diff & ~hit_gap
        alive &= ~hit
        s = np.where(alive, s_after, s)
        if step % n_sub == 0 and step < n_steps and np.any(alive):
            t_left = t - step * dt
            new_delta = np.zeros(n_paths)
            new_delta[alive] = call_delta(s[alive], strike, barrier, t_left, r, q, sigma)
            cash = np.where(alive, cash - (new_delta - shares) * s, cash)
            shares = np.where(alive, new_delta, shares)
    payoff = np.where(alive, np.maximum(s - strike, 0.0), 0.0)
    return {
        "error": cash + shares * s - payoff,
        "payoff": payoff,
        "gap_knockout": gap,
        "diffusion_knockout": diffusion_hit,
    }


def discounted_stock(spot, t, r, q, sigma, jump_intensity, jump_multiplier, n_steps, n_paths, seed):
    """Mean of the discounted terminal spot. Should sit near the spot if q is 0."""
    rng = np.random.default_rng(seed)
    dt = t / n_steps
    k = jump_multiplier - 1.0
    drift = (r - q - jump_intensity * k - 0.5 * sigma ** 2) * dt
    scale = sigma * np.sqrt(dt)
    p_jump = jump_intensity * dt
    s = np.full(n_paths, spot)
    for _ in range(n_steps):
        s = s * np.exp(drift + scale * rng.standard_normal(n_paths))
        jump = rng.random(n_paths) < p_jump
        s = np.where(jump, s * jump_multiplier, s)
    return float(np.exp(-r * t) * s.mean())
