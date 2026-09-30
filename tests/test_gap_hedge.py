import numpy as np

from src.gap_hedge import discounted_stock, down_out_call, hedge_book


def test_discounted_spot_stays_near_the_forward():
    # q = 0, so the mean of the discounted spot should be the spot.
    got = discounted_stock(100, 1.0, 0.02, 0.0, 0.2, 0.5, 0.85, n_steps=252, n_paths=80_000, seed=1)
    assert abs(got - 100.0) < 1.0


def test_no_jump_hedge_matches_the_published_scale():
    premium = down_out_call(100, 100, 90, 1.0, 0.02, 0.0, 0.2)
    book = hedge_book(
        100, 100, 90, 1.0, 0.02, 0.0, 0.2, premium,
        jump_intensity=0.0, jump_multiplier=1.0,
        n_days=252, n_sub=4, n_paths=6_000, seed=2,
    )
    assert abs(book["error"].mean()) < 0.15
    assert book["error"].std(ddof=1) < 0.5
    assert book["gap_knockout"].sum() == 0


def test_a_down_jump_from_the_spot_crosses_the_barrier():
    premium = down_out_call(100, 100, 90, 1.0, 0.02, 0.0, 0.2)
    book = hedge_book(
        100, 100, 90, 1.0, 0.02, 0.0, 0.2, premium,
        jump_intensity=0.5, jump_multiplier=0.85,
        n_days=252, n_sub=2, n_paths=4_000, seed=3,
    )
    assert book["gap_knockout"].mean() > 0.05
