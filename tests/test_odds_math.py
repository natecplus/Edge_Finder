from pytest import approx

from edge.odds_math import (american_from_prob, edge, expected_value, fair_probs,
                            implied_prob, profit_if_win, vig)


def test_implied():
    assert implied_prob(-150) == approx(0.60)
    assert implied_prob(130) == approx(0.4348, abs=1e-4)
    assert implied_prob(100) == approx(0.5)


def test_fair_sums_to_one():
    h, a = fair_probs(-150, 130)
    assert h + a == approx(1.0)
    assert h == approx(0.5798, abs=1e-4)


def test_vig():
    assert vig(-110, -110) == approx(0.0476, abs=1e-4)


def test_profit():
    assert profit_if_win(-150) == approx(66.67, abs=0.01)
    assert profit_if_win(130) == approx(130)


def test_expected_value_zero_at_fair_price():
    # at the break-even probability, EV is zero
    assert expected_value(implied_prob(-150), -150) == approx(0.0, abs=1e-9)
    assert expected_value(0.5, 130) == approx(15.0)


def test_edge():
    assert edge(0.49, 0.42) == approx(0.07)


def test_american_round_trip():
    for p in (0.3, 0.5, 0.62, 0.8):
        assert implied_prob(american_from_prob(p)) == approx(p, abs=0.005)
