"""The edge math. Small, pure functions, each one unit-tested.

American odds: -150 means bet $150 to win $100; +130 means bet $100 to win $130.
"""


def implied_prob(american: int) -> float:
    """Break-even win probability for a price (includes the book's cut)."""
    if american < 0:
        return -american / (-american + 100)
    return 100 / (american + 100)


def fair_probs(home_ml: int, away_ml: int) -> tuple[float, float]:
    """Remove the vig: scale both implied probabilities so they sum to 1."""
    h, a = implied_prob(home_ml), implied_prob(away_ml)
    return h / (h + a), a / (h + a)


def vig(home_ml: int, away_ml: int) -> float:
    return implied_prob(home_ml) + implied_prob(away_ml) - 1


def profit_if_win(american: int, stake: float = 100) -> float:
    return stake * 100 / -american if american < 0 else stake * american / 100


def expected_value(p_win: float, american: int, stake: float = 100) -> float:
    """Average profit per bet if p_win is the true win probability."""
    return p_win * profit_if_win(american, stake) - (1 - p_win) * stake


def edge(model_p: float, fair_p: float) -> float:
    """0.07 means the model gives the team 7 more points of win probability than the market."""
    return model_p - fair_p


def american_from_prob(p: float) -> int:
    """Fair American price for a probability (used by the demo data)."""
    p = min(max(p, 0.01), 0.99)
    return round(-100 * p / (1 - p)) if p >= 0.5 else round(100 * (1 - p) / p)
