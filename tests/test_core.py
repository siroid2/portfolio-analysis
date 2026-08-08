import unittest

import numpy as np
import pandas as pd

from portfolio_analytics.core import (
    calendar_returns,
    convert_to_base_currency,
    drawdown_series,
    normalize_weights,
    performance_metrics,
    portfolio_wealth,
    relative_metrics,
)


class PortfolioAnalyticsTests(unittest.TestCase):
    def setUp(self):
        self.index = pd.bdate_range("2023-01-02", periods=300)

    def test_normalize_weights(self):
        result = normalize_weights(pd.Series({"A": 60.0, "B": 40.0}))
        self.assertAlmostEqual(result.sum(), 1.0)
        self.assertAlmostEqual(result["A"], 0.6)

    def test_buy_and_hold_wealth_matches_units(self):
        prices = pd.DataFrame(
            {"A": [100.0, 110.0, 121.0], "B": [100.0, 100.0, 100.0]},
            index=pd.bdate_range("2024-01-02", periods=3),
        )
        wealth = portfolio_wealth(prices, {"A": 0.5, "B": 0.5}, rebalancing="Aucun")
        self.assertAlmostEqual(wealth.iloc[-1], 1.105)

    def test_daily_rebalancing(self):
        prices = pd.DataFrame(
            {"A": [100.0, 110.0, 121.0], "B": [100.0, 100.0, 100.0]},
            index=pd.bdate_range("2024-01-02", periods=3),
        )
        wealth = portfolio_wealth(prices, {"A": 0.5, "B": 0.5}, rebalancing="Quotidien")
        self.assertAlmostEqual(wealth.iloc[-1], 1.1025)

    def test_currency_conversion(self):
        prices = pd.DataFrame(
            {"US": [100.0, 101.0], "CA": [50.0, 51.0]},
            index=pd.bdate_range("2024-01-02", periods=2),
        )
        fx = pd.Series([1.35, 1.36], index=prices.index)
        converted = convert_to_base_currency(prices, {"US": "USD", "CA": "CAD"}, "CAD", fx)
        self.assertAlmostEqual(converted.iloc[0]["US"], 135.0)
        self.assertAlmostEqual(converted.iloc[1]["CA"], 51.0)

    def test_drawdown_and_metrics(self):
        wealth = pd.Series([1.0, 1.2, 0.9, 1.3], index=pd.bdate_range("2024-01-02", periods=4))
        drawdowns = drawdown_series(wealth)
        self.assertAlmostEqual(drawdowns.min(), -0.25)
        metrics = performance_metrics(wealth)
        self.assertAlmostEqual(metrics["total_return"], 0.3)
        self.assertAlmostEqual(metrics["max_drawdown"], -0.25)

    def test_relative_metrics_identical_series(self):
        returns = pd.Series(np.linspace(-0.01, 0.015, len(self.index)), index=self.index)
        wealth = (1.0 + returns).cumprod()
        result = relative_metrics(wealth, wealth)
        self.assertAlmostEqual(result["beta"], 1.0)
        self.assertAlmostEqual(result["correlation"], 1.0)
        self.assertAlmostEqual(result["tracking_error"], 0.0)

    def test_calendar_returns(self):
        index = pd.to_datetime(["2023-01-03", "2023-12-29", "2024-01-02", "2024-12-31"])
        wealth = pd.Series([1.0, 1.1, 1.2, 1.44], index=index)
        result = calendar_returns(wealth)
        self.assertAlmostEqual(result["2023"], 0.1)
        self.assertAlmostEqual(result["2024"], 0.2)


if __name__ == "__main__":
    unittest.main()
