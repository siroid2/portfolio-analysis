"""Calculs financiers purs, indépendants de l'interface et de la source de données."""

from __future__ import annotations

from collections.abc import Mapping
import math

import numpy as np
import pandas as pd


TRADING_DAYS = 252
VALID_REBALANCING = {"Aucun", "Quotidien", "Mensuel", "Trimestriel", "Annuel"}


def _safe_divide(numerator: float, denominator: float) -> float:
    if not np.isfinite(denominator) or abs(denominator) < 1e-15:
        return float("nan")
    return float(numerator / denominator)


def normalize_weights(weights: pd.Series | Mapping[str, float]) -> pd.Series:
    """Valide et normalise une série de pondérations afin qu'elle totalise 1."""
    series = pd.Series(weights, dtype=float)
    if series.empty:
        raise ValueError("Au moins une pondération est requise.")
    if not np.isfinite(series.to_numpy()).all():
        raise ValueError("Les pondérations doivent être des nombres finis.")
    if (series < 0).any():
        raise ValueError("Les pondérations ne peuvent pas être négatives.")
    total = float(series.sum())
    if total <= 0:
        raise ValueError("La somme des pondérations doit être supérieure à zéro.")
    return series / total


def prepare_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Nettoie, aligne et valide un tableau de prix."""
    if prices.empty:
        raise ValueError("Aucun prix n'est disponible pour la période choisie.")

    clean = prices.copy()
    clean.index = pd.to_datetime(clean.index)
    if getattr(clean.index, "tz", None) is not None:
        clean.index = clean.index.tz_localize(None)
    clean = clean.sort_index()
    clean = clean.loc[~clean.index.duplicated(keep="last")]
    clean = clean.replace([np.inf, -np.inf], np.nan).ffill().dropna(how="any")

    if clean.empty:
        raise ValueError("Les séries de prix ne se chevauchent pas sur la période choisie.")
    if (clean <= 0).any().any():
        raise ValueError("Une série contient un prix nul ou négatif.")
    return clean.astype(float)


def convert_to_base_currency(
    prices: pd.DataFrame,
    currencies: Mapping[str, str],
    base_currency: str,
    usd_cad: pd.Series | None = None,
) -> pd.DataFrame:
    """Convertit des prix CAD/USD vers la devise de base.

    ``usd_cad`` représente le nombre de dollars canadiens pour un dollar US
    (symbole Yahoo Finance ``CAD=X``).
    """
    base_currency = base_currency.upper()
    if base_currency not in {"CAD", "USD"}:
        raise ValueError("La devise de base doit être CAD ou USD.")

    converted = prices.copy().astype(float)
    different_currency = any(
        currencies.get(str(column), "").upper() != base_currency for column in converted.columns
    )
    if different_currency:
        if usd_cad is None:
            raise ValueError("Le taux de change USD/CAD est requis pour convertir les prix.")
        fx = pd.Series(usd_cad, dtype=float).reindex(converted.index).ffill()
        if fx.isna().any() or (fx <= 0).any():
            raise ValueError("Le taux de change USD/CAD est incomplet ou invalide.")
    else:
        fx = None

    for column in converted.columns:
        currency = currencies.get(str(column), "").upper()
        if currency not in {"CAD", "USD"}:
            raise ValueError(f"Devise inconnue pour {column}: {currency or 'vide'}.")
        if currency == base_currency:
            continue
        if base_currency == "CAD" and currency == "USD":
            converted[column] = converted[column] * fx
        elif base_currency == "USD" and currency == "CAD":
            converted[column] = converted[column] / fx

    return converted


def _period_key(timestamp: pd.Timestamp, rebalancing: str) -> tuple[int, ...]:
    if rebalancing == "Mensuel":
        return (timestamp.year, timestamp.month)
    if rebalancing == "Trimestriel":
        return (timestamp.year, (timestamp.month - 1) // 3 + 1)
    if rebalancing == "Annuel":
        return (timestamp.year,)
    return (timestamp.year, timestamp.month, timestamp.day)


def portfolio_wealth(
    prices: pd.DataFrame,
    weights: pd.Series | Mapping[str, float],
    rebalancing: str = "Mensuel",
    name: str = "Portefeuille",
) -> pd.Series:
    """Construit un indice de richesse partant de 1.

    Le rééquilibrage a lieu à la clôture du dernier jour de négociation de la
    période choisie. ``Aucun`` représente une stratégie acheter-conserver.
    """
    if rebalancing not in VALID_REBALANCING:
        raise ValueError(f"Fréquence de rééquilibrage inconnue: {rebalancing}.")

    clean = prepare_prices(prices)
    normalized = normalize_weights(weights)
    missing = [symbol for symbol in normalized.index if symbol not in clean.columns]
    if missing:
        raise ValueError(f"Prix manquant pour: {', '.join(map(str, missing))}.")

    normalized = normalized.reindex(clean.columns, fill_value=0.0)
    if float(normalized.sum()) <= 0:
        raise ValueError("Aucune pondération ne correspond aux séries de prix.")
    normalized = normalize_weights(normalized)

    returns = clean.pct_change(fill_method=None).fillna(0.0)
    asset_values = normalized.copy()
    values = np.ones(len(clean), dtype=float)

    for position in range(1, len(clean)):
        asset_values = asset_values * (1.0 + returns.iloc[position])
        total = float(asset_values.sum())
        values[position] = total

        if position >= len(clean) - 1 or rebalancing == "Aucun":
            continue
        current_date = pd.Timestamp(clean.index[position])
        next_date = pd.Timestamp(clean.index[position + 1])
        should_rebalance = rebalancing == "Quotidien" or (
            _period_key(current_date, rebalancing) != _period_key(next_date, rebalancing)
        )
        if should_rebalance:
            asset_values = normalized * total

    return pd.Series(values, index=clean.index, name=name)


def drawdown_series(wealth: pd.Series) -> pd.Series:
    """Retourne la baisse par rapport au plus haut précédent."""
    clean = pd.Series(wealth, dtype=float).dropna()
    if clean.empty:
        return clean
    return (clean / clean.cummax() - 1.0).rename(wealth.name)


def _monthly_returns(wealth: pd.Series) -> pd.Series:
    month_ends = wealth.resample("ME").last()
    return month_ends.pct_change(fill_method=None).dropna()


def performance_metrics(wealth: pd.Series, risk_free_rate: float = 0.0) -> dict[str, float]:
    """Calcule les principales statistiques absolues de rendement et de risque."""
    clean = pd.Series(wealth, dtype=float).dropna()
    if len(clean) < 2:
        raise ValueError("Au moins deux observations sont requises.")
    if (clean <= 0).any():
        raise ValueError("L'indice de richesse doit demeurer positif.")

    daily = clean.pct_change(fill_method=None).dropna()
    calendar_days = max((clean.index[-1] - clean.index[0]).days, 1)
    years = calendar_days / 365.25
    total_return = float(clean.iloc[-1] / clean.iloc[0] - 1.0)
    cagr = float((clean.iloc[-1] / clean.iloc[0]) ** (1.0 / years) - 1.0)
    volatility = float(daily.std(ddof=1) * math.sqrt(TRADING_DAYS))
    arithmetic_return = float(daily.mean() * TRADING_DAYS)

    risk_free_daily = (1.0 + risk_free_rate) ** (1.0 / TRADING_DAYS) - 1.0
    excess = daily - risk_free_daily
    sharpe = _safe_divide(float(excess.mean() * math.sqrt(TRADING_DAYS)), float(daily.std(ddof=1)))
    downside_daily = float(np.sqrt(np.mean(np.minimum(excess.to_numpy(), 0.0) ** 2)))
    sortino = _safe_divide(float(excess.mean() * TRADING_DAYS), downside_daily * math.sqrt(TRADING_DAYS))

    drawdowns = drawdown_series(clean)
    max_drawdown = float(drawdowns.min())
    calmar = _safe_divide(cagr, abs(max_drawdown))
    var_95 = float(daily.quantile(0.05))
    tail = daily[daily <= var_95]
    cvar_95 = float(tail.mean()) if not tail.empty else float("nan")
    monthly = _monthly_returns(clean)

    return {
        "observations": float(len(daily)),
        "years": float(years),
        "total_return": total_return,
        "cagr": cagr,
        "arithmetic_return": arithmetic_return,
        "volatility": volatility,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
        "var_95": var_95,
        "cvar_95": cvar_95,
        "best_month": float(monthly.max()) if not monthly.empty else float("nan"),
        "worst_month": float(monthly.min()) if not monthly.empty else float("nan"),
        "positive_months": float((monthly > 0).mean()) if not monthly.empty else float("nan"),
    }


def relative_metrics(
    portfolio_wealth_series: pd.Series,
    benchmark_wealth_series: pd.Series,
    risk_free_rate: float = 0.0,
) -> dict[str, float]:
    """Calcule les statistiques relatives au benchmark."""
    aligned = pd.concat(
        [
            pd.Series(portfolio_wealth_series, dtype=float).rename("portfolio"),
            pd.Series(benchmark_wealth_series, dtype=float).rename("benchmark"),
        ],
        axis=1,
    ).dropna()
    returns = aligned.pct_change(fill_method=None).dropna()
    if len(returns) < 2:
        raise ValueError("Pas assez d'observations communes avec le benchmark.")

    portfolio = returns["portfolio"]
    benchmark = returns["benchmark"]
    benchmark_variance = float(benchmark.var(ddof=1))
    beta = _safe_divide(float(portfolio.cov(benchmark)), benchmark_variance)
    correlation = float(portfolio.corr(benchmark))
    risk_free_daily = (1.0 + risk_free_rate) ** (1.0 / TRADING_DAYS) - 1.0
    alpha_daily = (portfolio.mean() - risk_free_daily) - beta * (benchmark.mean() - risk_free_daily)
    alpha = float(alpha_daily * TRADING_DAYS) if np.isfinite(beta) else float("nan")

    active = portfolio - benchmark
    tracking_error = float(active.std(ddof=1) * math.sqrt(TRADING_DAYS))
    information_ratio = _safe_divide(float(active.mean() * TRADING_DAYS), tracking_error)

    up = benchmark > 0
    down = benchmark < 0
    upside_capture = _safe_divide(float(portfolio[up].mean()), float(benchmark[up].mean()))
    downside_capture = _safe_divide(float(portfolio[down].mean()), float(benchmark[down].mean()))

    return {
        "alpha": alpha,
        "beta": beta,
        "correlation": correlation,
        "r_squared": correlation**2 if np.isfinite(correlation) else float("nan"),
        "tracking_error": tracking_error,
        "information_ratio": information_ratio,
        "upside_capture": upside_capture,
        "downside_capture": downside_capture,
    }


def calendar_returns(wealth: pd.Series) -> pd.Series:
    """Calcule le rendement de chaque année civile, incluant les années partielles."""
    clean = pd.Series(wealth, dtype=float).dropna()
    by_year = clean.groupby(clean.index.year)
    result = by_year.last() / by_year.first() - 1.0
    result.index = result.index.astype(str)
    return result


def holding_statistics(prices: pd.DataFrame, weights: pd.Series) -> pd.DataFrame:
    """Retourne quelques statistiques descriptives pour chaque position."""
    clean = prepare_prices(prices)
    normalized = normalize_weights(weights)
    rows: list[dict[str, float | str]] = []
    for symbol in normalized.index:
        wealth = (clean[symbol] / clean[symbol].iloc[0]).rename(symbol)
        metrics = performance_metrics(wealth)
        rows.append(
            {
                "Symbole": symbol,
                "Poids": float(normalized[symbol]),
                "Rendement total": metrics["total_return"],
                "Rendement annualisé": metrics["cagr"],
                "Volatilité": metrics["volatility"],
                "Baisse maximale": metrics["max_drawdown"],
            }
        )
    return pd.DataFrame(rows)
