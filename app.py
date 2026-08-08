"""Interface Streamlit de l'analyseur de portefeuille client."""

from __future__ import annotations

from datetime import date
import math
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from portfolio_analytics import (
    calendar_returns,
    convert_to_base_currency,
    drawdown_series,
    holding_statistics,
    performance_metrics,
    portfolio_wealth,
    relative_metrics,
)
from portfolio_analytics.core import prepare_prices
from portfolio_analytics.data import MarketDataError, download_adjusted_closes


st.set_page_config(
    page_title="Prisme | Analyse de portefeuille",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)


BENCHMARK_OPTIONS = [
    {
        "key": "sp500",
        "label": "S&P 500",
        "ticker": "SPY",
        "currency": "USD",
        "description": "Grandes sociétés américaines",
        "default": 30.0,
    },
    {
        "key": "nasdaq",
        "label": "Nasdaq 100",
        "ticker": "QQQ",
        "currency": "USD",
        "description": "Croissance et technologie US",
        "default": 20.0,
    },
    {
        "key": "dow",
        "label": "Dow Jones",
        "ticker": "DIA",
        "currency": "USD",
        "description": "30 grandes sociétés américaines",
        "default": 10.0,
    },
    {
        "key": "tsx",
        "label": "Canada / TSX",
        "ticker": "XIC.TO",
        "currency": "CAD",
        "description": "Marché canadien diversifié",
        "default": 40.0,
    },
    {
        "key": "international",
        "label": "MSCI international",
        "ticker": "XEF.TO",
        "currency": "CAD",
        "description": "Marchés développés hors Amérique",
        "default": 0.0,
    },
    {
        "key": "bonds_us",
        "label": "Obligations US",
        "ticker": "AGG",
        "currency": "USD",
        "description": "Marché obligataire américain",
        "default": 0.0,
    },
    {
        "key": "bonds_cad",
        "label": "Obligations Canada",
        "ticker": "XBB.TO",
        "currency": "CAD",
        "description": "Marché obligataire canadien",
        "default": 0.0,
    },
    {
        "key": "bonds_intl",
        "label": "Obligations internationales",
        "ticker": "BNDX",
        "currency": "USD",
        "description": "Obligations mondiales hors États-Unis",
        "default": 0.0,
    },
]


# EXAMPLE_HOLDINGS = pd.DataFrame(
#     [
#         {"Nom": "Actions américaines", "Symbole": "VUN.TO", "Type": "FNB", "Devise": "CAD", "Poids (%)": 45.0},
#         {"Nom": "Actions canadiennes", "Symbole": "XIC.TO", "Type": "FNB", "Devise": "CAD", "Poids (%)": 20.0},
#         {"Nom": "Actions internationales", "Symbole": "XEF.TO", "Type": "FNB", "Devise": "CAD", "Poids (%)": 15.0},
#         {"Nom": "Obligations canadiennes", "Symbole": "XBB.TO", "Type": "FNB obligataire", "Devise": "CAD", "Poids (%)": 20.0},
#     ]
# )

EXAMPLE_HOLDINGS = pd.DataFrame(
    [
        {"Nom": "AAPL",   "Symbole": "AAPL",   "Type": "Action", "Devise": "USD", "Poids (%)": 5},
        {"Nom": "AEM.TO", "Symbole": "AEM.TO", "Type": "Action", "Devise": "CAD", "Poids (%)": 3},
        {"Nom": "AMZN",   "Symbole": "AMZN",   "Type": "Action", "Devise": "USD", "Poids (%)": 5},
        {"Nom": "ATD.TO", "Symbole": "ATD.TO", "Type": "Action", "Devise": "CAD", "Poids (%)": 5},
        {"Nom": "BRK-B",  "Symbole": "BRK-B",  "Type": "Action", "Devise": "USD", "Poids (%)": 5},
        {"Nom": "CNR.TO", "Symbole": "CNR.TO", "Type": "Action", "Devise": "CAD", "Poids (%)": 5},
        {"Nom": "COKE",   "Symbole": "COKE",   "Type": "Action", "Devise": "USD", "Poids (%)": 5},
        {"Nom": "ED",     "Symbole": "ED",     "Type": "Action", "Devise": "USD", "Poids (%)": 5},
        {"Nom": "ENB.TO", "Symbole": "ENB.TO", "Type": "Action", "Devise": "CAD", "Poids (%)": 5},
        {"Nom": "FTS.TO", "Symbole": "FTS.TO", "Type": "Action", "Devise": "CAD", "Poids (%)": 5},
        {"Nom": "GOOGL",  "Symbole": "GOOGL",  "Type": "Action", "Devise": "USD", "Poids (%)": 5},
        {"Nom": "IAG.TO", "Symbole": "IAG.TO", "Type": "Action", "Devise": "CAD", "Poids (%)": 5},
        {"Nom": "JPM",    "Symbole": "JPM",    "Type": "Action", "Devise": "USD", "Poids (%)": 5},
        {"Nom": "MCD",    "Symbole": "MCD",    "Type": "Action", "Devise": "USD", "Poids (%)": 2},
        {"Nom": "MSFT",   "Symbole": "MSFT",   "Type": "Action", "Devise": "USD", "Poids (%)": 5},
        {"Nom": "RY.TO",  "Symbole": "RY.TO",  "Type": "Action", "Devise": "CAD", "Poids (%)": 5},
        {"Nom": "TD.TO",  "Symbole": "TD.TO",  "Type": "Action", "Devise": "CAD", "Poids (%)": 5},
        {"Nom": "UNH",    "Symbole": "UNH",    "Type": "Action", "Devise": "USD", "Poids (%)": 5},
        {"Nom": "WCN.TO", "Symbole": "WCN.TO", "Type": "Action", "Devise": "CAD", "Poids (%)": 5},
        {"Nom": "WMT",    "Symbole": "WMT",    "Type": "Action", "Devise": "USD", "Poids (%)": 5},
        {"Nom": "XOM",    "Symbole": "XOM",    "Type": "Action", "Devise": "USD", "Poids (%)": 5},
    ]
)


EMPTY_HOLDINGS = pd.DataFrame(
    [{"Nom": "", "Symbole": "", "Type": "FNB", "Devise": "CAD", "Poids (%)": 0.0}]
)


REBALANCING_HELP = {
    "Aucun": "Les pondérations dérivent avec les marchés (acheter-conserver).",
    "Quotidien": "Les pondérations cibles sont rétablies chaque jour.",
    "Mensuel": "Les pondérations cibles sont rétablies à la fin de chaque mois.",
    "Trimestriel": "Les pondérations cibles sont rétablies à la fin de chaque trimestre.",
    "Annuel": "Les pondérations cibles sont rétablies à la fin de chaque année.",
}


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp { background: #f7f5ef; }
        .block-container { max-width: 1320px; padding-top: 2.1rem; padding-bottom: 4rem; }
        [data-testid="stSidebar"] { background: #192923; }
        [data-testid="stSidebar"] * { color: #f4f0e7; }
        [data-testid="stSidebar"] input,
        [data-testid="stSidebar"] [data-baseweb="select"] * { color: #17251f !important; }
        [data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.14); }
        h1, h2, h3 { color: #17251f; letter-spacing: -0.025em; }
        h1 { font-size: clamp(2.15rem, 4vw, 3.65rem) !important; line-height: 1.02 !important; }
        h2 { margin-top: 1.4rem; }
        .eyebrow { color: #a8562f; font-size: .76rem; font-weight: 800; letter-spacing: .16em; text-transform: uppercase; }
        .hero-copy { color: #56635e; font-size: 1.08rem; max-width: 780px; line-height: 1.65; margin: .5rem 0 1.25rem; }
        .section-kicker { color: #a8562f; font-size: .72rem; font-weight: 800; letter-spacing: .13em; text-transform: uppercase; margin-bottom: -.45rem; }
        .allocation-note { padding: .75rem 1rem; border-radius: 10px; background: #ece9df; color: #41504a; font-size: .9rem; }
        .allocation-ok { background: #e0ebe3; color: #24533b; }
        .allocation-bad { background: #f3e2dc; color: #853e29; }
        div[data-testid="stMetric"] { background: #fffdf8; border: 1px solid #e4dfd3; border-radius: 14px; padding: 1rem 1.05rem; min-height: 118px; box-shadow: 0 5px 22px rgba(23,37,31,.045); }
        div[data-testid="stMetricLabel"] { color: #65716c; }
        div[data-testid="stMetricValue"] { color: #17251f; letter-spacing: -.035em; }
        div[data-testid="stDataFrame"], div[data-testid="stDataEditor"] { border: 1px solid #e1ddd2; border-radius: 12px; overflow: hidden; }
        .stButton > button[kind="primary"] { background: #c96e3b; border-color: #c96e3b; color: white; border-radius: 10px; font-weight: 750; min-height: 3rem; }
        .stButton > button[kind="secondary"] { border-color: #cfc9bb; border-radius: 9px; }
        .stTabs [data-baseweb="tab-list"] { gap: 1.4rem; border-bottom: 1px solid #ddd8cd; }
        .stTabs [data-baseweb="tab"] { padding-left: 0; padding-right: 0; font-weight: 700; }
        .stTabs [aria-selected="true"] { color: #a8562f; }
        .benchmark-caption { color: #68736f; font-size: .88rem; margin-top: -.5rem; margin-bottom: .5rem; }
        .fine-print { color: #77817d; font-size: .78rem; line-height: 1.5; border-top: 1px solid #dfdbd0; margin-top: 2.5rem; padding-top: 1rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def years_ago(today: date, years: int) -> date:
    try:
        return today.replace(year=today.year - years)
    except ValueError:
        return today.replace(year=today.year - years, day=28)


def reset_holdings(frame: pd.DataFrame) -> None:
    st.session_state["holdings_data"] = frame.copy()
    st.session_state["holdings_version"] = st.session_state.get("holdings_version", 0) + 1


def allocation_badge(total: float, label: str) -> None:
    is_valid = math.isclose(total, 100.0, abs_tol=0.05)
    css_class = "allocation-ok" if is_valid else "allocation-bad"
    status = "prête" if is_valid else "à ajuster"
    st.markdown(
        f'<div class="allocation-note {css_class}">{label} : <strong>{total:.1f} %</strong> · {status}</div>',
        unsafe_allow_html=True,
    )


def clean_holdings(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    for column in ["Nom", "Symbole", "Type", "Devise"]:
        working[column] = working[column].fillna("").astype(str).str.strip()
    working["Symbole"] = working["Symbole"].str.upper()
    working["Devise"] = working["Devise"].str.upper()
    working["Poids (%)"] = pd.to_numeric(working["Poids (%)"], errors="coerce").fillna(0.0)

    incomplete = working[(working["Symbole"] == "") & (working["Poids (%)"] > 0)]
    if not incomplete.empty:
        raise ValueError("Chaque ligne pondérée doit contenir un symbole.")
    working = working[(working["Symbole"] != "") & (working["Poids (%)"] > 0)].copy()
    if working.empty:
        raise ValueError("Ajoutez au moins une position avec une pondération positive.")
    if (~working["Devise"].isin(["CAD", "USD"])).any():
        raise ValueError("Chaque position doit utiliser la devise CAD ou USD.")

    total = float(working["Poids (%)"].sum())
    if not math.isclose(total, 100.0, abs_tol=0.05):
        raise ValueError(f"Les pondérations du portefeuille totalisent {total:.1f} %, plutôt que 100 %.")

    conflicts = working.groupby("Symbole")["Devise"].nunique()
    conflict_symbols = conflicts[conflicts > 1].index.tolist()
    if conflict_symbols:
        raise ValueError("Devise contradictoire pour: " + ", ".join(conflict_symbols) + ".")

    return working.rename(
        columns={"Nom": "name", "Symbole": "ticker", "Type": "type", "Devise": "currency", "Poids (%)": "weight"}
    )[["name", "ticker", "type", "currency", "weight"]]


def build_benchmark() -> tuple[pd.DataFrame, float]:
    rows = []
    for option in BENCHMARK_OPTIONS:
        weight = float(st.session_state.get(f"bench_weight_{option['key']}", option["default"]))
        ticker = str(st.session_state.get(f"bench_ticker_{option['key']}", option["ticker"])).strip().upper()
        currency = str(st.session_state.get(f"bench_currency_{option['key']}", option["currency"])).upper()
        if weight > 0:
            rows.append(
                {
                    "name": option["label"],
                    "ticker": ticker,
                    "type": "Benchmark",
                    "currency": currency,
                    "weight": weight,
                }
            )
    return pd.DataFrame(rows, columns=["name", "ticker", "type", "currency", "weight"]), sum(
        float(st.session_state.get(f"bench_weight_{option['key']}", option["default"]))
        for option in BENCHMARK_OPTIONS
    )


def validate_benchmark(frame: pd.DataFrame, total: float) -> pd.DataFrame:
    if frame.empty:
        raise ValueError("Attribuez une pondération positive à au moins un bloc du benchmark.")
    if not math.isclose(total, 100.0, abs_tol=0.05):
        raise ValueError(f"Les pondérations du benchmark totalisent {total:.1f} %, plutôt que 100 %.")
    if (frame["ticker"].str.len() == 0).any():
        raise ValueError("Chaque bloc pondéré du benchmark doit avoir un symbole.")
    if (~frame["currency"].isin(["CAD", "USD"])).any():
        raise ValueError("Chaque bloc du benchmark doit utiliser la devise CAD ou USD.")

    conflicts = frame.groupby("ticker")["currency"].nunique()
    conflict_symbols = conflicts[conflicts > 1].index.tolist()
    if conflict_symbols:
        raise ValueError("Devise contradictoire dans le benchmark pour: " + ", ".join(conflict_symbols) + ".")
    return frame


@st.cache_data(ttl=3600, show_spinner=False)
def cached_prices(symbols: tuple[str, ...], start: date, end: date) -> pd.DataFrame:
    return download_adjusted_closes(symbols, start, end)


def aggregate_allocations(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        frame.groupby(["ticker", "currency"], as_index=False)
        .agg(weight=("weight", "sum"), name=("name", "first"), type=("type", "first"))
    )
    grouped["weight"] = grouped["weight"] / grouped["weight"].sum()
    return grouped


def run_analysis(
    holdings: pd.DataFrame,
    benchmark: pd.DataFrame,
    start: date,
    end: date,
    base_currency: str,
    rebalancing: str,
    risk_free_rate: float,
) -> dict[str, object]:
    holdings_agg = aggregate_allocations(holdings)
    benchmark_agg = aggregate_allocations(benchmark)
    combined = pd.concat([holdings_agg, benchmark_agg], ignore_index=True)

    conflicts = combined.groupby("ticker")["currency"].nunique()
    conflict_symbols = conflicts[conflicts > 1].index.tolist()
    if conflict_symbols:
        raise ValueError(
            "Le même symbole ne peut pas avoir deux devises différentes: " + ", ".join(conflict_symbols) + "."
        )

    currency_map = combined.drop_duplicates("ticker").set_index("ticker")["currency"].to_dict()
    symbols = list(dict.fromkeys(combined["ticker"].tolist()))
    needs_fx = any(currency != base_currency for currency in currency_map.values())
    download_symbols = symbols + (["CAD=X"] if needs_fx and "CAD=X" not in symbols else [])

    raw = cached_prices(tuple(download_symbols), start, end)
    fx = raw["CAD=X"] if needs_fx else None
    converted = convert_to_base_currency(raw[symbols], currency_map, base_currency, fx)
    converted = converted.loc[(converted.index.date >= start) & (converted.index.date <= end)]
    prices = prepare_prices(converted)
    if len(prices) < 40:
        raise ValueError(
            "Moins de 40 observations communes sont disponibles. Choisissez une période plus longue ou vérifiez les symboles."
        )

    portfolio_weights = holdings_agg.set_index("ticker")["weight"]
    benchmark_weights = benchmark_agg.set_index("ticker")["weight"]
    portfolio = portfolio_wealth(
        prices[portfolio_weights.index.tolist()], portfolio_weights, rebalancing, "Portefeuille client"
    )
    benchmark_wealth = portfolio_wealth(
        prices[benchmark_weights.index.tolist()], benchmark_weights, rebalancing, "Benchmark"
    )

    aligned = pd.concat([portfolio, benchmark_wealth], axis=1).dropna()
    portfolio = aligned["Portefeuille client"]
    benchmark_wealth = aligned["Benchmark"]
    portfolio_metrics = performance_metrics(portfolio, risk_free_rate)
    benchmark_metrics = performance_metrics(benchmark_wealth, risk_free_rate)

    position_details = holding_statistics(prices[portfolio_weights.index.tolist()], portfolio_weights)
    meta = holdings_agg.set_index("ticker")
    position_details.insert(1, "Nom", position_details["Symbole"].map(meta["name"]))
    position_details.insert(2, "Type", position_details["Symbole"].map(meta["type"]))
    position_details.insert(3, "Devise", position_details["Symbole"].map(meta["currency"]))

    calendar = pd.concat(
        [calendar_returns(portfolio).rename("Portefeuille"), calendar_returns(benchmark_wealth).rename("Benchmark")],
        axis=1,
    )
    benchmark_description = " + ".join(
        f"{row.weight * 100:.0f}% {row.ticker}" for row in benchmark_agg.itertuples()
    )

    return {
        "prices": prices,
        "portfolio": portfolio,
        "benchmark": benchmark_wealth,
        "portfolio_metrics": portfolio_metrics,
        "benchmark_metrics": benchmark_metrics,
        "relative_metrics": relative_metrics(portfolio, benchmark_wealth, risk_free_rate),
        "portfolio_drawdown": drawdown_series(portfolio),
        "benchmark_drawdown": drawdown_series(benchmark_wealth),
        "positions": position_details,
        "calendar": calendar,
        "benchmark_description": benchmark_description,
        "actual_start": prices.index[0],
        "actual_end": prices.index[-1],
    }


def fmt_pct(value: float, decimals: int = 1) -> str:
    if not np.isfinite(value):
        return "—"
    return f"{value * 100:,.{decimals}f} %".replace(",", " ").replace("-", "−")


def fmt_num(value: float, decimals: int = 2) -> str:
    return "—" if not np.isfinite(value) else f"{value:.{decimals}f}".replace("-", "−")


def metric_delta(portfolio_value: float, benchmark_value: float, percent: bool = True) -> str | None:
    difference = portfolio_value - benchmark_value
    if not np.isfinite(difference):
        return None
    return (f"{difference:+.1%}" if percent else f"{difference:+.2f}") + " vs benchmark"


def base_chart_layout(title: str, y_title: str) -> dict[str, object]:
    return {
        "title": {"text": title, "x": 0, "xanchor": "left", "font": {"size": 19, "color": "#17251f"}},
        "height": 430,
        "margin": {"l": 10, "r": 18, "t": 58, "b": 12},
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "#fffdf8",
        "hovermode": "x unified",
        "legend": {"orientation": "h", "y": 1.08, "x": 1, "xanchor": "right"},
        "xaxis": {"showgrid": False, "linecolor": "#ded9cf"},
        "yaxis": {"title": y_title, "gridcolor": "#e9e5dc", "zerolinecolor": "#cfc9bd"},
        "font": {"family": "Arial, sans-serif", "color": "#485650"},
    }


def growth_chart(
    portfolio: pd.Series,
    benchmark: pd.Series,
    initial_value: float,
    currency: str,
    logarithmic: bool = False,
) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=portfolio.index,
            y=portfolio * initial_value,
            name="Portefeuille client",
            mode="lines",
            line={"color": "#1b5c4a", "width": 3},
            hovertemplate=f"%{{y:,.0f}} {currency}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=benchmark.index,
            y=benchmark * initial_value,
            name="Benchmark",
            mode="lines",
            line={"color": "#c96e3b", "width": 2.3},
            hovertemplate=f"%{{y:,.0f}} {currency}<extra></extra>",
        )
    )
    title = "Croissance d’un placement hypothétique"
    if logarithmic:
        title += " · échelle logarithmique"
    figure.update_layout(**base_chart_layout(title, f"Valeur ({currency})"))
    figure.update_yaxes(type="log" if logarithmic else "linear", tickformat="~s" if logarithmic else ",.0f")
    return figure


def drawdown_chart(portfolio: pd.Series, benchmark: pd.Series) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=portfolio.index,
            y=portfolio,
            name="Portefeuille client",
            mode="lines",
            fill="tozeroy",
            line={"color": "#1b5c4a", "width": 2.2},
            fillcolor="rgba(27,92,74,.16)",
            hovertemplate="%{y:.1%}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=benchmark.index,
            y=benchmark,
            name="Benchmark",
            mode="lines",
            line={"color": "#c96e3b", "width": 2},
            hovertemplate="%{y:.1%}<extra></extra>",
        )
    )
    figure.update_layout(**base_chart_layout("Baisses depuis le dernier sommet", "Baisse"))
    figure.update_yaxes(tickformat=".0%")
    return figure


def formatted_comparison_table(result: dict[str, object]) -> pd.DataFrame:
    portfolio = result["portfolio_metrics"]
    benchmark = result["benchmark_metrics"]
    definitions = [
        ("Rendement total", "total_return", "pct"),
        ("Rendement annualisé", "cagr", "pct"),
        ("Rendement moyen annualisé", "arithmetic_return", "pct"),
        ("Volatilité annualisée", "volatility", "pct"),
        ("Baisse maximale", "max_drawdown", "pct"),
        ("Sharpe", "sharpe", "num"),
        ("Sortino", "sortino", "num"),
        ("Calmar", "calmar", "num"),
        ("VaR historique 95 % — 1 jour", "var_95", "pct"),
        ("CVaR historique 95 % — 1 jour", "cvar_95", "pct"),
        ("Meilleur mois", "best_month", "pct"),
        ("Pire mois", "worst_month", "pct"),
        ("Mois positifs", "positive_months", "pct"),
    ]
    rows = []
    for label, key, kind in definitions:
        formatter = fmt_pct if kind == "pct" else fmt_num
        rows.append(
            {
                "Mesure": label,
                "Portefeuille": formatter(portfolio[key]),
                "Benchmark": formatter(benchmark[key]),
            }
        )
    return pd.DataFrame(rows)


def relative_table(result: dict[str, object]) -> pd.DataFrame:
    metrics = result["relative_metrics"]
    rows = [
        ("Alpha annualisé", fmt_pct(metrics["alpha"]), "Rendement excédentaire ajusté au bêta."),
        ("Bêta", fmt_num(metrics["beta"]), "Sensibilité du portefeuille aux mouvements du benchmark."),
        ("Corrélation", fmt_num(metrics["correlation"]), "Proximité des mouvements, de −1 à 1."),
        ("R²", fmt_pct(metrics["r_squared"]), "Part des variations expliquée par le benchmark."),
        ("Erreur de suivi", fmt_pct(metrics["tracking_error"]), "Volatilité annualisée de l'écart de rendement."),
        ("Ratio d'information", fmt_num(metrics["information_ratio"]), "Rendement actif par unité d'erreur de suivi."),
        ("Capture haussière", fmt_pct(metrics["upside_capture"]), "Participation moyenne aux journées positives du benchmark."),
        ("Capture baissière", fmt_pct(metrics["downside_capture"]), "Participation moyenne aux journées négatives du benchmark."),
    ]
    return pd.DataFrame(rows, columns=["Mesure", "Valeur", "Interprétation"])


def render_results(result: dict[str, object], initial_value: float, base_currency: str, client_name: str) -> None:
    portfolio_metrics = result["portfolio_metrics"]
    benchmark_metrics = result["benchmark_metrics"]
    portfolio = result["portfolio"]
    benchmark = result["benchmark"]

    st.markdown("---")
    st.markdown('<p class="section-kicker">Résultats</p>', unsafe_allow_html=True)
    title_suffix = f" · {client_name}" if client_name.strip() else ""
    st.subheader(f"Portrait du portefeuille{title_suffix}")
    st.caption(
        f"Période commune : {result['actual_start']:%d %b %Y} au {result['actual_end']:%d %b %Y} · "
        f"{int(portfolio_metrics['observations']):,} rendements quotidiens · devise {base_currency}"
    )

    row_one = st.columns(3)
    row_one[0].metric(
        "Valeur finale hypothétique",
        f"{portfolio.iloc[-1] * initial_value:,.0f} {base_currency}",
        f"{(portfolio.iloc[-1] - benchmark.iloc[-1]) * initial_value:+,.0f} vs benchmark",
    )
    row_one[1].metric(
        "Rendement total",
        fmt_pct(portfolio_metrics["total_return"]),
        metric_delta(portfolio_metrics["total_return"], benchmark_metrics["total_return"]),
    )
    row_one[2].metric(
        "Rendement annualisé",
        fmt_pct(portfolio_metrics["cagr"]),
        metric_delta(portfolio_metrics["cagr"], benchmark_metrics["cagr"]),
    )

    row_two = st.columns(3)
    row_two[0].metric("Volatilité annualisée", fmt_pct(portfolio_metrics["volatility"]))
    row_two[1].metric("Baisse maximale", fmt_pct(portfolio_metrics["max_drawdown"]))
    row_two[2].metric(
        "Ratio de Sharpe",
        fmt_num(portfolio_metrics["sharpe"]),
        metric_delta(portfolio_metrics["sharpe"], benchmark_metrics["sharpe"], percent=False),
    )

    st.markdown(
        f'<p class="benchmark-caption"><strong>Benchmark :</strong> {result["benchmark_description"]}</p>',
        unsafe_allow_html=True,
    )

    if portfolio_metrics["total_return"] > 10:
        st.info(
            f"Le rendement total exceptionnel équivaut à multiplier le capital par "
            f"{portfolio.iloc[-1]:,.1f}. Le rendement annualisé est de "
            f"{fmt_pct(portfolio_metrics['cagr'])}. Les prix sont déjà ajustés pour les fractionnements; "
            "ceux-ci ne sont donc pas comptés deux fois."
        )

    overview_tab, risk_tab, positions_tab, data_tab = st.tabs(
        ["Vue d’ensemble", "Risque & benchmark", "Positions", "Rendements & export"]
    )

    with overview_tab:
        positive_values = pd.concat([portfolio, benchmark])
        positive_values = positive_values[positive_values > 0]
        dispersion = float(positive_values.max() / positive_values.min()) if not positive_values.empty else 1.0
        default_logarithmic = dispersion >= 50
        scale = st.radio(
            "Échelle du graphique de croissance",
            ["Logarithmique", "Linéaire"],
            index=0 if default_logarithmic else 1,
            horizontal=True,
            help="L'échelle logarithmique rend les périodes anciennes visibles lorsque le capital a été multiplié plusieurs fois.",
            key="growth_chart_scale",
        )
        st.plotly_chart(
            growth_chart(
                portfolio,
                benchmark,
                initial_value,
                base_currency,
                logarithmic=scale == "Logarithmique",
            ),
            width="stretch",
            config={"displayModeBar": False, "responsive": True},
        )
        st.plotly_chart(
            drawdown_chart(result["portfolio_drawdown"], result["benchmark_drawdown"]),
            width="stretch",
            config={"displayModeBar": False, "responsive": True},
        )

    with risk_tab:
        left, right = st.columns([1.05, 1])
        with left:
            st.markdown("#### Statistiques absolues")
            st.dataframe(formatted_comparison_table(result), hide_index=True, width="stretch")
        with right:
            st.markdown("#### Statistiques relatives")
            st.dataframe(relative_table(result), hide_index=True, width="stretch")
        st.info(
            "La VaR et la CVaR sont calculées historiquement sur les rendements quotidiens. "
            "Elles décrivent l'échantillon observé et ne constituent pas une prévision de perte maximale."
        )

    with positions_tab:
        st.markdown("#### Comportement historique des positions")
        st.caption("Chaque ligne est mesurée séparément dans la devise de base, sur la même période commune.")
        positions = result["positions"].copy()
        percent_columns = ["Poids", "Rendement total", "Rendement annualisé", "Volatilité", "Baisse maximale"]
        styled = positions.style.format({column: "{:.1%}" for column in percent_columns}, na_rep="—")
        st.dataframe(styled, hide_index=True, width="stretch")

    with data_tab:
        st.markdown("#### Rendements par année civile")
        st.caption("La première et la dernière année peuvent être partielles.")
        calendar = result["calendar"].copy()
        calendar.index.name = "Année"
        st.dataframe(calendar.style.format("{:.1%}", na_rep="—"), width="stretch")

        export = pd.DataFrame(
            {
                "Portefeuille": portfolio,
                "Benchmark": benchmark,
                "Drawdown portefeuille": result["portfolio_drawdown"],
                "Drawdown benchmark": result["benchmark_drawdown"],
            }
        )
        safe_client = re.sub(r"[^A-Za-z0-9_-]+", "_", client_name.strip()) or "client"
        st.download_button(
            "Télécharger les séries en CSV",
            data=export.to_csv(index_label="Date").encode("utf-8"),
            file_name=f"analyse_portefeuille_{safe_client}.csv",
            mime="text/csv",
        )


inject_styles()

if "holdings_data" not in st.session_state:
    st.session_state["holdings_data"] = EXAMPLE_HOLDINGS.copy()
if "holdings_version" not in st.session_state:
    st.session_state["holdings_version"] = 0


today = date.today()
with st.sidebar:
    st.markdown("## ◈ Prisme")
    st.caption("Analyse de portefeuille")
    st.markdown("---")
    client_name = st.text_input("Nom du client", placeholder="Client ou ménage")
    base_currency = st.selectbox("Devise de l’analyse", ["CAD", "USD"], index=0)
    start_date = st.date_input("Date de début", value=years_ago(today, 5), max_value=today)
    end_date = st.date_input("Date de fin", value=today, max_value=today)
    rebalancing = st.selectbox(
        "Rééquilibrage",
        ["Mensuel", "Trimestriel", "Annuel", "Aucun", "Quotidien"],
        index=0,
    )
    st.caption(REBALANCING_HELP[rebalancing])
    risk_free_percent = st.number_input(
        "Taux sans risque annuel (%)",
        min_value=-5.0,
        max_value=20.0,
        value=2.0,
        step=0.25,
        help="Utilisé pour les ratios de Sharpe, Sortino et l'alpha.",
    )
    initial_value = st.number_input(
        f"Placement hypothétique ({base_currency})",
        min_value=100.0,
        value=10000.0,
        step=1000.0,
    )
    st.markdown("---")
    st.caption("Les prix sont mis en cache pendant une heure afin d'accélérer les analyses répétées.")


st.markdown('<p class="eyebrow">Portefeuille · Risque · Benchmark</p>', unsafe_allow_html=True)
st.title("Analysez le portefeuille. Voyez ce qui compte.")
st.markdown(
    '<p class="hero-copy">Composez le portefeuille du client, définissez un benchmark sur mesure et obtenez '
    'un portrait clair du rendement, du risque et des baisses historiques.</p>',
    unsafe_allow_html=True,
)

st.markdown('<p class="section-kicker">Étape 1</p>', unsafe_allow_html=True)
st.subheader("Portefeuille client")
st.caption(
    "Utilisez les symboles reconnus par Yahoo Finance. Les titres canadiens portent généralement le suffixe .TO. "
    "Les obligations sont plus fiables sous forme de FNB obligataires."
)

button_left, button_right, spacer = st.columns([1, 1, 5])
if button_left.button("Charger l’exemple", width="stretch"):
    reset_holdings(EXAMPLE_HOLDINGS)
    st.rerun()
if button_right.button("Effacer", width="stretch"):
    reset_holdings(EMPTY_HOLDINGS)
    st.rerun()

editor_key = f"holdings_editor_{st.session_state['holdings_version']}"
edited_holdings = st.data_editor(
    st.session_state["holdings_data"],
    key=editor_key,
    num_rows="dynamic",
    hide_index=True,
    width="stretch",
    column_config={
        "Nom": st.column_config.TextColumn("Nom / description", width="medium"),
        "Symbole": st.column_config.TextColumn("Symbole Yahoo", required=True, width="small"),
        "Type": st.column_config.SelectboxColumn(
            "Type",
            options=["Action", "FNB", "Fonds mutuel", "FNB obligataire", "Obligation"],
            required=True,
            width="medium",
        ),
        "Devise": st.column_config.SelectboxColumn("Devise", options=["CAD", "USD"], required=True, width="small"),
        "Poids (%)": st.column_config.NumberColumn(
            "Poids (%)", min_value=0.0, max_value=100.0, step=1.0, format="%.1f", required=True, width="small"
        ),
    },
)
# ``st.data_editor`` mémorise ses modifications comme des écarts par rapport
# au tableau fourni. Ne pas remplacer ce tableau source à chaque rerun : cela
# réappliquerait les mêmes écarts sur une nouvelle base et ferait disparaître
# des lignes ajoutées ou rétablirait d'anciennes pondérations.
holdings_total = float(pd.to_numeric(edited_holdings["Poids (%)"], errors="coerce").fillna(0.0).sum())
allocation_badge(holdings_total, "Total du portefeuille")

with st.expander("Aide sur les symboles et les fonds mutuels"):
    st.markdown(
        """
        - Titres américains : `AAPL`, `SPY`, `AGG`.
        - Titres et FNB canadiens : `RY.TO`, `XIC.TO`, `XBB.TO`.
        - Fonds mutuels : entrez leur symbole Yahoo Finance lorsqu'il existe. Certains codes de fonds canadiens ne sont
          pas couverts; une source de données professionnelle pourra être branchée plus tard.
        - Obligations individuelles : elles ne sont analysables que si un historique de prix existe pour leur symbole.
          Pour une V1 robuste, utilisez plutôt un FNB obligataire.
        """
    )

st.markdown('<p class="section-kicker">Étape 2</p>', unsafe_allow_html=True)
st.subheader("Benchmark personnalisé")
st.caption("Répartissez 100 % entre les blocs ci-dessous. Chaque bloc utilise un FNB liquide comme approximation.")

for row_start in range(0, len(BENCHMARK_OPTIONS), 4):
    columns = st.columns(4)
    for column, option in zip(columns, BENCHMARK_OPTIONS[row_start : row_start + 4]):
        with column:
            st.markdown(f"**{option['label']}**")
            st.caption(f"{option['ticker']} · {option['description']}")
            st.number_input(
                f"Poids {option['label']} (%)",
                min_value=0.0,
                max_value=100.0,
                value=option["default"],
                step=5.0,
                key=f"bench_weight_{option['key']}",
                label_visibility="collapsed",
            )

with st.expander("Modifier les FNB utilisés pour le benchmark"):
    st.caption("Les valeurs proposées sont des approximations de portefeuilles indiciels standards.")
    for row_start in range(0, len(BENCHMARK_OPTIONS), 4):
        columns = st.columns(4)
        for column, option in zip(columns, BENCHMARK_OPTIONS[row_start : row_start + 4]):
            with column:
                st.text_input(option["label"], value=option["ticker"], key=f"bench_ticker_{option['key']}")
                st.selectbox(
                    f"Devise · {option['label']}",
                    ["CAD", "USD"],
                    index=0 if option["currency"] == "CAD" else 1,
                    key=f"bench_currency_{option['key']}",
                )

benchmark_frame, benchmark_total = build_benchmark()
allocation_badge(benchmark_total, "Total du benchmark")

st.markdown('<p class="section-kicker">Étape 3</p>', unsafe_allow_html=True)
st.subheader("Lancer l’analyse")

dates_valid = start_date < end_date
allocations_valid = math.isclose(holdings_total, 100.0, abs_tol=0.05) and math.isclose(
    benchmark_total, 100.0, abs_tol=0.05
)
if not dates_valid:
    st.error("La date de début doit précéder la date de fin.")

analyze = st.button(
    "Analyser le portefeuille",
    type="primary",
    width="stretch",
    disabled=not (dates_valid and allocations_valid),
)

if analyze:
    try:
        cleaned_holdings = clean_holdings(edited_holdings)
        cleaned_benchmark = validate_benchmark(benchmark_frame, benchmark_total)
        with st.spinner("Téléchargement des prix et calcul des statistiques…"):
            st.session_state["analysis_result"] = run_analysis(
                cleaned_holdings,
                cleaned_benchmark,
                start_date,
                end_date,
                base_currency,
                rebalancing,
                risk_free_percent / 100.0,
            )
            st.session_state["analysis_settings"] = {
                "initial_value": initial_value,
                "base_currency": base_currency,
                "client_name": client_name,
            }
    except (ValueError, MarketDataError) as exc:
        st.session_state.pop("analysis_result", None)
        st.error(str(exc))
    except Exception as exc:
        st.session_state.pop("analysis_result", None)
        st.error(f"L'analyse n'a pas pu être complétée: {exc}")

if "analysis_result" in st.session_state:
    settings = st.session_state.get(
        "analysis_settings",
        {"initial_value": initial_value, "base_currency": base_currency, "client_name": client_name},
    )
    render_results(
        st.session_state["analysis_result"],
        settings["initial_value"],
        settings["base_currency"],
        settings["client_name"],
    )

st.markdown(
    '<p class="fine-print">Outil d’analyse historique à vocation informative. Les résultats reposent sur des prix '
    'ajustés provenant de Yahoo Finance et supposent des pondérations constantes selon la fréquence de rééquilibrage '
    'choisie. Ils n’intègrent pas les dépôts, retraits, impôts, frais de conseil ni coûts de transaction et ne '
    'constituent pas une recommandation de placement.</p>',
    unsafe_allow_html=True,
)
