"""Accès aux prix de marché ajustés via Yahoo Finance."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path
import ssl
import sys
import tempfile

import certifi
from curl_cffi import requests as curl_requests
import pandas as pd
import yfinance as yf


class MarketDataError(RuntimeError):
    """Erreur lisible liée au téléchargement des données de marché."""


@lru_cache(maxsize=1)
def _configure_yfinance_cache() -> None:
    """Place les bases de cache yfinance dans un dossier local inscriptible."""
    cache_directory = Path(tempfile.gettempdir()) / "prisme-portfolio" / "yfinance-cache"
    cache_directory.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(cache_directory))


@lru_cache(maxsize=1)
def _market_session() -> curl_requests.Session:
    """Crée une session HTTPS compatible avec les certificats Windows.

    Certaines entreprises ajoutent leur propre autorité de certification au
    magasin Windows. ``curl_cffi`` (utilisé par yfinance) consulte normalement
    seulement le bundle Mozilla de certifi. On fusionne ici les deux sources de
    confiance, sans jamais désactiver la validation SSL.
    """
    if sys.platform != "win32" or not hasattr(ssl, "enum_certificates"):
        return curl_requests.Session(impersonate="chrome")

    try:
        bundle = Path(certifi.where()).read_bytes()
        windows_certificates: set[bytes] = set()
        for store_name in ("ROOT", "CA"):
            for certificate, encoding, _trust in ssl.enum_certificates(store_name):
                if encoding == "x509_asn":
                    pem = ssl.DER_cert_to_PEM_cert(certificate).encode("ascii")
                    windows_certificates.add(pem)

        merged = bundle.rstrip() + b"\n" + b"\n".join(sorted(windows_certificates)) + b"\n"
        cache_directory = Path(tempfile.gettempdir()) / "prisme-portfolio"
        cache_directory.mkdir(parents=True, exist_ok=True)
        bundle_path = cache_directory / "windows-ca-bundle.pem"
        if not bundle_path.exists() or bundle_path.read_bytes() != merged:
            bundle_path.write_bytes(merged)
        return curl_requests.Session(impersonate="chrome", verify=str(bundle_path))
    except (OSError, ssl.SSLError, ValueError):
        # Le bundle certifi demeure sécurisé si le magasin Windows est
        # inaccessible ou contient un certificat non exportable.
        return curl_requests.Session(impersonate="chrome")


def _extract_close(raw: pd.DataFrame, requested: list[str]) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        first_level = raw.columns.get_level_values(0)
        second_level = raw.columns.get_level_values(1)
        if "Close" in first_level:
            close = raw.xs("Close", axis=1, level=0)
        elif "Close" in second_level:
            close = raw.xs("Close", axis=1, level=1)
        else:
            return pd.DataFrame()
    elif "Close" in raw.columns:
        close = raw[["Close"]].copy()
        close.columns = [requested[0]]
    else:
        return pd.DataFrame()

    if isinstance(close, pd.Series):
        close = close.to_frame(name=requested[0])
    close.columns = [str(column).upper() for column in close.columns]
    close = close.loc[:, ~close.columns.duplicated(keep="first")]
    return close


def _download(tickers: list[str], start: date, end_exclusive: date) -> pd.DataFrame:
    _configure_yfinance_cache()
    raw = yf.download(
        tickers=tickers,
        start=start.isoformat(),
        end=end_exclusive.isoformat(),
        interval="1d",
        auto_adjust=True,
        repair=False,
        actions=False,
        progress=False,
        threads=True,
        group_by="column",
        multi_level_index=True,
        timeout=20,
        session=_market_session(),
    )
    return _extract_close(raw, tickers)


def download_adjusted_closes(
    tickers: Iterable[str],
    start: date,
    end: date,
) -> pd.DataFrame:
    """Télécharge les cours ajustés, avec une nouvelle tentative symbole par symbole."""
    requested = list(dict.fromkeys(str(ticker).strip().upper() for ticker in tickers if str(ticker).strip()))
    if not requested:
        raise MarketDataError("Aucun symbole n'a été fourni.")
    if start >= end:
        raise MarketDataError("La date de début doit précéder la date de fin.")

    warmup_start = start - timedelta(days=10)
    end_exclusive = end + timedelta(days=1)
    try:
        close = _download(requested, warmup_start, end_exclusive)
    except Exception as exc:  # yfinance regroupe plusieurs types d'erreurs réseau
        raise MarketDataError(f"Impossible de joindre la source de données: {exc}") from exc

    missing = [
        ticker
        for ticker in requested
        if ticker not in close.columns or close[ticker].dropna().shape[0] < 2
    ]
    for ticker in missing:
        try:
            single = _download([ticker], warmup_start, end_exclusive)
        except Exception:
            continue
        if ticker in single.columns and single[ticker].dropna().shape[0] >= 2:
            close[ticker] = single[ticker]

    still_missing = [
        ticker
        for ticker in requested
        if ticker not in close.columns or close[ticker].dropna().shape[0] < 2
    ]
    if still_missing:
        raise MarketDataError(
            "Aucune donnée exploitable pour: "
            + ", ".join(still_missing)
            + ". Vérifiez les symboles Yahoo Finance et la connexion. "
            + "Sur un réseau d'entreprise, un certificat SSL local peut aussi bloquer Yahoo Finance."
        )

    close = close.reindex(columns=requested)
    close.index = pd.to_datetime(close.index)
    if getattr(close.index, "tz", None) is not None:
        close.index = close.index.tz_localize(None)
    return close.sort_index()
