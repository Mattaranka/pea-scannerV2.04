"""
Composant de graphique réutilisable (cours + EMA9/EMA20), partagé par
plusieurs pages Streamlit pour éviter de dupliquer le code de récupération
et de tracé sur chaque page.
"""

import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from src import config


def render_stock_chart(tickers_disponibles: list[str], key: str, titre: str = "Visualiser une action") -> None:
    """
    Affiche un sélecteur d'action puis un graphique chandelier + EMA9/EMA20
    pour l'action choisie.

    `tickers_disponibles` : liste des tickers proposés dans le sélecteur
    (typiquement la liste déjà filtrée sur la page appelante).
    `key` : préfixe unique par page, pour éviter les conflits entre widgets
    Streamlit portant le même nom sur des pages différentes.
    """
    if not tickers_disponibles:
        return

    st.divider()
    st.subheader(titre)

    ticker_choisi = st.selectbox(
        "Choisir une action pour afficher son graphique",
        sorted(set(tickers_disponibles)),
        key=f"selectbox_{key}",
    )
    if not ticker_choisi:
        return

    hist = yf.Ticker(ticker_choisi).history(
        period=config.HISTORY_PERIOD, interval=config.HISTORY_INTERVAL, auto_adjust=False
    )
    if hist.empty:
        st.warning("Impossible de récupérer l'historique pour cette action.")
        return

    hist[f"EMA_{config.EMA_SHORT}"] = hist["Close"].ewm(span=config.EMA_SHORT, adjust=False).mean()
    hist[f"EMA_{config.EMA_LONG}"] = hist["Close"].ewm(span=config.EMA_LONG, adjust=False).mean()

    # Les EMA sont calculées sur l'historique complet (2 ans, nécessaire pour
    # leur stabilisation) mais on n'affiche que les 6 derniers mois pour que
    # le graphique reste lisible.
    hist_affiche = hist.tail(130)

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=hist_affiche.index, open=hist_affiche["Open"], high=hist_affiche["High"],
        low=hist_affiche["Low"], close=hist_affiche["Close"],
        name="Cours",
    ))
    fig.add_trace(go.Scatter(x=hist_affiche.index, y=hist_affiche[f"EMA_{config.EMA_SHORT}"],
                              name=f"EMA{config.EMA_SHORT}", line=dict(color="orange", width=1.5)))
    fig.add_trace(go.Scatter(x=hist_affiche.index, y=hist_affiche[f"EMA_{config.EMA_LONG}"],
                              name=f"EMA{config.EMA_LONG}", line=dict(color="blue", width=1.5)))
    fig.update_layout(
        title=f"{ticker_choisi} — Cours et EMA{config.EMA_SHORT}/{config.EMA_LONG}",
        xaxis_rangeslider_visible=False,
        height=550,
    )
    st.plotly_chart(fig, use_container_width=True)
