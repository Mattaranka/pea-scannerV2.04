"""
Page dédiée : setups d'achat les plus intéressants pour du swing trading,
classés par score d'opportunité (combinaison tendance + croisement + cassure
+ confirmation volume + prudence RSI).

Le PEA ne permettant pas la vente à découvert, cette page se concentre sur
les setups acheteurs (score positif). Les signaux baissiers restent visibles
sur les pages dédiées (Croisements EMA, Cassures) comme signal de prudence /
sortie sur une position déjà ouverte.
"""

import os

import pandas as pd
import streamlit as st

from src import config

st.set_page_config(page_title="Opportunités - Scanner PEA", page_icon="🎯", layout="wide")
st.title("🎯 Opportunités de swing trading")
st.caption(
    "Score combiné = score de tendance + croisement (confirmé par le volume) + cassure "
    "(confirmée par le volume) − pénalité si RSI en zone extrême. Setups acheteurs uniquement."
)

if not os.path.exists(config.SCAN_RESULTS_FILE):
    st.warning("Aucune donnée disponible pour le moment.")
    st.stop()


@st.cache_data(ttl=300)
def load_data(mtime: float) -> pd.DataFrame:
    return pd.read_csv(config.SCAN_RESULTS_FILE)


mtime = os.path.getmtime(config.SCAN_RESULTS_FILE)
df = load_data(mtime)

seuil = st.slider(
    "Score minimum affiché",
    min_value=-10, max_value=18, value=config.OPPORTUNITY_MIN_SCORE,
)

df_opp = df[df["score_opportunite"] >= seuil].copy().sort_values("score_opportunite", ascending=False)

if df_opp.empty:
    st.info("Aucune action n'atteint ce score aujourd'hui. Essaie d'abaisser le seuil.")
    st.stop()

st.metric("Opportunités trouvées", len(df_opp))
st.divider()

st.dataframe(
    df_opp[[
        "ticker", "nom", "secteur", "dernier_cours", "score_opportunite", "score_tendance",
        f"rsi_{config.RSI_PERIOD}", "croisement", "jours_depuis_croisement",
        "cassure_20j", "jours_depuis_cassure", "volume_confirme",
        "stop_suggere", "objectif_suggere",
    ]].rename(columns={
        "ticker": "Ticker",
        "nom": "Nom",
        "secteur": "Secteur",
        "dernier_cours": "Dernier cours (€)",
        "score_opportunite": "Score opportunité",
        "score_tendance": "Score tendance",
        f"rsi_{config.RSI_PERIOD}": f"RSI{config.RSI_PERIOD}",
        "croisement": "Croisement",
        "jours_depuis_croisement": "Jours depuis croisement",
        "cassure_20j": "Cassure 20j",
        "jours_depuis_cassure": "Jours depuis cassure",
        "volume_confirme": "Volume confirmé",
        "stop_suggere": "Stop suggéré (€)",
        "objectif_suggere": "Objectif suggéré (€)",
    }),
    use_container_width=True,
    hide_index=True,
)

st.caption(
    f"⚠️ Stop et objectif sont calculés à partir de l'ATR{config.ATR_PERIOD} "
    f"(stop = cours − {config.ATR_STOP_MULTIPLIER}×ATR, objectif = cours + {config.ATR_TARGET_MULTIPLIER}×ATR). "
    "Ce sont des repères indicatifs basés sur la volatilité récente, pas une recommandation "
    "d'investissement — à ajuster selon le contexte du titre (supports/résistances, actualité)."
)
