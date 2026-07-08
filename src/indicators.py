"""
Calcul d'indicateurs techniques.

Conçu pour être étendu facilement : chaque nouvel indicateur (RSI, Bollinger,
MACD...) est une fonction indépendante prenant un DataFrame OHLCV et
retournant une Series ou colonne ajoutée. Le scanner orchestre l'appel de
ces fonctions, il n'a pas besoin de connaître leur détail interne.
"""

import pandas as pd


def add_ema(df: pd.DataFrame, period: int, column: str = "Close") -> pd.DataFrame:
    """Ajoute une colonne EMA_{period} au DataFrame (calculée sur 'Close' par défaut)."""
    df[f"EMA_{period}"] = df[column].ewm(span=period, adjust=False).mean()
    return df


def detect_ema_crossover(df: pd.DataFrame, short: int, long: int) -> str | None:
    """
    Détecte si un croisement EMA court / EMA long a eu lieu entre l'avant-dernière
    et la dernière bougie disponible (= "aujourd'hui" par rapport à "hier").

    Retourne :
        "haussier" si EMA_short passe au-dessus de EMA_long (golden cross)
        "baissier" si EMA_short passe en dessous de EMA_long (death cross)
        None si aucun croisement
    """
    short_col, long_col = f"EMA_{short}", f"EMA_{long}"
    if short_col not in df.columns or long_col not in df.columns:
        raise ValueError("Les colonnes EMA doivent être calculées avant la détection.")

    if len(df) < 2:
        return None

    prev = df.iloc[-2]
    curr = df.iloc[-1]

    was_below = prev[short_col] < prev[long_col]
    is_above = curr[short_col] > curr[long_col]
    was_above = prev[short_col] > prev[long_col]
    is_below = curr[short_col] < curr[long_col]

    if was_below and is_above:
        return "haussier"
    if was_above and is_below:
        return "baissier"
    return None


def add_rsi(df: pd.DataFrame, period: int, column: str = "Close") -> pd.DataFrame:
    """
    Ajoute une colonne RSI_{period} (indice de force relative), calculé avec
    le lissage de Wilder (méthode standard, identique à la plupart des
    plateformes de trading).
    """
    delta = df[column].diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    avg_gain = gains.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = losses.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(100)  # si avg_loss == 0 sur la période : RSI = 100 (pas de baisse)

    df[f"RSI_{period}"] = rsi
    return df


def add_volume_stats(df: pd.DataFrame, period: int) -> pd.DataFrame:
    """
    Ajoute la moyenne de volume des `period` jours précédant chaque bougie
    (le jour même est exclu du calcul, pour comparer le volume du jour à une
    référence "normale" qui ne l'inclut pas déjà).
    """
    df[f"volume_avg_{period}"] = df["Volume"].shift(1).rolling(period).mean()
    return df


def add_range_stats(df: pd.DataFrame, period: int, column_high: str = "High", column_low: str = "Low",
                     column_close: str = "Close") -> pd.DataFrame:
    """
    Ajoute la variation moyenne (amplitude haut-bas, en % du cours de clôture)
    calculée sur les `period` derniers jours - un indicateur simple de
    volatilité récente.
    """
    daily_range_pct = (df[column_high] - df[column_low]) / df[column_close] * 100
    df[f"range_avg_{period}"] = daily_range_pct.rolling(period).mean()
    return df


def add_high_low_window(df: pd.DataFrame, window: int, label: str,
                         column_high: str = "High", column_low: str = "Low") -> pd.DataFrame:
    """
    Ajoute le plus haut et le plus bas atteints sur les `window` derniers
    jours (jour courant inclus). `label` sert à nommer les colonnes, par
    exemple label="52w" -> colonnes "high_52w" / "low_52w".
    """
    df[f"high_{label}"] = df[column_high].rolling(window, min_periods=1).max()
    df[f"low_{label}"] = df[column_low].rolling(window, min_periods=1).min()
    return df


def add_breakout_levels(df: pd.DataFrame, period: int,
                         column_high: str = "High", column_low: str = "Low") -> pd.DataFrame:
    """
    Ajoute le plus haut et le plus bas des `period` jours PRÉCÉDENT chaque
    bougie (jour courant exclu). C'est le niveau de référence utilisé pour
    détecter une cassure : si le cours du jour dépasse ce plus haut (ou passe
    sous ce plus bas), il y a cassure.
    """
    df[f"high_{period}d_prior"] = df[column_high].shift(1).rolling(period).max()
    df[f"low_{period}d_prior"] = df[column_low].shift(1).rolling(period).min()
    return df


def detect_breakout(df: pd.DataFrame, period: int, column_close: str = "Close") -> str | None:
    """
    Détecte si le cours de clôture du jour a cassé le plus haut ou le plus
    bas des `period` jours précédents.

    Retourne "haussière", "baissière", ou None si aucune cassure.
    """
    high_col, low_col = f"high_{period}d_prior", f"low_{period}d_prior"
    if high_col not in df.columns or low_col not in df.columns or len(df) < 1:
        return None

    curr = df.iloc[-1]
    if pd.isna(curr[high_col]) or pd.isna(curr[low_col]):
        return None

    if curr[column_close] > curr[high_col]:
        return "haussière"
    if curr[column_close] < curr[low_col]:
        return "baissière"
    return None


# --- Score de tendance -----------------------------------------------------------
# Classement (du plus haut au plus bas) des 4 niveaux Prix/EMA9/EMA20/EMA200.
# Seules les 8 configurations "typiques" ci-dessous sont notées ; toute autre
# combinaison retourne un score neutre de 0 (voir compute_trend_score).
TREND_SCORE_TABLE = {
    ("Prix", "EMA9", "EMA20", "EMA200"): 7,
    ("Prix", "EMA20", "EMA9", "EMA200"): 4,
    ("EMA9", "Prix", "EMA20", "EMA200"): 4,
    ("EMA20", "EMA9", "Prix", "EMA200"): 2,
    ("EMA20", "EMA200", "Prix", "EMA9"): -2,
    ("EMA200", "Prix", "EMA20", "EMA9"): -4,
    ("EMA200", "EMA9", "Prix", "EMA20"): -4,
    ("EMA200", "EMA20", "EMA9", "Prix"): -7,
}


def compute_trend_score(price: float, ema9: float, ema20: float, ema200: float) -> tuple[int, str]:
    """
    Calcule le score de tendance en comparant l'ordre relatif de Prix, EMA9,
    EMA20 et EMA200.

    Retourne (score, configuration) où `configuration` est une chaîne du
    type "Prix > EMA9 > EMA20 > EMA200", utile pour vérifier visuellement
    quel classement a été détecté (y compris pour les configurations non
    notées, qui obtiennent un score de 0).
    """
    valeurs = {"Prix": price, "EMA9": ema9, "EMA20": ema20, "EMA200": ema200}
    classement = tuple(sorted(valeurs, key=lambda k: valeurs[k], reverse=True))
    score = TREND_SCORE_TABLE.get(classement, 0)
    configuration = " > ".join(classement)
    return score, configuration


# --- ATR (volatilité) et suggestions de stop / objectif --------------------------

def add_atr(df: pd.DataFrame, period: int, column_high: str = "High", column_low: str = "Low",
            column_close: str = "Close") -> pd.DataFrame:
    """
    Ajoute une colonne ATR_{period} (Average True Range, lissage de Wilder).
    Le True Range prend en compte les gaps (écarts entre la clôture de la
    veille et le haut/bas du jour), contrairement à un simple haut-bas.
    """
    prev_close = df[column_close].shift(1)
    tr = pd.concat([
        df[column_high] - df[column_low],
        (df[column_high] - prev_close).abs(),
        (df[column_low] - prev_close).abs(),
    ], axis=1).max(axis=1)

    df[f"ATR_{period}"] = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    return df


# --- Fraîcheur des signaux --------------------------------------------------------
# Un croisement ou une cassure gardent souvent leur pertinence quelques jours
# après leur apparition (le temps que le mouvement se développe). Ces
# fonctions cherchent, dans une fenêtre récente, le dernier jour où le
# signal a eu lieu et retournent son ancienneté en jours (0 = aujourd'hui).

def days_since_ema_crossover(df: pd.DataFrame, short: int, long: int, max_lookback: int) -> int | None:
    short_col, long_col = f"EMA_{short}", f"EMA_{long}"
    for age in range(max_lookback):
        if len(df) <= age + 2:
            break
        curr = df.iloc[-(age + 1)]
        prev = df.iloc[-(age + 2)]
        was_below, is_above = prev[short_col] < prev[long_col], curr[short_col] > curr[long_col]
        was_above, is_below = prev[short_col] > prev[long_col], curr[short_col] < curr[long_col]
        if (was_below and is_above) or (was_above and is_below):
            return age
    return None


def days_since_breakout(df: pd.DataFrame, period: int, max_lookback: int, column_close: str = "Close") -> int | None:
    high_col, low_col = f"high_{period}d_prior", f"low_{period}d_prior"
    for age in range(max_lookback):
        if len(df) <= age + 1:
            break
        curr = df.iloc[-(age + 1)]
        if pd.isna(curr[high_col]) or pd.isna(curr[low_col]):
            continue
        if curr[column_close] > curr[high_col] or curr[column_close] < curr[low_col]:
            return age
    return None


# --- Score d'opportunité combiné --------------------------------------------------
# Combine le score de tendance avec les autres signaux (croisement, cassure,
# confirmation par le volume, zone RSI extrême) en un score unique, pensé
# pour prioriser les setups d'achat en swing trading. C'est un point de
# départ empirique : à ajuster avec l'expérience et, plus tard, avec les
# résultats du backtesting sur data/history/.
def compute_opportunity_score(
    score_tendance: int | None,
    croisement: str | None,
    croisement_confirme: bool,
    cassure: str | None,
    cassure_confirmee: bool,
    rsi: float | None,
) -> int:
    score = score_tendance if score_tendance is not None else 0

    if croisement == "haussier":
        score += 3 if croisement_confirme else 1
    elif croisement == "baissier":
        score += -3 if croisement_confirme else -1

    if cassure == "haussière":
        score += 3 if cassure_confirmee else 1
    elif cassure == "baissière":
        score += -3 if cassure_confirmee else -1

    if rsi is not None and (rsi > 70 or rsi < 30):
        score -= 1  # zone extrême = prudence, quel que soit le sens

    return score
