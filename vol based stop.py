# Volatility-Based Stop Calculator v5.2
# =====================================
# DISCLAIMER: Educational heuristic for risk sizing. Not a regime model.
# Data source: Yahoo Finance (not institutional grade, may have gaps).
# Continuous futures can have roll artifacts that affect ATR and realized vol.
# Coefficients are hand-tuned, not calibrated to optimize any objective.
# This is NOT trading advice.
#
# Dependencies:
#   pip install yfinance pandas numpy scikit-learn
# Tested on Python 3.11+
#
# Usage:
#   python vol_stops_v5_2.py

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass, field
from typing import Optional, List, Tuple


# =========================
# CONFIG
# =========================
FUTURES = ["ES=F", "NQ=F"]
ETFS = ["QQQ", "SPY"]
ASSETS = ETFS + FUTURES

TICK_SIZE = {"QQQ": 0.01, "SPY": 0.01, "ES=F": 0.25, "NQ=F": 0.25}
POINT_VALUE_USD = {"QQQ": 1.0, "SPY": 1.0, "ES=F": 50.0, "NQ=F": 20.0}

DATA_HORIZON = "2y"
ATR_EMA_SPAN = 14
VIX_WINDOW = 252
N_BUCKETS = 4
OVERNIGHT_GAP_LOOKBACK = 20
SMOOTH_WINDOW = 5
DATA_FRESHNESS_DAYS = 3

SEVERITY_WEIGHTS = {
    "realized_vol": 1.0,
    "vix_chg_5d": 0.5,
    "vix_accel": 0.3,
}

K_BUCKET_BASE = {0: 0.06, 1: 0.09, 2: 0.13, 3: 0.18}

OUT_STOPS_CSV = "vol_stops_v5_2.csv"
OUT_DECOMP_CSV = "vol_stops_v5_2_decomp.csv"


@dataclass
class DataBundle:
    asset_data: pd.DataFrame
    vix_close: pd.Series
    vix9d_close: Optional[pd.Series] = None
    warnings: List[str] = field(default_factory=list)
    effective_rows: int = 0
    missing_pct: dict = field(default_factory=dict)

    def warn(self, msg: str, echo: bool = True):
        self.warnings.append(msg)
        if echo:
            print(f"  ⚠ {msg}")


def _safe_series(df: pd.DataFrame, field: str, symbol: str) -> Optional[pd.Series]:
    """
    yfinance returns a multiindex column frame: df[field][symbol].
    This helper returns a Series or None with a warning handled upstream.
    """
    try:
        if field not in df.columns:
            return None
        sub = df[field]
        if isinstance(sub, pd.DataFrame):
            if symbol not in sub.columns:
                return None
            s = sub[symbol]
        else:
            # Unlikely for multi-asset download, but keep safe
            s = sub
        return s
    except Exception:
        return None


def fetch_all_data() -> DataBundle:
    bundle = DataBundle(asset_data=pd.DataFrame(), vix_close=pd.Series(dtype=float))

    print(f"\n[1/3] Downloading asset data ({DATA_HORIZON})...")
    bundle.asset_data = yf.download(
        ASSETS, period=DATA_HORIZON, interval="1d", auto_adjust=False, progress=False
    )
    if bundle.asset_data.empty:
        raise ValueError("Failed to fetch asset data.")

    print(f"[2/3] Downloading VIX data ({DATA_HORIZON})...")
    vix_data = yf.download(
        "^VIX", period=DATA_HORIZON, interval="1d", auto_adjust=False, progress=False
    )
    if vix_data.empty:
        raise ValueError("Failed to fetch VIX data.")
    bundle.vix_close = vix_data["Close"].squeeze().ffill(limit=3).dropna()

    print("[3/3] Downloading VIX9D data...")
    try:
        vix9d_data = yf.download(
            "^VIX9D", period=DATA_HORIZON, interval="1d", auto_adjust=False, progress=False
        )
        if not vix9d_data.empty:
            bundle.vix9d_close = vix9d_data["Close"].squeeze().ffill(limit=3).dropna()
            print(f"    ✓ VIX9D: {len(bundle.vix9d_close)} days")
    except Exception as e:
        bundle.warn(f"VIX9D fetch failed: {e}")

    print("\n  Data Completeness:")
    for a in ASSETS:
        close = _safe_series(bundle.asset_data, "Close", a)
        if close is None:
            print(f"    ✗ {a}: MISSING Close")
            bundle.missing_pct[a] = 100.0
            continue
        total = len(close)
        valid = close.notna().sum()
        pct_missing = (1 - valid / total) * 100 if total > 0 else 100
        bundle.missing_pct[a] = pct_missing
        status = "⚠" if pct_missing > 5 else "✓"
        print(f"    {status} {a}: {valid}/{total} days ({pct_missing:.1f}% missing)")

    return bundle


def compute_vix_percentile_np(arr: np.ndarray) -> float:
    """
    Percentile of the last value in the rolling window with NaN handling.
    Uses the true last value (even if earlier values include NaNs).
    """
    if arr.size == 0:
        return np.nan
    last_val = arr[-1]
    if np.isnan(last_val):
        return np.nan
    valid = arr[~np.isnan(arr)]
    if valid.size == 0:
        return np.nan
    return float((valid <= last_val).mean() * 100.0)


def compute_true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def compute_atr_ema(high: pd.Series, low: pd.Series, close: pd.Series, span: int) -> pd.Series:
    tr = compute_true_range(high, low, close)
    atr = tr.ewm(span=span, adjust=False).mean()
    atr.iloc[: span - 1] = np.nan
    return atr


def compute_overnight_gap_risk(
    open_price: pd.Series,
    close: pd.Series,
    atr: pd.Series,
    last_date: pd.Timestamp,
    lookback: int = 20,
) -> Tuple[float, float]:
    idx = open_price.index.intersection(close.index).intersection(atr.index)
    idx = idx[idx <= last_date]
    if len(idx) < lookback + 1:
        return 0.08, 0.15

    open_price = open_price.loc[idx]
    close = close.loc[idx]
    atr = atr.loc[idx]

    overnight_gap = (open_price - close.shift(1)).abs()
    gap_ratio = overnight_gap / atr.shift(1)
    gap_ratio = gap_ratio.replace([np.inf, -np.inf], np.nan)

    valid = gap_ratio.dropna().tail(lookback)
    if len(valid) < 5:
        return 0.08, 0.15
    return float(valid.mean()), float(valid.quantile(0.90))


def compute_realized_vol(returns: pd.Series, last_date: pd.Timestamp, window: int = 20) -> float:
    aligned = returns.loc[:last_date].tail(window)
    if len(aligned) < window // 2:
        return np.nan
    return float(aligned.std() * np.sqrt(252))


def build_features(vix_close: pd.Series, asset_data: pd.DataFrame) -> pd.DataFrame:
    """
    Features for severity scoring.
    VIX percentile handled separately (as vix_adj) to avoid double counting.
    """
    features = pd.DataFrame(index=vix_close.index)
    features["vix_chg_5d"] = vix_close.pct_change(5)
    vix_chg = vix_close.pct_change()
    features["vix_accel"] = vix_chg.diff(5)

    spy_close = _safe_series(asset_data, "Close", "SPY")
    if spy_close is not None:
        spy_returns = spy_close.dropna().pct_change()
        features["realized_vol"] = spy_returns.rolling(20).std() * np.sqrt(252)
    else:
        features["realized_vol"] = 0.15

    for col in ["vix_chg_5d", "vix_accel"]:
        if col in features.columns:
            lower = features[col].quantile(0.01)
            upper = features[col].quantile(0.99)
            features[col] = features[col].clip(lower=lower, upper=upper)

    return features.dropna()


def compute_severity_score(features: pd.DataFrame) -> pd.Series:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features.values)
    X_df = pd.DataFrame(X_scaled, index=features.index, columns=features.columns)

    available_cols = [c for c in SEVERITY_WEIGHTS.keys() if c in X_df.columns]
    weights = {c: SEVERITY_WEIGHTS[c] for c in available_cols}
    weight_sum = sum(weights.values()) if weights else 1.0

    severity = pd.Series(0.0, index=features.index)
    for col, w in weights.items():
        severity += (w / weight_sum) * X_df[col].rank(pct=True)

    return severity


def assign_buckets(severity: pd.Series, n_buckets: int = 4) -> Tuple[pd.Series, dict]:
    ranked = severity.rank(method="first")
    buckets = pd.qcut(ranked, q=n_buckets, labels=False).astype(int)

    bucket_stats = {}
    for b in range(n_buckets):
        mask = buckets == b
        if mask.sum() > 0:
            bucket_stats[b] = {
                "count": int(mask.sum()),
                "severity_min": float(severity[mask].min()),
                "severity_max": float(severity[mask].max()),
                "severity_mean": float(severity[mask].mean()),
            }
        else:
            bucket_stats[b] = {"count": 0, "severity_min": 0, "severity_max": 0, "severity_mean": 0}

    return buckets, bucket_stats


def get_current_bucket(severity: pd.Series, buckets: pd.Series, window: int = 5) -> int:
    recent = buckets.tail(window).dropna()
    if len(recent) == 0:
        return 1
    return int(recent.mode().iloc[0])


def calculate_dynamic_k(
    base_k: float,
    vix_percentile: float,
    gap_p90: float,
    asset_realized_vol: float,
    market_realized_vol: float,
    vix_level: float,
    vix9d_level: Optional[float] = None,
) -> Tuple[float, dict]:
    decomp = {
        "base_k": base_k,
        "vix_adj": 0.0,
        "gap_adj": 0.0,
        "vol_adj": 0.0,
        "term_factor": 1.0,
        "term_adj_equiv": 0.0,
    }

    k = base_k

    z = (vix_percentile - 50.0) / 50.0
    vix_adj = 0.02 * z
    k += vix_adj
    decomp["vix_adj"] = vix_adj

    excess = max(0.0, gap_p90 - 0.10)
    gap_premium = min(excess * 0.20, 0.04)
    k += gap_premium
    decomp["gap_adj"] = gap_premium

    if asset_realized_vol > 0 and market_realized_vol > 0 and not np.isnan(asset_realized_vol):
        vol_ratio = asset_realized_vol / market_realized_vol
        if vol_ratio > 1.15:
            vol_adj = (vol_ratio - 1.0) * 0.05
            vol_adj = min(vol_adj, 0.03)
            k += vol_adj
            decomp["vol_adj"] = vol_adj

    k_before_term = k
    if vix9d_level is not None and vix_level > 0:
        raw_stress = vix9d_level / vix_level - 1.0
        stress = np.tanh(5 * raw_stress)
        term_factor = 1.0 + 0.03 * stress
        k *= term_factor
        decomp["term_factor"] = float(term_factor)
        decomp["term_adj_equiv"] = float(k_before_term * (term_factor - 1.0))

    final_k = max(0.05, min(0.25, k))
    decomp["final_k"] = float(final_k)
    return float(final_k), decomp


def round_to_tick(value: float, tick: float) -> float:
    return float(round(value / tick) * tick)


def pts_to_usd(asset: str, pts: float) -> float:
    return float(pts) * float(POINT_VALUE_USD.get(asset, 1.0))


def main(as_of_date: Optional[str] = None):
    print("=" * 70)
    print("VOLATILITY-BASED STOP CALCULATOR v5.2")
    print("=" * 70)
    print("⚠ DISCLAIMER: Educational heuristic. Not trading advice.")
    print("   Data: Yahoo Finance (not institutional grade)")
    print("   Buckets: quantile-based severity, not clustering")
    print("=" * 70)

    today = pd.Timestamp.now().normalize()

    bundle = fetch_all_data()
    if len(bundle.vix_close) < VIX_WINDOW:
        raise ValueError(f"Insufficient VIX data: {len(bundle.vix_close)} days (need {VIX_WINDOW}).")

    print("\n[4/6] Calculating ATR per asset...")
    atr_dict: dict[str, pd.Series] = {}
    last_close_dict: dict[str, pd.Series] = {}

    for a in ASSETS:
        high = _safe_series(bundle.asset_data, "High", a)
        low = _safe_series(bundle.asset_data, "Low", a)
        close = _safe_series(bundle.asset_data, "Close", a)

        missing = [f for f, s in [("High", high), ("Low", low), ("Close", close)] if s is None]
        if missing:
            bundle.warn(f"{a} missing fields from Yahoo: {', '.join(missing)}. Skipping ATR.")
            atr_dict[a] = pd.Series(dtype=float)
            last_close_dict[a] = pd.Series(dtype=float)
            continue

        high = high.dropna()
        low = low.dropna()
        close = close.dropna()
        idx = high.index.intersection(low.index).intersection(close.index)
        high = high.loc[idx]
        low = low.loc[idx]
        close = close.loc[idx]

        if len(idx) < (ATR_EMA_SPAN + 5):
            bundle.warn(f"{a} too few rows for ATR (have {len(idx)}). Skipping ATR.")
            atr_dict[a] = pd.Series(dtype=float)
            last_close_dict[a] = close
            continue

        atr = compute_atr_ema(high, low, close, ATR_EMA_SPAN)
        atr_dict[a] = atr
        last_close_dict[a] = close

    atr_combined = pd.DataFrame(atr_dict)

    vix_percentile = bundle.vix_close.rolling(window=VIX_WINDOW, min_periods=VIX_WINDOW).apply(
        compute_vix_percentile_np, raw=True
    )

    print("[5/6] Building features & severity score...")
    features = build_features(bundle.vix_close, bundle.asset_data)
    bundle.effective_rows = len(features)

    # Determine last_date using intersection of: VIX, features, and at least 2 valid ATRs
    atr_ok = atr_combined.notna().sum(axis=1) >= 2
    valid_atr_dates = atr_combined.index[atr_ok]
    valid_dates = valid_atr_dates.intersection(bundle.vix_close.index).intersection(features.index)

    if len(valid_dates) == 0:
        raise ValueError("No valid dates with sufficient data (ATR + VIX + features).")

    if as_of_date:
        target = pd.Timestamp(as_of_date)
        valid_dates = valid_dates[valid_dates <= target]
        if len(valid_dates) == 0:
            raise ValueError(f"No data available for as_of_date={as_of_date}")

    last_date = valid_dates.max()

    if pd.Timestamp(last_date).date() == today.date():
        bundle.warn("Latest bar may be incomplete (intraday run)")

    vix_pct_series = vix_percentile.loc[:last_date].dropna()
    if len(vix_pct_series) == 0:
        bundle.warn("VIX percentile unavailable, defaulting to 50.0")
        vix_pct = 50.0
    else:
        vix_pct = float(vix_pct_series.iloc[-1])

    vix_level = float(bundle.vix_close.loc[last_date])
    atr_last = atr_combined.loc[last_date]

    vix9d_level = None
    if bundle.vix9d_close is not None and len(bundle.vix9d_close) > 0:
        common = bundle.vix_close.index.intersection(bundle.vix9d_close.index)
        common = common[common <= last_date]
        if len(common) > 0:
            vix9d_level = float(bundle.vix9d_close.loc[common[-1]])

    print("[6/6] Computing severity & bucket assignment...")
    train_features = features.loc[:last_date]

    if train_features.isna().any().any():
        raise ValueError("NaNs in training features after dropna. Data alignment issue.")
    if len(train_features) < 10:
        raise ValueError(f"Too few training rows ({len(train_features)}) for stable scoring.")

    severity = compute_severity_score(train_features)
    n_eff = min(N_BUCKETS, len(severity))
    if n_eff < 2:
        raise ValueError(f"Not enough rows ({len(severity)}) to form buckets.")

    buckets, bucket_stats = assign_buckets(severity, n_eff)
    current_bucket = get_current_bucket(severity, buckets, SMOOTH_WINDOW)
    current_severity = float(severity.iloc[-1])

    if n_eff == N_BUCKETS:
        scaled_bucket = current_bucket
    else:
        scaled_bucket = int(round(current_bucket * (N_BUCKETS - 1) / (n_eff - 1)))
        scaled_bucket = max(0, min(N_BUCKETS - 1, scaled_bucket))

    base_k = K_BUCKET_BASE.get(scaled_bucket, 0.10)

    # Market vol baselines
    spy_close = _safe_series(bundle.asset_data, "Close", "SPY")
    market_vol_etf = 0.15
    if spy_close is not None:
        spy_returns = spy_close.dropna().pct_change()
        mv = compute_realized_vol(spy_returns, last_date, window=20)
        market_vol_etf = mv if not np.isnan(mv) else 0.15

    es_close = _safe_series(bundle.asset_data, "Close", "ES=F")
    market_vol_es = market_vol_etf
    if es_close is not None:
        es_returns = es_close.dropna().pct_change()
        mv = compute_realized_vol(es_returns, last_date, window=20)
        market_vol_es = mv if not np.isnan(mv) else market_vol_etf

    nq_close = _safe_series(bundle.asset_data, "Close", "NQ=F")
    market_vol_nq = market_vol_es
    if nq_close is not None:
        nq_returns = nq_close.dropna().pct_change()
        mv = compute_realized_vol(nq_returns, last_date, window=20)
        market_vol_nq = mv if not np.isnan(mv) else market_vol_es

    rows = []
    decomp_rows = []

    for a in ASSETS:
        atr_pts = float(atr_last[a]) if (a in atr_last.index and not np.isnan(atr_last[a])) else 0.0
        if atr_pts == 0:
            bundle.warn(f"{a} ATR is zero or missing on last_date. Skipping.")
            continue

        try:
            last_close = float(last_close_dict[a].loc[:last_date].iloc[-1])
        except Exception:
            last_close = 0.0

        # Realized vol for asset
        close_series = _safe_series(bundle.asset_data, "Close", a)
        asset_realized_vol = np.nan
        if close_series is not None:
            returns = close_series.dropna().pct_change()
            asset_realized_vol = compute_realized_vol(returns, last_date, window=20)

        if np.isnan(asset_realized_vol):
            if a in ETFS:
                asset_realized_vol = market_vol_etf
            else:
                asset_realized_vol = market_vol_nq if a == "NQ=F" else market_vol_es

        # Baseline vol
        if a in ETFS:
            market_vol = market_vol_etf
        else:
            market_vol = market_vol_nq if a == "NQ=F" else market_vol_es

        # Gap risk: apply to ETFs only
        if a in FUTURES:
            gap_mean, gap_p90 = 0.0, 0.0
        else:
            open_series = _safe_series(bundle.asset_data, "Open", a)
            close_series2 = _safe_series(bundle.asset_data, "Close", a)
            if open_series is None or close_series2 is None or a not in atr_dict:
                gap_mean, gap_p90 = 0.08, 0.15
            else:
                try:
                    gap_mean, gap_p90 = compute_overnight_gap_risk(
                        open_series.dropna(),
                        close_series2.dropna(),
                        atr_dict[a],
                        last_date,
                        OVERNIGHT_GAP_LOOKBACK,
                    )
                except Exception:
                    gap_mean, gap_p90 = 0.08, 0.15

        k, decomp = calculate_dynamic_k(
            base_k=base_k,
            vix_percentile=vix_pct,
            gap_p90=gap_p90,
            asset_realized_vol=float(asset_realized_vol),
            market_realized_vol=float(market_vol),
            vix_level=vix_level,
            vix9d_level=vix9d_level,
        )

        tick = float(TICK_SIZE.get(a, 0.01))
        stop_pts_raw = atr_pts * k
        stop_pts = round_to_tick(stop_pts_raw, tick)

        # Targets as price levels from last close
        tp1_long = round_to_tick(last_close + stop_pts, tick)
        tp2_long = round_to_tick(last_close + stop_pts * 2, tick)
        tp3_long = round_to_tick(last_close + stop_pts * 3, tick)
        tp1_short = round_to_tick(last_close - stop_pts, tick)
        tp2_short = round_to_tick(last_close - stop_pts * 2, tick)
        tp3_short = round_to_tick(last_close - stop_pts * 3, tick)

        rows.append(
            {
                "Asset": a,
                "Type": "ETF" if a in ETFS else "Futures",
                "Close": round(last_close, 2),
                "ATR_pts": round(atr_pts, 2),
                "RealVol": round(float(asset_realized_vol), 4),
                "GapP90": round(float(gap_p90), 4),
                "k": round(float(k), 4),
                "Stop_pts": stop_pts,
                "StopRisk_USD": round(pts_to_usd(a, stop_pts), 2),
                "TP1_L": tp1_long,
                "TP2_L": tp2_long,
                "TP3_L": tp3_long,
                "TP1_S": tp1_short,
                "TP2_S": tp2_short,
                "TP3_S": tp3_short,
            }
        )

        decomp_rows.append(
            {
                "Asset": a,
                "base_k": decomp["base_k"],
                "vix_adj": decomp["vix_adj"],
                "gap_adj": decomp["gap_adj"],
                "vol_adj": decomp["vol_adj"],
                "term_factor": decomp["term_factor"],
                "final_k": decomp["final_k"],
            }
        )

    out = pd.DataFrame(rows)
    decomp_df = pd.DataFrame(decomp_rows)

    if out.empty:
        raise ValueError("No output rows produced. Data fetch or ATR calculation failed.")

    next_trading_day = (pd.Timestamp(last_date) + pd.offsets.BDay(1)).to_pydatetime()

    # Freshness check (business days)
    last_bd = pd.offsets.BDay().rollback(today)
    try:
        stale_range = pd.bdate_range(pd.Timestamp(last_date) + pd.Timedelta(days=1), last_bd)
        days_stale = len(stale_range)
    except Exception:
        days_stale = int((today - pd.Timestamp(last_date)).days)

    if days_stale > DATA_FRESHNESS_DAYS:
        bundle.warn(f"Data is {days_stale} business days old")

    # =========================
    # OUTPUT
    # =========================
    print("\n" + "=" * 70)
    print("DATA SUMMARY")
    print("=" * 70)
    print(f"Data horizon:         {DATA_HORIZON}")
    print(f"Effective rows:       {bundle.effective_rows}")
    print(f"Data as of:           {pd.Timestamp(last_date).strftime('%B %d, %Y')}")
    print(f"For trading on:       {next_trading_day.strftime('%B %d, %Y')}")

    print("\n" + "=" * 70)
    print("VOLATILITY STATE")
    print("=" * 70)
    print(f"VIX Level:            {vix_level:.2f}")
    print(f"VIX Percentile (252d):{vix_pct:.1f}%")
    if vix9d_level is not None:
        ratio = vix9d_level / vix_level if vix_level > 0 else np.nan
        print(f"VIX9D/VIX ratio:      {ratio:.3f} (term structure, not calibrated)")
    print(f"Market Vol (ETF):     {market_vol_etf*100:.1f}%")
    print(f"Market Vol (ES):      {market_vol_es*100:.1f}%")
    print(f"Market Vol (NQ):      {market_vol_nq*100:.1f}%")
    print("Gap adjustments:      ETFs only (cash session). Futures gap set to 0.")

    print("\n" + "=" * 70)
    print("SEVERITY & BUCKET (quantile-based, no clustering)")
    print("=" * 70)
    print(f"Current severity:     {current_severity:.4f} (0=calm, 1=stress)")
    print(f"Current bucket:       {current_bucket} (of 0..{n_eff-1})")
    if n_eff < N_BUCKETS:
        print(f"Scaled bucket:        {scaled_bucket} (mapped to base_k table 0..{N_BUCKETS-1})")
    print(f"Base k:               {base_k:.4f}")
    print(f"Smoothed over:        {SMOOTH_WINDOW} days")
    print("\nBucket Ranges (in realized sample):")
    print(f"{'Bucket':<8} {'Count':<8} {'Sev Min':<10} {'Sev Max':<10}")
    for b in range(n_eff):
        stats = bucket_stats.get(b, {})
        print(
            f"{b:<8} {stats.get('count', 0):<8} "
            f"{stats.get('severity_min', 0):.4f}    {stats.get('severity_max', 0):.4f}"
        )

    print("\n" + "=" * 70)
    print("STOPS & TARGETS (rounded to tick)")
    print("Units: ATR_pts and Stop_pts are points. TP columns are price levels.")
    print("StopRisk_USD: USD per share (ETFs) or USD per contract (Futures).")
    print("=" * 70)
    display_df = out.copy()
    display_df["RealVol"] = display_df["RealVol"].apply(lambda x: f"{x*100:.1f}%")
    display_df["GapP90"] = display_df["GapP90"].apply(lambda x: f"{x*100:.1f}%")
    print(display_df.to_string(index=False))

    print("\n" + "=" * 70)
    print("k DECOMPOSITION")
    print("=" * 70)
    decomp_display = decomp_df.copy()
    for col in ["vix_adj", "gap_adj", "vol_adj"]:
        decomp_display[col] = decomp_display[col].apply(lambda x: f"{x:+.4f}")
    decomp_display["base_k"] = decomp_display["base_k"].apply(lambda x: f"{x:.4f}")
    decomp_display["term_factor"] = decomp_display["term_factor"].apply(lambda x: f"{x:.4f}")
    decomp_display["final_k"] = decomp_display["final_k"].apply(lambda x: f"{x:.4f}")
    print(decomp_display.to_string(index=False))

    # =========================
    # POSITION SIZING
    # =========================
    print("\n" + "=" * 70)
    print("POSITION SIZING - FUTURES (1% account risk, micro contracts)")
    print("=" * 70)
    print("Account Size | Max Risk (1%) | MNQ Micros | MES Micros")
    print("-" * 70)

    nq_row = out[out["Asset"] == "NQ=F"]
    es_row = out[out["Asset"] == "ES=F"]
    nq_stop_usd = float(nq_row["StopRisk_USD"].iloc[0]) if len(nq_row) > 0 else 0.0
    es_stop_usd = float(es_row["StopRisk_USD"].iloc[0]) if len(es_row) > 0 else 0.0

    for acct in [50_000, 100_000, 150_000, 250_000]:
        max_risk = acct * 0.01
        mnq = int(max_risk / (nq_stop_usd / 10)) if nq_stop_usd > 0 else 0
        mes = int(max_risk / (es_stop_usd / 10)) if es_stop_usd > 0 else 0
        print(f"${acct:>9,}   |    ${max_risk:>6,.0f}    |     {mnq:>3}     |     {mes:>3}")

    print("\n" + "=" * 70)
    print("POSITION SIZING - ETFs (1% account risk, shares)")
    print("=" * 70)
    print("Account Size | Max Risk (1%) | QQQ Shares | SPY Shares")
    print("-" * 70)

    qqq_row = out[out["Asset"] == "QQQ"]
    spy_row = out[out["Asset"] == "SPY"]
    qqq_stop_usd = float(qqq_row["StopRisk_USD"].iloc[0]) if len(qqq_row) > 0 else 0.0
    spy_stop_usd = float(spy_row["StopRisk_USD"].iloc[0]) if len(spy_row) > 0 else 0.0

    for acct in [50_000, 100_000, 150_000, 250_000]:
        max_risk = acct * 0.01
        qqq_shares = int(max_risk / qqq_stop_usd) if qqq_stop_usd > 0 else 0
        spy_shares = int(max_risk / spy_stop_usd) if spy_stop_usd > 0 else 0
        print(f"${acct:>9,}   |    ${max_risk:>6,.0f}    |     {qqq_shares:>4}    |     {spy_shares:>4}")

    if bundle.warnings:
        print("\n" + "=" * 70)
        print(f"WARNINGS ({len(bundle.warnings)})")
        print("=" * 70)
        for w in bundle.warnings:
            print(f"  • {w}")

    print("\n" + "=" * 70)
    print("CALIBRATION NOTES (not optimized, treat as starting points)")
    print("=" * 70)
    print("• Buckets: quantile-based on severity score, no clustering")
    print("• base_k: 0.06/0.09/0.13/0.18 by bucket")
    print("• vix_adj: ±0.02 per 50 percentile points")
    print("• gap_adj: 0.20 * (P90 gap - 0.10), capped at 0.04 (ETFs only)")
    print("• vol_adj: 0.05 * (vol_ratio - 1), capped at 0.03")
    print("• term_factor: 1 + 0.03 * tanh(5 * (VIX9D/VIX - 1))")
    print("• Daily bars, session differences not normalized")

    out.to_csv(OUT_STOPS_CSV, index=False)
    decomp_df.to_csv(OUT_DECOMP_CSV, index=False)
    print(f"\n✓ {OUT_STOPS_CSV}")
    print(f"✓ {OUT_DECOMP_CSV}")

    return out, decomp_df, bundle.warnings


if __name__ == "__main__":
    main()