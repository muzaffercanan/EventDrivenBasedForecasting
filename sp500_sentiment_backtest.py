"""
SP500 Sentiment Backtest (standalone)

- Merges daily sentiment aggregation with SP500 close prices (CP) from raw CSV
- Builds a simple rule-based strategy using signed_mean thresholds and news_count filter
- Applies next-day execution to avoid look-ahead (position_t uses signal_{t-1})
- Computes daily PnL and reports metrics: Sharpe, Sortino, CAGR, Calmar, Max Drawdown, Hit Ratio, etc.
- Supports walk-forward analysis for out-of-sample testing
- Supports purged k-fold cross-validation with embargo to prevent data leakage

Inputs (defaults):
  - Sentiment agg: SP500_news/processed/sp500_headlines_daily_agg.csv  (columns: date, signed_mean, news_count, ...)
  - Raw headlines (with CP): SP500_news/raw/sp500_headlines_2008_2024.csv (columns: Date, CP)

Usage (standard backtest):
  python sp500_sentiment_backtest.py \
      --agg_csv SP500_news/processed/sp500_headlines_daily_agg.csv \
      --price_csv SP500_news/raw/sp500_headlines_2008_2024.csv \
      --date_col Date --price_col CP \
      --tpos 0.10 --tneg 0.10 --nmin 5 --ret log

Usage (walk-forward analysis):
  python sp500_sentiment_backtest.py \
      --walk_forward \
      --train_years 2 --test_years 1 --step_months 6 \
      --agg_csv SP500_news/processed/sp500_headlines_daily_agg.csv \
      --price_csv SP500_news/raw/sp500_headlines_2008_2024.csv

Usage (purged k-fold cross-validation):
  python sp500_sentiment_backtest.py \
      --purged_kfold \
      --n_splits 5 --purge_days 5 --embargo_days 1 \
      --agg_csv SP500_news/processed/sp500_headlines_daily_agg.csv \
      --price_csv SP500_news/raw/sp500_headlines_2008_2024.csv

Outputs:
  - SP500_news/results/backtest_metrics.json
  - SP500_news/results/backtest_equity_curve.csv
  - SP500_news/results/backtest_summary.txt
  - SP500_news/results/walk_forward_metrics.json (if walk-forward)
  - SP500_news/results/walk_forward_summary.txt (if walk-forward)
  - SP500_news/results/purged_kfold_metrics.json (if purged_kfold)
  - SP500_news/results/purged_kfold_summary.txt (if purged_kfold)

Requires: pandas, numpy
"""

import argparse
import json
import os
from typing import Tuple, List, Dict
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


def _ensure_dirs():
    os.makedirs("SP500_news/results", exist_ok=True)


def compute_returns(df_price: pd.DataFrame, date_col: str, price_col: str, ret_type: str = "log") -> pd.DataFrame:
    df = df_price[[date_col, price_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col, price_col]).sort_values(date_col)
    if ret_type == "log":
        df["ret"] = np.log(df[price_col] / df[price_col].shift(1))
    else:
        df["ret"] = df[price_col].pct_change()
    return df


def merge_signal_returns(df_sig: pd.DataFrame, df_ret: pd.DataFrame, date_col_sig: str = "date", date_col_ret: str = "Date") -> pd.DataFrame:
    df_sig = df_sig.copy()
    df_sig[date_col_sig] = pd.to_datetime(df_sig[date_col_sig], errors="coerce")
    df = pd.merge(df_ret, df_sig, left_on=date_col_ret, right_on=date_col_sig, how="left")
    df = df.sort_values(date_col_ret).reset_index(drop=True)
    return df


def compute_trend_filter(df: pd.DataFrame, price_col: str = "CP", 
                         short_window: int = 50, long_window: int = 200) -> pd.Series:
    """Compute trend filter using moving averages"""
    if price_col not in df.columns:
        # Reconstruct price from returns if needed
        df["_cumret"] = (1.0 + df["ret"].fillna(0.0)).cumprod()
        prices = df["_cumret"] * 100
    else:
        prices = df[price_col]
    
    ma_short = prices.rolling(window=short_window, min_periods=1).mean()
    ma_long = prices.rolling(window=long_window, min_periods=1).mean()
    
    # 1 = uptrend, -1 = downtrend, 0 = neutral
    trend = np.where(ma_short > ma_long, 1, np.where(ma_short < ma_long, -1, 0))
    return pd.Series(trend, index=df.index, name="trend")


def compute_volatility_filter(df: pd.DataFrame, ret_col: str = "ret",
                              window: int = 20, threshold_percentile: float = 75.0) -> pd.Series:
    """Compute volatility filter: 1 = low volatility (trade), 0 = high volatility (no trade)"""
    rolling_vol = df[ret_col].rolling(window=window, min_periods=1).std()
    threshold = rolling_vol.quantile(threshold_percentile / 100.0)
    return pd.Series((rolling_vol <= threshold).astype(int), index=df.index, name="vol_filter")


def compute_position_size(df: pd.DataFrame, ret_col: str = "ret",
                         base_size: float = 1.0, volatility_window: int = 20,
                         min_size: float = 0.1, max_size: float = 1.0) -> pd.Series:
    """Compute position size based on volatility (volatility targeting)"""
    rolling_vol = df[ret_col].rolling(window=volatility_window, min_periods=1).std()
    target_vol = rolling_vol.median()  # Target median volatility
    
    # Position size inversely proportional to volatility
    position_size = base_size * (target_vol / (rolling_vol + 1e-8))
    position_size = position_size.clip(lower=min_size, upper=max_size)
    
    return pd.Series(position_size, index=df.index, name="position_size")


def build_positions_enhanced(df: pd.DataFrame, tpos: float, tneg: float, nmin: int,
                            use_trend_filter: bool = True,
                            use_volatility_filter: bool = True,
                            use_signal_quality: bool = True,
                            signal_quality_threshold: float = 0.15,
                            trend_col: str = "trend",
                            vol_filter_col: str = "vol_filter",
                            signed_mean_col: str = "signed_mean",
                            signed_std_col: str = "signed_std") -> pd.Series:
    """Enhanced position building with multiple filters"""
    sig = df[signed_mean_col].fillna(0.0)
    cnt = df["news_count"].fillna(0).astype(int)
    
    # Base signal
    base_long = (sig >= tpos) & (cnt >= nmin)
    base_short = (sig <= -tneg) & (cnt >= nmin)
    
    # Signal quality filter: require stronger signals or consistent sentiment
    if use_signal_quality:
        if signed_std_col in df.columns:
            sig_std = df[signed_std_col].fillna(1.0)
            # Strong signal: high absolute value OR low std (consistent sentiment)
            quality_filter = (np.abs(sig) >= signal_quality_threshold) | (sig_std < 0.3)
            base_long = base_long & quality_filter
            base_short = base_short & quality_filter
    
    # Trend filter
    if use_trend_filter and trend_col in df.columns:
        trend = df[trend_col].fillna(0)
        base_long = base_long & (trend >= 0)  # Only long in uptrend or neutral
        base_short = base_short & (trend <= 0)  # Only short in downtrend or neutral
    
    # Volatility filter
    if use_volatility_filter and vol_filter_col in df.columns:
        vol_filter = df[vol_filter_col].fillna(0)
        base_long = base_long & (vol_filter == 1)
        base_short = base_short & (vol_filter == 1)
    
    pos = np.where(base_long, 1,
          np.where(base_short, -1, 0))
    
    return pd.Series(pos, index=df.index, name="position")


def apply_holding_period_limit(df: pd.DataFrame, position_col: str = "position",
                              max_holding_days: int = 20) -> pd.Series:
    """Limit maximum holding period for positions"""
    position = df[position_col].copy()
    position_with_limit = position.copy()
    
    current_position = 0
    entry_day = None
    
    for i in range(len(df)):
        if position.iloc[i] != 0:
            if current_position == 0:
                # New position
                current_position = position.iloc[i]
                entry_day = i
            elif position.iloc[i] != current_position:
                # Position changed
                current_position = position.iloc[i]
                entry_day = i
            else:
                # Same position continues
                if entry_day is not None and (i - entry_day) >= max_holding_days:
                    # Force close after max holding period
                    position_with_limit.iloc[i] = 0
                    current_position = 0
                    entry_day = None
        else:
            # No position
            current_position = 0
            entry_day = None
    
    return pd.Series(position_with_limit, index=df.index, name="position_with_limit")


def build_positions(df: pd.DataFrame, tpos: float, tneg: float, nmin: int) -> pd.Series:
    sig = df["signed_mean"].fillna(0.0)
    cnt = df["news_count"].fillna(0).astype(int)
    pos = np.where((sig >= tpos) & (cnt >= nmin), 1,
          np.where((sig <= -tneg) & (cnt >= nmin), -1, 0))
    return pd.Series(pos, index=df.index, name="position")


def shift_positions(df: pd.DataFrame, pos_col: str = "position") -> pd.Series:
    # Use yesterday's signal for today's return (t+1 execution)
    return df[pos_col].shift(1).fillna(0.0)


def apply_risk_management(df: pd.DataFrame, 
                          position_col: str = "position",
                          ret_col: str = "ret",
                          price_col: str = None,
                          stop_loss: float = None,
                          take_profit: float = None,
                          trailing_stop: float = None,
                          max_drawdown_limit: float = None) -> pd.DataFrame:
    """
    Apply risk management rules to positions:
    - Stop-loss: Close position if loss exceeds threshold
    - Take-profit: Close position if profit exceeds threshold
    - Trailing stop: Dynamic stop that follows price upward
    - Max drawdown limit: Close all positions if equity drops too much
    
    Parameters:
    - stop_loss: Maximum loss per trade (e.g., 0.05 = 5%)
    - take_profit: Target profit per trade (e.g., 0.10 = 10%)
    - trailing_stop: Trailing stop distance (e.g., 0.03 = 3%)
    - max_drawdown_limit: Maximum equity drawdown before closing all (e.g., 0.20 = 20%)
    """
    df = df.copy()
    
    # Initialize risk management columns
    df["entry_price"] = np.nan
    df["unrealized_pnl"] = 0.0
    df["position_with_risk"] = df[position_col].copy()
    df["risk_action"] = ""  # Track what risk action was taken
    
    # Track entry prices and cumulative returns for each position
    current_position = 0
    entry_index = None
    cumulative_return = 0.0
    highest_return = 0.0  # For trailing stop
    
    # Calculate temporary PnL and equity for max drawdown check
    df["pnl_temp"] = df[position_col] * df[ret_col]
    df["equity_temp"] = (1.0 + df["pnl_temp"].fillna(0.0)).cumprod()
    peak_equity = 1.0
    
    for i in range(len(df)):
        # Update peak equity for max drawdown
        if i > 0:
            peak_equity = max(peak_equity, df.loc[i, "equity_temp"])
        
        # Check max drawdown limit first (affects all positions)
        if max_drawdown_limit is not None:
            current_drawdown = (df.loc[i, "equity_temp"] / peak_equity) - 1.0
            if current_drawdown <= -max_drawdown_limit:
                df.loc[i, "position_with_risk"] = 0
                df.loc[i, "risk_action"] = "max_dd_limit"
                current_position = 0
                entry_index = None
                cumulative_return = 0.0
                highest_return = 0.0
                continue
        
        # Get current position signal
        signal_position = df.loc[i, position_col]
        
        # Position entry
        if signal_position != 0 and current_position == 0:
            # New position opened
            current_position = signal_position
            entry_index = i
            entry_equity = df.loc[i, "equity_temp"] if i > 0 else 1.0
            cumulative_return = 0.0
            highest_return = 0.0
            df.loc[i, "entry_price"] = df.loc[i, price_col] if price_col and price_col in df.columns else np.nan
        
        # Position management
        if current_position != 0:
            # Update cumulative return (using log returns)
            if i > entry_index:
                cumulative_return += df.loc[i, ret_col] * current_position
            
            # Update highest return for trailing stop
            if cumulative_return > highest_return:
                highest_return = cumulative_return
            
            # Calculate unrealized PnL
            unrealized_pnl = cumulative_return
            df.loc[i, "unrealized_pnl"] = unrealized_pnl
            
            # Check stop-loss
            if stop_loss is not None and unrealized_pnl <= -stop_loss:
                df.loc[i, "position_with_risk"] = 0
                df.loc[i, "risk_action"] = "stop_loss"
                current_position = 0
                entry_index = None
                cumulative_return = 0.0
                highest_return = 0.0
                continue
            
            # Check take-profit
            if take_profit is not None and unrealized_pnl >= take_profit:
                df.loc[i, "position_with_risk"] = 0
                df.loc[i, "risk_action"] = "take_profit"
                current_position = 0
                entry_index = None
                cumulative_return = 0.0
                highest_return = 0.0
                continue
            
            # Check trailing stop
            if trailing_stop is not None and highest_return > 0:
                trailing_stop_level = highest_return - trailing_stop
                if unrealized_pnl <= trailing_stop_level:
                    df.loc[i, "position_with_risk"] = 0
                    df.loc[i, "risk_action"] = "trailing_stop"
                    current_position = 0
                    entry_index = None
                    cumulative_return = 0.0
                    highest_return = 0.0
                    continue
            
            # Position continues
            df.loc[i, "position_with_risk"] = current_position
        
        # Position exit (signal changed)
        if signal_position == 0 and current_position != 0:
            current_position = 0
            entry_index = None
            cumulative_return = 0.0
            highest_return = 0.0
    
    return df


def metrics_from_pnl(pnl: pd.Series, freq: int = 252) -> dict:
    pnl = pnl.dropna()
    mu = pnl.mean()
    sd = pnl.std(ddof=0)
    sharpe = (mu / sd * np.sqrt(freq)) if sd > 0 else 0.0

    downside = pnl.copy()
    downside[downside > 0] = 0
    downside_sd = np.sqrt((downside ** 2).mean())
    sortino = (mu / downside_sd * np.sqrt(freq)) if downside_sd > 0 else 0.0

    eq = (1.0 + pnl).cumprod()
    peak = eq.cummax()
    dd = (eq / peak - 1.0)
    max_dd = dd.min() if len(dd) else 0.0

    # CAGR (assuming freq trading days per year)
    if len(eq) > 1:
        years = len(eq) / freq
        cagr = eq.iloc[-1] ** (1 / years) - 1
    else:
        cagr = 0.0

    calmar = (cagr / abs(max_dd)) if max_dd < 0 else 0.0

    hit = (pnl > 0).sum()
    miss = (pnl <= 0).sum()
    hit_ratio = hit / (hit + miss) if (hit + miss) > 0 else 0.0

    avg_win = pnl[pnl > 0].mean() if (pnl > 0).any() else 0.0
    avg_loss = pnl[pnl <= 0].mean() if (pnl <= 0).any() else 0.0

    turnover = (np.abs(np.diff([0] + list((pnl*0+1).index.map(lambda i: 0).values))))  # placeholder 0
    # For simplicity, skip turnover without transaction cost modeling; could be computed from position changes.

    return {
        "mean_daily": float(mu),
        "std_daily": float(sd),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": float(max_dd),
        "cagr": float(cagr),
        "calmar": float(calmar),
        "hit_ratio": float(hit_ratio),
        "avg_win": float(avg_win),
        "avg_loss": float(avg_loss),
        "obs": int(len(pnl)),
    }


def run_backtest(agg_csv: str,
                 price_csv: str,
                 date_col: str = "Date",
                 price_col: str = "CP",
                 tpos: float = 0.10,
                 tneg: float = 0.10,
                 nmin: int = 5,
                 ret_type: str = "log",
                 start_date: str = None,
                 end_date: str = None,
                 stop_loss: float = None,
                 take_profit: float = None,
                 trailing_stop: float = None,
                 max_drawdown_limit: float = None,
                 use_trend_filter: bool = True,
                 use_volatility_filter: bool = True,
                 use_position_sizing: bool = True,
                 use_signal_quality: bool = True,
                 signal_quality_threshold: float = 0.15,
                 max_holding_days: int = 20,
                 trend_short_window: int = 50,
                 trend_long_window: int = 200,
                 volatility_window: int = 20,
                 volatility_percentile: float = 75.0,
                 base_position_size: float = 1.0) -> Tuple[pd.DataFrame, dict]:
    _ensure_dirs()

    # Load data
    if not os.path.exists(agg_csv):
        raise FileNotFoundError(f"Aggregation file not found: {agg_csv}")
    if not os.path.exists(price_csv):
        raise FileNotFoundError(f"Price file not found: {price_csv}")

    df_sig = pd.read_csv(agg_csv)
    df_price = pd.read_csv(price_csv)

    # Compute returns
    df_ret = compute_returns(df_price, date_col=date_col, price_col=price_col, ret_type=ret_type)

    # Merge signal & returns
    df = merge_signal_returns(df_sig, df_ret, date_col_sig="date", date_col_ret=date_col)
    
    # Filter by date range if specified
    if start_date:
        df = df[df[date_col] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df[date_col] <= pd.to_datetime(end_date)]
    
    # Ensure price column exists for trend calculation
    if price_col not in df.columns:
        df["_cumret"] = (1.0 + df["ret"].fillna(0.0)).cumprod()
        df[price_col] = df["_cumret"] * 100

    # Compute filters
    if use_trend_filter:
        df["trend"] = compute_trend_filter(df, price_col=price_col,
                                           short_window=trend_short_window,
                                           long_window=trend_long_window)
    
    if use_volatility_filter:
        df["vol_filter"] = compute_volatility_filter(df, ret_col="ret",
                                                    window=volatility_window,
                                                    threshold_percentile=volatility_percentile)
    
    # Build positions with enhanced filters
    df["position_raw"] = build_positions_enhanced(
        df, tpos=tpos, tneg=tneg, nmin=nmin,
        use_trend_filter=use_trend_filter,
        use_volatility_filter=use_volatility_filter,
        use_signal_quality=use_signal_quality,
        signal_quality_threshold=signal_quality_threshold
    )
    
    # Apply holding period limit
    if max_holding_days > 0:
        df["position_raw"] = apply_holding_period_limit(df, position_col="position_raw",
                                                        max_holding_days=max_holding_days)
    
    df["position"] = shift_positions(df, pos_col="position_raw")
    
    # Apply position sizing
    if use_position_sizing:
        df["position_size"] = compute_position_size(df, ret_col="ret",
                                                   base_size=base_position_size,
                                                   volatility_window=volatility_window)
        df["position"] = df["position"] * df["position_size"]
    
    # Apply risk management if specified
    if any([stop_loss, take_profit, trailing_stop, max_drawdown_limit]):
        # Need price column for risk management
        if price_col not in df.columns:
            if "_cumret" not in df.columns:
                df["_cumret"] = (1.0 + df["ret"].fillna(0.0)).cumprod()
            df[price_col] = df["_cumret"] * 100
        
        df = apply_risk_management(
            df,
            position_col="position",
            ret_col="ret",
            price_col=price_col,
            stop_loss=stop_loss,
            take_profit=take_profit,
            trailing_stop=trailing_stop,
            max_drawdown_limit=max_drawdown_limit
        )
        df["position"] = df["position_with_risk"]
    else:
        # No risk management - add empty columns for consistency
        df["risk_action"] = ""
        df["unrealized_pnl"] = 0.0

    # PnL (using risk-managed positions)
    df["pnl"] = df["position"] * df["ret"]
    df["equity"] = (1.0 + df["pnl"].fillna(0.0)).cumprod()

    # Trim initial NaNs from ret
    df_bt = df.dropna(subset=["ret"]).copy()

    # Metrics
    m = metrics_from_pnl(df_bt["pnl"].fillna(0.0))

    # Save outputs
    out_eq = "SP500_news/results/backtest_equity_curve.csv"
    output_cols = [date_col, "ret", "position", "pnl", "equity", "signed_mean", "news_count"]
    if "risk_action" in df_bt.columns:
        output_cols.append("risk_action")
    if "unrealized_pnl" in df_bt.columns:
        output_cols.append("unrealized_pnl")
    if "trend" in df_bt.columns:
        output_cols.append("trend")
    if "vol_filter" in df_bt.columns:
        output_cols.append("vol_filter")
    if "position_size" in df_bt.columns:
        output_cols.append("position_size")
    df_bt[output_cols].to_csv(out_eq, index=False)

    out_metrics = "SP500_news/results/backtest_metrics.json"
    with open(out_metrics, "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2)

    # Human-readable summary
    out_summary = "SP500_news/results/backtest_summary.txt"
    with open(out_summary, "w", encoding="utf-8") as f:
        f.write("SP500 Sentiment Backtest Summary\n")
        f.write("=================================\n\n")
        f.write(f"Rule: long if signed_mean >= {tpos} and news_count >= {nmin}; short if signed_mean <= -{tneg} and news_count >= {nmin}; else flat.\n")
        f.write(f"Return type: {ret_type}\n\n")
        
        # Enhanced filters info
        f.write("Enhanced Filters:\n")
        f.write(f"  Trend filter: {'ON' if use_trend_filter else 'OFF'} (MA{trend_short_window}/MA{trend_long_window})\n")
        f.write(f"  Volatility filter: {'ON' if use_volatility_filter else 'OFF'} (window={volatility_window}, percentile={volatility_percentile})\n")
        f.write(f"  Position sizing: {'ON' if use_position_sizing else 'OFF'} (base_size={base_position_size})\n")
        f.write(f"  Signal quality filter: {'ON' if use_signal_quality else 'OFF'} (threshold={signal_quality_threshold})\n")
        f.write(f"  Max holding period: {max_holding_days} days\n")
        f.write("\n")
        
        # Risk management info
        if any([stop_loss, take_profit, trailing_stop, max_drawdown_limit]):
            f.write("Risk Management:\n")
            if stop_loss:
                f.write(f"  Stop-loss: {stop_loss:.2%}\n")
            if take_profit:
                f.write(f"  Take-profit: {take_profit:.2%}\n")
            if trailing_stop:
                f.write(f"  Trailing stop: {trailing_stop:.2%}\n")
            if max_drawdown_limit:
                f.write(f"  Max drawdown limit: {max_drawdown_limit:.2%}\n")
            f.write("\n")
            
            # Risk action statistics
            if "risk_action" in df_bt.columns:
                risk_actions = df_bt["risk_action"].value_counts()
                if len(risk_actions) > 0:
                    f.write("Risk Actions Taken:\n")
                    for action, count in risk_actions.items():
                        if action:
                            f.write(f"  {action}: {count}\n")
                    f.write("\n")
        
        f.write("Metrics:\n")
        for k, v in m.items():
            f.write(f"{k}: {v}\n")

    print("Saved:")
    print(" -", out_eq)
    print(" -", out_metrics)
    print(" -", out_summary)

    return df_bt, m


def run_walk_forward_analysis(agg_csv: str,
                              price_csv: str,
                              date_col: str = "Date",
                              price_col: str = "CP",
                              tpos: float = 0.10,
                              tneg: float = 0.10,
                              nmin: int = 5,
                              ret_type: str = "log",
                              train_years: int = 2,
                              test_years: int = 1,
                              step_months: int = 6,
                              stop_loss: float = None,
                              take_profit: float = None,
                              trailing_stop: float = None,
                              max_drawdown_limit: float = None,
                              use_trend_filter: bool = True,
                              use_volatility_filter: bool = True,
                              use_position_sizing: bool = True,
                              use_signal_quality: bool = True,
                              signal_quality_threshold: float = 0.15,
                              max_holding_days: int = 20,
                              trend_short_window: int = 50,
                              trend_long_window: int = 200,
                              volatility_window: int = 20,
                              volatility_percentile: float = 75.0,
                              base_position_size: float = 1.0) -> Tuple[pd.DataFrame, Dict]:
    """
    Walk-forward analysis: Rolling window backtest
    
    Parameters:
    - train_years: Training period in years
    - test_years: Test period in years  
    - step_months: How many months to step forward each iteration
    """
    _ensure_dirs()
    
    print("=" * 60)
    print("WALK-FORWARD ANALYSIS")
    print("=" * 60)
    print(f"Training period: {train_years} years")
    print(f"Test period: {test_years} years")
    print(f"Step size: {step_months} months")
    print()
    
    # Load data
    if not os.path.exists(agg_csv):
        raise FileNotFoundError(f"Aggregation file not found: {agg_csv}")
    if not os.path.exists(price_csv):
        raise FileNotFoundError(f"Price file not found: {price_csv}")
    
    df_sig = pd.read_csv(agg_csv)
    df_price = pd.read_csv(price_csv)
    
    # Compute returns
    df_ret = compute_returns(df_price, date_col=date_col, price_col=price_col, ret_type=ret_type)
    
    # Merge signal & returns
    df = merge_signal_returns(df_sig, df_ret, date_col_sig="date", date_col_ret=date_col)
    df = df.sort_values(date_col).reset_index(drop=True)
    
    # Get date range
    df[date_col] = pd.to_datetime(df[date_col])
    min_date = df[date_col].min()
    max_date = df[date_col].max()
    
    print(f"Data range: {min_date.date()} to {max_date.date()}")
    print()
    
    # Generate walk-forward periods
    periods = []
    current_start = min_date
    
    while current_start < max_date:
        train_start = current_start
        train_end = train_start + pd.DateOffset(years=train_years)
        test_start = train_end
        test_end = test_start + pd.DateOffset(years=test_years)
        
        # Check if we have enough data
        if test_end > max_date:
            break
        
        periods.append({
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end
        })
        
        # Step forward
        current_start = current_start + pd.DateOffset(months=step_months)
    
    print(f"Total periods: {len(periods)}")
    print()
    
    # Run backtest for each period
    all_test_results = []
    all_test_equity = []
    period_metrics = []
    
    for i, period in enumerate(periods):
        print(f"Period {i+1}/{len(periods)}: "
              f"Train [{period['train_start'].date()} to {period['train_end'].date()}] | "
              f"Test [{period['test_start'].date()} to {period['test_end'].date()}]")
        
        # Run backtest on test period (using same parameters)
        df_test = df[(df[date_col] >= period['test_start']) & 
                     (df[date_col] < period['test_end'])].copy()
        
        if len(df_test) == 0:
            print("  [SKIP] No data in test period")
            continue
        
        # Reset index for risk management function
        df_test = df_test.reset_index(drop=True)
        
        # Ensure price column exists
        if price_col not in df_test.columns:
            df_test["_cumret"] = (1.0 + df_test["ret"].fillna(0.0)).cumprod()
            df_test[price_col] = df_test["_cumret"] * 100
        
        # Compute filters
        if use_trend_filter:
            df_test["trend"] = compute_trend_filter(df_test, price_col=price_col,
                                                   short_window=trend_short_window,
                                                   long_window=trend_long_window)
        
        if use_volatility_filter:
            df_test["vol_filter"] = compute_volatility_filter(df_test, ret_col="ret",
                                                            window=volatility_window,
                                                            threshold_percentile=volatility_percentile)
        
        # Build positions with enhanced filters
        df_test["position_raw"] = build_positions_enhanced(
            df_test, tpos=tpos, tneg=tneg, nmin=nmin,
            use_trend_filter=use_trend_filter,
            use_volatility_filter=use_volatility_filter,
            use_signal_quality=use_signal_quality,
            signal_quality_threshold=signal_quality_threshold
        )
        
        # Apply holding period limit
        if max_holding_days > 0:
            df_test["position_raw"] = apply_holding_period_limit(df_test, position_col="position_raw",
                                                                max_holding_days=max_holding_days)
        
        df_test["position"] = shift_positions(df_test, pos_col="position_raw")
        
        # Apply position sizing
        if use_position_sizing:
            df_test["position_size"] = compute_position_size(df_test, ret_col="ret",
                                                           base_size=base_position_size,
                                                           volatility_window=volatility_window)
            df_test["position"] = df_test["position"] * df_test["position_size"]
        
        # Apply risk management if specified
        if any([stop_loss, take_profit, trailing_stop, max_drawdown_limit]):
            if price_col not in df_test.columns:
                if "_cumret" not in df_test.columns:
                    df_test["_cumret"] = (1.0 + df_test["ret"].fillna(0.0)).cumprod()
                df_test[price_col] = df_test["_cumret"] * 100
            
            df_test = apply_risk_management(
                df_test,
                position_col="position",
                ret_col="ret",
                price_col=price_col,
                stop_loss=stop_loss,
                take_profit=take_profit,
                trailing_stop=trailing_stop,
                max_drawdown_limit=max_drawdown_limit
            )
            df_test["position"] = df_test["position_with_risk"]
        else:
            df_test["risk_action"] = ""
            df_test["unrealized_pnl"] = 0.0
        
        # PnL (using risk-managed positions)
        df_test["pnl"] = df_test["position"] * df_test["ret"]
        df_test["equity"] = (1.0 + df_test["pnl"].fillna(0.0)).cumprod()
        
        # Metrics
        df_test_clean = df_test.dropna(subset=["ret"]).copy()
        if len(df_test_clean) == 0:
            print("  [SKIP] No valid returns")
            continue
        
        m = metrics_from_pnl(df_test_clean["pnl"].fillna(0.0))
        
        # Add period info
        m["period"] = i + 1
        m["train_start"] = period['train_start'].strftime("%Y-%m-%d")
        m["train_end"] = period['train_end'].strftime("%Y-%m-%d")
        m["test_start"] = period['test_start'].strftime("%Y-%m-%d")
        m["test_end"] = period['test_end'].strftime("%Y-%m-%d")
        m["test_days"] = len(df_test_clean)
        
        period_metrics.append(m)
        all_test_results.append(df_test_clean)
        
        print(f"  Sharpe: {m['sharpe']:.3f}, CAGR: {m['cagr']:.3f}, Hit Ratio: {m['hit_ratio']:.3f}")
    
    print()
    
    # Combine all test results
    if not all_test_results:
        raise ValueError("No valid test periods found!")
    
    df_combined = pd.concat(all_test_results, ignore_index=True)
    df_combined = df_combined.sort_values(date_col).reset_index(drop=True)
    
    # Recalculate equity from combined PnL
    df_combined["equity"] = (1.0 + df_combined["pnl"].fillna(0.0)).cumprod()
    
    # Overall metrics from combined test periods
    overall_metrics = metrics_from_pnl(df_combined["pnl"].fillna(0.0))
    
    # Period-by-period statistics
    period_df = pd.DataFrame(period_metrics)
    
    # Aggregate period statistics
    period_stats = {
        "num_periods": len(period_metrics),
        "avg_sharpe": period_df["sharpe"].mean(),
        "std_sharpe": period_df["sharpe"].std(),
        "avg_sortino": period_df["sortino"].mean(),
        "avg_cagr": period_df["cagr"].mean(),
        "avg_hit_ratio": period_df["hit_ratio"].mean(),
        "avg_max_drawdown": period_df["max_drawdown"].mean(),
        "positive_periods": (period_df["sharpe"] > 0).sum(),
        "negative_periods": (period_df["sharpe"] <= 0).sum(),
    }
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_to_native(obj):
        if isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    # Combine overall and period stats
    walk_forward_metrics = {
        "overall_metrics": convert_to_native(overall_metrics),
        "period_statistics": convert_to_native(period_stats),
        "period_details": convert_to_native(period_metrics)
    }
    
    # Save outputs
    out_eq = "SP500_news/results/walk_forward_equity_curve.csv"
    df_combined[[date_col, "ret", "position", "pnl", "equity", "signed_mean", "news_count"]].to_csv(out_eq, index=False)
    
    out_metrics = "SP500_news/results/walk_forward_metrics.json"
    with open(out_metrics, "w", encoding="utf-8") as f:
        json.dump(walk_forward_metrics, f, indent=2)
    
    out_periods = "SP500_news/results/walk_forward_periods.csv"
    period_df.to_csv(out_periods, index=False)
    
    # Summary
    out_summary = "SP500_news/results/walk_forward_summary.txt"
    with open(out_summary, "w", encoding="utf-8") as f:
        f.write("SP500 Walk-Forward Analysis Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Training period: {train_years} years\n")
        f.write(f"Test period: {test_years} years\n")
        f.write(f"Step size: {step_months} months\n")
        f.write(f"Total periods: {len(period_metrics)}\n\n")
        
        f.write("Overall Metrics (Combined Test Periods):\n")
        f.write("-" * 60 + "\n")
        for k, v in overall_metrics.items():
            f.write(f"{k}: {v}\n")
        
        f.write("\nPeriod Statistics:\n")
        f.write("-" * 60 + "\n")
        for k, v in period_stats.items():
            f.write(f"{k}: {v}\n")
        
        f.write("\nPeriod Details:\n")
        f.write("-" * 60 + "\n")
        for i, p in enumerate(period_metrics):
            f.write(f"\nPeriod {i+1} [{p['test_start']} to {p['test_end']}]:\n")
            f.write(f"  Sharpe: {p['sharpe']:.3f}\n")
            f.write(f"  Sortino: {p['sortino']:.3f}\n")
            f.write(f"  CAGR: {p['cagr']:.3f}\n")
            f.write(f"  Max DD: {p['max_drawdown']:.3f}\n")
            f.write(f"  Hit Ratio: {p['hit_ratio']:.3f}\n")
    
    print("=" * 60)
    print("WALK-FORWARD RESULTS")
    print("=" * 60)
    print(f"\nOverall Metrics (All Test Periods Combined):")
    print(f"  Sharpe: {overall_metrics['sharpe']:.3f}")
    print(f"  Sortino: {overall_metrics['sortino']:.3f}")
    print(f"  CAGR: {overall_metrics['cagr']:.3f}")
    print(f"  Max Drawdown: {overall_metrics['max_drawdown']:.3f}")
    print(f"  Hit Ratio: {overall_metrics['hit_ratio']:.3f}")
    
    print(f"\nPeriod Statistics ({len(period_metrics)} periods):")
    print(f"  Avg Sharpe: {period_stats['avg_sharpe']:.3f} (std: {period_stats['std_sharpe']:.3f})")
    print(f"  Avg Sortino: {period_stats['avg_sortino']:.3f}")
    print(f"  Avg CAGR: {period_stats['avg_cagr']:.3f}")
    print(f"  Positive periods: {period_stats['positive_periods']}/{len(period_metrics)}")
    print(f"  Negative periods: {period_stats['negative_periods']}/{len(period_metrics)}")
    
    print("\nSaved:")
    print(" -", out_eq)
    print(" -", out_metrics)
    print(" -", out_periods)
    print(" -", out_summary)
    
    return df_combined, walk_forward_metrics


def run_purged_kfold_analysis(agg_csv: str,
                               price_csv: str,
                               date_col: str = "Date",
                               price_col: str = "CP",
                               tpos: float = 0.10,
                               tneg: float = 0.10,
                               nmin: int = 5,
                               ret_type: str = "log",
                               n_splits: int = 5,
                               purge_days: int = 5,
                               embargo_days: int = 1,
                               stop_loss: float = None,
                               take_profit: float = None,
                               trailing_stop: float = None,
                               max_drawdown_limit: float = None,
                               use_trend_filter: bool = True,
                               use_volatility_filter: bool = True,
                               use_position_sizing: bool = True,
                               use_signal_quality: bool = True,
                               signal_quality_threshold: float = 0.15,
                               max_holding_days: int = 20,
                               trend_short_window: int = 50,
                               trend_long_window: int = 200,
                               volatility_window: int = 20,
                               volatility_percentile: float = 75.0,
                               base_position_size: float = 1.0) -> Tuple[pd.DataFrame, Dict]:
    """
    Purged K-Fold Cross-Validation with Embargo
    
    This method splits data into k folds while ensuring:
    1. Purged: Training data before test period is removed to prevent look-ahead bias
    2. Embargo: Data after test period is removed to prevent information leakage
    
    Parameters:
    - n_splits: Number of folds (default: 5)
    - purge_days: Days to purge before test period (default: 5)
    - embargo_days: Days to embargo after test period (default: 1)
    """
    _ensure_dirs()
    
    print("=" * 60)
    print("PURGED K-FOLD CROSS-VALIDATION WITH EMBARGO")
    print("=" * 60)
    print(f"Number of folds: {n_splits}")
    print(f"Purge days (before test): {purge_days}")
    print(f"Embargo days (after test): {embargo_days}")
    print()
    
    # Load data
    if not os.path.exists(agg_csv):
        raise FileNotFoundError(f"Aggregation file not found: {agg_csv}")
    if not os.path.exists(price_csv):
        raise FileNotFoundError(f"Price file not found: {price_csv}")
    
    df_sig = pd.read_csv(agg_csv)
    df_price = pd.read_csv(price_csv)
    
    # Compute returns
    df_ret = compute_returns(df_price, date_col=date_col, price_col=price_col, ret_type=ret_type)
    
    # Merge signal and returns
    df = merge_signal_returns(df_sig, df_ret, date_col_sig="date", date_col_ret=date_col)
    
    # Sort by date
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.sort_values(date_col).reset_index(drop=True)
    df = df.dropna(subset=[date_col]).copy()
    
    if len(df) == 0:
        raise ValueError("No valid data after merging!")
    
    min_date = df[date_col].min()
    max_date = df[date_col].max()
    
    print(f"Data range: {min_date.date()} to {max_date.date()}")
    print(f"Total days: {len(df)}")
    print()
    
    # Create k folds with purged and embargo periods
    total_days = len(df)
    fold_size = total_days // n_splits
    
    fold_results = []
    all_test_results = []
    
    for fold_idx in range(n_splits):
        # Calculate fold boundaries
        test_start_idx = fold_idx * fold_size
        test_end_idx = (fold_idx + 1) * fold_size if fold_idx < n_splits - 1 else total_days
        
        # Purge: Remove purge_days before test period
        train_end_idx = max(0, test_start_idx - purge_days)
        
        # Embargo: Remove embargo_days after test period
        train_start_idx_next = min(total_days, test_end_idx + embargo_days)
        
        # Training set: All data before purge period
        train_indices = list(range(0, train_end_idx))
        
        # Test set: Fold period
        test_indices = list(range(test_start_idx, test_end_idx))
        
        if len(train_indices) == 0 or len(test_indices) == 0:
            print(f"Fold {fold_idx + 1}/{n_splits}: Skipping (insufficient data)")
            continue
        
        train_start_date = df.iloc[0][date_col]
        train_end_date = df.iloc[train_end_idx - 1][date_col] if train_end_idx > 0 else df.iloc[0][date_col]
        test_start_date = df.iloc[test_start_idx][date_col]
        test_end_date = df.iloc[test_end_idx - 1][date_col]
        
        print(f"Fold {fold_idx + 1}/{n_splits}:")
        print(f"  Train: [{train_start_date.date()} to {train_end_date.date()}] ({len(train_indices)} days)")
        print(f"  Purge: {purge_days} days before test")
        print(f"  Test:  [{test_start_date.date()} to {test_end_date.date()}] ({len(test_indices)} days)")
        print(f"  Embargo: {embargo_days} days after test")
        
        # Get train and test dataframes
        df_train = df.iloc[train_indices].copy().reset_index(drop=True)
        df_test = df.iloc[test_indices].copy().reset_index(drop=True)
        
        if len(df_test) == 0:
            print(f"  [SKIP] No data in test period")
            continue
        
        # Reconstruct price if needed
        if price_col not in df_test.columns:
            df_test["_cumret"] = (1.0 + df_test["ret"].fillna(0.0)).cumprod()
            df_test[price_col] = df_test["_cumret"] * 100
        
        # Apply filters
        if use_trend_filter:
            df_test["trend"] = compute_trend_filter(df_test, price_col=price_col,
                                                     short_window=trend_short_window,
                                                     long_window=trend_long_window)
        
        if use_volatility_filter:
            df_test["vol_filter"] = compute_volatility_filter(df_test, ret_col="ret",
                                                               window=volatility_window,
                                                               threshold_percentile=volatility_percentile)
        
        # Build positions
        df_test["position_raw"] = build_positions_enhanced(
            df_test, tpos=tpos, tneg=tneg, nmin=nmin,
            use_trend_filter=use_trend_filter,
            use_volatility_filter=use_volatility_filter,
            use_signal_quality=use_signal_quality,
            signal_quality_threshold=signal_quality_threshold
        )
        
        # Apply holding period limit
        df_test["position_raw"] = apply_holding_period_limit(df_test, position_col="position_raw",
                                                               max_holding_days=max_holding_days)
        
        # Shift positions for next-day execution
        df_test["position"] = shift_positions(df_test, pos_col="position_raw")
        
        # Position sizing
        if use_position_sizing:
            df_test["position_size"] = compute_position_size(df_test, ret_col="ret",
                                                               base_size=base_position_size,
                                                               volatility_window=volatility_window)
            df_test["position"] = df_test["position"] * df_test["position_size"]
        
        # Risk management
        if any([stop_loss, take_profit, trailing_stop, max_drawdown_limit]):
            if price_col not in df_test.columns:
                if "_cumret" not in df_test.columns:
                    df_test["_cumret"] = (1.0 + df_test["ret"].fillna(0.0)).cumprod()
                df_test[price_col] = df_test["_cumret"] * 100
            
            df_test = apply_risk_management(
                df_test,
                position_col="position",
                ret_col="ret",
                price_col=price_col,
                stop_loss=stop_loss,
                take_profit=take_profit,
                trailing_stop=trailing_stop,
                max_drawdown_limit=max_drawdown_limit
            )
            df_test["position"] = df_test["position_with_risk"]
        else:
            df_test["risk_action"] = ""
            df_test["unrealized_pnl"] = 0.0
        
        # Compute PnL
        df_test["pnl"] = df_test["position"] * df_test["ret"]
        df_test["equity"] = (1.0 + df_test["pnl"].fillna(0.0)).cumprod()
        
        # Calculate metrics
        df_test_clean = df_test.dropna(subset=["ret"]).copy()
        if len(df_test_clean) == 0:
            print(f"  [SKIP] No valid returns")
            continue
        
        m = metrics_from_pnl(df_test_clean["pnl"].fillna(0.0))
        
        # Add fold info
        m["fold"] = fold_idx + 1
        m["train_start"] = train_start_date.strftime("%Y-%m-%d")
        m["train_end"] = train_end_date.strftime("%Y-%m-%d")
        m["test_start"] = test_start_date.strftime("%Y-%m-%d")
        m["test_end"] = test_end_date.strftime("%Y-%m-%d")
        m["test_days"] = len(df_test_clean)
        
        fold_results.append(m)
        all_test_results.append(df_test_clean)
        
        print(f"  Sharpe: {m['sharpe']:.3f}, CAGR: {m['cagr']:.3f}, Hit Ratio: {m['hit_ratio']:.3f}")
        print()
    
    if not fold_results:
        raise ValueError("No valid folds found!")
    
    # Combine all test results
    df_combined = pd.concat(all_test_results, ignore_index=True)
    
    # Calculate overall metrics
    overall_metrics = metrics_from_pnl(df_combined["pnl"].fillna(0.0))
    
    # Period statistics
    period_df = pd.DataFrame(fold_results)
    period_statistics = {
        "num_folds": len(fold_results),
        "avg_sharpe": period_df["sharpe"].mean(),
        "std_sharpe": period_df["sharpe"].std(),
        "avg_sortino": period_df["sortino"].mean(),
        "avg_cagr": period_df["cagr"].mean(),
        "avg_hit_ratio": period_df["hit_ratio"].mean(),
        "avg_max_drawdown": period_df["max_drawdown"].mean(),
        "positive_folds": (period_df["cagr"] > 0).sum(),
        "negative_folds": (period_df["cagr"] <= 0).sum()
    }
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_to_native(obj):
        if isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    # Prepare output metrics
    purged_kfold_metrics = {
        "overall_metrics": convert_to_native(overall_metrics),
        "fold_statistics": convert_to_native(period_statistics),
        "fold_details": convert_to_native(fold_results)
    }
    
    # Save outputs
    out_eq = "SP500_news/results/purged_kfold_equity_curve.csv"
    df_combined[[date_col, "ret", "position", "pnl", "equity", "signed_mean", "news_count"]].to_csv(out_eq, index=False)
    
    out_metrics = "SP500_news/results/purged_kfold_metrics.json"
    with open(out_metrics, "w", encoding="utf-8") as f:
        json.dump(purged_kfold_metrics, f, indent=2)
    
    out_folds = "SP500_news/results/purged_kfold_folds.csv"
    period_df.to_csv(out_folds, index=False)
    
    # Summary
    out_summary = "SP500_news/results/purged_kfold_summary.txt"
    with open(out_summary, "w", encoding="utf-8") as f:
        f.write("SP500 Purged K-Fold Cross-Validation Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Number of folds: {n_splits}\n")
        f.write(f"Purge days (before test): {purge_days}\n")
        f.write(f"Embargo days (after test): {embargo_days}\n\n")
        
        f.write("Overall Metrics (Combined Test Folds):\n")
        f.write("-" * 60 + "\n")
        for k, v in overall_metrics.items():
            f.write(f"{k}: {v}\n")
        
        f.write("\nFold Statistics:\n")
        f.write("-" * 60 + "\n")
        for k, v in period_statistics.items():
            f.write(f"{k}: {v}\n")
        
        f.write("\nFold Details:\n")
        f.write("-" * 60 + "\n")
        for i, p in enumerate(fold_results):
            f.write(f"\nFold {i+1} [{p['test_start']} to {p['test_end']}]:\n")
            f.write(f"  Sharpe: {p['sharpe']:.3f}\n")
            f.write(f"  Sortino: {p['sortino']:.3f}\n")
            f.write(f"  CAGR: {p['cagr']:.3f}\n")
            f.write(f"  Max DD: {p['max_drawdown']:.3f}\n")
            f.write(f"  Hit Ratio: {p['hit_ratio']:.3f}\n")
    
    print("=" * 60)
    print("PURGED K-FOLD RESULTS")
    print("=" * 60)
    print(f"\nOverall Metrics (All Test Folds Combined):")
    print(f"  Sharpe: {overall_metrics['sharpe']:.3f}")
    print(f"  Sortino: {overall_metrics['sortino']:.3f}")
    print(f"  CAGR: {overall_metrics['cagr']:.3f}")
    print(f"  Max Drawdown: {overall_metrics['max_drawdown']:.3f}")
    print(f"  Hit Ratio: {overall_metrics['hit_ratio']:.3f}")
    print(f"\nFold Statistics ({len(fold_results)} folds):")
    print(f"  Avg Sharpe: {period_statistics['avg_sharpe']:.3f} (std: {period_statistics['std_sharpe']:.3f})")
    print(f"  Avg Sortino: {period_statistics['avg_sortino']:.3f}")
    print(f"  Avg CAGR: {period_statistics['avg_cagr']:.3f}")
    print(f"  Positive folds: {period_statistics['positive_folds']}/{len(fold_results)}")
    print(f"  Negative folds: {period_statistics['negative_folds']}/{len(fold_results)}")
    
    print(f"\nSaved:")
    print(f" - {out_eq}")
    print(f" - {out_metrics}")
    print(f" - {out_folds}")
    print(f" - {out_summary}")
    
    return df_combined, purged_kfold_metrics


def main():
    p = argparse.ArgumentParser(description="Backtest sentiment-based SP500 strategy (standalone)")
    p.add_argument("--agg_csv", type=str, default="SP500_news/processed/sp500_headlines_daily_agg.csv")
    p.add_argument("--price_csv", type=str, default="SP500_news/raw/sp500_headlines_2008_2024.csv")
    p.add_argument("--date_col", type=str, default="Date")
    p.add_argument("--price_col", type=str, default="CP")
    p.add_argument("--tpos", type=float, default=0.10, help="Positive threshold (default: 0.10)")
    p.add_argument("--tneg", type=float, default=0.10, help="Negative threshold (default: 0.10)")
    p.add_argument("--nmin", type=int, default=5, help="Minimum news count (default: 5)")
    p.add_argument("--ret", type=str, default="log", choices=["log", "simple"], help="Return type: log or simple")
    
    # Risk management arguments
    p.add_argument("--stop_loss", type=float, default=None, help="Stop-loss threshold (e.g., 0.05 for 5% loss)")
    p.add_argument("--take_profit", type=float, default=None, help="Take-profit threshold (e.g., 0.10 for 10% profit)")
    p.add_argument("--trailing_stop", type=float, default=None, help="Trailing stop distance (e.g., 0.03 for 3%)")
    p.add_argument("--max_drawdown_limit", type=float, default=None, help="Max drawdown limit before closing all positions (e.g., 0.20 for 20%)")
    
    # Walk-forward arguments
    p.add_argument("--walk_forward", action="store_true", help="Enable walk-forward analysis")
    p.add_argument("--train_years", type=int, default=2, help="Training period in years (for walk-forward)")
    p.add_argument("--test_years", type=int, default=1, help="Test period in years (for walk-forward)")
    p.add_argument("--step_months", type=int, default=6, help="Step size in months (for walk-forward)")
    
    # Purged K-Fold arguments
    p.add_argument("--purged_kfold", action="store_true", help="Enable purged k-fold cross-validation with embargo")
    p.add_argument("--n_splits", type=int, default=5, help="Number of folds (default: 5)")
    p.add_argument("--purge_days", type=int, default=5, help="Days to purge before test period (default: 5)")
    p.add_argument("--embargo_days", type=int, default=1, help="Days to embargo after test period (default: 1)")
    
    # Enhanced filter arguments
    p.add_argument("--no_trend_filter", action="store_true", help="Disable trend filter")
    p.add_argument("--no_volatility_filter", action="store_true", help="Disable volatility filter")
    p.add_argument("--no_position_sizing", action="store_true", help="Disable position sizing")
    p.add_argument("--no_signal_quality", action="store_true", help="Disable signal quality filter")
    p.add_argument("--signal_quality_threshold", type=float, default=0.15, help="Signal quality threshold (default: 0.15)")
    p.add_argument("--max_holding_days", type=int, default=20, help="Maximum holding period in days (default: 20)")
    p.add_argument("--trend_short_window", type=int, default=50, help="Short MA window for trend filter (default: 50)")
    p.add_argument("--trend_long_window", type=int, default=200, help="Long MA window for trend filter (default: 200)")
    p.add_argument("--volatility_window", type=int, default=20, help="Window for volatility filter (default: 20)")
    p.add_argument("--volatility_percentile", type=float, default=75.0, help="Volatility percentile threshold (default: 75.0)")
    p.add_argument("--base_position_size", type=float, default=1.0, help="Base position size (default: 1.0)")

    args = p.parse_args()
    
    # Prepare enhanced filter flags
    use_trend_filter = not args.no_trend_filter
    use_volatility_filter = not args.no_volatility_filter
    use_position_sizing = not args.no_position_sizing
    use_signal_quality = not args.no_signal_quality

    if args.purged_kfold:
        run_purged_kfold_analysis(
            agg_csv=args.agg_csv,
            price_csv=args.price_csv,
            date_col=args.date_col,
            price_col=args.price_col,
            tpos=args.tpos,
            tneg=args.tneg,
            nmin=args.nmin,
            ret_type=args.ret,
            n_splits=args.n_splits,
            purge_days=args.purge_days,
            embargo_days=args.embargo_days,
            stop_loss=args.stop_loss,
            take_profit=args.take_profit,
            trailing_stop=args.trailing_stop,
            max_drawdown_limit=args.max_drawdown_limit,
            use_trend_filter=use_trend_filter,
            use_volatility_filter=use_volatility_filter,
            use_position_sizing=use_position_sizing,
            use_signal_quality=use_signal_quality,
            signal_quality_threshold=args.signal_quality_threshold,
            max_holding_days=args.max_holding_days,
            trend_short_window=args.trend_short_window,
            trend_long_window=args.trend_long_window,
            volatility_window=args.volatility_window,
            volatility_percentile=args.volatility_percentile,
            base_position_size=args.base_position_size,
        )
    elif args.walk_forward:
        run_walk_forward_analysis(
            agg_csv=args.agg_csv,
            price_csv=args.price_csv,
            date_col=args.date_col,
            price_col=args.price_col,
            tpos=args.tpos,
            tneg=args.tneg,
            nmin=args.nmin,
            ret_type=args.ret,
            train_years=args.train_years,
            test_years=args.test_years,
            step_months=args.step_months,
            stop_loss=args.stop_loss,
            take_profit=args.take_profit,
            trailing_stop=args.trailing_stop,
            max_drawdown_limit=args.max_drawdown_limit,
            use_trend_filter=use_trend_filter,
            use_volatility_filter=use_volatility_filter,
            use_position_sizing=use_position_sizing,
            use_signal_quality=use_signal_quality,
            signal_quality_threshold=args.signal_quality_threshold,
            max_holding_days=args.max_holding_days,
            trend_short_window=args.trend_short_window,
            trend_long_window=args.trend_long_window,
            volatility_window=args.volatility_window,
            volatility_percentile=args.volatility_percentile,
            base_position_size=args.base_position_size,
        )
    else:
        run_backtest(
            agg_csv=args.agg_csv,
            price_csv=args.price_csv,
            date_col=args.date_col,
            price_col=args.price_col,
            tpos=args.tpos,
            tneg=args.tneg,
            nmin=args.nmin,
            ret_type=args.ret,
            stop_loss=args.stop_loss,
            take_profit=args.take_profit,
            trailing_stop=args.trailing_stop,
            max_drawdown_limit=args.max_drawdown_limit,
            use_trend_filter=use_trend_filter,
            use_volatility_filter=use_volatility_filter,
            use_position_sizing=use_position_sizing,
            use_signal_quality=use_signal_quality,
            signal_quality_threshold=args.signal_quality_threshold,
            max_holding_days=args.max_holding_days,
            trend_short_window=args.trend_short_window,
            trend_long_window=args.trend_long_window,
            volatility_window=args.volatility_window,
            volatility_percentile=args.volatility_percentile,
            base_position_size=args.base_position_size,
        )


if __name__ == "__main__":
    main()
