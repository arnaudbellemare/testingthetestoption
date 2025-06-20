import streamlit as st
import datetime as dt
import numpy as np
import pandas as pd
import requests
import ccxt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants ---
BASE_URL = "https://thalex.com/api/v2/public"
INSTRUMENTS_ENDPOINT = "instruments"
URL_INSTRUMENTS = f"{BASE_URL}/{INSTRUMENTS_ENDPOINT}"
MARK_PRICE_ENDPOINT = "mark_price_historical_data"
URL_MARK_PRICE = f"{BASE_URL}/{MARK_PRICE_ENDPOINT}"
TICKER_ENDPOINT = "ticker"
URL_TICKER = f"{BASE_URL}/{TICKER_ENDPOINT}"
COLUMNS = ["ts", "mark_price_open", "mark_price_high", "mark_price_low", "mark_price_close", "iv_open", "iv_high", "iv_low", "iv_close", "mark_volume"]
REQUEST_TIMEOUT = 15
TRANSACTION_COST_BPS = 2
RETRY_ATTEMPTS = 3
RETRY_DELAY = 2

# Current date and time: 04:55 PM EDT, June 20, 2025 = 20:55 UTC
CURRENT_TIME_UTC = pd.Timestamp("2025-06-20 20:55:00", tz="UTC")

# Initialize exchange
exchange1 = None
try:
    exchange1 = ccxt.bitget({'enableRateLimit': True})
    logging.info("Successfully initialized Bitget exchange (exchange1).")
except Exception as e:
    st.error(f"Failed to connect to Bitget: {e}")
    logging.error(f"Failed to initialize Bitget exchange (exchange1): {e}", exc_info=True)

# --- Utility Functions ---

# Login Functions
@st.cache_data(ttl=3600)
def load_credentials():
    try:
        with open("usernames.txt", "r") as f_user:
            users = [line.strip() for line in f_user if line.strip()]
        with open("passwords.txt", "r") as f_pass:
            pwds = [line.strip() for line in f_pass if line.strip()]
        if len(users) != len(pwds):
            st.error("Number of usernames and passwords mismatch.")
            return {}
        return dict(zip(users, pwds))
    except FileNotFoundError:
        st.error("Credential files (usernames.txt, passwords.txt) not found.")
        return {}
    except Exception as e:
        st.error(f"Error loading credentials: {e}")
        return {}

def login():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if not st.session_state.logged_in:
        st.title("Please Log In")
        creds = load_credentials()
        if not creds:
            st.stop()
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if username in creds and creds[username] == password:
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("Invalid username or password")
        st.stop()

def safe_get_in(keys, data_dict, default=None):
    current = data_dict
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        elif isinstance(current, list) and isinstance(key, int) and 0 <= key < len(current):
            current = current[key]
        else:
            return default
    return current

# Fetch and Filter Functions
@st.cache_data(ttl=600)
def fetch_instruments():
    try:
        resp = requests.get(URL_INSTRUMENTS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.json().get("result", [])
    except Exception as e:
        st.error(f"Error fetching instruments: {e}")
        return []

@retry(stop=stop_after_attempt(RETRY_ATTEMPTS), wait=wait_fixed(RETRY_DELAY), retry=retry_if_exception_type(requests.exceptions.RequestException))
@st.cache_data(ttl=600)
def fetch_ticker(instr_name):
    try:
        r = requests.get(URL_TICKER, params={"instrument_name": instr_name}, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        return r.json().get("result", {})
    except requests.exceptions.RequestException as e:
        logging.warning(f"Error fetching ticker {instr_name}: {e}")
        return {}

@st.cache_data(ttl=600)
def fetch_ticker_batch(instrument_names):
    ticker_data = {}
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_to_instr = {executor.submit(fetch_ticker, instr): instr for instr in instrument_names}
            for future in as_completed(future_to_instr):
                instr = future_to_instr[future]
                ticker_data[instr] = future.result()
    except Exception as e:
        logging.warning(f"Batch ticker fetch failed: {e}. Falling back to sequential.")
        for instr in instrument_names:
            ticker_data[instr] = fetch_ticker(instr)
    return ticker_data

@st.cache_data(ttl=600)
def fetch_data(instruments_tuple, historical_lookback_days=7):
    instr = list(instruments_tuple)
    if not instr:
        logging.warning("fetch_data: instruments_tuple is empty.")
        return pd.DataFrame()
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(lambda x: fetch_single_instrument_data(x, historical_lookback_days), instr))
    dfs = [df for df in results if not df.empty]
    if not dfs:
        return pd.DataFrame()
    dfc = pd.concat(dfs, ignore_index=True)
    dfc['date_time'] = pd.to_datetime(dfc['ts'], unit='s', utc=True)
    dfc['k'] = dfc['instrument_name'].str.split('-').str[2].astype(float)
    dfc['option_type'] = dfc['instrument_name'].str.split('-').str[-1]
    # Ensure expiry_datetime_col is timezone-aware (UTC)
    dfc['expiry_datetime_col'] = pd.to_datetime(dfc['instrument_name'].str.split('-').str[1], format="%d%b%y", utc=True)
    dfc['expiry_datetime_col'] = dfc['expiry_datetime_col'].dt.floor('D') + pd.Timedelta(hours=8)
    if dfc['expiry_datetime_col'].dt.tz is None:
        logging.warning("expiry_datetime_col is naive after creation. Localizing to UTC.")
        dfc['expiry_datetime_col'] = dfc['expiry_datetime_col'].dt.tz_localize('UTC')
    return dfc.dropna(subset=['k', 'option_type', 'mark_price_close', 'iv_close', 'expiry_datetime_col', 'date_time']).sort_values('date_time')

@st.cache_data(ttl=600)
def fetch_single_instrument_data(name, days):
    try:
        params = params_historical(name, days)
        resp = requests.get(URL_MARK_PRICE, params=params, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        marks = safe_get_in(["result", "mark"], resp.json(), default=[])
        if marks:
            return pd.DataFrame(marks, columns=COLUMNS).assign(instrument_name=name)
        return pd.DataFrame()
    except Exception as e:
        logging.error(f"fetch_single_instrument_data: Error for {name}: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=600)
def params_historical(instrument_name, days=7):
    now = CURRENT_TIME_UTC
    start_dt = now - dt.timedelta(days=days)
    return {"from": int(start_dt.timestamp()), "to": int(now.timestamp()), "resolution": "5m", "instrument_name": instrument_name}

@st.cache_data(ttl=600)
def fetch_kraken_data(coin="BTC", days=7):
    try:
        k = ccxt.kraken()
        now = CURRENT_TIME_UTC
        start = now - dt.timedelta(days=days)
        ohlcv = k.fetch_ohlcv(f"{coin}/USD", timeframe="5m", since=int(start.timestamp() * 1000))
        if ohlcv:
            return pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]).assign(
                date_time=lambda x: pd.to_datetime(x["timestamp"], unit="ms", utc=True)
            ).dropna(subset=['date_time']).sort_values("date_time").reset_index(drop=True)
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error fetching Kraken data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=600)
def fetch_funding_rates(exchange_instance, symbol='BTC/USDT', days=7):
    if exchange_instance is None:
        return pd.DataFrame()
    try:
        now = CURRENT_TIME_UTC
        start = now - dt.timedelta(days=days)
        hist = exchange_instance.fetch_funding_rate_history(symbol=symbol, since=int(start.timestamp() * 1000))
        if hist:
            return pd.DataFrame([{'date_time': pd.to_datetime(e['timestamp'], unit='ms', utc=True), 'funding_rate': e['fundingRate'] * 365 * 3} for e in hist])
        return pd.DataFrame()
    except Exception as e:
        logging.error(f"Error fetching funding rates for {symbol}: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=600)
def get_valid_expiration_options(current_date_utc):
    instruments = fetch_instruments()
    if not instruments:
        return []
    # Ensure all timestamps are UTC-aware
    expiry_dates = [
        pd.to_datetime(i.get("instrument_name", "").split("-")[1], format="%d%b%y", utc=True).replace(hour=8)
        for i in instruments
        if len(i.get("instrument_name", "").split("-")) >= 3 and i.get("instrument_name", "").split("-")[-1] in ['C', 'P']
    ]
    current_date_utc_ts = pd.Timestamp(current_date_utc, tz='UTC')
    mask = [exp > current_date_utc_ts for exp in expiry_dates]
    return [exp for exp, m in zip(expiry_dates, mask) if m]

@st.cache_data(ttl=600)
def get_option_instruments(instruments, option_type, expiry_str, coin):
    return np.array([
        i["instrument_name"] for i in instruments
        if i.get("instrument_name", "").startswith(f"{coin}-{expiry_str}") and i.get("instrument_name", "").endswith(f"-{option_type}")
    ])

# --- Greek Calculations (Vectorized) ---
@st.cache_data(ttl=600)
def compute_greeks_vectorized(df, spot_price, snapshot_time_utc, risk_free_rate=0.0):
    if df.empty or 'k' not in df.columns or 'iv_close' not in df.columns or 'option_type' not in df.columns or 'expiry_datetime_col' not in df.columns:
        return df.assign(delta=np.nan, gamma=np.nan, vega=np.nan, theta=np.nan)
    
    # Ensure snapshot_time_utc is a timezone-aware pd.Timestamp
    snapshot_time_utc_ts = pd.Timestamp(snapshot_time_utc, tz='UTC')
    
    # Ensure expiry_datetime_col is timezone-aware (UTC)
    df['expiry_datetime_col'] = pd.to_datetime(df['expiry_datetime_col'], utc=True, errors='coerce')
    if df['expiry_datetime_col'].isna().any():
        logging.warning("Some expiry_datetime_col values are NaT. Filling with snapshot_time_utc.")
        df['expiry_datetime_col'] = df['expiry_datetime_col'].fillna(snapshot_time_utc_ts)
    
    # Calculate time to expiry (T) in years
    T = (df['expiry_datetime_col'] - snapshot_time_utc_ts).dt.total_seconds().fillna(0) / (365 * 24 * 3600)
    T = np.where(T < 1e-9, 1e-9, T)
    
    sigma = df['iv_close'].values
    k = df['k'].values
    S = np.full(len(df), spot_price) if np.isscalar(spot_price) else spot_price
    option_type = df['option_type'].values == 'C'

    sigma_sqrt_T = sigma * np.sqrt(T)
    d1 = (np.log(S / k) + (risk_free_rate + 0.5 * sigma ** 2) * T) / sigma_sqrt_T
    d2 = d1 - sigma_sqrt_T

    delta = np.where(option_type, norm.cdf(d1), norm.cdf(d1) - 1.0)
    gamma = norm.pdf(d1) / (S * sigma_sqrt_T)
    vega = S * norm.pdf(d1) * np.sqrt(T) * 0.01
    theta = np.where(
        option_type,
        -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - risk_free_rate * k * np.exp(-risk_free_rate * T) * norm.cdf(d2),
        -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + risk_free_rate * k * np.exp(-risk_free_rate * T) * norm.cdf(-d2)
    ) / 365.0

    return df.assign(delta=delta, gamma=gamma, vega=vega, theta=theta)

# --- Hedging Analysis ---
@st.cache_data(ttl=600)
def display_mm_gamma_adjustment_analysis(dft_latest_snap, spot_price, snapshot_time_utc, risk_free_rate=0.0):
    st.subheader("MM Indicative Delta-Gamma Hedge Adjustment (Selected Expiry)")
    st.caption("Assumes Market Maker is short the entire displayed option book for this expiry.")

    if dft_latest_snap.empty or not all(c in dft_latest_snap.columns for c in ['instrument_name', 'k', 'option_type', 'iv_close', 'open_interest', 'expiry_datetime_col']):
        st.warning("Cannot perform analysis: Latest snapshot data missing required columns or empty.")
        return
    if pd.isna(spot_price) or spot_price <= 0:
        st.warning(f"Invalid spot price: {spot_price}.")
        return

    df_book = dft_latest_snap.copy()
    df_book['open_interest'] = pd.to_numeric(df_book['open_interest'], errors='coerce').fillna(0)
    mask = df_book['open_interest'] > 0
    df_book = df_book[mask]

    if df_book.empty:
        st.info("No options with open interest in the latest snapshot for this expiry.")
        return

    df_book = compute_greeks_vectorized(df_book, spot_price, snapshot_time_utc, risk_free_rate)
    mm_net_delta_initial = -np.sum(df_book['delta'] * df_book['open_interest'])
    mm_net_gamma_initial = -np.sum(df_book['gamma'] * df_book['open_interest'])
    mm_net_theta_initial = -np.sum(df_book['theta'] * df_book['open_interest'])

    st.metric("MM Initial Net Delta (Book)", f"{mm_net_delta_initial:,.2f}")
    st.metric("MM Initial Net Gamma (Book)", f"{mm_net_gamma_initial:,.4f}")
    st.metric("MM Initial Net Theta (Book)", f"{mm_net_theta_initial:,.2f}")

    gamma_hedger_selected = None
    calls_in_book = df_book[df_book['option_type'] == 'C']
    if not calls_in_book.empty:
        calls_in_book['moneyness_dist'] = np.abs(calls_in_book['k'] - spot_price)
        atm_call_idx = np.argmin(calls_in_book['moneyness_dist']) if not calls_in_book[calls_in_book['k'] >= spot_price].empty else np.argmin(calls_in_book['moneyness_dist'])
        gamma_hedger_selected = calls_in_book.iloc[atm_call_idx]

    theta_hedger_selected = None
    puts_in_book = df_book[df_book['option_type'] == 'P']
    if not puts_in_book.empty:
        puts_in_book['moneyness_dist'] = np.abs(puts_in_book['k'] - spot_price)
        atm_put_idx = np.argmin(puts_in_book[puts_in_book['k'] <= spot_price]['moneyness_dist']) if not puts_in_book[puts_in_book['k'] <= spot_price].empty else np.argmin(puts_in_book['moneyness_dist'])
        theta_hedger_selected = puts_in_book.iloc[atm_put_idx]

    if gamma_hedger_selected is None and theta_hedger_selected is None:
        st.warning("Could not select suitable Call or Put option from the book for hedging.")
        return

    st.info(f"Selected Gamma Hedging Instrument: {gamma_hedger_selected['instrument_name'] if gamma_hedger_selected is not None else 'N/A'}")
    st.info(f"Selected Theta Hedging Instrument: {theta_hedger_selected['instrument_name'] if theta_hedger_selected is not None else 'N/A'}")

    steth_position = st.sidebar.number_input("stETH Position Size", value=0.0, key="steth_pos")
    inverse_position = st.sidebar.number_input("Inverse Position Size", value=0.0, key="inverse_pos")
    short_put_position = st.sidebar.number_input("Short Put Position Size", value=0.0, key="short_put_pos")
    funding_rate = fetch_funding_rates(exchange1, f"{st.session_state.selected_coin}/USDT").iloc[-1]['funding_rate'] if not fetch_funding_rates(exchange1, f"{st.session_state.selected_coin}/USDT").empty else 0.0
    inverse_delta = -1.0 * (1 + funding_rate)
    steth_delta = 1.0
    put_delta = theta_hedger_selected['delta'] if theta_hedger_selected is not None else 0.0
    put_theta = theta_hedger_selected['theta'] if theta_hedger_selected is not None else 0.0

    total_delta_from_extras = np.sum([steth_position * steth_delta, inverse_position * inverse_delta, short_put_position * put_delta])
    total_theta_from_extras = short_put_position * put_theta

    N_h = -mm_net_gamma_initial / gamma_hedger_selected['gamma'] if gamma_hedger_selected is not None and abs(gamma_hedger_selected['gamma']) > 1e-7 else 0.0
    delta_from_gamma_hedge = N_h * gamma_hedger_selected['delta'] if gamma_hedger_selected is not None else 0.0

    N_p = -mm_net_theta_initial / put_theta if theta_hedger_selected is not None and abs(put_theta) > 1e-7 else 0.0
    delta_from_theta_hedge = N_p * put_delta if theta_hedger_selected is not None else 0.0
    theta_from_theta_hedge = N_p * put_theta if theta_hedger_selected is not None else 0.0

    mm_net_delta_post_gamma_hedge = mm_net_delta_initial + delta_from_gamma_hedge + delta_from_theta_hedge + total_delta_from_extras
    mm_net_theta_post_hedge = mm_net_theta_initial + theta_from_theta_hedge + total_theta_from_extras
    underlying_hedge_qty = -mm_net_delta_post_gamma_hedge

    st.markdown("#### Indicative Gamma & Theta Hedge Adjustments:")
    cols_hedge = st.columns(4)
    with cols_hedge[0]:
        st.metric("Gamma Hedger Delta (Dₕ)", f"{gamma_hedger_selected['delta']:.4f}" if gamma_hedger_selected is not None else "N/A")
    with cols_hedge[1]:
        st.metric("Gamma Hedger Gamma (Gₕ)", f"{gamma_hedger_selected['gamma']:.6f}" if gamma_hedger_selected is not None else "N/A")
    with cols_hedge[2]:
        st.metric(f"Gamma Hedge Qty ({'Buy' if N_h > 0 else 'Sell'})", f"{abs(N_h):,.2f} units" if gamma_hedger_selected is not None else "N/A")
    with cols_hedge[3]:
        st.metric("Put Delta (Dₚ)", f"{put_delta:.4f}" if theta_hedger_selected is not None else "N/A")

    st.metric("Delta Change from Gamma Hedge", f"{delta_from_gamma_hedge:,.2f}")
    st.metric("Delta Change from Theta Hedge", f"{delta_from_theta_hedge:,.2f}")
    st.metric("Delta from stETH/Inverse/Short Puts", f"{total_delta_from_extras:,.2f}")
    st.metric("Theta from Theta Hedge", f"{theta_from_theta_hedge:,.2f}")
    st.metric("Theta from Short Puts", f"{total_theta_from_extras:,.2f}")

    st.markdown("#### Indicative Final Delta & Theta Hedge:")
    st.metric("MM Net Delta (After All Hedges)", f"{mm_net_delta_post_gamma_hedge:,.2f}")
    st.metric("MM Net Theta (After All Hedges)", f"{mm_net_theta_post_hedge:,.2f}")
    action_underlying = "Buy" if underlying_hedge_qty > 0 else "Sell" if underlying_hedge_qty < 0 else "Hold"
    st.metric(f"Final Underlying Hedge ({action_underlying} Spot/Perp)", f"{abs(underlying_hedge_qty):,.2f} {st.session_state.selected_coin}")

    final_net_delta_book = mm_net_delta_post_gamma_hedge + underlying_hedge_qty
    st.success(f"**Resulting Book Net Delta (Post-All Adjustments):** {final_net_delta_book:,.4f} (should be ~0)")
    st.caption("This indicates the spot/perp hedge needed for the MM to become delta-neutral.")

# --- Plotting Functions (Simplified) ---
@st.cache_data(ttl=300)
def plot_delta_balance(ticker_list, spot_price):
    if not ticker_list or pd.isna(spot_price):
        st.info("Data insufficient for Delta Balance plot.")
        return
    df = pd.DataFrame(ticker_list)
    calls_delta = np.sum(df[df['option_type'] == 'C']['delta'] * df[df['option_type'] == 'C']['open_interest'])
    puts_delta = np.sum(df[df['option_type'] == 'P']['delta'] * df[df['option_type'] == 'P']['open_interest'])
    data = pd.DataFrame({'Option Type': ['Calls', 'Puts'], 'Total Weighted Delta': [calls_delta, abs(puts_delta)]})
    fig = px.bar(data, x='Option Type', y='Total Weighted Delta', color='Option Type', color_discrete_map={"Calls": "mediumseagreen", "Puts": "lightcoral"})
    fig.add_annotation(text=f"Net Delta Exposure: {calls_delta + puts_delta:,.2f}", align='center', showarrow=False, xref='paper', yref='paper', x=0.5, y=1.05)
    st.plotly_chart(fig, use_container_width=True)

@st.cache_data(ttl=300)
def plot_open_interest_delta(ticker_list, spot_price):
    if not ticker_list or pd.isna(spot_price):
        st.info("Data insufficient for OI & Delta Bubble chart.")
        return
    df = pd.DataFrame(ticker_list).dropna(subset=['strike', 'open_interest', 'delta', 'iv'])
    df['moneyness'] = df['strike'] / spot_price
    fig = px.scatter(
        df, x="strike", y="delta", size="open_interest", color="moneyness",
        color_continuous_scale=px.colors.diverging.RdYlBu_r, range_color=[0.8, 1.2],
        hover_data=["instrument", "open_interest", "iv"], size_max=50
    )
    fig.add_vline(x=spot_price, line_dash="dot", line_color="black")
    fig.add_hline(y=0.5, line_dash="dot", line_color="grey")
    fig.add_hline(y=-0.5, line_dash="dot", line_color="grey")
    st.plotly_chart(fig, use_container_width=True)

# Placeholder for undefined plotting functions
def plot_intraday_premium_spikes(df_options_hist, df_spot_hist, coin_symbol, spot_price_latest, selected_expiry_str):
    st.subheader(f"Intraday Premium Spikes ({coin_symbol} - {selected_expiry_str})")
    st.info("Premium Spikes plot not implemented due to missing function definition.")

def compute_and_plot_itm_gex_ratio(dft, df_krak_5m, spot_price_latest, selected_expiry_obj):
    st.subheader(f"ITM Put/Call GEX Ratio (Expiry: {selected_expiry_obj.strftime('%d%b%y')})")
    st.info("GEX Ratio plot not implemented due to missing function definition.")
    return pd.DataFrame()

# --- Main Function ---
def main():
    st.set_page_config(layout="wide", page_title="Advanced Options Hedging & MM Dashboard")
    login()

    # Initialize session state
    if 'selected_coin' not in st.session_state:
        st.session_state.selected_coin = "BTC"
    elif not isinstance(st.session_state.selected_coin, str):
        st.session_state.selected_coin = str(st.session_state.selected_coin)
    if 'snapshot_time' not in st.session_state:
        st.session_state.snapshot_time = pd.Timestamp(CURRENT_TIME_UTC, tz='UTC')
    if 'risk_free_rate_input' not in st.session_state:
        st.session_state.risk_free_rate_input = 0.01

    st.title(f"{st.session_state.selected_coin} Options: Advanced Hedging & MM Perspective")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

    st.sidebar.header("Configuration")
    coin_options = ["BTC", "ETH"]
    current_coin_idx = coin_options.index(st.session_state.selected_coin) if st.session_state.selected_coin in coin_options else 0
    selected_coin_widget = st.sidebar.selectbox("Cryptocurrency", coin_options, index=current_coin_idx, key="main_coin_select_adv_vFull2_final")
    if selected_coin_widget != st.session_state.selected_coin:
        st.session_state.selected_coin = selected_coin_widget
        st.rerun()
    coin = st.session_state.selected_coin

    st.session_state.risk_free_rate_input = st.sidebar.number_input(
        "Risk-Free Rate (Annualized)", value=st.session_state.risk_free_rate_input, min_value=0.0, max_value=0.2, step=0.001, format="%.3f", key="main_rf_rate_adv_vFull2_final"
    )
    risk_free_rate = st.session_state.risk_free_rate_input
    st.session_state.snapshot_time = pd.Timestamp(CURRENT_TIME_UTC, tz='UTC')

    pair_sim_lookback_days = st.sidebar.number_input("Hist. Lookback (days)", min_value=7, max_value=365, value=30, step=7, key="pair_sim_lookback_days_adv_vFull2_final")
    spot_merge_tolerance_minutes = st.sidebar.number_input("Spot Merge Tolerance (minutes)", min_value=1, max_value=240, value=15, step=1, key="spot_merge_tolerance_adv_vFull2_final")
    df_krak_5m = fetch_kraken_data(coin=coin, days=max(10, pair_sim_lookback_days + 2))
    df_spot_hist = df_krak_5m
    spot_price = df_krak_5m["close"].iloc[-1] if not df_krak_5m.empty else np.nan

    all_instruments_list = fetch_instruments()
    if not all_instruments_list:
        st.error("Failed to fetch instruments list.")
        st.stop()

    current_snapshot_time = st.session_state.snapshot_time
    valid_expiries = get_valid_expiration_options(current_snapshot_time)
    if not valid_expiries:
        st.error(f"No valid expiries for {coin}.")
        st.stop()

    default_exp_idx = 0
    if valid_expiries:
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_expiry = {
                executor.submit(
                    lambda x: (
                        x,
                        np.sum([
                            fetch_ticker(instr).get('open_interest', 0)
                            for instr in get_option_instruments(all_instruments_list, "C", x.strftime("%d%b%y").upper(), coin)[:50] +
                            get_option_instruments(all_instruments_list, "P", x.strftime("%d%b%y").upper(), coin)[:50]
                        ])
                    ),
                    exp
                ): exp for exp in valid_expiries
            }
            expiry_oi_map = {future.result()[0]: future.result()[1] for future in as_completed(future_to_expiry)}
        if expiry_oi_map:
            best_expiry_by_oi = max(expiry_oi_map.items(), key=lambda x: x[1])[0]
            default_exp_idx = valid_expiries.index(best_expiry_by_oi)

    selected_expiry = st.sidebar.selectbox(
        "Choose Expiry", valid_expiries, index=default_exp_idx, format_func=lambda d: d.strftime("%d %b %Y"),
        key=f"main_expiry_select_adv_vFull2_final_{coin}_oi_default"
    )
    e_str = selected_expiry.strftime("%d%b%y").upper() if selected_expiry else "N/A"
    all_calls_expiry = get_option_instruments(all_instruments_list, "C", e_str, coin)
    all_puts_expiry = get_option_instruments(all_instruments_list, "P", e_str, coin)
    all_instr_selected_expiry = np.sort(np.concatenate([all_calls_expiry, all_puts_expiry]))

    st.header(f"Analysis: {coin} | Expiry: {e_str} | Spot: ${spot_price:,.2f}" if not np.isnan(spot_price) else f"Analysis: {coin} | Expiry: {e_str} | Spot: N/A")
    st.markdown(f"*Snapshot: {st.session_state.snapshot_time.strftime('%Y-%m-%d %H:%M:%S UTC')} | RF Rate: {risk_free_rate:.3%}*")

    if not all_instr_selected_expiry.size:
        st.error(f"No options for {e_str}.")
        st.stop()
    dft_raw = fetch_data(all_instr_selected_expiry, historical_lookback_days=pair_sim_lookback_days)
    ticker_data = fetch_ticker_batch(all_instr_selected_expiry)
    valid_tickers = np.array([k for k, v in ticker_data.items() if v and isinstance(v.get('iv'), (int, float)) and v.get('iv', 0) > 1e-4 and pd.notna(v.get('open_interest'))])
    dft = dft_raw[dft_raw['instrument_name'].isin(valid_tickers)].copy() if not dft_raw.empty else pd.DataFrame()
    if not dft.empty:
        dft['open_interest'] = np.array([ticker_data.get(x, {}).get('open_interest', 0.0) for x in dft['instrument_name']])
        dft['iv_close'] = pd.to_numeric(dft['iv_close'], errors='coerce')

    dft_with_hist_greeks = pd.DataFrame()
    if not dft.empty and not df_krak_5m.empty:
        merged_hist = pd.merge_asof(
            dft.sort_values('date_time'),
            df_krak_5m[['date_time', 'close']].rename(columns={'close': 'spot_hist'}),
            on='date_time', direction='nearest', tolerance=pd.Timedelta(minutes=spot_merge_tolerance_minutes)
        ).dropna(subset=['spot_hist'])
        if not merged_hist.empty:
            with st.spinner("Calculating Greeks (Delta, Gamma, Vega, Theta) on historical data..."):
                dft_with_hist_greeks = compute_greeks_vectorized(
                    merged_hist, merged_hist['spot_hist'].values, merged_hist['date_time'].values, risk_free_rate
                )

    dft_latest = pd.DataFrame()
    if not dft.empty and not np.isnan(spot_price):
        dft_latest_idx = dft.groupby('instrument_name')['date_time'].idxmax()
        dft_latest = dft.loc[dft_latest_idx].copy()
        if not dft_latest.empty:
            dft_latest['open_interest'] = np.array([ticker_data.get(x, {}).get('open_interest', 0.0) for x in dft_latest['instrument_name']])
            dft_latest = compute_greeks_vectorized(dft_latest, spot_price, st.session_state.snapshot_time, risk_free_rate)

    def safe_plot_exec(plot_func, *args, **kwargs):
        try:
            plot_func(*args, **kwargs)
        except Exception as e:
            st.error(f"Plot error in {plot_func.__name__}: {e}")
            logging.error(f"Plot error in {plot_func.__name__}", exc_info=True)

    if not dft_latest.empty:
        ticker_list_latest_snap = np.array([
            {
                'instrument': row['instrument_name'], 'strike': int(row['k']), 'option_type': row['option_type'],
                'open_interest': float(row['open_interest']), 'delta': float(row['delta']),
                'gamma': float(row['gamma']), 'iv': float(row['iv_close'])
            } for _, row in dft_latest.iterrows()
        ])
        if ticker_list_latest_snap.size:
            safe_plot_exec(plot_open_interest_delta, ticker_list_latest_snap, spot_price)
            safe_plot_exec(plot_delta_balance, ticker_list_latest_snap, spot_price)

    st.markdown("---"); st.header("Market Maker Perspective")
    if not dft_latest.empty:
        st.subheader("Net Greek Exposures (Latest Snapshot - MM Short Book)")
        cols_greeks_adv = st.columns(5)
        net_d_mm = -np.sum(dft_latest['delta'] * dft_latest['open_interest'])
        cols_greeks_adv[0].metric("Net Delta", f"{net_d_mm:,.2f}")
        net_g_mm = -np.sum(dft_latest['gamma'] * dft_latest['open_interest'])
        cols_greeks_adv[1].metric("Net Gamma", f"{net_g_mm:,.4f}")
        net_v_mm = -np.sum(dft_latest['vega'] * dft_latest['open_interest'])
        cols_greeks_adv[2].metric("Net Vega", f"{net_v_mm:,.0f}")
        net_t_mm = -np.sum(dft_latest['theta'] * dft_latest['open_interest'])
        cols_greeks_adv[3].metric("Net Theta", f"{net_t_mm:,.2f}")
        display_mm_gamma_adjustment_analysis(dft_latest, spot_price, st.session_state.snapshot_time, risk_free_rate)

    st.markdown("---"); st.header("Raw Data Tables (Debug)")
    with st.expander("dft_raw (Initial Fetch - Head)"):
        st.dataframe(dft_raw.head(20))
    with st.expander("dft (After Current Ticker Filter - Head)"):
        st.dataframe(dft.head(20))
    with st.expander("dft_with_hist_greeks (For Sims - Head)"):
        st.dataframe(dft_with_hist_greeks.head(20))

    gc.collect()
    logging.info(f"--- ADVANCED Dashboard rendering complete for {coin} {e_str} ---")

if __name__ == "__main__":
    main()
