import streamlit as st
import datetime as dt
import numpy as np
import pandas as pd
import requests
import ccxt
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import norm
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants ---
BASE_URL = "https://thalex.com/api/v2/public"
INSTRUMENTS_ENDPOINT = "instruments"
URL_INSTRUMENTS = f"{BASE_URL}/{INSTRUMENTS_ENDPOINT}"
MARK_PRICE_ENDPOINT = "mark_price_historical_data"
URL_MARK_PRICE = f"{BASE_URL}/{MARK_PRICE_ENDPOINT}"
TICKER_ENDPOINT = "ticker"
URL_TICKER = f"{BASE_URL}/{TICKER_ENDPOINT}"
COLUMNS = ["ts", "mark_price_open", "mark_price_high", "mark_price_low", "mark_price_close", "iv_open", "iv_high", "iv_low", "iv_close", "mark_volume"]
REQUEST_TIMEOUT = 10
TRANSACTION_COST_BPS = 2
RETRY_ATTEMPTS = 3
RETRY_DELAY = 1
CURRENT_TIME_UTC = pd.Timestamp("2025-06-20 20:55:00", tz="UTC")

# Initialize exchange
exchange1 = None
try:
    exchange1 = ccxt.bitget({'enableRateLimit': True})
    logger.info("Successfully initialized Bitget exchange.")
except Exception as e:
    st.error(f"Failed to connect to Bitget: {e}")
    logger.error(f"Failed to initialize Bitget exchange: {e}")

# --- Utility Functions ---

@st.cache_data(ttl=3600)
def load_credentials():
    try:
        with open("usernames.txt", "r") as f_user:
            users = [line.strip() for line in f_user if line.strip()]
        with open("passwords.txt", "r") as f_pass:
            pwds = [line.strip() for line in f_pass if line.strip()]
        if len(users) != len(pwds):
            logger.error("Number of usernames and passwords mismatch.")
            return {}
        return dict(zip(users, pwds))
    except FileNotFoundError:
        logger.error("Credential files not found.")
        return {}
    except Exception as e:
        logger.error(f"Error loading credentials: {e}")
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

@st.cache_data(ttl=600)
def fetch_instruments():
    try:
        resp = requests.get(URL_INSTRUMENTS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.json().get("result", [])
    except Exception as e:
        logger.warning(f"Error fetching instruments: {e}")
        return []

@retry(stop=stop_after_attempt(RETRY_ATTEMPTS), wait=wait_fixed(RETRY_DELAY), retry=retry_if_exception_type(requests.exceptions.RequestException))
@st.cache_data(ttl=600)
def fetch_ticker(instr_name):
    try:
        r = requests.get(URL_TICKER, params={"instrument_name": instr_name}, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        return r.json().get("result", {})
    except requests.exceptions.RequestException as e:
        logger.warning(f"Error fetching ticker {instr_name}: {e}")
        return {}

@st.cache_data(ttl=600)
def fetch_ticker_batch(instrument_names):
    ticker_data = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_instr = {executor.submit(fetch_ticker, instr): instr for instr in instrument_names}
        for future in as_completed(future_to_instr):
            instr = future_to_instr[future]
            try:
                ticker_data[instr] = future.result()
            except Exception as e:
                logger.warning(f"Failed to fetch ticker {instr}: {e}")
                ticker_data[instr] = {}
    return ticker_data

@st.cache_data(ttl=600)
def fetch_data(instruments_tuple, historical_lookback_days=7):
    instruments = list(instruments_tuple)
    if not instruments:
        logger.warning("No instruments provided.")
        return pd.DataFrame()
    with ThreadPoolExecutor(max_workers=6) as executor:
        results = executor.map(lambda x: fetch_single_instrument_data(x, historical_lookback_days), instruments)
    dfs = [df for df in results if not df.empty]
    if not dfs:
        return pd.DataFrame()
    dfc = pd.concat(dfs, ignore_index=True)
    if dfc.empty:
        return dfc
    dfc['date_time'] = pd.to_datetime(dfc['ts'], unit='s', utc=True)
    dfc['k'] = dfc['instrument_name'].str.split('-').str[2].astype(float, errors='ignore')
    dfc['option_type'] = dfc['instrument_name'].str.split('-').str[-1]
    dfc['expiry_datetime_col'] = pd.to_datetime(
        dfc['instrument_name'].str.split('-').str[1], format="%d%b%y", utc=True, errors='coerce'
    ).dt.floor('D') + pd.Timedelta(hours=8)
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
        logger.warning(f"Error fetching data for {name}: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=600)
def params_historical(instrument_name, days=7):
    start_dt = CURRENT_TIME_UTC - dt.timedelta(days=days)
    return {"from": int(start_dt.timestamp()), "to": int(CURRENT_TIME_UTC.timestamp()), "resolution": "5m", "instrument_name": instrument_name}

@st.cache_data(ttl=600)
def fetch_kraken_data(coin="BTC", days=7):
    try:
        k = ccxt.kraken({'enableRateLimit': True})
        start = CURRENT_TIME_UTC - dt.timedelta(days=days)
        ohlcv = k.fetch_ohlcv(f"{coin}/USD", timeframe="5m", since=int(start.timestamp() * 1000))
        if ohlcv:
            df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df['date_time'] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            return df.dropna(subset=['date_time']).sort_values("date_time").reset_index(drop=True)
        return pd.DataFrame()
    except Exception as e:
        logger.warning(f"Error fetching Kraken data for {coin}: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=600)
def fetch_funding_rates(exchange_instance, symbol='BTC/USDT', days=7):
    if exchange_instance is None:
        return pd.DataFrame()
    try:
        start = CURRENT_TIME_UTC - dt.timedelta(days=days)
        hist = exchange_instance.fetch_funding_rate_history(symbol=symbol, since=int(start.timestamp() * 1000))
        if hist:
            return pd.DataFrame([
                {'date_time': pd.to_datetime(e['timestamp'], unit='ms', utc=True), 'funding_rate': e['fundingRate'] * 365 * 3}
                for e in hist
            ])
        return pd.DataFrame()
    except Exception as e:
        logger.warning(f"Error fetching funding rates for {symbol}: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=600)
def get_valid_expiration_options(current_date_utc):
    instruments = fetch_instruments()
    if not instruments:
        return []
    expiry_dates = [
        pd.to_datetime(i.get("instrument_name", "").split("-")[1], format="%d%b%y", utc=True).replace(hour=8)
        for i in instruments
        if len(i.get("instrument_name", "").split("-")) >= 3 and i.get("instrument_name", "").endswith(('C', 'P'))
    ]
    current_date_utc_ts = pd.Timestamp(current_date_utc, tz='UTC')
    return [exp for exp in expiry_dates if exp > current_date_utc_ts]

@st.cache_data(ttl=600)
def get_option_instruments(instruments, option_type, expiry_str, coin):
    return [
        i["instrument_name"] for i in instruments
        if i.get("instrument_name", "").startswith(f"{coin}-{expiry_str}") and i.get("instrument_name", "").endswith(f"-{option_type}")
    ]

@st.cache_data(ttl=600)
def compute_greeks_vectorized(df, spot_price, snapshot_time_utc, risk_free_rate=0.0):
    if df.empty or not all(col in df for col in ['k', 'iv_close', 'option_type', 'expiry_datetime_col']):
        return df.assign(delta=np.nan, gamma=np.nan, vega=np.nan, theta=np.nan)
    
    snapshot_time_utc_ts = pd.Timestamp(snapshot_time_utc, tz='UTC')
    df['expiry_datetime_col'] = pd.to_datetime(df['expiry_datetime_col'], utc=True, errors='coerce')
    df['expiry_datetime_col'] = df['expiry_datetime_col'].fillna(snapshot_time_utc_ts)
    
    T = (df['expiry_datetime_col'] - snapshot_time_utc_ts).dt.total_seconds() / (365 * 24 * 3600)
    T = np.clip(T, 1e-9, None)
    
    sigma = df['iv_close'].values
    k = df['k'].values
    S = np.full_like(k, spot_price) if np.isscalar(spot_price) else spot_price
    is_call = df['option_type'].values == 'C'

    sigma_sqrt_T = sigma * np.sqrt(T)
    d1 = np.where(sigma_sqrt_T > 0, (np.log(S / k) + (risk_free_rate + 0.5 * sigma ** 2) * T) / sigma_sqrt_T, 0)
    d2 = d1 - sigma_sqrt_T

    delta = np.where(is_call, norm.cdf(d1), norm.cdf(d1) - 1.0)
    gamma = np.where(sigma_sqrt_T > 0, norm.pdf(d1) / (S * sigma_sqrt_T), 0)
    vega = np.where(sigma_sqrt_T > 0, S * norm.pdf(d1) * np.sqrt(T) * 0.01, 0)
    theta = np.where(
        is_call,
        -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - risk_free_rate * k * np.exp(-risk_free_rate * T) * norm.cdf(d2),
        -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + risk_free_rate * k * np.exp(-risk_free_rate * T) * norm.cdf(-d2)
    ) / 365.0
    theta = np.where(T > 1e-9, theta, 0)

    return df.assign(delta=delta, gamma=gamma, vega=vega, theta=theta)

@st.cache_data(ttl=600)
def display_mm_gamma_adjustment_analysis(dft_latest_snap, spot_price, snapshot_time_utc, risk_free_rate=0.0):
    st.subheader("MM Indicative Delta-Gamma Hedge Adjustment")
    if dft_latest_snap.empty or not all(c in dft_latest_snap for c in ['instrument_name', 'k', 'option_type', 'iv_close', 'open_interest', 'expiry_datetime_col']):
        st.warning("No data for analysis.")
        return
    if pd.isna(spot_price) or spot_price <= 0:
        st.warning(f"Invalid spot price: {spot_price}.")
        return

    df_book = dft_latest_snap.copy()
    df_book['open_interest'] = pd.to_numeric(df_book['open_interest'], errors='coerce').fillna(0)
    df_book = df_book[df_book['open_interest'] > 0]
    if df_book.empty:
        st.info("No options with open interest.")
        return

    df_book = compute_greeks_vectorized(df_book, spot_price, snapshot_time_utc, risk_free_rate)
    mm_net_delta_initial = -df_book['delta'].mul(df_book['open_interest']).sum()
    mm_net_gamma_initial = -df_book['gamma'].mul(df_book['open_interest']).sum()
    mm_net_theta_initial = -df_book['theta'].mul(df_book['open_interest']).sum()

    st.metric("MM Initial Net Delta", f"{mm_net_delta_initial:,.2f}")
    st.metric("MM Initial Net Gamma", f"{mm_net_gamma_initial:,.4f}")
    st.metric("MM Initial Net Theta", f"{mm_net_theta_initial:,.2f}")

    calls_in_book = df_book[df_book['option_type'] == 'C']
    gamma_hedger = None
    if not calls_in_book.empty:
        calls_in_book = calls_in_book.assign(moneyness_dist=np.abs(calls_in_book['k'] - spot_price))
        gamma_hedger = calls_in_book.loc[calls_in_book['moneyness_dist'].idxmin()]

    puts_in_book = df_book[df_book['option_type'] == 'P']
    theta_hedger = None
    if not puts_in_book.empty:
        puts_in_book = puts_in_book.assign(moneyness_dist=np.abs(puts_in_book['k'] - spot_price))
        theta_hedger = puts_in_book.loc[puts_in_book['moneyness_dist'].idxmin()]

    if gamma_hedger is None and theta_hedger is None:
        st.warning("No suitable hedging instruments.")
        return

    st.info(f"Gamma Hedger: {gamma_hedger['instrument_name'] if gamma_hedger is not None else 'N/A'}")
    st.info(f"Theta Hedger: {theta_hedger['instrument_name'] if theta_hedger is not None else 'N/A'}")

    steth_position = st.sidebar.number_input("stETH Position Size", value=0.0, step=0.1)
    inverse_position = st.sidebar.number_input("Inverse Position Size", value=0.0, step=0.1)
    short_put_position = st.sidebar.number_input("Short Put Position Size", value=0.0, step=0.1)
    funding_rate = fetch_funding_rates(exchange1, f"{st.session_state.selected_coin}/USDT").iloc[-1]['funding_rate'] if not fetch_funding_rates(exchange1, f"{st.session_state.selected_coin}/USDT").empty else 0.0
    inverse_delta = -1.0 * (1 + funding_rate)
    steth_delta = 1.0
    put_delta = theta_hedger['delta'] if theta_hedger is not None else 0.0
    put_theta = theta_hedger['theta'] if theta_hedger is not None else 0.0

    total_delta_extras = steth_position * steth_delta + inverse_position * inverse_delta + short_put_position * put_delta
    total_theta_extras = short_put_position * put_theta

    N_h = -mm_net_gamma_initial / gamma_hedger['gamma'] if gamma_hedger is not None and abs(gamma_hedger['gamma']) > 1e-7 else 0.0
    delta_from_gamma_hedge = N_h * gamma_hedger['delta'] if gamma_hedger is not None else 0.0
    N_p = -mm_net_theta_initial / put_theta if theta_hedger is not None and abs(put_theta) > 1e-7 else 0.0
    delta_from_theta_hedge = N_p * put_delta if theta_hedger is not None else 0.0
    theta_from_theta_hedge = N_p * put_theta if theta_hedger is not None else 0.0

    mm_net_delta_post_hedge = mm_net_delta_initial + delta_from_gamma_hedge + delta_from_theta_hedge + total_delta_extras
    mm_net_theta_post_hedge = mm_net_theta_initial + theta_from_theta_hedge + total_theta_extras
    underlying_hedge_qty = -mm_net_delta_post_hedge

    st.markdown("#### Hedge Adjustments")
    cols = st.columns(4)
    cols[0].metric("Gamma Hedger Delta", f"{gamma_hedger['delta']:.4f}" if gamma_hedger is not None else "N/A")
    cols[1].metric("Gamma Hedger Gamma", f"{gamma_hedger['gamma']:.6f}" if gamma_hedger is not None else "N/A")
    cols[2].metric(f"Gamma Hedge Qty ({'Buy' if N_h > 0 else 'Sell'})", f"{abs(N_h):,.2f}" if gamma_hedger is not None else "N/A")
    cols[3].metric("Put Delta", f"{put_delta:.4f}" if theta_hedger is not None else "N/A")

    st.metric("Delta from Gamma Hedge", f"{delta_from_gamma_hedge:,.2f}")
    st.metric("Delta from Theta Hedge", f"{delta_from_theta_hedge:,.2f}")
    st.metric("Delta from Extras", f"{total_delta_extras:,.2f}")
    st.metric("Theta from Theta Hedge", f"{theta_from_theta_hedge:,.2f}")
    st.metric("Theta from Short Puts", f"{total_theta_extras:,.2f}")

    st.markdown("#### Final Hedge")
    st.metric("MM Net Delta (Post-Hedge)", f"{mm_net_delta_post_hedge:,.2f}")
    st.metric("MM Net Theta (Post-Hedge)", f"{mm_net_theta_post_hedge:,.2f}")
    action = "Buy" if underlying_hedge_qty > 0 else "Sell" if underlying_hedge_qty < 0 else "Hold"
    st.metric(f"Underlying Hedge ({action})", f"{abs(underlying_hedge_qty):,.2f} {st.session_state.selected_coin}")

    final_net_delta = mm_net_delta_post_hedge + underlying_hedge_qty
    st.success(f"**Resulting Net Delta:** {final_net_delta:,.4f}")

@st.cache_data(ttl=300)
def plot_delta_balance(ticker_list, spot_price):
    if not ticker_list or pd.isna(spot_price):
        st.info("No data for Delta Balance plot.")
        return
    df = pd.DataFrame(ticker_list)
    calls_delta = df[df['option_type'] == 'C']['delta'].mul(df['open_interest']).sum()
    puts_delta = df[df['option_type'] == 'P']['delta'].mul(df['open_interest']).sum()
    data = pd.DataFrame({'Type': ['Calls', 'Puts'], 'Delta': [calls_delta, abs(puts_delta)]})
    fig = px.bar(data, x='Type', y='Delta', color='Type', color_discrete_map={"Calls": "mediumseagreen", "Puts": "lightcoral"})
    fig.add_annotation(text=f"Net Delta: {calls_delta + puts_delta:,.2f}", xref='paper', yref='paper', x=0.5, y=1.05, showarrow=False)
    st.plotly_chart(fig, use_container_width=True)

@st.cache_data(ttl=300)
def plot_open_interest_delta(ticker_list, spot_price):
    if not ticker_list or pd.isna(spot_price):
        st.info("No data for OI & Delta plot.")
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

def main():
    st.set_page_config(layout="wide", page_title="Options Hedging Dashboard")
    login()

    if 'selected_coin' not in st.session_state:
        st.session_state.selected_coin = "BTC"
    if 'snapshot_time' not in st.session_state:
        st.session_state.snapshot_time = CURRENT_TIME_UTC
    if 'risk_free_rate_input' not in st.session_state:
        st.session_state.risk_free_rate_input = 0.01

    st.title(f"{st.session_state.selected_coin} Options Hedging Dashboard")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

    st.sidebar.header("Configuration")
    coin = st.sidebar.selectbox("Cryptocurrency", ["BTC", "ETH"], index=["BTC", "ETH"].index(st.session_state.selected_coin))
    if coin != st.session_state.selected_coin:
        st.session_state.selected_coin = coin
        st.rerun()

    risk_free_rate = st.sidebar.number_input("Risk-Free Rate", value=st.session_state.risk_free_rate_input, min_value=0.0, max_value=0.2, step=0.001, format="%.3f")
    st.session_state.risk_free_rate_input = risk_free_rate
    lookback_days = st.sidebar.number_input("Lookback (days)", min_value=7, max_value=90, value=30, step=7)
    merge_tolerance = st.sidebar.number_input("Merge Tolerance (min)", min_value=1, max_value=60, value=15, step=1)

    with st.spinner("Fetching spot data..."):
        df_krak_5m = fetch_kraken_data(coin=coin, days=lookback_days + 2)
    spot_price = df_krak_5m["close"].iloc[-1] if not df_krak_5m.empty else np.nan

    with st.spinner("Fetching instruments..."):
        all_instruments_list = fetch_instruments()
    if not all_instruments_list:
        st.error("Failed to fetch instruments.")
        st.stop()

    valid_expiries = get_valid_expiration_options(CURRENT_TIME_UTC)
    if not valid_expiries:
        st.error(f"No valid expiries for {coin}.")
        st.stop()

    expiry_oi_map = {}
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {
            executor.submit(
                lambda x: (
                    x,
                    sum(
                        fetch_ticker(instr).get('open_interest', 0)
                        for instr in get_option_instruments(all_instruments_list, "C", x.strftime("%d%b%y").upper(), coin)[:50] +
                        get_option_instruments(all_instruments_list, "P", x.strftime("%d%b%y").upper(), coin)[:50]
                    )
                ),
                exp
            ): exp for exp in valid_expiries
        }
        for future in as_completed(futures):
            try:
                exp, oi = future.result()
                expiry_oi_map[exp] = oi
            except Exception as e:
                logger.warning(f"Error calculating OI for expiry: {e}")

    default_exp = max(expiry_oi_map.items(), key=lambda x: x[1])[0] if expiry_oi_map else valid_expiries[0]
    selected_expiry = st.sidebar.selectbox(
        "Expiry", valid_expiries, index=valid_expiries.index(default_exp), format_func=lambda d: d.strftime("%d %b %Y")
    )
    e_str = selected_expiry.strftime("%d%b%y").upper()
    all_instr_selected_expiry = get_option_instruments(all_instruments_list, "C", e_str, coin) + get_option_instruments(all_instruments_list, "P", e_str, coin)

    st.header(f"{coin} | Expiry: {e_str} | Spot: ${spot_price:,.2f}" if not np.isnan(spot_price) else f"{coin} | Expiry: {e_str} | Spot: N/A")
    st.markdown(f"*Snapshot: {CURRENT_TIME_UTC.strftime('%Y-%m-%d %H:%M:%S UTC')} | RF Rate: {risk_free_rate:.3%}*")

    if not all_instr_selected_expiry:
        st.error(f"No options for {e_str}.")
        st.stop()

    with st.spinner("Fetching options data..."):
        dft_raw = fetch_data(tuple(all_instr_selected_expiry), historical_lookback_days=lookback_days)
    with st.spinner("Fetching ticker data..."):
        ticker_data = fetch_ticker_batch(all_instr_selected_expiry)
    valid_tickers = [k for k, v in ticker_data.items() if v.get('iv', 0) > 1e-4 and pd.notna(v.get('open_interest'))]
    dft = dft_raw[dft_raw['instrument_name'].isin(valid_tickers)].copy() if not dft_raw.empty else pd.DataFrame()
    if not dft.empty:
        dft['open_interest'] = [ticker_data.get(x, {}).get('open_interest', 0.0) for x in dft['instrument_name']]
        dft['iv_close'] = pd.to_numeric(dft['iv_close'], errors='coerce')

    dft_with_hist_greeks = pd.DataFrame()
    if not dft.empty and not df_krak_5m.empty:
        with st.spinner("Merging and computing Greeks..."):
            merged_hist = pd.merge_asof(
                dft.sort_values('date_time'),
                df_krak_5m[['date_time', 'close']].rename(columns={'close': 'spot_hist'}),
                on='date_time', direction='nearest', tolerance=pd.Timedelta(minutes=merge_tolerance)
            ).dropna(subset=['spot_hist'])
            if not merged_hist.empty:
                dft_with_hist_greeks = compute_greeks_vectorized(
                    merged_hist, merged_hist['spot_hist'].values, merged_hist['date_time'].values, risk_free_rate
                )

    dft_latest = pd.DataFrame()
    if not dft.empty and not np.isnan(spot_price):
        dft_latest_idx = dft.groupby('instrument_name')['date_time'].idxmax()
        dft_latest = dft.loc[dft_latest_idx].copy()
        if not dft_latest.empty:
            dft_latest['open_interest'] = [ticker_data.get(x, {}).get('open_interest', 0.0) for x in dft_latest['instrument_name']]
            dft_latest = compute_greeks_vectorized(dft_latest, spot_price, CURRENT_TIME_UTC, risk_free_rate)

    if not dft_latest.empty:
        ticker_list_latest_snap = [
            {
                'instrument': row['instrument_name'], 'strike': row['k'], 'option_type': row['option_type'],
                'open_interest': float(row['open_interest']), 'delta': float(row['delta']),
                'gamma': float(row['gamma']), 'iv': float(row['iv_close'])
            } for _, row in dft_latest.iterrows()
        ]
        plot_open_interest_delta(ticker_list_latest_snap, spot_price)
        plot_delta_balance(ticker_list_latest_snap, spot_price)

    st.markdown("---")
    st.header("Market Maker Perspective")
    if not dft_latest.empty:
        st.subheader("Net Greek Exposures")
        cols = st.columns(4)
        for i, (greek, value) in enumerate([
            ("Delta", -dft_latest['delta'].mul(dft_latest['open_interest']).sum()),
            ("Gamma", -dft_latest['gamma'].mul(dft_latest['open_interest']).sum()),
            ("Vega", -dft_latest['vega'].mul(dft_latest['open_interest']).sum()),
            ("Theta", -dft_latest['theta'].mul(dft_latest['open_interest']).sum())
        ]):
            cols[i].metric(f"Net {greek}", f"{value:,.2f}")
        display_mm_gamma_adjustment_analysis(dft_latest, spot_price, CURRENT_TIME_UTC, risk_free_rate)

    st.markdown("---")
    st.header("Debug Tables")
    with st.expander("Raw Data"):
        st.dataframe(dft_raw.head(20))
    with st.expander("Filtered Data"):
        st.dataframe(dft.head(20))
    with st.expander("Historical Greeks"):
        st.dataframe(dft_with_hist_greeks.head(20))

    gc.collect()
    logger.info(f"Dashboard rendering complete for {coin} {e_str}")

if __name__ == "__main__":
    main()
