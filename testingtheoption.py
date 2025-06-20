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
                    merged_hist, merged_hist['spot_hist'].values, CURRENT_TIME_UTC, risk_free_rate
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

        # Move widget inputs here
        st.sidebar.header("Hedging Positions")
        steth_position = st.sidebar.number_input("stETH Position Size", value=0.0, step=0.1, key="steth_pos")
        inverse_position = st.sidebar.number_input("Inverse Position Size", value=0.0, step=0.1, key="inverse_pos")
        short_put_position = st.sidebar.number_input("Short Put Position Size", value=0.0, step=0.1, key="short_put_pos")
        
        # Fetch funding rate outside cached function
        funding_rate_df = fetch_funding_rates(None, f"{coin}/USDT", days=7)
        funding_rate = funding_rate_df.iloc[-1]['funding_rate'] if not funding_rate_df.empty else 0.0

        # Call analysis with precomputed inputs
        with st.spinner("Computing hedging analysis..."):
            analysis_result = display_mm_gamma_adjustment_analysis(
                dft_latest, spot_price, CURRENT_TIME_UTC, risk_free_rate,
                steth_position, inverse_position, short_put_position, funding_rate
            )

        # Display results
        st.subheader("MM Delta-Gamma Hedge Adjustment")
        if analysis_result:
            st.metric("MM Initial Net Delta", f"{analysis_result['mm_net_delta_initial']:,.2f}")
            st.metric("MM Initial Net Gamma", f"{analysis_result['mm_net_gamma_initial']:,.4f}")
            st.metric("MM Initial Net Theta", f"{analysis_result['mm_net_theta_initial']:,.2f}")

            if analysis_result['gamma_hedger'] is not None:
                st.info(f"Gamma Hedger: {analysis_result['gamma_hedger']['instrument_name']}")
            if analysis_result['theta_hedger'] is not None:
                st.info(f"Theta Hedger: {analysis_result['theta_hedger']['instrument_name']}")

            st.markdown("#### Hedge Adjustments")
            cols = st.columns(4)
            cols[0].metric("Gamma Hedger Delta", f"{analysis_result['metrics']['gamma_hedger_delta']:.4f}")
            cols[1].metric("Gamma Hedger Gamma", f"{analysis_result['metrics']['gamma_hedger_gamma']:.6f}")
            cols[2].metric(f"Gamma Hedge Qty ({analysis_result['metrics']['gamma_hedge_action']})", f"{analysis_result['metrics']['gamma_hedge_qty']:,.2f}")
            cols[3].metric("Put Delta", f"{analysis_result['metrics']['put_delta']:.4f}")

            st.metric("Delta from Gamma Hedge", f"{analysis_result['metrics']['delta_from_gamma_hedge']:,.2f}")
            st.metric("Delta from Theta Hedge", f"{analysis_result['metrics']['delta_from_theta_hedge']:,.2f}")
            st.metric("Delta from Extras", f"{analysis_result['metrics']['total_delta_extras']:,.2f}")
            st.metric("Theta from Theta Hedge", f"{analysis_result['metrics']['theta_from_theta_hedge']:,.2f}")
            st.metric("Theta from Short Puts", f"{analysis_result['metrics']['total_theta_extras']:,.2f}")

            st.markdown("#### Final Hedge")
            st.metric("MM Net Delta (Post-Hedge)", f"{analysis_result['metrics']['mm_net_delta_post_hedge']:,.2f}")
            st.metric("MM Net Theta (Post-Hedge)", f"{analysis_result['metrics']['mm_net_theta_post_hedge']:,.2f}")
            st.metric(f"Underlying Hedge ({analysis_result['metrics']['underlying_hedge_action']})", f"{analysis_result['metrics']['underlying_hedge_qty']:,.2f} {coin}")
            st.success(f"**Resulting Net Delta:** {analysis_result['metrics']['final_net_delta']:,.4f}")
        else:
            st.warning("No valid hedging analysis results.")

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
