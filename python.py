import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Configuration
INITIAL_INVESTMENT_CAD = 1000000.00
BENCHMARK_TICKERS = ["^GSPTSE", "^GSPC"]
FX_TICKER = "CADUSD=X" # For converting USD benchmark to CAD

st.set_page_config(layout="wide", page_title="Portfolio Visualizer")

st.title("💰 Portfolio Visualizer (CAD)")
st.caption(f"Initial Investment: {INITIAL_INVESTMENT_CAD:,.2f} CAD. All prices are in CAD. Note - doesn't account for fees but the result should be close enough for visualization purposes.")

@st.cache_data(ttl=300)
def fetch_historical_data(tickers, start_date):
    """Fetches historical adjusted close prices for a list of tickers."""
    try:
        data = yf.download(tickers, start=start_date)['Close']
        # For a single ticker, the result is a Series, convert to DataFrame
        if isinstance(data, pd.Series):
            data = data.to_frame()
        return data.ffill().dropna()
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

def calculate_portfolio_metrics(portfolio_df, prices_df, fx_rates):
    """Calculates daily portfolio value, returns, and initial weights."""
    if portfolio_df.empty or prices_df.empty:
        return None, None, None

    # 1. Prepare Price Data (Align Index and apply FX)
    prices = prices_df.copy()
    
    portfolio_tickers = portfolio_df['Ticker'].tolist()
    
    # Assume non-.TO tickers are USD and require conversion using the fetched CADUSD=X rate
    usd_tickers = [t for t in portfolio_tickers if not t.endswith('.TO')]
    
    if fx_rates is not None and len(usd_tickers) > 0:
        fx_series = fx_rates.rename('FX_RATE')
        
        # Align prices and FX rates
        prices = pd.concat([prices_df, fx_series], axis=1).dropna()
        
        for ticker in usd_tickers:
            if ticker in prices.columns:
                # CRITICAL FIX: CADUSD=X is USD per CAD, so to convert USD to CAD we divide
                # P_CAD = P_USD / (USD per CAD) = P_USD * (CAD per USD)
                prices[ticker] = prices[ticker] / prices['FX_RATE']

        # Drop FX rate column before filtering to portfolio tickers
        prices = prices.drop(columns=['FX_RATE'])
    else:
        # If FX rates are missing, use prices_df as is
        prices = prices_df.copy()

    # Filter prices to only include portfolio tickers
    prices = prices[prices.columns.intersection(portfolio_tickers)]

    if prices.empty:
        st.error("No valid price data available for your portfolio tickers.")
        return None, None, None

    # Merge prices and holdings data
    holdings = portfolio_df.set_index('Ticker')['Shares']
    
    initial_prices = prices.iloc[0].sort_index()
    
    # Calculate initial market value for each holding based on initial price
    initial_market_values = holdings * initial_prices.reindex(holdings.index, fill_value=0)
    
    # Calculate initial weights based on market values
    total_initial_market_value = initial_market_values.sum()
    initial_weights = initial_market_values / total_initial_market_value

    if total_initial_market_value == 0:
        st.error("Total initial market value is zero. Check prices or shares.")
        return None, None, None
        
    # Scale initial weights to match INITIAL_INVESTMENT_CAD
    scaling_factor = INITIAL_INVESTMENT_CAD / total_initial_market_value
    
    # Calculate daily portfolio value by multiplying shares by daily prices
    portfolio_value_components = prices * holdings
    portfolio_value = portfolio_value_components.sum(axis=1)
    
    # Apply the scaling factor to the entire portfolio value series
    scaled_portfolio_value = portfolio_value * scaling_factor
    
    # Calculate Daily Returns
    daily_returns = scaled_portfolio_value.pct_change().dropna()
    
    # Prepare the initial weighting DataFrame for the pie chart
    weighting_df = pd.DataFrame({
        'Ticker': initial_weights.index,
        'Weight': initial_weights.values * 100
    })

    return scaled_portfolio_value, daily_returns, weighting_df.sort_values(by='Weight', ascending=False)


def calculate_benchmark_performance(prices_df, fx_rates):
    """Calculates the combined daily return for S&P 500 (^GSPC) and TSX Composite (^GSPTSE) (50/50 weighted), converted to CAD."""
    
    # Get benchmark prices
    benchmark_prices = prices_df[BENCHMARK_TICKERS].copy()

    if FX_TICKER in prices_df.columns:
        # CADUSD=X gives USD per CAD (e.g., 0.72 USD per 1 CAD)
        # To convert USD to CAD, we need CAD per USD, which is 1/FX_RATE
        # Or equivalently: P_CAD = P_USD / FX_RATE
        fx_rates = prices_df[FX_TICKER].rename('FX_RATE')
        
        if not benchmark_prices.empty and not fx_rates.empty:
            # Reindex to the common date range
            combined_data = pd.concat([benchmark_prices, fx_rates], axis=1).dropna()
            
            if '^GSPC' in combined_data.columns:
                # Convert ^GSPC (USD) to CAD by dividing by (USD per CAD) rate
                combined_data['^GSPC_CAD'] = combined_data['^GSPC'] / combined_data['FX_RATE']
                
                benchmark_prices['^GSPC_CAD'] = combined_data['^GSPC_CAD']
                
                # Drop original USD price
                benchmark_prices = benchmark_prices.drop(columns=['^GSPC'])
                benchmark_prices = benchmark_prices.rename(columns={'^GSPC_CAD': '^GSPC'})
            
    # Calculate daily returns for benchmarks
    benchmark_returns = benchmark_prices.pct_change().dropna()
    
    # Create combined benchmark (50% TSX, 50% S&P 500 CAD)
    benchmark_returns['Combined_Benchmark'] = (
        benchmark_returns['^GSPTSE'] * 0.5 + 
        benchmark_returns['^GSPC'] * 0.5
    )
    
    # Calculate cumulative return starting from INITIAL_INVESTMENT_CAD
    initial_value = INITIAL_INVESTMENT_CAD
    
    # Cumulative value: (1 + R1) * (1 + R2) * ... * Initial_Value
    cumulative_benchmark_value = (1 + benchmark_returns['Combined_Benchmark']).cumprod() * initial_value
    
    return cumulative_benchmark_value.to_frame(name='Combined_Benchmark_Value')


def plot_individual_performance(portfolio_df, prices_df, fx_rates):
    # adds a new plot to show individual performance of each ticker in the portfolio 

    st.write("### Individual Ticker Performance")
    st.write("Select tickers to display")

    selected_tickers = st.multiselect(
        "Choose tickers:",
        options=portfolio_df['Ticker'].unique().tolist(),
        default=portfolio_df['Ticker'].unique().tolist()
    )
    
    fig = go.Figure()
    fig.update_layout(width=1200, height=600)

    for index, row in portfolio_df.iterrows():
        ticker = row['Ticker']

        if ticker not in selected_tickers:
            continue

        if ticker in prices_df.columns:
            price_series = prices_df[ticker].copy()

            # Convert to CAD if necessary
            if not ticker.endswith('.TO') and fx_rates is not None:
                fx_series = fx_rates.rename('FX_RATE')
                # CRITICAL FIX: Divide by FX_RATE to convert USD to CAD
                price_series = price_series / fx_series
            
            # Calculate individual investment value over time (cumulative return starting at 1.0)
            returns = price_series.pct_change().dropna()
            cumulative_return = (1 + returns).cumprod()
            
            # Convert to percentage for display
            cumulative_return_pct = (cumulative_return - 1) * 100
            
            fig.add_trace(go.Scatter(x=cumulative_return_pct.index, y=cumulative_return_pct,
                                     mode='lines', name=f'{ticker}'))
    
    fig.update_layout(
        title='Individual Ticker Performance (% Return)',
        xaxis_title='Date',
        yaxis_title='Return (%)',
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )

    fig.update_yaxes(ticksuffix="%")

    st.plotly_chart(fig, use_container_width=True)


def plot_performance(portfolio_value_df, benchmark_value_df):
    """Generates a Plotly line chart comparing portfolio and benchmark performance."""
    
    combined_df = pd.concat([portfolio_value_df, benchmark_value_df], axis=1).dropna()
    
    fig = go.Figure()
    fig.update_layout(width=1200, height=600)

    fig.add_trace(go.Scatter(x=combined_df.index, y=combined_df['Portfolio_Value'],
                             mode='lines', name='Portfolio Value (CAD)'))

    fig.add_trace(go.Scatter(x=combined_df.index, y=combined_df['Combined_Benchmark_Value'],
                             mode='lines', name='Combined Benchmark (CAD)'))
    
    fig.update_layout(
        title='Portfolio vs. Combined Benchmark Performance',
        xaxis_title='Date',
        yaxis_title='Value (CAD)',
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    fig.update_yaxes(tickprefix="$")

    st.plotly_chart(fig, use_container_width=True)


# --- Streamlit Main Application ---

# slider for number of TRADING days to look back
num_trading_days = st.slider("Number of trading days to look back", min_value=1, max_value=63, value=30, step=1)

# Convert trading days to calendar days (approximate: trading days * 1.4 to account for weekends/holidays)
# This ensures we fetch enough data to get the desired number of trading days
calendar_days_to_fetch = int(num_trading_days * 1.4) + 5  # Add buffer
start_date = (datetime.now() - timedelta(days=calendar_days_to_fetch)).strftime('%Y-%m-%d')

uploaded_file = st.file_uploader("Upload CSV file (must contain 'Ticker', 'Weight (%)', and 'Shares' columns)", type="csv")

if uploaded_file is not None:
    try:
        portfolio_df = pd.read_csv(uploaded_file)
        
        # Define required columns
        TICKER_COL = 'Ticker'
        SHARES_COL = 'Shares'
        WEIGHT_COL = 'Weight (%)'
        REQUIRED_COLS = [TICKER_COL, SHARES_COL, WEIGHT_COL]

        # Validation
        if not all(col in portfolio_df.columns for col in REQUIRED_COLS):
            st.error(f"The uploaded CSV must contain columns: {REQUIRED_COLS}")
            portfolio_df = pd.DataFrame()
        else:
            # Clean data:
            
            # Ticker: string, uppercase
            portfolio_df[TICKER_COL] = portfolio_df[TICKER_COL].astype(str).str.strip().str.upper()
            
            # Shares: numeric
            portfolio_df[SHARES_COL] = pd.to_numeric(portfolio_df[SHARES_COL], errors='coerce')
            
            # Weight (%): numeric, converted to fraction (used only for display, but validated)
            portfolio_df[WEIGHT_COL] = pd.to_numeric(portfolio_df[WEIGHT_COL], errors='coerce')
            portfolio_df[WEIGHT_COL] = portfolio_df[WEIGHT_COL] / 100.0
            
            # Drop rows with invalid data in required columns
            portfolio_df = portfolio_df.dropna(subset=[TICKER_COL, SHARES_COL, WEIGHT_COL])
            
            portfolio_tickers = portfolio_df[TICKER_COL].unique().tolist()

            if not portfolio_tickers:
                st.warning("No valid tickers or shares found in the CSV.")
            else:
                st.success(f"Portfolio loaded with {len(portfolio_tickers)} unique tickers.")
                
                all_tickers = portfolio_tickers + BENCHMARK_TICKERS + [FX_TICKER]
                
                # Fetch data, caching the result
                with st.spinner(f"Fetching data for all assets..."):
                    all_prices_df = fetch_historical_data(all_tickers, start_date)
                
                if all_prices_df.empty:
                    st.error("Could not retrieve sufficient market data. Please check ticker symbols.")
                else:
                    # Trim data to only include the requested number of trading days
                    if len(all_prices_df) > num_trading_days:
                        all_prices_df = all_prices_df.iloc[-num_trading_days:]
                    
                    st.write("---")
                    
                    # 2. Benchmark Calculation
                    benchmark_value_df = calculate_benchmark_performance(all_prices_df, all_prices_df.get(FX_TICKER))
                    
                    # 3. Portfolio Calculation
                    portfolio_value, daily_returns, weighting_df = calculate_portfolio_metrics(
                        portfolio_df, all_prices_df, all_prices_df.get(FX_TICKER)
                    )

                    if portfolio_value is not None and benchmark_value_df is not None:
                        
                        # Align data to the intersection of dates
                        final_data = pd.concat([portfolio_value.rename('Portfolio_Value'),
                                                 benchmark_value_df['Combined_Benchmark_Value']],
                                                axis=1).dropna()
                        
                        if len(final_data) > 0:
                            
                            # Display Key Metrics
                            current_portfolio_value = final_data['Portfolio_Value'].iloc[-1]
                            current_benchmark_value = final_data['Combined_Benchmark_Value'].iloc[-1]
                            
                            portfolio_return = (current_portfolio_value / INITIAL_INVESTMENT_CAD) - 1
                            benchmark_return = (current_benchmark_value / INITIAL_INVESTMENT_CAD) - 1
                            
                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric("Current Portfolio Value (CAD)", f"${current_portfolio_value:,.2f}")
                            col2.metric("Portfolio Return", f"{portfolio_return:.2%}", delta=f"{portfolio_return - benchmark_return:.2%} vs Benchmark")
                            col3.metric("Combined Benchmark Value (CAD)", f"${current_benchmark_value:,.2f}")
                            col4.metric("Benchmark Return", f"{benchmark_return:.2%}")
                            
                            st.write("---")
                            
                            # 4. Plot Performance
                            plot_performance(final_data['Portfolio_Value'].to_frame(), final_data['Combined_Benchmark_Value'].to_frame('Combined_Benchmark_Value'))
                            
                            plot_individual_performance(portfolio_df, all_prices_df, all_prices_df.get(FX_TICKER))

                        else:
                            st.warning("Insufficient data to plot performance.")
                    
    except Exception as e:
        st.error(f"An error occurred while processing the CSV file or market data: {e}")