"""
Stock Price Prediction Web Application
Interactive dashboard for real-time predictions using trained LSTM models
Run with: streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import load_model
import ta
from ta.trend import MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
import pickle
import warnings
warnings.filterwarnings('ignore')

# Import model_utils classes at the top
from model_utils import ModelManager, PredictionEngine

# ============================================
# PAGE CONFIGURATION
# ============================================

st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon="📈",
       layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================
# HELPER FUNCTIONS
# ============================================



# @st.cache_data
def load_stock_data(ticker, start_date, end_date):
    """Load stock data from Yahoo Finance with extra validation"""
    try:
        data = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=True  # adjust for splits/dividends
        )

        if data is None or data.empty:
            st.error(f"❌ No data found for {ticker}. Try another ticker or shorter date range.")
            return None

        # Debug info
        st.write(f"✅ Data loaded: {ticker}")
        st.write("Shape:", data.shape)
        st.dataframe(data.head())

        return data

    except Exception as e:
        st.error(f"⚠️ Error fetching data for {ticker}: {e}")
        return None



def add_technical_indicators(data):
    """Add technical indicators to the DataFrame (matches training_notebook.py)"""
    df = data.copy()
    # Moving Averages
    close_series = df['Close'].squeeze().astype(float)
    df['MA_20'] = close_series.rolling(window=20).mean()
    df['MA_50'] = close_series.rolling(window=50).mean()
    # RSI
    rsi = RSIIndicator(close=close_series, window=14)
    df['RSI'] = rsi.rsi()
    # MACD
    macd = MACD(close=close_series)
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    # Bollinger Bands
    bollinger = BollingerBands(close=close_series)
    df['BB_High'] = bollinger.bollinger_hband()
    df['BB_Low'] = bollinger.bollinger_lband()
    # Returns
    df['Returns'] = df['Close'].pct_change()
    # Drop NaN values
    df = df.dropna()
    return df

def plot_technical_indicators(data):
    """Plot technical indicators"""
    fig = go.Figure()
    # Flatten all y-values
    close = np.array(data['Close']).flatten()
    ma20 = np.array(data['MA_20']).flatten()
    ma50 = np.array(data['MA_50']).flatten()
    bb_high = np.array(data['BB_High']).flatten()
    bb_low = np.array(data['BB_Low']).flatten()

    fig.add_trace(go.Scatter(x=data.index, y=close, name='Close', line=dict(color='black')))
    fig.add_trace(go.Scatter(x=data.index, y=ma20, name='MA 20', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=data.index, y=ma50, name='MA 50', line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=data.index, y=bb_high, name='BB High', line=dict(color='gray', dash='dash')))
    fig.add_trace(go.Scatter(x=data.index, y=bb_low, name='BB Low', line=dict(color='gray', dash='dash')))

    fig.update_layout(
        title='Technical Indicators',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        hovermode='x unified',
        template='plotly_white',
        height=400
    )
    return fig

def plot_rsi_macd(data):
    """Plot RSI and MACD"""
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('RSI', 'MACD'),
        vertical_spacing=0.15
    )
    
    # RSI
    fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], name='RSI'), row=1, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=1)
    
    # MACD
    fig.add_trace(go.Scatter(x=data.index, y=data['MACD'], name='MACD'), row=2, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['MACD_Signal'], name='Signal'), row=2, col=1)
    
    fig.update_layout(height=600, template='plotly_white', showlegend=True)
    
    return fig

def create_mock_model(input_shape):
    """Creates a mock Keras model for demonstration purposes."""
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, InputLayer

    model = Sequential([
        InputLayer(input_shape=input_shape),
        LSTM(16, activation='relu', return_sequences=False),
        Dense(1)
    ])
    # Compile with dummy optimizer/loss
    model.compile(optimizer='adam', loss='mse')
    return model

# Utility functions

# def make_predictions(model, data_with_indicators, scaler_target, feature_columns, lookback):
#     # TODO: Implement actual prediction logic
#     # For now, return last 10 closing prices as dummy predictions
#     return data_with_indicators['Close'].tail(10).values

def plot_stock_price(data, predictions=None, title="Stock Price History"):
    import plotly.graph_objects as go
    fig = go.Figure()
    # Flatten actual prices
    actual_prices = np.array(data['Close']).flatten()
    fig.add_trace(go.Scatter(x=data.index, y=actual_prices, mode='lines', name='Actual'))
    if predictions is not None:
        pred_prices = np.array(predictions).flatten()
        fig.add_trace(go.Scatter(x=data.index, y=pred_prices, mode='lines', name='Predicted'))
    fig.update_layout(title=title, xaxis_title='Date', yaxis_title='Price', template='plotly_white')
    return fig

# ============================================
# SIDEBAR
# ============================================

st.sidebar.image("C:\COLLEGE\PROJECT1\stock-favicon.png", width=100)
st.sidebar.title("⚙️ Configuration")

# Stock selection
popular_stocks = {
    'Apple': 'AAPL',
    'Google': 'GOOGL',
    'Microsoft': 'MSFT',
    'Tesla': 'TSLA',
    'Amazon': 'AMZN',
    'Meta': 'META',
    'Netflix': 'NFLX',
    'NVIDIA': 'NVDA'
}

stock_name = st.sidebar.selectbox("Select Stock", list(popular_stocks.keys()))
ticker = popular_stocks[stock_name]

# Custom ticker option
custom_ticker = st.sidebar.text_input("Or enter custom ticker", "")
if custom_ticker:
    ticker = custom_ticker.upper()
    stock_name = ticker

# Date range
st.sidebar.subheader("📅 Date Range")
end_date = datetime.now()
start_date = end_date - timedelta(days=5*365)  # 5 years

start_date_input = st.sidebar.date_input("Start Date", start_date)
end_date_input = st.sidebar.date_input("End Date", end_date)

# Model selection
st.sidebar.subheader("🤖 Model Selection")
model_type = st.sidebar.selectbox(
    "Choose Model",
    ['LSTM', 'BiLSTM', 'GRU']
)

# Prediction days
prediction_days = st.sidebar.slider("Future Prediction Days", 1, 30, 7)

# Load data button
load_button = st.sidebar.button("🔄 Load Data & Predict", use_container_width=True)

# Clear cache button
if st.sidebar.button("🗑️ Clear Cache & Reset", use_container_width=True):
    st.cache_data.clear()
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.success("✅ Cache cleared! Reloading...")
    st.rerun()

# ============================================
# MAIN CONTENT
# ============================================

# Header
st.markdown('<p class="main-header">📈 Stock Price Prediction System</p>', unsafe_allow_html=True)
st.markdown(f"### Analyzing **{stock_name}** ({ticker})")

# Load and process data
if load_button or 'data' not in st.session_state:
    with st.spinner(f'Loading data for {ticker}...'):
        data = load_stock_data(ticker, start_date_input, end_date_input)
        if data is not None and not data.empty and 'Close' in data.columns:
            st.session_state['data'] = data
            st.session_state['ticker'] = ticker
            st.success(f"✅ Successfully loaded {len(data)} days of data!")
        else:
            st.warning("⚠️ No valid data loaded. Please check your internet connection, ticker, and date range.")
            st.session_state['data'] = None

# Process data if available

if (
    'data' in st.session_state
    and st.session_state['data'] is not None
    and not st.session_state['data'].empty
    and 'Close' in st.session_state['data'].columns
):
    data = st.session_state['data']
    # Add technical indicators
    data_with_indicators = add_technical_indicators(data)
    # Current metrics
    st.subheader("📊 Current Metrics")
    col1, col2, col3, col4, col5 = st.columns(5)
    current_price = float(data['Close'].iloc[-1])
    prev_price = float(data['Close'].iloc[-2])
    price_change = current_price - prev_price
    price_change_pct = (price_change / prev_price) * 100
    col1.metric("Current Price", f"${current_price:.2f}", f"{price_change:.2f} ({price_change_pct:.2f}%)")
    col2.metric("High (Today)", f"${float(data['High'].iloc[-1]):.2f}")
    col3.metric("Low (Today)", f"${float(data['Low'].iloc[-1]):.2f}")
    col4.metric("Volume", f"{int(data['Volume'].iloc[-1]):,}")
    col5.metric("52-Week High", f"${float(data['Close'].tail(252).max()):.2f}")
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Price Prediction", "📊 Technical Analysis", "📉 Historical Data", "🎯 Model Performance"])
    
    # ===== TAB 1: PRICE PREDICTION =====
    with tab1:
        st.subheader("🔮 Price Prediction")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Prepare data for prediction
            feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume',
                             'MA_20', 'MA_50', 'RSI', 'MACD', 'MACD_Signal',
                             'BB_High', 'BB_Low', 'Returns']
            
            lookback = 60
            
            # Initialize or reset model when ticker changes
            if 'current_ticker' not in st.session_state or st.session_state['current_ticker'] != ticker:
                st.session_state['current_ticker'] = ticker
                st.session_state.pop('model', None)
                st.session_state.pop('predictions', None)
            
            # Create or load model
            if 'model' not in st.session_state:
                try:
                    manager = ModelManager()
                    model = manager.load_model(model_type, ticker)
                    scaler_target = manager.load_scaler('target', ticker)
                    scaler_features = manager.load_scaler('features', ticker)

                    # Helper to check if scaler is fitted
                    def is_fitted(scaler):
                        return hasattr(scaler, 'n_samples_seen_') or hasattr(scaler, 'scale_')

                    # Fit scalers if not fitted
                    if scaler_target is not None and not is_fitted(scaler_target):
                        scaler_target.fit(data_with_indicators[['Close']])
                    if scaler_features is not None and not is_fitted(scaler_features):
                        scaler_features.fit(data_with_indicators[feature_columns])

                    if model is not None and scaler_target is not None and scaler_features is not None:
                        st.success(f"✅ Loaded trained {model_type} model for {ticker}!")
                        st.session_state['model'] = model
                        st.session_state['scaler_target'] = scaler_target
                        st.session_state['scaler_features'] = scaler_features
                    else:
                        # Fallback to mock model if not found
                        st.warning(f"⚠️ No trained model found for {ticker}. Using demonstration mode.")
                        st.info("💡 Train models first: `python training_notebook.py`")
                        input_shape = (lookback, len(feature_columns))
                        model = create_mock_model(input_shape)
                        scaler_target = MinMaxScaler()
                        scaler_target.fit(data_with_indicators[['Close']])
                        scaler_features = MinMaxScaler()
                        scaler_features.fit(data_with_indicators[feature_columns])
                        st.session_state['model'] = model
                        st.session_state['scaler_target'] = scaler_target
                        st.session_state['scaler_features'] = scaler_features
                except Exception as e:
                    st.error(f"Error loading model/scaler: {e}")
                    # Fallback to mock model
                    input_shape = (lookback, len(feature_columns))
                    model = create_mock_model(input_shape)
                    scaler_target = MinMaxScaler()
                    scaler_target.fit(data_with_indicators[['Close']])
                    scaler_features = MinMaxScaler()
                    scaler_features.fit(data_with_indicators[feature_columns])
                    st.session_state['model'] = model
                    st.session_state['scaler_target'] = scaler_target
                    st.session_state['scaler_features'] = scaler_features
            
            
 
            model = st.session_state['model']
            scaler_target = st.session_state['scaler_target']
            scaler_features = st.session_state['scaler_features']

            # Make predictions with progress bar
            if 'predictions' not in st.session_state:
                progress_bar = st.progress(0)
                status_text = st.empty()

                try:
                    status_text.text('📊 Preparing data...')
                    progress_bar.progress(25)

                    status_text.text('🤖 Generating predictions...')
                    progress_bar.progress(50)

                    # Create PredictionEngine instance
                    engine = PredictionEngine(
                        model,
                        scaler_features,
                        scaler_target,
                        feature_columns,
                        lookback
                    )

                    # Make predictions on the processed data
                    predictions = engine.predict(data_with_indicators)

                    progress_bar.progress(75)
                    status_text.text('✅ Predictions complete!')
                    progress_bar.progress(100)

                    st.session_state['predictions'] = predictions

                    # Clear progress indicators
                    import time
                    time.sleep(0.5)
                    progress_bar.empty()
                    status_text.empty()

                except Exception as e:
                    st.error(f"❌ Prediction error: {str(e)}")
                    st.info("💡 Try: 1) Check if data has all required features, 2) Retrain models, 3) Check data quality")
                    progress_bar.empty()
                    status_text.empty()
            
            # Display predictions if available
            if 'predictions' in st.session_state:
                predictions = st.session_state['predictions']
                actual_data = data_with_indicators.iloc[lookback:]
                # Debug info
                st.write(f"Actual data shape: {actual_data['Close'].shape}")
                st.write(f"Predictions shape: {np.shape(predictions)}")
                st.write(f"First 5 actual prices: {actual_data['Close'].head().values}")
                st.write(f"First 5 predictions: {predictions[:5]}")

                # Check for empty or all NaN
                if (predictions is not None and len(predictions) == len(actual_data)
                    and not np.all(np.isnan(predictions))
                    and not np.all(np.isnan(actual_data['Close'].values))):
                    fig = plot_stock_price(
                        actual_data, 
                        predictions,
                        f"{ticker} - Actual vs Predicted Prices"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning(f"⚠️ Cannot plot Actual vs Predicted: Length mismatch, missing predictions, or all values are NaN. Actual data length: {len(actual_data)}, Predictions length: {len(predictions) if predictions is not None else 'None'}.")
            else:
                st.warning("⚠️ No predictions available. Click 'Load Data & Predict' in the sidebar.")
        
        with col2:
            st.markdown("### 📈 Prediction Summary")
            
            if 'predictions' in st.session_state:
                predictions = st.session_state['predictions']
                last_actual = float(data['Close'].iloc[-1])
                last_predicted = float(predictions[-1][0]) if hasattr(predictions[-1], '__iter__') else float(predictions[-1])
                
                # Calculate prediction difference
                pred_diff = last_predicted - last_actual
                pred_diff_pct = (pred_diff / last_actual) * 100
                
                # Future trend
                if pred_diff > 0:
                    future_trend = "📈 Bullish (Upward)"
                    trend_color = "green"
                else:
                    future_trend = "📉 Bearish (Downward)"
                    trend_color = "red"
                
                st.markdown(f"""
                <div class="metric-card" style="background-color: #000; color: #fff;">
                    <h4>Latest Prediction</h4>
                    <p style="font-size: 2rem; font-weight: bold; color: #1f77b4;">
                        ${last_predicted:.2f}
                    </p>
                    <p>Actual: ${last_actual:.2f}</p>
                    <p style="color: {trend_color};">
                        Difference: ${pred_diff:.2f} ({pred_diff_pct:+.2f}%)
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="metric-card" style="margin-top: 1rem; background-color: #000; color: #fff;">
                <h4>Trend Direction</h4>
                <p style="font-size: 1.5rem; color: {trend_color};">{future_trend}</p>
                <p style="font-size: 0.9rem;">Based on {model_type} prediction</p>
                </div>
                """, unsafe_allow_html=True)
                
                
                # Calculate accuracy metrics
                actual_prices = data_with_indicators['Close'].iloc[lookback:].values
                pred_prices = predictions.flatten()
                
                if len(actual_prices) == len(pred_prices):
                    mae = np.mean(np.abs(actual_prices - pred_prices))
                    mape = np.mean(np.abs((actual_prices - pred_prices) / actual_prices)) * 100
                    accuracy = max(0, 100 - mape)
                    st.markdown(f"""
                    <div class="metric-card" style="margin-top: 1rem; background-color: #000; color: #fff;">
                        <h4>Model Accuracy</h4>
                        <p style="font-size: 1.5rem; font-weight: bold; color: green;">
                            {accuracy:.2f}%
                        </p>
                        <p style="font-size: 0.9rem;">MAE: ${mae:.2f}</p>
                        <p style="font-size: 0.9rem;">MAPE: {mape:.2f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("👈 Click 'Load Data & Predict' to see predictions")
            
            st.markdown("---")
            st.markdown("### ⚙️ Model Info")
            st.info(f"""
            **Model Type:** {model_type}  
            **Lookback Period:** {lookback} days  
            **Features:** {len(feature_columns)}  
            **Training Data:** {len(data)} days
            """)
            
            # Add refresh button
            if st.button("🔄 Refresh Predictions", use_container_width=True):
                st.session_state.pop('predictions', None)
                st.rerun()
    
    # ===== TAB 2: TECHNICAL ANALYSIS =====
    with tab2:
        st.subheader("📊 Technical Indicators Analysis")
        
        # Technical indicators plot
        fig_tech = plot_technical_indicators(data_with_indicators.tail(180))
        st.plotly_chart(fig_tech, use_container_width=True)
        
        # RSI and MACD
        fig_rsi_macd = plot_rsi_macd(data_with_indicators.tail(180))
        st.plotly_chart(fig_rsi_macd, use_container_width=True)
        
        # Trading signals
        st.subheader("🎯 Trading Signals")
        
        col1, col2, col3 = st.columns(3)
        
        # RSI Signal
        if 'RSI' in data_with_indicators.columns and not data_with_indicators['RSI'].empty:
            current_rsi = data_with_indicators['RSI'].iloc[-1]
            if current_rsi > 70:
                rsi_signal = "🔴 Overbought"
                rsi_color = "red"
            elif current_rsi < 30:
                rsi_signal = "🟢 Oversold"
                rsi_color = "green"
            else:
                rsi_signal = "🟡 Neutral"
                rsi_color = "orange"
            col1.markdown(f"""
            <div class="metric-card" style="background-color: #000; color: #fff;">
                <h4>RSI Signal</h4>
                <p style="font-size: 1.5rem; color: {rsi_color};">{rsi_signal}</p>
                <p>Value: {current_rsi:.2f}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            col1.warning("RSI data not available.")

        # MACD Signal
        if ('MACD' in data_with_indicators.columns and not data_with_indicators['MACD'].empty and
            'MACD_Signal' in data_with_indicators.columns and not data_with_indicators['MACD_Signal'].empty):
            current_macd = data_with_indicators['MACD'].iloc[-1]
            current_signal = data_with_indicators['MACD_Signal'].iloc[-1]
            if current_macd > current_signal:
                macd_signal = "🟢 Bullish"
                macd_color = "green"
            else:
                macd_signal = "🔴 Bearish"
                macd_color = "red"
            col2.markdown(f"""
            <div class="metric-card" style="background-color: #000; color: #fff;">
                <h4>MACD Signal</h4>
                <p style="font-size: 1.5rem; color: {macd_color};">{macd_signal}</p>
                <p>MACD: {current_macd:.2f}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            col2.warning("MACD data not available.")

        # Bollinger Bands Signal
        if ('BB_High' in data_with_indicators.columns and not data_with_indicators['BB_High'].empty and
            'BB_Low' in data_with_indicators.columns and not data_with_indicators['BB_Low'].empty and
            'Close' in data_with_indicators.columns and not data_with_indicators['Close'].empty):
            current_close = float(data_with_indicators['Close'].iloc[-1])
            bb_high = float(data_with_indicators['BB_High'].iloc[-1])
            bb_low = float(data_with_indicators['BB_Low'].iloc[-1])
            if current_close > bb_high:
                bb_signal = "🔴 Above Upper Band"
                bb_color = "red"
            elif current_close < bb_low:
                bb_signal = "🟢 Below Lower Band"
                bb_color = "green"
            else:
                bb_signal = "🟡 Within Bands"
                bb_color = "orange"
            col3.markdown(f"""
            <div class="metric-card" style="background-color: #000; color: #fff;">
                <h4>Bollinger Bands</h4>
                <p style="font-size: 1.5rem; color: {bb_color};">{bb_signal}</p>
                <p>Price: ${current_close:.2f}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            col3.warning("Bollinger Bands data not available.")
        # col3.markdown(f"""
        # <div class="metric-card">
        #     <h4>Bollinger Bands</h4>
        #     <p style="font-size: 1.5rem; color: {bb_color};">{bb_signal}</p>
        #     <p>Price: ${current_close:.2f}</p>
        # </div>
        # """, unsafe_allow_html=True)
    
    # ===== TAB 3: HISTORICAL DATA =====
    with tab3:
        st.subheader("📉 Historical Price Data")
        
        # Candlestick chart
        last90 = data.tail(90)
        x_vals = last90.index
        open_vals = np.array(last90['Open']).flatten()
        high_vals = np.array(last90['High']).flatten()
        low_vals = np.array(last90['Low']).flatten()
        close_vals = np.array(last90['Close']).flatten()
        fig_candle = go.Figure(data=[go.Candlestick(
            x=x_vals,
            open=open_vals,
            high=high_vals,
            low=low_vals,
            close=close_vals
        )])

        fig_candle.update_layout(
            title=f'{ticker} - Last 90 Days Candlestick Chart',
            yaxis_title='Price ($)',
            xaxis_title='Date',
            template='plotly_white',
            height=500
        )

        st.plotly_chart(fig_candle, use_container_width=True)
        
        # Volume chart
        last90 = data.tail(90)
        x_vals = last90.index
        volume_vals = np.array(last90['Volume']).flatten()
        fig_volume = go.Figure()
        fig_volume.add_trace(go.Bar(
            x=x_vals,
            y=volume_vals,
            name='Volume',
            marker_color='lightblue'
        ))

        fig_volume.update_layout(
            title='Trading Volume',
            yaxis_title='Volume',
            xaxis_title='Date',
            template='plotly_white',
            height=300
        )

        st.plotly_chart(fig_volume, use_container_width=True)
        
        # Data table
        st.subheader("📋 Recent Data")
        display_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA_20', 'MA_50', 'RSI']
        st.dataframe(
            data_with_indicators[display_columns].tail(20).style.format({
                'Open': '${:.2f}',
                'High': '${:.2f}',
                'Low': '${:.2f}',
                'Close': '${:.2f}',
                'Volume': '{:,.0f}',
                'MA_20': '${:.2f}',
                'MA_50': '${:.2f}',
                'RSI': '{:.2f}'
            }),
            use_container_width=True
        )
        
        # Download data
        csv = data_with_indicators.to_csv()
        st.download_button(
            label="📥 Download Data as CSV",
            data=csv,
            file_name=f"{ticker}_stock_data.csv",
            mime="text/csv"
        )
    
    # ===== TAB 4: MODEL PERFORMANCE =====
    with tab4:
        st.subheader("🎯 Model Performance Metrics")
        
        # Mock performance comparison
        performance_data = {
            'Model': ['LSTM', 'BiLSTM', 'GRU'],
            'RMSE': [12.45, 11.87, 13.21],
            'MAE': [9.32, 8.95, 10.11],
            'MAPE': [3.45, 3.12, 3.89],
            'Accuracy': [96.55, 96.88, 96.11]
        }
        
        df_performance = pd.DataFrame(performance_data)
        
        # Display metrics
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 📊 Model Comparison")
            st.dataframe(
                df_performance.style.highlight_min(
                    subset=['RMSE', 'MAE', 'MAPE'],
                    color='lightgreen'
                ).highlight_max(
                    subset=['Accuracy'],
                    color='lightgreen'
                ),
                use_container_width=True
            )
        
        with col2:
            # Bar chart comparison
            fig_comparison = go.Figure()
            
            metrics = ['RMSE', 'MAE', 'MAPE']
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
            
            for i, metric in enumerate(metrics):
                fig_comparison.add_trace(go.Bar(
                    name=metric,
                    x=df_performance['Model'],
                    y=df_performance[metric],
                    marker_color=colors[i]
                ))
            
            fig_comparison.update_layout(
                title='Model Performance Comparison',
                barmode='group',
                template='plotly_white',
                height=400
            )
            
            st.plotly_chart(fig_comparison, use_container_width=True)
        
        st.markdown("---")
        
        # Training information
        col1, col2, col3 = st.columns(3)
        
        col1.markdown("""
    <div class="metric-card" style="background-color: #000; color: #fff;">
            <h4>Training Details</h4>
            <p>Epochs: 50</p>
            <p>Batch Size: 32</p>
            <p>Optimizer: Adam</p>
            <p>Loss: MSE</p>
        </div>
        """, unsafe_allow_html=True)
        
        col2.markdown("""
    <div class="metric-card" style="background-color: #000; color: #fff;">
            <h4>Dataset Info</h4>
            <p>Train: 80%</p>
            <p>Test: 20%</p>
            <p>Features: 13</p>
            <p>Lookback: 60 days</p>
        </div>
        """, unsafe_allow_html=True)
        
        col3.markdown("""
    <div class="metric-card" style="background-color: #000; color: #fff;">
            <h4>Best Model</h4>
            <p style="font-size: 1.5rem; color: green; font-weight: bold;">BiLSTM</p>
            <p>RMSE: 11.87</p>
            <p>Accuracy: 96.88%</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.info("""
        **Note:** These are demonstration metrics. Train actual models using the provided training notebook 
        with your specific stock data to get real performance metrics.
        """)

else:
    st.info("👈 Please configure settings in the sidebar and click 'Load Data & Predict' to begin.")
# ============================================
# FOOTER
# ============================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray;">
    <p>📈 Stock Price Prediction System | Built with LSTM Deep Learning</p>
    <p style="font-size: 0.8rem;">
        Disclaimer: This is for educational purposes only. Not financial advice.
    </p>
</div>
""", unsafe_allow_html=True)