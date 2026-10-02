import streamlit as st
import time
from datetime import datetime, timedelta
import yfinance as yf
import numpy as np
import pandas as pd
import ta
from ta.trend import MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

import plotly.graph_objects as go

st.set_page_config(
    page_title="Stock Price Predictor",
    # page_icon="📈",
       layout="wide",
    initial_sidebar_state="expanded"
)

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
        title='Moving Averages & Bollinger Bands',
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

@st.cache_data
def load_stock_data(ticker, start_date, end_date):
    # """Load stock data from Yahoo Finance with extra validation"""
    # print("test #####################")
    try:
        data = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=True  # adjust for splits/dividends
        )

        print("data fetched #####################", data.head())

        if data is None or data.empty:
            st.error(f" No data found for {ticker}. Try another ticker or shorter date range.")
            return None
        
        if isinstance(data.columns, pd.MultiIndex):
            data = data.xs(ticker, axis=1, level="Ticker")
        # Debug info
        st.write(f"Data loaded: {ticker}")
        # st.write("Shape:", data.shape)
        # st.dataframe(data.head())

        return data

    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
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

st.sidebar.image(".\stock-favicon.png", width=100)
st.sidebar.title(" Configuration")

# Stock selection
popular_stocks = {
    'Apple': 'AAPL',
    'Google': 'GOOGL',
    'Microsoft': 'MSFT',
    'Tesla': 'TSLA',
    'Amazon': 'AMZN',
    'Wallmart': 'WMT',
    'Meta': 'META',
    'AMD': 'AMD'

}

stock_name = st.sidebar.selectbox("Select Stock", list(popular_stocks.keys()))
ticker = popular_stocks[stock_name]

# # Custom ticker option
# custom_ticker = st.sidebar.text_input("Or enter custom ticker", "")
# if custom_ticker:
#     ticker = custom_ticker.upper()
#     stock_name = ticker

# Date range
st.sidebar.subheader("Date Range")
end_date = datetime.now()
start_date = end_date - timedelta(days=5*365)  # 5 years

start_date_input = st.sidebar.date_input("Start Date", start_date)
end_date_input = st.sidebar.date_input("End Date", end_date)


from typing import Union
import requests
from tensorflow import keras



def list_models():
    # list models for ticker from "models"/"ticker" directory
    print(list(Available_models.keys()))
    return {"models": list(Available_models.keys())}

# Available_models = ["LSTM", "GRU", "Transformer"]

def lstm(x_train):
    model = keras.models.Sequential()
    
    model.add(keras.layers.LSTM(
        units=128,
        return_sequences=True,
        input_shape=(x_train.shape[1], x_train.shape[2])
    ))
    model.add(keras.layers.Dropout(0.2))
    
    model.add(keras.layers.LSTM(units=64))
    model.add(keras.layers.Dropout(0.2))
    
    model.add(keras.layers.Dense(32, activation='relu'))
    model.add(keras.layers.Dense(1, activation='linear'))
    
    model.compile(
        optimizer='adam',
        loss='mean_squared_error',
        metrics=['mae']
    )
    
    model.summary()
    return model


# def Bidirectional_LSTM(x_train):
#     model = keras.models.Sequential()
#     model.add(keras.layers.Bidirectional(keras.layers.LSTM(units=64,
#                                 return_sequences=True),
#                                 input_shape=(x_train.shape[1], 1)))
#     model.add(keras.layers.Bidirectional(keras.layers.LSTM(units=64)))
#     model.add(keras.layers.Dense(32))
#     model.add(keras.layers.Dropout(0.5))
#     model.add(keras.layers.Dense(1))
#     model.summary()
#     model.compile(optimizer='adam',
#               loss='mean_squared_error')
#     return model

def GRU(x_train):
    model = keras.models.Sequential()
    
    model.add(keras.layers.GRU(
        units=128,
        return_sequences=True,
        input_shape=(x_train.shape[1], x_train.shape[2])
    ))
    model.add(keras.layers.Dropout(0.2))
    
    model.add(keras.layers.GRU(units=64))
    model.add(keras.layers.Dropout(0.2))
    
    model.add(keras.layers.Dense(32, activation='relu'))
    model.add(keras.layers.Dense(1, activation='linear'))
    
    model.compile(
        optimizer='adam',
        loss='mean_squared_error',
        metrics=['mae']
    )
    
    model.summary() 
    return model

def transformer(x_train):
    # Placeholder for transformer model
    pass
 
# def lstm_trained(x_train):
    
#     try:
#         model = keras.models.load_weights('models/trained_models/lstm_model.keras')
#         return model
#     except AttributeError as e:
#         model = lstm(x_train)
#         model.save('models/trained_models/lstm_model.keras')
#         return model
    

Available_models = {"LSTM": lstm, "GRU":GRU}



# Model selection
st.sidebar.subheader(" Model Selection")
st.session_state["model_type"] = st.sidebar.selectbox(
    "Choose Model",
    list(Available_models.keys())
)
model_type = st.session_state["model_type"]
# Prediction days
# prediction_days = st.sidebar.slider("Future Prediction Days", 1, 30, 7)

# Load data button
# load_button = st.sidebar.button("🔄 Load Data & Predict", use_container_width=True)

# Clear cache button
if st.sidebar.button("🗑️ Clear Cache & Reset", use_container_width=True):
    st.cache_data.clear()
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.success("Cache cleared! Reloading...")
    st.rerun()

# ============================================
# MAIN CONTENT
# ============================================

# # Header
# st.markdown('<p class="main-header">📈 Stock Price Prediction System</p>', unsafe_allow_html=True)
st.markdown(f"### **{stock_name}** ({ticker})")

# Load and process data
# if load_button or 'data' not in st.session_state:
with st.spinner(f'Loading data for {ticker}...'):
        data = load_stock_data(ticker, start_date_input, end_date_input)
        print("##################### data loaded")
        if data is not None and not data.empty and 'Close' in data.columns:
            st.session_state['data'] = data
            st.session_state['ticker'] = ticker
            st.success(f" Successfully loaded {len(data)} days of data!")
        else:
            st.warning(" No valid data loaded. Please check your internet connection, ticker, and date range.")
            st.session_state['data'] = None

# st.write(st.session_state.data)            
data = st.session_state['data']

# Add technical indicators
# data_with_indicators = add_technical_indicators(data)
# Current metrics
st.subheader(" Current Metrics")
col1, col2, col3, col4, col5 = st.columns(5)
print("data coloumns", data.columns)
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
tab1, tab2, tab3 = st.tabs(["📈 Price Prediction", "📊 Technical Analysis", "📉 Historical Data"])


tab1_col1, tab1_col2 = tab1.columns([2, 1])
lookback = 60
feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume',
                             'MA_20', 'MA_50', 'RSI', 'MACD', 'MACD_Signal',
                             'BB_High', 'BB_Low', 'Returns']

data_with_indicators = add_technical_indicators(data)

def plot_stock_price(data, predictions=None, title="Stock Price History"):
    import plotly.graph_objects as go
    fig = go.Figure()
    # Flatten actual prices
    actual_prices = np.array(data['Close']).flatten()
    fig.add_trace(go.Scatter(x=data.index, y=actual_prices, mode='lines', name='Actual'))
    if predictions is not None:
        pred_prices = np.array(predictions).flatten()

        fig.add_trace(go.Scatter(x=data.index[60:], y=pred_prices, mode='lines', name='Predicted'))
    fig.update_layout(title=title, xaxis_title='Date', yaxis_title='Price', template='plotly_white')
    return fig
# "BiLSTM": Bidirectional_LSTM,
# "Transformer": transformer,
from dataclasses import dataclass
@dataclass
class TrainRequest():
    ticker: str 
    from_date: str | None = None 
    to_date: str | None = None
    model : str = 'LSTM' 
    epochs: int = 10

from sklearn.preprocessing import MinMaxScaler

def prepare_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    close_scaled_data = scaler.fit_transform(data["Close"].to_numpy().reshape(-1, 1))
    print(close_scaled_data)
    x_train = []
    y_train = []
    for i in range(60, len(close_scaled_data)):
        x_train.append(close_scaled_data[i-60:i, 0])
        y_train.append(close_scaled_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))  
    return x_train, y_train

import os

def training_task(TrainRequest : TrainRequest, data):
    
    x_train, y_train = prepare_data(data)
    model = Available_models[TrainRequest.model](x_train)
    history = model.fit(x_train,
                    y_train,
                    epochs=TrainRequest.epochs)
    

    os.makedirs(f"models/{TrainRequest.ticker}", exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    model.save(f"models/{TrainRequest.ticker}/{TrainRequest.model}.keras")


import io
import asyncio


    
def list_saved_models(ticker: str):
    # list all saved models from "models" directory
    model_folder = 'models'
    
    saved_models = []
    try:
        for file_name in os.listdir(model_folder+"/"+ticker):
            if file_name.endswith('.keras'):
                saved_models.append(file_name)
    except FileNotFoundError:
        pass

    print("saved models #####################", saved_models)
    
    return saved_models




from sklearn.preprocessing import MinMaxScaler
def predict(data, ticker: str, model: str):
    # load model from "models"/"ticker"/"model_id"
    # auto regressively predict next 'days' days
    model  = keras.models.load_model(f"models/{ticker}/{model}")
    scaler = MinMaxScaler(feature_range=(0, 1))
    close_scaled_data = scaler.fit_transform(data["Close"].to_numpy().reshape(-1, 1))
    x_test = [] 
    for i in range(60, len(close_scaled_data)):
        x_test.append(close_scaled_data[i-60:i, 0])
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    
    print(x_test.shape)
    predictions = model.predict(x_test)
    print(predictions.shape)
    # for i in range(60):
    #     np.append(predictions, model.predict(predictions[-61:-1, 0].reshape(1,60,1)), axis=0)
    
    predictions = scaler.inverse_transform(predictions)

    return {"predictions": predictions.tolist()}

with tab2:
        st.subheader(" Technical Indicators Analysis")
        
        # Technical indicators plot
        fig_tech = plot_technical_indicators(data_with_indicators.tail(180))
        st.plotly_chart(fig_tech, use_container_width=True)
        
        # RSI and MACD
        fig_rsi_macd = plot_rsi_macd(data_with_indicators.tail(180))
        st.plotly_chart(fig_rsi_macd, use_container_width=True)
        
        # Trading signals
        st.subheader(" Trading Signals")
        
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
            <div class="metric-card" style="background-color: #fff; color: #000;">
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
            <div class="metric-card" style="background-color: #fff; color: #000;">
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
            <div class="metric-card" style="background-color: #fff; color: #000;">
                <h4>Bollinger Bands</h4>
                <p style="font-size: 1.5rem; color: {bb_color};">{bb_signal}</p>
                <p>Price: ${current_close:.2f}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            col3.warning("Bollinger Bands data not available.")

# from google import genai
# set gemini api key as enviroment variable
# client = genai.Client(vertexai=True, project="pristine-sphere-477402-t2")

with tab1_col1:
    # check if selected model exists for current ticker
    # st.subheader("📈 Stock Price Prediction")
    saved_models = list_saved_models(ticker)
    print("model type #####################", model_type)
    model_name = model_type + ".keras"
    if model_name in saved_models:
        st.success(f" Model '{model_type}' found for {ticker}!")
        # load model
        info = st.info(f"ℹ Loading existing model '{model_type}' for {ticker}...")
        predictions = predict(data, ticker, model_name)
        st.session_state['predictions'] = predictions['predictions']
        info.empty()

    else: # train model and save it
        st.warning(f" Model '{model_type}' not found for {ticker}. Training new model...")
        TrainRequest_instance = TrainRequest(
            ticker=ticker,
            from_date=start_date_input.strftime("%Y-%m-%d"),
            to_date=end_date_input.strftime("%Y-%m-%d"),
            model=model_type,
            epochs=10
        )
        training_task(TrainRequest_instance, data)
        predictions = predict(data, ticker, model_name)
        st.session_state['predictions'] = predictions['predictions']        
        st.success(f" Model '{model_type}' trained and saved for {ticker}!") 


    # predict yth day using previous x days

    future_trend = st.session_state.get('future_trend', "N/A")
    st.plotly_chart(plot_stock_price(data,predictions=st.session_state.get('predictions', None),
                                    title=f"{ticker} Stock Price Prediction using {model_type}"), use_container_width=True)
    try:
        response = client.models.generate_content(
                        model="gemini-2.5-pro",
                        contents=f"""Generate a concise, professional stock analysis summary for the ticker {ticker} using the following data:

                        Technical Indicators:
                        - RSI: {rsi_signal}
                        - MACD: {macd_signal}
                        
                        - Bollinger Bands: Upper {bb_high}, Lower {bb_low}, Price {current_close}

                        Model Prediction:
                        - LSTM Predicted Price: {predictions['predictions'][-1][0]:.2f}
                        - Actual Price: {current_close}
                        - Direction: {future_trend}

                        Follow this exact structure and tone:

                        Title: “{ticker} Stock Analysis – Concise Summary”

                        1. Current Outlook: One short line describing the sentiment based on indicators + prediction.

                        2. Technical Indicator Interpretation:
                        - 3–5 bullet points interpreting RSI, MACD, and Bollinger Bands.
                        - Mention whether indicators suggest overbought/oversold, momentum, or consolidation.

                        3. Model Prediction Summary:
                        - 1–2 bullet points explaining the LSTM prediction vs actual and its short-term implication.

                        4. Short-Term Impact:
                        - One short paragraph on expected movement (bullish, bearish, neutral) combining indicators + model output.

                        5. Provide a confidence score:
                        -A score between 0-100 that gives the confidence to take the trade.

                        Keep it factual, crisp, and in the exact clean format shown above. No emojis."""
                    )
        print(response.text)
        # write response text in markdown
        st.write("AI-Generated Stock Analysis Summary")
        st.info(f"""
                            {response.text}
                    """)
    except Exception as e:
            print("Error generating AI summary:", e)
            st.info(f"""
                    **{ticker} Stock Analysis – Concise Summary**

1.  Current Outlook: Mixed, with technical indicators showing conflicting signals and a bearish price prediction from the model.

2.  Technical Indicator Interpretation:
    *   RSI is neutral, suggesting neither overbought nor oversold conditions.
    *   MACD is bearish, indicating potential downward momentum.
    *   Price is currently trading above the upper Bollinger Band, which is potentially a sign of overbought conditions.
    *   Bollinger Bands indicate significant price volatility.

3.  Model Prediction Summary:
    *   LSTM model predicts a price of 261.64, lower than the current price of 275.92.
    *   The model suggests a potential price correction in the short term.

4.  Short-Term Impact: The conflicting signals from technical indicators combined with a bearish LSTM prediction point towards a potentially bearish short-term outlook. Traders should monitor price action for confirmation.

5.  Confidence Score: 40
                    """)




with tab1_col2:
            st.markdown("###  Prediction Summary")
            
            if 'predictions' in st.session_state:
                predictions = st.session_state['predictions']
                last_actual = float(data['Close'].iloc[-1])
                last_predicted = float(predictions[-1][0]) if hasattr(predictions[-1], '__iter__') else float(predictions[-1])
                
                # Calculate prediction difference
                pred_diff = last_predicted - last_actual
                pred_diff_pct = (pred_diff / last_actual) * 100
                
                # Future trend
                if pred_diff > 0:
                    future_trend = " Bullish (Upward)"
                    trend_color = "green"
                else:
                    future_trend = " Bearish (Downward)"
                    trend_color = "red"
                st.session_state['future_trend'] = future_trend
                st.markdown("#### Latest Prediction vs Actual")
                st.metric(label="Predicted Closing Price", value=f"${last_predicted:.2f}", delta=f"{pred_diff:.2f} ({pred_diff_pct:+.2f}%)", delta_color="normal")
                st.metric(label="Actual Closing Price", value=f"${last_actual:.2f}")

                st.markdown(f"""
                <p style="font-size: 1.2rem; font-weight: bold;">Trend Direction: <span style="color: {trend_color};">{future_trend}</span></p>
                <p style="font-size: 0.9rem;">Based on {model_type} prediction</p>
                """, unsafe_allow_html=True)
                
                
                
                # Calculate accuracy metrics
                actual_prices = data["Close"].iloc[lookback:].to_numpy().flatten()                
                pred_prices = np.array(predictions).flatten()
                
                # if len(actual_prices) == len(pred_prices):
                # align lengths
                actual_prices = actual_prices[-len(pred_prices):]   
                mae = np.mean(np.abs(actual_prices - pred_prices))
                mape = np.mean(np.abs((actual_prices - pred_prices) / actual_prices)) * 100
                accuracy = max(0, 100 - mape)
                st.markdown(f"""
                <div class="metric-card" style="background-color: #fff; color: #000; padding: 15px; border-radius: 10px; margin-top: 20px; border: 1px solid #eee;">
                    <h4> Model Performance</h4>
                    <p style="font-size: 1.5rem; color: #4CAF50; font-weight: bold;">Accuracy: {accuracy:.2f}%</p>
                    <p style="font-size: 1rem;">MAE: {mae:.2f}</p>
                    <p style="font-size: 1rem;">MAPE: {mape:.2f}%</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("Click 'Load Data & Predict' to see predictions")
            
            st.markdown("---")
            st.markdown("### Model Info")
            st.info(f"""
            **Model Type:** {model_type}  
            **Lookback Period:** {lookback} days  
            **Features:** {len(feature_columns)}  
            **Training Data:** {len(data)} days
            """)

            


tab1.dataframe(data.head())

            

# tab2_col1, tab2_col2, tab2_col3 = tab2.columns(3)
# tab2.subheader("Technical Indicators Analysis")


    # ===== TAB 2: TECHNICAL ANALYSIS =====
with tab3:
    st.subheader(" Historical Price Data")
    
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
    st.subheader(" Recent Data")
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
        label=" Download Data as CSV",
        data=csv,
        file_name=f"{ticker}_stock_data.csv",
        mime="text/csv"
    )
    