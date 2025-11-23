"""
Stock Price Prediction using LSTM - Training Notebook
Complete implementation with technical indicators and sentiment analysis
"""

# ============================================
# PART 1: IMPORTS AND SETUP
# ============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Deep Learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Bidirectional, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Technical Indicators
import ta
from ta.trend import MACD, EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

# Sentiment Analysis
from textblob import TextBlob
import requests

print("TensorFlow Version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))

# ============================================
# PART 2: DATA COLLECTION
# ============================================

class StockDataCollector:
    """Collect and preprocess stock data with technical indicators"""
    
    def __init__(self, ticker, start_date, end_date):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        
    def fetch_data(self):
        """Download stock data from Yahoo Finance"""
        print(f"Fetching data for {self.ticker}...")
        stock = yf.Ticker(self.ticker)
        self.data = stock.history(start=self.start_date, end=self.end_date)
        print(f"Downloaded {len(self.data)} records")
        return self.data
    
    def add_technical_indicators(self):
        """Add technical indicators to the dataset"""
        df = self.data.copy()
        
        # Moving Averages
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        df['MA_50'] = df['Close'].rolling(window=50).mean()
        df['MA_200'] = df['Close'].rolling(window=200).mean()
        
        # RSI
        rsi = RSIIndicator(close=df['Close'], window=14)
        df['RSI'] = rsi.rsi()
        
        # MACD
        macd = MACD(close=df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Diff'] = macd.macd_diff()
        
        # Bollinger Bands
        bollinger = BollingerBands(close=df['Close'])
        df['BB_High'] = bollinger.bollinger_hband()
        df['BB_Low'] = bollinger.bollinger_lband()
        df['BB_Mid'] = bollinger.bollinger_mavg()
        
        # Volume indicators
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        
        # Price changes
        df['Returns'] = df['Close'].pct_change()
        df['Price_Change'] = df['Close'].diff()
        
        # Drop NaN values
        df = df.dropna()
        
        self.data = df
        print(f"Added technical indicators. Shape: {df.shape}")
        return df
    
    def get_sentiment_score(self, symbol):
        """
        Get sentiment score from news (simplified version)
        In production, use NewsAPI or Twitter API
        """
        # Placeholder for sentiment analysis
        # You would fetch real news here and analyze sentiment
        # For now, return neutral sentiment
        return 0.5

# ============================================
# PART 3: DATA PREPROCESSING
# ============================================

class DataPreprocessor:
    """Preprocess data for LSTM training"""
    
    def __init__(self, data, feature_columns, target_column='Close', lookback=60):
        self.data = data
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.lookback = lookback
        self.scaler_features = MinMaxScaler()
        self.scaler_target = MinMaxScaler()
        
    def scale_data(self):
        """Scale features and target"""
        # Scale features
        scaled_features = self.scaler_features.fit_transform(
            self.data[self.feature_columns]
        )
        
        # Scale target
        scaled_target = self.scaler_target.fit_transform(
            self.data[[self.target_column]]
        )
        
        return scaled_features, scaled_target
    
    def create_sequences(self, features, target):
        """Create sequences for LSTM"""
        X, y = [], []
        
        for i in range(self.lookback, len(features)):
            X.append(features[i-self.lookback:i])
            y.append(target[i])
        
        return np.array(X), np.array(y)
    
    def prepare_data(self, train_split=0.8):
        """Complete data preparation pipeline"""
        # Scale data
        scaled_features, scaled_target = self.scale_data()
        
        # Create sequences
        X, y = self.create_sequences(scaled_features, scaled_target)
        
        # Train-test split
        split_idx = int(len(X) * train_split)
        
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"Training set: X={X_train.shape}, y={y_train.shape}")
        print(f"Test set: X={X_test.shape}, y={y_test.shape}")
        
        return X_train, X_test, y_train, y_test

# ============================================
# PART 4: MODEL ARCHITECTURES
# ============================================

class StockLSTMModels:
    """Different LSTM architectures for comparison"""
    
    @staticmethod
    def build_lstm(input_shape, units=[50, 50], dropout=0.2):
        """Standard LSTM model"""
        model = Sequential([
            LSTM(units[0], return_sequences=True, input_shape=input_shape),
            Dropout(dropout),
            LSTM(units[1], return_sequences=False), 
            Dropout(dropout),
            Dense(25, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    @staticmethod
    def build_gru(input_shape, units=[50, 50], dropout=0.2):
        """GRU model for comparison"""
        model = Sequential([
            GRU(units[0], return_sequences=True, input_shape=input_shape),
            Dropout(dropout),
            GRU(units[1], return_sequences=False),
            Dropout(dropout),
            Dense(25, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    @staticmethod
    def build_bidirectional_lstm(input_shape, units=[50, 50], dropout=0.2):
        """Bidirectional LSTM model"""
        model = Sequential([
            Bidirectional(LSTM(units[0], return_sequences=True), input_shape=input_shape),
            Dropout(dropout),
            Bidirectional(LSTM(units[1], return_sequences=False)),
            Dropout(dropout),
            Dense(25, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    @staticmethod
    def build_deep_lstm(input_shape, units=[64, 64, 32], dropout=0.2):
        """Deep LSTM with 3 layers"""
        model = Sequential([
            LSTM(units[0], return_sequences=True, input_shape=input_shape),
            Dropout(dropout),
            LSTM(units[1], return_sequences=True),
            Dropout(dropout),
            LSTM(units[2], return_sequences=False),
            Dropout(dropout),
            Dense(25, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model

# ============================================
# PART 5: TRAINING AND EVALUATION
# ============================================

class ModelTrainer:
    """Train and evaluate models"""
    
    def __init__(self, model, model_name):
        self.model = model
        self.model_name = model_name
        self.history = None
        
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        """Train the model with callbacks"""
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7),
            ModelCheckpoint(f'{self.model_name}_best.h5', save_best_only=True)
        ]
        
        print(f"\nTraining {self.model_name}...")
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def plot_training_history(self):
        """Plot training and validation loss"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['loss'], label='Train Loss')
        plt.plot(self.history.history['val_loss'], label='Val Loss')
        plt.title(f'{self.model_name} - Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['mae'], label='Train MAE')
        plt.plot(self.history.history['val_mae'], label='Val MAE')
        plt.title(f'{self.model_name} - MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{self.model_name}_training_history.png')
        plt.show()

class ModelEvaluator:
    """Evaluate model performance"""
    
    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler
        
    def evaluate(self, X_test, y_test):
        """Calculate evaluation metrics"""
        predictions = self.model.predict(X_test)
        
        # Inverse transform
        y_test_actual = self.scaler.inverse_transform(y_test)
        predictions_actual = self.scaler.inverse_transform(predictions)
        
        # Metrics
        rmse = np.sqrt(mean_squared_error(y_test_actual, predictions_actual))
        mae = mean_absolute_error(y_test_actual, predictions_actual)
        mape = np.mean(np.abs((y_test_actual - predictions_actual) / y_test_actual)) * 100
        
        # Directional accuracy
        y_test_direction = np.diff(y_test_actual.flatten()) > 0
        pred_direction = np.diff(predictions_actual.flatten()) > 0
        directional_accuracy = np.mean(y_test_direction == pred_direction) * 100
        
        metrics = {
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'Directional_Accuracy': directional_accuracy
        }
        
        return metrics, predictions_actual, y_test_actual
    
    def plot_predictions(self, predictions, actual, model_name):
        """Plot predictions vs actual"""
        plt.figure(figsize=(15, 6))
        
        plt.plot(actual, label='Actual Price', linewidth=2)
        plt.plot(predictions, label='Predicted Price', linewidth=2, alpha=0.7)
        plt.title(f'{model_name} - Predictions vs Actual', fontsize=16)
        plt.xlabel('Time Steps', fontsize=12)
        plt.ylabel('Stock Price ($)', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{model_name}_predictions.png')
        plt.show()

# ============================================
# PART 6: MAIN EXECUTION
# ============================================

def main():
    """Main execution pipeline"""
    
    # Configuration
    TICKER = 'AAPL'  # Change to any stock: GOOGL, TSLA, MSFT, etc.
    START_DATE = '2019-01-01'
    END_DATE = '2024-01-01'
    LOOKBACK = 60
    
    # Step 1: Collect Data
    print("="*50)
    print("STEP 1: DATA COLLECTION")
    print("="*50)
    
    collector = StockDataCollector(TICKER, START_DATE, END_DATE)
    data = collector.fetch_data()
    data = collector.add_technical_indicators()
    
    # Display data info
    print("\nData Sample:")
    print(data.tail())
    print(f"\nData Shape: {data.shape}")
    print(f"\nColumns: {data.columns.tolist()}")
    
    # Step 2: Prepare Data
    print("\n" + "="*50)
    print("STEP 2: DATA PREPROCESSING")
    print("="*50)
    
    # Select features
    feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume',
                      'MA_20', 'MA_50', 'RSI', 'MACD', 'MACD_Signal',
                      'BB_High', 'BB_Low', 'Returns']
    
    preprocessor = DataPreprocessor(data, feature_columns, lookback=LOOKBACK)
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(train_split=0.8)
    
    # Step 3: Build and Train Models
    print("\n" + "="*50)
    print("STEP 3: MODEL TRAINING")
    print("="*50)
    
    input_shape = (X_train.shape[1], X_train.shape[2])
    
    models_to_train = {
        'LSTM': StockLSTMModels.build_lstm(input_shape),
        'GRU': StockLSTMModels.build_gru(input_shape),
        'BiLSTM': StockLSTMModels.build_bidirectional_lstm(input_shape)
    }
    
    results = {}
    
    from model_utils import ModelManager
    manager = ModelManager()
    for model_name, model in models_to_train.items():
        print(f"\n{'='*50}")
        print(f"Training {model_name}")
        print(f"{'='*50}")
        # Train
        trainer = ModelTrainer(model, model_name)
        trainer.train(X_train, y_train, X_test, y_test, epochs=50, batch_size=32)
        trainer.plot_training_history()
        # Evaluate
        evaluator = ModelEvaluator(model, preprocessor.scaler_target)
        metrics, predictions, actual = evaluator.evaluate(X_test, y_test)
        evaluator.plot_predictions(predictions, actual, model_name)
        results[model_name] = metrics
        print(f"\n{model_name} Results:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        # Save model and scalers using ModelManager
        manager.save_model(model, model_name, TICKER)
        manager.save_scaler(preprocessor.scaler_features, 'features', TICKER)
        manager.save_scaler(preprocessor.scaler_target, 'target', TICKER)
        # Save config
        config = {
            'ticker': TICKER,
            'model_type': model_name,
            'lookback': LOOKBACK,
            'features': feature_columns,
            'training_date': END_DATE,
            'epochs': 50,
            'batch_size': 32
        }
        manager.save_config(config, TICKER)
    
    # Step 4: Compare Models
    print("\n" + "="*50)
    print("STEP 4: MODEL COMPARISON")
    print("="*50)
    
    comparison_df = pd.DataFrame(results).T
    print("\nComparison Table:")
    print(comparison_df)
    
    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    metrics_list = ['RMSE', 'MAE', 'MAPE', 'Directional_Accuracy']
    for idx, metric in enumerate(metrics_list):
        ax = axes[idx//2, idx%2]
        comparison_df[metric].plot(kind='bar', ax=ax, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax.set_title(f'{metric} Comparison', fontsize=14)
        ax.set_ylabel(metric, fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=0)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.show()
    
    # Save best model
    best_model_name = comparison_df['RMSE'].idxmin()
    print(f"\nBest Model: {best_model_name}")
    print(f"Lowest RMSE: {comparison_df.loc[best_model_name, 'RMSE']:.4f}")
    
    # Save results
    comparison_df.to_csv('model_comparison_results.csv')
    print("\nResults saved to 'model_comparison_results.csv'")

if __name__ == "__main__":
    main()
    print("\n" + "="*50)
    print("TRAINING COMPLETE!")
    print("="*50)