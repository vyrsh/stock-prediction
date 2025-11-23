"""
Model Utilities - Helper functions for saving/loading models and scalers
Use these functions in both training and web app
"""

import os
import pickle
import json
import numpy as np
from tensorflow.keras.models import load_model, save_model
from sklearn.preprocessing import MinMaxScaler

# ============================================
# MODEL MANAGEMENT
# ============================================

class ModelManager:
    """Manage model saving and loading"""
    
    def __init__(self, models_dir='models', scalers_dir='scalers'):
        self.models_dir = models_dir
        self.scalers_dir = scalers_dir
        
        # Create directories if they don't exist
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(scalers_dir, exist_ok=True)
    
    def save_model(self, model, model_name, ticker):
        """Save trained model"""
        filename = f"{ticker}_{model_name}_model.h5"
        filepath = os.path.join(self.models_dir, filename)
        model.save(filepath)
        print(f"✅ Model saved: {filepath}")
        return filepath
    
    def load_model(self, model_name, ticker):
        """Load trained model"""
        filename = f"{ticker}_{model_name}_model.h5"
        filepath = os.path.join(self.models_dir, filename)
        
        if os.path.exists(filepath):
            # Use compile=False to avoid deserialization issues
            model = load_model(filepath, compile=False)
            print(f"✅ Model loaded: {filepath}")
            return model
        else:
            print(f"❌ Model not found: {filepath}")
            return None
    
    def save_scaler(self, scaler, scaler_name, ticker):
        """Save data scaler"""
        filename = f"{ticker}_{scaler_name}_scaler.pkl"
        filepath = os.path.join(self.scalers_dir, filename)
        
        with open(filepath, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"✅ Scaler saved: {filepath}")
        return filepath
    
    def load_scaler(self, scaler_name, ticker):
        """Load data scaler"""
        filename = f"{ticker}_{scaler_name}_scaler.pkl"
        filepath = os.path.join(self.scalers_dir, filename)
        
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                scaler = pickle.load(f)
            print(f"✅ Scaler loaded: {filepath}")
            return scaler
        else:
            print(f"❌ Scaler not found: {filepath}")
            return None
    
    def save_config(self, config, ticker):
        """Save model configuration"""
        filename = f"{ticker}_config.json"
        filepath = os.path.join(self.models_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"✅ Config saved: {filepath}")
    
    def load_config(self, ticker):
        """Load model configuration"""
        filename = f"{ticker}_config.json"
        filepath = os.path.join(self.models_dir, filename)
        
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                config = json.load(f)
            print(f"✅ Config loaded: {filepath}")
            return config
        else:
            print(f"❌ Config not found: {filepath}")
            return None
    
    def list_available_models(self):
        """List all available models"""
        models = [f for f in os.listdir(self.models_dir) if f.endswith('.h5')]
        return models
    
    def get_model_info(self, ticker):
        """Get information about available models for a ticker"""
        config = self.load_config(ticker)
        models = [f for f in self.list_available_models() if f.startswith(ticker)]
        
        return {
            'ticker': ticker,
            'available_models': models,
            'config': config
        }

# ============================================
# PREDICTION UTILITIES
# ============================================

class PredictionEngine:
    """Engine for making predictions with trained models"""
    
    def __init__(self, model, scaler_features, scaler_target, feature_columns, lookback=60):
        self.model = model
        self.scaler_features = scaler_features
        self.scaler_target = scaler_target
        self.feature_columns = feature_columns
        self.lookback = lookback
    
    def prepare_input(self, data):
        """Prepare data for prediction"""
        # Select features
        features = data[self.feature_columns].values
        
        # Scale features
        scaled_features = self.scaler_features.transform(features)
        
        # Create sequences
        if len(scaled_features) < self.lookback:
            raise ValueError(f"Need at least {self.lookback} data points")
        
        X = []
        for i in range(self.lookback, len(scaled_features)):
            X.append(scaled_features[i-self.lookback:i])
        
        return np.array(X)
    
    def predict(self, data):
        """Make predictions on new data"""
        X = self.prepare_input(data)
        predictions_scaled = self.model.predict(X, verbose=0)
        predictions = self.scaler_target.inverse_transform(predictions_scaled)
        return predictions
    
    def predict_next_days(self, data, n_days=7):
        """Predict next N days"""
        predictions = []
        current_data = data.copy()
        
        for _ in range(n_days):
            # Prepare input
            X = self.prepare_input(current_data)
            
            # Predict next day
            pred_scaled = self.model.predict(X[-1:], verbose=0)
            pred = self.scaler_target.inverse_transform(pred_scaled)[0, 0]
            predictions.append(pred)
            
            # Add prediction to data for next iteration
            # (This is simplified - in production, you'd update all features)
            new_row = current_data.iloc[-1].copy()
            new_row['Close'] = pred
            current_data = pd.concat([current_data, new_row.to_frame().T])
        
        return np.array(predictions)

# ============================================
# EVALUATION UTILITIES
# ============================================

class PerformanceEvaluator:
    """Evaluate model performance comprehensively"""
    
    @staticmethod
    def calculate_metrics(y_true, y_pred):
        """Calculate all performance metrics"""
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        # Basic metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        
        # Percentage metrics
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        # R-squared
        r2 = r2_score(y_true, y_pred)
        
        # Directional accuracy
        y_true_direction = np.diff(y_true.flatten()) > 0
        y_pred_direction = np.diff(y_pred.flatten()) > 0
        directional_acc = np.mean(y_true_direction == y_pred_direction) * 100
        
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'R2_Score': r2,
            'Directional_Accuracy': directional_acc
        }
    
    @staticmethod
    def calculate_trading_metrics(y_true, y_pred, initial_capital=10000):
        """Calculate trading strategy metrics"""
        # Simple strategy: buy when prediction is higher, sell when lower
        returns = []
        capital = initial_capital
        position = 0
        
        for i in range(1, len(y_pred)):
            if y_pred[i] > y_true[i-1] and position == 0:
                # Buy signal
                position = capital / y_true[i]
                capital = 0
            elif y_pred[i] < y_true[i-1] and position > 0:
                # Sell signal
                capital = position * y_true[i]
                returns.append((capital - initial_capital) / initial_capital * 100)
                position = 0
        
        # Close any open position
        if position > 0:
            capital = position * y_true[-1]
            returns.append((capital - initial_capital) / initial_capital * 100)
        
        total_return = (capital - initial_capital) / initial_capital * 100 if capital > 0 else 0
        
        return {
            'Total_Return': total_return,
            'Final_Capital': capital if capital > 0 else initial_capital,
            'Number_of_Trades': len(returns)
        }
    
    @staticmethod
    def create_performance_report(y_true, y_pred):
        """Create comprehensive performance report"""
        basic_metrics = PerformanceEvaluator.calculate_metrics(y_true, y_pred)
        trading_metrics = PerformanceEvaluator.calculate_trading_metrics(y_true, y_pred)
        
        report = {
            **basic_metrics,
            **trading_metrics
        }
        
        return report

# ============================================
# DATA VALIDATION
# ============================================

class DataValidator:
    """Validate data quality before training/prediction"""
    
    @staticmethod
    def check_missing_values(data):
        """Check for missing values"""
        missing = data.isnull().sum()
        if missing.sum() > 0:
            print("⚠️ Missing values found:")
            print(missing[missing > 0])
            return False
        print("✅ No missing values")
        return True
    
    @staticmethod
    def check_data_length(data, min_length=100):
        """Check if data has sufficient length"""
        if len(data) < min_length:
            print(f"❌ Insufficient data: {len(data)} rows (need at least {min_length})")
            return False
        print(f"✅ Sufficient data: {len(data)} rows")
        return True
    
    @staticmethod
    def check_price_anomalies(data, threshold=0.5):
        """Check for unusual price movements"""
        returns = data['Close'].pct_change()
        anomalies = returns[abs(returns) > threshold]
        
        if len(anomalies) > 0:
            print(f"⚠️ Found {len(anomalies)} unusual price movements (>{threshold*100}% change)")
            print(anomalies)
            return False
        print("✅ No unusual price movements detected")
        return True
    
    @staticmethod
    def validate_all(data, min_length=100):
        """Run all validation checks"""
        print("\n" + "="*50)
        print("DATA VALIDATION")
        print("="*50)
        
        checks = [
            DataValidator.check_data_length(data, min_length),
            DataValidator.check_missing_values(data),
            DataValidator.check_price_anomalies(data)
        ]
        
        if all(checks):
            print("\n✅ All validation checks passed!")
            return True
        else:
            print("\n❌ Some validation checks failed!")
            return False

# ============================================
# EXAMPLE USAGE
# ============================================

def example_save_and_load():
    """Example: How to save and load models"""
    import pandas as pd
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    from tensorflow.keras.losses import MeanSquaredError
    
    print("\n" + "="*50)
    print("EXAMPLE: SAVE AND LOAD MODEL")
    print("="*50)
    
    # Initialize manager
    manager = ModelManager()
    
    # Create a simple model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(60, 13)),
        LSTM(50, return_sequences=False),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss=MeanSquaredError())
    
    # Create scalers
    scaler_features = MinMaxScaler()
    scaler_target = MinMaxScaler()
    
    # Save model and scalers
    ticker = 'AAPL'
    manager.save_model(model, 'LSTM', ticker)
    manager.save_scaler(scaler_features, 'features', ticker)
    manager.save_scaler(scaler_target, 'target', ticker)
    
    # Save configuration
    config = {
        'ticker': ticker,
        'model_type': 'LSTM',
        'lookback': 60,
        'features': ['Open', 'High', 'Low', 'Close', 'Volume', 
                    'MA_20', 'MA_50', 'RSI', 'MACD', 'MACD_Signal',
                    'BB_High', 'BB_Low', 'Returns'],
        'training_date': '2024-01-01',
        'epochs': 50,
        'batch_size': 32
    }
    manager.save_config(config, ticker)
    
    print("\n--- Loading saved model ---")
    
    # Load model and scalers
    loaded_model = manager.load_model('LSTM', ticker)
    loaded_scaler_features = manager.load_scaler('features', ticker)
    loaded_scaler_target = manager.load_scaler('target', ticker)
    loaded_config = manager.load_config(ticker)
    
    print("\n--- Model Info ---")
    info = manager.get_model_info(ticker)
    print(f"Ticker: {info['ticker']}")
    print(f"Available models: {info['available_models']}")
    print(f"Config: {info['config']}")

def example_prediction():
    """Example: How to make predictions"""
    import pandas as pd
    import yfinance as yf
    
    print("\n" + "="*50)
    print("EXAMPLE: MAKING PREDICTIONS")
    print("="*50)
    
    # Load data
    ticker = 'AAPL'
    data = yf.Ticker(ticker).history(period='1y')
    
    # Add simple moving average as example feature
    data['MA_20'] = data['Close'].rolling(20).mean()
    data['Returns'] = data['Close'].pct_change()
    data = data.dropna()
    
    print(f"\nLoaded {len(data)} days of data for {ticker}")
    
    # Validate data
    validator = DataValidator()
    if validator.validate_all(data):
        print("\n✅ Data is ready for prediction")
    
    # Create prediction engine (assuming model is loaded)
    feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA_20', 'Returns']
    
    print("\n💡 To use prediction engine:")
    print("""
    # Load model and scalers
    manager = ModelManager()
    model = manager.load_model('LSTM', 'AAPL')
    scaler_features = manager.load_scaler('features', 'AAPL')
    scaler_target = manager.load_scaler('target', 'AAPL')
    
    # Create prediction engine
    engine = PredictionEngine(
        model, 
        scaler_features, 
        scaler_target, 
        feature_columns
    )
    
    # Make predictions
    predictions = engine.predict(data)
    
    # Predict next 7 days
    future_predictions = engine.predict_next_days(data, n_days=7)
    """)

def example_evaluation():
    """Example: How to evaluate model performance"""
    print("\n" + "="*50)
    print("EXAMPLE: MODEL EVALUATION")
    print("="*50)
    
    # Simulate predictions
    y_true = np.random.randn(100) * 10 + 150  # Simulated actual prices
    y_pred = y_true + np.random.randn(100) * 5  # Simulated predictions with noise
    
    # Calculate metrics
    evaluator = PerformanceEvaluator()
    report = evaluator.create_performance_report(y_true, y_pred)
    
    print("\nPerformance Report:")
    print("-" * 50)
    for metric, value in report.items():
        print(f"{metric:.<30} {value:.4f}")
    
    print("\n✅ Evaluation complete!")

# ============================================
# BATCH PROCESSING
# ============================================

class BatchProcessor:
    """Process multiple stocks at once"""
    
    def __init__(self, tickers, start_date, end_date):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.results = {}
    
    def download_all_data(self):
        """Download data for all tickers"""
        import yfinance as yf
        
        print(f"\nDownloading data for {len(self.tickers)} stocks...")
        
        for ticker in self.tickers:
            try:
                stock = yf.Ticker(ticker)
                data = stock.history(start=self.start_date, end=self.end_date)
                
                if len(data) > 0:
                    self.results[ticker] = {
                        'data': data,
                        'status': 'success'
                    }
                    print(f"✅ {ticker}: {len(data)} days")
                else:
                    self.results[ticker] = {
                        'data': None,
                        'status': 'no_data'
                    }
                    print(f"❌ {ticker}: No data")
            
            except Exception as e:
                self.results[ticker] = {
                    'data': None,
                    'status': 'error',
                    'error': str(e)
                }
                print(f"❌ {ticker}: Error - {e}")
        
        return self.results
    
    def get_summary(self):
        """Get summary of batch processing"""
        total = len(self.tickers)
        success = sum(1 for r in self.results.values() if r['status'] == 'success')
        failed = total - success
        
        return {
            'total': total,
            'success': success,
            'failed': failed,
            'success_rate': (success / total) * 100
        }

# ============================================
# MAIN DEMONSTRATION
# ============================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print(" " * 15 + "MODEL UTILITIES DEMONSTRATION")
    print("="*70)
    
    # Run examples
    example_save_and_load()
    example_prediction()
    example_evaluation()
    
    print("\n" + "="*70)
    print(" " * 20 + "BATCH PROCESSING DEMO")
    print("="*70)
    
    # Batch processing example
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
    processor = BatchProcessor(tickers, '2023-01-01', '2024-01-01')
    results = processor.download_all_data()
    
    print("\n--- Batch Summary ---")
    summary = processor.get_summary()
    print(f"Total stocks: {summary['total']}")
    print(f"Successful: {summary['success']}")
    print(f"Failed: {summary['failed']}")
    print(f"Success rate: {summary['success_rate']:.1f}%")
    
    print("\n" + "="*70)
    print("UTILITIES DEMONSTRATION COMPLETE!")
    print("="*70)
    print("\n💡 Use these utilities in your training and web app scripts!")
    print("   - ModelManager: Save/load models and scalers")
    print("   - PredictionEngine: Make predictions easily")
    print("   - PerformanceEvaluator: Evaluate model performance")
    print("   - DataValidator: Validate data quality")
    print("   - BatchProcessor: Process multiple stocks")
    print("="*70)