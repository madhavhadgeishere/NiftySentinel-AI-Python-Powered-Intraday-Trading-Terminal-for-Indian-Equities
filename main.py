import sys
import time
import pickle
import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import yfinance as yf
import pandas_ta as ta
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Machine Learning
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# PyQt5
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

# ============= CONFIGURATION =============
INITIAL_CAPITAL = 100000.00  # ₹1 Lakh
BUY_SCORE_THRESHOLD = 3
SELL_SCORE_THRESHOLD = -3
RISK_PERCENTAGE = 0.015  # 1.5% risk per trade
TRAILING_STOP_ATR_MULTIPLIER = 1.8
MODEL_RETRAIN_INTERVAL = 1800  # 30 minutes
DATA_SAVE_FILE = "indian_stock_trading_state.pkl"
MIN_TIME_BETWEEN_TRADES = 180  # 3 minutes
MAX_DAILY_TRADES = 30
UPDATE_INTERVAL = 30  # 30 seconds

# Top Nifty 50 stocks with NSE symbols (yfinance format)
TOP_NIFTY50_STOCKS = {
    'RELIANCE.NS': 'Reliance Industries',
    'TCS.NS': 'Tata Consultancy Services',
    'HDFCBANK.NS': 'HDFC Bank',
    'INFY.NS': 'Infosys',
    'ICICIBANK.NS': 'ICICI Bank',
    'HINDUNILVR.NS': 'Hindustan Unilever',
    'ITC.NS': 'ITC Limited',
    'SBIN.NS': 'State Bank of India',
    'BHARTIARTL.NS': 'Bharti Airtel',
    'KOTAKBANK.NS': 'Kotak Mahindra Bank',
    'LT.NS': 'Larsen & Toubro',
    'HCLTECH.NS': 'HCL Technologies',
    'AXISBANK.NS': 'Axis Bank',
    'BAJFINANCE.NS': 'Bajaj Finance',
    'ASIANPAINT.NS': 'Asian Paints',
    'MARUTI.NS': 'Maruti Suzuki',
    'WIPRO.NS': 'Wipro',
    'TITAN.NS': 'Titan Company',
    'ULTRACEMCO.NS': 'UltraTech Cement',
    'SUNPHARMA.NS': 'Sun Pharma',
    'NESTLEIND.NS': 'Nestle India',
    'TATASTEEL.NS': 'Tata Steel',
    'POWERGRID.NS': 'Power Grid Corporation',
    'NTPC.NS': 'NTPC Limited',
    'M&M.NS': 'Mahindra & Mahindra',
    'TECHM.NS': 'Tech Mahindra',
    'BAJAJFINSV.NS': 'Bajaj Finserv',
    'ONGC.NS': 'Oil & Natural Gas Corp',
    'TATAMOTORS.NS': 'Tata Motors',
    'ADANIPORTS.NS': 'Adani Ports',
}

# Market hours for NSE (IST: 9:15 AM - 3:30 PM)
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 15
MARKET_CLOSE_HOUR = 15
MARKET_CLOSE_MINUTE = 30

# ============= HELPER FUNCTIONS =============

def is_market_open():
    """Check if NSE market is currently open."""
    now = datetime.now()
    current_time = now.time()
    
    # Check if weekend
    if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
        return False
    
    market_open = datetime.strptime(f"{MARKET_OPEN_HOUR}:{MARKET_OPEN_MINUTE}", "%H:%M").time()
    market_close = datetime.strptime(f"{MARKET_CLOSE_HOUR}:{MARKET_CLOSE_MINUTE}", "%H:%M").time()
    
    return market_open <= current_time <= market_close

def get_daily_data_and_pivots(ticker):
    """Fetches daily data with enhanced indicators for Indian stocks."""
    try:
        daily_df = yf.download(ticker, period='200d', interval='1d', progress=False)
        if daily_df.empty or len(daily_df) < 50:
            return {'trend': "Undetermined", 'pivots': None, 'error': "Not enough data."}
        
        if isinstance(daily_df.columns, pd.MultiIndex):
            daily_df.columns = daily_df.columns.droplevel(1)
        
        # Multiple timeframe SMAs
        daily_df.ta.sma(length=20, col_names=('SMA_20d',), append=True)
        daily_df.ta.sma(length=50, col_names=('SMA_50d',), append=True)
        daily_df.ta.sma(length=200, col_names=('SMA_200d',), append=True)
        daily_df.ta.ema(length=12, col_names=('EMA_12d',), append=True)
        daily_df.ta.ema(length=26, col_names=('EMA_26d',), append=True)
        
        latest_daily = daily_df.iloc[-1]
        
        # Enhanced trend detection
        trend_score = 0
        if pd.notna(latest_daily.get('SMA_50d')) and pd.notna(latest_daily.get('SMA_200d')):
            if latest_daily['SMA_50d'] > latest_daily['SMA_200d']:
                trend_score += 2
            else:
                trend_score -= 2
        
        if pd.notna(latest_daily.get('EMA_12d')) and pd.notna(latest_daily.get('EMA_26d')):
            if latest_daily['EMA_12d'] > latest_daily['EMA_26d']:
                trend_score += 1
            else:
                trend_score -= 1
        
        trend = "Strong Uptrend" if trend_score >= 2 else "Uptrend" if trend_score > 0 else "Downtrend" if trend_score < -1 else "Sideways"
        
        # Enhanced Pivot Points (Fibonacci levels)
        prev_day = daily_df.iloc[-2]
        p = (prev_day['High'] + prev_day['Low'] + prev_day['Close']) / 3
        r1 = p + 0.382 * (prev_day['High'] - prev_day['Low'])
        s1 = p - 0.382 * (prev_day['High'] - prev_day['Low'])
        r2 = p + 0.618 * (prev_day['High'] - prev_day['Low'])
        s2 = p - 0.618 * (prev_day['High'] - prev_day['Low'])
        r3 = p + 1.0 * (prev_day['High'] - prev_day['Low'])
        s3 = p - 1.0 * (prev_day['High'] - prev_day['Low'])
        
        pivots = {'pivot': p, 'r1': r1, 's1': s1, 'r2': r2, 's2': s2, 'r3': r3, 's3': s3}
        return {'trend': trend, 'pivots': pivots, 'error': None, 'trend_score': trend_score}
        
    except Exception as e:
        return {'trend': "Error", 'pivots': None, 'error': str(e), 'trend_score': 0}

def get_intraday_data(ticker, period='7d', interval='5m'):
    """Fetches intraday data with better error handling."""
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        
        if not df.empty:
            try:
                if df.index.tz is None:
                    df.index = df.index.tz_localize('UTC')
                # Convert to IST for display
                df.index = df.index.tz_convert('Asia/Kolkata')
            except:
                pass
        return df
    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()

def create_ml_features(df):
    """Creates comprehensive ML features for Indian stock trading."""
    df_copy = df.copy()
    
    # Technical Indicators
    df_copy.ta.rsi(length=14, append=True)
    df_copy.ta.rsi(length=7, append=True)
    df_copy.ta.macd(append=True)
    df_copy.ta.bbands(length=20, append=True)
    df_copy.ta.atr(length=14, append=True)
    df_copy.ta.sma(length=10, append=True)
    df_copy.ta.sma(length=20, append=True)
    df_copy.ta.sma(length=50, append=True)
    df_copy.ta.ema(length=9, append=True)
    df_copy.ta.ema(length=12, append=True)
    df_copy.ta.ema(length=26, append=True)
    df_copy.ta.stoch(append=True)
    df_copy.ta.adx(append=True)
    df_copy.ta.obv(append=True)  # On-Balance Volume
    
    # Price-based features
    df_copy['returns'] = df_copy['Close'].pct_change()
    df_copy['log_returns'] = np.log(df_copy['Close'] / df_copy['Close'].shift(1))
    df_copy['volatility_10'] = df_copy['returns'].rolling(window=10).std()
    df_copy['volatility_20'] = df_copy['returns'].rolling(window=20).std()
    df_copy['momentum_5'] = df_copy['Close'] - df_copy['Close'].shift(5)
    df_copy['momentum_10'] = df_copy['Close'] - df_copy['Close'].shift(10)
    df_copy['price_range'] = df_copy['High'] - df_copy['Low']
    df_copy['price_range_pct'] = df_copy['price_range'] / df_copy['Close']
    
    # Enhanced lagged features
    for lag in [1, 2, 3, 5, 10]:
        df_copy[f'close_lag_{lag}'] = df_copy['Close'].shift(lag)
        df_copy[f'volume_lag_{lag}'] = df_copy['Volume'].shift(lag)
        df_copy[f'rsi_lag_{lag}'] = df_copy['RSI_14'].shift(lag)
        df_copy[f'returns_lag_{lag}'] = df_copy['returns'].shift(lag)
    
    # Time-based features (IST timezone aware)
    df_copy['hour'] = df_copy.index.hour
    df_copy['day_of_week'] = df_copy.index.dayofweek
    df_copy['hour_sin'] = np.sin(2 * np.pi * df_copy['hour'] / 24)
    df_copy['hour_cos'] = np.cos(2 * np.pi * df_copy['hour'] / 24)
    
    # Volume analysis
    df_copy['volume_ma_10'] = df_copy['Volume'].rolling(window=10).mean()
    df_copy['volume_ma_20'] = df_copy['Volume'].rolling(window=20).mean()
    df_copy['volume_ratio_10'] = df_copy['Volume'] / df_copy['volume_ma_10']
    df_copy['volume_ratio_20'] = df_copy['Volume'] / df_copy['volume_ma_20']
    
    # Price position in range
    df_copy['price_position'] = (df_copy['Close'] - df_copy['Low']) / (df_copy['High'] - df_copy['Low'] + 1e-10)
    
    return df_copy

def train_ml_model(df):
    """Trains enhanced ML model for stock prediction."""
    try:
        if len(df) < 100:
            return None, None
        
        df_featured = create_ml_features(df)
        
        # Predict next candle direction
        df_featured['target'] = (df_featured['Close'].shift(-1) > df_featured['Close']).astype(int)
        
        df_featured.dropna(inplace=True)
        
        if len(df_featured) < 50:
            return None, None
        
        feature_cols = [col for col in df_featured.columns if col not in 
                        ['target', 'Open', 'High', 'Low', 'Close', 'Volume'] and 
                        not col.startswith('BBL_') and not col.startswith('BBM_') and 
                        not col.startswith('BBU_')]
        
        X = df_featured[feature_cols].fillna(0)
        y = df_featured['target']
        
        # Gradient Boosting for better performance
        model = make_pipeline(
            StandardScaler(), 
            GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
        )
        model.fit(X, y)
        
        return model, feature_cols
        
    except Exception as e:
        print(f"ML training error: {e}")
        return None, None

def predict_with_model(model, feature_cols, df):
    """Enhanced prediction with confidence."""
    try:
        if model is None or feature_cols is None or len(df) < 2:
            return 0, 0
        
        df_featured = create_ml_features(df)
        df_featured.dropna(inplace=True)
        
        if len(df_featured) == 0:
            return 0, 0
        
        X_latest = df_featured[feature_cols].iloc[-1:].fillna(0)
        prediction = model.predict(X_latest)[0]
        
        try:
            proba = model.predict_proba(X_latest)[0]
            confidence = max(proba) - 0.5
        except:
            confidence = 0
        
        signal = 1 if prediction == 1 else -1
        
        return signal, confidence
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return 0, 0

# ============= WORKER THREAD =============

class AnalysisWorker(QObject):
    """Enhanced worker with better signal quality."""
    analysis_complete = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    status_update = pyqtSignal(str)
    
    def __init__(self, stock_symbol, daily_info):
        super().__init__()
        self.stock_symbol = stock_symbol
        self.daily_info = daily_info
        self.is_running = True
        self.df_historical = pd.DataFrame()
        self.ml_model = None
        self.feature_cols = None
        self.last_model_train = None
        self.mutex = QMutex()
        
    def run(self):
        """Main analysis loop."""
        try:
            self.status_update.emit("Fetching historical data...")
            self.df_historical = get_intraday_data(self.stock_symbol, period='30d', interval='5m')
            
            if self.df_historical.empty:
                self.error_occurred.emit("Failed to fetch data")
                return
            
            self.status_update.emit("Training ML model...")
            self.ml_model, self.feature_cols = train_ml_model(self.df_historical)
            self.last_model_train = time.time()
            
            if self.ml_model:
                self.status_update.emit("ML model ready")
            
            while self.is_running:
                try:
                    # Check if market is open
                    if not is_market_open():
                        self.status_update.emit("Market Closed (NSE: 9:15 AM - 3:30 PM IST)")
                        time.sleep(60)
                        continue
                    
                    df = get_intraday_data(self.stock_symbol, period='7d', interval='5m')
                    
                    if df.empty or len(df) < 50:
                        time.sleep(30)
                        continue
                    
                    self.mutex.lock()
                    self.df_historical = df.copy()
                    self.mutex.unlock()
                    
                    # Retrain model periodically
                    if time.time() - (self.last_model_train or 0) >= MODEL_RETRAIN_INTERVAL:
                        self.status_update.emit("Retraining model...")
                        new_model, new_features = train_ml_model(self.df_historical)
                        if new_model:
                            self.mutex.lock()
                            self.ml_model = new_model
                            self.feature_cols = new_features
                            self.mutex.unlock()
                            self.last_model_train = time.time()
                    
                    results = self._perform_analysis()
                    if results:
                        self.analysis_complete.emit(results)
                    
                    time.sleep(UPDATE_INTERVAL)
                    
                except Exception as e:
                    self.error_occurred.emit(f"Analysis error: {str(e)}")
                    time.sleep(30)
                    
        except Exception as e:
            self.error_occurred.emit(f"Worker error: {str(e)}")

    def _perform_analysis(self):
        """Enhanced analysis with better scoring for Indian stocks."""
        try:
            df = self.df_historical.copy()
            if df.empty or len(df) < 50:
                return {'error': 'Insufficient data'}
            
            # Calculate all indicators
            df.ta.rsi(length=14, append=True)
            df.ta.rsi(length=7, append=True)
            df.ta.macd(append=True)
            df.ta.bbands(length=20, append=True)
            df.ta.atr(length=14, append=True)
            df.ta.sma(length=10, append=True)
            df.ta.sma(length=20, append=True)
            df.ta.sma(length=50, append=True)
            df.ta.ema(length=9, append=True)
            df.ta.ema(length=12, append=True)
            df.ta.stoch(append=True)
            df.ta.adx(append=True)
            df.ta.obv(append=True)
            
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            last_price = latest['Close']
            
            # Get indicator values
            rsi_14 = latest.get('RSI_14', 50)
            rsi_7 = latest.get('RSI_7', 50)
            macd = latest.get('MACD_12_26_9', 0)
            macd_signal = latest.get('MACDs_12_26_9', 0)
            macd_hist = latest.get('MACDh_12_26_9', 0)
            atr_value = latest.get('ATRr_14', 0)
            bb_upper = latest.get('BBU_20_2.0', last_price * 1.02)
            bb_lower = latest.get('BBL_20_2.0', last_price * 0.98)
            bb_middle = latest.get('BBM_20_2.0', last_price)
            stoch_k = latest.get('STOCHk_14_3_3', 50)
            stoch_d = latest.get('STOCHd_14_3_3', 50)
            adx = latest.get('ADX_14', 25)
            obv = latest.get('OBV', 0)
            
            # Initialize score
            score = 0
            signals = []
            
            # 1. RSI Signals
            if pd.notna(rsi_14):
                if rsi_14 < 25:
                    score += 2
                    signals.append("RSI14: Oversold")
                elif rsi_14 < 35:
                    score += 1
                    signals.append("RSI14: Slightly Oversold")
                elif rsi_14 > 75:
                    score -= 2
                    signals.append("RSI14: Overbought")
                elif rsi_14 > 65:
                    score -= 1
                    signals.append("RSI14: Slightly Overbought")
            
            if pd.notna(rsi_7):
                if rsi_7 < 30 and rsi_14 < 40:
                    score += 1
                    signals.append("RSI7: Confirming Oversold")
                elif rsi_7 > 70 and rsi_14 > 60:
                    score -= 1
                    signals.append("RSI7: Confirming Overbought")
            
            # 2. MACD Signals
            if pd.notna(macd) and pd.notna(macd_signal) and pd.notna(macd_hist):
                prev_macd_hist = prev.get('MACDh_12_26_9', 0)
                
                if macd > macd_signal:
                    if macd_hist > prev_macd_hist:
                        score += 2
                        signals.append("MACD: Strong Bullish")
                    else:
                        score += 1
                        signals.append("MACD: Bullish")
                elif macd < macd_signal:
                    if macd_hist < prev_macd_hist:
                        score -= 2
                        signals.append("MACD: Strong Bearish")
                    else:
                        score -= 1
                        signals.append("MACD: Bearish")
            
            # 3. Moving Averages
            sma_10 = latest.get('SMA_10', None)
            sma_20 = latest.get('SMA_20', None)
            sma_50 = latest.get('SMA_50', None)
            ema_9 = latest.get('EMA_9', None)
            ema_12 = latest.get('EMA_12', None)
            
            ma_score = 0
            if pd.notna(ema_9) and last_price > ema_9:
                ma_score += 1
            if pd.notna(ema_12) and last_price > ema_12:
                ma_score += 1
            if pd.notna(sma_20) and last_price > sma_20:
                ma_score += 1
            if pd.notna(sma_50) and last_price > sma_50:
                ma_score += 1
            
            if ma_score >= 3:
                score += 2
                signals.append(f"MAs: Strong Bullish ({ma_score}/4)")
            elif ma_score >= 2:
                score += 1
                signals.append(f"MAs: Bullish ({ma_score}/4)")
            elif ma_score <= 1:
                score -= 1
                signals.append(f"MAs: Bearish ({ma_score}/4)")
            
            # 4. Bollinger Bands
            bb_position = (last_price - bb_lower) / (bb_upper - bb_lower + 1e-10)
            
            if last_price < bb_lower:
                score += 2
                signals.append("BB: Below Lower Band")
            elif bb_position < 0.2:
                score += 1
                signals.append("BB: Near Lower Band")
            elif last_price > bb_upper:
                score -= 2
                signals.append("BB: Above Upper Band")
            elif bb_position > 0.8:
                score -= 1
                signals.append("BB: Near Upper Band")
            
            # 5. Stochastic
            if pd.notna(stoch_k) and pd.notna(stoch_d):
                if stoch_k < 20 and stoch_d < 20:
                    score += 1
                    signals.append("Stoch: Oversold")
                elif stoch_k > 80 and stoch_d > 80:
                    score -= 1
                    signals.append("Stoch: Overbought")
                
                prev_stoch_k = prev.get('STOCHk_14_3_3', 50)
                prev_stoch_d = prev.get('STOCHd_14_3_3', 50)
                if prev_stoch_k < prev_stoch_d and stoch_k > stoch_d:
                    score += 1
                    signals.append("Stoch: Bullish Cross")
                elif prev_stoch_k > prev_stoch_d and stoch_k < stoch_d:
                    score -= 1
                    signals.append("Stoch: Bearish Cross")
            
            # 6. ADX (Trend Strength)
            trend_strength_multiplier = 1.0
            if pd.notna(adx):
                if adx > 30:
                    trend_strength_multiplier = 1.3
                    signals.append(f"ADX: Strong Trend ({adx:.1f})")
                elif adx > 25:
                    trend_strength_multiplier = 1.1
                elif adx < 20:
                    trend_strength_multiplier = 0.8
                    signals.append(f"ADX: Weak Trend ({adx:.1f})")
            
            # 7. Volume Analysis (OBV)
            if pd.notna(obv) and len(df) > 10:
                obv_ma = df['OBV'].rolling(window=10).mean().iloc[-1]
                if obv > obv_ma:
                    score += 1
                    signals.append("Volume: Bullish (OBV rising)")
                elif obv < obv_ma:
                    score -= 1
                    signals.append("Volume: Bearish (OBV falling)")
            
            # 8. Daily Trend Alignment
            if self.daily_info:
                trend_score = self.daily_info.get('trend_score', 0)
                if trend_score >= 2:
                    score += 2
                    signals.append("Daily: Strong Uptrend")
                elif trend_score > 0:
                    score += 1
                    signals.append("Daily: Uptrend")
                elif trend_score <= -2:
                    score -= 2
                    signals.append("Daily: Strong Downtrend")
                elif trend_score < 0:
                    score -= 1
                    signals.append("Daily: Downtrend")
                
                pivots = self.daily_info.get('pivots')
                if pivots:
                    if last_price > pivots['r1']:
                        score += 1
                        signals.append("Price: Above R1")
                    elif last_price < pivots['s1']:
                        score -= 1
                        signals.append("Price: Below S1")
            
            # 9. ML Model Prediction
            ml_signal, ml_confidence = 0, 0
            if self.ml_model and self.feature_cols:
                ml_signal, ml_confidence = predict_with_model(self.ml_model, self.feature_cols, df)
                if ml_confidence > 0.1:
                    score += ml_signal * 2
                    signals.append(f"ML: {'Bullish' if ml_signal > 0 else 'Bearish'} ({ml_confidence:.2f})")
            
            # Apply trend strength multiplier
            score = int(score * trend_strength_multiplier)
            
            return {
                'dataframe': df,
                'last_price': last_price,
                'rsi': rsi_14 if pd.notna(rsi_14) else 50,
                'rsi_7': rsi_7 if pd.notna(rsi_7) else 50,
                'atr_value': atr_value if pd.notna(atr_value) else 0,
                'score': score,
                'ml_signal': ml_signal,
                'ml_confidence': ml_confidence,
                'adx': adx if pd.notna(adx) else 25,
                'obv': obv if pd.notna(obv) else 0,
                'signals': signals,
                'error': None
            }
            
        except Exception as e:
            return {'error': f'Analysis error: {str(e)}'}

    def stop(self):
        self.is_running = False

# ============= MAIN APPLICATION =============

class IndianStockTradingTerminal(QMainWindow):
    def __init__(self):
        super().__init__()
        self.selected_stock = None
        self.portfolio = self._create_initial_portfolio()
        self.daily_info = None
        self.last_intraday_data = pd.DataFrame()
        self.worker = None
        self.worker_thread = None
        self.enable_trailing_stop = True
        self.dark_mode = True
        self.last_signals = []
        
        self._load_state()
        self.setWindowTitle("🇮🇳 Indian Stock Auto Trading Terminal - Paper Trading")
        self.setGeometry(50, 50, 1900, 1000)
        self.init_ui()
        self.apply_theme()

    def _create_initial_portfolio(self):
        return {
            'capital': INITIAL_CAPITAL,
            'position_open': False,
            'position_type': None,
            'trade_log': [],
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'last_trade_time': None,
            'daily_trades': 0,
            'daily_reset_date': datetime.now().date()
        }

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Top toolbar
        toolbar = QWidget()
        toolbar.setObjectName("toolbar")
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(15, 10, 15, 10)
        
        title = QLabel("🇮🇳 INDIAN STOCK AUTO TRADER - Paper Trading")
        title.setObjectName("app_title")
        toolbar_layout.addWidget(title)
        
        toolbar_layout.addStretch()
        
        # Market status indicator
        self.market_status_label = QLabel()
        self.market_status_label.setObjectName("market_status")
        self.update_market_status()
        toolbar_layout.addWidget(self.market_status_label)
        
        # Theme toggle
        self.theme_btn = QPushButton("☀️ Light Mode")
        self.theme_btn.setObjectName("theme_btn")
        self.theme_btn.clicked.connect(self.toggle_theme)
        toolbar_layout.addWidget(self.theme_btn)
        
        main_layout.addWidget(toolbar)
        
        # Main content area
        content_splitter = QSplitter(Qt.Horizontal)
        content_splitter.setHandleWidth(3)
        
        # ===== LEFT PANEL =====
        left_panel = QWidget()
        left_panel.setObjectName("panel")
        left_panel.setMinimumWidth(320)
        left_panel.setMaximumWidth(400)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(15, 15, 15, 15)
        left_layout.setSpacing(10)
        
        # Stock selection
        left_layout.addWidget(self._create_section_header("📊 Select Stock (Nifty 50)"))
        
        self.stock_input = QLineEdit()
        self.stock_input.setPlaceholderText("Enter symbol (e.g., RELIANCE.NS)")
        self.stock_input.returnPressed.connect(self.start_trading)
        left_layout.addWidget(self.stock_input)
        
        self.stock_dropdown = QComboBox()
        self.stock_dropdown.addItem("-- Top Nifty 50 Stocks --")
        for symbol, name in TOP_NIFTY50_STOCKS.items():
            self.stock_dropdown.addItem(f"{symbol} - {name}", symbol)
        self.stock_dropdown.currentIndexChanged.connect(self.on_dropdown_select)
        left_layout.addWidget(self.stock_dropdown)
        
        # Active trading info
        self.active_stock_label = QLabel("None")
        self.active_stock_label.setObjectName("active_stock")
        self.active_stock_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.active_stock_label)
        
        left_layout.addSpacing(10)
        
        # Strategy options
        left_layout.addWidget(self._create_section_header("⚙️ Strategy Options"))
        
        options_frame = QFrame()
        options_layout = QVBoxLayout(options_frame)
        options_layout.setContentsMargins(10, 5, 10, 5)
        
        self.trailing_checkbox = QCheckBox("Enable Trailing Stop-Loss")
        self.trailing_checkbox.setChecked(True)
        self.trailing_checkbox.stateChanged.connect(lambda s: setattr(self, 'enable_trailing_stop', s == Qt.Checked))
        options_layout.addWidget(self.trailing_checkbox)
        
        # Risk info label
        risk_label = QLabel(f"Risk per trade: {RISK_PERCENTAGE*100:.1f}%")
        risk_label.setStyleSheet("font-size: 11px; color: #888;")
        options_layout.addWidget(risk_label)
        
        left_layout.addWidget(options_frame)
        left_layout.addSpacing(10)
        
        # Control buttons
        self.start_button = QPushButton("▶ START AUTO TRADING")
        self.start_button.setObjectName("start_btn")
        self.start_button.clicked.connect(self.start_trading)
        left_layout.addWidget(self.start_button)
        
        self.stop_button = QPushButton("⏹ STOP TRADING")
        self.stop_button.setObjectName("stop_btn")
        self.stop_button.clicked.connect(self.stop_trading)
        self.stop_button.setEnabled(False)
        left_layout.addWidget(self.stop_button)
        
        self.save_button = QPushButton("💾 SAVE STATE")
        self.save_button.setObjectName("save_btn")
        self.save_button.clicked.connect(self._save_state)
        left_layout.addWidget(self.save_button)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(False)
        left_layout.addWidget(self.progress_bar)
        
        left_layout.addSpacing(15)
        
        # Current signals
        left_layout.addWidget(self._create_section_header("🎯 Active Signals"))
        
        self.signals_list = QListWidget()
        self.signals_list.setMaximumHeight(150)
        left_layout.addWidget(self.signals_list)
        
        left_layout.addSpacing(10)
        
        # Status
        left_layout.addWidget(self._create_section_header("📡 Status"))
        self.status_label = QLabel("Ready to trade")
        self.status_label.setWordWrap(True)
        self.status_label.setObjectName("status_label")
        left_layout.addWidget(self.status_label)
        
        left_layout.addStretch()
        
        # ===== CENTER PANEL (Chart) =====
        center_panel = QWidget()
        center_layout = QVBoxLayout(center_panel)
        center_layout.setContentsMargins(0, 0, 0, 0)
        
        self.fig = Figure(figsize=(12, 8), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)
        center_layout.addWidget(self.canvas)
        
        # ===== RIGHT PANEL =====
        right_panel = QWidget()
        right_panel.setObjectName("panel")
        right_panel.setMinimumWidth(350)
        right_panel.setMaximumWidth(450)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(15, 15, 15, 15)
        right_layout.setSpacing(10)
        
        # Portfolio metrics
        right_layout.addWidget(self._create_section_header("💰 Portfolio Metrics (Paper)"))
        
        metrics_widget = QWidget()
        metrics_layout = QGridLayout(metrics_widget)
        metrics_layout.setSpacing(8)
        
        self.capital_val = self._add_metric(metrics_layout, "Capital:", f"₹{INITIAL_CAPITAL:,.2f}", 0)
        self.pnl_total_val = self._add_metric(metrics_layout, "Total P&L:", "₹0.00", 1)
        self.pnl_open_val = self._add_metric(metrics_layout, "Open P&L:", "₹0.00", 2)
        self.win_rate_val = self._add_metric(metrics_layout, "Win Rate:", "0%", 3)
        self.daily_trades_val = self._add_metric(metrics_layout, "Daily Trades:", "0", 4)
        self.position_val = self._add_metric(metrics_layout, "Position:", "None", 5)
        
        right_layout.addWidget(metrics_widget)
        right_layout.addSpacing(10)
        
        # Market indicators
        right_layout.addWidget(self._create_section_header("📈 Market Indicators"))
        
        indicators_widget = QWidget()
        indicators_layout = QGridLayout(indicators_widget)
        indicators_layout.setSpacing(8)
        
        self.daily_trend_val = self._add_metric(indicators_layout, "Daily Trend:", "N/A", 0)
        self.ai_signal_val = self._add_metric(indicators_layout, "AI Score:", "N/A", 1)
        self.rsi_val = self._add_metric(indicators_layout, "RSI (14):", "N/A", 2)
        self.adx_val = self._add_metric(indicators_layout, "ADX:", "N/A", 3)
        
        right_layout.addWidget(indicators_widget)
        right_layout.addSpacing(15)
        
        # Trade log
        right_layout.addWidget(self._create_section_header("📝 Trade Log"))
        
        self.trade_table = QTableWidget(0, 5)
        self.trade_table.setHorizontalHeaderLabels(["Time", "Type", "Price", "P&L", "Balance"])
        self.trade_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.trade_table.setAlternatingRowColors(True)
        self.trade_table.setEditTriggers(QTableWidget.NoEditTriggers)
        right_layout.addWidget(self.trade_table)
        
        # Add panels to splitter
        content_splitter.addWidget(left_panel)
        content_splitter.addWidget(center_panel)
        content_splitter.addWidget(right_panel)
        content_splitter.setSizes([350, 1000, 400])
        
        main_layout.addWidget(content_splitter)
        
        self.update_metrics_display()
        
        # Timer for market status updates
        self.market_timer = QTimer()
        self.market_timer.timeout.connect(self.update_market_status)
        self.market_timer.start(60000)  # Update every minute

    def update_market_status(self):
        """Update market open/closed status."""
        if is_market_open():
            self.market_status_label.setText("🟢 Market OPEN")
            self.market_status_label.setStyleSheet("color: #00ff88; font-weight: bold;")
        else:
            self.market_status_label.setText("🔴 Market CLOSED")
            self.market_status_label.setStyleSheet("color: #ff4444; font-weight: bold;")

    def _create_section_header(self, text):
        label = QLabel(text)
        label.setObjectName("section_header")
        return label
    
    def _add_metric(self, layout, name, value, row):
        name_label = QLabel(name)
        name_label.setObjectName("metric_name")
        value_label = QLabel(value)
        value_label.setObjectName("metric_value")
        layout.addWidget(name_label, row, 0)
        layout.addWidget(value_label, row, 1)
        return value_label

    def apply_theme(self):
        """Apply light or dark theme."""
        if self.dark_mode:
            self.theme_btn.setText("☀️ Light Mode")
            style = """
                QMainWindow, QWidget { background-color: #1a1a1a; color: #e0e0e0; }
                #toolbar { background-color: #2a2a2a; border-bottom: 2px solid #ff9933; }
                #app_title { color: #ff9933; font-weight: bold; font-size: 18px; }
                #market_status { font-size: 12px; margin-right: 15px; }
                #panel { background-color: #222; border-radius: 8px; }
                #section_header { 
                    color: #ff9933; font-weight: bold; font-size: 14px; 
                    padding: 8px 0px; border-bottom: 2px solid #333;
                }
                #active_stock { 
                    color: #00ff88; font-size: 22px; font-weight: bold; 
                    background-color: #2a2a2a; padding: 15px; border-radius: 8px;
                }
                #metric_name { color: #aaa; font-size: 12px; }
                #metric_value { color: #e0e0e0; font-size: 13px; font-weight: bold; }
                #status_label { 
                    background-color: #2a2a2a; padding: 10px; 
                    border-radius: 6px; font-size: 11px; color: #ccc;
                }
                QPushButton {
                    background-color: #ff9933; color: white; padding: 12px;
                    border-radius: 6px; font-weight: bold; font-size: 13px; border: none;
                }
                QPushButton:hover { background-color: #cc7729; }
                QPushButton:disabled { background-color: #444; color: #888; }
                #start_btn { background-color: #00aa00; }
                #start_btn:hover { background-color: #008800; }
                #stop_btn { background-color: #cc0000; }
                #stop_btn:hover { background-color: #aa0000; }
                #save_btn { background-color: #0078d4; }
                #theme_btn { background-color: #444; padding: 8px 15px; }
                QLineEdit, QComboBox {
                    background-color: #2a2a2a; color: #e0e0e0; border: 1px solid #444;
                    padding: 8px; border-radius: 4px; font-size: 12px;
                }
                QLineEdit:focus, QComboBox:focus { border: 1px solid #ff9933; }
                QListWidget, QTableWidget {
                    background-color: #2a2a2a; color: #e0e0e0; border: 1px solid #333;
                    alternate-background-color: #252525; gridline-color: #333;
                }
                QHeaderView::section {
                    background-color: #333; color: #e0e0e0; padding: 8px;
                    border: none; font-weight: bold; font-size: 11px;
                }
                QProgressBar {
                    border: 1px solid #444; border-radius: 4px; text-align: center;
                    background-color: #2a2a2a; color: #e0e0e0;
                }
                QProgressBar::chunk { background-color: #ff9933; }
                QCheckBox { spacing: 8px; font-size: 12px; color: #e0e0e0; }
                QCheckBox::indicator { width: 18px; height: 18px; }
                QSplitter::handle { background-color: #333; }
            """
        else:
            self.theme_btn.setText("🌙 Dark Mode")
            style = """
                QMainWindow, QWidget { background-color: #f5f5f5; color: #333; }
                #toolbar { background-color: #fff; border-bottom: 2px solid #ff9933; }
                #app_title { color: #ff9933; font-weight: bold; font-size: 18px; }
                #market_status { font-size: 12px; margin-right: 15px; }
                #panel { background-color: #fff; border: 1px solid #ddd; border-radius: 8px; }
                #section_header { 
                    color: #ff9933; font-weight: bold; font-size: 14px; 
                    padding: 8px 0px; border-bottom: 2px solid #e0e0e0;
                }
                #active_stock { 
                    color: #00aa00; font-size: 22px; font-weight: bold; 
                    background-color: #f0f0f0; padding: 15px; border-radius: 8px;
                }
                #metric_name { color: #666; font-size: 12px; }
                #metric_value { color: #333; font-size: 13px; font-weight: bold; }
                #status_label { 
                    background-color: #f9f9f9; padding: 10px; 
                    border-radius: 6px; font-size: 11px; color: #555; border: 1px solid #ddd;
                }
                QPushButton {
                    background-color: #ff9933; color: white; padding: 12px;
                    border-radius: 6px; font-weight: bold; font-size: 13px; border: none;
                }
                QPushButton:hover { background-color: #cc7729; }
                QPushButton:disabled { background-color: #ccc; color: #999; }
                #start_btn { background-color: #00aa00; }
                #start_btn:hover { background-color: #008800; }
                #stop_btn { background-color: #cc0000; }
                #stop_btn:hover { background-color: #aa0000; }
                #save_btn { background-color: #0078d4; }
                #theme_btn { background-color: #ddd; color: #333; padding: 8px 15px; }
                #theme_btn:hover { background-color: #ccc; }
                QLineEdit, QComboBox {
                    background-color: #fff; color: #333; border: 1px solid #ccc;
                    padding: 8px; border-radius: 4px; font-size: 12px;
                }
                QLineEdit:focus, QComboBox:focus { border: 1px solid #ff9933; }
                QListWidget, QTableWidget {
                    background-color: #fff; color: #333; border: 1px solid #ddd;
                    alternate-background-color: #f9f9f9; gridline-color: #e0e0e0;
                }
                QHeaderView::section {
                    background-color: #f0f0f0; color: #333; padding: 8px;
                    border: none; font-weight: bold; font-size: 11px;
                }
                QProgressBar {
                    border: 1px solid #ccc; border-radius: 4px; text-align: center;
                    background-color: #fff; color: #333;
                }
                QProgressBar::chunk { background-color: #ff9933; }
                QCheckBox { spacing: 8px; font-size: 12px; color: #333; }
                QCheckBox::indicator { width: 18px; height: 18px; }
                QSplitter::handle { background-color: #ddd; }
            """
        
        self.setStyleSheet(style)
        self.setup_chart_style()
        if not self.last_intraday_data.empty:
            self.update_chart()

    def toggle_theme(self):
        """Toggle between light and dark mode."""
        self.dark_mode = not self.dark_mode
        self.apply_theme()

    def setup_chart_style(self):
        """Setup chart appearance based on theme."""
        if self.dark_mode:
            bg_color = '#1a1a1a'
            plot_bg = '#2a2a2a'
            text_color = '#e0e0e0'
            grid_color = '#3a3a3a'
        else:
            bg_color = '#f5f5f5'
            plot_bg = '#ffffff'
            text_color = '#333333'
            grid_color = '#e0e0e0'
        
        self.fig.set_facecolor(bg_color)
        self.ax.set_facecolor(plot_bg)
        self.ax.tick_params(colors=text_color)
        for spine in self.ax.spines.values():
            spine.set_color(grid_color)
        self.ax.xaxis.label.set_color(text_color)
        self.ax.yaxis.label.set_color(text_color)
        self.ax.title.set_color(text_color)
        self.ax.grid(True, color=grid_color, linestyle='--', alpha=0.3)

    def on_dropdown_select(self, index):
        if index > 0:
            symbol = self.stock_dropdown.itemData(index)
            self.stock_input.setText(symbol)

    def start_trading(self):
        """Start auto trading."""
        stock = self.stock_input.text().strip().upper()
        
        if not stock:
            QMessageBox.warning(self, "Input Error", "Please select a stock")
            return
        
        # Validate NSE format
        if not stock.endswith('.NS'):
            stock += ".NS"
            self.stock_input.setText(stock)
        
        if self.worker_thread and self.worker_thread.isRunning():
            self.stop_trading()
        
        # Reset daily trades if new day
        if self.portfolio.get('daily_reset_date') != datetime.now().date():
            self.portfolio['daily_trades'] = 0
            self.portfolio['daily_reset_date'] = datetime.now().date()
        
        is_continuing = (self.selected_stock == stock and self.portfolio.get('total_trades', 0) > 0)
        
        if not is_continuing:
            self.selected_stock = stock
            self.portfolio = self._create_initial_portfolio()
            self.trade_table.setRowCount(0)
            self.signals_list.clear()
        else:
            self.selected_stock = stock
        
        # Get stock name
        stock_name = TOP_NIFTY50_STOCKS.get(stock, stock)
        
        self.status_label.setText(f"Initializing {stock_name}...")
        self.active_stock_label.setText(f"{stock}\n{stock_name}")
        self.progress_bar.setVisible(True)
        
        self.daily_info = get_daily_data_and_pivots(self.selected_stock)
        
        if self.daily_info and not self.daily_info.get('error'):
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.stock_input.setEnabled(False)
            self.stock_dropdown.setEnabled(False)
            
            if is_continuing:
                self.status_label.setText(f"Continuing {stock_name} - Capital: ₹{self.portfolio['capital']:,.2f}")
            
            self._start_worker_thread()
        else:
            error_msg = self.daily_info.get('error', 'Unknown error')
            self.status_label.setText(f"Failed: {error_msg}")
            QMessageBox.critical(self, "Data Error", f"Could not fetch data for {stock}.\n\n{error_msg}")
            self.selected_stock = None
            self.active_stock_label.setText("None")
            self.progress_bar.setVisible(False)

    def _start_worker_thread(self):
        """Initialize worker thread."""
        self.worker = AnalysisWorker(self.selected_stock, self.daily_info)
        self.worker_thread = QThread()
        
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.analysis_complete.connect(self.on_analysis_complete)
        self.worker.error_occurred.connect(self.on_worker_error)
        self.worker.status_update.connect(self.on_status_update)
        
        self.worker_thread.start()

    def stop_trading(self):
        """Stop auto trading."""
        if self.worker:
            self.worker.stop()
        
        if self.worker_thread:
            self.worker_thread.quit()
            self.worker_thread.wait()
        
        if self.portfolio.get('position_open') and not self.last_intraday_data.empty:
            self._liquidate_position("Manual Stop")
        
        final_pnl = self.portfolio['capital'] - INITIAL_CAPITAL
        self.status_label.setText(f"Stopped. Final P&L: ₹{final_pnl:,.2f}")
        
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.stock_input.setEnabled(True)
        self.stock_dropdown.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        win_rate = (self.portfolio['winning_trades'] / self.portfolio['total_trades'] * 100) if self.portfolio['total_trades'] > 0 else 0
        
        QMessageBox.information(
            self, "Trading Stopped",
            f"Paper trading for {self.selected_stock} stopped.\n\n"
            f"Initial Capital: ₹{INITIAL_CAPITAL:,.2f}\n"
            f"Final Capital: ₹{self.portfolio['capital']:,.2f}\n"
            f"NET P&L: {'+' if final_pnl >= 0 else ''}₹{final_pnl:,.2f}\n"
            f"Total Trades: {self.portfolio['total_trades']}\n"
            f"Win Rate: {win_rate:.1f}%"
        )

    def on_analysis_complete(self, results):
        """Handle analysis results."""
        if results.get('error'):
            self.status_label.setText(f"Error: {results['error']}")
            return
        
        self.last_intraday_data = results['dataframe']
        self.last_signals = results.get('signals', [])
        
        # Update signals list
        self.signals_list.clear()
        for signal in self.last_signals[:8]:
            self.signals_list.addItem(signal)
        
        self._execute_trading_logic(results)
        self.update_chart()
        self.update_metrics_display()
    
    def on_worker_error(self, error_msg):
        self.status_label.setText(f"⚠️ {error_msg}")
    
    def on_status_update(self, status_msg):
        self.status_label.setText(status_msg)

    def _execute_trading_logic(self, results):
        """Enhanced trading logic for Indian stocks."""
        last_price = results['last_price']
        atr_value = results['atr_value']
        score = results['score']
        rsi = results['rsi']
        adx = results.get('adx', 25)
        
        # Update indicators display
        if score >= BUY_SCORE_THRESHOLD:
            signal_text = "BUY"
            signal_color = "#00ff88" if self.dark_mode else "#00aa00"
        elif score <= SELL_SCORE_THRESHOLD:
            signal_text = "SELL"
            signal_color = "#ff4444" if self.dark_mode else "#cc0000"
        else:
            signal_text = "NEUTRAL"
            signal_color = "#e0e0e0" if self.dark_mode else "#666"
        
        self.ai_signal_val.setText(f"{score} ({signal_text})")
        self.ai_signal_val.setStyleSheet(f"color: {signal_color}; font-weight: bold;")
        
        self.rsi_val.setText(f"{rsi:.1f}")
        self.rsi_val.setStyleSheet(
            f"color: {'#ff4444' if self.dark_mode else '#cc0000'};" if rsi > 70 else
            f"color: {'#00ff88' if self.dark_mode else '#00aa00'};" if rsi < 30 else
            f"color: {'#e0e0e0' if self.dark_mode else '#666'};"
        )
        
        self.adx_val.setText(f"{adx:.1f}")
        self.adx_val.setStyleSheet(
            f"color: {'#00ff88' if self.dark_mode else '#00aa00'};" if adx > 25 else
            f"color: {'#ffaa00' if self.dark_mode else '#ff8800'};"
        )
        
        # Entry logic
        if not self.portfolio['position_open']:
            # Check daily trade limit
            if self.portfolio['daily_trades'] >= MAX_DAILY_TRADES:
                self.status_label.setText(f"Daily trade limit reached ({MAX_DAILY_TRADES})")
                return
            
            # Check cooldown
            last_trade_time = self.portfolio.get('last_trade_time')
            if last_trade_time:
                time_since = (datetime.now() - last_trade_time).total_seconds()
                if time_since < MIN_TIME_BETWEEN_TRADES:
                    remaining = int(MIN_TIME_BETWEEN_TRADES - time_since)
                    self.status_label.setText(f"Cooldown: {remaining}s (Score: {score})")
                    return
            
            # Long entry (only long positions for Indian stocks - no shorting by default)
            if score >= BUY_SCORE_THRESHOLD and pd.notna(atr_value) and atr_value > 0:
                self._enter_long_position(last_price, atr_value)
            else:
                self.status_label.setText(f"Waiting... Score: {score}, Signals: {len(self.last_signals)}")
        else:
            self._check_exit_conditions(last_price, atr_value, score)
        
        self.update_portfolio_display()

    def _enter_long_position(self, entry_price, atr_value):
        """Enter long position with risk management."""
        sl = entry_price - (2.0 * atr_value)
        tp = entry_price + (3.0 * (entry_price - sl))  # 3:1 R:R
        
        risk_amount = self.portfolio['capital'] * RISK_PERCENTAGE
        price_risk = entry_price - sl
        qty = risk_amount / price_risk if price_risk > 0 else 0
        
        # Ensure we don't exceed 95% of capital
        max_qty = (self.portfolio['capital'] * 0.95) / entry_price
        qty = min(qty, max_qty)
        
        if qty > 0:
            self.portfolio.update({
                'position_open': True,
                'position_type': 'LONG',
                'entry_price': entry_price,
                'quantity': qty,
                'stop_loss': sl,
                'take_profit': tp,
                'highest_price': entry_price,
                'last_trade_time': datetime.now()
            })
            self.portfolio['capital'] -= entry_price * qty
            self.portfolio['daily_trades'] += 1
            self.portfolio['trade_log'].append({
                'type': 'BUY (LONG)',
                'price': entry_price,
                'qty': qty,
                'time': self.last_intraday_data.index[-1],
                'balance': self.portfolio['capital']
            })
    def _check_exit_conditions(self, current_price, atr_value, score):
        """Check all exit conditions."""
        exit_reason = None
        position_type = self.portfolio['position_type']
        
        if position_type == 'LONG':
            # Update trailing stop
            if self.enable_trailing_stop and current_price > self.portfolio['highest_price']:
                self.portfolio['highest_price'] = current_price
                new_sl = current_price - (TRAILING_STOP_ATR_MULTIPLIER * atr_value)
                if new_sl > self.portfolio['stop_loss']:
                    self.portfolio['stop_loss'] = new_sl
            
            # Check exit conditions
            if current_price >= self.portfolio['take_profit']:
                exit_reason = "Take-Profit"
            elif current_price <= self.portfolio['stop_loss']:
                exit_reason = "Stop-Loss"
            elif score <= SELL_SCORE_THRESHOLD:
                exit_reason = "Signal Reversal"
        
        if exit_reason:
            self._close_position(current_price, exit_reason)

    def _close_position(self, exit_price, reason):
        """Close open position."""
        position_type = self.portfolio['position_type']
        qty = self.portfolio['quantity']
        entry_price = self.portfolio['entry_price']
        
        pnl = (exit_price - entry_price) * qty
        self.portfolio['capital'] += exit_price * qty
        
        self.portfolio['total_trades'] += 1
        if pnl > 0:
            self.portfolio['winning_trades'] += 1
        else:
            self.portfolio['losing_trades'] += 1
        
        self.portfolio['trade_log'].append({
            'type': f'CLOSE {position_type} ({reason})',
            'price': exit_price,
            'pnl': pnl,
            'time': self.last_intraday_data.index[-1],
            'balance': self.portfolio['capital']
        })
        
        pnl_emoji = "💰" if pnl > 0 else "📉"
        self.status_label.setText(
            f"{pnl_emoji} CLOSED {position_type}: ₹{exit_price:.2f} | {reason} | P&L: ₹{pnl:.2f}"
        )
        
        self.portfolio['position_open'] = False
        self.portfolio['position_type'] = None

    def _liquidate_position(self, reason):
        """Force close position."""
        if not self.last_intraday_data.empty:
            final_price = self.last_intraday_data['Close'].iloc[-1]
            self._close_position(final_price, reason)

    def update_metrics_display(self):
        """Update all metrics."""
        if self.daily_info:
            trend = self.daily_info['trend']
            self.daily_trend_val.setText(trend)
            if 'Strong Up' in trend or trend == 'Uptrend':
                color = "#00ff88" if self.dark_mode else "#00aa00"
            elif 'Down' in trend:
                color = "#ff4444" if self.dark_mode else "#cc0000"
            else:
                color = "#e0e0e0" if self.dark_mode else "#666"
            self.daily_trend_val.setStyleSheet(f"color: {color}; font-weight: bold;")
        
        # Calculate total value
        total_value = self.portfolio['capital']
        if self.portfolio['position_open'] and not self.last_intraday_data.empty:
            last_price = self.last_intraday_data['Close'].iloc[-1]
            total_value += last_price * self.portfolio['quantity']
        
        total_pnl = total_value - INITIAL_CAPITAL
        
        self.capital_val.setText(f"₹{self.portfolio['capital']:,.2f}")
        self.pnl_total_val.setText(f"{'+' if total_pnl >= 0 else ''}₹{total_pnl:,.2f}")
        pnl_color = "#00ff88" if self.dark_mode else "#00aa00" if total_pnl >= 0 else "#ff4444" if self.dark_mode else "#cc0000"
        self.pnl_total_val.setStyleSheet(f"color: {pnl_color}; font-weight: bold; font-size: 14px;")
        
        # Win rate
        if self.portfolio['total_trades'] > 0:
            win_rate = (self.portfolio['winning_trades'] / self.portfolio['total_trades']) * 100
            self.win_rate_val.setText(f"{win_rate:.1f}%")
            wr_color = "#00ff88" if self.dark_mode else "#00aa00" if win_rate >= 50 else "#ff4444" if self.dark_mode else "#cc0000"
            self.win_rate_val.setStyleSheet(f"color: {wr_color}; font-weight: bold;")
        
        # Daily trades
        self.daily_trades_val.setText(f"{self.portfolio['daily_trades']}/{MAX_DAILY_TRADES}")
        
        # Position
        if self.portfolio['position_open']:
            qty = self.portfolio['quantity']
            entry = self.portfolio['entry_price']
            pos_type = self.portfolio['position_type']
            self.position_val.setText(f"📈 {pos_type}: {qty:.2f} @ ₹{entry:.2f}")
            pos_color = "#ff9933" if self.dark_mode else "#ff7700"
            self.position_val.setStyleSheet(f"color: {pos_color}; font-weight: bold;")
        else:
            self.position_val.setText("None")
            self.position_val.setStyleSheet(f"color: {'#e0e0e0' if self.dark_mode else '#666'};")
        
        # Open P&L
        if self.portfolio['position_open'] and not self.last_intraday_data.empty:
            last_price = self.last_intraday_data['Close'].iloc[-1]
            unrealized_pnl = (last_price - self.portfolio['entry_price']) * self.portfolio['quantity']
            
            self.pnl_open_val.setText(f"{'+' if unrealized_pnl >= 0 else ''}₹{unrealized_pnl:,.2f}")
            open_color = "#00ff88" if self.dark_mode else "#00aa00" if unrealized_pnl >= 0 else "#ff4444" if self.dark_mode else "#cc0000"
            self.pnl_open_val.setStyleSheet(f"color: {open_color}; font-weight: bold;")
        else:
            self.pnl_open_val.setText("₹0.00")
            self.pnl_open_val.setStyleSheet(f"color: {'#e0e0e0' if self.dark_mode else '#666'};")

    def update_portfolio_display(self):
        """Update trade log table."""
        log = self.portfolio['trade_log']
        self.trade_table.setRowCount(len(log))
        
        for row, trade in enumerate(log):
            time_str = trade['time'].strftime('%m/%d %H:%M') if hasattr(trade['time'], 'strftime') else str(trade['time'])
            time_item = QTableWidgetItem(time_str)
            time_item.setTextAlignment(Qt.AlignCenter)
            self.trade_table.setItem(row, 0, time_item)
            
            type_item = QTableWidgetItem(trade['type'])
            type_item.setTextAlignment(Qt.AlignCenter)
            if 'BUY' in trade['type'] or 'LONG' in trade['type']:
                color = QColor("#00ff88" if self.dark_mode else "#00aa00")
            else:
                color = QColor("#ff4444" if self.dark_mode else "#cc0000")
            type_item.setForeground(color)
            self.trade_table.setItem(row, 1, type_item)
            
            price_item = QTableWidgetItem(f"₹{trade['price']:.2f}")
            price_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.trade_table.setItem(row, 2, price_item)
            
            pnl = trade.get('pnl')
            if pnl is not None:
                pnl_item = QTableWidgetItem(f"{'+' if pnl >= 0 else ''}₹{pnl:.2f}")
                pnl_color = QColor("#00ff88" if self.dark_mode else "#00aa00") if pnl >= 0 else QColor("#ff4444" if self.dark_mode else "#cc0000")
                pnl_item.setForeground(pnl_color)
                pnl_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.trade_table.setItem(row, 3, pnl_item)
            else:
                self.trade_table.setItem(row, 3, QTableWidgetItem("-"))
            
            balance = trade.get('balance', 0)
            balance_item = QTableWidgetItem(f"₹{balance:,.2f}")
            balance_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.trade_table.setItem(row, 4, balance_item)
        
        self.trade_table.scrollToBottom()

    def update_chart(self):
        """Update chart with enhanced visuals."""
        df = self.last_intraday_data
        if df.empty or self.daily_info is None or self.daily_info['pivots'] is None:
            return

        self.ax.clear()
        
        # Show last 3 hours
        if len(df.index) > 1:
            end_time = df.index[-1]
            start_time = end_time - timedelta(hours=3)
            df_view = df[df.index >= start_time]
        else:
            df_view = df
        
        if len(df_view) == 0:
            return
        
        # Colors
        if self.dark_mode:
            line_color = '#ff9933'
            fill_color = '#ff9933'
            up_color = '#00ff88'
            down_color = '#ff4444'
        else:
            line_color = '#ff7700'
            fill_color = '#ff9933'
            up_color = '#00aa00'
            down_color = '#cc0000'
        
        # Main price line
        self.ax.plot(df_view.index, df_view['Close'], label='Price', 
                     color=line_color, linewidth=2.5, zorder=3)
        self.ax.fill_between(df_view.index, df_view['Close'], 
                             alpha=0.15, color=fill_color, zorder=1)
        
        # Plot moving averages
        if 'SMA_20' in df_view.columns:
            self.ax.plot(df_view.index, df_view['SMA_20'], 
                        color='#00aaff', linewidth=1.5, alpha=0.7, label='SMA 20', linestyle='--')
        if 'EMA_12' in df_view.columns:
            self.ax.plot(df_view.index, df_view['EMA_12'], 
                        color='#aa00ff', linewidth=1.5, alpha=0.7, label='EMA 12', linestyle=':')
        
        # Plot pivots
        pivots = self.daily_info['pivots']
        pivot_color = '#888888' if self.dark_mode else '#666666'
        self.ax.axhline(pivots['pivot'], color=pivot_color, linestyle='--', 
                        label='Pivot', zorder=2, linewidth=1.5, alpha=0.6)
        self.ax.axhline(pivots['r1'], color=down_color, linestyle=':', 
                        label='R1', zorder=2, linewidth=1, alpha=0.5)
        self.ax.axhline(pivots['s1'], color=up_color, linestyle=':', 
                        label='S1', zorder=2, linewidth=1, alpha=0.5)
        
        # Plot position levels
        if self.portfolio['position_open']:
            self.ax.axhline(self.portfolio['stop_loss'], color=down_color, 
                            linestyle='-', linewidth=2, alpha=0.8, label='Stop Loss', zorder=4)
            self.ax.axhline(self.portfolio['take_profit'], color=up_color, 
                            linestyle='-', linewidth=2, alpha=0.8, label='Take Profit', zorder=4)
            self.ax.axhline(self.portfolio['entry_price'], color='#ffaa00', 
                            linestyle='-', linewidth=2, alpha=0.8, label='Entry', zorder=4)
        
        # Plot trades
        for trade in self.portfolio['trade_log']:
            trade_time = trade['time']
            if trade_time < df_view.index[0] or trade_time > df_view.index[-1]:
                continue
            
            trade_price = trade['price']
            
            if 'BUY' in trade['type']:
                self.ax.scatter(trade_time, trade_price, marker='^', 
                                color=up_color, s=250, zorder=5, edgecolors='white', linewidths=2)
            elif 'CLOSE' in trade['type']:
                pnl = trade.get('pnl', 0)
                color = up_color if pnl >= 0 else down_color
                self.ax.scatter(trade_time, trade_price, marker='v', 
                                color=color, s=250, zorder=5, edgecolors='white', linewidths=2)
        
        # Format axes
        if len(df_view.index) > 1:
            self.ax.set_xlim(df_view.index[0], df_view.index[-1])
        
        self.ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        self.ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=30))
        plt.setp(self.ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        text_color = '#e0e0e0' if self.dark_mode else '#333333'
        self.ax.set_xlabel("Time (IST)", color=text_color, fontweight='bold', fontsize=11)
        self.ax.set_ylabel("Price (₹)", color=text_color, fontweight='bold', fontsize=11)
        
        legend_bg = '#2a2a2a' if self.dark_mode else '#f9f9f9'
        legend_edge = '#555' if self.dark_mode else '#ccc'
        self.ax.legend(loc='upper left', facecolor=legend_bg, edgecolor=legend_edge, 
                       labelcolor=text_color, framealpha=0.95, fontsize=9)
        
        # Enhanced title
        last_price = df_view['Close'].iloc[-1]
        total_value = self.portfolio['capital']
        
        if self.portfolio['position_open']:
            total_value += last_price * self.portfolio['quantity']
            unrealized_pnl = (last_price - self.portfolio['entry_price']) * self.portfolio['quantity']
            
            title = (f"{self.selected_stock} | Price: ₹{last_price:.2f} | "
                     f"Capital: ₹{self.portfolio['capital']:,.2f} | "
                     f"Total: ₹{total_value:,.2f} | "
                     f"Open P&L: {'+' if unrealized_pnl >= 0 else ''}₹{unrealized_pnl:.2f}")
        else:
            total_pnl = total_value - INITIAL_CAPITAL
            title = (f"{self.selected_stock} | Price: ₹{last_price:.2f} | "
                     f"Capital: ₹{self.portfolio['capital']:,.2f} | "
                     f"Total P&L: {'+' if total_pnl >= 0 else ''}₹{total_pnl:,.2f}")
        
        self.ax.set_title(title, color=text_color, fontsize=12, fontweight='bold', pad=12)
        
        self.fig.tight_layout()
        self.canvas.draw()

    def _save_state(self):
        """Save trading state."""
        try:
            state = {
                'portfolio': self.portfolio,
                'selected_stock': self.selected_stock,
                'daily_info': self.daily_info,
                'timestamp': datetime.now()
            }
            
            with open(DATA_SAVE_FILE, 'wb') as f:
                pickle.dump(state, f)
            
            QMessageBox.information(self, "State Saved", 
                                  f"Trading state saved!\n\n"
                                  f"Stock: {self.selected_stock}\n"
                                  f"Capital: ₹{self.portfolio['capital']:,.2f}\n"
                                  f"Total Trades: {self.portfolio['total_trades']}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save:\n{str(e)}")
    
    def _load_state(self):
        """Load saved state."""
        if not os.path.exists(DATA_SAVE_FILE):
            return
        
        try:
            with open(DATA_SAVE_FILE, 'rb') as f:
                state = pickle.load(f)
            
            if 'portfolio' not in state:
                raise ValueError("Invalid state file")
            
            saved_portfolio = state['portfolio']
            total_trades = saved_portfolio.get('total_trades', 0)
            win_rate = 0
            if total_trades > 0:
                win_rate = (saved_portfolio.get('winning_trades', 0) / total_trades) * 100
            
            timestamp = state.get('timestamp', datetime.now())
            timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M:%S') if isinstance(timestamp, datetime) else str(timestamp)
            
            QTimer.singleShot(100, lambda: self._show_restore_dialog(state, saved_portfolio, total_trades, win_rate, timestamp_str))
        
        except Exception as e:
            print(f"Load error: {e}")
            try:
                os.remove(DATA_SAVE_FILE)
            except:
                pass
    
    def _show_restore_dialog(self, state, saved_portfolio, total_trades, win_rate, timestamp_str):
        """Show restore confirmation."""
        try:
            reply = QMessageBox.question(
                self, "Restore State?",
                f"Found saved state from {timestamp_str}\n\n"
                f"Stock: {state.get('selected_stock', 'Unknown')}\n"
                f"Capital: ₹{saved_portfolio['capital']:,.2f}\n"
                f"Total Trades: {total_trades}\n"
                f"Win Rate: {win_rate:.1f}%\n\n"
                f"Restore this state?",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.portfolio = saved_portfolio
                self.selected_stock = state.get('selected_stock')
                
                if self.selected_stock:
                    stock_name = TOP_NIFTY50_STOCKS.get(self.selected_stock, self.selected_stock)
                    self.active_stock_label.setText(f"{self.selected_stock}\n{stock_name} (Restored)")
                    self.stock_input.setText(self.selected_stock)
                
                self.update_portfolio_display()
                self.update_metrics_display()
                
                QMessageBox.information(self, "State Restored",
                    f"Successfully restored!\n\n"
                    f"Capital: ₹{self.portfolio['capital']:,.2f}\n"
                    f"Total Trades: {total_trades}\n\n"
                    f"Click 'START AUTO TRADING' to continue.")
        except Exception as e:
            print(f"Restore error: {e}")

    def closeEvent(self, event):
        """Handle app closure."""
        if self.worker_thread and self.worker_thread.isRunning():
            reply = QMessageBox.question(
                self, 'Close Application?',
                'Trading is active. Close anyway?\n\nOpen positions will be liquidated.',
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.stop_trading()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = IndianStockTradingTerminal()
    window.show()
    sys.exit(app.exec_())
