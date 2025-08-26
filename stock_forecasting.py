import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import yfinance as yf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler

from pmdarima.arima import auto_arima
from prophet import Prophet

# Deep learning libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

# ----------------------
# Reproducibility
# ----------------------
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ----------------------
# 1. Download stock data
# ----------------------
ticker = "AAPL"
start_date = "2020-01-01"
end_date = "2023-12-31"

# Ensure auto_adjust to avoid future warning
data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
series = data['Close']

# ----------------------
# 2. Preprocessing
# ----------------------
series = series.ffill().squeeze()
series_log = np.log(series)

train_size = int(len(series) * 0.8)
idx_split = series.index[train_size]
train_log, test_log = series_log[:train_size], series_log[train_size:]

# Prophet DataFrame
series_df = pd.DataFrame({"ds": pd.to_datetime(series.index), "y": series.values})
train_prophet = series_df.iloc[:train_size]
test_prophet = series_df.iloc[train_size:]

# LSTM scaling
scaler = MinMaxScaler(feature_range=(0,1))
train_vals = series.values[:train_size].reshape(-1,1)
scaled_train = scaler.fit_transform(train_vals)
scaled_full = scaler.transform(series.values.reshape(-1,1))

# ----------------------
# 3. ARIMA Model
# ----------------------
auto_model = auto_arima(train_log, seasonal=False, stepwise=True, suppress_warnings=True)
model_arima = ARIMA(train_log, order=auto_model.order)
fit_arima = model_arima.fit()
forecast_log = fit_arima.forecast(steps=len(test_log))
forecast_arima = np.exp(forecast_log)
actual_arima = np.exp(test_log)

# ----------------------
# 4. Prophet Model
# ----------------------
prophet_model = Prophet(daily_seasonality=True)
prophet_model.fit(train_prophet)
future = prophet_model.make_future_dataframe(periods=len(test_prophet), freq='B')
forecast_prophet_df = prophet_model.predict(future)

# Merge to avoid missing index issues
prophet_pred_df = test_prophet[['ds']].merge(forecast_prophet_df[['ds','yhat']], on='ds', how='left')
prophet_pred = prophet_pred_df['yhat'].ffill().values

# ----------------------
# 5. LSTM Model
# ----------------------
WINDOW = 60

def make_sequences(values, window_size=WINDOW):
    X, y = [], []
    for i in range(window_size, len(values)):
        X.append(values[i-window_size:i,0])
        y.append(values[i,0])
    return np.array(X).reshape(-1,window_size,1), np.array(y)

X_all, y_all = make_sequences(scaled_full)
index_seq = series.index[WINDOW:]
split_pos = np.where(index_seq >= idx_split)[0][0]
X_train, y_train = X_all[:split_pos], y_all[:split_pos]
X_test, y_test = X_all[split_pos:], y_all[split_pos:]
idx_test = index_seq[split_pos:]

# Use Input layer to remove Keras input_shape warning
model_lstm = Sequential([
    Input(shape=(WINDOW,1)),
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(1)
])
model_lstm.compile(optimizer='adam', loss='mse')
es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_lstm.fit(X_train, y_train, validation_split=0.2, epochs=100, batch_size=32, callbacks=[es], verbose=0)
pred_scaled = model_lstm.predict(X_test, verbose=0)
pred_prices = scaler.inverse_transform(pred_scaled)
true_prices = scaler.inverse_transform(y_test.reshape(-1,1))

# ----------------------
# 6. Evaluation Function
# ----------------------
def evaluate_model(name, y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"\n{name} Evaluation:")
    print(f"MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}

metrics = {}
metrics['ARIMA'] = evaluate_model('ARIMA', actual_arima, forecast_arima)
metrics['Prophet'] = evaluate_model('Prophet', test_prophet['y'].values, prophet_pred)
metrics['LSTM'] = evaluate_model('LSTM', true_prices.ravel(), pred_prices.ravel())

# ----------------------
# 7. Plots
# ----------------------
plt.figure(figsize=(12,6))
plt.plot(series.index, series, label='Actual')
plt.plot(actual_arima.index, forecast_arima, label='ARIMA Forecast')
plt.title(f"{ticker} ARIMA Forecast")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()

plt.figure(figsize=(12,6))
plt.plot(series.index, series, label='Actual')
plt.plot(test_prophet['ds'], prophet_pred, label='Prophet Forecast')
plt.title(f"{ticker} Prophet Forecast")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()

plt.figure(figsize=(12,6))
plt.plot(series.index, series, label='Actual')
plt.plot(idx_test, pred_prices.ravel(), label='LSTM Forecast')
plt.title(f"{ticker} LSTM Forecast")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()

# ----------------------
# 8. Comparison Table
# ----------------------
print("\nModel comparison:")
comp_df = pd.DataFrame(metrics).T
print(comp_df.round(4))
