# StockForecasting

### Stock Price Forecasting with ARIMA, Prophet & LSTM ###

This project demonstrates how to forecast stock prices using three different approaches:
- ARIMA (classical time series model)
- Prophet (Facebook’s additive model for time series forecasting)
- LSTM (deep learning recurrent neural network)

The project includes  data preprocessing, visualization, model training, evaluation, and comparison between these models.

---

## Features
- Automatic data fetching from Yahoo Finance
- Preprocessing (missing value handling, log transformation, scaling)
- Model training with:
  - Auto-ARIMA (best parameters via AIC)
  - Prophet with daily seasonality
  - LSTM (with dropout & early stopping)
- Model evaluation with metrics: MSE, RMSE, MAE, R²
- Forecasting & visualization
- Comparison table for models

---

##  Project Structure
── stock_forecasting.py # Main script with all models
├── README.md # Project documentation
└── requirements.txt # Dependencies


# Outputs


Comments: ARIMA and Prophet did not performed well with the presented dataset.With both approaches having negative Rˆ2 value. While LSTM, an AI architecture indicated for learn non-linear relationships and temporal dependencies may be the suitable one for this specific task.

ARIMA Evaluation:
MSE: 962.0496, RMSE: 31.0169, MAE: 29.0030, R2: -6.9584

Prophet Evaluation:
MSE: 3487.6850, RMSE: 59.0566, MAE: 57.2262, R2: -27.8511

LSTM Evaluation:
MSE: 28.2707, RMSE: 5.3170, MAE: 4.6738, R2: 0.7661

Model comparison:
               MSE     RMSE      MAE       R2
ARIMA     962.0496  31.0169  29.0030  -6.9584
Prophet  3487.6850  59.0566  57.2262 -27.8511
LSTM       28.2707   5.3170   4.6738   0.7661
