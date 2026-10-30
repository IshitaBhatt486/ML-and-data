## Model Comparison

Four forecasting methods were evaluated on the same held-out 30-day period: naive, seasonal naive, exponential smoothing (ETS), and seasonal ARIMA.

| Model | MAE (£) | RMSE (£) | sMAPE (%) |
|---|---:|---:|---:|
| Seasonal Naive | 14,535.73 | 28,649.68 | 24.08 |
| ETS | 16,768.10 | 30,677.97 | 46.59 |
| ARIMA | 16,659.05 | 32,845.27 | 47.89 |
| Naive | 26,772.84 | 39,371.94 | 53.20 |

### Findings

- Seasonal naive achieved the lowest MAE, RMSE, and sMAPE among the four tested methods.
- Its RMSE was approximately 12.8% lower than ARIMA's.
- ARIMA's Ljung–Box test returned a p-value of 0.0269, indicating evidence of remaining residual autocorrelation at the 5% significance level.
- Results are based on one test window and may not generalize to other periods.

### Conclusion

A simple seasonal baseline outperformed the more complex forecasting models on this test set.