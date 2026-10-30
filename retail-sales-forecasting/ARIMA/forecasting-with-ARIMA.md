# Retail Sales Forecasting with ARIMA

Time-series forecasting using retail transaction data to analyze daily revenue and evaluate 30-day sales forecasts using R.

## Dataset
- Source: [UCI Online Retail Dataset](https://archive.ics.uci.edu/dataset/352/online+retail)
- Original records: 541,909
- Records after cleaning: 530,104
- Daily observations: 374
- Target variable: Daily revenue (£)

## Methodology
1. Clean transaction records by excluding cancellations, returns, invalid quantities and prices.
2. Calculate revenue as quantity multiplied by unit price.
3. Aggregate revenue by day and fill missing calendar dates with zero.
4. Reserve the last 30 days for testing.
5. Fit a seasonal ARIMA model using `auto.arima()`.
6. Evaluate predictions using MAE, RMSE and sMAPE.

## Results

Selected model: `ARIMA(1,0,1)(2,1,1)[7] with drift`

| Metric | Result |
|---|---:|
| MAE | £16,659.05 |
| RMSE | £32,845.27 |
| sMAPE | 47.89% |
| Training period | 344 days |
| Test period | 30 days |

## Interpretation
The model produced forecasts for a held-out 30-day period, but the observed errors indicate considerable differences between predicted and actual daily revenue. RMSE exceeding MAE suggests that some larger errors influenced overall performance.

The results provide a baseline for further experimentation rather than evidence of high forecasting accuracy. Comparing ARIMA with a seasonal-naive model would help establish whether the added model complexity improves predictions.

## Limitations
- Only one 30-day test period was evaluated.
- Forecast accuracy may be affected by unusual sales spikes and zero-sales days.
- The dataset covers a limited historical period.
- No baseline model comparison was performed.