# Retail Sales Forecasting using Seasonal Naive

Time-series forecasting in R that analyzes real-world retail transactions, compares four forecasting models, and predicts daily revenue. It uses historical transaction data to aggregate daily revenue, evaluate forecasting methods, and generate a 30-day forecast using the best-performing model.

## Dataset

- Source: [UCI Online Retail Dataset](https://archive.ics.uci.edu/dataset/352/online+retail)
- Original records: 541,909
- Records after cleaning: 530,104
- Daily observations: 374
- Target variable: Daily revenue (£)

## Methodology

1. Clean transactions by excluding cancellations, non-positive quantities, and non-positive prices.
2. Calculate revenue as `Quantity × UnitPrice`.
3. Aggregate revenue by date and fill missing calendar dates with zero.
4. Split the data into 344 training days and 30 test days.
5. Compare Naive, Seasonal Naive, ETS, and ARIMA forecasts.
6. Select the best model based on test-set RMSE.
7. Forecast the next 30 days using Seasonal Naive.

## Model Comparison

Results on the held-out 30-day test period:

| Model | MAE (£) | RMSE (£) | sMAPE (%) |
|---|---:|---:|---:|
| Seasonal Naive | 14,535.73 | 28,649.68 | 24.08 |
| ETS | 16,768.10 | 30,677.97 | 46.59 |
| ARIMA | 16,659.05 | 32,845.27 | 47.89 |
| Naive | 26,772.84 | 39,371.94 | 53.20 |

Seasonal Naive achieved the lowest error across all three reported metrics.

## Forecasting Method

The Seasonal Naive model predicts revenue using the observed revenue from the same weekday in the previous week. It was selected because it outperformed the other tested models on the held-out test period.

The final model was fitted to all available observations to forecast the 30 days following December 9, 2011.

**Forecast period:** December 10, 2011 – January 8, 2012.

## Key Findings

- Seasonal Naive achieved an RMSE of £28,649.68 and an MAE of £14,535.73.
- Its RMSE was approximately 12.8% lower than ARIMA's.
- The results demonstrate that a simple seasonal baseline can outperform more complex models.
- Forecasts repeat the previous week's revenue pattern, so they may not capture holidays, promotions, or changing demand.

## Limitations

- Model performance was evaluated on a single 30-day test period.
- The dataset covers a limited historical period.
- Missing calendar dates were assigned zero revenue.
- The model does not explicitly account for holidays or external factors.
- Negative lower prediction bounds were clipped to zero for the final forecast output; this does not improve model accuracy.

## Conclusion

Seasonal Naive performed best on the test set, emphasizing the importance of benchmarking complex models against simple alternatives.