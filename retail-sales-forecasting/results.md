> # Retail Sales Forecasting
> # Dataset: UCI Online Retail
> 
> library(readxl)
> library(forecast)
> library(ggplot2)
> 
> set.seed(42)
> 
> # 1. Load dataset
> retail <- read_excel("Online_Retail.xlsx")
>                                                                                            
> cat("Original dataset dimensions:", dim(retail), "\n")
Original dataset dimensions: 541909 8 
> cat("Missing values:\n")
Missing values:
> print(colSums(is.na(retail)))
  InvoiceNo   StockCode Description    Quantity InvoiceDate   UnitPrice  CustomerID 
          0           0        1454           0           0           0      135080 
    Country 
          0 
> 
> # 2. Clean data
> # Exclude cancellations, returns and invalid prices/quantities
> retail$InvoiceNo <- as.character(retail$InvoiceNo)
> 
> retail <- retail[
+   !is.na(retail$InvoiceDate) &
+     !is.na(retail$Quantity) &
+     !is.na(retail$UnitPrice) &
+     !grepl("^C", retail$InvoiceNo, ignore.case = TRUE) &
+     retail$Quantity > 0 &
+     retail$UnitPrice > 0,
+ ]
> 
> # Revenue per transaction line
> retail$Revenue <- retail$Quantity * retail$UnitPrice
> retail$Date <- as.Date(retail$InvoiceDate)
> 
> cat("Rows after cleaning:", nrow(retail), "\n")
Rows after cleaning: 530104 
> 
> # 3. Aggregate revenue by day
> daily_sales <- aggregate(
+   Revenue ~ Date,
+   data = retail,
+   FUN = sum
+ )
> 
> # Fill missing calendar days with zero sales
> all_dates <- data.frame(
+   Date = seq(
+     min(daily_sales$Date),
+     max(daily_sales$Date),
+     by = "day"
+   )
+ )
> 
> daily_sales <- merge(all_dates, daily_sales, by = "Date", all.x = TRUE)
> daily_sales$Revenue[is.na(daily_sales$Revenue)] <- 0
> daily_sales <- daily_sales[order(daily_sales$Date), ]
> 
> write.csv(daily_sales, "daily_sales.csv", row.names = FALSE)
> 
> # 4. Visualize historical sales
> p1 <- ggplot(daily_sales, aes(x = Date, y = Revenue)) +
+   geom_line() +
+   labs(
+     title = "Daily Retail Revenue",
+     x = "Date",
+     y = "Revenue (£)"
+   ) +
+   theme_minimal()
> 
> print(p1)
> 
> ggsave("daily_sales.png", p1, width = 10, height = 5)
> 
> # 5. Split into training and test sets
> # Reserve the last 30 days for evaluation
> h <- 30
> n <- nrow(daily_sales)
> 
> if (n <= h + 30) {
+   stop("Not enough daily observations for this split.")
+ }
> 
> train <- daily_sales$Revenue[1:(n - h)]
> test <- daily_sales$Revenue[(n - h + 1):n]
> test_dates <- daily_sales$Date[(n - h + 1):n]
> 
> # Weekly seasonality: frequency = 7
> train_ts <- ts(train, frequency = 7)
> 
> 
> # 6. Fit and compare forecasting models
> 
> library(forecast)
> 
> # ARIMA
> fit_arima <- auto.arima(train_ts, seasonal = TRUE)
> fc_arima <- forecast(fit_arima, h = h, level = 95)
> 
> # Exponential smoothing
> fit_ets <- ets(train_ts)
> fc_ets <- forecast(fit_ets, h = h, level = 95)
> 
> # Seasonal naive: repeats the previous week's pattern
> fc_snaive <- snaive(train_ts, h = h, level = 95)
> 
> # Naive: repeats the last observed value
> fc_naive <- naive(train_ts, h = h, level = 95)
> 
> # 7. Compare models on the same test period
> 
> forecasts <- list(
+   Naive = as.numeric(fc_naive$mean),
+   Seasonal_Naive = as.numeric(fc_snaive$mean),
+   ETS = as.numeric(fc_ets$mean),
+   ARIMA = as.numeric(fc_arima$mean)
+ )
> 
> evaluate <- function(actual, predicted) {
+   denom <- abs(actual) + abs(predicted)
+   valid <- denom > 0
+   
+   c(
+     MAE = mean(abs(actual - predicted)),
+     RMSE = sqrt(mean((actual - predicted)^2)),
+     sMAPE = if (any(valid)) {
+       mean(200 * abs(actual[valid] - predicted[valid]) /
+              denom[valid])
+     } else {
+       NA_real_
+     }
+   )
+ }
> 
> metrics <- as.data.frame(
+   t(sapply(forecasts, function(x) evaluate(test, x)))
+ )
> 
> metrics$Model <- rownames(metrics)
> rownames(metrics) <- NULL
> 
> metrics <- metrics[, c("Model", "MAE", "RMSE", "sMAPE")]
> metrics <- metrics[order(metrics$RMSE), ]
> 
> print(metrics)
           Model      MAE     RMSE    sMAPE
2 Seasonal_Naive 14535.73 28649.68 24.07901
3            ETS 16768.10 30677.97 46.58628
4          ARIMA 16659.05 32845.27 47.89467
1          Naive 26772.84 39371.94 53.19624
> write.csv(metrics, "model_comparison.csv", row.names = FALSE)
> 
> # 8. Plot all forecasts against actual revenue
> 
> comparison <- data.frame(
+   Date = test_dates,
+   Actual = test,
+   Naive = forecasts$Naive,
+   Seasonal_Naive = forecasts$Seasonal_Naive,
+   ETS = forecasts$ETS,
+   ARIMA = forecasts$ARIMA
+ )
> 
> comparison_long <- reshape(
+   comparison,
+   varying = c("Naive", "Seasonal_Naive", "ETS", "ARIMA"),
+   v.names = "Forecast",
+   timevar = "Model",
+   times = c("Naive", "Seasonal Naive", "ETS", "ARIMA"),
+   direction = "long"
+ )
> 
> p <- ggplot() +
+   geom_line(
+     data = comparison,
+     aes(x = Date, y = Actual, colour = "Actual"),
+     linewidth = 0.8
+   ) +
+   geom_line(
+     data = comparison_long,
+     aes(x = Date, y = Forecast, colour = Model),
+     linewidth = 0.7
+   ) +
+   labs(
+     title = "Retail Revenue: Forecast Model Comparison",
+     x = "Date",
+     y = "Revenue (£)",
+     colour = "Series"
+   ) +
+   theme_minimal()
> 
> print(p)
> ggsave("model_comparison.png", p, width = 10, height = 5)
> 
> # 9. Diagnose ARIMA residuals
> checkresiduals(fit_arima)

	Ljung-Box test

data:  Residuals from ARIMA(1,0,1)(2,1,1)[7] with drift
Q* = 18.807, df = 9, p-value = 0.02689

Model df: 5.   Total lags used: 14

> 
> # 10. Identify the lowest-RMSE model
> cat("\nBest model by test RMSE:", metrics$Model[1], "\n")

Best model by test RMSE: Seasonal_Naive 
> cat("Test RMSE:", round(metrics$RMSE[1], 2), "\n")
Test RMSE: 28649.68