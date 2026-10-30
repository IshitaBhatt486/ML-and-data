# Retail Sales Forecasting
# Dataset: UCI Online Retail
# Forecasting method: Automatic ARIMA

library(readxl)
library(forecast)
library(ggplot2)

set.seed(42)

# 1. Load dataset
retail <- read_excel("Online_Retail.xlsx")

cat("Original dataset dimensions:", dim(retail), "\n")
cat("Missing values:\n")
print(colSums(is.na(retail)))

# 2. Clean data
# Exclude cancellations, returns and invalid prices/quantities
retail$InvoiceNo <- as.character(retail$InvoiceNo)

retail <- retail[
  !is.na(retail$InvoiceDate) &
    !is.na(retail$Quantity) &
    !is.na(retail$UnitPrice) &
    !grepl("^C", retail$InvoiceNo, ignore.case = TRUE) &
    retail$Quantity > 0 &
    retail$UnitPrice > 0,
]

# Revenue per transaction line
retail$Revenue <- retail$Quantity * retail$UnitPrice
retail$Date <- as.Date(retail$InvoiceDate)

cat("Rows after cleaning:", nrow(retail), "\n")

# 3. Aggregate revenue by day
daily_sales <- aggregate(
  Revenue ~ Date,
  data = retail,
  FUN = sum
)

# Fill missing calendar days with zero sales
all_dates <- data.frame(
  Date = seq(
    min(daily_sales$Date),
    max(daily_sales$Date),
    by = "day"
  )
)

daily_sales <- merge(all_dates, daily_sales, by = "Date", all.x = TRUE)
daily_sales$Revenue[is.na(daily_sales$Revenue)] <- 0
daily_sales <- daily_sales[order(daily_sales$Date), ]

write.csv(daily_sales, "daily_sales.csv", row.names = FALSE)

# 4. Visualize historical sales
p1 <- ggplot(daily_sales, aes(x = Date, y = Revenue)) +
  geom_line() +
  labs(
    title = "Daily Retail Revenue",
    x = "Date",
    y = "Revenue (£)"
  ) +
  theme_minimal()

print(p1)

ggsave("daily_sales.png", p1, width = 10, height = 5)

# 5. Split into training and test sets
# Reserve the last 30 days for evaluation
h <- 30
n <- nrow(daily_sales)

if (n <= h + 30) {
  stop("Not enough daily observations for this split.")
}

train <- daily_sales$Revenue[1:(n - h)]
test <- daily_sales$Revenue[(n - h + 1):n]
test_dates <- daily_sales$Date[(n - h + 1):n]

# Weekly seasonality: frequency = 7
train_ts <- ts(train, frequency = 7)

# 6. Fit automatic ARIMA model
model <- auto.arima(train_ts, seasonal = TRUE)
print(model)

# 7. Forecast the next 30 days
pred <- forecast(model, h = h, level = 95)

predicted <- as.numeric(pred$mean)

# 8. Evaluate forecast accuracy
mae <- mean(abs(test - predicted))
rmse <- sqrt(mean((test - predicted)^2))

# Symmetric MAPE avoids division by zero
denominator <- abs(test) + abs(predicted)
valid <- denominator > 0

smape <- if (any(valid)) {
  mean(200 * abs(test[valid] - predicted[valid]) /
         denominator[valid])
} else {
  NA_real_
}

metrics <- data.frame(
  Metric = c("MAE", "RMSE", "sMAPE (%)"),
  Value = c(mae, rmse, smape)
)

cat("\nForecast evaluation:\n")
print(metrics)

write.csv(metrics, "forecast_metrics.csv", row.names = FALSE)

# 9. Save forecast results
forecast_results <- data.frame(
  Date = test_dates,
  Actual = test,
  Forecast = predicted,
  Lower_95 = as.numeric(pred$lower[, 1]),
  Upper_95 = as.numeric(pred$upper[, 1])
)

write.csv(forecast_results, "forecast_results.csv", row.names = FALSE)

# 10. Plot actual vs forecast
p2 <- ggplot(forecast_results, aes(x = Date)) +
  geom_ribbon(
    aes(ymin = Lower_95, ymax = Upper_95),
    alpha = 0.2
  ) +
  geom_line(aes(y = Actual, colour = "Actual")) +
  geom_line(aes(y = Forecast, colour = "Forecast")) +
  scale_colour_manual(
    values = c("Actual" = "black", "Forecast" = "blue")
  ) +
  labs(
    title = "Actual vs Forecast Retail Revenue",
    subtitle = "Last 30 days held out for evaluation",
    x = "Date",
    y = "Revenue (£)",
    colour = "Series"
  ) +
  theme_minimal()

print(p2)

ggsave("actual_vs_forecast.png", p2, width = 10, height = 5)

cat("\nModel:", as.character(model), "\n")
cat("Forecast horizon:", h, "days\n")
cat("Total daily observations:", n, "\n")
cat("Training observations:", length(train), "\n")
cat("Test observations:", length(test), "\n")
cat("\nFiles saved successfully.\n")