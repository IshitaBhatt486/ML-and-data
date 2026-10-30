# Retail Sales Forecasting using Seasonal Naive
# Dataset: UCI Online Retail

library(readxl)
library(forecast)
library(ggplot2)

# 1. Load dataset
retail <- read_excel("Online_Retail.xlsx")

cat("Original dataset dimensions:", dim(retail), "\n")

# 2. Clean data
retail$InvoiceNo <- as.character(retail$InvoiceNo)

retail <- retail[
  !is.na(retail$InvoiceDate) &
    !is.na(retail$Quantity) &
    !is.na(retail$UnitPrice) &
    !grepl("^C", retail$InvoiceNo, ignore.case = TRUE) &
    retail$Quantity > 0 &
    retail$UnitPrice > 0,
]

retail$Revenue <- retail$Quantity * retail$UnitPrice
retail$Date <- as.Date(retail$InvoiceDate)

cat("Rows after cleaning:", nrow(retail), "\n")

# 3. Aggregate revenue by day
daily_sales <- aggregate(
  Revenue ~ Date,
  data = retail,
  FUN = sum
)

# Fill missing calendar dates with zero revenue
all_dates <- data.frame(
  Date = seq(
    min(daily_sales$Date),
    max(daily_sales$Date),
    by = "day"
  )
)

daily_sales <- merge(
  all_dates,
  daily_sales,
  by = "Date",
  all.x = TRUE
)

daily_sales$Revenue[is.na(daily_sales$Revenue)] <- 0
daily_sales <- daily_sales[order(daily_sales$Date), ]

write.csv(
  daily_sales,
  "daily_sales.csv",
  row.names = FALSE
)

# 4. Fit Seasonal Naive model
# Repeats the previous week's daily revenue pattern
sales_ts <- ts(daily_sales$Revenue, frequency = 7)

h <- 30
model <- snaive(sales_ts, h = h, level = 95)

# 5. Generate forecasts for the next 30 days
future_dates <- seq(
  max(daily_sales$Date) + 1,
  by = "day",
  length.out = h
)

forecast_results <- data.frame(
  Date = future_dates,
  Forecast = as.numeric(model$mean),
  Lower_95 = as.numeric(model$lower[, 1]),
  Upper_95 = as.numeric(model$upper[, 1])
)

# Prevent negative lower prediction bounds
forecast_results$Lower_95 <- pmax(
  0,
  forecast_results$Lower_95
)

# Display forecast results
cat("\n30-Day Revenue Forecast:\n")
print(forecast_results)

write.csv(
  forecast_results,
  "seasonal_naive_forecast.csv",
  row.names = FALSE
)

# 6. Plot historical revenue
p1 <- ggplot(
  daily_sales,
  aes(x = Date, y = Revenue)
) +
  geom_line() +
  labs(
    title = "Historical Daily Retail Revenue",
    x = "Date",
    y = "Revenue (£)"
  ) +
  theme_minimal()

print(p1)

ggsave(
  "daily_sales.png",
  p1,
  width = 10,
  height = 5
)

# 7. Plot historical revenue and future forecast
p2 <- ggplot() +
  geom_line(
    data = daily_sales,
    aes(x = Date, y = Revenue),
    colour = "black"
  ) +
  geom_ribbon(
    data = forecast_results,
    aes(
      x = Date,
      ymin = Lower_95,
      ymax = Upper_95
    ),
    fill = "skyblue",
    alpha = 0.35
  ) +
  geom_line(
    data = forecast_results,
    aes(x = Date, y = Forecast),
    colour = "blue",
    linewidth = 0.9
  ) +
  labs(
    title = "30-Day Retail Revenue Forecast",
    subtitle = "Seasonal Naive Model with 95% Prediction Intervals",
    x = "Date",
    y = "Revenue (£)"
  ) +
  theme_minimal()

print(p2)

ggsave(
  "seasonal_naive_forecast.png",
  p2,
  width = 10,
  height = 5
)

# 8. Summary
cat("\nForecast complete.\n")
cat("Model: Seasonal Naive\n")
cat("Forecast horizon:", h, "days\n")
cat("Last historical date:", as.character(max(daily_sales$Date)), "\n")
cat(
  "Forecast period:",
  as.character(min(future_dates)),
  "to",
  as.character(max(future_dates)),
  "\n"
)
cat("Files saved successfully.\n")