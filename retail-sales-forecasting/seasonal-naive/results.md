> # Retail Sales Forecasting using Seasonal Naive
> # Dataset: UCI Online Retail
> 
> library(readxl)
> library(forecast)
> library(ggplot2)
> 
> # 1. Load and clean data
> retail <- read_excel("C:/Users/hp/OneDrive - Shiv Nadar Institution of Eminence/Desktop/github/data-and-ML/retail-sales-forecasting/Online Retail.xlsx")
>                                                                                            
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
> retail$Revenue <- retail$Quantity * retail$UnitPrice
> retail$Date <- as.Date(retail$InvoiceDate)
> 
> # 2. Aggregate daily revenue
> daily_sales <- aggregate(
+   Revenue ~ Date,
+   data = retail,
+   sum
+ )
> 
> all_dates <- data.frame(
+   Date = seq(min(daily_sales$Date),
+              max(daily_sales$Date),
+              by = "day")
+ )
> 
> daily_sales <- merge(
+   all_dates, daily_sales,
+   by = "Date", all.x = TRUE
+ )
> 
> daily_sales$Revenue[is.na(daily_sales$Revenue)] <- 0
> daily_sales <- daily_sales[order(daily_sales$Date), ]
> 
> # 3. Fit Seasonal Naive model using all available data
> sales_ts <- ts(daily_sales$Revenue, frequency = 7)
> 
> model <- snaive(sales_ts, h = 30, level = 95)
> 
> # 4. Generate next 30 days of forecasts
> future_dates <- seq(
+   max(daily_sales$Date) + 1,
+   by = "day",
+   length.out = 30
+ )
> 
> forecast_results <- data.frame(
+   Date = future_dates,
+   Forecast = as.numeric(model$mean),
+   Lower_95 = as.numeric(model$lower[, 1]),
+   Upper_95 = as.numeric(model$upper[, 1])
+ )
> 
> print(forecast_results)
         Date  Forecast   Lower_95  Upper_95
1  2011-12-10      0.00 -35761.268  35761.27
2  2011-12-11  24621.43 -11139.838  60382.70
3  2011-12-12  88741.96  52980.692 124503.23
4  2011-12-13  56713.21  20951.942  92474.48
5  2011-12-14  75439.16  39677.892 111200.43
6  2011-12-15  82495.00  46733.732 118256.27
7  2011-12-16 200920.60 165159.332 236681.87
8  2011-12-17      0.00 -50574.070  50574.07
9  2011-12-18  24621.43 -25952.640  75195.50
10 2011-12-19  88741.96  38167.890 139316.03
11 2011-12-20  56713.21   6139.140 107287.28
12 2011-12-21  75439.16  24865.090 126013.23
13 2011-12-22  82495.00  31920.930 133069.07
14 2011-12-23 200920.60 150346.530 251494.67
15 2011-12-24      0.00 -61940.333  61940.33
16 2011-12-25  24621.43 -37318.903  86561.76
17 2011-12-26  88741.96  26801.627 150682.29
18 2011-12-27  56713.21  -5227.123 118653.54
19 2011-12-28  75439.16  13498.827 137379.49
20 2011-12-29  82495.00  20554.667 144435.33
21 2011-12-30 200920.60 138980.267 262860.93
22 2011-12-31      0.00 -71522.536  71522.54
23 2012-01-01  24621.43 -46901.106  96143.97
24 2012-01-02  88741.96  17219.424 160264.50
25 2012-01-03  56713.21 -14809.326 128235.75
26 2012-01-04  75439.16   3916.624 146961.70
27 2012-01-05  82495.00  10972.464 154017.54
28 2012-01-06 200920.60 129398.064 272443.14
29 2012-01-07      0.00 -79964.626  79964.63
30 2012-01-08  24621.43 -55343.196 104586.06
> 
> write.csv(
+   forecast_results,
+   "seasonal_naive_forecast.csv",
+   row.names = FALSE
+ )
> 
> # 5. Plot historical revenue and future forecast
> plot_data <- data.frame(
+   Date = daily_sales$Date,
+   Revenue = daily_sales$Revenue
+ )
> 
> p <- ggplot() +
+   geom_line(
+     data = plot_data,
+     aes(x = Date, y = Revenue),
+     colour = "black"
+   ) +
+   geom_ribbon(
+     data = forecast_results,
+     aes(x = Date, ymin = Lower_95, ymax = Upper_95),
+     fill = "skyblue",
+     alpha = 0.35
+   ) +
+   geom_line(
+     data = forecast_results,
+     aes(x = Date, y = Forecast),
+     colour = "blue",
+     linewidth = 0.9
+   ) +
+   labs(
+     title = "30-Day Retail Revenue Forecast",
+     subtitle = "Seasonal Naive Model",
+     x = "Date",
+     y = "Revenue (£)"
+   ) +
+   theme_minimal()
> 
> print(p)
> 
> ggsave(
+   "seasonal_naive_forecast.png",
+   p,
+   width = 10,
+   height = 5
+ )
> 
> cat("\nForecast complete.\n")

Forecast complete.
> cat("Model: Seasonal Naive\n")
Model: Seasonal Naive
> cat("Forecast horizon: 30 days\n")
Forecast horizon: 30 days
> cat("Files saved successfully.\n")
Files saved successfully.
> # Retail Sales Forecasting using Seasonal Naive
> # Dataset: UCI Online Retail
> 
> library(readxl)
> library(forecast)
> library(ggplot2)
> 
> # 1. Load dataset
> retail <- read_excel("C:/Users/hp/OneDrive - Shiv Nadar Institution of Eminence/Desktop/github/data-and-ML/retail-sales-forecasting/Online Retail.xlsx")
>                                                                                            
> cat("Original dataset dimensions:", dim(retail), "\n")
Original dataset dimensions: 541909 8 
> 
> # 2. Clean data
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
> # Fill missing calendar dates with zero revenue
> all_dates <- data.frame(
+   Date = seq(
+     min(daily_sales$Date),
+     max(daily_sales$Date),
+     by = "day"
+   )
+ )
> 
> daily_sales <- merge(
+   all_dates,
+   daily_sales,
+   by = "Date",
+   all.x = TRUE
+ )
> 
> daily_sales$Revenue[is.na(daily_sales$Revenue)] <- 0
> daily_sales <- daily_sales[order(daily_sales$Date), ]
> 
> write.csv(
+   daily_sales,
+   "daily_sales.csv",
+   row.names = FALSE
+ )
> 
> # 4. Fit Seasonal Naive model
> # Repeats the previous week's daily revenue pattern
> sales_ts <- ts(daily_sales$Revenue, frequency = 7)
> 
> h <- 30
> model <- snaive(sales_ts, h = h, level = 95)
> 
> # 5. Generate forecasts for the next 30 days
> future_dates <- seq(
+   max(daily_sales$Date) + 1,
+   by = "day",
+   length.out = h
+ )
> 
> forecast_results <- data.frame(
+   Date = future_dates,
+   Forecast = as.numeric(model$mean),
+   Lower_95 = as.numeric(model$lower[, 1]),
+   Upper_95 = as.numeric(model$upper[, 1])
+ )
> 
> # Prevent negative lower prediction bounds
> forecast_results$Lower_95 <- pmax(
+   0,
+   forecast_results$Lower_95
+ )
> 
> # Display forecast results
> cat("\n30-Day Revenue Forecast:\n")

30-Day Revenue Forecast:
> print(forecast_results)
         Date  Forecast   Lower_95  Upper_95
1  2011-12-10      0.00      0.000  35761.27
2  2011-12-11  24621.43      0.000  60382.70
3  2011-12-12  88741.96  52980.692 124503.23
4  2011-12-13  56713.21  20951.942  92474.48
5  2011-12-14  75439.16  39677.892 111200.43
6  2011-12-15  82495.00  46733.732 118256.27
7  2011-12-16 200920.60 165159.332 236681.87
8  2011-12-17      0.00      0.000  50574.07
9  2011-12-18  24621.43      0.000  75195.50
10 2011-12-19  88741.96  38167.890 139316.03
11 2011-12-20  56713.21   6139.140 107287.28
12 2011-12-21  75439.16  24865.090 126013.23
13 2011-12-22  82495.00  31920.930 133069.07
14 2011-12-23 200920.60 150346.530 251494.67
15 2011-12-24      0.00      0.000  61940.33
16 2011-12-25  24621.43      0.000  86561.76
17 2011-12-26  88741.96  26801.627 150682.29
18 2011-12-27  56713.21      0.000 118653.54
19 2011-12-28  75439.16  13498.827 137379.49
20 2011-12-29  82495.00  20554.667 144435.33
21 2011-12-30 200920.60 138980.267 262860.93
22 2011-12-31      0.00      0.000  71522.54
23 2012-01-01  24621.43      0.000  96143.97
24 2012-01-02  88741.96  17219.424 160264.50
25 2012-01-03  56713.21      0.000 128235.75
26 2012-01-04  75439.16   3916.624 146961.70
27 2012-01-05  82495.00  10972.464 154017.54
28 2012-01-06 200920.60 129398.064 272443.14
29 2012-01-07      0.00      0.000  79964.63
30 2012-01-08  24621.43      0.000 104586.06
> 
> write.csv(
+   forecast_results,
+   "seasonal_naive_forecast.csv",
+   row.names = FALSE
+ )
> 
> # 6. Plot historical revenue
> p1 <- ggplot(
+   daily_sales,
+   aes(x = Date, y = Revenue)
+ ) +
+   geom_line() +
+   labs(
+     title = "Historical Daily Retail Revenue",
+     x = "Date",
+     y = "Revenue (£)"
+   ) +
+   theme_minimal()
> 
> print(p1)
> 
> ggsave(
+   "daily_sales.png",
+   p1,
+   width = 10,
+   height = 5
+ )
> 
> # 7. Plot historical revenue and future forecast
> p2 <- ggplot() +
+   geom_line(
+     data = daily_sales,
+     aes(x = Date, y = Revenue),
+     colour = "black"
+   ) +
+   geom_ribbon(
+     data = forecast_results,
+     aes(
+       x = Date,
+       ymin = Lower_95,
+       ymax = Upper_95
+     ),
+     fill = "skyblue",
+     alpha = 0.35
+   ) +
+   geom_line(
+     data = forecast_results,
+     aes(x = Date, y = Forecast),
+     colour = "blue",
+     linewidth = 0.9
+   ) +
+   labs(
+     title = "30-Day Retail Revenue Forecast",
+     subtitle = "Seasonal Naive Model with 95% Prediction Intervals",
+     x = "Date",
+     y = "Revenue (£)"
+   ) +
+   theme_minimal()
> 
> print(p2)
> 
> ggsave(
+   "seasonal_naive_forecast.png",
+   p2,
+   width = 10,
+   height = 5
+ )
> 
> # 8. Summary
> cat("\nForecast complete.\n")

Forecast complete.
> cat("Model: Seasonal Naive\n")
Model: Seasonal Naive
> cat("Forecast horizon:", h, "days\n")
Forecast horizon: 30 days
> cat("Last historical date:", as.character(max(daily_sales$Date)), "\n")
Last historical date: 2011-12-09 
> cat(
+   "Forecast period:",
+   as.character(min(future_dates)),
+   "to",
+   as.character(max(future_dates)),
+   "\n"
+ )
Forecast period: 2011-12-10 to 2012-01-08 
> cat("Files saved successfully.\n")
Files saved successfully.