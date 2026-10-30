
# Marketing A/B Testing Analysis
# Dataset: marketing_AB.csv
# Compares ad exposure against a public service announcement (PSA).

# 1. Install packages if needed
if (!requireNamespace("ggplot2", quietly = TRUE)) {
  install.packages("ggplot2")
}
library(ggplot2)

# 2. Load data
data <- read.csv("marketing_AB.csv", stringsAsFactors = FALSE)

# Handle common column-name variations
names(data) <- tolower(gsub("[^A-Za-z0-9]+", "_", names(data)))
names(data) <- gsub("_$", "", names(data))

required <- c("test_group", "converted")
if (!all(required %in% names(data))) {
  stop("Required columns not found. Check names(data) and the CSV format.")
}

# 3. Clean data
data <- data[!is.na(data$test_group) & !is.na(data$converted), ]
data$test_group <- tolower(trimws(data$test_group))
data$converted <- tolower(trimws(as.character(data$converted)))

# Keep only valid experiment groups and binary outcomes
data <- data[
  data$test_group %in% c("ad", "psa") &
    data$converted %in% c("true", "false", "1", "0"),
]

data$converted <- data$converted %in% c("true", "1")

# 4. Summary statistics
summary <- aggregate(
  converted ~ test_group,
  data = data,
  FUN = function(x) c(
    visitors = length(x),
    conversions = sum(x),
    rate = mean(x)
  )
)

cat("\n--- Conversion Summary ---\n")
print(summary)

ad <- data[data$test_group == "ad", ]
psa <- data[data$test_group == "psa", ]

if (nrow(ad) == 0 || nrow(psa) == 0) {
  stop("Both ad and psa groups must contain observations.")
}

ad_conversions <- sum(ad$converted)
psa_conversions <- sum(psa$converted)

ad_rate <- mean(ad$converted)
psa_rate <- mean(psa$converted)

# 5. Two-proportion hypothesis test
# H0: Ad and PSA conversion rates are equal.
# H1: Ad and PSA conversion rates differ.

test <- prop.test(
  x = c(ad_conversions, psa_conversions),
  n = c(nrow(ad), nrow(psa)),
  alternative = "two.sided",
  correct = FALSE
)

cat("\n--- Hypothesis Test ---\n")
print(test)

# 6. Effect size
difference <- ad_rate - psa_rate
relative_lift <- if (psa_rate > 0) {
  difference / psa_rate * 100
} else {
  NA_real_
}

# Approximate 95% CI for the difference in proportions
se <- sqrt(
  ad_rate * (1 - ad_rate) / nrow(ad) +
    psa_rate * (1 - psa_rate) / nrow(psa)
)

ci <- difference + c(-1, 1) * qnorm(0.975) * se

cat("\n--- Effect Size ---\n")
cat("Ad conversion rate:", round(ad_rate * 100, 3), "%\n")
cat("PSA conversion rate:", round(psa_rate * 100, 3), "%\n")
cat("Absolute difference:", round(difference * 100, 3),
    "percentage points\n")
cat("Relative lift vs PSA:", round(relative_lift, 2), "%\n")
cat("Approx. 95% CI for difference:",
    round(ci[1] * 100, 3), "to",
    round(ci[2] * 100, 3), "percentage points\n")
cat("P-value:", format.pval(test$p.value, digits = 5), "\n")

if (test$p.value < 0.05) {
  cat("Conclusion: statistically significant difference at 5%.\n")
} else {
  cat("Conclusion: insufficient evidence of a difference at 5%.\n")
}

# 7. Plot conversion rates
plot_data <- data.frame(
  Group = c("Ad", "PSA"),
  Rate = c(ad_rate, psa_rate),
  Visitors = c(nrow(ad), nrow(psa))
)

p1 <- ggplot(plot_data, aes(x = Group, y = Rate * 100, fill = Group)) +
  geom_col(width = 0.6) +
  geom_text(
    aes(label = sprintf("%.2f%%", Rate * 100)),
    vjust = -0.5
  ) +
  labs(
    title = "Conversion Rate: Ad vs PSA",
    x = "Experiment group",
    y = "Conversion rate (%)"
  ) +
  theme_minimal() +
  theme(legend.position = "none")

ggsave("conversion_rates.png", p1, width = 8, height = 5)

# 8. Explore conversion by number of ads, if available
if ("total_ads" %in% names(data)) {
  p2 <- ggplot(data, aes(x = total_ads, fill = converted)) +
    geom_histogram(binwidth = 10, position = "identity", alpha = 0.6) +
    coord_cartesian(xlim = c(0, quantile(data$total_ads, 0.99,
                                         na.rm = TRUE))) +
    labs(
      title = "Ad Exposure Distribution by Conversion",
      x = "Number of ads seen",
      y = "Number of users",
      fill = "Converted"
    ) +
    theme_minimal()
  
  ggsave("ad_exposure.png", p2, width = 8, height = 5)
}

# 9. Export results
write.csv(plot_data, "conversion_summary.csv", row.names = FALSE)

metrics <- data.frame(
  Metric = c(
    "Ad conversion rate (%)",
    "PSA conversion rate (%)",
    "Absolute difference (percentage points)",
    "Relative lift (%)",
    "95% CI lower (percentage points)",
    "95% CI upper (percentage points)",
    "P-value"
  ),
  Value = c(
    ad_rate * 100,
    psa_rate * 100,
    difference * 100,
    relative_lift,
    ci[1] * 100,
    ci[2] * 100,
    test$p.value
  )
)

write.csv(metrics, "ab_test_results.csv", row.names = FALSE)

cat("\nFiles saved: conversion_rates.png, conversion_summary.csv,",
    "ab_test_results.csv\n")
if ("total_ads" %in% names(data)) cat("ad_exposure.png\n")
