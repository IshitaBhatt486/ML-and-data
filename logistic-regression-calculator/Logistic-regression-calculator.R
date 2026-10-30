
# Logistic Regression Calculator
# Calculates probabilities using coefficients and the sigmoid function.

# 1. Example model parameters
beta_0 <- -2.0  # Intercept
beta_1 <- 0.8   # Study hours coefficient
beta_2 <- 0.5   # Attendance coefficient

# 2. Input values
study_hours <- 4
attendance <- 3
threshold <- 0.5

# 3. Logistic regression calculations
z <- beta_0 + beta_1 * study_hours + beta_2 * attendance

sigmoid <- function(z) {
  1 / (1 + exp(-z))
}

probability <- sigmoid(z)
predicted_class <- ifelse(probability >= threshold, 1, 0)

# 4. Display results
cat("=== LOGISTIC REGRESSION CALCULATOR ===\n")
cat("Intercept:", beta_0, "\n")
cat("Study hours:", study_hours, "\n")
cat("Attendance score:", attendance, "\n")
cat("Weighted sum (z):", round(z, 4), "\n")
cat("Predicted probability:", round(probability, 4), "\n")
cat("Classification threshold:", threshold, "\n")
cat("Predicted class:", predicted_class, "\n")

# 5. Visualize probability as study hours change
study_range <- seq(0, 10, by = 0.1)

probabilities <- sigmoid(
  beta_0 + beta_1 * study_range + beta_2 * attendance
)

plot(
  study_range, probabilities,
  type = "l",
  lwd = 2,
  col = "blue",
  ylim = c(0, 1),
  xlab = "Study Hours",
  ylab = "Predicted Probability",
  main = "Logistic Regression: Input vs Probability"
)

abline(h = threshold, col = "red", lty = 2)
abline(v = study_hours, col = "darkgreen", lty = 3)

points(study_hours, probability, pch = 19, col = "darkgreen")

legend(
  "bottomright",
  legend = c("Probability curve", "Decision threshold", "Current input"),
  col = c("blue", "red", "darkgreen"),
  lty = c(1, 2, 3),
  pch = c(NA, NA, 19),
  bty = "n"
)

# 6. Explore different classification thresholds
thresholds <- seq(0.1, 0.9, by = 0.1)
classes <- ifelse(probability >= thresholds, 1, 0)

cat("\n=== THRESHOLD EXPERIMENT ===\n")
print(data.frame(
  Threshold = thresholds,
  Probability = round(probability, 4),
  Predicted_Class = classes
))
