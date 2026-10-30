
# Regression Decision Tree: Automobile Price Prediction

# 1. Install and load packages
packages <- c("rpart", "rpart.plot", "ggplot2")

for (p in packages) {
  if (!requireNamespace(p, quietly = TRUE)) {
    install.packages(p)
  }
}

library(rpart)
library(rpart.plot)
library(ggplot2)

set.seed(42)

# 2. Load the UCI Automobile dataset
url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"

cols <- c(
  "symboling", "normalized_losses", "make", "fuel_type",
  "aspiration", "num_doors", "body_style", "drive_wheels",
  "engine_location", "wheel_base", "length", "width", "height",
  "curb_weight", "engine_type", "num_cylinders", "engine_size",
  "fuel_system", "bore", "stroke", "compression_ratio",
  "horsepower", "peak_rpm", "city_mpg", "highway_mpg", "price"
)

cars <- read.csv(
  url, header = FALSE, col.names = cols,
  na.strings = "?", stringsAsFactors = FALSE
)

# 3. Select predictors and target
features <- c(
  "make", "fuel_type", "aspiration", "num_doors",
  "body_style", "drive_wheels", "engine_location",
  "wheel_base", "length", "width", "height", "curb_weight",
  "engine_type", "num_cylinders", "engine_size",
  "fuel_system", "horsepower", "peak_rpm",
  "city_mpg", "highway_mpg"
)

cars <- cars[, c(features, "price")]
cars$price <- as.numeric(cars$price)

# Remove rows with missing target values
cars <- cars[!is.na(cars$price), ]

# Convert categorical variables to factors
categorical <- c(
  "make", "fuel_type", "aspiration", "num_doors",
  "body_style", "drive_wheels", "engine_location",
  "engine_type", "num_cylinders", "fuel_system"
)

cars[categorical] <- lapply(cars[categorical], factor)

# Impute missing numeric values using the median
numeric_cols <- setdiff(
  names(cars)[sapply(cars, is.numeric)], "price"
)

for (col in numeric_cols) {
  cars[[col]][is.na(cars[[col]])] <-
    median(cars[[col]], na.rm = TRUE)
}

# Impute missing categorical values using the mode
for (col in categorical) {
  mode_value <- names(which.max(table(cars[[col]])))[1]
  cars[[col]][is.na(cars[[col]])] <- mode_value
  cars[[col]] <- droplevels(cars[[col]])
}

cat("Dataset rows after cleaning:", nrow(cars), "\n")

# 4. Split into training and testing data
train_ids <- sample(
  seq_len(nrow(cars)),
  size = floor(0.8 * nrow(cars))
)

train <- cars[train_ids, ]
test <- cars[-train_ids, ]

# 5. Train three regression trees

# Shallow tree
shallow <- rpart(
  price ~ ., data = train, method = "anova",
  control = rpart.control(
    maxdepth = 2, cp = 0.001, minbucket = 8
  )
)

# Deep tree
deep <- rpart(
  price ~ ., data = train, method = "anova",
  control = rpart.control(
    maxdepth = 8, cp = 0.0005, minbucket = 3
  )
)

# Pruned tree: choose CP using cross-validation
best_cp <- deep$cptable[
  which.min(deep$cptable[, "xerror"]), "CP"
]

pruned <- prune(deep, cp = best_cp)

# 6. Evaluate model performance
evaluate <- function(model, name) {
  predicted <- predict(model, newdata = test)
  actual <- test$price

  rmse <- sqrt(mean((actual - predicted)^2))
  mae <- mean(abs(actual - predicted))

  r2 <- 1 - sum((actual - predicted)^2) /
    sum((actual - mean(actual))^2)

  data.frame(
    Model = name,
    RMSE = rmse,
    MAE = mae,
    R_squared = r2,
    Leaves = sum(model$frame$var == "<leaf>")
  )
}

results <- rbind(
  evaluate(shallow, "Shallow Tree"),
  evaluate(deep, "Deep Tree"),
  evaluate(pruned, "Pruned Tree")
)

results <- results[order(results$RMSE), ]

cat("\nMODEL COMPARISON\n")
print(results, row.names = FALSE, digits = 4)

write.csv(results, "model_comparison.csv", row.names = FALSE)

# Select model with lowest test RMSE
models <- list(
  "Shallow Tree" = shallow,
  "Deep Tree" = deep,
  "Pruned Tree" = pruned
)

best_name <- results$Model[1]
best_model <- models[[best_name]]

cat("\nBest model by test RMSE:", best_name, "\n")

# 7. Plot the selected regression tree
png("regression_tree.png", width = 1400, height = 900, res = 150)

rpart.plot(
  best_model,
  type = 2,
  extra = 101,
  fallen.leaves = TRUE,
  main = paste("Automobile Price Prediction:", best_name)
)

dev.off()

# 8. Actual vs predicted prices
predictions <- data.frame(
  Actual = test$price,
  Predicted = as.numeric(predict(best_model, newdata = test))
)

predictions$Error <- predictions$Actual - predictions$Predicted

write.csv(
  predictions, "test_predictions.csv", row.names = FALSE
)

p <- ggplot(predictions, aes(x = Actual, y = Predicted)) +
  geom_point(alpha = 0.8) +
  geom_abline(
    intercept = 0, slope = 1,
    linetype = "dashed"
  ) +
  labs(
    title = "Actual vs Predicted Car Prices",
    x = "Actual Price",
    y = "Predicted Price"
  ) +
  theme_minimal()

ggsave(
  "actual_vs_predicted.png", p,
  width = 7, height = 5, dpi = 160
)

# 9. Feature importance
importance <- best_model$variable.importance

if (!is.null(importance) && length(importance) > 0) {
  importance_df <- data.frame(
    Feature = names(importance),
    Importance = as.numeric(importance)
  )

  importance_df <- importance_df[
    order(importance_df$Importance, decreasing = TRUE), ]

  write.csv(
    importance_df, "feature_importance.csv",
    row.names = FALSE
  )

  p2 <- ggplot(
    head(importance_df, 10),
    aes(x = reorder(Feature, Importance), y = Importance)
  ) +
    geom_col() +
    coord_flip() +
    labs(
      title = "Top 10 Important Features",
      x = "Feature",
      y = "Importance"
    ) +
    theme_minimal()

  ggsave(
    "feature_importance.png", p2,
    width = 7, height = 5, dpi = 160
  )
}

# 10. Final interpretation
cat("\nINTERPRETATION GUIDE\n")
cat("Lower RMSE and MAE indicate better predictions.\n")
cat("Higher R-squared generally indicates a better fit.\n")
cat("A deeper tree may overfit the training data.\n")
cat("Pruning reduces unnecessary branches.\n")
cat("Feature importance indicates predictive usefulness, not causation.\n")

cat("\nFiles generated successfully if no errors occurred.\n")
