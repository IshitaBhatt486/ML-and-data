# Car Evaluation: Decision Tree Rule Generator

library(rpart)
library(rpart.plot)
library(caret)

set.seed(42)

# 1. Load dataset
url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"

car <- read.csv(
  url,
  header = FALSE,
  stringsAsFactors = FALSE
)

colnames(car) <- c(
  "buying", "maint", "doors",
  "persons", "lug_boot", "safety", "class"
)

car[] <- lapply(car, factor)

# 2. Dataset summary
cat("====================================\n")
cat("       CAR EVALUATION DATASET\n")
cat("====================================\n")

cat("\nDATASET OVERVIEW\n")
cat("Source: UCI Machine Learning Repository\n")
cat("Total records:", nrow(car), "\n")
cat("Total columns:", ncol(car), "\n")
cat("Predictor variables:", ncol(car) - 1, "\n")
cat("Missing values:", sum(is.na(car)), "\n")

cat("\nDATASET DESCRIPTION\n")
cat("The dataset evaluates car acceptability using six attributes.\n")
cat("All predictors are categorical.\n")
cat("The target variable is the car acceptability class.\n")

# 3. Variable descriptions
cat("\nVARIABLES\n")
cat("buying   : Buying price\n")
cat("maint    : Maintenance cost\n")
cat("doors    : Number of doors\n")
cat("persons  : Passenger capacity\n")
cat("lug_boot : Luggage boot size\n")
cat("safety   : Estimated safety level\n")
cat("class    : Overall acceptability\n")

# 4. Class distribution
cat("\nTARGET CLASS DISTRIBUTION\n")
print(table(car$class))

cat("\nCLASS PERCENTAGES\n")
print(round(prop.table(table(car$class)) * 100, 2))

barplot(
  table(car$class),
  col = "steelblue",
  main = "Distribution of Car Acceptability",
  xlab = "Acceptability Class",
  ylab = "Number of Cars"
)

# 5. Distribution of predictor categories
cat("\nPREDICTOR CATEGORY DISTRIBUTIONS\n")

for (variable in names(car)[1:6]) {
  cat("\n", variable, "\n")
  print(table(car[[variable]]))
}

# 6. Train/test split
idx <- createDataPartition(
  car$class,
  p = 0.7,
  list = FALSE
)

train <- car[idx, ]
test <- car[-idx, ]

cat("\nTRAINING RECORDS:", nrow(train), "\n")
cat("TESTING RECORDS:", nrow(test), "\n")

# 7. Train decision tree
tree <- rpart(
  class ~ .,
  data = train,
  method = "class",
  control = rpart.control(
    maxdepth = 4,
    minsplit = 20,
    cp = 0.005
  )
)

# 8. Visualize tree
rpart.plot(
  tree,
  type = 3,
  extra = 104,
  fallen.leaves = TRUE,
  main = "Car Acceptability Decision Tree"
)

# 9. Extract rules
cat("\nDECISION TREE RULES\n")
print(tree)

cat("\nREADABLE IF-THEN RULES\n")
print(rpart.rules(tree, cover = TRUE))

# 10. Evaluate model
pred <- predict(tree, test, type = "class")

cat("\nCONFUSION MATRIX\n")
cm <- confusionMatrix(pred, test$class)
print(cm)

cat("\nTEST ACCURACY\n")
print(round(cm$overall["Accuracy"], 4))

# 11. Feature importance
cat("\nFEATURE IMPORTANCE\n")
importance <- sort(
  tree$variable.importance,
  decreasing = TRUE
)
print(importance)

if (length(importance) > 0) {
  barplot(
    importance,
    las = 2,
    col = "steelblue",
    main = "Decision Tree Feature Importance",
    ylab = "Importance"
  )
}

# 12. Show example predictions
cat("\n=== SAMPLE PREDICTIONS ===\n")

results <- data.frame(
  test[, c("buying", "maint", "persons", "safety")],
  Actual = test$class,
  Predicted = pred
)

print(head(results, 10))