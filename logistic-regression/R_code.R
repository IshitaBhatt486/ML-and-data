churn_data <- read.csv(
  "C:/Users/hp/OneDrive - Shiv Nadar Institution of Eminence/Desktop/github/logistic-regression/telecom_customer_churn_synthetic.csv",
  stringsAsFactors = TRUE
)

# first 6 observations
head(churn_data)

# view data
View(churn_data)

# number of observations and variables
dim(churn_data)

# variable names and datatypes
str(churn_data)

# Convert churn into a labeled binary categorical variable
churn_data$Churn <- factor(
  churn_data$Churn,
  levels = c(0, 1),
  labels = c("Retained", "Churned")
)

# number of retained and churned customers
table(churn_data$Churn)

# summary
summary(churn_data)

# check for missing values
colSums(is.na(churn_data))

# Reproducible random seed
set.seed(123)

# Random 70% training sample
training_index <- sample(
  1:nrow(churn_data),
  size = round(0.70 * nrow(churn_data))
)

# Split the data into training and testing sets
training_data <- churn_data[training_index, ]
testing_data <- churn_data[-training_index, ]

# Split sizes
nrow(training_data)  # Should be 175
nrow(testing_data)   # Should be 75

# Check churn distribution in both datasets
prop.table(table(training_data$Churn))
prop.table(table(testing_data$Churn))

# Logistic regression model using the training data
churn_model <- glm(
  Churn ~ TenureMonths +
    MonthlyCharges +
    ContractType +
    SupportTicketsLast6Months +
    PaymentMethod +
    AvgMonthlyDataGB,
  data = training_data,
  family = binomial
)

# View the model results
summary(churn_model)

# Convert coefficients to odds ratios
exp(coef(churn_model))

# 95% confidence intervals for odds ratios
exp(confint(churn_model))

# Odds ratios
exp(coef(churn_model))

# Generate predicted probabilities of churn for the testing dataset
testing_data$PredictedProbability <- predict(
  churn_model,
  newdata = testing_data,
  type = "response"
)

# Use 0.50 as the classification threshold
threshold <- 0.50

# Convert probabilities into predicted classes
testing_data$PredictedClass <- ifelse(
  testing_data$PredictedProbability >= threshold,
  "Churned",
  "Retained"
)

# Convert predicted classes to a factor with the same levels as Churn
testing_data$PredictedClass <- factor(
  testing_data$PredictedClass,
  levels = c("Retained", "Churned")
)

# View the first few predictions
head(testing_data[, c(
  "CustomerID",
  "Churn",
  "PredictedProbability",
  "PredictedClass"
)])

# Confusion matrix
confusion_matrix <- table(
  Actual = testing_data$Churn,
  Predicted = testing_data$PredictedClass
)

confusion_matrix

# Extract the four outcomes
TN <- confusion_matrix["Retained", "Retained"]
FP <- confusion_matrix["Retained", "Churned"]
FN <- confusion_matrix["Churned", "Retained"]
TP <- confusion_matrix["Churned", "Churned"]

# Calculate performance metrics
accuracy <- (TP + TN) / (TP + TN + FP + FN)

precision <- TP / (TP + FP)

recall <- TP / (TP + FN)

specificity <- TN / (TN + FP)

f1_score <- 2 * (precision * recall) / (precision + recall)

# Display results
metrics <- data.frame(
  Metric = c("Accuracy", "Precision", "Recall (Sensitivity)",
             "Specificity", "F1-score"),
  Value = round(c(accuracy, precision, recall, specificity, f1_score), 3)
)

metrics

# Model-fit criteria

# AIC
AIC(churn_model)

# McFadden's pseudo R-squared
null_model <- glm(
  Churn ~ 1,
  data = training_data,
  family = binomial
)

mcfadden_r2 <- 1 - (
  as.numeric(logLik(churn_model)) /
    as.numeric(logLik(null_model))
)

round(mcfadden_r2, 3)