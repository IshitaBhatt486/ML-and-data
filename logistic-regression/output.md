# Outputs

## 1. Load the Dataset

```r
churn_data <- read.csv(
  "C:/Users/hp/OneDrive - Shiv Nadar Institution of Eminence/Desktop/github/logistic-regression/telecom_customer_churn_synthetic.csv",
  stringsAsFactors = TRUE
)
```

## 2. First Six Observations

```r
head(churn_data)
```

```text
  CustomerID TenureMonths MonthlyCharges   ContractType SupportTicketsLast6Months
1      C0001           13          31.18 Month-to-month                          1
2      C0002           20          95.76       Two year                          1
3      C0003           37         108.48        One year                          3
4      C0004           28          71.02        One year                          4
5      C0005           50          68.44        One year                          4
6      C0006            2          46.54        One year                          5

       PaymentMethod AvgMonthlyDataGB Churn
1       Mailed check             41.4     0
2      Bank transfer            112.9     0
3   Electronic check             33.7     0
4        Credit card             79.5     0
5   Electronic check             54.0     0
6        Credit card             25.3     0
```

## 3. View the Dataset

```r
View(churn_data)
```

## 4. Number of Observations and Variables

```r
dim(churn_data)
```

**Output:**

```text
[1] 250 8
```

## 5. Variable Names and Data Types

```r
str(churn_data)
```

**Output:**

```text
'data.frame': 250 obs. of 8 variables:
 $ CustomerID               : Factor w/ 250 levels "C0001","C0002",..: 1 2 3 4 5 6 7 8 9 10 ...
 $ TenureMonths              : int 13 20 37 28 50 2 72 41 18 8 ...
 $ MonthlyCharges            : num 31.2 95.8 108.5 71 68.4 ...
 $ ContractType              : Factor w/ 3 levels "Month-to-month",..: 1 3 2 2 2 2 1 1 2 3 ...
 $ SupportTicketsLast6Months : int 1 1 3 4 4 5 2 0 2 2 ...
 $ PaymentMethod             : Factor w/ 4 levels "Bank transfer",..: 4 1 3 2 3 2 2 3 3 3 ...
 $ AvgMonthlyDataGB           : num 41.4 112.9 33.7 79.5 54 ...
 $ Churn                     : int 0 0 0 0 0 0 0 0 0 0 ...
```

## 6. Convert Churn into a Labeled Binary Categorical Variable

```r
churn_data$Churn <- factor(
  churn_data$Churn,
  levels = c(0, 1),
  labels = c("Retained", "Churned")
)
```

## 7. Number of Retained and Churned Customers

```r
table(churn_data$Churn)
```

**Output:**

```text
Retained Churned
     201      49
```

## 8. Summary Statistics

```r
summary(churn_data)
```

**Output:**

```text
   CustomerID   TenureMonths   MonthlyCharges         ContractType SupportTicketsLast6Months
 C0001  :  1   Min.   : 1.00   Min.   : 25.00   Month-to-month:116   Min.   :0.000
 C0002  :  1   1st Qu.: 7.25   1st Qu.: 60.27   One year      : 55   1st Qu.:1.000
 C0003  :  1   Median :17.00   Median : 76.31   Two year      : 79   Median :2.000
 C0004  :  1   Mean   :21.50   Mean   : 76.09                       Mean   :1.972
 C0005  :  1   3rd Qu.:30.00   3rd Qu.: 91.37                       3rd Qu.:3.000
 C0006  :  1   Max.   :72.00   Max.   :132.52                       Max.   :8.000
 (Other):244

          PaymentMethod AvgMonthlyDataGB      Churn
 Bank transfer   :60    Min.   :  2.00   Retained:201
 Credit card     :45    1st Qu.: 35.90   Churned : 49
 Electronic check:93    Median : 54.00
 Mailed check    :52    Mean   : 57.82
                        3rd Qu.: 79.97
                        Max.   :144.10
```

## 9. Check for Missing Values

```r
colSums(is.na(churn_data))
```

**Output:**

```text
CustomerID                0
TenureMonths              0
MonthlyCharges             0
ContractType               0
SupportTicketsLast6Months  0
PaymentMethod              0
AvgMonthlyDataGB            0
Churn                      0
```

## 10. Set a Reproducible Random Seed

```r
set.seed(123)
```

## 11. Create a Random 70% Training Sample

```r
training_index <- sample(
  1:nrow(churn_data),
  size = round(0.70 * nrow(churn_data))
)
```

## 12. Split the Data into Training and Testing Sets

```r
training_data <- churn_data[training_index, ]
testing_data <- churn_data[-training_index, ]
```

## 13. Check the Split Sizes

```r
nrow(training_data)  # Should be 175
nrow(testing_data)   # Should be 75
```

**Output:**

```text
[1] 175
[1] 75
```

## 14. Check Churn Distribution in Both Datasets

```r
prop.table(table(training_data$Churn))
```

**Training Data Output:**

```text
Retained   Churned
0.8171429  0.1828571
```

```r
prop.table(table(testing_data$Churn))
```

**Testing Data Output:**

```text
Retained   Churned
0.7733333  0.2266667
```

## 15. Build the Logistic Regression Model

```r
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
```

## 16. View the Model Results

```r
summary(churn_model)
```

**Output:**

```text
Call:
glm(formula = Churn ~ TenureMonths + MonthlyCharges + ContractType +
    SupportTicketsLast6Months + PaymentMethod + AvgMonthlyDataGB,
    family = binomial, data = training_data)

Coefficients:
                            Estimate Std. Error z value Pr(>|z|)
(Intercept)                 -3.468116   1.321062  -2.625 0.008659 **
TenureMonths                -0.044981   0.017052  -2.638 0.008343 **
MonthlyCharges                0.050836   0.013391   3.796 0.000147 ***
ContractTypeOne year         -2.215824   0.694243  -3.192 0.001414 **
ContractTypeTwo year         -1.829436   0.648622  -2.820 0.004795 **
SupportTicketsLast6Months     0.451878   0.185116   2.441 0.014644 *
PaymentMethodCredit card     -0.441523   0.748877  -0.590 0.555472
PaymentMethodElectronic check 0.168980   0.616704   0.274 0.784081
PaymentMethodMailed check     0.554606   0.698966   0.793 0.427506
AvgMonthlyDataGB             -0.028158   0.008668  -3.249 0.001160 **

---
Signif. codes:
0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

(Dispersion parameter for binomial family taken to be 1)

    Null deviance: 166.49  on 174  degrees of freedom
Residual deviance: 119.47  on 165  degrees of freedom
AIC: 139.47

Number of Fisher Scoring iterations: 6
```