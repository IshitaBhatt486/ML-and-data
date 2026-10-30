> # Car Evaluation: Decision Tree Rule Generator
> 
> library(rpart)
> library(rpart.plot)
> library(caret)
Loading required package: ggplot2
Loading required package: lattice
> 
> set.seed(42)
> 
> # 1. Load dataset
> url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"
> 
> car <- read.csv(
+   url,
+   header = FALSE,
+   stringsAsFactors = FALSE
+ )
> 
> colnames(car) <- c(
+   "buying", "maint", "doors",
+   "persons", "lug_boot", "safety", "class"
+ )
> 
> car[] <- lapply(car, factor)
> 
> # 2. Dataset summary
> cat("====================================\n")
====================================
> cat("       CAR EVALUATION DATASET\n")
       CAR EVALUATION DATASET
> cat("====================================\n")
====================================
> 
> cat("\nDATASET OVERVIEW\n")

DATASET OVERVIEW
> cat("Source: UCI Machine Learning Repository\n")
Source: UCI Machine Learning Repository
> cat("Total records:", nrow(car), "\n")
Total records: 1728 
> cat("Total columns:", ncol(car), "\n")
Total columns: 7 
> cat("Predictor variables:", ncol(car) - 1, "\n")
Predictor variables: 6 
> cat("Missing values:", sum(is.na(car)), "\n")
Missing values: 0 
> 
> cat("\nDATASET DESCRIPTION\n")

DATASET DESCRIPTION
> cat("The dataset evaluates car acceptability using six attributes.\n")
The dataset evaluates car acceptability using six attributes.
> cat("All predictors are categorical.\n")
All predictors are categorical.
> cat("The target variable is the car acceptability class.\n")
The target variable is the car acceptability class.
> 
> # 3. Variable descriptions
> cat("\nVARIABLES\n")

VARIABLES
> cat("buying   : Buying price\n")
buying   : Buying price
> cat("maint    : Maintenance cost\n")
maint    : Maintenance cost
> cat("doors    : Number of doors\n")
doors    : Number of doors
> cat("persons  : Passenger capacity\n")
persons  : Passenger capacity
> cat("lug_boot : Luggage boot size\n")
lug_boot : Luggage boot size
> cat("safety   : Estimated safety level\n")
safety   : Estimated safety level
> cat("class    : Overall acceptability\n")
class    : Overall acceptability
> 
> # 4. Class distribution
> cat("\nTARGET CLASS DISTRIBUTION\n")

TARGET CLASS DISTRIBUTION
> print(table(car$class))

  acc  good unacc vgood 
  384    69  1210    65 
> 
> cat("\nCLASS PERCENTAGES\n")

CLASS PERCENTAGES
> print(round(prop.table(table(car$class)) * 100, 2))

  acc  good unacc vgood 
22.22  3.99 70.02  3.76 
> 
> barplot(
+   table(car$class),
+   col = "steelblue",
+   main = "Distribution of Car Acceptability",
+   xlab = "Acceptability Class",
+   ylab = "Number of Cars"
+ )
> 
> # 5. Distribution of predictor categories
> cat("\nPREDICTOR CATEGORY DISTRIBUTIONS\n")

PREDICTOR CATEGORY DISTRIBUTIONS
> 
> for (variable in names(car)[1:6]) {
+   cat("\n", variable, "\n")
+   print(table(car[[variable]]))
+ }

 buying 

 high   low   med vhigh 
  432   432   432   432 

 maint 

 high   low   med vhigh 
  432   432   432   432 

 doors 

    2     3     4 5more 
  432   432   432   432 

 persons 

   2    4 more 
 576  576  576 

 lug_boot 

  big   med small 
  576   576   576 

 safety 

high  low  med 
 576  576  576 
> 
> # 6. Train/test split
> idx <- createDataPartition(
+   car$class,
+   p = 0.7,
+   list = FALSE
+ )
> 
> train <- car[idx, ]
> test <- car[-idx, ]
> 
> cat("\nTRAINING RECORDS:", nrow(train), "\n")

TRAINING RECORDS: 1211 
> cat("TESTING RECORDS:", nrow(test), "\n")
TESTING RECORDS: 517 
> 
> # 7. Train decision tree
> tree <- rpart(
+   class ~ .,
+   data = train,
+   method = "class",
+   control = rpart.control(
+     maxdepth = 4,
+     minsplit = 20,
+     cp = 0.005
+   )
+ )
> 
> # 8. Visualize tree
> rpart.plot(
+   tree,
+   type = 3,
+   extra = 104,
+   fallen.leaves = TRUE,
+   main = "Car Acceptability Decision Tree"
+ )
> 
> # 9. Extract rules
> cat("\nDECISION TREE RULES\n")

DECISION TREE RULES
> print(tree)
n= 1211 

node), split, n, loss, yval, (yprob)
      * denotes terminal node

 1) root 1211 364 unacc (0.22213047 0.04046243 0.69942197 0.03798514)  
   2) persons=4,more 802 364 unacc (0.33541147 0.06109726 0.54613466 0.05735661)  
     4) safety=high,med 541 272 acc (0.49722736 0.09057301 0.32717190 0.08502773)  
       8) buying=low,med 278 132 acc (0.52517986 0.17625899 0.13309353 0.16546763)  
        16) maint=high,vhigh 138  36 acc (0.73913043 0.00000000 0.21014493 0.05072464) *
        17) maint=low,med 140  91 good (0.31428571 0.35000000 0.05714286 0.27857143) *
       9) buying=high,vhigh 263 123 unacc (0.46768061 0.00000000 0.53231939 0.00000000)  
        18) maint=low,med 131  34 acc (0.74045802 0.00000000 0.25954198 0.00000000) *
        19) maint=high,vhigh 132  26 unacc (0.19696970 0.00000000 0.80303030 0.00000000) *
     5) safety=low 261   0 unacc (0.00000000 0.00000000 1.00000000 0.00000000) *
   3) persons=2 409   0 unacc (0.00000000 0.00000000 1.00000000 0.00000000) *
> 
> cat("\nREADABLE IF-THEN RULES\n")

READABLE IF-THEN RULES
> print(rpart.rules(tree, cover = TRUE))
 class   acc good unac vgoo                                                                                                         cover
   acc [ .74  .00  .21  .05] when persons is 4 or more & safety is high or med & buying is    low or med & maint is high or vhigh     11%
   acc [ .74  .00  .26  .00] when persons is 4 or more & safety is high or med & buying is high or vhigh & maint is    low or med     11%
  good [ .31  .35  .06  .28] when persons is 4 or more & safety is high or med & buying is    low or med & maint is    low or med     12%
 unacc [ .20  .00  .80  .00] when persons is 4 or more & safety is high or med & buying is high or vhigh & maint is high or vhigh     11%
 unacc [ .00  .00 1.00  .00] when persons is 4 or more & safety is         low                                                        22%
 unacc [ .00  .00 1.00  .00] when persons is         2                                                                                34%
> 
> # 10. Evaluate model
> pred <- predict(tree, test, type = "class")
> 
> cat("\nCONFUSION MATRIX\n")

CONFUSION MATRIX
> cm <- confusionMatrix(pred, test$class)
> print(cm)
Confusion Matrix and Statistics

          Reference
Prediction acc good unacc vgood
     acc    86    0    23     6
     good   19   20     0    13
     unacc  10    0   340     0
     vgood   0    0     0     0

Overall Statistics
                                        
               Accuracy : 0.8627        
                 95% CI : (0.83, 0.8912)
    No Information Rate : 0.7021        
    P-Value [Acc > NIR] : < 2.2e-16     
                                        
                  Kappa : 0.7086        
                                        
 Mcnemar's Test P-Value : NA            

Statistics by Class:

                     Class: acc Class: good Class: unacc Class: vgood
Sensitivity              0.7478     1.00000       0.9366      0.00000
Specificity              0.9279     0.93561       0.9351      1.00000
Pos Pred Value           0.7478     0.38462       0.9714          NaN
Neg Pred Value           0.9279     1.00000       0.8623      0.96325
Prevalence               0.2224     0.03868       0.7021      0.03675
Detection Rate           0.1663     0.03868       0.6576      0.00000
Detection Prevalence     0.2224     0.10058       0.6770      0.00000
Balanced Accuracy        0.8378     0.96781       0.9359      0.50000
> 
> cat("\nTEST ACCURACY\n")

TEST ACCURACY
> print(round(cm$overall["Accuracy"], 4))
Accuracy 
  0.8627 
> 
> # 11. Feature importance
> cat("\nFEATURE IMPORTANCE\n")

FEATURE IMPORTANCE
> importance <- sort(
+   tree$variable.importance,
+   decreasing = TRUE
+ )
> print(importance)
     safety     persons       maint      buying       doors    lug_boot 
127.3246310  88.1713614  65.1335084  32.1716427   2.6991214   0.3810385 
> 
> if (length(importance) > 0) {
+   barplot(
+     importance,
+     las = 2,
+     col = "steelblue",
+     main = "Decision Tree Feature Importance",
+     ylab = "Importance"
+   )
+ }
> 
> # 12. Show example predictions
> cat("\n=== SAMPLE PREDICTIONS ===\n")

=== SAMPLE PREDICTIONS ===
> 
> results <- data.frame(
+   test[, c("buying", "maint", "persons", "safety")],
+   Actual = test$class,
+   Predicted = pred
+ )
> 
> print(head(results, 10))
   buying maint persons safety Actual Predicted
2   vhigh vhigh       2    med  unacc     unacc
4   vhigh vhigh       2    low  unacc     unacc
5   vhigh vhigh       2    med  unacc     unacc
9   vhigh vhigh       2   high  unacc     unacc
20  vhigh vhigh    more    med  unacc     unacc
23  vhigh vhigh    more    med  unacc     unacc
24  vhigh vhigh    more   high  unacc     unacc
28  vhigh vhigh       2    low  unacc     unacc
29  vhigh vhigh       2    med  unacc     unacc
31  vhigh vhigh       2    low  unacc     unacc