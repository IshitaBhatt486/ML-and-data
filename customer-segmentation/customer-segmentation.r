> # Customer Segmentation using K-means
> # Dataset: Mall_Customers.csv
> 
> set.seed(42)
> 
> # 1. Load the dataset
> customers <- read.csv("Mall_Customers.csv",
+                       check.names = FALSE)
> 
> # 2. Inspect and validate the dataset
> cat("Dataset dimensions:", dim(customers), "\n")
Dataset dimensions: 200 5 
> print(head(customers))
  CustomerID Gender Age Annual Income (k$) Spending Score (1-100)
1          1   Male  19                 15                     39
2          2   Male  21                 15                     81
3          3 Female  20                 16                      6
4          4 Female  23                 16                     77
5          5 Female  31                 17                     40
6          6 Female  22                 17                     76
> print(str(customers))
'data.frame':	200 obs. of  5 variables:
 $ CustomerID            : int  1 2 3 4 5 6 7 8 9 10 ...
 $ Gender                : chr  "Male" "Male" "Female" "Female" ...
 $ Age                   : int  19 21 20 23 31 22 35 23 64 30 ...
 $ Annual Income (k$)    : int  15 15 16 16 17 17 18 18 19 19 ...
 $ Spending Score (1-100): int  39 81 6 77 40 76 6 94 3 72 ...
NULL
> cat("\nMissing values:\n")

Missing values:
> print(colSums(is.na(customers)))
            CustomerID                 Gender                    Age     Annual Income (k$) 
                     0                      0                      0                      0 
Spending Score (1-100) 
                     0 
> 
> # Select numerical features
> features <- customers[, c(
+   "Annual Income (k$)",
+   "Spending Score (1-100)"
+ )]
> 
> # Remove incomplete observations if any
> valid <- complete.cases(features)
> customers <- customers[valid, ]
> features <- features[valid, ]
> 
> # 3. Standardize features
> scaled_data <- scale(features)
> 
> # 4. Elbow method to compare cluster counts
> wss <- sapply(1:10, function(k) {
+   kmeans(scaled_data, centers = k, nstart = 25)$tot.withinss
+ })
> 
> plot(1:10, wss,
+      type = "b", pch = 19,
+      xlab = "Number of clusters (K)",
+      ylab = "Within-cluster sum of squares",
+      main = "Elbow Method for K-means")
> 
> # 5. Fit K-means with 5 clusters
> # Five is a starting choice; inspect the elbow plot.
> set.seed(42)
> k <- 5
> 
> model <- kmeans(scaled_data,
+                 centers = k,
+                 nstart = 25,
+                 iter.max = 100)
> 
> customers$Cluster <- factor(model$cluster)
> 
> # 6. Print cluster summaries
> cat("\nCustomers analyzed:", nrow(customers), "\n")

Customers analyzed: 200 
> cat("Number of clusters:", k, "\n")
Number of clusters: 5 
> cat("Within-cluster sum of squares:",
+     round(model$tot.withinss, 2), "\n")
Within-cluster sum of squares: 65.24 
> 
> cat("\nCluster sizes:\n")

Cluster sizes:
> print(table(customers$Cluster))

 1  2  3  4  5 
23 22 39 81 35 
> 
> cat("\nAverage income and spending by cluster:\n")

Average income and spending by cluster:
> print(aggregate(
+   features,
+   by = list(Cluster = customers$Cluster),
+   FUN = mean
+ ))
  Cluster Annual Income (k$) Spending Score (1-100)
1       1           26.30435               20.91304
2       2           25.72727               79.36364
3       3           86.53846               82.12821
4       4           55.29630               49.51852
5       5           88.20000               17.11429
> 
> # 7. Plot customer segments
> cluster_colors <- rainbow(k)
> 
> plot(features[[1]], features[[2]],
+      col = cluster_colors[as.integer(customers$Cluster)],
+      pch = 19,
+      xlab = "Annual Income (thousands of dollars)",
+      ylab = "Spending Score (1-100)",
+      main = "Mall Customer Segmentation")
> 
> legend("topright",
+        legend = paste("Cluster", 1:k),
+        col = cluster_colors,
+        pch = 19,
+        cex = 0.8)
> 
> # 8. Save cluster assignments
> write.csv(customers,
+           "customer_segments.csv",
+           row.names = FALSE)
> 
> cat("\nSaved customer_segments.csv successfully.\n")

Saved customer_segments.csv successfully.