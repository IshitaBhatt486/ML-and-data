# Customer Segmentation using K-means

Customer segmentation using K-means clustering in R on the Mall Customers dataset. Groups customers based on annual income and spending score to identify distinct purchasing profiles to identify customer segments that businesses can use to develop targeted marketing campaigns, personalized promotions, and customer engagement strategies.

## Dataset

- **Source:** Mall Customers dataset
- **Observations:** 200 customers
- **Features used:** Annual Income (k$), Spending Score (1–100)
- **Missing values:** 0 in all columns
- **Preprocessing:** Standardization using `scale()`

## Methodology

1. Loaded and inspected the dataset.
2. Selected annual income and spending score as clustering features.
3. Standardized the features to give them comparable scales.
4. Used the elbow method to examine candidate cluster counts from 1 to 10.
5. Applied K-means with 5 clusters, 25 random initializations, and a fixed random seed.
6. Calculated cluster sizes and average feature values.
7. Visualized the clusters and exported customer assignments to CSV.

## Results

- Customers analyzed: 200
- Clusters: 5
- Within-cluster sum of squares: 65.24 (standardized feature space)

### Clusters

| Cluster | Customers | Avg. Income (k$) | Avg. Spending Score |
|---|---:|---:|---:|
| 1 | 23 | 26.30 | 20.91 |
| 2 | 22 | 25.73 | 79.36 |
| 3 | 39 | 86.54 | 82.13 |
| 4 | 81 | 55.30 | 49.52 |
| 5 | 35 | 88.20 | 17.11 |

## Graph Interpretation

The scatter plot shows five customer groups according to annual income and spending score.

- **Cluster 1 — Lower-income, low-spending customers:** 23 customers with average income of $26.30k and spending score of 20.91. Affordable offers and introductory promotions may help attract these customers.
- **Cluster 2 — Lower-income, high-spending customers:** 22 customers with average income of $25.73k and spending score of 79.36. Loyalty rewards and value-oriented products may help retain this group.
- **Cluster 3 — High-income, high-spending customers:** 39 customers with average income of $86.54k and spending score of 82.13. These customers are potential targets for premium products and personalized offers.
- **Cluster 4 — Moderate-income, moderate-spending customers:** 81 customers with average income of $55.30k and spending score of 49.52. This is the largest segment and may respond to targeted promotions and product recommendations.
- **Cluster 5 — High-income, low-spending customers:** 35 customers with average income of $88.20k and spending score of 17.11. Exclusive experiences or more relevant product recommendations could be tested to increase engagement.

The five groups reveal different combinations of income and spending behavior. Income alone does not explain customer spending, making segmentation useful for designing differentiated marketing strategies.

## Conclusion

K-means clustering partitioned 200 customers into five groups with distinct average income and spending characteristics. The analysis identified high-income, high-spending customers, lower-income customers with contrasting spending behavior, and a large middle segment.

The model's within-cluster sum of squares was 65.24 on standardized features. This measures the total squared distance from customers to their assigned cluster centers.

## Limitations

- K-means requires the number of clusters to be specified in advance.
- Results depend on the selected features, scaling, and cluster initialization.
- The analysis uses only income and spending score; age and gender were not included.