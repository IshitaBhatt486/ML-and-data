# Automobile Price Prediction Using Regression Trees

Predicts automobile prices using regression decision trees in R. It compares shallow, deep, and pruned trees to examine the trade-off between model complexity and predictive accuracy.

## Dataset
- Source: UCI Automobile Dataset
- Observations after cleaning: 201
- Target variable: Automobile price
- Features: Engine size, horsepower, curb weight, fuel type, body style, drive wheels, and other vehicle specifications.

## Models
- **Shallow Tree:** Maximum depth of 2
- **Deep Tree:** Maximum depth of 8
- **Pruned Tree:** Reduced complexity using cross-validation-based complexity parameters

## Results

| Model | RMSE | MAE | R² | Leaves |
|---|---:|---:|---:|---:|
| Shallow Tree | 3154 | 2437 | 0.8802 | 3 |
| Deep Tree | 2615 | 1800 | 0.9176 | 19 |
| Pruned Tree | 2627 | 1799 | 0.9169 | 16 |

The deep tree achieved the lowest test RMSE and highest R². The pruned tree achieved almost identical performance with fewer leaves.

## Conclusion
Regression trees predicted automobile prices effectively, with test R² values above 0.88 for all three models. Increasing tree depth improved predictive accuracy, while pruning produced a simpler model with nearly the same performance. The pruned tree offers a reasonable balance between model complexity and accuracy.