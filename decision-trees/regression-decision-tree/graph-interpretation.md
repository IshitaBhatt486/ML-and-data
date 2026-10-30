# Graph Interpretation

## 1. Regression Tree (`regression_tree.png`)

The regression tree predicts automobile prices by splitting vehicles according to their specifications. Each internal node represents a decision rule, and each terminal leaf provides a predicted price.

- The root split identifies an important feature for separating vehicles into different price groups.
- Subsequent splits create increasingly specific groups.
- A deeper tree captures more complex patterns but can become harder to interpret and may overfit.

The deep tree contained 19 leaves, compared with 16 for the pruned tree and 3 for the shallow tree.

## 2. Actual vs. Predicted Prices (`actual_vs_predicted.png`)

This plot compares actual automobile prices with the prices predicted by the selected model.

- Points close to the diagonal represent accurate predictions.
- Points above the diagonal indicate overprediction.
- Points below the diagonal indicate underprediction.
- Large deviations from the diagonal indicate larger prediction errors.

## 3. Feature Importance (`feature_importance.png`)

This graph ranks the input variables according to their contribution to the regression tree's predictions.

- Higher importance indicates that a feature contributed more to the model's splits.
- Lower importance indicates a smaller contribution to the fitted tree.
- Importance does not establish causation or guarantee that a feature independently determines price.

## 4. Model Comparison

| Model | RMSE | MAE | R² |
|---|---:|---:|---:|
| Shallow Tree | 3154 | 2437 | 0.8802 |
| Deep Tree | 2615 | 1800 | 0.9176 |
| Pruned Tree | 2627 | 1799 | 0.9169 |

- **Shallow tree:** Simplest model, but it produced the largest prediction errors.
- **Deep tree:** Lowest RMSE and highest R², giving it the best test performance among the three models.
- **Pruned tree:** Nearly matched the deep tree while using three fewer leaves. Its MAE was also marginally lower.

## 5. Overall Conclusion

The results show that tree complexity affects predictive performance. The shallow tree may underfit the data, while the deep tree captures more detailed relationships. Pruning reduced complexity with almost no loss in accuracy.

The deep tree was best according to RMSE, while the pruned tree provided a simpler alternative with comparable error.