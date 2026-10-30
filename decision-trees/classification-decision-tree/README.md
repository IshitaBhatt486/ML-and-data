# Car Evaluation Decision Tree

## 1. Dataset
- 1,728 records, 6 categorical predictors, and no missing values.
- `unacc` represents 70.02% of records, while `good` and `vgood` each represent about 4%.
- The imbalanced classes mean accuracy alone is insufficient to evaluate the model.

## 2. Decision Tree Rules
The tree primarily uses `persons`, `safety`, `buying`, and `maint`.

- Cars for only 2 people are classified as `unacc`.
- Cars with low safety are classified as `unacc`.
- For cars with 4 or more seats and medium/high safety, buying and maintenance costs determine further classifications.

## 3. Model Performance
- Test accuracy: 86.27%
- 95% confidence interval: 83.00%–89.12%
- Majority-class baseline: 70.21%
- Kappa: 0.7086, indicating substantial agreement beyond chance.

The model performs better than the majority-class baseline, but performance varies across classes.

## 4. Confusion Matrix
- `unacc`: 93.66% sensitivity — performs well.
- `acc`: 74.78% sensitivity — reasonably good.
- `good`: 100% sensitivity, but only 38.46% precision — many predictions are incorrect.
- `vgood`: 0% sensitivity — the model failed to identify any `vgood` examples.

## 5. Feature Importance
1. `safety`: 127.32
2. `persons`: 88.17
3. `maint`: 65.13
4. `buying`: 32.17
5. `doors`: 2.70
6. `lug_boot`: 0.38

Safety and passenger capacity are the most influential features in this fitted tree. 

## Conclusion
The decision tree generates interpretable IF–THEN rules and achieves 86.27% test accuracy.