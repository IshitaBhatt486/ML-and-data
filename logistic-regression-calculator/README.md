# Logistic Regression Calculator

This R script demonstrates how logistic regression converts input values into a probability and then assigns a class using a chosen threshold.

## How it works
1. **Model parameters:** `beta_0`, `beta_1`, and `beta_2` are illustrative coefficients. They have not been learned from real data.
2. **Weighted sum:** The script combines the inputs and coefficients:
   `z = -2 + (0.8 × study_hours) + (0.5 × attendance)`
3. **Sigmoid function:** It converts `z` into a value between 0 and 1:
   `p = 1 / (1 + exp(-z))`
4. **Classification:** If the probability is at least the threshold (0.5), the predicted class is `1`; otherwise, it is `0`.
5. **Plot:** The graph shows how predicted probability changes with study hours while attendance stays fixed at 3.