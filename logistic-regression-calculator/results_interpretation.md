# Results and Graph Interpretation

## Numerical results
- **Weighted sum (`z`): 2.7** — the combined value calculated from the intercept, study hours, and attendance.
- **Predicted probability: 0.937** — the sigmoid converts `z = 2.7` into approximately 93.7%.
- **Predicted class: 1** — the probability is above the selected threshold of 0.5.

## Graph interpretation
The blue curve rises as study hours increase because the study-hours coefficient is positive (`0.8`). Attendance is held constant at 3, so the graph isolates the effect of study hours in this example.

The red horizontal line marks the 0.5 classification threshold. The green vertical line and point mark the current input of 4 study hours and its predicted probability.

## Threshold experiment
The probability remains `0.937` for every threshold because the inputs and model coefficients do not change. Only the rule used to assign the class changes.

Since `0.937` is greater than all thresholds from 0.1 through 0.8—and also 0.9—the predicted class should be `1` at every threshold in that range.

**Small output detail:** R may omit `0.9` from `seq(0.1, 0.9, by = 0.1)` due to floating-point rounding. Use `round(seq(0.1, 0.9, by = 0.1), 1)` to ensure the printed threshold list includes 0.9.

## Conclusion
With the chosen illustrative coefficients, both study hours and attendance increase the model's weighted sum, and the resulting probability is above the classification threshold. This shows how logistic regression produces a probability and how a threshold converts it into a class.
