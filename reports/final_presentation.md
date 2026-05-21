# Final Presentation

## Slide 1: Project Title

**Salifort Motors Employee Retention Prediction**  
Google Advanced Data Analytics Capstone

## Slide 2: Business Problem

Employee turnover is costly and disruptive. Salifort Motors needs to understand which employee factors are associated with leaving and how HR can intervene earlier.

## Slide 3: Dataset and Target

- Employee-level HR dataset
- Target variable: `left`
- Key features: satisfaction, evaluation, project count, monthly hours, tenure, promotion history, department, salary

## Slide 4: EDA Findings

- Attrition risk is concentrated in specific workload and tenure patterns.
- High project load and high monthly hours appear repeatedly among employees who left.
- Satisfaction varies meaningfully by tenure and attrition status.

## Slide 5: Modeling Approach

- Binary classification problem
- Models compared: Logistic Regression, Decision Tree, Random Forest
- Selected model: leakage-aware Random Forest

## Slide 6: Model Performance

| Metric | Score |
|---|---:|
| Precision | 0.870 |
| Recall | 0.904 |
| F1 | 0.887 |
| Accuracy | 0.962 |
| AUC | 0.938 |

## Slide 7: Most Important Signals

- Last evaluation
- Number of projects
- Tenure
- Overworked indicator
- Promotion and salary context

## Slide 8: Business Recommendations

- Review workload concentration.
- Investigate mid-tenure career stagnation.
- Audit promotion timing and fairness.
- Separate high evaluation from sustained overwork.
- Use model outputs for support, not discipline.

## Slide 9: Ethical Use

Predictions should trigger supportive conversations and policy review. The model should not be used as the sole basis for employment decisions.

## Slide 10: Next Steps

- Validate on current HR data.
- Add department-level fairness analysis.
- Tune thresholds based on HR outreach capacity.
- Build a dashboard for ongoing retention monitoring.
