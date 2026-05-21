# Executive Summary

## Business Problem

Salifort Motors is experiencing employee turnover that creates hiring, onboarding, training, and productivity costs. HR needs to understand which employee factors are most associated with attrition and how to identify at-risk employees early enough to support retention.

## Major Findings

- Attrition is not randomly distributed across the workforce.
- Employees with high project loads and high monthly hours show elevated turnover risk.
- Four-year-tenure employees appear to be an important retention-risk group, suggesting a possible career progression or role-fit issue.
- Satisfaction is highly informative, but it should be treated carefully because it may reflect late-stage disengagement.
- Evaluation scores and workload appear connected, which may indicate that strong performance is being achieved through unsustainable work patterns.

## Model Performance

The selected leakage-aware Random Forest model balanced strong predictive performance with realistic HR use:

| Metric | Score |
|---|---:|
| Precision | 0.870 |
| Recall | 0.904 |
| F1 | 0.887 |
| Accuracy | 0.962 |
| AUC | 0.938 |

Recall was prioritized because missing an employee likely to leave may prevent HR from offering timely support.

## Business Impact

The model and EDA can help HR move from reactive turnover reporting to proactive retention planning. The highest-value use case is not replacing HR judgment, but helping HR prioritize where to review workload, promotion processes, satisfaction trends, and management practices.

## Actionable Recommendations

- Review employees with unusually high project counts before workload becomes unsustainable.
- Audit teams where high evaluation scores are paired with high monthly hours.
- Investigate promotion and career-path friction among three- to five-year-tenure employees.
- Use model outputs to prioritize supportive check-ins, not disciplinary decisions.
- Monitor retention metrics by department and salary band to detect preventable operational issues.
