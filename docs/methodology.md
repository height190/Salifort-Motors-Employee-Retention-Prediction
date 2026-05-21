# Methodology

## Framework

This project follows the Google PACE workflow:

- **Plan:** Define the HR retention problem, stakeholders, target variable, risks, and success metrics.
- **Analyze:** Clean the dataset, review data quality, and explore relationships between attrition and employee factors.
- **Construct:** Engineer features, train classification models, and compare performance.
- **Execute:** Evaluate results, interpret model behavior, and develop stakeholder recommendations.

## Data Cleaning

Cleaning steps are implemented in `src/preprocessing.py` and demonstrated in `notebooks/01_data_cleaning.ipynb`.

Key steps:

- Standardize column names to snake_case.
- Rename `time_spend_company` to `tenure`.
- Normalize categorical fields such as department and salary.
- Remove duplicate employee records.
- Review missing values, data types, and outliers.

## Feature Engineering

Feature engineering is implemented in `src/feature_engineering.py`.

The primary engineered feature is `overworked`, based on average monthly hours. This feature gives HR a more interpretable workload signal than using raw monthly hours alone.

The project compares two modeling perspectives:

- A full feature set for performance benchmarking.
- A leakage-aware feature set for a more realistic early-intervention use case.

## Exploratory Data Analysis

EDA focuses on:

- Attrition class balance.
- Correlation patterns.
- Workload and project count.
- Satisfaction by tenure.
- Evaluation score and monthly hours.
- Promotion and salary context.

EDA outputs are saved to `images/eda/`.

## Modeling

The project trains three model families:

- Logistic Regression as an interpretable baseline.
- Decision Tree for nonlinear interpretability.
- Random Forest for stronger classification performance.

Models are evaluated using:

- Precision
- Recall
- F1 score
- Accuracy
- ROC AUC
- Confusion matrix
- ROC curve
- Feature importance

## Model Selection

The leakage-aware Random Forest is selected because it balances strong predictive performance with a more practical HR use case. Recall is prioritized because missing employees who may leave is costly for retention planning.

## Ethical Considerations

Employee attrition predictions can affect people directly. This model should be used only for supportive HR actions such as workload review, career development conversations, and retention planning.

The model should not be used to punish employees, reduce opportunities, or make automated employment decisions.
