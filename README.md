# Food Emissions High-Impact Classification

## Project Overview

- End-to-end machine learning workflow to predict whether a food product is environmentally high-impact based on lifecycle greenhouse gas emissions.

- Uses emissions from production stages such as farm, land use change, processing, transport, packaging, and retail.

- Focus on building reliable and interpretable models while demonstrating strong workflow management, feature analysis, and evaluation.

- Models progress from simple linear regression to logistic regression, random forest, and XGBoost to compare additive, nonlinear, and boosted approaches.

## Problem Framing

#### Objective: Predict whether a food product is in the top quartile of total lifecycle emissions (“High Impact”) using individual stage emissions.

#### Why classification: Rather than predicting exact emissions, classification is useful for applications like screening high-risk products, prioritizing environmental interventions, or guiding policy and supply-chain decisions.

#### Target variable: High_Impact = 1 if total emissions are greater than or equal to the 75th percentile, otherwise 0.

## Data

- #### Source: [Food production lifecycle emissions dataset](https://www.kaggle.com/datasets/selfvivek/environment-impact-of-food-production/data)

- #### Observations: Individual food products

- #### Features: Lifecycle emission stages

  - Land use change
  - Animal feed
  - Farm
  - Processing
  - Transport
  - Packaging
  - Retail

- The dataset was cleaned to standardize names, convert numeric fields, and handle missing values. Summary statistics and missing value reports are saved in the outputs folder.

## Modeling Approach

#### Linear Regression

- Predicted total emissions from lifecycle stages.

- Coefficients were approximately 1.0, confirming total emissions are the sum of the components.
  
- Used mainly to validate data integrity, not for prediction.

#### Logistic Regression (Baseline Model)

- Provides a simple, interpretable baseline.

- Captures linear, additive effects of each lifecycle stage on the probability of being high-impact.

- Coefficients are used to understand the direction and strength of influence.

- #### Initially, I expected Farm to dominate, but analysis showed that other stages also play significant roles in classification.

#### Random Forest Classifier (Nonlinear Benchmark)

- Tests whether nonlinear interactions improve predictions.

- Depth is limited to avoid overfitting, and class weighting addresses imbalance.

- Random Forest helps compare feature importance and capture complex patterns.

- #### I was surprised to see that Random Forest did not improve much over Logistic Regression. This suggests that most of the predictive signal is additive and can be captured with a linear model.

#### XGBoost (Gradient Boosting)

- Captures complex patterns using a boosted ensemble of decision trees.

- #### Workflow includes:

    - Baseline model training

    - Feature importance analysis: built-in, SHAP, and permutation importance
    
    - Hyperparameter tuning with GridSearchCV

Key observations:

  - Gain-based importance favors Farm
  - SHAP highlights Land Use Change as the strongest positive contributor
  - Permutation importance confirms Farm as highly predictive
  - Slightly worse performance than logistic regression and Random Forest, indicating additive signal dominates

## Diagnostics and Evaluation

- Stratified train/test split

- Confusion matrices and evaluation metrics: precision, recall, F1

- Feature importance comparisons to separate magnitude from predictive contribution

- SHAP plots for XGBoost interpretability

- Permutation importance to show unique feature contributions

- Boxplots comparing stage emissions across high-impact and low-impact foods

## Key Takeaways

- Lifecycle stage emissions can reliably predict high-impact foods.

- Feature importance differs from raw magnitude, highlighting the need for careful diagnostics.

- Additive models like logistic regression capture most of the predictive signal.

- Nonlinear models add limited improvement in this dataset.

- Demonstrates end-to-end ML workflow ownership, including preprocessing, modeling, interpretability, and hyperparameter tuning.

## Repository Structure

    Food-Emissions-High-Impact-Classification/
    │
    ├─ data/
    │   └─ Food_Production.csv      
    ├─ images/                        
    ├─ outputs/                       
    ├─ scripts/
    │   ├─ linear_regression.py
    │   ├─ baseline_models.py
    │   └─ xgboost_analysis.py
    ├─ README.md

## Usage
Clone the repository:

    git clone https://github.com/GelilaK12/Food-Emissions-High-Impact-Classification.git
    cd Food-Emissions-High-Impact-Classification

