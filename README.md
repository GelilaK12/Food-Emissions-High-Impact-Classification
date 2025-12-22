# Food Emissions High-Impact Classification
#### Predicting High-Impact Foods Using Lifecycle Emissions

## Project Overview

This project builds an end-to-end machine learning pipeline to predict whether a food product is environmentally high-impact based on lifecycle greenhouse gas emissions. Using emissions from different lifecycle stages such as farm, land use change, and processing, I framed this as a binary classification problem. The goal was not only to achieve reliable predictions but also to maintain interpretability, perform thorough diagnostics, and exercise sound modeling judgment.

I chose this dataset because lifecycle emissions naturally reflect both additive and stage-specific effects, making it a good test case for comparing linear and nonlinear models. The project demonstrates careful workflow management, feature analysis, and evaluation of model behavior in a way that mirrors real-world ML engineering practices.

## Problem Framing

#### Objective:
- Predict whether a food product is in the top quartile of total lifecycle emissions (“High Impact”) using individual stage emissions.

#### Why classification:
- Rather than predicting exact emissions, classification is useful for applications like screening high-risk products, prioritizing environmental interventions, or guiding policy and supply-chain decisions.

#### Target variable:
- High_Impact = 1 if total emissions are greater than or equal to the 75th percentile, otherwise 0.

## Data

- #### Source: [Food production lifecycle emissions dataset](https://www.kaggle.com/datasets/selfvivek/environment-impact-of-food-production/data)

- #### Observations: Individual food products

- #### Features: Lifecycle emission stages

  -Land use change

  -Animal feed

  -Farm

  -Processing

  -Transport

  -Packaging

  -Retail

- The dataset was cleaned to standardize names, convert numeric fields, and handle missing values. Summary statistics and missing value reports are saved in the outputs folder.

## Modeling Approach

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


## Model Performance

- Both models achieved similar results on the test set. Logistic Regression correctly predicted most high-impact foods while maintaining full accuracy on low-impact foods. Random Forest performance was comparable, showing that a simple linear model is sufficient in this context.

## Feature Interpretation

It was important to separate magnitude, correlation, and predictive importance:

- #### Magnitude (EDA): The Farm stage contributes the most to total emissions and is highly variable.

- #### Correlation with target: Processing, Animal Feed, and Land Use Change show the strongest linear relationships with High_Impact.

- #### Predictive importance (Random Forest): Land Use Change provides the most unique information for classification. Permutation tests confirmed that Farm, while large and variable, is partially redundant with other stages.

- I learned that a feature with high emissions is not necessarily the most useful for predicting high-impact foods. Predictive importance highlights which features actually help the model separate classes effectively.

## Diagnostics

#### Key diagnostics included:

- Stratified train/test split

- Confusion matrices and precision/recall evaluation

- Logistic Regression coefficient analysis

- Random Forest impurity-based and permutation-based feature importance

- Boxplots comparing stage emissions across high- and low-impact foods

- Correlation analysis with the target variable

These steps ensure that model results are interpretable, reliable, and defensible.


## Key Takeaways

- High-impact foods can be reliably predicted using lifecycle emissions.

- Model complexity beyond logistic regression did not meaningfully improve performance.

- Feature importance differs from absolute emissions, highlighting the importance of careful diagnostics.

- This project demonstrates end-to-end ML workflow ownership, strong reasoning, and interpretable results, which are core skills for an ML engineer.
