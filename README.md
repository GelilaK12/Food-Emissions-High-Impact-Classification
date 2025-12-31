# Food Emissions High-Impact Classification

## Project Overview

- End-to-end machine learning workflow to predict whether a food product is environmentally high-impact based on lifecycle greenhouse gas emissions.
- Uses emissions from production stages such as farm, land use change, processing, transport, packaging, and retail.
- Focus on building reliable and interpretable models while demonstrating strong workflow management, feature analysis, and evaluation.
- Models progress from linear regression to logistic regression, random forest, XGBoost, Multi-Layer Perceptron (MLP) neural network, and Bayesian Logistic Regression Model to compare additive, tree-based, boosted, and deep learning approaches.

## Problem Framing

#### Objective: 
Predict whether a food product is in the top quartile of total lifecycle emissions (“High Impact”) using individual stage emissions.

#### Why classification: 
Rather than predicting exact emissions, classification is useful for applications like screening high-risk products, prioritizing environmental interventions, or guiding policy and supply-chain decisions.

#### Target variable: 
`High_Impact = 1` if total emissions are greater than or equal to the 75th percentile, otherwise 0.

## Data

- #### Source: 
[Food production lifecycle emissions dataset](https://www.kaggle.com/datasets/selfvivek/environment-impact-of-food-production/data)
- #### Observations: 
Individual food products
- #### Features: Lifecycle emission stages
  - Land use change
  - Animal feed
  - Farm
  - Processing
  - Transport
  - Packaging
  - Retail
- The dataset was cleaned to standardize names, convert numeric fields, and handle missing values. Summary statistics and missing value reports are saved in the `outputs` folder.

## Modeling Approach

#### Linear Regression
- Predicted total emissions from lifecycle stages.
- Coefficients were approximately 1.0, confirming total emissions are the sum of the components.
- Used mainly to validate data integrity, not for prediction.

#### Logistic Regression (Baseline Model)
- Provides a simple, interpretable baseline.
- Captures linear, additive effects of each lifecycle stage on the probability of being high-impact.
- Coefficients are used to understand the direction and strength of influence.
- #### Initially, Farm was expected to dominate, but analysis showed other stages also play significant roles.

#### Random Forest Classifier (Nonlinear Benchmark)
- Tests whether nonlinear interactions improve predictions.
- Depth is limited to avoid overfitting, and class weighting addresses imbalance.
- Random Forest helps compare feature importance and capture complex patterns.
- #### Random Forest did not improve much over Logistic Regression, suggesting most predictive signal is additive.

#### XGBoost (Gradient Boosting)
- Captures complex patterns using a boosted ensemble of decision trees.
- #### Workflow includes:
  - Baseline model training
  - Feature importance analysis: built-in, SHAP, and permutation importance
  - Hyperparameter tuning with GridSearchCV
- Key observations:
  - Gain-based importance favors Farm
  - SHAP highlights Land Use Change as strongest positive contributor
  - Permutation importance confirms Farm as highly predictive
  - Slightly worse performance than logistic regression and Random Forest, indicating additive signal dominates

#### Multi-Layer Perceptron (MLP Neural Network)
- Introduces a feed-forward neural network to evaluate whether a deep learning model provides additional predictive power.
- #### Workflow includes:
  - Feature scaling using StandardScaler
  - Baseline MLP with two hidden layers
  - L2 regularization to reduce overfitting
  - Hyperparameter tuning of architecture, activation functions, and regularization strength
  - Calibration curve analysis
  - Permutation feature importance
- Key observations:
  - MLP performance comparable to Logistic Regression and Random Forest
  - Calibration curve shows mild overconfidence
  - Feature importance suggests Packaging and Transport appear more influential
  - Results reinforce additive signal dominates
 
#### Bayesian Logistic Regression Model

- Uses posterior means of coefficients (w) and bias (b) from Bayesian inference to create a deployable scikit-learn-style wrapper.
- Provides interpretable probabilities for high-impact classification.
- Model captures additive effects of all lifecycle stages, with Farm and Land Use Change as strongest contributors.
- Trained once and saved as `deployable_model.pkl` for reproducible predictions.
- Integrated into Prefect ETL and served via FastAPI for automated inference.


## Diagnostics and Evaluation

- Stratified train/test split
- Confusion matrices and evaluation metrics: precision, recall, F1
- Feature importance comparisons to separate magnitude from predictive contribution
- SHAP plots for XGBoost interpretability
- Permutation importance to show unique feature contributions
- Boxplots comparing stage emissions across high-impact and low-impact foods
- Calibration curves for MLP probability reliability
- Neural network training loss curves

## Key Takeaways

- Lifecycle stage emissions can reliably predict high-impact foods.
- Feature importance differs from raw magnitude, highlighting the need for careful diagnostics.
- Additive models like logistic regression capture most of the predictive signal.
- Nonlinear and neural network models add limited improvement.
- Demonstrates end-to-end ML workflow ownership, including preprocessing, modeling, interpretability, and hyperparameter tuning.

## CI/CD Pipeline & Deployment

- Automated **CI/CD pipeline** using **GitHub Actions**:
  - Runs the Prefect ETL pipeline
  - Builds a deployable Bayesian Logistic Regression model
  - Generates predictions and evaluation metrics
  - Builds and pushes a Docker image to Docker Hub
- **Secrets used**:
  - `DOCKERHUB_USERNAME` → Docker Hub username
  - `DOCKERHUB_TOKEN` → Personal access token
- **Prefect ETL pipeline** (`pipeline/etl.py`) handles:
  - Loading and validating dataset
  - Creating `high_impact` label
  - Training and saving deployable Bayesian Logistic model
  - Saving prediction outputs and metrics
- **Docker deployment**:
  - Dockerfile builds image with all dependencies
  - Run API locally:
    
        docker build -t food-emissions-api .
      
        docker run -p 8000:8000 food-emissions-api
  
  - FastAPI serves model via `/predict` endpoint, Swagger UI at `/docs`

## Repository Structure
Food-Emissions-High-Impact-Classification/

    │
    ├─ data/
    │   └─ Food_Production.csv
    ├─ images/
    ├─ api/
    ├─ outputs/
    ├─ scripts/
    │   └─ regular/
    |       ├─  linear_regression.py
    │       ├─ baseline_models.py
    │       ├─ xgboost_analysis.py
    │       ├─ bayesian_logistic
    │       └─ mlp_classifier.py  
    │       └─ export_deployable_model 
    |    └─ artifacts/
    |         ├─  deployable_model.pkl
    │         ├─ posterior_means.pkl
    │         ├─ bayesian_logreg_trace
    |    └─ artifacts/
    |         ├─  deployable_model.pkl
    │         ├─ posterior_means.pkl
    │         ├─ bayesian_logreg_trace
    ├─ pipeline/
    │   ├─ etl.py
    │   └─ validation.py
    ├─ wandb/
    ├─ .github/
    │ └─ workflows/
    │ └─ ci-cd.yml
    ├─ Dockerfile
    ├─ README.md


## Usage


#### Clone the repository:

    git clone https://github.com/GelilaK12/Food-Emissions-High-Impact-Classification.git

Run the Prefect ETL pipeline
      
    python -m pipeline.etl


### Run the API via Docker

docker build -t food-emissions-api .
docker run -p 8000:8000 food-emissions-api


### Access API

Swagger UI: 
        
        http://localhost:8000/docs

Send POST request to /predict with JSON input:

  Swagger UI:
          
          http://localhost:8000/docs

Send POST request to /predict with JSON input:

Returns high_impact prediction (0 or 1)


















