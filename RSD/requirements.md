# Requirements Specification Document

## 1. Project Overview

### 1.1 Project Objective
To predict customer no-shows for a hotel chain and develop policies to reduce associated expenses through data analysis and machine learning.

```markdown
# Requirements Specification Document (RSD)

## 1. Project Overview

### 1.1 Project Objective
To predict customer no-shows for a hotel chain and develop policies to reduce associated expenses through data analysis and machine learning.

### 1.2 Scope
- Development of predictive models for no-show prediction
- Analysis of factors influencing no-shows
- Implementation of an end-to-end machine learning pipeline (SQLite ingestion → preprocessing → feature engineering → model training/evaluation)
- Creation of policy recommendations based on findings

## 2. Task Requirements (detailed)

### 2.1 Task 1: Exploratory Data Analysis (EDA)

#### Deliverable
- `notebooks/eda.ipynb` (interactive Jupyter notebook)

#### Requirements
- Documented step-by-step process and rationale
- Data quality checks (missing values, duplicates, types)
- Descriptive statistics and distribution analysis
- Correlation and temporal analysis where applicable
- Visualizations (static & interactive) with interpretations
- Clear conclusions and recommendations for modeling and policy

### 2.2 Task 2: Machine Learning Pipeline (MLP)

#### Deliverables
1. `src/` directory with Python modules (.py) for each component
2. Executable `run.sh` script at project root (does not install deps)
3. `requirements.txt` listing dependencies
4. Updated `README.md` explaining pipeline design, usage and rationale

#### Functional Requirements
- Data ingestion via SQLite (or provided connector)
- Preprocessing including missing values, type fixes, scaling/encoding
- Feature engineering based on EDA (e.g., booking lead time, day-of-week, interaction terms)
- Train & evaluate at least 3 different algorithms (e.g., RandomForest, XGBoost, LightGBM)
- Support cross-validation and configurable hyperparameters
- Save trained model(s) and evaluation metrics

#### Non-functional Requirements
- Configurable via `config/params.yaml`, environment variables, or CLI
- Clear logging of steps and metrics
- Modular, testable code structure
- Reasonable runtime and memory use for the dataset size

## 3. README & Documentation Requirements (what README must include)

1. **Personal Information**
   - Full name (as in NRIC)
   - Email address

2. **Folder Structure**
   - Short description of key folders/files (e.g. `src/`, `config/`, `notebooks/`, `models/`)

3. **Execution Instructions**
   - How to set up environment, install deps, run pipeline and notebook
   - How to change parameters (yaml/env/CLI)

4. **Pipeline Flow**
   - High-level description of logical steps and where to change behavior

5. **EDA Summary**
   - Key findings that influenced preprocessing and features (kept short; full EDA in notebook)

6. **Feature Processing Table**
   - A table describing each feature, its type, how it is processed, and any engineering applied

7. **Modeling & Evaluation**
   - Models evaluated and why
   - Metrics used and interpretation
   - Summary of results (table)

## 4. Feature Processing (example table — fill with dataset details)

| Feature | Type | Processing | Engineered Feature(s) | Rationale |
|---------|------|------------|-----------------------|-----------|
| booking_id | numeric | dropped or used as id | - | Not predictive
| booking_date | datetime | parse, derive day_of_week, lead_time | lead_time_days | Lead time often correlates with no-shows
| customer_age | numeric | fill median, scale | age_group | Age may influence reliability
| room_type | categorical | fill mode, label/one-hot encode | room_popularity | Capture demand effect
| no_show | binary (target) | map to 0/1 | - | Target variable

_Note: Replace table rows with actual columns from the dataset._

## 5. Evaluation Metrics (recommended)
- Accuracy: overall correctness (useful baseline but can be misleading with imbalanced data)
- Precision: proportion of predicted no-shows that were correct (important if false positives are costly)
- Recall (Sensitivity): proportion of actual no-shows detected (important to catch as many no-shows as possible)
- F1-score: harmonic mean of precision and recall (balanced metric)
- ROC-AUC: overall separability capability of the model (probabilistic)

## 6. Deliverables & Acceptance Criteria
- `eda.ipynb`: shows clean analysis, charts, interpretation, and conclusions
- `src/`: modular pipeline that runs end-to-end with the provided config
- `run.sh`: executes the pipeline (no package installs inside the script)
- `requirements.txt`: captures versions used
- README & RSD: clear, complete, actionable documentation

## 7. Risks and Mitigations
- Missing or inconsistent data: mitigate with defensive preprocessing and validation steps
- Model performance drift: log metrics and periodically retrain with new data
- Dependency issues: pin versions in `requirements.txt` and document environment setup

## 8. Next Steps (optional enhancements)
- Add experiment tracking (e.g., MLflow) to log runs and metrics
- Add hyperparameter tuning (GridSearchCV / Optuna)
- Add unit tests and CI to validate pipeline on each commit
- Create a lightweight dashboard to review model metrics and feature importances

```