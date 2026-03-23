# Wine Quality Prediction — Project Plan

## 1. Problem Overview

| Item | Detail |
|------|--------|
| **Goal** | Predict the `quality` score (0–10, discrete) of wine from physicochemical features |
| **Dataset** | 5,198 training observations, 13 columns (11 chemical inputs + `is_red` indicator + `quality` target) |
| **Evaluation** | Model will be scored on a held-out `test.csv` |
| **Nature** | Ordinal / multi-class classification (7 observed classes: 3–9, heavily imbalanced toward 5 and 6) |

### Target Distribution (Training Set)

| Quality | Count | Proportion |
|---------|------:|----------:|
| 3       |    20 |   0.4%    |
| 4       |   178 |   3.4%    |
| 5       | 1,701 |  32.7%    |
| 6       | 2,269 |  43.7%    |
| 7       |   867 |  16.7%    |
| 8       |   158 |   3.0%    |
| 9       |     5 |   0.1%    |

### Feature List

| # | Feature | Type |
|---|---------|------|
| 1 | fixed acidity | continuous |
| 2 | volatile acidity | continuous |
| 3 | citric acid | continuous |
| 4 | residual sugar | continuous |
| 5 | chlorides | continuous |
| 6 | free sulfur dioxide | continuous |
| 7 | total sulfur dioxide | continuous |
| 8 | density | continuous |
| 9 | pH | continuous |
| 10 | sulphates | continuous |
| 11 | alcohol | continuous |
| 12 | is_red | binary (0/1) |

---

## 2. Candidate Models

All models below treat `quality` as a **class label** (multi-class classification). Models are grouped by family.

### 2.1 Baseline / Simple Models

| Model | Key Idea | Why Try It |
|-------|----------|------------|
| **Majority-class classifier** | Always predict the most frequent class (6) | Provides the "floor" accuracy any real model must beat |
| **Logistic Regression (multinomial)** | Linear decision boundaries via softmax | Simple, interpretable, fast; a strong linear baseline |
| **Linear Discriminant Analysis (LDA)** | Assumes shared Gaussian covariance per class | Classic parametric classifier; works well when assumptions roughly hold |
| **Quadratic Discriminant Analysis (QDA)** | Per-class covariance matrices | Relaxes LDA's equal-covariance assumption at the cost of more parameters |

### 2.2 Regularized Linear Models

| Model | Key Idea | Why Try It |
|-------|----------|------------|
| **Ridge (L2) Logistic Regression** | Penalizes large coefficients with L2 norm | Reduces overfitting when features are correlated (density ↔ alcohol, SO₂ features) |
| **Lasso (L1) Logistic Regression** | Penalizes with L1 norm, inducing sparsity | Automatic feature selection; reveals which inputs matter |
| **Elastic Net Logistic Regression** | Mix of L1 + L2 penalties | Best of both worlds; handles correlated groups of features |

### 2.3 Nearest-Neighbor Methods

| Model | Key Idea | Why Try It |
|-------|----------|------------|
| **K-Nearest Neighbors (KNN)** | Classify by majority vote of k closest training points | Non-parametric; captures local patterns; simple to implement |
| **Weighted KNN** | Closer neighbors get higher weight (e.g., 1/distance) | Reduces sensitivity to the choice of k |

### 2.4 Tree-Based Models

| Model | Key Idea | Why Try It |
|-------|----------|------------|
| **Decision Tree (CART)** | Recursive binary splits on features | Highly interpretable; captures non-linear interactions |
| **Bagged Trees (Bagging)** | Average many bootstrapped trees | Reduces variance of a single tree |
| **Random Forest** | Bagging + random feature subsets at each split | De-correlates trees → lower variance; often a top performer |
| **Extra-Trees (Extremely Randomized Trees)** | Like Random Forest but splits are chosen randomly | Even faster; sometimes better generalization |
| **Gradient Boosted Trees (GBM)** | Sequentially fit trees to residual errors | High accuracy; widely used in competitions |
| **XGBoost** | Regularized, parallelized GBM | State-of-the-art gradient boosting with built-in regularization |
| **LightGBM** | Leaf-wise tree growth, histogram-based splits | Fast training on moderate-to-large data; often matches or beats XGBoost |
| **CatBoost** | Ordered boosting; native handling of categoricals | Robust out-of-the-box; handles `is_red` natively |

### 2.5 Support Vector Machines

| Model | Key Idea | Why Try It |
|-------|----------|------------|
| **Linear SVM (one-vs-rest)** | Maximum-margin linear classifier | Strong when classes are near-linearly separable |
| **SVM with RBF kernel** | Maps to infinite-dimensional space via Gaussian kernel | Captures non-linear boundaries; cited as best performer in original wine quality paper |
| **SVM with Polynomial kernel** | Polynomial feature interactions implicitly | Alternative non-linear kernel; worth comparing |

### 2.6 Neural Networks

| Model | Key Idea | Why Try It |
|-------|----------|------------|
| **Multi-Layer Perceptron (MLP)** | Fully connected feedforward network (1–3 hidden layers) | Universal approximator; can model complex interactions |
| **MLP with Dropout + BatchNorm** | Regularized deeper network | Reduces overfitting on small-ish data |

### 2.7 Probabilistic / Bayesian Models

| Model | Key Idea | Why Try It |
|-------|----------|------------|
| **Naive Bayes (Gaussian)** | Assumes feature independence given class | Extremely fast; surprisingly effective baseline |
| **Bayesian Logistic Regression** | Places priors on coefficients | Principled uncertainty quantification |

### 2.8 Ensemble / Stacking Methods

| Model | Key Idea | Why Try It |
|-------|----------|------------|
| **Voting Classifier (hard/soft)** | Combine predictions of diverse models by vote or averaged probabilities | Easy way to boost accuracy by combining complementary models |
| **Stacking (Stacked Generalization)** | Train a meta-learner on out-of-fold predictions from base models | Often yields the best single prediction by leveraging diversity |
| **Blending** | Similar to stacking but uses a hold-out set instead of cross-validation | Simpler to implement; less prone to target leakage |

### 2.9 Ordinal-Aware Models

Since quality is **ordinal** (3 < 4 < 5 < … < 9), models that respect this ordering may help.

| Model | Key Idea | Why Try It |
|-------|----------|------------|
| **Ordinal Logistic Regression (Proportional Odds)** | Single set of coefficients + ordered thresholds | Directly models the ordinal structure of quality |
| **Ordinal Random Forest / Ordinal Boosting** | Tree methods adapted for ordinal targets | Preserves ordering during splits |
| **Cumulative Link Model (CLM)** | Generalized linear model for ordered categorical data | Standard statistical approach to ordinal outcomes |

### 2.10 Regression-then-Round Approach

Treat `quality` as a continuous variable, fit a regression model, then round predictions to the nearest integer.

| Model | Key Idea | Why Try It |
|-------|----------|------------|
| **Linear Regression → round** | OLS on quality, then discretize | Simple; leverages ordinal spacing directly |
| **Ridge / Lasso Regression → round** | Regularized regression then discretize | Handles multicollinearity while using ordinal info |
| **SVR (RBF) → round** | Support vector regression then discretize | Non-linear regression baseline |
| **Random Forest Regressor → round** | RF regression then discretize | Often competitive; compare vs. classification RF |
| **XGBoost Regressor → round** | Gradient boosted regression then discretize | Compare regression vs. classification framing |

---

## 3. Project Structure

```
wine-quality-prediction/
├── data/
│   ├── raw/                        # Original data files
│   │   └── winequality.names       # Dataset documentation
│   └── processed/
│       ├── train.csv               # Training data (provided)
│       └── test.csv                # Test data (to be provided)
├── notebooks/
│   ├── 01_eda.ipynb                # Exploratory data analysis
│   ├── 02_preprocessing.ipynb      # Feature engineering & preprocessing
│   ├── 03_model_training.ipynb     # Model fitting & tuning
│   ├── 04_model_comparison.ipynb   # Side-by-side model evaluation
│   └── 05_final_prediction.ipynb   # Generate test set predictions
├── src/
│   ├── preprocess.R                # Preprocessing functions
│   ├── train_models.R              # Model training utilities
│   ├── evaluate.R                  # Evaluation metrics & helpers
│   └── predict.R                   # Generate final predictions
├── outputs/
│   ├── distributions/              # EDA plots & tables
│   ├── models/                     # Saved model objects
│   └── results/                    # Comparison tables, final predictions
├── analyze_distributions.R         # Existing EDA script
├── PROJECT_PLAN.md                 # ← This file
├── AI_USAGE.md                     # AI usage documentation
└── .gitignore
```

---

## 4. Phase-by-Phase Plan

### Phase 1 — Exploratory Data Analysis (EDA)

| Step | Task | Output |
|------|------|--------|
| 1.1 | Load data; verify dimensions, types, missing values | Summary table |
| 1.2 | Visualize target distribution (bar chart) | `quality_distribution_bar.png` *(done)* |
| 1.3 | Histograms / density plots for each feature | Per-feature PNGs *(done)* |
| 1.4 | Correlation heatmap (feature–feature and feature–target) | Heatmap PNG |
| 1.5 | Box plots of each feature grouped by quality level | Box plot PNGs |
| 1.6 | Compare red vs. white wine distributions | Side-by-side plots |
| 1.7 | Detect outliers (IQR / z-score) | Outlier summary table |
| 1.8 | Document EDA findings and insights | Section in notebook |

### Phase 2 — Data Preprocessing & Feature Engineering

| Step | Task | Detail |
|------|------|--------|
| 2.1 | Handle class imbalance strategy decision | Evaluate: oversampling (SMOTE), undersampling, class weights, or none |
| 2.2 | Feature scaling | Standardize (z-score) or normalize (min-max) — required for SVM, KNN, NN |
| 2.3 | Feature selection exploration | Correlation filtering, Lasso coefficients, tree-based importances |
| 2.4 | Engineer new features (optional) | Ratios (free/total SO₂), interaction terms, polynomial features |
| 2.5 | Create preprocessing pipeline | Reproducible pipeline that can be applied to test set identically |

### Phase 3 — Model Training & Hyperparameter Tuning

| Step | Task | Detail |
|------|------|--------|
| 3.1 | Set up cross-validation | Stratified k-fold (k = 5 or 10) to preserve class proportions |
| 3.2 | Train baseline models | Majority class, logistic regression, decision tree |
| 3.3 | Train all candidate models (§2) | Fit each model family with default hyperparameters |
| 3.4 | Hyperparameter tuning | Grid search or random search for top-performing models |
| 3.5 | Record CV metrics for every model | Accuracy, macro F1, weighted F1, Cohen's kappa, confusion matrix |

### Phase 4 — Model Comparison & Selection

| Step | Task | Detail |
|------|------|--------|
| 4.1 | Build comparison table | Rank all models by primary metric (e.g., weighted F1 or accuracy) |
| 4.2 | Statistical significance testing | Paired t-test or Wilcoxon test on CV fold scores |
| 4.3 | Analyze confusion matrices | Identify which classes each model struggles with |
| 4.4 | Evaluate trade-offs | Accuracy vs. interpretability vs. training time |
| 4.5 | Select top 3–5 models for ensembling | Diverse models with complementary error patterns |
| 4.6 | Build ensemble (stacking / voting) | Combine top models; evaluate via CV |
| 4.7 | Choose the **"best" model** | Based on CV performance + simplicity |

### Phase 5 — Final Prediction & Reporting

| Step | Task | Detail |
|------|------|--------|
| 5.1 | Retrain best model on full training set | Use selected hyperparameters |
| 5.2 | Generate predictions on `test.csv` | Apply identical preprocessing pipeline |
| 5.3 | Save predictions | CSV with predicted quality for each test observation |
| 5.4 | Write final report | Model choice rationale, key findings, limitations |

---

## 5. Evaluation Metrics

Since this is an **imbalanced, ordinal, multi-class** problem, multiple metrics are warranted:

| Metric | Why |
|--------|-----|
| **Accuracy** | Simple overall correctness; easy to interpret but misleading with imbalance |
| **Weighted F1-score** | Balances precision and recall, weighted by class frequency |
| **Macro F1-score** | Equal weight to every class; highlights performance on rare classes (3, 4, 8, 9) |
| **Cohen's Kappa** | Adjusts for chance agreement; useful for ordinal classification |
| **Quadratic Weighted Kappa (QWK)** | Penalizes predictions farther from the true class more heavily; ideal for ordinal targets |
| **Confusion Matrix** | Visual breakdown of per-class errors |
| **MAE (on class label)** | Mean absolute error treating labels as integers; captures "how far off" predictions are |

**Primary metric recommendation:** **Quadratic Weighted Kappa** or **Weighted F1-score** (choose based on assignment rubric).

---

## 6. Key Considerations

1. **Class imbalance** — Classes 3 and 9 have very few samples. Use stratified CV, consider class weights or resampling (SMOTE), and evaluate with macro F1 / kappa rather than raw accuracy alone.

2. **Feature correlation** — Several features are correlated (e.g., density with alcohol and residual sugar; free SO₂ with total SO₂). Regularized models and feature selection will help.

3. **Ordinal nature** — Quality levels have a natural order. Ordinal models or the regression-then-round approach may outperform naive multi-class classifiers.

4. **No missing values** — The dataset is complete, so no imputation is needed.

5. **Small dataset** — ~5,200 rows is moderate. Complex models (deep neural nets) may overfit; simpler models with proper regularization could be competitive.

6. **Reproducibility** — Set random seeds everywhere. Document all preprocessing steps so the same pipeline applies to `test.csv`.
