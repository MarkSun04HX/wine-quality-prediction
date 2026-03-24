suppressPackageStartupMessages({
  library(nnet)
  library(glmnet)
  library(randomForest)
  library(xgboost)
  library(e1071)
  library(MASS)
})

# ---------------------------------------------------------------------------
# 1. OLS
# ---------------------------------------------------------------------------
train_and_predict_ols <- function(X_train, y_train, X_test) {
  df <- data.frame(quality = y_train, X_train, check.names = FALSE)
  fit <- lm(quality ~ ., data = df)
  preds <- predict(fit, newdata = data.frame(X_test, check.names = FALSE))
  list(predictions = as.numeric(preds), model = fit)
}

# ---------------------------------------------------------------------------
# 2. Multinomial logistic regression
# ---------------------------------------------------------------------------
train_and_predict_multinom <- function(X_train, y_train, X_test) {
  df <- data.frame(quality = factor(y_train), X_train, check.names = FALSE)
  fit <- suppressWarnings(multinom(quality ~ ., data = df, trace = FALSE, MaxNWts = 5000))
  preds <- predict(fit, newdata = data.frame(X_test, check.names = FALSE))
  list(predictions = as.integer(as.character(preds)), model = fit)
}

# ---------------------------------------------------------------------------
# 3. Ridge logistic (glmnet, alpha = 0)
# ---------------------------------------------------------------------------
train_and_predict_ridge_logistic <- function(X_train, y_train, X_test) {
  X_mat <- as.matrix(X_train)
  y_fac <- factor(y_train)
  set.seed(42)
  cv_fit <- cv.glmnet(X_mat, y_fac, family = "multinomial",
                       alpha = 0, type.measure = "class")
  preds <- predict(cv_fit, newx = as.matrix(X_test),
                   s = "lambda.min", type = "class")
  list(predictions = as.integer(preds), model = cv_fit)
}

# ---------------------------------------------------------------------------
# 4. Lasso logistic (glmnet, alpha = 1)
# ---------------------------------------------------------------------------
train_and_predict_lasso_logistic <- function(X_train, y_train, X_test) {
  X_mat <- as.matrix(X_train)
  y_fac <- factor(y_train)
  set.seed(42)
  cv_fit <- cv.glmnet(X_mat, y_fac, family = "multinomial",
                       alpha = 1, type.measure = "class")
  preds <- predict(cv_fit, newx = as.matrix(X_test),
                   s = "lambda.min", type = "class")
  list(predictions = as.integer(preds), model = cv_fit)
}

# ---------------------------------------------------------------------------
# 5. Elastic Net logistic (glmnet, alpha = 0.5)
# ---------------------------------------------------------------------------
train_and_predict_enet_logistic <- function(X_train, y_train, X_test) {
  X_mat <- as.matrix(X_train)
  y_fac <- factor(y_train)
  set.seed(42)
  cv_fit <- cv.glmnet(X_mat, y_fac, family = "multinomial",
                       alpha = 0.5, type.measure = "class")
  preds <- predict(cv_fit, newx = as.matrix(X_test),
                   s = "lambda.min", type = "class")
  list(predictions = as.integer(preds), model = cv_fit)
}

# ---------------------------------------------------------------------------
# 6. KNN (internal 3-fold CV to pick k from {5, 7, 11})
# ---------------------------------------------------------------------------
train_and_predict_knn <- function(X_train, y_train, X_test) {
  X_tr <- as.matrix(X_train)
  X_te <- as.matrix(X_test)
  y_fac <- factor(y_train)

  n <- nrow(X_tr)
  set.seed(123)
  fold_ids <- sample(rep(1:3, length.out = n))

  best_k   <- 5
  best_acc <- 0
  for (k_try in c(5, 7, 11)) {
    accs <- numeric(3)
    for (f in 1:3) {
      idx <- fold_ids == f
      p <- class::knn(X_tr[!idx, , drop = FALSE],
                      X_tr[ idx, , drop = FALSE],
                      y_fac[!idx], k = k_try)
      accs[f] <- mean(p == y_fac[idx])
    }
    if (mean(accs) > best_acc) {
      best_acc <- mean(accs)
      best_k   <- k_try
    }
  }

  preds <- class::knn(X_tr, X_te, y_fac, k = best_k)
  list(predictions = as.integer(as.character(preds)),
       model = list(k = best_k))
}

# ---------------------------------------------------------------------------
# 7. Random Forest classifier
# ---------------------------------------------------------------------------
train_and_predict_rf_class <- function(X_train, y_train, X_test) {
  set.seed(42)
  fit <- randomForest(x = X_train, y = factor(y_train), ntree = 500)
  preds <- predict(fit, newdata = X_test)
  list(predictions = as.integer(as.character(preds)), model = fit)
}

# ---------------------------------------------------------------------------
# 8. XGBoost classifier (multi:softmax)
# ---------------------------------------------------------------------------
train_and_predict_xgb_class <- function(X_train, y_train, X_test) {
  offset    <- min(y_train)
  labels    <- y_train - offset
  num_class <- max(labels) + 1L

  dtrain <- xgb.DMatrix(data = as.matrix(X_train), label = labels)
  dtest  <- xgb.DMatrix(data = as.matrix(X_test))

  params <- list(objective  = "multi:softmax",
                 num_class  = num_class,
                 max_depth  = 6,
                 eta        = 0.1,
                 nthread    = 1)
  set.seed(42)
  fit <- xgb.train(params, dtrain, nrounds = 200, verbose = 0)
  preds <- predict(fit, dtest) + offset
  list(predictions = as.integer(preds), model = fit)
}

# ---------------------------------------------------------------------------
# 9. CatBoost classifier
# ---------------------------------------------------------------------------
train_and_predict_catboost_class <- function(X_train, y_train, X_test) {
  if (!requireNamespace("catboost", quietly = TRUE)) {
    warning("catboost not installed — skipping")
    return(NULL)
  }

  offset <- min(y_train)
  labels <- y_train - offset

  train_pool <- catboost::catboost.load_pool(data = as.data.frame(X_train),
                                              label = labels)
  test_pool  <- catboost::catboost.load_pool(data = as.data.frame(X_test))

  params <- list(loss_function = "MultiClass",
                 iterations    = 200,
                 depth         = 6,
                 learning_rate = 0.1,
                 verbose       = 0,
                 thread_count  = 1)

  fit <- catboost::catboost.train(train_pool, params = params)
  preds <- catboost::catboost.predict(fit, test_pool,
                                       prediction_type = "Class")
  preds <- as.integer(preds) + offset
  list(predictions = preds, model = fit)
}

# ---------------------------------------------------------------------------
# 10. Naive Bayes (Gaussian)
# ---------------------------------------------------------------------------
train_and_predict_naive_bayes <- function(X_train, y_train, X_test) {
  fit <- naiveBayes(x = as.data.frame(X_train), y = factor(y_train))
  preds <- predict(fit, newdata = as.data.frame(X_test))
  list(predictions = as.integer(as.character(preds)), model = fit)
}

# ---------------------------------------------------------------------------
# 11. Ordinal logistic regression (proportional odds)
# ---------------------------------------------------------------------------
train_and_predict_ordinal <- function(X_train, y_train, X_test) {
  df <- data.frame(quality = factor(y_train, ordered = TRUE),
                   X_train, check.names = FALSE)
  fit <- suppressWarnings(polr(quality ~ ., data = df, Hess = TRUE))
  preds <- predict(fit, newdata = data.frame(X_test, check.names = FALSE))
  list(predictions = as.integer(as.character(preds)), model = fit)
}

# ---------------------------------------------------------------------------
# 12. Ridge regression (glmnet, alpha = 0, gaussian)
# ---------------------------------------------------------------------------
train_and_predict_ridge_reg <- function(X_train, y_train, X_test) {
  X_mat <- as.matrix(X_train)
  set.seed(42)
  cv_fit <- cv.glmnet(X_mat, y_train, family = "gaussian", alpha = 0)
  preds <- predict(cv_fit, newx = as.matrix(X_test), s = "lambda.min")
  list(predictions = as.numeric(preds), model = cv_fit)
}

# ---------------------------------------------------------------------------
# 13. Lasso regression (glmnet, alpha = 1, gaussian)
# ---------------------------------------------------------------------------
train_and_predict_lasso_reg <- function(X_train, y_train, X_test) {
  X_mat <- as.matrix(X_train)
  set.seed(42)
  cv_fit <- cv.glmnet(X_mat, y_train, family = "gaussian", alpha = 1)
  preds <- predict(cv_fit, newx = as.matrix(X_test), s = "lambda.min")
  list(predictions = as.numeric(preds), model = cv_fit)
}

# ---------------------------------------------------------------------------
# 14. Random Forest regressor
# ---------------------------------------------------------------------------
train_and_predict_rf_reg <- function(X_train, y_train, X_test) {
  set.seed(42)
  fit <- randomForest(x = X_train, y = y_train, ntree = 500)
  preds <- predict(fit, newdata = X_test)
  list(predictions = as.numeric(preds), model = fit)
}

# ---------------------------------------------------------------------------
# 15. XGBoost regressor (reg:squarederror)
# ---------------------------------------------------------------------------
train_and_predict_xgb_reg <- function(X_train, y_train, X_test) {
  dtrain <- xgb.DMatrix(data = as.matrix(X_train), label = y_train)
  dtest  <- xgb.DMatrix(data = as.matrix(X_test))

  params <- list(objective = "reg:squarederror",
                 max_depth = 6,
                 eta       = 0.1,
                 nthread   = 1)
  set.seed(42)
  fit <- xgb.train(params, dtrain, nrounds = 200, verbose = 0)
  preds <- predict(fit, dtest)
  list(predictions = as.numeric(preds), model = fit)
}

# ---------------------------------------------------------------------------
# MODEL_REGISTRY — name, function, regressor flag
# ---------------------------------------------------------------------------
MODEL_REGISTRY <- list(
  list(name = "Elastic Net Logistic", fn = train_and_predict_enet_logistic, is_reg = FALSE),
  list(name = "Random Forest (clf)", fn = train_and_predict_rf_class,      is_reg = FALSE),
  list(name = "XGBoost (clf)",      fn = train_and_predict_xgb_class,      is_reg = FALSE)
)
