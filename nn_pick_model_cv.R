#!/usr/bin/env Rscript
# Comparison: 10-Fold CV | Ensemble (ENet vs KNN) vs Standalone Models

suppressPackageStartupMessages({
  library(glmnet)
  library(FNN) # Fast Nearest Neighbors for regression
})

source("src/helpers.R")

K_FOLDS <- 10
CV_SEED <- 42
NEIGHBORS_FOR_PICKING <- 5 # K used to decide which model to use
NEIGHBORS_FOR_KNN_MODEL <- 7 # K used inside the KNN regression model itself

# --- Feature Engineering ---
expand_features <- function(X) {
  X_mat <- as.matrix(X)
  X_sq <- X_mat^2
  colnames(X_sq) <- paste0(colnames(X_mat), "_sq")
  X_interact <- model.matrix(~ .^2 - 1, data = as.data.frame(X_mat))
  cbind(X_interact, X_sq)
}

# --- Distance Logic for Model Selection ---
get_top_idx <- function(X_train, x_row, k) {
  d <- sqrt(rowSums(sweep(as.matrix(X_train), 2, as.numeric(x_row), FUN = "-")^2))
  order(d)[1:k]
}

# --- Core Comparison Logic ---
run_comparison_cycle <- function(X_train, y_train, X_test, seed) {
  # 1. Fit Expanded Elastic Net
  set.seed(seed)
  en_cv <- cv.glmnet(as.matrix(X_train), y_train, family = "gaussian", alpha = 0.5)
  en_pred_tr <- as.numeric(predict(en_cv, newx = as.matrix(X_train), s = "lambda.min"))
  en_pred_te <- as.numeric(predict(en_cv, newx = as.matrix(X_test), s = "lambda.min"))
  
  # 2. KNN Regression Model (The new competitor)
  # In-sample (train) prediction using leave-one-out to avoid overfit bias
  knn_pred_tr <- knn.reg(train = X_train, test = X_train, y = y_train, k = NEIGHBORS_FOR_KNN_MODEL)$pred
  # Out-of-sample (test) prediction
  knn_pred_te <- knn.reg(train = X_train, test = X_test, y = y_train, k = NEIGHBORS_FOR_KNN_MODEL)$pred
  
  # 3. Ensemble (KNN Picking)
  err_en <- abs(y_train - en_pred_tr)
  err_knn <- abs(y_train - knn_pred_tr)
  ens_pred_te <- numeric(nrow(X_test))
  
  for (i in seq_len(nrow(X_test))) {
    idx <- get_top_idx(X_train, X_test[i, , drop = FALSE], k = NEIGHBORS_FOR_PICKING)
    # If ENet performed better on the neighbors than the KNN model did...
    if (mean(err_en[idx]) <= mean(err_knn[idx])) {
      ens_pred_te[i] <- en_pred_te[i]
    } else {
      ens_pred_te[i] <- knn_pred_te[i]
    }
  }
  
  list(enet = en_pred_te, knn = knn_pred_te, ensemble = ens_pred_te)
}

# --- Main Execution ---
raw <- load_train_data()
folds <- create_cv_folds(raw$quality, k = K_FOLDS, seed = CV_SEED)

results <- data.frame(fold = 1:K_FOLDS, enet_acc = 0, knn_acc = 0, ensemble_acc = 0)

cat(sprintf("Starting 10-fold CV: ENet vs KNN vs Ensemble...\n\n"))

for (i in seq_along(folds)) {
  test_idx  <- folds[[i]]
  proc_tr   <- preprocess(raw[-test_idx, ])
  proc_te   <- apply_preprocess(raw[test_idx, ], proc_tr$scale_params)
  
  X_tr_exp  <- expand_features(proc_tr$X)
  X_te_exp  <- expand_features(proc_te$X)
  
  preds <- run_comparison_cycle(X_tr_exp, proc_tr$y, X_te_exp, CV_SEED + i)
  
  results$enet_acc[i]     <- compute_accuracy(proc_te$y, preds$enet)
  results$knn_acc[i]      <- compute_accuracy(proc_te$y, preds$knn)
  results$ensemble_acc[i] <- compute_accuracy(proc_te$y, preds$ensemble)
  
  cat(sprintf("Fold %d | ENet: %.4f | KNN: %.4f | Ensemble: %.4f\n", 
              i, results$enet_acc[i], results$knn_acc[i], results$ensemble_acc[i]))
}

# Summary Stats
summary_stats <- data.frame(
  Model = c("Expanded Elastic Net", "KNN Regression", "KNN-Picked Ensemble"),
  Mean_Accuracy = c(mean(results$enet_acc), mean(results$knn_acc), mean(results$ensemble_acc)),
  SD = c(sd(results$enet_acc), sd(results$knn_acc), sd(results$ensemble_acc))
)

cat("\n--- Final Performance Comparison ---\n")
print(summary_stats)