#!/usr/bin/env Rscript
# Same logic as nn_pick_model_cv.R: expanded features, k nearest training rows,
# pick Random Forest vs XGBoost by lower mean absolute error on those neighbors
# (in-sample train predictions), then use that model's test prediction per row.
# Reports 10-fold CV accuracy and RMSE for RF, XGB, and the NN-picked blend.

suppressPackageStartupMessages({
  library(randomForest)
  library(xgboost)
})

source("src/helpers.R")

K_FOLDS <- 10L
CV_SEED <- 123L
NEIGHBORS_FOR_PICKING <- 5L

expand_features <- function(X) {
  X_mat <- as.matrix(X)
  X_sq <- X_mat^2
  colnames(X_sq) <- paste0(colnames(X_mat), "_sq")
  X_interact <- model.matrix(~ .^2 - 1, data = as.data.frame(X_mat))
  cbind(X_interact, X_sq)
}

get_top_idx <- function(X_train, x_row, k) {
  d <- sqrt(rowSums(sweep(as.matrix(X_train), 2, as.numeric(x_row), FUN = "-")^2))
  order(d)[seq_len(min(k, length(d)))]
}

fit_predict_rf <- function(X_train, y_train, X_test, seed) {
  set.seed(seed)
  fit <- randomForest(x = X_train, y = factor(y_train), ntree = 500)
  p_tr <- as.numeric(as.character(predict(fit, newdata = X_train)))
  p_te <- as.numeric(as.character(predict(fit, newdata = X_test)))
  list(pred_train = p_tr, pred_test = p_te)
}

fit_predict_xgb <- function(X_train, y_train, X_test, seed) {
  offset <- min(y_train)
  labels <- as.integer(y_train - offset)
  num_class <- max(labels) + 1L
  dtrain <- xgb.DMatrix(data = as.matrix(X_train), label = labels)
  dtest <- xgb.DMatrix(data = as.matrix(X_test))
  params <- list(
    objective = "multi:softmax",
    num_class = num_class,
    max_depth = 6,
    eta = 0.1,
    nthread = 1
  )
  set.seed(seed)
  fit <- xgb.train(params, dtrain, nrounds = 200, verbose = 0)
  p_tr <- as.numeric(predict(fit, dtrain)) + offset
  p_te <- as.numeric(predict(fit, dtest)) + offset
  list(pred_train = p_tr, pred_test = p_te)
}

run_nn_pick_rf_xgb <- function(X_train, y_train, X_test, seed) {
  rf <- fit_predict_rf(X_train, y_train, X_test, seed)
  xgb <- fit_predict_xgb(X_train, y_train, X_test, seed + 791L)

  err_rf <- abs(y_train - rf$pred_train)
  err_xgb <- abs(y_train - xgb$pred_train)

  ens <- numeric(nrow(X_test))
  for (i in seq_len(nrow(X_test))) {
    idx <- get_top_idx(X_train, X_test[i, , drop = FALSE], k = NEIGHBORS_FOR_PICKING)
    if (mean(err_rf[idx]) <= mean(err_xgb[idx])) {
      ens[i] <- rf$pred_test[i]
    } else {
      ens[i] <- xgb$pred_test[i]
    }
  }

  list(rf = rf$pred_test, xgb = xgb$pred_test, ensemble = ens)
}

raw <- load_train_data()
folds <- create_cv_folds(raw$quality, k = K_FOLDS, seed = CV_SEED)

rf_acc <- numeric(K_FOLDS)
xgb_acc <- numeric(K_FOLDS)
ens_acc <- numeric(K_FOLDS)
rf_rmse <- numeric(K_FOLDS)
xgb_rmse <- numeric(K_FOLDS)
ens_rmse <- numeric(K_FOLDS)

cat(sprintf(
  "%d-fold CV: NN-pick (k=%d) Random Forest vs XGBoost\n\n",
  K_FOLDS, NEIGHBORS_FOR_PICKING
))

for (i in seq_along(folds)) {
  test_idx <- folds[[i]]
  proc_tr <- preprocess(raw[-test_idx, ])
  proc_te <- apply_preprocess(raw[test_idx, ], proc_tr$scale_params)

  X_tr <- expand_features(proc_tr$X)
  X_te <- expand_features(proc_te$X)
  y_tr <- proc_tr$y
  y_te <- proc_te$y

  preds <- run_nn_pick_rf_xgb(X_tr, y_tr, X_te, CV_SEED + 1000L * i)

  rf_acc[i] <- compute_accuracy(y_te, preds$rf)
  xgb_acc[i] <- compute_accuracy(y_te, preds$xgb)
  ens_acc[i] <- compute_accuracy(y_te, preds$ensemble)

  rf_rmse[i] <- compute_rmse(y_te, preds$rf)
  xgb_rmse[i] <- compute_rmse(y_te, preds$xgb)
  ens_rmse[i] <- compute_rmse(y_te, preds$ensemble)

  cat(sprintf(
    "Fold %2d | RF acc %.4f rmse %.4f | XGB acc %.4f rmse %.4f | NN-pick acc %.4f rmse %.4f\n",
    i, rf_acc[i], rf_rmse[i], xgb_acc[i], xgb_rmse[i], ens_acc[i], ens_rmse[i]
  ))
}

summary_df <- data.frame(
  model = c("Random Forest", "XGBoost", "NN-picked RF vs XGB"),
  mean_cv_accuracy = c(mean(rf_acc), mean(xgb_acc), mean(ens_acc)),
  sd_cv_accuracy = c(sd(rf_acc), sd(xgb_acc), sd(ens_acc)),
  mean_cv_rmse = c(mean(rf_rmse), mean(xgb_rmse), mean(ens_rmse)),
  sd_cv_rmse = c(sd(rf_rmse), sd(xgb_rmse), sd(ens_rmse)),
  stringsAsFactors = FALSE
)

cat("\n--- Summary (accuracy & RMSE) ---\n")
print(summary_df)

output_dir <- "outputs/results"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
write.csv(summary_df,
          file.path(output_dir, "nn_pick_rf_xgb_cv_summary.csv"),
          row.names = FALSE)
write.csv(
  data.frame(
    fold = seq_len(K_FOLDS),
    rf_accuracy = rf_acc, rf_rmse = rf_rmse,
    xgb_accuracy = xgb_acc, xgb_rmse = xgb_rmse,
    nn_pick_accuracy = ens_acc, nn_pick_rmse = ens_rmse
  ),
  file.path(output_dir, "nn_pick_rf_xgb_cv_by_fold.csv"),
  row.names = FALSE
)
cat("\nWrote:", file.path(output_dir, "nn_pick_rf_xgb_cv_summary.csv"), "\n")
