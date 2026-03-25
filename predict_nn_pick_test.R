#!/usr/bin/env Rscript
# Train on data/processed/train.csv, predict data/processed/test.csv using the
# same nn-pick rule as nn_pick_model_cv.R (expanded features, ENet vs KNN).
# Accuracy uses ceiling(predicted), clipped to integer quality 3–9.

suppressPackageStartupMessages({
  library(glmnet)
  library(FNN)
})

source("src/helpers.R")

TEST_PATH <- "data/processed/test.csv"
NEIGHBORS_FOR_PICKING <- 5L
NEIGHBORS_FOR_KNN_MODEL <- 7L
MODEL_SEED <- SEED

expand_features <- function(X) {
  X_mat <- as.matrix(X)
  X_sq <- X_mat^2
  colnames(X_sq) <- paste0(colnames(X_mat), "_sq")
  X_interact <- model.matrix(~ .^2 - 1, data = as.data.frame(X_mat))
  cbind(X_interact, X_sq)
}

get_top_idx <- function(X_train, x_row, k) {
  d <- sqrt(rowSums(sweep(as.matrix(X_train), 2, as.numeric(x_row), FUN = "-")^2))
  kk <- min(as.integer(k), length(d))
  order(d)[seq_len(kk)]
}

accuracy_ceiling <- function(actual, predicted) {
  pred_int <- as.integer(pmax(pmin(ceiling(predicted), 9L), 3L))
  mean(as.integer(actual) == pred_int)
}

run_nn_pick <- function(X_train, y_train, X_test, seed) {
  set.seed(seed)
  en_cv <- cv.glmnet(as.matrix(X_train), y_train, family = "gaussian", alpha = 0.5)
  en_pred_tr <- as.numeric(predict(en_cv, newx = as.matrix(X_train), s = "lambda.min"))
  en_pred_te <- as.numeric(predict(en_cv, newx = as.matrix(X_test), s = "lambda.min"))

  knn_pred_tr <- knn.reg(
    train = X_train, test = X_train, y = y_train,
    k = NEIGHBORS_FOR_KNN_MODEL
  )$pred
  knn_pred_te <- knn.reg(
    train = X_train, test = X_test, y = y_train,
    k = NEIGHBORS_FOR_KNN_MODEL
  )$pred

  err_en <- abs(y_train - en_pred_tr)
  err_knn <- abs(y_train - knn_pred_tr)
  ens_pred_te <- numeric(nrow(X_test))

  for (i in seq_len(nrow(X_test))) {
    idx <- get_top_idx(X_train, X_test[i, , drop = FALSE], k = NEIGHBORS_FOR_PICKING)
    if (mean(err_en[idx]) <= mean(err_knn[idx])) {
      ens_pred_te[i] <- en_pred_te[i]
    } else {
      ens_pred_te[i] <- knn_pred_te[i]
    }
  }

  list(
    enet = en_pred_te,
    knn = knn_pred_te,
    ensemble = ens_pred_te
  )
}

if (!file.exists(TEST_PATH)) {
  stop("Test file not found: ", TEST_PATH)
}

train_raw <- load_train_data()
test_raw <- read.csv(TEST_PATH, sep = ";", check.names = FALSE)

if (!"quality" %in% colnames(test_raw)) {
  stop("test.csv must contain a 'quality' column to compute accuracy.")
}

proc_train <- preprocess(train_raw)
proc_test <- apply_preprocess(test_raw, proc_train$scale_params)

X_tr <- expand_features(proc_train$X)
X_te <- expand_features(proc_test$X)
y_tr <- proc_train$y
y_te <- proc_test$y

preds <- run_nn_pick(X_tr, y_tr, X_te, MODEL_SEED)

acc_enet <- accuracy_ceiling(y_te, preds$enet)
acc_knn <- accuracy_ceiling(y_te, preds$knn)
acc_ens <- accuracy_ceiling(y_te, preds$ensemble)

cat("=== NN-pick (ENet + KNN) | train -> test ===\n")
cat(sprintf("Test rows: %d\n", nrow(test_raw)))
cat(sprintf("Accuracy (ceiling, clipped 3–9) — ENet:     %.4f\n", acc_enet))
cat(sprintf("Accuracy (ceiling, clipped 3–9) — KNN:     %.4f\n", acc_knn))
cat(sprintf("Accuracy (ceiling, clipped 3–9) — NN-pick: %.4f\n", acc_ens))

output_dir <- "outputs/results"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
out_df <- data.frame(
  actual_quality = y_te,
  pred_enet = preds$enet,
  pred_knn = preds$knn,
  pred_nn_pick = preds$ensemble,
  pred_nn_pick_ceiling = as.integer(pmax(pmin(ceiling(preds$ensemble), 9L), 3L)),
  stringsAsFactors = FALSE
)
write.csv(out_df, file.path(output_dir, "nn_pick_test_predictions.csv"), row.names = FALSE)

metrics <- data.frame(
  method = c("elastic_net", "knn_reg", "nn_pick_ensemble"),
  accuracy_ceiling_clipped = c(acc_enet, acc_knn, acc_ens),
  stringsAsFactors = FALSE
)
write.csv(metrics, file.path(output_dir, "nn_pick_test_accuracy.csv"), row.names = FALSE)

cat("\nWrote:\n  ", file.path(output_dir, "nn_pick_test_predictions.csv"), "\n", sep = "")
cat("  ", file.path(output_dir, "nn_pick_test_accuracy.csv"), "\n", sep = "")
