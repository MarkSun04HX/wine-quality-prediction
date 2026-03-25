#!/usr/bin/env Rscript
# ENet vs KNN vs NN-picked ensemble: 3-fold and 10-fold stratified CV, repeated
# over several fold-assignment seeds for different randomizations.

suppressPackageStartupMessages({
  library(glmnet)
  library(FNN)
})

source("src/helpers.R")

K_FOLDS_OPTIONS <- c(3L, 10L)
# Different seeds -> different stratified fold splits (and distinct model seeds per fold).
CV_SEEDS <- c(123L, 456L, 789L, 2024L, 42L)
NEIGHBORS_FOR_PICKING <- 5L
NEIGHBORS_FOR_KNN_MODEL <- 7L

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

run_comparison_cycle <- function(X_train, y_train, X_test, seed) {
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

  list(enet = en_pred_te, knn = knn_pred_te, ensemble = ens_pred_te)
}

raw <- load_train_data()

output_dir <- "outputs/results"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

run_one_cv <- function(k_folds, fold_seed) {
  folds <- create_cv_folds(raw$quality, k = k_folds, seed = fold_seed)
  n_f <- length(folds)
  enet_acc <- numeric(n_f)
  knn_acc <- numeric(n_f)
  ens_acc <- numeric(n_f)

  for (i in seq_along(folds)) {
    test_idx <- folds[[i]]
    proc_tr <- preprocess(raw[-test_idx, ])
    proc_te <- apply_preprocess(raw[test_idx, ], proc_tr$scale_params)

    X_tr_exp <- expand_features(proc_tr$X)
    X_te_exp <- expand_features(proc_te$X)

    model_seed <- as.integer(fold_seed) + 1000L * as.integer(i) + as.integer(k_folds) * 7919L
    preds <- run_comparison_cycle(X_tr_exp, proc_tr$y, X_te_exp, model_seed)

    enet_acc[i] <- compute_accuracy(proc_te$y, preds$enet)
    knn_acc[i] <- compute_accuracy(proc_te$y, preds$knn)
    ens_acc[i] <- compute_accuracy(proc_te$y, preds$ensemble)
  }

  list(
    k_folds = k_folds,
    fold_seed = fold_seed,
    enet_mean = mean(enet_acc),
    enet_sd = sd(enet_acc),
    knn_mean = mean(knn_acc),
    knn_sd = sd(knn_acc),
    ensemble_mean = mean(ens_acc),
    ensemble_sd = sd(ens_acc),
    fold_detail = data.frame(
      fold = seq_len(n_f),
      k_folds = k_folds,
      fold_seed = fold_seed,
      enet_acc = enet_acc,
      knn_acc = knn_acc,
      ensemble_acc = ens_acc
    )
  )
}

cat("=== NN-pick CV: 3-fold and 10-fold × multiple fold seeds ===\n\n")

summary_rows <- list()
fold_rows <- list()
row_id <- 0L

for (k in K_FOLDS_OPTIONS) {
  for (sv in CV_SEEDS) {
    row_id <- row_id + 1L
    cat(sprintf("--- k = %d  fold_seed = %d ---\n", k, sv))
    res <- run_one_cv(k, sv)
    summary_rows[[row_id]] <- data.frame(
      k_folds = res$k_folds,
      cv_fold_seed = res$fold_seed,
      enet_mean_acc = res$enet_mean,
      enet_sd_acc = res$enet_sd,
      knn_mean_acc = res$knn_mean,
      knn_sd_acc = res$knn_sd,
      ensemble_mean_acc = res$ensemble_mean,
      ensemble_sd_acc = res$ensemble_sd,
      stringsAsFactors = FALSE
    )
    fold_rows[[row_id]] <- res$fold_detail

    cat(sprintf(
      "  Mean acc | ENet: %.4f | KNN: %.4f | Ensemble: %.4f\n\n",
      res$enet_mean, res$knn_mean, res$ensemble_mean
    ))
  }
}

summary_df <- do.call(rbind, summary_rows)
fold_df <- do.call(rbind, fold_rows)

cat("--- Summary (all k × seeds) ---\n")
print(summary_df)

write.csv(
  summary_df,
  file.path(output_dir, "nn_pick_model_cv_summary_by_k_seed.csv"),
  row.names = FALSE
)
write.csv(
  fold_df,
  file.path(output_dir, "nn_pick_model_cv_by_fold.csv"),
  row.names = FALSE
)
cat("\nWrote:", file.path(output_dir, "nn_pick_model_cv_summary_by_k_seed.csv"), "\n")
