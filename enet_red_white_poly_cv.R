#!/usr/bin/env Rscript
# 10-fold stratified CV: separate Elastic Net (gaussian, alpha = 0.5) for
# red (is_red == 1) vs non-red wines. Features = linear + squares +
# pairwise interactions on continuous inputs (after log1p chlorides +
# scaling from the full training fold). Test rows are predicted with the
# model matching their is_red.

suppressPackageStartupMessages(library(glmnet))

source("src/helpers.R")

K_FOLDS <- 10L
CV_SEED <- 42L
ENET_ALPHA <- 0.5

expand_linear_sq_interact <- function(Xdf) {
  cn <- colnames(Xdf)
  p <- ncol(Xdf)
  out <- as.data.frame(Xdf, stringsAsFactors = FALSE)
  for (j in seq_len(p)) {
    out[[paste0(cn[j], "_sq")]] <- Xdf[[j]]^2
  }
  for (i in seq_len(p)) {
    for (j in seq_len(p)) {
      if (j <= i) next
      out[[paste0(cn[i], "_x_", cn[j])]] <- Xdf[[i]] * Xdf[[j]]
    }
  }
  out
}

as_mat <- function(df) {
  m <- as.matrix(df)
  storage.mode(m) <- "double"
  m
}

fit_cv_enet <- function(X_mat, y, seed) {
  if (length(y) < 15L) return(NULL)
  set.seed(seed)
  tryCatch(
    cv.glmnet(X_mat, y, family = "gaussian", alpha = ENET_ALPHA),
    error = function(e) NULL
  )
}

predict_enet <- function(fit, X_mat) {
  if (is.null(fit)) return(NULL)
  as.numeric(predict(fit, newx = X_mat, s = "lambda.min"))
}

cat("=== 10-fold CV: separate Elastic Net (poly2) by is_red ===\n")
cat(sprintf("alpha = %.2f  folds = %d  seed = %d\n\n", ENET_ALPHA, K_FOLDS, CV_SEED))

raw <- load_train_data()
if (!"is_red" %in% colnames(raw)) {
  stop("Column is_red not found in training data.")
}

folds <- create_cv_folds(raw$quality, k = K_FOLDS, seed = CV_SEED)
fold_acc <- numeric(length(folds))

for (fi in seq_along(folds)) {
  test_idx <- folds[[fi]]
  train_fold <- raw[-test_idx, ]
  test_fold <- raw[test_idx, ]

  proc_train <- preprocess(train_fold)
  proc_test <- apply_preprocess(test_fold, proc_train$scale_params)

  X_tr <- proc_train$X
  y_tr <- proc_train$y
  X_te <- proc_test$X
  y_te <- proc_test$y

  cont_cols <- setdiff(colnames(X_tr), "is_red")
  isr_tr <- X_tr$is_red == 1L
  isw_tr <- X_tr$is_red == 0L

  seed_r <- CV_SEED + 1000L * as.integer(fi)
  seed_w <- CV_SEED + 2000L * as.integer(fi)

  preds_te <- rep(NA_real_, nrow(X_te))

  fit_r <- NULL
  fit_w <- NULL

  if (any(isr_tr)) {
    Xr <- expand_linear_sq_interact(X_tr[isr_tr, cont_cols, drop = FALSE])
    Mr <- as_mat(Xr)
    fit_r <- fit_cv_enet(Mr, y_tr[isr_tr], seed_r)
  }
  if (any(isw_tr)) {
    Xw <- expand_linear_sq_interact(X_tr[isw_tr, cont_cols, drop = FALSE])
    Mw <- as_mat(Xw)
    fit_w <- fit_cv_enet(Mw, y_tr[isw_tr], seed_w)
  }

  isr_te <- X_te$is_red == 1L

  if (any(isr_te)) {
    Xr_te <- expand_linear_sq_interact(X_te[isr_te, cont_cols, drop = FALSE])
    Mr_te <- as_mat(Xr_te)
    if (!is.null(fit_r)) {
      pr <- predict_enet(fit_r, Mr_te)
      if (!is.null(pr)) preds_te[isr_te] <- pr
    }
  }
  if (any(!isr_te)) {
    Xw_te <- expand_linear_sq_interact(X_te[!isr_te, cont_cols, drop = FALSE])
    Mw_te <- as_mat(Xw_te)
    if (!is.null(fit_w)) {
      pw <- predict_enet(fit_w, Mw_te)
      if (!is.null(pw)) preds_te[!isr_te] <- pw
    }
  }

  bad <- is.na(preds_te)
  if (any(bad)) {
    Xpool <- expand_linear_sq_interact(X_tr[, cont_cols, drop = FALSE])
    Mpool <- as_mat(Xpool)
    fit_p <- fit_cv_enet(Mpool, y_tr, CV_SEED + 3000L * as.integer(fi))
    if (!is.null(fit_p)) {
      Xall_te <- expand_linear_sq_interact(X_te[, cont_cols, drop = FALSE])
      Mall_te <- as_mat(Xall_te)
      p_all <- predict_enet(fit_p, Mall_te)
      if (!is.null(p_all)) preds_te[bad] <- p_all[bad]
    }
  }

  fold_acc[fi] <- compute_accuracy(y_te, preds_te)
  cat(sprintf("Fold %2d accuracy: %.4f\n", fi, fold_acc[fi]))
}

mean_acc <- mean(fold_acc)
sd_acc <- sd(fold_acc)
cat("\n")
cat(sprintf("Mean CV accuracy: %.4f  SD: %.4f\n", mean_acc, sd_acc))

output_dir <- "outputs/results"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
write.csv(
  data.frame(
    method = "separate_elastic_net_by_is_red_poly2",
    k_folds = K_FOLDS,
    enet_alpha = ENET_ALPHA,
    features = "linear + squares + pairwise_interactions (continuous only)",
    mean_cv_accuracy = mean_acc,
    sd_cv_accuracy = sd_acc,
    stringsAsFactors = FALSE
  ),
  file.path(output_dir, "enet_red_white_poly_cv_summary.csv"),
  row.names = FALSE
)
write.csv(
  data.frame(fold = seq_along(fold_acc), cv_accuracy = fold_acc),
  file.path(output_dir, "enet_red_white_poly_cv_by_fold.csv"),
  row.names = FALSE
)
cat("Wrote:", file.path(output_dir, "enet_red_white_poly_cv_summary.csv"), "\n")
