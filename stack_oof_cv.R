#!/usr/bin/env Rscript
# 10-fold stratified CV (single seed) with stacking:
# Base learners: OLS, multinomial logistic (nnet), Elastic Net regression (glmnet),
# KNN classifier, Lasso multinomial logistic (glmnet). Feature engineering on raw
# rows before preprocess. Inner CV on each outer training set builds OOF preds;
# OLS meta-learner on OOF; bases refit on full outer train for test prediction.
# Reports stacked and per-base outer-fold accuracy & RMSE.

suppressPackageStartupMessages({
  library(nnet)
  library(glmnet)
})

source("src/helpers.R")

OUTER_K <- 10L
INNER_K <- 5L
CV_SEED <- 42L
KNN_K <- 7L

# Row-wise feature engineering (no global stats — safe before CV splits).
engineer_wine_features <- function(df) {
  out <- df
  out$ratio_free_total_so2 <- out[["free.sulfur.dioxide"]] /
    (out[["total.sulfur.dioxide"]] + 1e-6)
  out$alcohol_x_density <- out$alcohol * out$density
  out$fixed_plus_volatile_acidity <- out[["fixed.acidity"]] +
    out[["volatile.acidity"]]
  chem <- c(
    "fixed.acidity", "volatile.acidity", "citric.acid", "residual.sugar",
    "chlorides", "free.sulfur.dioxide", "total.sulfur.dioxide",
    "density", "pH", "sulphates", "alcohol"
  )
  for (nm in chem) {
    out[[paste0(nm, ".sq")]] <- out[[nm]]^2
  }
  out
}

pred_ols <- function(X_train, y_train, X_test) {
  df_tr <- data.frame(y = y_train, X_train, check.names = FALSE)
  fit <- lm(y ~ ., data = df_tr)
  as.numeric(predict(fit, newdata = data.frame(X_test, check.names = FALSE)))
}

pred_multinom <- function(X_train, y_train, X_test) {
  df_tr <- data.frame(y = factor(y_train), X_train, check.names = FALSE)
  fit <- suppressWarnings(
    multinom(y ~ ., data = df_tr, trace = FALSE, MaxNWts = 10000)
  )
  p <- predict(fit, newdata = data.frame(X_test, check.names = FALSE))
  as.numeric(as.integer(as.character(p)))
}

pred_enet_reg <- function(X_train, y_train, X_test, seed) {
  set.seed(seed)
  xm <- as.matrix(X_train)
  cvf <- cv.glmnet(xm, y_train, family = "gaussian", alpha = 0.5)
  as.numeric(predict(cvf, newx = as.matrix(X_test), s = "lambda.min"))
}

pred_knn <- function(X_train, y_train, X_test) {
  ntr <- nrow(X_train)
  k_use <- max(1L, min(as.integer(KNN_K), ntr - 1L))
  p <- class::knn(
    train = as.matrix(X_train),
    test = as.matrix(X_test),
    cl = factor(y_train),
    k = k_use
  )
  as.numeric(as.integer(as.character(p)))
}

pred_lasso_mn <- function(X_train, y_train, X_test, seed) {
  set.seed(seed)
  xm <- as.matrix(X_train)
  yf <- factor(y_train)
  cvf <- cv.glmnet(
    xm, yf, family = "multinomial", alpha = 1, type.measure = "class"
  )
  pr <- predict(cvf, newx = as.matrix(X_test), s = "lambda.min", type = "class")
  pv <- if (is.matrix(pr)) pr[, 1L] else pr
  as.numeric(as.integer(as.character(pv)))
}

BASE_NAMES <- c(
  "ols", "multinom", "enet_reg", "knn", "lasso_mn"
)

predict_base_row <- function(which_base, X_tr, y_tr, X_te, seed) {
  switch(
    which_base,
    ols = pred_ols(X_tr, y_tr, X_te),
    multinom = pred_multinom(X_tr, y_tr, X_te),
    enet_reg = pred_enet_reg(X_tr, y_tr, X_te, seed),
    knn = pred_knn(X_tr, y_tr, X_te),
    lasso_mn = pred_lasso_mn(X_tr, y_tr, X_te, seed),
    stop("unknown base ", which_base)
  )
}

build_oof_matrix <- function(df_train, seed_outer, seed_inner_offset) {
  n <- nrow(df_train)
  y <- df_train$quality
  oof <- matrix(NA_real_, nrow = n, ncol = length(BASE_NAMES))
  colnames(oof) <- BASE_NAMES

  set.seed(seed_outer + seed_inner_offset)
  inner_folds <- createFolds(factor(y), k = INNER_K, list = TRUE, returnTrain = FALSE)

  for (inf in seq_along(inner_folds)) {
    va <- inner_folds[[inf]]
    tr <- setdiff(seq_len(n), va)

    proc_tr <- preprocess(df_train[tr, , drop = FALSE])
    proc_va <- apply_preprocess(df_train[va, , drop = FALSE], proc_tr$scale_params)

    Xtr <- proc_tr$X
    ytr <- proc_tr$y
    Xva <- proc_va$X

    s_base <- seed_outer + seed_inner_offset + 17L * as.integer(inf)

    for (b in seq_along(BASE_NAMES)) {
      nm <- BASE_NAMES[b]
      oof[va, b] <- tryCatch(
        predict_base_row(nm, Xtr, ytr, Xva, s_base + b),
        error = function(e) rep(NA_real_, length(va))
      )
    }
  }

  oof
}

fit_meta_lm <- function(oof_mat, y) {
  ok <- rowSums(is.finite(oof_mat)) == ncol(oof_mat)
  if (sum(ok) < 50L) return(NULL)
  dd <- data.frame(
    y = y[ok],
    as.data.frame(oof_mat[ok, , drop = FALSE]),
    check.names = FALSE
  )
  tryCatch(
    lm(y ~ ., data = dd),
    error = function(e) NULL
  )
}

meta_predict <- function(meta_fit, pred_mat) {
  if (is.null(meta_fit)) {
    return(rep(NA_real_, nrow(pred_mat)))
  }
  nd <- as.data.frame(pred_mat, check.names = FALSE)
  colnames(nd) <- BASE_NAMES
  as.numeric(predict(meta_fit, newdata = nd))
}

cat("=== Stacked OOF (OLS meta) | 10-fold CV | seed ", CV_SEED, " ===\n", sep = "")
cat("Bases: OLS, multinomial logit, ENET regression, KNN, lasso multinomial\n")
cat("Inner folds for OOF: ", INNER_K, "\n\n", sep = "")

raw <- load_train_data()
eng <- engineer_wine_features(raw)

folds <- create_cv_folds(eng$quality, k = OUTER_K, seed = CV_SEED)

stack_acc <- numeric(OUTER_K)
stack_rmse <- numeric(OUTER_K)
base_acc <- matrix(NA_real_, OUTER_K, length(BASE_NAMES))
base_rmse <- matrix(NA_real_, OUTER_K, length(BASE_NAMES))
colnames(base_acc) <- BASE_NAMES
colnames(base_rmse) <- BASE_NAMES

for (of in seq_along(folds)) {
  te_idx <- folds[[of]]
  tr_idx <- setdiff(seq_len(nrow(eng)), te_idx)

  df_tr <- eng[tr_idx, , drop = FALSE]
  df_te <- eng[te_idx, , drop = FALSE]

  seed_o <- CV_SEED + 10000L * as.integer(of)

  oof <- build_oof_matrix(df_tr, seed_o, seed_o %% 997L)
  meta <- fit_meta_lm(oof, df_tr$quality)

  proc_full <- preprocess(df_tr)
  proc_te <- apply_preprocess(df_te, proc_full$scale_params)
  Xf <- proc_full$X
  yf <- proc_full$y
  Xe <- proc_te$X
  ye <- proc_te$y

  pred_te <- matrix(NA_real_, nrow = nrow(Xe), ncol = length(BASE_NAMES))
  colnames(pred_te) <- BASE_NAMES

  for (b in seq_along(BASE_NAMES)) {
    nm <- BASE_NAMES[b]
    pred_te[, b] <- tryCatch(
      predict_base_row(nm, Xf, yf, Xe, seed_o + 200L + b),
      error = function(e) rep(NA_real_, nrow(Xe))
    )
    base_acc[of, b] <- compute_accuracy(ye, pred_te[, b])
    base_rmse[of, b] <- compute_rmse(ye, pred_te[, b])
  }

  p_stack <- meta_predict(meta, pred_te)
  if (all(is.na(p_stack))) {
    p_stack <- rowMeans(pred_te, na.rm = TRUE)
  }
  stack_acc[of] <- compute_accuracy(ye, p_stack)
  stack_rmse[of] <- compute_rmse(ye, p_stack)

  cat(sprintf(
    "Fold %2d | stacked acc %.4f rmse %.4f\n",
    of, stack_acc[of], stack_rmse[of]
  ))
}

cat("\n--- Stacked (OOF + OLS meta) ---\n")
cat(sprintf(
  "Mean CV accuracy: %.4f (SD %.4f)\n",
  mean(stack_acc), sd(stack_acc)
))
cat(sprintf(
  "Mean CV RMSE:     %.4f (SD %.4f)\n",
  mean(stack_rmse), sd(stack_rmse)
))

cat("\n--- Base models (single fit on outer train, no stacking) ---\n")
for (b in seq_along(BASE_NAMES)) {
  nm <- BASE_NAMES[b]
  cat(sprintf(
    "  %-12s  acc %.4f (SD %.4f)  rmse %.4f (SD %.4f)\n",
    nm,
    mean(base_acc[, b]), sd(base_acc[, b]),
    mean(base_rmse[, b]), sd(base_rmse[, b])
  ))
}

output_dir <- "outputs/results"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

summary_stacked <- data.frame(
  method = "stacked_OOF_OLS_meta",
  outer_k = OUTER_K,
  inner_k = INNER_K,
  cv_seed = CV_SEED,
  mean_cv_accuracy = mean(stack_acc),
  sd_cv_accuracy = sd(stack_acc),
  mean_cv_rmse = mean(stack_rmse),
  sd_cv_rmse = sd(stack_rmse),
  stringsAsFactors = FALSE
)
write.csv(summary_stacked, file.path(output_dir, "stack_oof_cv_stacked_summary.csv"), row.names = FALSE)

base_summary <- data.frame(
  model = BASE_NAMES,
  mean_cv_accuracy = colMeans(base_acc),
  sd_cv_accuracy = apply(base_acc, 2, sd),
  mean_cv_rmse = colMeans(base_rmse),
  sd_cv_rmse = apply(base_rmse, 2, sd),
  stringsAsFactors = FALSE
)
write.csv(base_summary, file.path(output_dir, "stack_oof_cv_base_summary.csv"), row.names = FALSE)

write.csv(
  data.frame(
    fold = seq_len(OUTER_K),
    stacked_accuracy = stack_acc,
    stacked_rmse = stack_rmse,
    base_acc,
    check.names = FALSE
  ),
  file.path(output_dir, "stack_oof_cv_by_fold.csv"),
  row.names = FALSE
)

cat("\nWrote:\n  ", file.path(output_dir, "stack_oof_cv_stacked_summary.csv"), "\n", sep = "")
cat("  ", file.path(output_dir, "stack_oof_cv_base_summary.csv"), "\n", sep = "")
cat("  ", file.path(output_dir, "stack_oof_cv_by_fold.csv"), "\n", sep = "")
