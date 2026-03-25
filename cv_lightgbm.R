#!/usr/bin/env Rscript
# Stratified k-fold CV: LightGBM multiclass (wine quality).

if (!requireNamespace("lightgbm", quietly = TRUE)) {
  stop(
    "Package 'lightgbm' is not installed. Run: Rscript src/install_packages.R\n",
    "or: install.packages('lightgbm')"
  )
}

suppressPackageStartupMessages(library(lightgbm))

source("src/helpers.R")

K_FOLDS <- 5L
CV_SEED <- SEED

train_predict_lgb <- function(X_train, y_train, X_test, seed) {
  base <- as.integer(min(y_train))
  lab <- as.integer(y_train - base)
  nc <- max(lab) + 1L

  dtrain <- lgb.Dataset(data = as.matrix(X_train), label = lab)

  params <- list(
    objective = "multiclass",
    num_class = nc,
    learning_rate = 0.1,
    num_leaves = 31L,
    max_depth = 6L,
    verbosity = -1L,
    num_threads = 1L,
    seed = as.integer(seed)
  )

  fit <- lgb.train(
    params = params,
    data = dtrain,
    nrounds = 200L,
    verbose = -1L
  )

  n_te <- nrow(X_test)
  # lightgbm >= 4.x: use newdata= (data= is ignored and triggers "newdata missing")
  pv <- predict(fit, newdata = as.matrix(X_test))
  # Newer lightgbm drops reshape=TRUE; multiclass returns length n_te * nc (row-major).
  if (length(pv) != n_te * nc) {
    stop(
      "Unexpected predict length (", length(pv), ") for n=", n_te, " classes=", nc
    )
  }
  pm <- matrix(pv, nrow = n_te, ncol = nc, byrow = TRUE)
  pred_idx0 <- max.col(pm) - 1L
  as.numeric(base + pred_idx0)
}

cat("=== Cross-validation: LightGBM (multiclass) ===\n")
cat(sprintf("Folds: %d  stratify seed: %d\n\n", K_FOLDS, CV_SEED))

raw <- load_train_data()
folds <- create_cv_folds(raw$quality, k = K_FOLDS, seed = CV_SEED)

lgb_acc <- numeric(K_FOLDS)
lgb_rmse <- numeric(K_FOLDS)

for (i in seq_along(folds)) {
  test_idx <- folds[[i]]
  train_fold <- raw[-test_idx, ]
  test_fold <- raw[test_idx, ]

  proc_train <- preprocess(train_fold)
  proc_test <- apply_preprocess(test_fold, proc_train$scale_params)

  seed_i <- CV_SEED + 1000L * as.integer(i)
  p_lgb <- train_predict_lgb(proc_train$X, proc_train$y, proc_test$X, seed_i + 17L)
  lgb_acc[i] <- compute_accuracy(proc_test$y, p_lgb)
  lgb_rmse[i] <- compute_rmse(proc_test$y, p_lgb)

  cat(sprintf("Fold %d | acc %.4f | rmse %.4f\n", i, lgb_acc[i], lgb_rmse[i]))
}

summary_df <- data.frame(
  model = "LightGBM (multiclass)",
  k_folds = K_FOLDS,
  mean_cv_accuracy = mean(lgb_acc),
  sd_cv_accuracy = sd(lgb_acc),
  mean_cv_rmse = mean(lgb_rmse),
  sd_cv_rmse = sd(lgb_rmse),
  stringsAsFactors = FALSE
)

cat("\n--- Summary ---\n")
print(summary_df)

output_dir <- "outputs/results"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
write.csv(summary_df, file.path(output_dir, "cv_lightgbm_summary.csv"), row.names = FALSE)
write.csv(
  data.frame(
    fold = seq_len(K_FOLDS),
    accuracy = lgb_acc,
    rmse = lgb_rmse
  ),
  file.path(output_dir, "cv_lightgbm_by_fold.csv"),
  row.names = FALSE
)
cat("\nWrote:", file.path(output_dir, "cv_lightgbm_summary.csv"), "\n")
