#!/usr/bin/env Rscript
# Stratified k-fold CV for CatBoost multiclass (wine quality).

source("src/helpers.R")

if (!requireNamespace("catboost", quietly = TRUE)) {
  stop(
    "Package 'catboost' is not installed. Run: Rscript src/install_packages.R\n",
    "Or see README for CatBoost R installation."
  )
}

K_FOLDS <- 5L
CV_SEED <- SEED

train_predict_catboost <- function(X_train, y_train, X_test) {
  offset <- min(y_train)
  labels <- y_train - offset
  train_pool <- catboost::catboost.load_pool(
    data = as.data.frame(X_train),
    label = labels
  )
  test_pool <- catboost::catboost.load_pool(data = as.data.frame(X_test))
  params <- list(
    loss_function = "MultiClass",
    iterations = 200,
    depth = 6,
    learning_rate = 0.1,
    verbose = 0,
    thread_count = 1
  )
  fit <- catboost::catboost.train(train_pool, params = params)
  preds <- catboost::catboost.predict(
    fit, test_pool,
    prediction_type = "Class"
  )
  as.integer(preds) + offset
}

cat("=== CatBoost cross-validation ===\n")
cat(sprintf("Folds: %d  seed: %d\n\n", K_FOLDS, CV_SEED))

raw <- load_train_data()
folds <- create_cv_folds(raw$quality, k = K_FOLDS, seed = CV_SEED)

fold_acc <- numeric(length(folds))
fold_rmse <- numeric(length(folds))

for (i in seq_along(folds)) {
  test_idx <- folds[[i]]
  train_fold <- raw[-test_idx, ]
  test_fold <- raw[test_idx, ]

  proc_train <- preprocess(train_fold)
  proc_test <- apply_preprocess(test_fold, proc_train$scale_params)

  preds <- train_predict_catboost(proc_train$X, proc_train$y, proc_test$X)
  fold_acc[i] <- compute_accuracy(proc_test$y, preds)
  fold_rmse[i] <- compute_rmse(proc_test$y, preds)

  cat(sprintf("Fold %d  accuracy: %.4f  RMSE: %.4f\n", i, fold_acc[i], fold_rmse[i]))
}

mean_acc <- mean(fold_acc)
sd_acc <- sd(fold_acc)
mean_rmse <- mean(fold_rmse)
sd_rmse <- sd(fold_rmse)

cat("\n")
cat(sprintf("Mean CV accuracy: %.4f (SD %.4f)\n", mean_acc, sd_acc))
cat(sprintf("Mean CV RMSE:     %.4f (SD %.4f)\n", mean_rmse, sd_rmse))

output_dir <- "outputs/results"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
summary_df <- data.frame(
  model = "CatBoost (MultiClass)",
  k_folds = K_FOLDS,
  mean_cv_accuracy = mean_acc,
  sd_cv_accuracy = sd_acc,
  mean_cv_rmse = mean_rmse,
  sd_cv_rmse = sd_rmse,
  stringsAsFactors = FALSE
)
write.csv(summary_df, file.path(output_dir, "cv_catboost_summary.csv"), row.names = FALSE)
write.csv(
  data.frame(
    fold = seq_along(fold_acc),
    cv_accuracy = fold_acc,
    cv_rmse = fold_rmse
  ),
  file.path(output_dir, "cv_catboost_by_fold.csv"),
  row.names = FALSE
)
cat("\nWrote:", file.path(output_dir, "cv_catboost_summary.csv"), "\n")
