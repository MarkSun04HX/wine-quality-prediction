#!/usr/bin/env Rscript
# Cross-validation: mean of regression tree (rpart) and elastic net (glmnet)
# predictions, then round and clip to integer quality 3–9 for accuracy.

suppressPackageStartupMessages({
  library(glmnet)
  library(rpart)
})

source("src/helpers.R")

K_FOLDS <- 5
CV_SEED <- 42

fit_predict_enet_reg <- function(X_train, y_train, X_test, seed) {
  X_mat <- as.matrix(X_train)
  set.seed(seed)
  cv_fit <- cv.glmnet(X_mat, y_train, family = "gaussian", alpha = 0.5)
  as.numeric(predict(cv_fit, newx = as.matrix(X_test), s = "lambda.min"))
}

fit_predict_rpart_reg <- function(X_train, y_train, X_test) {
  df <- data.frame(quality = y_train, X_train, check.names = FALSE)
  fit <- rpart(quality ~ ., data = df, method = "anova")
  as.numeric(predict(fit, newdata = data.frame(X_test, check.names = FALSE)))
}

cat("=== CV: mean(Decision tree, Elastic net) -> round -> accuracy ===\n")
cat(sprintf("Folds: %d  stratify seed: %d\n\n", K_FOLDS, CV_SEED))

raw <- load_train_data()
folds <- create_cv_folds(raw$quality, k = K_FOLDS, seed = CV_SEED)

fold_acc <- numeric(length(folds))

for (i in seq_along(folds)) {
  test_idx <- folds[[i]]
  train_fold <- raw[-test_idx, ]
  test_fold <- raw[test_idx, ]

  proc_train <- preprocess(train_fold)
  proc_test <- apply_preprocess(test_fold, proc_train$scale_params)

  X_tr <- proc_train$X
  y_tr <- proc_train$y
  X_te <- proc_test$X
  y_te <- proc_test$y

  seed_i <- CV_SEED + 1000L * as.integer(i)
  p_tree <- fit_predict_rpart_reg(X_tr, y_tr, X_te)
  p_enet <- fit_predict_enet_reg(X_tr, y_tr, X_te, seed_i)
  p_mean <- (p_tree + p_enet) / 2

  fold_acc[i] <- compute_accuracy(y_te, p_mean)
  cat(sprintf("Fold %d accuracy: %.4f\n", i, fold_acc[i]))
}

mean_acc <- mean(fold_acc)
sd_acc <- sd(fold_acc)

cat("\n")
cat(sprintf("Mean CV accuracy: %.4f  SD: %.4f\n", mean_acc, sd_acc))

output_dir <- "outputs/results"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
out <- data.frame(
  model = "mean(rpart, elastic_net) then round",
  k_folds = K_FOLDS,
  mean_cv_accuracy = mean_acc,
  sd_cv_accuracy = sd_acc,
  stringsAsFactors = FALSE
)
write.csv(out, file.path(output_dir, "mean_tree_enet_cv_summary.csv"), row.names = FALSE)
write.csv(
  data.frame(fold = seq_along(fold_acc), cv_accuracy = fold_acc),
  file.path(output_dir, "mean_tree_enet_cv_by_fold.csv"),
  row.names = FALSE
)
cat("Wrote:", file.path(output_dir, "mean_tree_enet_cv_summary.csv"), "\n")
