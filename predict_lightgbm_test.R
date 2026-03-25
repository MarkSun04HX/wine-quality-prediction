#!/usr/bin/env Rscript
# Train LightGBM (multiclass) on data/processed/train.csv, predict test.csv.
# Accuracy: ceiling(prediction), clipped to integer quality 3–9.

if (!requireNamespace("lightgbm", quietly = TRUE)) {
  stop(
    "Package 'lightgbm' is not installed. Run: Rscript src/install_packages.R\n",
    "or: install.packages('lightgbm')"
  )
}

suppressPackageStartupMessages(library(lightgbm))

source("src/helpers.R")

TEST_PATH <- "data/processed/test.csv"
NROUNDS <- 200L
MODEL_SEED <- SEED

accuracy_ceiling <- function(actual, predicted) {
  pred_int <- as.integer(pmax(pmin(ceiling(as.numeric(predicted)), 9L), 3L))
  mean(as.integer(actual) == pred_int)
}

predict_lgb_multiclass <- function(X_train, y_train, X_test, seed) {
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
    nrounds = NROUNDS,
    verbose = -1L
  )

  n_te <- nrow(X_test)
  pv <- predict(fit, newdata = as.matrix(X_test))
  if (length(pv) != n_te * nc) {
    stop(
      "Unexpected predict length (", length(pv), ") for n=", n_te, " classes=", nc
    )
  }
  pm <- matrix(pv, nrow = n_te, ncol = nc, byrow = TRUE)

  pred_idx0 <- max.col(pm) - 1L
  as.numeric(base + pred_idx0)
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

pred_num <- predict_lgb_multiclass(
  proc_train$X, proc_train$y, proc_test$X, MODEL_SEED
)
y_te <- proc_test$y
acc <- accuracy_ceiling(y_te, pred_num)

cat("=== LightGBM (multiclass) | train -> test ===\n")
cat(sprintf("Rounds: %d  seed: %d\n", NROUNDS, MODEL_SEED))
cat(sprintf("Test rows: %d\n", nrow(test_raw)))
cat(sprintf("Accuracy (ceiling, clipped 3–9): %.4f\n", acc))

output_dir <- "outputs/results"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

out_df <- data.frame(
  actual_quality = y_te,
  pred_lgb_class = pred_num,
  pred_lgb_ceiling = as.integer(pmax(pmin(ceiling(pred_num), 9L), 3L)),
  stringsAsFactors = FALSE
)
write.csv(out_df, file.path(output_dir, "lgb_test_predictions.csv"), row.names = FALSE)
write.csv(
  data.frame(
    model = sprintf("LightGBM multiclass (%d rounds)", NROUNDS),
    accuracy_ceiling_clipped = acc,
    stringsAsFactors = FALSE
  ),
  file.path(output_dir, "lgb_test_accuracy.csv"),
  row.names = FALSE
)

cat("\nWrote:\n  ", file.path(output_dir, "lgb_test_predictions.csv"), "\n", sep = "")
cat("  ", file.path(output_dir, "lgb_test_accuracy.csv"), "\n", sep = "")
