#!/usr/bin/env Rscript
# Train XGBoost (multiclass) on data/processed/train.csv, predict test.csv.
# Accuracy: ceiling(prediction), clipped to integer quality 3–9.

suppressPackageStartupMessages(library(xgboost))

source("src/helpers.R")

TEST_PATH <- "data/processed/test.csv"
NROUNDS <- 200L
MODEL_SEED <- SEED

accuracy_ceiling <- function(actual, predicted) {
  pred_int <- as.integer(pmax(pmin(ceiling(as.numeric(predicted)), 9L), 3L))
  mean(as.integer(actual) == pred_int)
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

X_tr <- as.matrix(proc_train$X)
X_te <- as.matrix(proc_test$X)
y_tr <- proc_train$y
y_te <- proc_test$y

offset <- min(y_tr)
labels <- as.integer(y_tr - offset)
num_class <- max(labels) + 1L

dtrain <- xgb.DMatrix(data = X_tr, label = labels)
dtest <- xgb.DMatrix(data = X_te)

params <- list(
  objective = "multi:softmax",
  num_class = num_class,
  max_depth = 6,
  eta = 0.1,
  nthread = 1
)

set.seed(MODEL_SEED)
fit <- xgb.train(params, dtrain, nrounds = NROUNDS, verbose = 0)

pred_num <- as.numeric(predict(fit, dtest)) + offset
acc <- accuracy_ceiling(y_te, pred_num)

cat("=== XGBoost (multi:softmax) | train -> test ===\n")
cat(sprintf("Rounds: %d  seed: %d\n", NROUNDS, MODEL_SEED))
cat(sprintf("Test rows: %d\n", nrow(test_raw)))
cat(sprintf("Accuracy (ceiling, clipped 3–9): %.4f\n", acc))

output_dir <- "outputs/results"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

out_df <- data.frame(
  actual_quality = y_te,
  pred_xgb_class = pred_num,
  pred_xgb_ceiling = as.integer(pmax(pmin(ceiling(pred_num), 9L), 3L)),
  stringsAsFactors = FALSE
)
write.csv(out_df, file.path(output_dir, "xgb_test_predictions.csv"), row.names = FALSE)
write.csv(
  data.frame(
    model = sprintf("XGBoost multiclass (%d rounds)", NROUNDS),
    accuracy_ceiling_clipped = acc,
    stringsAsFactors = FALSE
  ),
  file.path(output_dir, "xgb_test_accuracy.csv"),
  row.names = FALSE
)

cat("\nWrote:\n  ", file.path(output_dir, "xgb_test_predictions.csv"), "\n", sep = "")
cat("  ", file.path(output_dir, "xgb_test_accuracy.csv"), "\n", sep = "")
