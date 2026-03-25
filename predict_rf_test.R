#!/usr/bin/env Rscript
# Train Random Forest on data/processed/train.csv, predict data/processed/test.csv.
# Accuracy: ceiling(prediction), clipped to integer quality 3–9 (same rule as nn_pick test script).

suppressPackageStartupMessages(library(randomForest))

source("src/helpers.R")

TEST_PATH <- "data/processed/test.csv"
NTREE <- 500L
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

set.seed(MODEL_SEED)
fit <- randomForest(
  x = proc_train$X,
  y = factor(proc_train$y),
  ntree = NTREE
)

pred_factor <- predict(fit, newdata = proc_test$X)
pred_num <- as.numeric(as.character(pred_factor))
y_te <- proc_test$y

acc <- accuracy_ceiling(y_te, pred_num)

cat("=== Random Forest | train -> test ===\n")
cat(sprintf("Trees: %d  seed: %d\n", NTREE, MODEL_SEED))
cat(sprintf("Test rows: %d\n", nrow(test_raw)))
cat(sprintf("Accuracy (ceiling, clipped 3–9): %.4f\n", acc))

output_dir <- "outputs/results"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

out_df <- data.frame(
  actual_quality = y_te,
  pred_rf_class = pred_num,
  pred_rf_ceiling = as.integer(pmax(pmin(ceiling(pred_num), 9L), 3L)),
  stringsAsFactors = FALSE
)
write.csv(out_df, file.path(output_dir, "rf_test_predictions.csv"), row.names = FALSE)
write.csv(
  data.frame(
    model = sprintf("Random Forest (%d trees)", NTREE),
    accuracy_ceiling_clipped = acc,
    stringsAsFactors = FALSE
  ),
  file.path(output_dir, "rf_test_accuracy.csv"),
  row.names = FALSE
)

cat("\nWrote:\n  ", file.path(output_dir, "rf_test_predictions.csv"), "\n", sep = "")
cat("  ", file.path(output_dir, "rf_test_accuracy.csv"), "\n", sep = "")
