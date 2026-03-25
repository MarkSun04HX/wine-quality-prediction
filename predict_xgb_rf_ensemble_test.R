#!/usr/bin/env Rscript
# Train Random Forest + XGBoost on train.csv, predict test.csv.
# Ensemble: mean(RF class, XGB class); accuracy uses ceiling + clip to 3–9.

suppressPackageStartupMessages({
  library(randomForest)
  library(xgboost)
})

source("src/helpers.R")

TEST_PATH <- "data/processed/test.csv"
NTREE <- 500L
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

set.seed(MODEL_SEED)
fit_rf <- randomForest(
  x = proc_train$X,
  y = factor(y_tr),
  ntree = NTREE
)
pred_rf <- as.numeric(as.character(predict(fit_rf, newdata = proc_test$X)))

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
fit_xgb <- xgb.train(params, dtrain, nrounds = NROUNDS, verbose = 0)
pred_xgb <- as.numeric(predict(fit_xgb, dtest)) + offset

pred_mean <- (pred_rf + pred_xgb) / 2

acc_rf <- accuracy_ceiling(y_te, pred_rf)
acc_xgb <- accuracy_ceiling(y_te, pred_xgb)
acc_ens <- accuracy_ceiling(y_te, pred_mean)

cat("=== XGBoost + Random Forest ensemble (mean) | train -> test ===\n")
cat(sprintf("RF trees: %d | XGB rounds: %d | seed: %d\n", NTREE, NROUNDS, MODEL_SEED))
cat(sprintf("Test rows: %d\n", nrow(test_raw)))
cat(sprintf("Accuracy (ceiling, clipped 3–9) — RF:       %.4f\n", acc_rf))
cat(sprintf("Accuracy (ceiling, clipped 3–9) — XGBoost:  %.4f\n", acc_xgb))
cat(sprintf("Accuracy (ceiling, clipped 3–9) — Ensemble: %.4f\n", acc_ens))

output_dir <- "outputs/results"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

out_df <- data.frame(
  actual_quality = y_te,
  pred_rf = pred_rf,
  pred_xgb = pred_xgb,
  pred_ensemble_mean = pred_mean,
  pred_ensemble_ceiling = as.integer(pmax(pmin(ceiling(pred_mean), 9L), 3L)),
  stringsAsFactors = FALSE
)
write.csv(
  out_df,
  file.path(output_dir, "xgb_rf_ensemble_test_predictions.csv"),
  row.names = FALSE
)

acc_df <- data.frame(
  model = c(
    sprintf("Random Forest (%d trees)", NTREE),
    sprintf("XGBoost (%d rounds)", NROUNDS),
    "Mean(RF, XGB) + ceiling"
  ),
  accuracy_ceiling_clipped = c(acc_rf, acc_xgb, acc_ens),
  stringsAsFactors = FALSE
)
write.csv(
  acc_df,
  file.path(output_dir, "xgb_rf_ensemble_test_accuracy.csv"),
  row.names = FALSE
)

cat("\nWrote:\n  ", file.path(output_dir, "xgb_rf_ensemble_test_predictions.csv"), "\n", sep = "")
cat("  ", file.path(output_dir, "xgb_rf_ensemble_test_accuracy.csv"), "\n", sep = "")
