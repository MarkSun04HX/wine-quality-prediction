source("src/helpers.R")
source("src/models.R")

cat("=== CV Accuracy: Elastic Net, Random Forest, XGBoost ===\n\n")

output_dir <- "outputs/results"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# You can edit these seeds as needed.
seeds <- c(42, 7, 123, 2024, 999)
k_folds <- 5

selected_model_names <- c(
  "Elastic Net Logistic",
  "Random Forest (clf)",
  "XGBoost (clf)"
)

selected_models <- MODEL_REGISTRY[sapply(MODEL_REGISTRY, function(m) {
  m$name %in% selected_model_names
})]

if (length(selected_models) != length(selected_model_names)) {
  stop("Could not find all selected models in MODEL_REGISTRY.")
}

raw <- load_train_data()
all_results <- data.frame()

for (seed in seeds) {
  cat("Running seed:", seed, "\n")
  folds <- create_cv_folds(raw$quality, k = k_folds, seed = seed)

  fold_data <- vector("list", length(folds))
  for (i in seq_along(folds)) {
    test_idx <- folds[[i]]
    train_fold <- raw[-test_idx, ]
    test_fold <- raw[test_idx, ]

    proc_train <- preprocess(train_fold)
    proc_test <- apply_preprocess(test_fold, proc_train$scale_params)

    fold_data[[i]] <- list(
      X_train = proc_train$X, y_train = proc_train$y,
      X_test = proc_test$X, y_test = proc_test$y
    )
  }

  for (m in selected_models) {
    fold_acc <- numeric(length(fold_data))
    skip_model <- FALSE

    for (i in seq_along(fold_data)) {
      fd <- fold_data[[i]]
      res <- tryCatch(
        m$fn(fd$X_train, fd$y_train, fd$X_test),
        error = function(e) {
          cat("  ", m$name, "fold", i, "error:", conditionMessage(e), "\n")
          NULL
        }
      )

      if (is.null(res)) {
        skip_model <- TRUE
        break
      }

      fold_acc[i] <- compute_accuracy(fd$y_test, res$predictions)
    }

    if (!skip_model) {
      mean_acc <- mean(fold_acc)
      all_results <- rbind(
        all_results,
        data.frame(
          seed = seed,
          model = m$name,
          mean_cv_accuracy = mean_acc,
          stringsAsFactors = FALSE
        )
      )
      cat(sprintf("  %-22s mean CV accuracy: %.4f\n", m$name, mean_acc))
    }
  }
  cat("\n")
}

summary_results <- aggregate(mean_cv_accuracy ~ model, data = all_results, FUN = mean)
colnames(summary_results)[2] <- "mean_cv_accuracy_across_seeds"

write.csv(all_results,
          file.path(output_dir, "three_models_cv_by_seed.csv"),
          row.names = FALSE)
write.csv(summary_results,
          file.path(output_dir, "three_models_cv_mean_across_seeds.csv"),
          row.names = FALSE)

cat("Per-seed results saved to:",
    file.path(output_dir, "three_models_cv_by_seed.csv"), "\n")
cat("Mean across seeds saved to:",
    file.path(output_dir, "three_models_cv_mean_across_seeds.csv"), "\n\n")

cat("=== Mean CV Accuracy Across Seeds ===\n")
print(summary_results[order(-summary_results$mean_cv_accuracy_across_seeds), ])
