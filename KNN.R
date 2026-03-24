#!/usr/bin/env Rscript
# KNN Parameter Tuning: 10-Fold CV with Accuracy Plotting

suppressPackageStartupMessages({
  library(FNN)
  library(ggplot2)
})

source("src/helpers.R")

K_FOLDS <- 10
CV_SEED <- 42
K_VALUES <- 2:50

# Load and prepare data
raw <- load_train_data()
folds <- create_cv_folds(raw$quality, k = K_FOLDS, seed = CV_SEED)

# Matrix to store accuracy
tuning_results <- matrix(0, nrow = K_FOLDS, ncol = length(K_VALUES))
colnames(tuning_results) <- paste0("K_", K_VALUES)

cat(sprintf("Running 10-fold CV for K = 2 to 50...\n"))

for (i in seq_along(folds)) {
  test_idx <- folds[[i]]
  proc_tr <- preprocess(raw[-test_idx, ])
  proc_te <- apply_preprocess(raw[test_idx, ], proc_tr$scale_params)
  
  X_tr <- as.matrix(proc_tr$X)
  y_tr <- proc_tr$y
  X_te <- as.matrix(proc_te$X)
  
  for (j in seq_along(K_VALUES)) {
    k_val <- K_VALUES[j]
    knn_mod <- knn.reg(train = X_tr, test = X_te, y = y_tr, k = k_val)
    tuning_results[i, j] <- compute_accuracy(proc_te$y, knn_mod$pred)
  }
}

# --- Analyze Results ---
final_stats <- data.frame(
  K = K_VALUES,
  Mean_Accuracy = colMeans(tuning_results),
  SD = apply(tuning_results, 2, sd)
)

best_k <- final_stats$K[which.max(final_stats$Mean_Accuracy)]
max_acc <- max(final_stats$Mean_Accuracy)

# --- Generate Diagram ---
# We use error bars (Mean +/- SD) to show the stability of each K
plot <- ggplot(final_stats, aes(x = K, y = Mean_Accuracy)) +
  geom_line(color = "#2c3e50", size = 1) +
  geom_point(color = "#e74c3c", size = 2) +
  geom_ribbon(aes(ymin = Mean_Accuracy - SD, ymax = Mean_Accuracy + SD), 
              fill = "#3498db", alpha = 0.2) +
  geom_vline(xintercept = best_k, linetype = "dashed", color = "darkgreen") +
  annotate("text", x = best_k + 5, y = max_acc, 
           label = paste("Best K =", best_k), color = "darkgreen") +
  labs(title = "KNN Tuning: Mean Accuracy vs. K (Neighbors)",
       subtitle = "10-Fold CV (Shaded area represents ±1 Standard Deviation)",
       x = "Number of Neighbors (K)",
       y = "Mean Accuracy") +
  theme_minimal()

# Save the plot
output_dir <- "outputs/results"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
ggsave(file.path(output_dir, "knn_tuning_plot.png"), plot, width = 8, height = 5)

cat("\n--- Result ---\n")
cat(sprintf("Best K: %d with Accuracy: %.4f\n", best_k, max_acc))
cat("Diagram saved to:", file.path(output_dir, "knn_tuning_plot.png"), "\n")