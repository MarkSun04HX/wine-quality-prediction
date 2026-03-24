#!/usr/bin/env Rscript
# Ensemble CV: five base learners -> pick top 3 by RMSE -> majority vote,
# or if all three rounded labels disagree, mean of the two lowest-RMSE models.

suppressPackageStartupMessages({
  library(glmnet)
  library(nnet)
  library(rpart)
})

source("src/helpers.R")

cat("=== 10-fold CV: 5 base models -> top-3 RMSE ensemble ===\n\n")

output_dir <- "outputs/results"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

K_FOLDS <- 10
CV_FOLD_SEED <- 42

# --- KNN regression (Euclidean), k chosen by 3-fold RMSE on training fold ---

knn_reg_one <- function(X_tr, y_tr, X_te, k) {
  k <- max(1L, min(as.integer(k), nrow(X_tr)))
  X_tr <- as.matrix(X_tr)
  X_te <- as.matrix(X_te)
  apply(X_te, 1, function(x) {
    d <- sqrt(rowSums(sweep(X_tr, 2, x, FUN = "-")^2))
    ii <- order(d)[seq_len(k)]
    mean(y_tr[ii])
  })
}

tune_knn_reg <- function(X_train, y_train, seed) {
  n <- nrow(X_train)
  set.seed(seed)
  fold_ids <- sample(rep(1:3, length.out = n))
  best_k <- 5L
  best_rmse <- Inf
  for (k_try in c(5L, 7L, 11L)) {
    if (k_try >= n) next
    rmses <- numeric(3)
    ok <- TRUE
    for (f in 1:3) {
      tr <- fold_ids != f
      va <- fold_ids == f
      if (!any(va) || sum(tr) < k_try) {
        ok <- FALSE
        break
      }
      pred_va <- knn_reg_one(X_train[tr, , drop = FALSE], y_train[tr],
                             X_train[va, , drop = FALSE], k_try)
      rmses[f] <- sqrt(mean((y_train[va] - pred_va)^2))
    }
    if (!ok) next
    m <- mean(rmses)
    if (m < best_rmse) {
      best_rmse <- m
      best_k <- k_try
    }
  }
  best_k
}

fit_predict_enet_reg <- function(X_train, y_train, X_test, seed) {
  X_mat <- as.matrix(X_train)
  set.seed(seed)
  cv_fit <- cv.glmnet(X_mat, y_train, family = "gaussian", alpha = 0.5)
  as.numeric(predict(cv_fit, newx = as.matrix(X_test), s = "lambda.min"))
}

fit_predict_ols <- function(X_train, y_train, X_test) {
  df <- data.frame(quality = y_train, X_train, check.names = FALSE)
  fit <- lm(quality ~ ., data = df)
  as.numeric(predict(fit, newdata = data.frame(X_test, check.names = FALSE)))
}

fit_predict_multinom <- function(X_train, y_train, X_test) {
  df <- data.frame(quality = factor(y_train), X_train, check.names = FALSE)
  fit <- suppressWarnings(
    multinom(quality ~ ., data = df, trace = FALSE, MaxNWts = 5000)
  )
  preds <- predict(fit, newdata = data.frame(X_test, check.names = FALSE))
  as.numeric(as.integer(as.character(preds)))
}

fit_predict_rpart_reg <- function(X_train, y_train, X_test) {
  df <- data.frame(quality = y_train, X_train, check.names = FALSE)
  fit <- rpart(quality ~ ., data = df, method = "anova")
  as.numeric(predict(fit, newdata = data.frame(X_test, check.names = FALSE)))
}

fit_predict_knn_reg <- function(X_train, y_train, X_test, seed) {
  k <- tune_knn_reg(X_train, y_train, seed)
  as.numeric(knn_reg_one(X_train, y_train, X_test, k))
}

BASE_MODELS <- list(
  list(name = "Elastic Net (reg)", fit = fit_predict_enet_reg),
  list(name = "Linear regression (OLS)", fit = fit_predict_ols),
  list(name = "Multinomial logistic", fit = fit_predict_multinom),
  list(name = "Regression tree (rpart)", fit = fit_predict_rpart_reg),
  list(name = "KNN regression", fit = fit_predict_knn_reg)
)

ensemble_from_preds <- function(pred_mat, top3_rmse_order) {
  # pred_mat: n x 3, columns in order of increasing RMSE (best .. third)
  # top3_rmse_order: 1:3 mapping already reflected in column order
  n <- nrow(pred_mat)
  out <- integer(n)
  best_two <- pred_mat[, 1:2, drop = FALSE]
  for (i in seq_len(n)) {
    p <- pred_mat[i, ]
    r <- as.integer(round(pmin(pmax(p, 3), 9)))
    tab <- table(r)
    mx <- max(tab)
    if (any(tab >= 2L)) {
      out[i] <- as.integer(names(tab)[which.max(tab)])
    } else {
      m <- mean(best_two[i, ])
      out[i] <- as.integer(round(m))
      out[i] <- as.integer(pmin(pmax(out[i], 3L), 9L))
    }
  }
  out
}

raw <- load_train_data()
folds <- create_cv_folds(raw$quality, k = K_FOLDS, seed = CV_FOLD_SEED)

n_models <- length(BASE_MODELS)
rmse_mat <- matrix(NA_real_, nrow = length(folds), ncol = n_models)
colnames(rmse_mat) <- vapply(BASE_MODELS, function(m) m$name, character(1))

fold_data <- vector("list", length(folds))
for (i in seq_along(folds)) {
  test_idx <- folds[[i]]
  train_fold <- raw[-test_idx, ]
  test_fold <- raw[test_idx, ]

  proc_train <- preprocess(train_fold)
  proc_test <- apply_preprocess(test_fold, proc_train$scale_params)

  fd <- list(
    X_train = proc_train$X, y_train = proc_train$y,
    X_test = proc_test$X, y_test = proc_test$y
  )
  fold_data[[i]] <- fd

  seed_i <- CV_FOLD_SEED + 1000L * i

  for (j in seq_len(n_models)) {
    m <- BASE_MODELS[[j]]
    pred <- if (identical(m$fit, fit_predict_enet_reg) ||
      identical(m$fit, fit_predict_knn_reg)) {
      m$fit(fd$X_train, fd$y_train, fd$X_test, seed_i)
    } else {
      m$fit(fd$X_train, fd$y_train, fd$X_test)
    }
    rmse_mat[i, j] <- compute_rmse(fd$y_test, pred)
  }
}

mean_rmse <- colMeans(rmse_mat)
rank_idx <- order(mean_rmse)
top3_idx <- rank_idx[1:3]

cat("Mean RMSE per base model (10-fold):\n")
for (j in seq_len(n_models)) {
  cat(sprintf("  %-28s %.4f\n", colnames(rmse_mat)[j], mean_rmse[j]))
}
cat("\nTop 3 (lowest mean RMSE):\n")
for (k in 1:3) {
  j <- top3_idx[k]
  cat(sprintf("  %d. %-28s mean RMSE = %.4f\n", k, colnames(rmse_mat)[j], mean_rmse[j]))
}
cat("\n")

ensemble_fold_acc <- numeric(length(folds))
for (i in seq_along(folds)) {
  fd <- fold_data[[i]]
  seed_i <- CV_FOLD_SEED + 1000L * i
  pred_mat <- matrix(NA_real_, nrow = nrow(fd$X_test), ncol = 3)
  for (k in 1:3) {
    j <- top3_idx[k]
    m <- BASE_MODELS[[j]]
    pred_mat[, k] <- if (identical(m$fit, fit_predict_enet_reg) ||
      identical(m$fit, fit_predict_knn_reg)) {
      m$fit(fd$X_train, fd$y_train, fd$X_test, seed_i)
    } else {
      m$fit(fd$X_train, fd$y_train, fd$X_test)
    }
  }
  ens <- ensemble_from_preds(pred_mat, 1:3)
  ensemble_fold_acc[i] <- compute_accuracy(fd$y_test, ens)
}

mean_ens_acc <- mean(ensemble_fold_acc)
sd_ens_acc <- sd(ensemble_fold_acc)

cat(sprintf(
  "Ensemble 10-fold CV accuracy: mean = %.4f  SD = %.4f\n",
  mean_ens_acc, sd_ens_acc
))

summary_df <- data.frame(
  model = "Top-3 RMSE ensemble (vote / mean-2-best)",
  k_folds = K_FOLDS,
  mean_cv_accuracy = mean_ens_acc,
  sd_cv_accuracy = sd_ens_acc,
  base_top3 = paste(colnames(rmse_mat)[top3_idx], collapse = "; "),
  stringsAsFactors = FALSE
)
base_rmse_df <- data.frame(
  model = colnames(rmse_mat),
  mean_cv_rmse = as.numeric(mean_rmse),
  stringsAsFactors = FALSE
)

write.csv(summary_df,
          file.path(output_dir, "ensemble_top3_cv_summary.csv"),
          row.names = FALSE)
write.csv(base_rmse_df,
          file.path(output_dir, "ensemble_base_mean_rmse.csv"),
          row.names = FALSE)
write.csv(
  data.frame(fold = seq_along(ensemble_fold_acc), cv_accuracy = ensemble_fold_acc),
  file.path(output_dir, "ensemble_fold_accuracy.csv"),
  row.names = FALSE
)

cat("\nSaved:",
    file.path(output_dir, "ensemble_top3_cv_summary.csv"),
    "\n")
