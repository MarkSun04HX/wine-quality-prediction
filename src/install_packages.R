cat("R version:", R.version.string, "\n")
if (as.numeric(R.version$major) < 4) {
  warning("R >= 4.0 is recommended. You are running ", R.version.string)
}

options(repos = c(CRAN = "https://cloud.r-project.org"))

cran_packages <- c(
  "ggplot2", "reshape2",
  "nnet", "glmnet", "class", "randomForest", "xgboost",
  "e1071", "MASS", "caret", "remotes", "rpart"
)

for (pkg in cran_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    cat("Installing", pkg, "...\n")
    install.packages(pkg, dependencies = TRUE)
  } else {
    cat(pkg, "already installed.\n")
  }
}

if (!requireNamespace("catboost", quietly = TRUE)) {
  cat("Installing catboost from GitHub (this may take several minutes)...\n")
  tryCatch({
    remotes::install_github("catboost/catboost", subdir = "catboost/R-package")
  }, error = function(e) {
    warning(
      "Failed to install catboost: ", e$message, "\n",
      "CatBoost models will be skipped during training.\n",
      "See https://catboost.ai/en/docs/installation/r-installation-binary-installation"
    )
  })
} else {
  cat("catboost already installed.\n")
}

cat("\n--- Package verification ---\n")
all_pkgs <- c(cran_packages, "catboost")
all_ok <- TRUE
for (pkg in all_pkgs) {
  ok <- requireNamespace(pkg, quietly = TRUE)
  cat(sprintf("  %-15s %s\n", pkg, if (ok) "OK" else "MISSING"))
  if (!ok && pkg != "catboost") all_ok <- FALSE
}

if (!all_ok) {
  stop("Some required CRAN packages are missing. Check output above.")
}
cat("\nAll required packages are available.\n")
