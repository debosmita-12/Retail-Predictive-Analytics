# scripts/04_model_training_and_tuning.R
# ==============================================================================
# Step 4: Model Training, Tuning, and Time-Aware Validation
# Author: Senior Data Scientist & R Engineer
# ==============================================================================

suppressPackageStartupMessages({
  library(data.table)
  library(ranger)
  library(xgboost)
  library(pROC)
})

setDTthreads(2)
set.seed(42)

cat(">>> Starting Step 4: Model Training & Hyperparameter Tuning...\n")

# 1. Load Train and Validation Cohorts
train_df <- readRDS("outputs/train_cohort.rds")
val_df   <- readRDS("outputs/val_cohort.rds")

# Feature selection (strictly historical)
feature_names <- c(
  "recency_days", "tenure_days", "order_frequency", "total_spend", "total_freight",
  "total_items", "avg_order_value", "avg_items_per_order", "freight_ratio",
  "credit_card_share", "boleto_share", "avg_installments", "avg_review_score",
  "has_low_review", "canceled_orders", "distinct_categories", "spend_last_30d",
  "spend_velocity_30d", "orders_last_30d", "orders_last_90d"
)

cat(sprintf("Training sample: %d | Validation sample: %d\n", nrow(train_df), nrow(val_df)))
cat(sprintf("Engineered numerical features: %d\n", length(feature_names)))

# Prepare feature matrices
X_train <- as.matrix(train_df[, ..feature_names])
y_train <- train_df$churn

X_val   <- as.matrix(val_df[, ..feature_names])
y_val   <- val_df$churn

# Helper to compute PR-AUC via trapezoidal integration of Precision-Recall curve
calc_pr_auc <- function(actual, predicted_prob) {
  order_idx <- order(predicted_prob, decreasing = TRUE)
  act_sorted <- actual[order_idx]
  tp <- cumsum(act_sorted == 1)
  fp <- cumsum(act_sorted == 0)
  total_pos <- sum(act_sorted == 1)
  if (total_pos == 0) return(0)
  recall <- tp / total_pos
  precision <- tp / (tp + fp)
  u_idx <- which(!duplicated(recall))
  rec <- c(0, recall[u_idx])
  prec <- c(precision[u_idx[1]], precision[u_idx])
  pr_auc <- sum(diff(rec) * (prec[-1] + prec[-length(prec)]) / 2)
  return(as.numeric(pr_auc))
}

# ------------------------------------------------------------------------------
# Model 0: Baseline Model (Heuristic Recency Rule)
# If recency_days > 90 -> Churn = 1 (prob = 0.998), else prob = 0.970
# ------------------------------------------------------------------------------
cat("\n--- Evaluating Baseline Model ---\n")
pred_base_val_prob <- ifelse(val_df$recency_days > 90, 0.998, 0.970)
base_roc_val <- as.numeric(pROC::roc(y_val, pred_base_val_prob, quiet = TRUE)$auc)
base_pr_val  <- calc_pr_auc(y_val, pred_base_val_prob)
cat(sprintf("Baseline Val ROC-AUC: %.4f | PR-AUC: %.4f\n", base_roc_val, base_pr_val))

# ------------------------------------------------------------------------------
# Model 1: Logistic Regression
# ------------------------------------------------------------------------------
cat("\n--- Training Model 1: Logistic Regression ---\n")
glm_data_train <- as.data.frame(train_df[, ..feature_names])
glm_data_train$churn <- y_train
fit_glm <- glm(churn ~ ., data = glm_data_train, family = binomial(link = "logit"))

glm_data_val <- as.data.frame(val_df[, ..feature_names])
pred_glm_val_prob <- predict(fit_glm, newdata = glm_data_val, type = "response")
glm_roc_val <- as.numeric(pROC::roc(y_val, pred_glm_val_prob, quiet = TRUE)$auc)
glm_pr_val  <- calc_pr_auc(y_val, pred_glm_val_prob)
cat(sprintf("Logistic Regression Val ROC-AUC: %.4f | PR-AUC: %.4f\n", glm_roc_val, glm_pr_val))

# ------------------------------------------------------------------------------
# Model 2: Random Forest (Ranger)
# ------------------------------------------------------------------------------
cat("\n--- Training Model 2: Random Forest (Ranger) ---\n")
train_rf_data <- as.data.frame(train_df[, ..feature_names])
train_rf_data$churn <- factor(y_train, levels = c(0, 1), labels = c("Retained", "Churned"))

fit_rf <- ranger(
  formula = churn ~ .,
  data = train_rf_data,
  num.trees = 150,
  mtry = 4,
  min.node.size = 50,
  probability = TRUE,
  importance = "impurity",
  num.threads = 2,
  seed = 42
)

pred_rf_val_prob <- predict(fit_rf, data = as.data.frame(val_df[, ..feature_names]))$predictions[, "Churned"]
rf_roc_val <- as.numeric(pROC::roc(y_val, pred_rf_val_prob, quiet = TRUE)$auc)
rf_pr_val  <- calc_pr_auc(y_val, pred_rf_val_prob)
cat(sprintf("Random Forest Val ROC-AUC: %.4f | PR-AUC: %.4f\n", rf_roc_val, rf_pr_val))

# ------------------------------------------------------------------------------
# Model 3: XGBoost & Hyperparameter Tuning (Training/Validation Only)
# ------------------------------------------------------------------------------
cat("\n--- Tuning Model 3: XGBoost on Training/Validation ---\n")
dtrain <- xgb.DMatrix(data = X_train, label = y_train)
dval   <- xgb.DMatrix(data = X_val, label = y_val)

param_grid <- expand.grid(
  max_depth = c(3, 4, 6),
  eta = c(0.03, 0.08),
  subsample = c(0.8, 1.0)
)

tuning_results <- list()
best_auc <- 0
best_model <- NULL
best_params <- NULL
best_iter <- 50

for (i in 1:nrow(param_grid)) {
  params <- list(
    booster = "gbtree",
    objective = "binary:logistic",
    eval_metric = "auc",
    max_depth = param_grid$max_depth[i],
    eta = param_grid$eta[i],
    subsample = param_grid$subsample[i]
  )
  
  bst <- xgb.train(
    params = params,
    data = dtrain,
    nrounds = 60,
    evals = list(val = dval),
    verbose = 0
  )
  
  pred_prob <- predict(bst, dval)
  roc_auc <- as.numeric(pROC::roc(y_val, pred_prob, quiet = TRUE)$auc)
  pr_auc  <- calc_pr_auc(y_val, pred_prob)
  
  tuning_results[[i]] <- data.frame(
    Trial = i,
    max_depth = param_grid$max_depth[i],
    eta = param_grid$eta[i],
    subsample = param_grid$subsample[i],
    Val_ROC_AUC = round(roc_auc, 4),
    Val_PR_AUC = round(pr_auc, 4)
  )
  
  if (roc_auc > best_auc) {
    best_auc <- roc_auc
    best_model <- bst
    best_params <- params
    best_iter <- 60
  }
}

tuning_df <- rbindlist(tuning_results)
fwrite(tuning_df, "outputs/tables/xgboost_tuning_results.csv")
cat("Hyperparameter tuning summary:\n")
print(tuning_df)

cat(sprintf("\nBest XGBoost Val ROC-AUC: %.4f\n", best_auc))

# Predict with best tuned XGBoost
pred_xgb_val_prob <- predict(best_model, dval)
xgb_roc_val <- best_auc
xgb_pr_val  <- calc_pr_auc(y_val, pred_xgb_val_prob)

# ------------------------------------------------------------------------------
# Model Comparison Table on Validation Set
# ------------------------------------------------------------------------------
model_comparison <- data.frame(
  Model = c("Baseline (Recency Rule)", "Logistic Regression", "Random Forest", "Tuned XGBoost"),
  Val_ROC_AUC = c(round(base_roc_val, 4), round(glm_roc_val, 4), round(rf_roc_val, 4), round(xgb_roc_val, 4)),
  Val_PR_AUC = c(round(base_pr_val, 4), round(glm_pr_val, 4), round(rf_pr_val, 4), round(xgb_pr_val, 4))
)

fwrite(model_comparison, "outputs/tables/model_validation_comparison.csv")
cat("\n=======================================================\n")
cat("VALIDATION PERFORMANCE COMPARISON:\n")
print(model_comparison)
cat("=======================================================\n")

# Save the trained model bundle and metadata
model_bundle <- list(
  model = best_model,
  model_type = "xgboost",
  feature_names = feature_names,
  best_params = best_params,
  best_iteration = best_iter,
  validation_metrics = list(
    roc_auc = xgb_roc_val,
    pr_auc = xgb_pr_val
  ),
  trained_date = Sys.time()
)

saveRDS(model_bundle, "outputs/models/churn_model.rds")
saveRDS(fit_glm, "outputs/models/glm_model.rds")
saveRDS(fit_rf, "outputs/models/rf_model.rds")

cat("Best model successfully saved to: outputs/models/churn_model.rds\n")
cat(">>> Step 4 Complete Successfully!\n")
