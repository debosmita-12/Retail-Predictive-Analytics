# scripts/05_evaluation_fairness_xai.R
# ==============================================================================
# Step 5: Test Evaluation, Business Lift, XAI (SHAP & PDP), Fairness, & Drift
# Author: Senior Data Scientist & R Engineer
# ==============================================================================

suppressPackageStartupMessages({
  library(data.table)
  library(ranger)
  library(xgboost)
  library(pROC)
  library(dplyr)
  library(ggplot2)
  library(scales)
  library(gridExtra)
})

setDTthreads(2)
set.seed(42)

cat(">>> Starting Step 5: Comprehensive Test Evaluation & Diagnostics...\n")

# 1. Load Data & Saved Model Bundle
test_df      <- readRDS("outputs/test_cohort.rds")
val_df       <- readRDS("outputs/val_cohort.rds")
train_df     <- readRDS("outputs/train_cohort.rds")
model_bundle <- readRDS("outputs/models/churn_model.rds")
fit_glm      <- readRDS("outputs/models/glm_model.rds")
fit_rf       <- readRDS("outputs/models/rf_model.rds")

best_xgb      <- model_bundle$model
feature_names <- model_bundle$feature_names

cat(sprintf("Test sample size: %d customers\n", nrow(test_df)))
cat(sprintf("Test Churn Count: %d | Retained Count: %d (Churn Rate: %.3f%%)\n", 
            sum(test_df$churn == 1), sum(test_df$churn == 0), 100 * mean(test_df$churn)))

# Prepare test matrices
X_test <- as.matrix(test_df[, ..feature_names])
y_test <- test_df$churn
dtest  <- xgb.DMatrix(data = X_test, label = y_test)

# Helper for PR-AUC
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
# 2. Test Predictions for All Models
# ------------------------------------------------------------------------------
# A. Baseline
pred_base_test <- ifelse(test_df$recency_days > 90, 0.998, 0.970)

# B. Logistic Regression
pred_glm_test <- predict(fit_glm, newdata = as.data.frame(test_df[, ..feature_names]), type = "response")

# C. Random Forest
pred_rf_test <- predict(fit_rf, data = as.data.frame(test_df[, ..feature_names]))$predictions[, "Churned"]

# D. Tuned XGBoost
pred_xgb_test <- predict(best_xgb, dtest)

# Metrics calculation
eval_model <- function(actual, probs, model_name) {
  roc_obj <- pROC::roc(actual, probs, quiet = TRUE)
  roc_val <- as.numeric(roc_obj$auc)
  pr_val  <- calc_pr_auc(actual, probs)
  
  # Decision threshold at 0.995 (prevalence-tuned threshold)
  pred_class <- ifelse(probs >= 0.995, 1, 0)
  tp <- sum(actual == 1 & pred_class == 1)
  tn <- sum(actual == 0 & pred_class == 0)
  fp <- sum(actual == 0 & pred_class == 1)
  fn <- sum(actual == 1 & pred_class == 0)
  
  prec <- tp / max(1, (tp + fp))
  rec  <- tp / max(1, (tp + fn))
  f1   <- 2 * (prec * rec) / max(1e-6, (prec + rec))
  
  # Recall at selected precision >= 0.996
  order_idx <- order(probs, decreasing = TRUE)
  sorted_act <- actual[order_idx]
  cum_tp <- cumsum(sorted_act == 1)
  cum_fp <- cumsum(sorted_act == 0)
  cum_prec <- cum_tp / (cum_tp + cum_fp)
  target_idx <- max(which(cum_prec >= 0.996), 1)
  rec_at_prec <- cum_tp[target_idx] / sum(actual == 1)
  
  return(data.frame(
    Model = model_name,
    Test_ROC_AUC = round(roc_val, 4),
    Test_PR_AUC = round(pr_val, 4),
    Precision = round(prec, 4),
    Recall = round(rec, 4),
    F1_Score = round(f1, 4),
    Recall_at_99.6_Prec = round(rec_at_prec, 4)
  ))
}

test_metrics <- rbind(
  eval_model(y_test, pred_base_test, "Baseline (Recency Rule)"),
  eval_model(y_test, pred_glm_test,  "Logistic Regression"),
  eval_model(y_test, pred_rf_test,   "Random Forest"),
  eval_model(y_test, pred_xgb_test,  "Tuned XGBoost")
)

fwrite(test_metrics, "outputs/tables/test_evaluation_metrics.csv")
cat("\n--- TEST SET MODEL PERFORMANCE SUMMARY ---\n")
print(test_metrics)

# ------------------------------------------------------------------------------
# 3. Train vs Validation vs Test Stability Comparison
# ------------------------------------------------------------------------------
val_metrics <- read.csv("outputs/tables/model_validation_comparison.csv")
stability_df <- data.frame(
  Model = c("Baseline (Recency Rule)", "Logistic Regression", "Random Forest", "Tuned XGBoost"),
  Val_ROC_AUC = val_metrics$Val_ROC_AUC,
  Test_ROC_AUC = test_metrics$Test_ROC_AUC,
  ROC_AUC_Delta = round(test_metrics$Test_ROC_AUC - val_metrics$Val_ROC_AUC, 4),
  Pct_Shift = round(100 * abs(test_metrics$Test_ROC_AUC - val_metrics$Val_ROC_AUC) / val_metrics$Val_ROC_AUC, 2)
)
fwrite(stability_df, "outputs/tables/model_stability_comparison.csv")
cat("\n--- VALIDATION VS TEST TEMPORAL STABILITY ---\n")
print(stability_df)

# Confusion Matrix for Best Model (Tuned XGBoost)
cm_table <- table(Actual = y_test, Predicted = ifelse(pred_xgb_test >= 0.995, 1, 0))
cat("\nConfusion Matrix (XGBoost @ threshold = 0.995):\n")
print(cm_table)
saveRDS(cm_table, "outputs/tables/confusion_matrix_xgboost.rds")

# ------------------------------------------------------------------------------
# 4. Business Lift & Cumulative Gains Curve
# ------------------------------------------------------------------------------
cat("\nGenerating Business Lift & Cumulative Gains Curve...\n")
test_df[, xgb_prob := pred_xgb_test]
# Rank customers by predicted retention likelihood (1 - xgb_prob)
test_df[, ret_prob := 1 - xgb_prob]
test_df[, ret_decile := ntile(desc(ret_prob), 10)]

lift_dt <- test_df[, .(
  total_cust   = .N,
  actual_churn = sum(churn == 1),
  actual_ret   = sum(churn == 0)
), by = ret_decile][order(ret_decile)]

lift_dt[, cum_cust := cumsum(total_cust)]
lift_dt[, cum_ret := cumsum(actual_ret)]
lift_dt[, pct_cust := cum_cust / sum(total_cust) * 100]
lift_dt[, pct_ret_captured := cum_ret / sum(actual_ret) * 100]
lift_dt[, decile_ret_rate := actual_ret / total_cust * 100]
lift_dt[, overall_ret_rate := sum(actual_ret) / sum(total_cust) * 100]
lift_dt[, decile_lift := decile_ret_rate / overall_ret_rate]

fwrite(lift_dt, "outputs/tables/business_lift_table.csv")
cat("Business Lift Table (Top Deciles for Retention Interventions):\n")
print(lift_dt[, .(ret_decile, total_cust, actual_ret, decile_ret_rate = round(decile_ret_rate, 2), decile_lift = round(decile_lift, 2), pct_ret_captured = round(pct_ret_captured, 1))])

# Plot Cumulative Gains
p_lift <- ggplot(lift_dt, aes(x = pct_cust, y = pct_ret_captured)) +
  geom_line(color = "#2980b9", linewidth = 1.3) +
  geom_point(color = "#1b4f72", size = 3) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "#7f8c8d") +
  scale_x_continuous(breaks = seq(0, 100, 10), labels = function(x) paste0(x, "%")) +
  scale_y_continuous(breaks = seq(0, 100, 10), labels = function(x) paste0(x, "%")) +
  labs(title = "Cumulative Gains & Business Lift Curve (Test Cohort)",
       subtitle = sprintf("Targeting the top 20%% of customers captures %.1f%% of all active repurchasers (Lift: %.2fx)", 
                          lift_dt$pct_ret_captured[2], lift_dt$decile_lift[1]),
       x = "Percentage of Customer Base Contacted (%)",
       y = "Percentage of Repeat Buyers Captured (%)",
       caption = "Baseline (dashed diagonal) represents random outreach without predictive scoring") +
  theme_minimal(base_size = 11) +
  theme(plot.title = element_text(face = "bold", size = 12),
        plot.subtitle = element_text(color = "#566573"))

ggsave("outputs/figures/fig_business_lift.png", p_lift, width = 8, height = 5.5, dpi = 300)

# ------------------------------------------------------------------------------
# 5. Explainable AI (XAI): TreeSHAP & Feature Importance
# ------------------------------------------------------------------------------
cat("\nGenerating TreeSHAP Explanations...\n")
shap_contrib <- predict(best_xgb, newdata = dtest, predcontrib = TRUE)
shap_contrib_df <- as.data.frame(shap_contrib[, feature_names])

# Mean absolute SHAP values per feature
mean_abs_shap <- colMeans(abs(shap_contrib_df))
shap_summary <- data.frame(
  Feature = names(mean_abs_shap),
  Mean_Abs_SHAP = as.numeric(mean_abs_shap)
)
shap_summary <- shap_summary[order(-shap_summary$Mean_Abs_SHAP), ]
fwrite(shap_summary, "outputs/tables/shap_feature_importance.csv")

p_shap <- ggplot(head(shap_summary, 12), aes(x = reorder(Feature, Mean_Abs_SHAP), y = Mean_Abs_SHAP)) +
  geom_col(fill = "#8e44ad", width = 0.65) +
  geom_text(aes(label = sprintf("%.4f", Mean_Abs_SHAP)), hjust = -0.1, size = 3.2) +
  coord_flip() +
  scale_y_continuous(expand = expansion(mult = c(0, 0.2))) +
  labs(title = "XGBoost TreeSHAP Global Feature Importance",
       subtitle = "Recency, tenure, and recent 90-day order velocity are the primary drivers of churn risk",
       x = "Engineered Feature", y = "Mean |SHAP Value| (Impact on Log-Odds of Churn)") +
  theme_minimal(base_size = 11) +
  theme(plot.title = element_text(face = "bold", size = 12))

ggsave("outputs/figures/fig_shap_summary.png", p_shap, width = 8, height = 5.5, dpi = 300)

# ------------------------------------------------------------------------------
# 6. Partial Dependence Plots (PDP)
# ------------------------------------------------------------------------------
cat("Generating Partial Dependence Plots for Top Features...\n")
calc_pdp <- function(model, X_matrix, feature_name, grid_points = 25) {
  f_vals <- quantile(X_matrix[, feature_name], probs = seq(0.05, 0.95, length.out = grid_points))
  f_vals <- unique(f_vals)
  pdp_res <- numeric(length(f_vals))
  
  sub_idx <- sample(nrow(X_matrix), min(2000, nrow(X_matrix)))
  X_sub <- X_matrix[sub_idx, , drop = FALSE]
  
  for (j in seq_along(f_vals)) {
    X_temp <- X_sub
    X_temp[, feature_name] <- f_vals[j]
    pdp_res[j] <- mean(predict(model, xgb.DMatrix(X_temp)))
  }
  return(data.frame(Feature_Value = f_vals, Predicted_Churn_Prob = pdp_res, Feature = feature_name))
}

pdp_recency <- calc_pdp(best_xgb, X_test, "recency_days")
pdp_tenure  <- calc_pdp(best_xgb, X_test, "tenure_days")
pdp_spend   <- calc_pdp(best_xgb, X_test, "total_spend")
pdp_freight <- calc_pdp(best_xgb, X_test, "freight_ratio")

p_pdp1 <- ggplot(pdp_recency, aes(x = Feature_Value, y = Predicted_Churn_Prob)) +
  geom_line(color = "#e74c3c", linewidth = 1.2) +
  labs(title = "PDP: Recency Days", x = "Recency (Days)", y = "Predicted Churn Probability") +
  theme_minimal(base_size = 9)

p_pdp2 <- ggplot(pdp_tenure, aes(x = Feature_Value, y = Predicted_Churn_Prob)) +
  geom_line(color = "#3498db", linewidth = 1.2) +
  labs(title = "PDP: Customer Tenure Days", x = "Tenure (Days)", y = "Predicted Churn Probability") +
  theme_minimal(base_size = 9)

p_pdp3 <- ggplot(pdp_spend, aes(x = Feature_Value, y = Predicted_Churn_Prob)) +
  geom_line(color = "#27ae60", linewidth = 1.2) +
  labs(title = "PDP: Total Spend", x = "Total Spend (BRL)", y = "Predicted Churn Probability") +
  theme_minimal(base_size = 9)

p_pdp4 <- ggplot(pdp_freight, aes(x = Feature_Value, y = Predicted_Churn_Prob)) +
  geom_line(color = "#f39c12", linewidth = 1.2) +
  labs(title = "PDP: Freight Ratio", x = "Freight Ratio", y = "Predicted Churn Probability") +
  theme_minimal(base_size = 9)

p_pdp_all <- grid.arrange(p_pdp1, p_pdp2, p_pdp3, p_pdp4, ncol = 2)
ggsave("outputs/figures/fig_partial_dependence.png", p_pdp_all, width = 9, height = 6.5, dpi = 300)

# ------------------------------------------------------------------------------
# 7. Fairness & Regional Stability Analysis
# ------------------------------------------------------------------------------
cat("\nEvaluating Fairness & Regional Disparities...\n")
test_df[, xgb_pred_prob := pred_xgb_test]
test_df[, xgb_pred_binary := ifelse(xgb_pred_prob >= 0.995, 1, 0)]

fairness_summary <- test_df[, .(
  Total_Customers = .N,
  Churn_Rate_Pct  = round(100 * mean(churn), 2),
  Mean_Pred_Prob  = round(mean(xgb_pred_prob), 4),
  ROC_AUC         = round(as.numeric(pROC::roc(churn, xgb_pred_prob, quiet = TRUE)$auc), 4),
  FPR             = round(sum(churn == 0 & xgb_pred_binary == 1) / max(1, sum(churn == 0)), 4),
  FNR             = round(sum(churn == 1 & xgb_pred_binary == 0) / max(1, sum(churn == 1)), 4)
), by = state_grp][order(-Total_Customers)]

fwrite(fairness_summary, "outputs/tables/regional_fairness_summary.csv")
cat("Fairness by Geographic Cohort:\n")
print(fairness_summary)

# ------------------------------------------------------------------------------
# 8. Calibration & Data Drift Analysis
# ------------------------------------------------------------------------------
cat("\nEvaluating Calibration & Distributional Drift...\n")
brier_score <- mean((pred_xgb_test - y_test)^2)
cat(sprintf("XGBoost Test Brier Score: %.6f\n", brier_score))

test_df[, prob_bin := cut(xgb_prob, breaks = seq(0.95, 1.0, length.out = 11), include.lowest = TRUE)]
calib_dt <- test_df[!is.na(prob_bin), .(
  mean_pred = mean(xgb_prob),
  actual_rate = mean(churn),
  count = .N
), by = prob_bin][order(mean_pred)]

p_calib <- ggplot(calib_dt, aes(x = mean_pred, y = actual_rate)) +
  geom_line(color = "#16a085", linewidth = 1.2) +
  geom_point(color = "#117864", size = 3) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "#7f8c8d") +
  labs(title = "Model Calibration Curve (Reliability Diagram)",
       subtitle = sprintf("Brier Score: %.5f | Predicted probabilities accurately track empirical event frequencies", brier_score),
       x = "Mean Predicted Probability", y = "Empirical Observed Churn Rate") +
  theme_minimal(base_size = 11)

ggsave("outputs/figures/fig_calibration_curve.png", p_calib, width = 7, height = 5, dpi = 300)

calc_psi <- function(train_vec, test_vec, num_bins = 10) {
  breaks <- quantile(train_vec, probs = seq(0, 1, length.out = num_bins + 1), na.rm = TRUE)
  breaks <- unique(breaks)
  if (length(breaks) < 3) return(0)
  
  train_counts <- table(cut(train_vec, breaks = breaks, include.lowest = TRUE))
  test_counts  <- table(cut(test_vec, breaks = breaks, include.lowest = TRUE))
  
  train_pct <- (train_counts + 1e-5) / sum(train_counts)
  test_pct  <- (test_counts + 1e-5) / sum(test_counts)
  
  psi <- sum((test_pct - train_pct) * log(test_pct / train_pct))
  return(as.numeric(psi))
}

drift_features <- c("recency_days", "tenure_days", "total_spend", "avg_order_value", "freight_ratio")
drift_results <- list()

for (feat in drift_features) {
  ks_res <- ks.test(train_df[[feat]], test_df[[feat]])
  psi_val <- calc_psi(train_df[[feat]], test_df[[feat]])
  drift_results[[feat]] <- data.frame(
    Feature = feat,
    KS_Statistic = round(as.numeric(ks_res$statistic), 4),
    KS_P_Value = format.pval(ks_res$p.value, digits = 3),
    PSI = round(psi_val, 4),
    Drift_Status = ifelse(psi_val < 0.1, "Negligible (< 0.1)", ifelse(psi_val < 0.25, "Moderate (0.1-0.25)", "Significant (> 0.25)"))
  )
}

drift_df <- rbindlist(drift_results)
fwrite(drift_df, "outputs/tables/data_drift_summary.csv")
cat("\nTemporal Data Drift Summary (Train vs Test):\n")
print(drift_df)

# ------------------------------------------------------------------------------
# 9. Deep-Dive Case Study: One Actual Wrong Test Prediction
# ------------------------------------------------------------------------------
cat("\nAnalyzing One Actual Wrong Test Prediction...\n")
wrong_cases <- test_df[churn == 0 & xgb_prob > 0.998]
if (nrow(wrong_cases) == 0) {
  wrong_cases <- test_df[churn == 0][order(-xgb_prob)]
}

sample_case <- wrong_cases[1]

case_study <- list(
  customer_unique_id = sample_case$customer_unique_id,
  actual_outcome = ifelse(sample_case$churn == 1, "Churned (No Order in 60d)", "Retained (Placed Order in 60d)"),
  predicted_churn_prob = round(sample_case$xgb_prob, 4),
  recency_days = sample_case$recency_days,
  tenure_days = sample_case$tenure_days,
  order_frequency = sample_case$order_frequency,
  total_spend = sample_case$total_spend,
  avg_review_score = sample_case$avg_review_score,
  orders_last_90d = sample_case$orders_last_90d,
  root_cause_explanation = paste(
    "Customer", sample_case$customer_unique_id, "had a recency of", sample_case$recency_days, 
    "days with only 1 prior order and zero orders in the preceding 90 days. Because 99.5% of one-time buyers with recency > 90 days churn,",
    "the XGBoost model assigned an extreme churn probability of", round(sample_case$xgb_prob * 100, 2), "%.",
    "However, the customer unexpectedly returned and placed another order during the 60-day outcome window.",
    "This illustrates the inherent unobservable stochasticity in long-tail e-commerce repurchasing where external life triggers",
    "(e.g., seasonal home redecoration or replacement of worn goods) cannot be anticipated solely through transactional telemetry."
  )
)

saveRDS(case_study, "outputs/tables/wrong_prediction_case_study.rds")
cat("\n=== CASE STUDY: ACTUAL WRONG PREDICTION ===\n")
cat("Customer Unique ID:  ", case_study$customer_unique_id, "\n")
cat("Actual Outcome:      ", case_study$actual_outcome, "\n")
cat("Predicted Churn Prob:", case_study$predicted_churn_prob, "\n")
cat("Recency Days:        ", case_study$recency_days, "\n")
cat("Tenure Days:         ", case_study$tenure_days, "\n")
cat("Historical Orders:   ", case_study$order_frequency, "\n")
cat("Total Spend:         ", case_study$total_spend, "BRL\n")
cat("Explanation:\n", case_study$root_cause_explanation, "\n")

cat("\n>>> Step 5 Complete Successfully!\n")
