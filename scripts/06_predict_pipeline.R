# scripts/06_predict_pipeline.R
# ==============================================================================
# Step 6: Production-Grade Inference Pipeline for Retail Churn Scoring (Vectorized)
# Author: Senior Data Scientist & R Engineer
# ==============================================================================

suppressPackageStartupMessages({
  library(data.table)
  library(xgboost)
})

setDTthreads(2)

#' Predict Customer Churn and Retention Risk
#'
#' @param raw_orders_dt data.table or data.frame containing historical orders with columns:
#'   customer_unique_id, purchase_date, total_item_price, total_freight, total_items,
#'   max_installments, dominant_payment, primary_category, review_score, order_status
#' @param snapshot_date Date or string (YYYY-MM-DD). Cutoff date for scoring.
#' @param model_path File path to saved model bundle RDS.
#' @return data.table with customer_unique_id, churn_probability, retention_probability, risk_tier, recommended_action
predict_customer_churn <- function(raw_orders_dt, 
                                   snapshot_date = as.Date("2018-07-01"), 
                                   model_path = "outputs/models/churn_model.rds") {
  
  if (!file.exists(model_path)) {
    stop("Trained model file not found at: ", model_path)
  }
  
  model_bundle <- readRDS(model_path)
  bst_model <- model_bundle$model
  req_features <- model_bundle$feature_names
  
  dt <- as.data.table(copy(raw_orders_dt))
  t_date <- as.Date(snapshot_date)
  t_lookback <- t_date - 365
  d30_cutoff <- t_date - 30
  d90_cutoff <- t_date - 90
  
  if (!inherits(dt$purchase_date, "Date")) {
    dt[, purchase_date := as.Date(purchase_date)]
  }
  
  # Filter strictly historical records prior to snapshot date
  hist_dt <- dt[purchase_date >= t_lookback & purchase_date < t_date]
  if (nrow(hist_dt) == 0) {
    stop("No orders found within lookback window prior to snapshot date.")
  }
  
  # Vectorized precomputations for blazing speed
  hist_dt[, is_30d := as.integer(purchase_date >= d30_cutoff)]
  hist_dt[, is_90d := as.integer(purchase_date >= d90_cutoff)]
  hist_dt[, spend_30d := ifelse(is_30d == 1L, total_item_price, 0)]
  hist_dt[, is_cc := as.integer(dominant_payment == "credit_card")]
  hist_dt[, is_bol := as.integer(dominant_payment == "boleto")]
  hist_dt[, is_canc := as.integer(order_status == "canceled")]
  hist_dt[, is_low_rev := as.integer(!is.na(review_score) & review_score <= 2)]
  
  # Distinct categories precomputation
  cat_counts <- unique(hist_dt[, .(customer_unique_id, primary_category)])[, .(distinct_categories = .N), by = customer_unique_id]
  
  # Customer level aggregation
  cust_dt <- hist_dt[, .(
    recency_days        = as.numeric(t_date - max(purchase_date)),
    tenure_days         = as.numeric(t_date - min(purchase_date)),
    order_frequency     = .N,
    total_spend         = sum(total_item_price, na.rm = TRUE),
    total_freight       = sum(total_freight, na.rm = TRUE),
    total_items         = sum(total_items, na.rm = TRUE),
    avg_installments    = mean(max_installments, na.rm = TRUE),
    avg_review_score    = mean(review_score, na.rm = TRUE),
    has_low_review      = max(is_low_rev, na.rm = TRUE),
    canceled_orders     = sum(is_canc, na.rm = TRUE),
    credit_card_orders  = sum(is_cc, na.rm = TRUE),
    boleto_orders       = sum(is_bol, na.rm = TRUE),
    spend_last_30d      = sum(spend_30d, na.rm = TRUE),
    orders_last_30d     = sum(is_30d, na.rm = TRUE),
    orders_last_90d     = sum(is_90d, na.rm = TRUE)
  ), by = customer_unique_id]
  
  cust_dt <- merge(cust_dt, cat_counts, by = "customer_unique_id", all.x = TRUE)
  cust_dt[is.na(distinct_categories), distinct_categories := 1L]
  
  cust_dt[, avg_order_value := total_spend / pmax(1, order_frequency)]
  cust_dt[, avg_items_per_order := total_items / pmax(1, order_frequency)]
  cust_dt[, freight_ratio := total_freight / pmax(1, (total_spend + total_freight))]
  cust_dt[, credit_card_share := credit_card_orders / pmax(1, order_frequency)]
  cust_dt[, boleto_share := boleto_orders / pmax(1, order_frequency)]
  cust_dt[, spend_velocity_30d := spend_last_30d / pmax(1, total_spend)]
  
  # Impute any missing values
  cust_dt[is.na(avg_review_score), avg_review_score := 4.0]
  cust_dt[is.na(has_low_review), has_low_review := 0L]
  cust_dt[is.na(freight_ratio), freight_ratio := 0.15]
  
  # Format feature matrix in exact training order
  X_pred <- as.matrix(cust_dt[, ..req_features])
  dpred <- xgb.DMatrix(data = X_pred)
  
  # Predict probabilities
  churn_prob <- predict(bst_model, dpred)
  ret_prob   <- 1 - churn_prob
  
  # Risk stratification
  risk_tier <- ifelse(ret_prob >= 0.008, "High Retention Potential (Top Decile)",
               ifelse(ret_prob >= 0.005, "Medium Potential", "Standard Churn Risk"))
  
  rec_action <- ifelse(ret_prob >= 0.008, "Immediate VIP Re-engagement Offer & Priority Support",
                ifelse(ret_prob >= 0.005, "Targeted Catalog Email & Win-Back Discount",
                       "Standard Automated Lifecycle Cadence"))
  
  results <- data.table(
    customer_unique_id    = cust_dt$customer_unique_id,
    recency_days          = cust_dt$recency_days,
    total_spend           = round(cust_dt$total_spend, 2),
    churn_probability     = round(churn_prob, 5),
    retention_probability = round(ret_prob, 5),
    risk_tier             = risk_tier,
    recommended_action    = rec_action
  )
  
  setorder(results, -retention_probability)
  return(results)
}

# ------------------------------------------------------------------------------
# Self-Test Verification Block
# ------------------------------------------------------------------------------
cat("Testing inference pipeline with a sample batch of historical orders...\n")
orders_master <- readRDS("outputs/cleaned_orders_master.rds")
sample_orders <- orders_master[purchase_date >= as.Date("2018-01-01") & purchase_date < as.Date("2018-07-01")]

scored_sample <- predict_customer_churn(sample_orders, snapshot_date = as.Date("2018-07-01"))
cat(sprintf("Pipeline successfully scored %s customers!\n", format(nrow(scored_sample), big.mark = ",")))
cat("\nTop 5 Scored Customers (Highest Retention Likelihood):\n")
print(head(scored_sample, 5))

# Save scored sample to tables
fwrite(head(scored_sample, 100), "outputs/tables/sample_scored_predictions.csv")
cat("\nSample scored predictions saved to: outputs/tables/sample_scored_predictions.csv\n")
cat(">>> Step 6 Complete Successfully!\n")
