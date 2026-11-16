# scripts/03_target_and_feature_engineering.R
# ==============================================================================
# Step 3: Target Definition & Customer Feature Engineering
# Author: Senior Data Scientist & R Engineer
# ==============================================================================

suppressPackageStartupMessages({
  library(data.table)
  library(lubridate)
})

setDTthreads(2)

cat(">>> Starting Step 3: Target Definition & Feature Engineering...\n")

# Load cleaned master orders table
orders <- readRDS("outputs/cleaned_orders_master.rds")

# Function to engineer customer-level features at a given snapshot date T
# Prevents target leakage by filtering strictly: purchase_date < snapshot_date
engineer_snapshot_cohort <- function(orders_dt, snapshot_date, outcome_days = 60, lookback_days = 365) {
  t_date <- as.Date(snapshot_date)
  t_outcome_end <- t_date + outcome_days
  t_lookback_start <- t_date - lookback_days
  
  cat(sprintf("\n--- Processing Cohort at Snapshot: %s ---\n", as.character(t_date)))
  cat(sprintf("  Lookback window: [%s to %s)\n", as.character(t_lookback_start), as.character(t_date)))
  cat(sprintf("  Outcome window:  [%s to %s]\n", as.character(t_date), as.character(t_outcome_end)))
  
  # 1. Historical orders strictly prior to snapshot date T
  hist_orders <- orders_dt[purchase_date >= t_lookback_start & purchase_date < t_date]
  cat(sprintf("  Historical orders in lookback: %d\n", nrow(hist_orders)))
  
  # 2. Future orders in the 60-day outcome window [T, T + 60]
  fut_orders <- orders_dt[purchase_date >= t_date & purchase_date <= t_outcome_end]
  fut_retained_ids <- unique(fut_orders$customer_unique_id)
  cat(sprintf("  Total future orders in outcome window: %d (Unique buyers: %d)\n", nrow(fut_orders), length(fut_retained_ids)))
  
  # 3. Customer-level aggregation
  # Rolling window cutoffs
  d30_cutoff <- t_date - 30
  d90_cutoff <- t_date - 90
  
  # Basic customer RFM & aggregates
  cust_features <- hist_orders[, .(
    recency_days        = as.numeric(t_date - max(purchase_date)),
    tenure_days         = as.numeric(t_date - min(purchase_date)),
    order_frequency     = .N,
    total_spend         = sum(total_item_price, na.rm = TRUE),
    total_freight       = sum(total_freight, na.rm = TRUE),
    total_items         = sum(total_items, na.rm = TRUE),
    avg_installments    = mean(max_installments, na.rm = TRUE),
    avg_review_score    = mean(review_score, na.rm = TRUE),
    has_low_review      = as.integer(any(review_score <= 2)),
    canceled_orders     = sum(order_status == "canceled"),
    credit_card_orders  = sum(dominant_payment == "credit_card"),
    boleto_orders       = sum(dominant_payment == "boleto"),
    distinct_categories = uniqueN(primary_category),
    
    # Rolling features
    spend_last_30d      = sum(total_item_price[purchase_date >= d30_cutoff], na.rm = TRUE),
    orders_last_30d     = sum(purchase_date >= d30_cutoff),
    orders_last_90d     = sum(purchase_date >= d90_cutoff),
    
    # Geographic location
    customer_state      = head(customer_state, 1)
  ), by = customer_unique_id]
  
  # Derived ratio features
  cust_features[, avg_order_value := total_spend / pmax(1, order_frequency)]
  cust_features[, avg_items_per_order := total_items / pmax(1, order_frequency)]
  cust_features[, freight_ratio := total_freight / pmax(1, (total_spend + total_freight))]
  cust_features[, credit_card_share := credit_card_orders / pmax(1, order_frequency)]
  cust_features[, boleto_share := boleto_orders / pmax(1, order_frequency)]
  cust_features[, spend_velocity_30d := spend_last_30d / pmax(1, total_spend)]
  
  # Standardize state: Top 3 states + Other
  cust_features[, state_grp := ifelse(customer_state %in% c("SP", "RJ", "MG"), customer_state, "Other")]
  cust_features[, state_grp := factor(state_grp, levels = c("SP", "RJ", "MG", "Other"))]
  
  # 4. Target definition: Next-60-day Churn
  # Churn = 1 if customer did NOT place any order in [T, T + 60], 0 if they placed at least 1 order
  cust_features[, churn := ifelse(customer_unique_id %in% fut_retained_ids, 0L, 1L)]
  cust_features[, retained := ifelse(churn == 1L, 0L, 1L)]
  
  # Add snapshot metadata
  cust_features[, snapshot_date := t_date]
  
  churn_count <- sum(cust_features$churn == 1)
  ret_count   <- sum(cust_features$churn == 0)
  total_cust  <- nrow(cust_features)
  cat(sprintf("  Total Active Customers: %d\n", total_cust))
  cat(sprintf("  - Retained (Repeated in 60d): %d (%.3f%%)\n", ret_count, 100 * ret_count / total_cust))
  cat(sprintf("  - Churned  (No order in 60d): %d (%.3f%%)\n", churn_count, 100 * churn_count / total_cust))
  
  return(cust_features)
}

# Generate 3 strictly non-overlapping cohorts
# Train: Snapshot = 2018-01-01 (Outcome: 2018-01-01 to 2018-03-02)
# Val:   Snapshot = 2018-03-15 (Outcome: 2018-03-15 to 2018-05-14)
# Test:  Snapshot = 2018-06-01 (Outcome: 2018-06-01 to 2018-07-31)

train_df <- engineer_snapshot_cohort(orders, "2018-01-01", outcome_days = 60, lookback_days = 365)
val_df   <- engineer_snapshot_cohort(orders, "2018-03-15", outcome_days = 60, lookback_days = 365)
test_df  <- engineer_snapshot_cohort(orders, "2018-06-01", outcome_days = 60, lookback_days = 365)

# Save datasets
saveRDS(train_df, "outputs/train_cohort.rds")
saveRDS(val_df,   "outputs/val_cohort.rds")
saveRDS(test_df,  "outputs/test_cohort.rds")

# Feature dictionary summary
feat_cols <- c(
  "recency_days", "tenure_days", "order_frequency", "total_spend", "total_freight",
  "total_items", "avg_order_value", "avg_items_per_order", "freight_ratio",
  "credit_card_share", "boleto_share", "avg_installments", "avg_review_score",
  "has_low_review", "canceled_orders", "distinct_categories", "spend_last_30d",
  "spend_velocity_30d", "orders_last_30d", "orders_last_90d", "state_grp"
)

feat_dict <- data.frame(
  Feature = feat_cols,
  Type = sapply(train_df[, ..feat_cols], function(x) class(x)[1]),
  Description = c(
    "Days since most recent order prior to snapshot",
    "Days since very first order prior to snapshot",
    "Total historical order count in lookback period",
    "Total cumulative merchandise spend (BRL)",
    "Total cumulative freight charges paid (BRL)",
    "Total physical items purchased across all orders",
    "Average spend per order (total_spend / order_frequency)",
    "Average item count per order (total_items / order_frequency)",
    "Ratio of freight costs to gross spend (freight / (spend + freight))",
    "Fraction of orders paid via credit card",
    "Fraction of orders paid via boleto bancario",
    "Average installment count across transactions",
    "Mean customer review rating (1 to 5 stars)",
    "Binary flag: 1 if customer ever gave a 1 or 2-star review",
    "Count of canceled orders historically",
    "Number of distinct product categories purchased",
    "Merchandise spend in the 30 days preceding snapshot",
    "Ratio of recent 30-day spend to total spend (velocity)",
    "Order count in the 30 days preceding snapshot",
    "Order count in the 90 days preceding snapshot",
    "Customer geographic region (SP, RJ, MG, Other)"
  )
)

fwrite(feat_dict, "outputs/tables/feature_dictionary.csv")
cat("\nFeature dictionary saved to: outputs/tables/feature_dictionary.csv\n")
cat(sprintf("Total engineered features: %d (Requirement: >= 10)\n", length(feat_cols)))

# Cohort comparison table
cohort_summary <- data.frame(
  Split = c("Train (Snapshot 1)", "Validation (Snapshot 2)", "Test (Snapshot 3)"),
  Snapshot_Date = c("2018-01-01", "2018-03-15", "2018-06-01"),
  Outcome_Window = c("2018-01-01 to 2018-03-02", "2018-03-15 to 2018-05-14", "2018-06-01 to 2018-07-31"),
  Active_Customers = c(nrow(train_df), nrow(val_df), nrow(test_df)),
  Retained_Count = c(sum(train_df$churn == 0), sum(val_df$churn == 0), sum(test_df$churn == 0)),
  Churn_Count = c(sum(train_df$churn == 1), sum(val_df$churn == 1), sum(test_df$churn == 1)),
  Churn_Rate_Pct = c(round(100 * mean(train_df$churn), 3), round(100 * mean(val_df$churn), 3), round(100 * mean(test_df$churn), 3))
)
fwrite(cohort_summary, "outputs/tables/cohort_split_summary.csv")
print(cohort_summary)

cat(">>> Step 3 Complete Successfully!\n")
