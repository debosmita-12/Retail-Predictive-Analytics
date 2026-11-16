# scripts/00_initial_data_audit.R
# Comprehensive initial audit of all 9 raw CSV files in Data/Raw/

library(data.table)

raw_dir <- "Data/Raw"
files <- list.files(raw_dir, pattern = "\\.csv$", full.names = TRUE)

cat("=================================================================\n")
cat("            RETAIL PREDICTIVE ANALYTICS - DATA AUDIT              \n")
cat("=================================================================\n\n")

for (f in files) {
  fname <- basename(f)
  cat("-----------------------------------------------------------------\n")
  cat("Auditing File:", fname, "\n")
  cat("File size:", sprintf("%.2f MB", file.size(f) / (1024 * 1024)), "\n")
  
  dt <- fread(f, nrows = Inf, na.strings = c("", "NA", "null"))
  nrows <- nrow(dt)
  ncols <- ncol(dt)
  cat("Rows:", format(nrows, big.mark = ","), " | Columns:", ncols, "\n")
  cat("Columns:\n")
  for (col in names(dt)) {
    null_cnt <- sum(is.na(dt[[col]]))
    cat(sprintf("  - %-32s [%-10s] | NAs: %d (%.2f%%)\n", 
                col, class(dt[[col]])[1], null_cnt, 100 * null_cnt / max(1, nrows)))
  }
  
  # Duplicates check (sample if huge)
  if (nrows > 200000) {
    dup_cnt <- anyDuplicated(dt[1:50000])
    cat("Duplicate rows in first 50k rows:", dup_cnt > 0, "\n")
  } else {
    dup_cnt <- anyDuplicated(dt)
    cat("Duplicate rows detected:", dup_cnt > 0, "\n")
  }
  
  # Specific dataset checks
  if (fname == "olist_orders_dataset.csv") {
    cat(">> Orders Dataset Analysis:\n")
    print(dt[, .N, by = order_status][order(-N)])
    cat("   Purchase date range:", min(dt$order_purchase_timestamp, na.rm=TRUE), "to", max(dt$order_purchase_timestamp, na.rm=TRUE), "\n")
    cat("   Unique order_ids:", uniqueN(dt$order_id), "\n")
    cat("   Unique customer_ids:", uniqueN(dt$customer_id), "\n")
  }
  
  if (fname == "olist_customers_dataset.csv") {
    cat(">> Customers Dataset Analysis:\n")
    cat("   Unique customer_ids (per-order):", uniqueN(dt$customer_id), "\n")
    cat("   Unique customer_unique_ids:", uniqueN(dt$customer_unique_id), "\n")
    repeat_cust <- dt[, .N, by = customer_unique_id][N > 1]
    cat("   Repeat customer_unique_ids (>1 order):", nrow(repeat_cust), sprintf("(%.2f%%)\n", 100 * nrow(repeat_cust) / uniqueN(dt$customer_unique_id)))
    cat("   Top customer states:\n")
    print(head(dt[, .N, by = customer_state][order(-N)], 5))
  }
  
  if (fname == "olist_order_items_dataset.csv") {
    cat(">> Order Items Dataset Analysis:\n")
    cat("   Unique orders in items:", uniqueN(dt$order_id), "\n")
    cat("   Price range:", min(dt$price, na.rm=TRUE), "to", max(dt$price, na.rm=TRUE), " | Mean:", sprintf("%.2f", mean(dt$price, na.rm=TRUE)), "\n")
    cat("   Freight range:", min(dt$freight_value, na.rm=TRUE), "to", max(dt$freight_value, na.rm=TRUE), " | Mean:", sprintf("%.2f", mean(dt$freight_value, na.rm=TRUE)), "\n")
    cat("   FACT: No GST or tax-specific columns exist in Olist schema.\n")
  }
  
  if (fname == "olist_order_payments_dataset.csv") {
    cat(">> Order Payments Dataset Analysis:\n")
    print(dt[, .N, by = payment_type][order(-N)])
    cat("   Payment value sum:", format(sum(dt$payment_value, na.rm=TRUE), big.mark = ","), "\n")
  }
  
  if (fname == "olist_products_dataset.csv") {
    cat(">> Products Dataset Analysis:\n")
    cat("   Unique product_ids:", uniqueN(dt$product_id), "\n")
    cat("   Unique categories:", uniqueN(dt$product_category_name), "\n")
    cat("   Missing category names:", sum(is.na(dt$product_category_name)), "\n")
  }
  
  cat("\n")
}

cat("=================================================================\n")
cat("AUDIT COMPLETE\n")
cat("=================================================================\n")
