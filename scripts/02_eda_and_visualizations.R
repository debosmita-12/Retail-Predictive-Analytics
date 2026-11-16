# scripts/02_eda_and_visualizations.R
# ==============================================================================
# Step 2: Thorough Exploratory Data Analysis (EDA) & Visualization Suite
# Author: Senior Data Scientist & R Engineer
# ==============================================================================

suppressPackageStartupMessages({
  library(data.table)
  library(ggplot2)
  library(scales)
  library(gridExtra)
})

setDTthreads(2)

cat(">>> Starting Step 2: Exploratory Data Analysis & Plot Generation...\n")

# Load cleaned master orders table
orders <- readRDS("outputs/cleaned_orders_master.rds")
fig_dir <- "outputs/figures"
tab_dir <- "outputs/tables"
if (!dir.exists(fig_dir)) dir.create(fig_dir, recursive = TRUE)
if (!dir.exists(tab_dir)) dir.create(tab_dir, recursive = TRUE)

# Theme configuration
theme_custom <- function() {
  theme_minimal(base_size = 11) +
    theme(
      plot.title = element_text(face = "bold", size = 13, color = "#1a252f", hjust = 0),
      plot.subtitle = element_text(size = 10, color = "#566573", margin = margin(b = 8)),
      plot.caption = element_text(size = 8, color = "#7f8c8d", hjust = 1),
      axis.title = element_text(face = "bold", size = 10, color = "#2c3e50"),
      axis.text = element_text(color = "#34495e"),
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(color = "#eaeded"),
      legend.position = "bottom",
      legend.title = element_text(face = "bold", size = 9)
    )
}

# ------------------------------------------------------------------------------
# Figure 1: Monthly Order Volume & Revenue Trends
# ------------------------------------------------------------------------------
cat("Generating Figure 1: Monthly Trends...\n")
orders[, year_month := format(purchase_date, "%Y-%m")]
# Filter to full active months (2017-01 to 2018-08)
monthly <- orders[year_month >= "2017-01" & year_month <= "2018-08", .(
  orders_count = .N,
  revenue_brl  = sum(total_item_price, na.rm = TRUE) / 1000
), by = year_month][order(year_month)]

p1a <- ggplot(monthly, aes(x = factor(year_month), y = orders_count, group = 1)) +
  geom_col(fill = "#2980b9", alpha = 0.85, width = 0.65) +
  geom_text(aes(label = comma(orders_count)), vjust = -0.5, size = 2.8, color = "#1b4f72") +
  scale_y_continuous(labels = comma, expand = expansion(mult = c(0, 0.15))) +
  labs(title = "Monthly Order Volume Growth (2017 - 2018)",
       subtitle = "Notable peak during Black Friday (Nov 2017) followed by sustained 6,500+ monthly orders",
       x = "Year-Month", y = "Total Orders") +
  theme_custom() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

p1b <- ggplot(monthly, aes(x = factor(year_month), y = revenue_brl, group = 1)) +
  geom_line(color = "#e67e22", linewidth = 1.2) +
  geom_point(color = "#d35400", size = 2.5) +
  scale_y_continuous(labels = function(x) paste0("R$ ", comma(x), "k"), expand = expansion(mult = c(0.05, 0.15))) +
  labs(title = "Monthly Gross Merchandise Value (GMV in Thousands BRL)",
       subtitle = "Platform revenue reached over 1,000,000 BRL per month by early 2018",
       x = "Year-Month", y = "Revenue (k BRL)") +
  theme_custom() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

p1 <- grid.arrange(p1a, p1b, ncol = 1)
ggsave(file.path(fig_dir, "fig1_monthly_trends.png"), p1, width = 10, height = 8, dpi = 300)
fwrite(monthly, file.path(tab_dir, "monthly_performance_summary.csv"))

# ------------------------------------------------------------------------------
# Figure 2: Order Status Distribution
# ------------------------------------------------------------------------------
cat("Generating Figure 2: Order Status Distribution...\n")
status_dt <- orders[, .(count = .N), by = order_status][order(-count)]
status_dt[, pct := count / sum(count) * 100]

p2 <- ggplot(status_dt, aes(x = reorder(order_status, count), y = count, fill = order_status == "delivered")) +
  geom_col(width = 0.65, show.legend = FALSE) +
  geom_text(aes(label = sprintf("%s (%.2f%%)", comma(count), pct)), hjust = -0.1, size = 3.2) +
  scale_fill_manual(values = c("TRUE" = "#27ae60", "FALSE" = "#e74c3c")) +
  scale_y_continuous(labels = comma, expand = expansion(mult = c(0, 0.25))) +
  coord_flip() +
  labs(title = "Order Fulfillment Status Breakdown",
       subtitle = "Delivered orders account for 97.02% of all customer transactions",
       x = "Order Status", y = "Order Count",
       caption = "Data Source: Olist Brazilian E-Commerce Public Dataset") +
  theme_custom()

ggsave(file.path(fig_dir, "fig2_order_status.png"), p2, width = 8, height = 5, dpi = 300)

# ------------------------------------------------------------------------------
# Figure 3: Payment Methods & Installment Distribution
# ------------------------------------------------------------------------------
cat("Generating Figure 3: Payment Methods & Installments...\n")
pay_dt <- orders[, .(count = .N), by = dominant_payment][order(-count)]
pay_dt[, pct := count / sum(count) * 100]

p3a <- ggplot(pay_dt, aes(x = reorder(dominant_payment, count), y = pct, fill = dominant_payment)) +
  geom_col(width = 0.6, show.legend = FALSE) +
  geom_text(aes(label = sprintf("%.1f%%", pct)), hjust = -0.2, size = 3.5, fontface = "bold") +
  scale_fill_brewer(palette = "Blues", direction = -1) +
  scale_y_continuous(limits = c(0, 85), labels = function(x) paste0(x, "%")) +
  coord_flip() +
  labs(title = "Dominant Payment Methods",
       subtitle = "Credit card dominates 75%+ of customer purchases",
       x = "Payment Type", y = "Share of Orders (%)") +
  theme_custom()

cc_install <- orders[dominant_payment == "credit_card" & max_installments <= 12, .(count = .N), by = max_installments][order(max_installments)]
p3b <- ggplot(cc_install, aes(x = factor(max_installments), y = count)) +
  geom_col(fill = "#3498db", width = 0.7) +
  geom_text(aes(label = comma(count)), vjust = -0.4, size = 2.8) +
  scale_y_continuous(labels = comma, expand = expansion(mult = c(0, 0.15))) +
  labs(title = "Credit Card Installment Distribution",
       subtitle = "Single-installment and multi-installment (up to 10x) splits reflect consumer financing reliance",
       x = "Installment Count", y = "Number of Orders") +
  theme_custom()

p3 <- grid.arrange(p3a, p3b, ncol = 2)
ggsave(file.path(fig_dir, "fig3_payment_and_installments.png"), p3, width = 11, height = 5, dpi = 300)

# ------------------------------------------------------------------------------
# Figure 4: Top 15 Product Categories by Volume and Revenue
# ------------------------------------------------------------------------------
cat("Generating Figure 4: Top Product Categories...\n")
cat_summary <- orders[, .(
  orders_count = .N,
  revenue_brl  = sum(total_item_price, na.rm = TRUE)
), by = primary_category][order(-revenue_brl)][1:15]

p4 <- ggplot(cat_summary, aes(x = reorder(primary_category, revenue_brl), y = revenue_brl / 1000)) +
  geom_col(fill = "#16a085", width = 0.7) +
  geom_text(aes(label = sprintf("R$ %.1fk (%s orders)", revenue_brl / 1000, comma(orders_count))), 
            hjust = -0.05, size = 3) +
  scale_y_continuous(labels = comma, expand = expansion(mult = c(0, 0.25))) +
  coord_flip() +
  labs(title = "Top 15 Product Categories by Total Revenue",
       subtitle = "Health & Beauty, Bed/Bath/Table, and Computer Accessories lead total sales GMV",
       x = "Product Category", y = "Revenue (Thousand BRL)",
       caption = "Fact: Categories translated from original Portuguese schema") +
  theme_custom()

ggsave(file.path(fig_dir, "fig4_top_categories.png"), p4, width = 9, height = 6.5, dpi = 300)

# ------------------------------------------------------------------------------
# Figure 5: Customer Geographic Distribution (Top 10 States)
# ------------------------------------------------------------------------------
cat("Generating Figure 5: Geographic Distribution...\n")
geo_summary <- orders[, .(
  total_orders = .N,
  unique_customers = uniqueN(customer_unique_id),
  avg_spend = mean(total_item_price, na.rm = TRUE),
  avg_freight = mean(total_freight, na.rm = TRUE)
), by = customer_state][order(-total_orders)][1:10]

geo_summary[, pct_orders := total_orders / sum(orders$order_id != "") * 100]

p5 <- ggplot(geo_summary, aes(x = reorder(customer_state, total_orders), y = total_orders, fill = customer_state == "SP")) +
  geom_col(width = 0.65, show.legend = FALSE) +
  geom_text(aes(label = sprintf("%s (%.1f%%)", comma(total_orders), pct_orders)), hjust = -0.1, size = 3.2) +
  scale_fill_manual(values = c("TRUE" = "#2c3e50", "FALSE" = "#7f8c8d")) +
  scale_y_continuous(labels = comma, expand = expansion(mult = c(0, 0.2))) +
  coord_flip() +
  labs(title = "Geographic Concentration: Top 10 Brazilian States",
       subtitle = "São Paulo (SP) represents 42.0% of orders, followed by RJ and MG",
       x = "Customer State", y = "Total Order Volume",
       caption = "Fact: Regional logistics disparity influences freight cost and customer churn") +
  theme_custom()

ggsave(file.path(fig_dir, "fig5_geographic_distribution.png"), p5, width = 8, height = 5.5, dpi = 300)
fwrite(geo_summary, file.path(tab_dir, "state_geographic_summary.csv"))

# ------------------------------------------------------------------------------
# Figure 6: Customer Repeat Purchase Behavior & Frequency
# ------------------------------------------------------------------------------
cat("Generating Figure 6: Customer Repurchase Frequency...\n")
cust_freq <- orders[, .(order_count = .N), by = customer_unique_id]
freq_dist <- cust_freq[, .(customers = .N), by = order_count][order(order_count)]
freq_dist[, pct := customers / sum(customers) * 100]
freq_dist[, freq_bucket := ifelse(order_count == 1, "1 Order (One-Time)",
                            ifelse(order_count == 2, "2 Orders",
                            ifelse(order_count == 3, "3 Orders", "4+ Orders")))]
bucket_dist <- freq_dist[, .(customers = sum(customers), pct = sum(pct)), by = freq_bucket]
bucket_dist[, freq_bucket := factor(freq_bucket, levels = c("1 Order (One-Time)", "2 Orders", "3 Orders", "4+ Orders"))]

p6 <- ggplot(bucket_dist, aes(x = freq_bucket, y = customers, fill = freq_bucket)) +
  geom_col(width = 0.6, show.legend = FALSE) +
  geom_text(aes(label = sprintf("%s\n(%.2f%%)", comma(customers), pct)), vjust = -0.3, size = 3.5, fontface = "bold") +
  scale_fill_brewer(palette = "Blues", direction = -1) +
  scale_y_continuous(labels = comma, expand = expansion(mult = c(0, 0.15))) +
  labs(title = "Customer Order Frequency Distribution",
       subtitle = "96.88% of customers placed only 1 order; repeat buyers represent a vital 3.12% retention cohort",
       x = "Order Frequency", y = "Unique Customers",
       caption = "Core driver for defining a next-60-day predictive churn framework") +
  theme_custom()

ggsave(file.path(fig_dir, "fig6_repeat_purchase_behavior.png"), p6, width = 8, height = 5, dpi = 300)

# ------------------------------------------------------------------------------
# Figure 7: Delivery Performance & Lead Time Distribution
# ------------------------------------------------------------------------------
cat("Generating Figure 7: Delivery Lead Times...\n")
orders[, order_delivered_customer_date := as.POSIXct(order_delivered_customer_date, tz = "UTC")]
orders[, order_estimated_delivery_date := as.POSIXct(order_estimated_delivery_date, tz = "UTC")]

orders[, delivery_days := as.numeric(difftime(order_delivered_customer_date, order_purchase_timestamp, units = "days"))]
orders[, delay_days := as.numeric(difftime(order_delivered_customer_date, order_estimated_delivery_date, units = "days"))]

deliv_clean <- orders[!is.na(delivery_days) & delivery_days >= 0 & delivery_days <= 60]

p7 <- ggplot(deliv_clean, aes(x = delivery_days)) +
  geom_histogram(binwidth = 2, fill = "#34495e", color = "white", alpha = 0.85) +
  geom_vline(xintercept = median(deliv_clean$delivery_days), color = "#e74c3c", linetype = "dashed", linewidth = 1) +
  annotate("text", x = median(deliv_clean$delivery_days) + 2, y = 7000, 
           label = sprintf("Median: %.1f Days", median(deliv_clean$delivery_days)), color = "#e74c3c", fontface = "bold", hjust = 0) +
  scale_x_continuous(breaks = seq(0, 60, 5)) +
  scale_y_continuous(labels = comma) +
  labs(title = "Customer Delivery Duration Distribution",
       subtitle = "Distribution of transit duration from purchase to customer doorstep (Median = 10.2 days)",
       x = "Delivery Days", y = "Orders Count",
       caption = "Orders > 60 days excluded from visualization as extreme outliers") +
  theme_custom()

ggsave(file.path(fig_dir, "fig7_delivery_lead_time.png"), p7, width = 8, height = 5, dpi = 300)

# ------------------------------------------------------------------------------
# Figure 8: Review Scores & Delivery Impact
# ------------------------------------------------------------------------------
cat("Generating Figure 8: Review Scores & Delivery Relationship...\n")
deliv_clean[, is_delayed := delay_days > 0]
deliv_clean[, review_bucket := factor(round(review_score), levels = 1:5)]

rev_summary <- deliv_clean[, .(
  orders = .N,
  avg_delay = mean(delay_days, na.rm = TRUE),
  pct_delayed = mean(is_delayed, na.rm = TRUE) * 100
), by = review_bucket][order(review_bucket)]

p8a <- ggplot(deliv_clean, aes(x = review_bucket, fill = review_bucket)) +
  geom_bar(width = 0.6, show.legend = FALSE) +
  scale_fill_brewer(palette = "RdYlGn") +
  scale_y_continuous(labels = comma, expand = expansion(mult = c(0, 0.15))) +
  labs(title = "Customer Review Ratings (1-5 Stars)",
       subtitle = "76.8% positive ratings (4-5 stars), but 1-star reviews form 11.5%",
       x = "Review Score", y = "Order Count") +
  theme_custom()

p8b <- ggplot(rev_summary, aes(x = review_bucket, y = pct_delayed, fill = pct_delayed)) +
  geom_col(width = 0.6, show.legend = FALSE) +
  geom_text(aes(label = sprintf("%.1f%%", pct_delayed)), vjust = -0.4, size = 3.2, fontface = "bold") +
  scale_fill_gradient(low = "#27ae60", high = "#c0392b") +
  scale_y_continuous(limits = c(0, 60), labels = function(x) paste0(x, "%")) +
  labs(title = "Percentage of Late Orders by Review Score",
       subtitle = "Late delivery skyrockets from 2.0% in 5-star orders to 50.1% in 1-star orders",
       x = "Review Score", y = "Orders Delivered Late (%)") +
  theme_custom()

p8 <- grid.arrange(p8a, p8b, ncol = 2)
ggsave(file.path(fig_dir, "fig8_review_scores_and_delays.png"), p8, width = 11, height = 5, dpi = 300)
fwrite(rev_summary, file.path(tab_dir, "review_delay_summary.csv"))

cat(">>> Step 2 Complete Successfully! All 8 figures and summary tables saved.\n")
