# scripts/run_all.R
# ==============================================================================
# Master Execution Pipeline: Retail Predictive Analytics
# Runs all 6 modules sequentially and verifies complete execution
# ==============================================================================

cat("================================================================================\n")
cat("       RETAIL PREDICTIVE ANALYTICS - END-TO-END MASTER PIPELINE EXECUTION        \n")
cat("================================================================================\n\n")

start_time <- Sys.time()

scripts <- c(
  "scripts/01_data_audit_and_cleaning.R",
  "scripts/02_eda_and_visualizations.R",
  "scripts/03_target_and_feature_engineering.R",
  "scripts/04_model_training_and_tuning.R",
  "scripts/05_evaluation_fairness_xai.R",
  "scripts/06_predict_pipeline.R"
)

for (s in scripts) {
  cat(sprintf("\n>>> Executing: %s ...\n", s))
  step_start <- Sys.time()
  res <- tryCatch({
    source(s)
    cat(sprintf(">>> SUCCESS: %s completed in %.2f seconds.\n", s, as.numeric(difftime(Sys.time(), step_start, units="secs"))))
    TRUE
  }, error = function(e) {
    cat(sprintf(">>> ERROR in %s: %s\n", s, e$message))
    FALSE
  })
  if (!res) {
    stop(sprintf("Pipeline halted due to error in %s", s))
  }
}

total_elapsed <- as.numeric(difftime(Sys.time(), start_time, units="secs"))

cat("\n================================================================================\n")
cat(sprintf("ALL 6 MODULES COMPLETED SUCCESSFULLY in %.2f seconds (%.2f minutes)!\n", total_elapsed, total_elapsed / 60))
cat("================================================================================\n")

# Verify all key outputs exist
key_outputs <- c(
  "outputs/cleaned_orders_master.rds",
  "outputs/figures/fig1_monthly_trends.png",
  "outputs/figures/fig6_repeat_purchase_behavior.png",
  "outputs/figures/fig_business_lift.png",
  "outputs/figures/fig_shap_summary.png",
  "outputs/figures/fig_partial_dependence.png",
  "outputs/figures/fig_calibration_curve.png",
  "outputs/models/churn_model.rds",
  "outputs/tables/test_evaluation_metrics.csv",
  "outputs/tables/model_stability_comparison.csv",
  "outputs/tables/business_lift_table.csv",
  "outputs/tables/regional_fairness_summary.csv",
  "outputs/tables/data_drift_summary.csv",
  "outputs/tables/sample_scored_predictions.csv",
  "notebooks/retail_predictive_analytics.Rmd",
  "reports/retail_predictive_analytics.html",
  "reports/model_card.md",
  "reports/presentation_slides.html"
)

cat("\n--- DELIVERABLES VERIFICATION ---\n")
all_found <- TRUE
for (out in key_outputs) {
  exists_flag <- file.exists(out)
  if (!exists_flag) all_found <- FALSE
  cat(sprintf(" [%s] %-50s (%.1f KB)\n", 
              ifelse(exists_flag, "EXISTS", "MISSING"), 
              out, 
              ifelse(exists_flag, file.size(out)/1024, 0)))
}

cat("\nStatus: ", ifelse(all_found, "ALL DELIVERABLES VERIFIED AND PRESENT!", "SOME ARTIFACTS MISSING"), "\n")
