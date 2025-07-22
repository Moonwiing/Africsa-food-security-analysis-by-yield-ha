# Africsa-food-security-analysis-by-yield-ha

# Load required libraries
library(dplyr)
library(tidyr)
library(ggplot2)
library(xgboost)
library(caret)
library(forecast)

# Assume africa_data is loaded: Area (chr), Item (chr), Year (num), Value (num), Flag Description (chr)

# Step 1: Data Audit
cat("Data Audit:\n")
cat("Missing Values:\n")
colSums(is.na(africa_data))
cat("Unique Years:", length(unique(africa_data$Year)), "\n")
cat("Unique Crops:", length(unique(africa_data$Item)), "\n")
cat("Unique Countries:", length(unique(africa_data$Area)), "\n")
outliers <- africa_data %>%
  group_by(Item) %>%
  summarise(median_yield = median(Value), threshold = median_yield * 10) %>%
  left_join(africa_data, by = "Item") %>%
  filter(Value > threshold)
cat("Outliers (yields > 10x median per crop):", nrow(outliers), "\n")

# Step 2: KPI Calculation
normalize <- function(x) {
  if (max(x, na.rm = TRUE) == min(x, na.rm = TRUE)) return(rep(50, length(x)))
  (x - min(x, na.rm = TRUE)) / (max(x, na.rm = TRUE) - min(x, na.rm = TRUE)) * 100
}

kpi_data <- africa_data %>%
  select(Area, Item, Year, Value) %>%
  group_by(Area, Item) %>%
  summarise(
    mean_yield = mean(Value, na.rm = TRUE),
    sd_yield = sd(Value, na.rm = TRUE),
    cv_yield = ifelse(mean_yield == 0, 0, sd_yield / mean_yield),
    growth_rate = ifelse(first(Value) == 0, 0, 
                         (last(Value) - first(Value)) / first(Value) * 100 / (max(Year) - min(Year))),
    .groups = "drop"
  ) %>%
  group_by(Item) %>%
  mutate(
    continent_mean = mean(mean_yield, na.rm = TRUE),
    relative_performance = ifelse(continent_mean == 0, 100, (mean_yield / continent_mean) * 100)
  ) %>%
  ungroup() %>%
  mutate(
    norm_yield = normalize(mean_yield),
    norm_growth = normalize(growth_rate),
    norm_stability = normalize(-cv_yield),
    kpi_score = 0.4 * norm_yield + 0.3 * norm_growth + 0.3 * norm_stability
  ) %>%
  filter(!is.na(kpi_score) & is.finite(kpi_score))

# Step 3: EDA for Docet Questions
# A. Crop Yield Patterns and Drivers
cat("\nCrops with Consistent Yield Growth (Top 5):\n")
growth_crops <- kpi_data %>%
  arrange(desc(growth_rate)) %>%
  select(Item, growth_rate) %>%
  dplyr::slice_head(n = 5)
growth_crops

cat("\nCrops with Significant Yield Decline (Top 5):\n")
decline_crops <- kpi_data %>%
  arrange(growth_rate) %>%
  select(Item, growth_rate) %>%
  dplyr::slice_head(n = 5)
decline_crops

cat("\nCrops with Highest Volatility (Top 5):\n")
volatile_crops <- kpi_data %>%
  arrange(desc(cv_yield)) %>%
  select(Item, cv_yield) %>%
  dplyr::slice_head(n = 5)
volatile_crops

cat("\nAverage Yield per Crop (Top 5):\n")
avg_yield <- kpi_data %>%
  group_by(Item) %>%
  summarise(avg_yield = mean(mean_yield)) %>%
  arrange(desc(avg_yield)) %>%
  dplyr::slice_head(n = 5)
avg_yield

# Cyclical patterns (LOESS smoothing for Maize)
p1 <- africa_data %>%
  filter(Item == "Maize (corn)") %>%
  group_by(Year) %>%
  summarise(mean_yield = mean(Value)) %>%
  ggplot(aes(x = Year, y = mean_yield)) +
  geom_line() +
  geom_smooth(method = "loess", se = FALSE, color = "red") +
  labs(title = "Maize Yield Trend with LOESS Smoothing", x = "Year", y = "Mean Yield (hg/ha)")
p1

# B. Temporal and Regional Dynamics
cat("\nYearly Average Yield Across All Crops:\n")
yearly_yield <- africa_data %>%
  group_by(Year) %>%
  summarise(mean_yield = mean(Value)) %>%
  arrange(Year)
yearly_yield

cat("\nTop 3 Years with Highest Yield:\n")
yearly_yield %>% arrange(desc(mean_yield)) %>% dplyr::slice_head(n = 3)

cat("\nTop 3 Years with Lowest Yield:\n")
yearly_yield %>% arrange(mean_yield) %>% dplyr::slice_head(n = 3)

cat("\nPeriod Comparison (2000-2007 vs 2015-2023):\n")
period_comparison <- africa_data %>%
  mutate(Period = ifelse(Year <= 2007, "2000-2007", "2015-2023")) %>%
  filter(Period != "2008-2014") %>%
  group_by(Period) %>%
  summarise(mean_yield = mean(Value))
period_comparison

# C. Cross-Crop Comparative Analysis
cat("\nMost Efficient Crops (Top 5 by Mean Yield):\n")
avg_yield

cat("\nStaple vs High-Value Crops Comparison:\n")
staple_high_value <- kpi_data %>%
  filter(Item %in% c("Maize (corn)", "Sorghum", "Avocados", "Mangoes, guavas and mangosteens")) %>%
  group_by(Item) %>%
  summarise(
    mean_yield = mean(mean_yield),
    mean_growth = mean(growth_rate),
    mean_stability = mean(norm_stability)
  )
staple_high_value

# D. Country-Level Analysis
cat("\nCountries with Highest Mean Yield (Top 5, All Crops):\n")
country_yield <- kpi_data %>%
  group_by(Area) %>%
  summarise(avg_yield = mean(mean_yield)) %>%
  arrange(desc(avg_yield)) %>%
  dplyr::slice_head(n = 5)
country_yield

cat("\nCountries with Consistent Yield Growth (Top 5, All Crops):\n")
country_growth <- kpi_data %>%
  group_by(Area) %>%
  summarise(avg_growth = mean(growth_rate)) %>%
  arrange(desc(avg_growth)) %>%
  dplyr::slice_head(n = 5)
country_growth

cat("\nCountries with Highest Volatility (Top 5, All Crops):\n")
country_volatility <- kpi_data %>%
  group_by(Area) %>%
  summarise(avg_cv_yield = mean(cv_yield)) %>%
  arrange(desc(avg_cv_yield)) %>%
  dplyr::slice_head(n = 5)
country_volatility

# Maize yield trends for top/bottom countries
top_bottom_countries <- kpi_data %>%
  filter(Item == "Maize (corn)") %>%
  group_by(Area) %>%
  summarise(avg_yield = mean(mean_yield)) %>%
  arrange(desc(avg_yield)) %>%
  dplyr::slice(c(1:3, (n()-2):n())) %>%
  pull(Area)

p2 <- africa_data %>%
  filter(Item == "Maize (corn)", Area %in% top_bottom_countries) %>%
  group_by(Area, Year) %>%
  summarise(mean_yield = mean(Value), .groups = "drop") %>%
  ggplot(aes(x = Year, y = mean_yield, color = Area)) +
  geom_line() +
  labs(title = "Maize Yield Trends for Top/Bottom Countries", x = "Year", y = "Mean Yield (hg/ha)")
p2

# Step 4: XGBoost with Fine-Tuning
# Encode Area and Item numerically
area_encoding <- kpi_data %>%
  group_by(Area) %>%
  summarise(area_yield = mean(mean_yield)) %>%
  mutate(area_id = row_number())

item_encoding <- kpi_data %>%
  group_by(Item) %>%
  summarise(item_yield = mean(mean_yield)) %>%
  mutate(item_id = row_number())

ml_data <- kpi_data %>%
  left_join(area_encoding, by = "Area") %>%
  left_join(item_encoding, by = "Item") %>%
  select(area_id, item_id, mean_yield, growth_rate, cv_yield, kpi_score) %>%
  filter(is.finite(mean_yield) & is.finite(growth_rate) & is.finite(cv_yield))

set.seed(123)
train_index <- createDataPartition(ml_data$kpi_score, p = 0.8, list = FALSE)
train_data <- ml_data[train_index, ]
test_data <- ml_data[-train_index, ]

# Grid search for XGBoost hyperparameters
param_grid <- expand.grid(
  max_depth = c(3, 6, 9),
  eta = c(0.01, 0.1, 0.3),
  nrounds = c(50, 100, 200)
)

best_rmse <- Inf
best_model <- NULL
best_params <- NULL

for (i in 1:nrow(param_grid)) {
  xgb_matrix <- xgb.DMatrix(
    data = as.matrix(train_data[, c("area_id", "item_id", "mean_yield", "growth_rate", "cv_yield")]),
    label = train_data$kpi_score
  )
  xgb_model <- xgboost(
    data = xgb_matrix,
    nrounds = param_grid$nrounds[i],
    max_depth = param_grid$max_depth[i],
    eta = param_grid$eta[i],
    objective = "reg:squarederror",
    verbose = 0
  )
  xgb_test_matrix <- xgb.DMatrix(
    data = as.matrix(test_data[, c("area_id", "item_id", "mean_yield", "growth_rate", "cv_yield")])
  )
  predictions <- predict(xgb_model, xgb_test_matrix)
  rmse <- sqrt(mean((predictions - test_data$kpi_score)^2))
  if (rmse < best_rmse) {
    best_rmse <- rmse
    best_model <- xgb_model
    best_params <- param_grid[i, ]
  }
}

cat("\nBest XGBoost RMSE:", best_rmse, "\n")
cat("Best Parameters: max_depth =", best_params$max_depth, 
    ", eta =", best_params$eta, ", nrounds =", best_params$nrounds, "\n")

# Feature importance
cat("\nFeature Importance (Best Model):\n")
importance_matrix <- xgb.importance(model = best_model)
importance_matrix

# Plot feature importance
p3 <- xgb.plot.importance(importance_matrix, main = "Feature Importance for KPI Score (XGBoost)")
p3

# Step 5: ARIMA for KPI Trend Prediction (Maize)
maize_ts <- africa_data %>%
  filter(Item == "Maize (corn)") %>%
  group_by(Year) %>%
  summarise(mean_yield = mean(Value)) %>%
  arrange(Year)

maize_ts$kpi_score <- normalize(maize_ts$mean_yield)

arima_model <- auto.arima(maize_ts$kpi_score)
forecast_result <- forecast(arima_model, h = 1)
cat("\nPredicted KPI Score for Next Year (Maize):", as.numeric(forecast_result$mean) * 100, "\n")

p4 <- autoplot(forecast_result) +
  labs(title = "Maize KPI Forecast (ARIMA)", x = "Year", y = "KPI Score (0-100)")
p4

# Step 6: SDG Mapping
cat("\nSDG 2 (Zero Hunger) Insights:\n")
cat("Top crops by mean yield (food security): ", paste(avg_yield$Item[1:3], collapse = ", "), "\n")
cat("Top countries by mean yield (food security): ", paste(country_yield$Area[1:3], collapse = ", "), "\n")
cat("Most stable crops (low volatility): ", paste(kpi_data %>% 
                                                    arrange(cv_yield) %>% 
                                                    select(Item) %>% 
                                                    dplyr::slice_head(n = 3) %>% 
                                                    pull(Item), collapse = ", "), "\n")
cat("Most stable countries (low volatility): ", paste(kpi_data %>% 
                                                        arrange(cv_yield) %>% 
                                                        select(Area) %>% 
                                                        dplyr::slice_head(n = 3) %>% 
                                                        pull(Area), collapse = ", "), "\n")

cat("\nSDG 12 (Responsible Consumption) Insights:\n")
cat("Most efficient crops (high yield, low volatility): ", 
    paste(kpi_data %>% arrange(desc(kpi_score)) %>% 
            select(Item) %>% dplyr::slice_head(n = 3) %>% pull(Item), collapse = ", "), "\n")
cat("Most efficient countries (high KPI score): ", 
    paste(kpi_data %>% arrange(desc(kpi_score)) %>% 
            select(Area) %>% dplyr::slice_head(n = 3) %>% pull(Area), collapse = ", "), "\n")

# Step 7: Summary
cat("\nAnalysis Completed. Key Outputs:\n")
cat("- Data audit results\n")
cat("- Crop yield patterns: growth, decline, volatility, efficiency\n")
cat("- Country-level analysis: yield, growth, volatility, Maize trends\n")
cat("- Temporal dynamics: yearly trends, top/bottom years, period comparison\n")
cat("- Fine-tuned XGBoost RMSE and feature importance\n")
cat("- ARIMA forecast for Maize KPI\n")
cat("- SDG insights for Zero Hunger and Responsible Consumption\n")
