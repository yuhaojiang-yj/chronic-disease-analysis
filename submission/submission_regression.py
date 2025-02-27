#!/usr/bin/env python
# Step 1: Import necessary libraries
import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Step 2: Load the cleaned dataset
df = pd.read_csv("Cleaned_13100906.csv")

# Step 3: Filter dataset for relevant health indicators
relevant_indicators = [
    "Perceived health, very good or excellent",
    "Diagnosed chronic conditions",
    "Mental health, fair or poor"
]
df_filtered = df[df["Indicators"].isin(relevant_indicators)].copy()

# Step 4: Convert categorical income levels to numerical values
income_mapping = {
    "Household income, first quintile": 1,
    "Household income, second quintile": 2,
    "Household income, third quintile": 3,
    "Household income, fourth quintile": 4,
    "Household income, fifth quintile": 5
}
df_filtered["Income_Quintile"] = df_filtered["Selected characteristic"].map(income_mapping)

# Step 5: Convert categorical columns to numerical codes
df_filtered["Indicators_Code"] = df_filtered["Indicators"].astype("category").cat.codes
df_filtered["Characteristics_Code"] = df_filtered["Characteristics"].astype("category").cat.codes
df_filtered["GEO_Code"] = df_filtered["GEO"].astype("category").cat.codes  # Regional variable

# Step 6: Apply Log-Transformation to `VALUE` (Fix skewness issue)
df_filtered["Log_VALUE"] = np.log1p(df_filtered["VALUE"])  # log(1 + VALUE) to avoid log(0) issues

# Step 7: Drop missing values before regression
df_filtered = df_filtered.dropna(subset=["Income_Quintile", "Log_VALUE", "Indicators_Code", "Characteristics_Code", "GEO_Code"])

# Ensure all variables are numeric
df_filtered["Income_Quintile"] = pd.to_numeric(df_filtered["Income_Quintile"], errors="coerce")
df_filtered["Indicators_Code"] = pd.to_numeric(df_filtered["Indicators_Code"], errors="coerce")
df_filtered["Characteristics_Code"] = pd.to_numeric(df_filtered["Characteristics_Code"], errors="coerce")
df_filtered["GEO_Code"] = pd.to_numeric(df_filtered["GEO_Code"], errors="coerce")
df_filtered["Log_VALUE"] = pd.to_numeric(df_filtered["Log_VALUE"], errors="coerce")

# Step 8: Check for Multicollinearity (Variance Inflation Factor - VIF)
X = df_filtered[["Income_Quintile", "Indicators_Code", "Characteristics_Code", "GEO_Code"]]
X = sm.add_constant(X)  # Add intercept

# Compute VIF for each predictor
vif_data = pd.DataFrame()
vif_data["Variable"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

print("\n🔹 Variance Inflation Factor (VIF) Analysis:\n", vif_data)

# Drop variables with VIF > 5 (indicating strong multicollinearity)
X = X.drop(columns=[col for col, vif in zip(vif_data["Variable"], vif_data["VIF"]) if vif > 5])

# Step 9: Investigate `Indicators_Code`
print("\n🔹 Unique Values in Indicators_Code:\n", df_filtered["Indicators_Code"].value_counts())

# Step 10: Add More Predictors (Test Additional Features)
df_filtered["Interaction_Term"] = df_filtered["Income_Quintile"] * df_filtered["Characteristics_Code"]  # Example of interaction

# Define final X and Y for regression
X["Interaction_Term"] = df_filtered["Interaction_Term"]
y = df_filtered["Log_VALUE"]

# Drop any remaining NaN or infinite values
X = X.replace([float("inf"), -float("inf")], float("nan")).dropna()
y = y.replace([float("inf"), -float("inf")], float("nan")).dropna()

# Ensure X and y match in length
X = X.loc[y.index]

# Step 11: Fit Enhanced Multiple Linear Regression Model
model = sm.OLS(y, X).fit()

# Step 12: Display Regression Results
print(model.summary())

# Step 13: Visualizing the Results
plt.figure(figsize=(8, 5))
sns.regplot(x=df_filtered["Income_Quintile"], y=df_filtered["Log_VALUE"], scatter_kws={"color": "blue"}, line_kws={"color": "red"})
plt.xlabel("Income Quintile (1 = Lowest, 5 = Highest)")
plt.ylabel("Log(Chronic Disease Rate + 1)")
plt.title("Improved Regression: Income & Health Indicators vs Log Chronic Disease Prevalence")
plt.show()
