#!/usr/bin/env python
# Step 1: Import necessary libraries
import os
import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Step 2: Ensure the results directory exists
os.makedirs("results", exist_ok=True)

# Step 3: Load the cleaned dataset
df = pd.read_csv("Cleaned_13100906.csv")

# Step 4: Filter dataset for relevant health indicators
relevant_indicators = [
    "Perceived health, very good or excellent",
    "Diagnosed chronic conditions",
    "Mental health, fair or poor"
]
df_filtered = df[df["Indicators"].isin(relevant_indicators)].copy()

# Step 5: Convert categorical income levels to numerical values
income_mapping = {
    "Household income, first quintile": 1,
    "Household income, second quintile": 2,
    "Household income, third quintile": 3,
    "Household income, fourth quintile": 4,
    "Household income, fifth quintile": 5
}
df_filtered["Income_Quintile"] = df_filtered["Selected characteristic"].map(income_mapping)

# Step 6: Apply Log-Transformation to `VALUE`
df_filtered["Log_VALUE"] = np.log1p(df_filtered["VALUE"])

# Step 7: Drop missing values before regression
df_filtered = df_filtered.dropna(subset=["Income_Quintile", "Log_VALUE"])

# Step 8: Define independent and dependent variables
X = df_filtered[["Income_Quintile"]]
X = sm.add_constant(X)  # Add intercept
y = df_filtered["Log_VALUE"]

# Step 9: Fit Regression Model
model = sm.OLS(y, X).fit()

# Step 10: Save regression summary
with open("results/enhanced_regression_summary.txt", "w") as f:
    f.write(str(model.summary()))

# Step 11: Save regression plot
plt.figure(figsize=(8, 5))
sns.regplot(x=df_filtered["Income_Quintile"], y=df_filtered["Log_VALUE"], scatter_kws={"color": "blue"}, line_kws={"color": "red"})
plt.xlabel("Income Quintile (1 = Lowest, 5 = Highest)")
plt.ylabel("Log(Chronic Disease Rate + 1)")
plt.title("Regression: Income & Health Indicators vs Log Chronic Disease Prevalence")
plt.savefig("results/enhanced_regression_plot.png")
plt.close()
