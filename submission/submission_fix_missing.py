#!/usr/bin/env python
# Step 1: Import necessary libraries
import os
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

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

# Step 6: Convert categorical "Indicators" column to numerical codes
df_filtered["Indicators_Code"] = df_filtered["Indicators"].astype("category").cat.codes

# Step 7: Drop missing values before regression
df_filtered = df_filtered.dropna(subset=["Income_Quintile", "VALUE", "Indicators_Code"])

# Step 8: Define dependent (Y) and independent (X) variables
X = df_filtered[["Income_Quintile", "Indicators_Code"]]
y = df_filtered["VALUE"]

X = sm.add_constant(X)  # Add a constant for OLS regression

# Step 9: Fit Multiple Linear Regression Model
model = sm.OLS(y, X).fit()

# Step 10: Save regression summary
with open("results/regression_summary.txt", "w") as f:
    f.write(str(model.summary()))

# Step 11: Save regression plot
plt.figure(figsize=(8, 5))
sns.regplot(x=df_filtered["Income_Quintile"], y=df_filtered["VALUE"], scatter_kws={"color": "blue"}, line_kws={"color": "red"})
plt.xlabel("Income Quintile (1 = Lowest, 5 = Highest)")
plt.ylabel("Chronic Disease Rate (%)")
plt.title("Multiple Regression: Income & Health Indicators vs Chronic Disease Prevalence")
plt.savefig("results/regression_plot.png")
plt.close()
