#!/usr/bin/env python
# Step 1: Import necessary libraries
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

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

# Step 5: Convert categorical "Indicators" column to numerical codes
df_filtered["Indicators_Code"] = df_filtered["Indicators"].astype("category").cat.codes

# Step 6: Drop missing values before regression
df_filtered = df_filtered.dropna(subset=["Income_Quintile", "VALUE", "Indicators_Code"])

# Ensure all variables are numeric
df_filtered["Income_Quintile"] = pd.to_numeric(df_filtered["Income_Quintile"], errors="coerce")
df_filtered["Indicators_Code"] = pd.to_numeric(df_filtered["Indicators_Code"], errors="coerce")
df_filtered["VALUE"] = pd.to_numeric(df_filtered["VALUE"], errors="coerce")

# Step 7: Define dependent (Y) and independent (X) variables for Multiple Regression
X = df_filtered[["Income_Quintile", "Indicators_Code"]]  # Independent variables
y = df_filtered["VALUE"]  # Dependent variable (Chronic disease prevalence)

# Drop any remaining NaN or infinite values
X = X.replace([float("inf"), -float("inf")], float("nan")).dropna()
y = y.replace([float("inf"), -float("inf")], float("nan")).dropna()

# Ensure the sizes of X and y match
X = X.loc[y.index]

# Add a constant for OLS regression
X = sm.add_constant(X)

# Step 8: Fit Multiple Linear Regression Model
model = sm.OLS(y, X).fit()

# Step 9: Display Regression Results
print(model.summary())

# Step 10: Visualizing the Results
plt.figure(figsize=(8, 5))
sns.regplot(x=df_filtered["Income_Quintile"], y=df_filtered["VALUE"], scatter_kws={"color": "blue"}, line_kws={"color": "red"})
plt.xlabel("Income Quintile (1 = Lowest, 5 = Highest)")
plt.ylabel("Chronic Disease Rate (%)")
plt.title("Multiple Regression: Income & Health Indicators vs Chronic Disease Prevalence")
plt.show()
