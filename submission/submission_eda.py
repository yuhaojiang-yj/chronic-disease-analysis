#!/usr/bin/env python
# Step 1: Import necessary libraries
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Ensure the results directory exists
os.makedirs("results", exist_ok=True)

# Step 3: Load the cleaned dataset
df = pd.read_csv("data/raw/Cleaned_13100906.csv")

# Step 4: Convert `REF_DATE` to datetime properly
df["REF_DATE"] = pd.to_datetime(df["REF_DATE"], format="%Y", errors="coerce")

# Step 5: Display basic summary statistics
summary_stats = df.describe()
summary_stats.to_csv("results/summary_statistics.csv")
print("\n🔹 Summary Statistics Saved.")

# Step 6: Check for missing values
missing_values = df.isna().sum()
missing_values.to_csv("results/missing_values.csv")
print("\n🔹 Missing Values Report Saved.")

# Step 7: Distribution of Chronic Disease Prevalence (`VALUE` column)
plt.figure(figsize=(8, 5))
sns.histplot(df["VALUE"], bins=30, kde=True)
plt.title("Distribution of Chronic Disease Prevalence")
plt.xlabel("Chronic Disease Rate (%)")
plt.ylabel("Frequency")
plt.savefig("results/distribution_plot.png")  # Save plot
plt.close()

# Step 8: Explore Income Quintiles vs Chronic Disease Rate (if available)
if "Selected characteristic" in df.columns:
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="Selected characteristic", y="VALUE")
    plt.xticks(rotation=45)
    plt.title("Income Quintiles vs Chronic Disease Rate")
    plt.xlabel("Income Quintile")
    plt.ylabel("Chronic Disease Rate (%)")
    plt.savefig("results/income_vs_disease.png")  # Save plot
    plt.close()

# Step 9: Correlation heatmap
df_numeric = df.select_dtypes(include=["number"])

if not df_numeric.empty:
    plt.figure(figsize=(8, 5))
    sns.heatmap(df_numeric.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap of Numerical Variables")
    plt.savefig("results/correlation_heatmap.png")  # Save heatmap
    plt.close()
else:
    print("\n⚠️ No numeric columns found for correlation analysis.")
