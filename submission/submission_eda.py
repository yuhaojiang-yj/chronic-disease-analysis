#!/usr/bin/env python
# Step 1: Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Load the cleaned dataset
df = pd.read_csv("Cleaned_13100906.csv")

# Step 3: Convert `REF_DATE` to datetime properly
df["REF_DATE"] = pd.to_datetime(df["REF_DATE"], format="%Y", errors="coerce")

# Step 4: Display basic summary statistics
print("\n🔹 Summary Statistics:")
print(df.describe())

# Step 5: Check for missing values
print("\n🔹 Missing Values Per Column:")
print(df.isna().sum())

# Step 6: Distribution of Chronic Disease Prevalence (`VALUE` column)
plt.figure(figsize=(8, 5))
sns.histplot(df["VALUE"], bins=30, kde=True)
plt.title("Distribution of Chronic Disease Prevalence")
plt.xlabel("Chronic Disease Rate (%)")
plt.ylabel("Frequency")
plt.show()

# Step 7: Explore Income Quintiles vs Chronic Disease Rate (if available in dataset)
if "Selected characteristic" in df.columns:
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="Selected characteristic", y="VALUE")
    plt.xticks(rotation=45)
    plt.title("Income Quintiles vs Chronic Disease Rate")
    plt.xlabel("Income Quintile")
    plt.ylabel("Chronic Disease Rate (%)")
    plt.show()

# Step 8: Correlation heatmap (Fixing the error)
# Select only numeric columns
df_numeric = df.select_dtypes(include=["number"])

# Check if there are enough numeric columns
if not df_numeric.empty:
    plt.figure(figsize=(8, 5))
    sns.heatmap(df_numeric.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap of Numerical Variables")
    plt.show()
else:
    print("\n⚠️ No numeric columns found for correlation analysis.")
