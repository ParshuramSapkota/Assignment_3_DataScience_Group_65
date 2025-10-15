# HIT140 Objective 2 – Investigation A & B
# Author: [Your Name]
# Date: October 2025
# ===============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression

# ===============================================
# 1. Load both datasets
# ===============================================
bat_df = pd.read_csv("dataset1.csv")   # bat behaviours
rat_df = pd.read_csv("dataset2.csv")   # rat activities

print("\n--- Dataset Shapes ---")
print(f"Bat dataset: {bat_df.shape}")
print(f"Rat dataset: {rat_df.shape}")

# ===============================================
# 2. Data Cleaning – General
# ===============================================
# Drop duplicates
bat_df = bat_df.drop_duplicates()
rat_df = rat_df.drop_duplicates()

# Handle missing values
print("\nMissing values before cleaning:")
print("Bat dataset:\n", bat_df.isnull().sum())
print("Rat dataset:\n", rat_df.isnull().sum())

# Drop rows missing critical values
bat_df = bat_df.dropna(subset=['month', 'risk', 'reward'])
rat_df = rat_df.dropna(subset=['month', 'bat_landing_number', 'rat_minutes'])

# Convert month to string (for season mapping)
bat_df['month'] = bat_df['month'].astype(str)
rat_df['month'] = rat_df['month'].astype(str)

# ===============================================
# 3. Season Mapping Function
# ===============================================
def month_to_season(m):
    m = str(m).strip().lower()
    if m in ['12', '1', '2', 'dec', 'december', 'jan', 'january', 'feb', 'february']:
        return 'Summer'
    elif m in ['3', '4', '5', 'mar', 'march', 'apr', 'april', 'may']:
        return 'Autumn'
    elif m in ['6', '7', '8', 'jun', 'june', 'jul', 'july', 'aug', 'august']:
        return 'Winter'
    elif m in ['9', '10', '11', 'sep', 'september', 'oct', 'october', 'nov', 'november']:
        return 'Spring'
    else:
        return np.nan

bat_df['season'] = bat_df['month'].apply(month_to_season)
rat_df['season'] = rat_df['month'].apply(month_to_season)

# Remove unmapped rows
bat_df = bat_df.dropna(subset=['season'])
rat_df = rat_df.dropna(subset=['season'])

# ===============================================
# 4. Export Cleaned Datasets
# ===============================================
bat_df.to_csv("clean_bat_dataset.csv", index=False)
rat_df.to_csv("clean_rat_dataset.csv", index=False)
print("\n✅ Cleaned datasets saved as 'clean_bat_dataset.csv' and 'clean_rat_dataset.csv'.")

# ===============================================
# 5. Investigation A – Relationship Between Rat and Bat Behaviour
# ===============================================

# Merge datasets on common temporal context (month + season)
merged_df = pd.merge(rat_df, bat_df, on=['month', 'season'], how='inner', suffixes=('_rat', '_bat'))
print(f"\nMerged dataset shape: {merged_df.shape}")

# Explore correlation between rat activity and bat landing numbers
corr = merged_df[['rat_minutes', 'rat_arrival_number', 'bat_landing_number']].corr()
plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation: Rat Activity vs Bat Landings")
plt.show()

# Linear regression – does rat presence affect bat landings?
X = merged_df[['rat_minutes']]
y = merged_df['bat_landing_number']
model = LinearRegression()
model.fit(X, y)
r2 = model.score(X, y)

print(f"\nLinear Regression Results (Investigation A)")
print(f"R² = {r2:.3f}")
print(f"Equation: Bat Landings = {model.coef_[0]:.3f} * Rat Minutes + {model.intercept_:.3f}")

plt.figure(figsize=(7,5))
sns.regplot(data=merged_df, x='rat_minutes', y='bat_landing_number', scatter_kws={'alpha':0.6})
plt.title("Effect of Rat Presence on Bat Landings")
plt.xlabel("Rat Minutes (duration of presence)")
plt.ylabel("Bat Landings (count)")
plt.show()

# ===============================================
# 6. Investigation B – Seasonal Differences
# ===============================================
# Seasonal comparison for both datasets
bat_season = bat_df.groupby('season')[['risk', 'reward']].mean().reset_index()
rat_season = rat_df.groupby('season')[['bat_landing_number', 'rat_minutes', 'rat_arrival_number']].mean().reset_index()

print("\nAverage Bat Behaviour by Season:")
print(bat_season)
print("\nAverage Rat Activity by Season:")
print(rat_season)

# Plot – Seasonal changes in rat activity
plt.figure(figsize=(8,6))
sns.barplot(data=rat_season.melt(id_vars='season', var_name='Variable', value_name='Average'),
            x='season', y='Average', hue='Variable')
plt.title("Seasonal Comparison: Rat Activity & Bat Landings")
plt.xlabel("Season")
plt.ylabel("Average Count/Duration")
plt.show()

# Plot – Bat risk vs reward behaviour by season
plt.figure(figsize=(6,5))
bat_plot = bat_season.melt(id_vars='season', var_name='Behaviour', value_name='Average')
sns.barplot(data=bat_plot, x='season', y='Average', hue='Behaviour')
plt.title("Bat Behaviour Across Seasons (Risk vs Reward)")
plt.xlabel("Season")
plt.ylabel("Proportion")
plt.show()

# ===============================================
# 7. Statistical Testing
# ===============================================
# ANOVA for seasonal difference in bat landings
anova_groups = [group["bat_landing_number"].values for name, group in rat_df.groupby("season")]
if len(anova_groups) >= 2:
    anova = stats.f_oneway(*anova_groups)
    print(f"\nANOVA p-value for seasonal difference in bat landings: {anova.pvalue:.4f}")
else:
    print("\nANOVA could not be performed (need at least two seasons).")

# ===============================================
# 8. Summary Findings
# ===============================================
print("\n===== SUMMARY =====")
if model.coef_[0] < 0:
    print("→ Negative relationship: More rat presence reduces bat landings (risk avoidance).")
else:
    print("→ Positive relationship: More rat presence increases bat landings (risk-taking).")

if len(anova_groups) >= 2:
    if anova.pvalue < 0.05:
        print("→ Significant seasonal differences detected in bat behaviour.")
    else:
        print("→ No strong evidence of seasonal variation in bat landings.")
print("===================")
