import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np

df = pd.read_csv("/Users/ruhirayamajhi/Desktop/Python/HIT140/cleaned_bat_data.csv")
print(df.shape)
print(df.columns)
print(df.dtypes)
print(df.isna().sum())

# Conversion of time columns to datetime format
df['start_time'] = pd.to_datetime(df['start_time'], errors='coerce')
df['rat_period_start'] = pd.to_datetime(df['rat_period_start'], errors='coerce')
df['rat_period_end'] = pd.to_datetime(df['rat_period_end'], errors='coerce')
df['sunset_time'] = pd.to_datetime(df['sunset_time'], errors='coerce')
# df['rat_arrival_time'] = df['start_time'] - pd.to_timedelta(df['seconds_after_rat_arrival'], unit='s')
# Ensure hours_after_sunset numeric
df['hours_after_sunset'] = pd.to_numeric(df['hours_after_sunset'], errors='coerce')

df_model = df.dropna(subset=['risk','seconds_after_rat_arrival','hours_after_sunset']).copy()
df_with_habit = df_model.dropna(subset=['habit']).copy()

#Boxplot: Seconds After Rat Arrival vs Bat Risk Behaviour
plt.figure(figsize=(6,5))
sns.boxplot(data=df, x='risk', y='seconds_after_rat_arrival')
plt.xlabel('Risk (0=no, 1=yes)')
plt.ylabel('Seconds after rat arrival')
plt.title('Seconds After Rat Arrival vs Bat Risk Behaviour')
plt.tight_layout()
plt.savefig('fig_seconds_after_rat_by_risk.png', dpi=200)
plt.show()

#Histogram / density of seconds_after_rat_arrival and hours_after_sunset
plt.figure(figsize=(6,4))
sns.histplot(df['seconds_after_rat_arrival'], kde=True)
plt.title('Distribution of seconds_after_rat_arrival')
plt.tight_layout()
plt.savefig('fig_seconds_after_rat_hist.png', dpi=200)
plt.show()

plt.figure(figsize=(6,4))
sns.histplot(df['hours_after_sunset'].dropna(), kde=True)
plt.title('Distribution of hours_after_sunset')
plt.tight_layout()
plt.savefig('fig_hours_after_sunset_hist.png', dpi=200)
plt.show()

#Correlation matrix
numcols = ['seconds_after_rat_arrival','hours_after_sunset','bat_landing_to_food','reward','risk']
corr = df[numcols].corr()
plt.figure(figsize=(6,5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='vlag', center=0)
plt.title('Correlation matrix (numeric)')
plt.tight_layout()
plt.savefig('fig_corr_matrix.png', dpi=200)
plt.show()

# Prepare variables
X = df[['seconds_after_rat_arrival','hours_after_sunset']].copy()
X = X.fillna(0)  # or better: dropna if you prefer
X = sm.add_constant(X)
y = df['risk']  # assume 0/1 values

# Fit logistic model
model = sm.Logit(y, X).fit(disp=False)
print(model.summary())

# Odds ratios and 95% CI
params = model.params
conf = model.conf_int()
odds = np.exp(params)
ci_lower = np.exp(conf[0])
ci_upper = np.exp(conf[1])
or_table = pd.DataFrame({
    'coef': params,
    'odds_ratio': odds,
    'ci_lower': ci_lower,
    'ci_upper': ci_upper,
    'pvalue': model.pvalues
})
print(or_table)

# ROC and AUC
preds = model.predict(X)
auc = roc_auc_score(y, preds)
fpr, tpr, _ = roc_curve(y, preds)
plt.figure(figsize=(5,5))
plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
plt.plot([0,1],[0,1],'--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.legend()
plt.tight_layout()
plt.savefig('fig_logit_roc.png', dpi=200)
plt.show()

from statsmodels.stats.outliers_influence import variance_inflation_factor
X_vif = X.drop(columns='const').assign(const=1)  # VIF expects no constant at front
vifs = pd.Series([variance_inflation_factor(X_vif.values, i) 
                  for i in range(X_vif.shape[1])], index=X_vif.columns)
print(vifs)

df['sec_q'] = pd.qcut(df['seconds_after_rat_arrival'], 4, labels=False)
# then fit model with sec_q as categorical via get_dummies

or_table.to_csv('logit_odds_ratios.csv', index=True)

sec_grid = np.linspace(df['seconds_after_rat_arrival'].min(), df['seconds_after_rat_arrival'].max(), 200)
X_pred = pd.DataFrame({
    'const':1,
    'seconds_after_rat_arrival': sec_grid,
    'hours_after_sunset': df['hours_after_sunset'].median()
})
probs = model.predict(X_pred)
plt.figure(figsize=(6,4))
plt.plot(sec_grid, probs)
plt.xlabel('Seconds after rat arrival')
plt.ylabel('Predicted probability of taking risk')
plt.title('Predicted probability of risk vs seconds after rat arrival')
plt.tight_layout()
plt.savefig('fig_pred_prob_vs_seconds.png', dpi=200)
plt.show()
