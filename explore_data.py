import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Setup ---
sns.set_style("whitegrid")
if not os.path.exists('advanced_plots'):
    os.makedirs('advanced_plots')

# --- 1. Load the Data ---
try:
    df = pd.read_csv('train.csv')
except FileNotFoundError:
    print("Error: 'train.csv' not found. Please make sure the file is in the same directory.")
    exit()

# --- 2. Generate Advanced Visualizations ---

# Plot 1: Donut Chart for Loan Status Distribution
plt.figure(figsize=(8, 8))
status_counts = df['loan_status'].value_counts()
labels = ['Not Defaulted (0)', 'Defaulted (1)']
colors = ['#4a90e2', '#d0021b'] # Blue for safe, Red for default
plt.pie(status_counts, labels=labels, autopct='%1.1f%%', startangle=90, wedgeprops=dict(width=0.4), colors=colors, textprops={'fontsize': 14})
plt.title('Proportion of Loan Status', fontsize=18, weight='bold')
plt.savefig('advanced_plots/1_loan_status_donut.png')
print("Saved plot: 1_loan_status_donut.png")
plt.close()

# Plot 2: Stacked Percentage Bar Chart for Home Ownership
plt.figure(figsize=(12, 7))
# Create a crosstab and normalize it to get percentages
crosstab_norm = pd.crosstab(df['person_home_ownership'], df['loan_status'], normalize='index') * 100
crosstab_norm.plot(kind='bar', stacked=True, color=['#4a90e2', '#d0021b'], width=0.7)
plt.title('Default Rate by Home Ownership', fontsize=18, weight='bold')
plt.xlabel('Home Ownership Type', fontsize=12)
plt.ylabel('Percentage of Applicants (%)', fontsize=12)
plt.xticks(rotation=0)
plt.legend(['Not Defaulted', 'Defaulted'], loc='upper right')
plt.savefig('advanced_plots/2_home_ownership_stacked.png')
print("Saved plot: 2_home_ownership_stacked.png")
plt.close()

# Plot 3: Violin Plot for Income vs. Loan Status
plt.figure(figsize=(12, 8))
sns.violinplot(x='loan_status', y='person_income', data=df, palette=['#4a90e2', '#d0021b'])
plt.title('Income Distribution by Loan Status', fontsize=18, weight='bold')
plt.xlabel('Loan Status', fontsize=12)
plt.ylabel('Applicant Income', fontsize=12)
plt.xticks([0, 1], ['Not Defaulted', 'Defaulted'])
# Limit y-axis to see the main distribution clearly
plt.ylim(0, 250000)
plt.savefig('advanced_plots/3_income_violin_plot.png')
print("Saved plot: 3_income_violin_plot.png")
plt.close()


# Plot 4: Correlation Heatmap
# Select only numerical columns for correlation calculation
numerical_df = df.select_dtypes(include=['number'])
plt.figure(figsize=(16, 12))
correlation_matrix = numerical_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
plt.title('Correlation Matrix of Numerical Features', fontsize=18, weight='bold')
plt.savefig('advanced_plots/4_correlation_heatmap.png')
print("Saved plot: 4_correlation_heatmap.png")
plt.close()

print("\n--- Advanced EDA Script Finished ---")
print("Check the 'advanced_plots' folder for your new visualizations.")
