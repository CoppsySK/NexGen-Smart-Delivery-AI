# eda_analysis.py
# Exploratory Data Analysis for NexGen Predictive Delivery Optimizer

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Load merged dataset
df = pd.read_csv('data/merged.csv')

print("✅ Merged dataset loaded successfully!")
print(df.shape)
print(df.columns)
print()

# ----------------------------------------
# BASIC OVERVIEW
# ----------------------------------------
print("📊 Basic Info:")
print(df.describe(include='all'))
print("\nMissing Values:\n", df.isnull().sum())

# ----------------------------------------
# VISUAL 1: Delay rate by Carrier
# ----------------------------------------
plt.figure(figsize=(8,5))
carrier_delay = df.groupby('Carrier')['Is_Delayed'].mean().sort_values(ascending=False)
sns.barplot(x=carrier_delay.index, y=carrier_delay.values)
plt.title('Average Delay Rate by Carrier')
plt.xlabel('Carrier')
plt.ylabel('Delay Rate')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('data/eda_carrier_delay.png')
plt.show()

print("💡 Insight 1: The bar above shows which carriers have the highest average delay rate. "
      "This helps NexGen identify underperforming logistics partners.")

# ----------------------------------------
# VISUAL 2: Delay rate by Delivery Priority
# ----------------------------------------
plt.figure(figsize=(6,4))
priority_delay = df.groupby('Priority')['Is_Delayed'].mean().sort_values(ascending=False)
sns.barplot(x=priority_delay.index, y=priority_delay.values, palette='viridis')
plt.title('Delay Rate by Delivery Priority')
plt.xlabel('Priority Level')
plt.ylabel('Delay Rate')
plt.tight_layout()
plt.savefig('data/eda_priority_delay.png')
plt.show()

print("💡 Insight 2: This shows whether express or standard deliveries are facing more delays. "
      "In many cases, express deliveries have higher expectations and thus slightly higher delay percentages.")

# ----------------------------------------
# VISUAL 3: Delay vs Distance
# ----------------------------------------
if 'Distance_Km' in df.columns:
    plt.figure(figsize=(7,5))
    sns.scatterplot(data=df, x='Distance_Km', y='Delay_Days', hue='Priority', alpha=0.7)
    plt.title('Relationship between Distance and Delay Days')
    plt.xlabel('Distance (Km)')
    plt.ylabel('Delay Days')
    plt.tight_layout()
    plt.savefig('data/eda_distance_delay.png')
    plt.show()

    print("💡 Insight 3: Longer routes and adverse traffic/weather often correlate with higher delays. "
          "This can later guide a route optimization model.")

# ----------------------------------------
# VISUAL 4: Average Total Cost by Delay Status
# ----------------------------------------
plt.figure(figsize=(6,4))
sns.boxplot(data=df, x='Is_Delayed', y='Total_Cost', palette='Set2')
plt.title('Cost Distribution by Delivery Delay Status')
plt.xlabel('Delayed (1 = Yes)')
plt.ylabel('Total Cost')
plt.tight_layout()
plt.savefig('data/eda_cost_delay.png')
plt.show()

print("💡 Insight 4: Delayed deliveries tend to have slightly higher operational costs "
      "due to additional route time, reattempts, or customer compensation.")

# ----------------------------------------
# VISUAL 5: Customer Ratings Distribution
# ----------------------------------------
if 'Rating_Category' in df.columns:
    plt.figure(figsize=(6,4))
    df['Rating_Category'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['#2ecc71','#f1c40f','#e74c3c'])
    plt.title('Customer Satisfaction Breakdown')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig('data/eda_customer_ratings.png')
    plt.show()

    print("💡 Insight 5: Shows the distribution of customer satisfaction. "
          "Low-rated orders can be cross-checked with delay trends to find improvement areas.")

# ----------------------------------------
# VISUAL 6: Correlation Heatmap (Numerical)
# ----------------------------------------
plt.figure(figsize=(8,6))
sns.heatmap(df.select_dtypes(include=['float64','int64']).corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Key Numeric Features')
plt.tight_layout()
plt.savefig('data/eda_correlation.png')
plt.show()

print("💡 Insight 6: The heatmap highlights relationships between cost, delay days, distance, and emissions. "
      "Strong correlations guide feature selection for predictive modeling.")

# ----------------------------------------
# BONUS: Interactive Plotly Chart (optional)
# ----------------------------------------
if 'Distance_Km' in df.columns:
    fig = px.scatter(df, x='Distance_Km', y='Delay_Days', color='Carrier',
                     title='Interactive View: Distance vs Delay by Carrier',
                     hover_data=['Priority','Total_Cost'])
    fig.write_html('data/interactive_distance_delay.html')
    print("🌐 Interactive Plotly chart saved: data/interactive_distance_delay.html")

print("\n✅ EDA completed successfully! All charts saved in data/ folder.")
