import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Read the CSV file with explicit dtypes and error handling
dtype_dict = {
    'Domain': str,
    'SSE1_Type': str,
    'SSE1_Start': float,
    'SSE1_End': float,
    'SSE1_Length': float,
    'SSE2_Type': str,
    'SSE2_Start': float,
    'SSE2_End': float,
    'SSE2_Length': float,
    'Loop_Length': float
}

df = pd.read_csv('smotif_database2.csv')
# Clean the data
df = df.dropna()  # Remove rows with any NaN values
df = df[df['SSE1_Type'].isin(['H', 'E']) & df['SSE2_Type'].isin(['H', 'E'])]  # Keep only valid SSE types

# Convert numeric columns to integers
numeric_columns = ['SSE1_Start', 'SSE1_End', 'SSE1_Length', 'SSE2_Start', 'SSE2_End', 'SSE2_Length', 'Loop_Length']
for col in numeric_columns:
    df[col] = df[col].astype(int)

# Create a function to generate Smotif type
def get_smotif_type(row):
    return f"{row['SSE1_Type']}{row['SSE2_Type']}"

df['Smotif_Type'] = df.apply(get_smotif_type, axis=1)

# Calculate total Smotif size
df['Total_Size'] = df['SSE1_Length'] + df['SSE2_Length'] + df['Loop_Length']

# Set up the plot
plt.figure(figsize=(20, 20))
plt.suptitle('Smotif Statistics', fontsize=24)

# 1. Distribution of Smotif Types
plt.subplot(3, 2, 1)
smotif_type_counts = df['Smotif_Type'].value_counts()
sns.barplot(x=smotif_type_counts.index, y=smotif_type_counts.values)
plt.title('Distribution of Smotif Types')
plt.ylabel('Count')
plt.xticks(rotation=45)

# 2. Distribution of Total Smotif Sizes
plt.subplot(3, 2, 2)
sns.histplot(df['Total_Size'], bins=50, kde=True)
plt.title('Distribution of Total Smotif Sizes')
plt.xlabel('Total Size (residues)')
plt.ylabel('Count')

# 3. Distribution of Loop Lengths
plt.subplot(3, 2, 3)
sns.histplot(df['Loop_Length'], bins=50, kde=True)
plt.title('Distribution of Loop Lengths')
plt.xlabel('Loop Length (residues)')
plt.ylabel('Count')

# 4. SSE1 Length vs SSE2 Length
plt.subplot(3, 2, 4)
sns.scatterplot(data=df, x='SSE1_Length', y='SSE2_Length', hue='Smotif_Type', alpha=0.5)
plt.title('SSE1 Length vs SSE2 Length')
plt.xlabel('SSE1 Length (residues)')
plt.ylabel('SSE2 Length (residues)')

# 5. Average Smotif Size by Type
plt.subplot(3, 2, 5)
avg_size_by_type = df.groupby('Smotif_Type')['Total_Size'].mean().sort_values(ascending=False)
sns.barplot(x=avg_size_by_type.index, y=avg_size_by_type.values)
plt.title('Average Smotif Size by Type')
plt.ylabel('Average Size (residues)')
plt.xticks(rotation=45)

# 6. Correlation Heatmap
plt.subplot(3, 2, 6)
corr_matrix = df[['SSE1_Length', 'SSE2_Length', 'Loop_Length', 'Total_Size']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')

# Adjust layout and save the plot
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('smotif_statistics.png', dpi=300, bbox_inches='tight')
plt.close()

# Print some additional statistics
print(f"Total number of Smotifs: {len(df)}")
print("\nSmotif Type Distribution:")
print(df['Smotif_Type'].value_counts(normalize=True))
print("\nAverage Smotif Size:", df['Total_Size'].mean())
print("Median Smotif Size:", df['Total_Size'].median())
print("\nAverage Loop Length:", df['Loop_Length'].mean())
print("Median Loop Length:", df['Loop_Length'].median())

# Top 10 domains with the most Smotifs
top_domains = df['Domain'].value_counts().head(10)
print("\nTop 10 domains with the most Smotifs:")
print(top_domains)