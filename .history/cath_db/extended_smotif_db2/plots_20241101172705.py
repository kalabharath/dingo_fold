import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np

# Set the style for better visualization
plt.style.use('seaborn')
sns.set_palette("husl")

# Read the data
df = pd.read_csv('extended_smotif_db_with_contacts.csv')

# Create a figure with multiple subplots
plt.figure(figsize=(20, 25))

# 1. Distribution of SSE types combinations
plt.subplot(4, 2, 1)
sns.countplot(data=df, x='smotif_type')
plt.title('Distribution of Secondary Structure Element Combinations')
plt.xticks(rotation=45)

# 2. Distribution of sequence lengths with gaussian fit
plt.subplot(4, 2, 2)
# Combine both SSE lengths
all_lengths = np.concatenate([df['sse1_length'], df['sse2_length']])
sns.histplot(all_lengths, kde=True)
# Fit gaussian
mu, sigma = stats.norm.fit(all_lengths)
x = np.linspace(min(all_lengths), max(all_lengths), 100)
gaussian = stats.norm.pdf(x, mu, sigma)
plt.plot(x, gaussian * len(all_lengths) * (max(all_lengths) - min(all_lengths)) / 100, 
         'r-', lw=2, label=f'Gaussian fit\nμ={mu:.1f}, σ={sigma:.1f}')
plt.title('Distribution of SSE Lengths')
plt.xlabel('Length')
plt.legend()

# 3. Loop length distribution
plt.subplot(4, 2, 3)
sns.histplot(data=df, x='loop_length', kde=True)
plt.title('Distribution of Loop Lengths')
plt.xlabel('Loop Length')

# 4. Center of Mass (COM) distance distribution
plt.subplot(4, 2, 4)
sns.histplot(data=df, x='com_distance', kde=True)
plt.title('Distribution of Center of Mass Distances')
plt.xlabel('COM Distance')

# 5. Total contacts distribution
plt.subplot(4, 2, 5)
sns.histplot(data=df, x='total_contacts', kde=True)
plt.title('Distribution of Total Contacts')
plt.xlabel('Total Contacts')

# 6. Scatter plot: Loop length vs COM distance
plt.subplot(4, 2, 6)
sns.scatterplot(data=df, x='loop_length', y='com_distance', hue='smotif_type', alpha=0.6)
plt.title('Loop Length vs COM Distance')
plt.xlabel('Loop Length')
plt.ylabel('COM Distance')

# 7. Box plot of COM distances by SSE type
plt.subplot(4, 2, 7)
sns.boxplot(data=df, x='smotif_type', y='com_distance')
plt.title('COM Distance Distribution by SSE Type')
plt.xticks(rotation=45)

# 8. Correlation between various numerical features
plt.subplot(4, 2, 8)
numerical_cols = ['sse1_length', 'sse2_length', 'loop_length', 'com_distance', 'total_contacts']
correlation_matrix = df[numerical_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix of Numerical Features')

# Adjust layout
plt.tight_layout()

# Additional statistics
print("\nSummary Statistics:")
print(df[numerical_cols].describe())

# Calculate percentages of SSE types
sse_type_dist = df['smotif_type'].value_counts(normalize=True) * 100
print("\nSSE Type Distribution (%):")
print(sse_type_dist)

# Save the plot
plt.savefig('structural_motif_analysis.png', dpi=300, bbox_inches='tight')
plt.close()