import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the data
df = pd.read_csv('./processed_extended_smotif_database.csv')

# Set up the plotting style
plt.style.use('default')  # Use the default matplotlib style
sns.set_palette("Set2")

# Rest of the code remains the same...

# 1. Distribution of Smotif types
plt.figure(figsize=(10, 6))
df['smotif_type'].value_counts().plot(kind='bar')
plt.title('Distribution of Smotif Types')
plt.xlabel('Smotif Type')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('smotif_type_distribution.png')
plt.close()

# 2. Distribution of geometric parameters
fig, axs = plt.subplots(2, 2, figsize=(15, 15))
axs[0, 0].hist(df['D'], bins=50)
axs[0, 0].set_title('Distribution of Distance (D)')
axs[0, 1].hist(df['delta'], bins=50)
axs[0, 1].set_title('Distribution of Hoist Angle (δ)')
axs[1, 0].hist(df['theta'], bins=50)
axs[1, 0].set_title('Distribution of Packing Angle (θ)')
axs[1, 1].hist(df['rho'], bins=50)
axs[1, 1].set_title('Distribution of Meridian Angle (ρ)')
plt.tight_layout()
plt.savefig('geometric_parameters_distribution.png')
plt.close()

# 3. Loop length distribution
plt.figure(figsize=(10, 6))
df['loop_length'].value_counts().sort_index().plot(kind='bar')
plt.title('Distribution of Loop Lengths')
plt.xlabel('Loop Length')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('loop_length_distribution.png')
plt.close()

# 4. Correlation heatmap of geometric parameters
plt.figure(figsize=(10, 8))
sns.heatmap(df[['D', 'delta', 'theta', 'rho']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap of Geometric Parameters')
plt.tight_layout()
plt.savefig('geometric_parameters_correlation.png')
plt.close()

# 5. Scatter plot of D vs delta for each Smotif type
plt.figure(figsize=(12, 8))
for smotif_type in df['smotif_type'].unique():
    subset = df[df['smotif_type'] == smotif_type]
    plt.scatter(subset['D'], subset['delta'], alpha=0.5, label=smotif_type)
plt.title('Distance vs Hoist Angle for Different Smotif Types')
plt.xlabel('Distance (D)')
plt.ylabel('Hoist Angle (δ)')
plt.legend()
plt.tight_layout()
plt.savefig('distance_vs_hoist_angle.png')
plt.close()

# 6. Box plot of geometric parameters for each Smotif type
fig, axs = plt.subplots(2, 2, figsize=(15, 15))
sns.boxplot(x='smotif_type', y='D', data=df, ax=axs[0, 0])
axs[0, 0].set_title('Distribution of Distance (D) by Smotif Type')
sns.boxplot(x='smotif_type', y='delta', data=df, ax=axs[0, 1])
axs[0, 1].set_title('Distribution of Hoist Angle (δ) by Smotif Type')
sns.boxplot(x='smotif_type', y='theta', data=df, ax=axs[1, 0])
axs[1, 0].set_title('Distribution of Packing Angle (θ) by Smotif Type')
sns.boxplot(x='smotif_type', y='rho', data=df, ax=axs[1, 1])
axs[1, 1].set_title('Distribution of Meridian Angle (ρ) by Smotif Type')
plt.tight_layout()
plt.savefig('geometric_parameters_by_smotif_type.png')
plt.close()

# Print some statistics
print("Smotif Statistics:")
print(f"Total number of Smotifs: {len(df)}")
print("\nSmotif Type Distribution:")
print(df['smotif_type'].value_counts())
print("\nAverage Geometric Parameters:")
print(df[['D', 'delta', 'theta', 'rho']].mean())
print("\nMedian Geometric Parameters:")
print(df[['D', 'delta', 'theta', 'rho']].median())
print("\nAverage Loop Length:")
print(df['loop_length'].mean())
print("\nUnique Smotif IDs:")
print(df['smotif_id'].nunique())