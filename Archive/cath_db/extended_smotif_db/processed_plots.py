import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the data
standard_df = pd.read_csv('../smotif_db/processed_smotif_database3.csv')
extended_df = pd.read_csv('processed_extended_smotif_database.csv')

# Set up the plotting style
plt.style.use('default')
sns.set_palette("Set2")

def plot_comparison(standard_data, extended_data, title, filename, xlabel=None, ylabel=None):
    plt.figure(figsize=(12, 6))
    sns.histplot(standard_data, kde=True, stat="density", label="Standard Smotifs", alpha=0.5)
    sns.histplot(extended_data, kde=True, stat="density", label="Extended Smotifs", alpha=0.5)
    plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# 1. Distribution of Smotif types
plt.figure(figsize=(12, 6))
standard_counts = standard_df['smotif_type'].value_counts()
extended_counts = extended_df['smotif_type'].value_counts()
bar_width = 0.35
index = np.arange(len(standard_counts.index))
plt.bar(index, standard_counts.values, bar_width, label='Standard Smotifs')
plt.bar(index + bar_width, extended_counts.values, bar_width, label='Extended Smotifs')
plt.title('Distribution of Smotif Types')
plt.xlabel('Smotif Type')
plt.ylabel('Count')
plt.xticks(index + bar_width/2, standard_counts.index)
plt.legend()
plt.tight_layout()
plt.savefig('smotif_type_distribution_comparison.png')
plt.close()

# 2. Distribution of geometric parameters
for param in ['D', 'delta', 'theta', 'rho']:
    plot_comparison(standard_df[param], extended_df[param], 
                    f'Distribution of {param}',
                    f'geometric_parameter_{param}_comparison.png',
                    xlabel=param, ylabel='Density')

# Distribution of com_distance for extended Smotifs only
plt.figure(figsize=(12, 6))
sns.histplot(extended_df['com_distance'], kde=True, stat="density")
plt.title('Distribution of Center of Mass Distance (Extended Smotifs)')
plt.xlabel('Center of Mass Distance')
plt.ylabel('Density')
plt.tight_layout()
plt.savefig('com_distance_distribution_extended.png')
plt.close()

# 3. Correlation heatmap of geometric parameters
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
sns.heatmap(standard_df[['D', 'delta', 'theta', 'rho']].corr(), annot=True, cmap='coolwarm', ax=ax1)
ax1.set_title('Correlation Heatmap of Geometric Parameters (Standard Smotifs)')
sns.heatmap(extended_df[['D', 'delta', 'theta', 'rho', 'com_distance']].corr(), annot=True, cmap='coolwarm', ax=ax2)
ax2.set_title('Correlation Heatmap of Geometric Parameters (Extended Smotifs)')
plt.tight_layout()
plt.savefig('geometric_parameters_correlation_comparison.png')
plt.close()

# 4. Scatter plot of D vs delta for each Smotif type (as com_distance is not available for standard Smotifs)
plt.figure(figsize=(12, 8))
for smotif_type in standard_df['smotif_type'].unique():
    standard_subset = standard_df[standard_df['smotif_type'] == smotif_type]
    extended_subset = extended_df[extended_df['smotif_type'] == smotif_type]
    plt.scatter(standard_subset['D'], standard_subset['delta'], alpha=0.5, label=f'Standard {smotif_type}')
    plt.scatter(extended_subset['D'], extended_subset['delta'], alpha=0.5, label=f'Extended {smotif_type}', marker='x')
plt.title('Distance vs Hoist Angle for Different Smotif Types')
plt.xlabel('Distance (D)')
plt.ylabel('Hoist Angle (δ)')
plt.legend()
plt.tight_layout()
plt.savefig('distance_vs_hoist_angle_comparison.png')
plt.close()

# 5. Box plot of geometric parameters for each Smotif type
for param in ['D', 'delta', 'theta', 'rho']:
    plt.figure(figsize=(12, 6))
    standard_data = standard_df[['smotif_type', param]].copy()
    standard_data['Type'] = 'Standard'
    extended_data = extended_df[['smotif_type', param]].copy()
    extended_data['Type'] = 'Extended'
    combined_data = pd.concat([standard_data, extended_data])
    sns.boxplot(x='smotif_type', y=param, hue='Type', data=combined_data)
    plt.title(f'Distribution of {param} by Smotif Type')
    plt.tight_layout()
    plt.savefig(f'geometric_parameter_{param}_by_smotif_type_comparison.png')
    plt.close()

# Box plot for com_distance (Extended Smotifs only)
plt.figure(figsize=(12, 6))
sns.boxplot(x='smotif_type', y='com_distance', data=extended_df)
plt.title('Distribution of Center of Mass Distance by Smotif Type (Extended Smotifs)')
plt.tight_layout()
plt.savefig('com_distance_by_smotif_type_extended.png')
plt.close()

# Print some statistics
print("Smotif Statistics:")
print(f"Total number of Standard Smotifs: {len(standard_df)}")
print(f"Total number of Extended Smotifs: {len(extended_df)}")
print("\nSmotif Type Distribution:")
print("Standard Smotifs:")
print(standard_df['smotif_type'].value_counts())
print("\nExtended Smotifs:")
print(extended_df['smotif_type'].value_counts())
print("\nAverage Geometric Parameters:")
print("Standard Smotifs:")
print(standard_df[['D', 'delta', 'theta', 'rho']].mean())
print("\nExtended Smotifs:")
print(extended_df[['D', 'delta', 'theta', 'rho', 'com_distance']].mean())
print("\nMedian Geometric Parameters:")
print("Standard Smotifs:")
print(standard_df[['D', 'delta', 'theta', 'rho']].median())
print("\nExtended Smotifs:")
print(extended_df[['D', 'delta', 'theta', 'rho', 'com_distance']].median())
print("\nUnique Smotif IDs:")
print(f"Standard Smotifs: {standard_df['smotif_id'].nunique()}")
print(f"Extended Smotifs: {extended_df['smotif_id'].nunique()}")