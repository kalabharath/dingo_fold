import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np

# Set style parameters
plt.style.use('default')
sns.set_theme(style="whitegrid")

# Read the data
df = pd.read_csv('extended_smotif_db2.csv')

def create_plots():
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(20, 25))
    
    # 1. Distribution of SSE types combinations
    ax1 = plt.subplot(4, 2, 1)
    sns.countplot(data=df, x='smotif_type', ax=ax1)
    ax1.set_title('Distribution of Secondary Structure Element Combinations')
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. Distribution of sequence lengths with gaussian fit
    ax2 = plt.subplot(4, 2, 2)
    all_lengths = np.concatenate([df['sse1_length'], df['sse2_length']])
    sns.histplot(all_lengths, kde=True, ax=ax2)
    # Fit gaussian
    mu, sigma = stats.norm.fit(all_lengths)
    x = np.linspace(min(all_lengths), max(all_lengths), 100)
    gaussian = stats.norm.pdf(x, mu, sigma)
    ax2.plot(x, gaussian * len(all_lengths) * (max(all_lengths) - min(all_lengths)) / 100, 
            'r-', lw=2, label=f'Gaussian fit\nμ={mu:.1f}, σ={sigma:.1f}')
    ax2.set_title('Distribution of SSE Lengths')
    ax2.set_xlabel('Length')
    ax2.legend()
    
    # 3. Loop length distribution
    ax3 = plt.subplot(4, 2, 3)
    sns.histplot(data=df, x='loop_length', kde=True, ax=ax3)
    ax3.set_title('Distribution of Loop Lengths')
    ax3.set_xlabel('Loop Length')
    
    # 4. Center of Mass (COM) distance distribution
    ax4 = plt.subplot(4, 2, 4)
    sns.histplot(data=df, x='com_distance', kde=True, ax=ax4)
    ax4.set_title('Distribution of Center of Mass Distances')
    ax4.set_xlabel('COM Distance')
    
    # 5. Total contacts distribution
    ax5 = plt.subplot(4, 2, 5)
    sns.histplot(data=df, x='total_contacts', kde=True, ax=ax5)
    ax5.set_title('Distribution of Total Contacts')
    ax5.set_xlabel('Total Contacts')
    
    # 6. Scatter plot: Loop length vs COM distance
    ax6 = plt.subplot(4, 2, 6)
    sns.scatterplot(data=df, x='loop_length', y='com_distance', hue='smotif_type', alpha=0.6, ax=ax6)
    ax6.set_title('Loop Length vs COM Distance')
    ax6.set_xlabel('Loop Length')
    ax6.set_ylabel('COM Distance')
    
    # 7. Box plot of COM distances by SSE type
    ax7 = plt.subplot(4, 2, 7)
    sns.boxplot(data=df, x='smotif_type', y='com_distance', ax=ax7)
    ax7.set_title('COM Distance Distribution by SSE Type')
    ax7.tick_params(axis='x', rotation=45)
    
    # 8. Correlation between various numerical features
    ax8 = plt.subplot(4, 2, 8)
    numerical_cols = ['sse1_length', 'sse2_length', 'loop_length', 'com_distance', 'total_contacts']
    correlation_matrix = df[numerical_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax8)
    ax8.set_title('Correlation Matrix of Numerical Features')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('structural_motif_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Print additional statistics
    print("\nSummary Statistics:")
    print(df[numerical_cols].describe())
    
    print("\nSSE Type Distribution (%):")
    sse_type_dist = df['smotif_type'].value_counts(normalize=True) * 100
    print(sse_type_dist)

if __name__ == "__main__":
    create_plots()