import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV file
df = pd.read_csv('/home/kalabharath/projects/dingo_fold/extended_smotif_training_data/extended_smotif_db_with_contacts.csv')

# delete the rows where total_contacts is 0
df = df[df['total_contacts'] > 0]
# Set up the plotting style
plt.style.use('ggplot')
sns.set_palette("deep")

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16))

# Violin plot
sns.violinplot(x='smotif_type', y='total_contacts', data=df, ax=ax1)
ax1.set_title('Distribution of Total Contacts by Smotif Type (Violin Plot)', fontsize=16)
ax1.set_xlabel('Smotif Type', fontsize=12)
ax1.set_ylabel('Total Contacts', fontsize=12)
ax1.tick_params(axis='both', which='major', labelsize=10)

# Box plot
sns.boxplot(x='smotif_type', y='total_contacts', data=df, ax=ax2)
ax2.set_title('Distribution of Total Contacts by Smotif Type (Box Plot)', fontsize=16)
ax2.set_xlabel('Smotif Type', fontsize=12)
ax2.set_ylabel('Total Contacts', fontsize=12)
ax2.tick_params(axis='both', which='major', labelsize=10)

# Add individual data points to the box plot
sns.stripplot(x='smotif_type', y='total_contacts', data=df, color="0.4", size=2, jitter=True, ax=ax2)

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig('smotif_contacts_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# Calculate and print summary statistics
summary_stats = df.groupby('smotif_type')['total_contacts'].describe()
print("Summary Statistics:")
print(summary_stats)

# Calculate and print the correlation between total_contacts and other numerical columns
numerical_columns = ['sse1_length', 'sse2_length', 'D', 'delta', 'theta', 'rho', 'com_distance']
correlations = df[['total_contacts'] + numerical_columns].corr()['total_contacts'].drop('total_contacts')
print("\nCorrelations with total_contacts:")
print(correlations)