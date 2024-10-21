import cudf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np

# Read the CSV file
df = cudf.read_csv('smotif_database2.csv')

# Create Smotif_Type column using string concatenation
df['Smotif_Type'] = df['SSE1_Type'] + df['SSE2_Type']

# Calculate total Smotif size
df['Total_Size'] = df['SSE1_Length'] + df['SSE2_Length'] + df['Loop_Length']

# Convert to pandas DataFrame for plotting
pdf = df.to_pandas()

# Create subplots
fig = make_subplots(rows=3, cols=2, subplot_titles=(
    'Distribution of Smotif Types', 'Distribution of Total Smotif Sizes',
    'Distribution of Loop Lengths', 'SSE1 Length vs SSE2 Length',
    'Average Smotif Size by Type', 'Correlation Heatmap'
))

# 1. Distribution of Smotif Types
smotif_type_counts = pdf['Smotif_Type'].value_counts()
fig.add_trace(go.Bar(x=smotif_type_counts.index, y=smotif_type_counts.values,
                     marker_color='indianred'), row=1, col=1)
fig.update_xaxes(title_text="Smotif Type", row=1, col=1)
fig.update_yaxes(title_text="Count", row=1, col=1)

# 2. Distribution of Total Smotif Sizes
fig.add_trace(go.Histogram(x=pdf['Total_Size'], nbinsx=50,
                           marker_color='lightseagreen'), row=1, col=2)
fig.update_xaxes(title_text="Total Size (residues)", row=1, col=2)
fig.update_yaxes(title_text="Count", row=1, col=2)

# 3. Distribution of Loop Lengths
fig.add_trace(go.Histogram(x=pdf['Loop_Length'], nbinsx=50,
                           marker_color='mediumorchid'), row=2, col=1)
fig.update_xaxes(title_text="Loop Length (residues)", row=2, col=1)
fig.update_yaxes(title_text="Count", row=2, col=1)

# 4. SSE1 Length vs SSE2 Length
scatter = go.Scatter(
    x=pdf['SSE1_Length'],
    y=pdf['SSE2_Length'],
    mode='markers',
    marker=dict(
        size=5,
        color=pdf['Smotif_Type'].astype('category').cat.codes,  # Use categorical codes for colors
        colorscale='Viridis',
        showscale=True,
        colorbar=dict(title='Smotif Type')
    ),
    text=pdf['Smotif_Type'],
    hoverinfo='text'
)
fig.add_trace(scatter, row=2, col=2)
fig.update_xaxes(title_text="SSE1 Length (residues)", row=2, col=2)
fig.update_yaxes(title_text="SSE2 Length (residues)", row=2, col=2)

# 5. Average Smotif Size by Type
avg_size_by_type = pdf.groupby('Smotif_Type')['Total_Size'].mean().sort_values(ascending=False)
fig.add_trace(go.Bar(x=avg_size_by_type.index,
                     y=avg_size_by_type.values,
                     marker_color='goldenrod'), row=3, col=1)
fig.update_xaxes(title_text="Smotif Type", row=3, col=1)
fig.update_yaxes(title_text="Average Size (residues)", row=3, col=1)

# 6. Correlation Heatmap
corr_matrix = pdf[['SSE1_Length', 'SSE2_Length', 'Loop_Length', 'Total_Size']].corr()
fig.add_trace(go.Heatmap(z=corr_matrix.values,
                         x=corr_matrix.columns,
                         y=corr_matrix.columns,
                         colorscale='Viridis',
                         zmin=-1, zmax=1,
                         text=corr_matrix.values,
                         texttemplate="%{text:.2f}",
                         textfont={"size":10},
                         hoverongaps=False), row=3, col=2)

# Update layout
fig.update_layout(height=1800, width=1200, title_text="Smotif Statistics",
                  showlegend=False)

# Save the plot as an interactive HTML file
fig.write_html("smotif_statistics_interactive.html")

# Also save as a static image
fig.write_image("smotif_statistics2.png")

# Print some additional statistics
print(f"Total number of Smotifs: {len(df)}")
print("\nSmotif Type Distribution:")
print(pdf['Smotif_Type'].value_counts(normalize=True))
print("\nAverage Smotif Size:", pdf['Total_Size'].mean())
print("Median Smotif Size:", pdf['Total_Size'].median())
print("\nAverage Loop Length:", pdf['Loop_Length'].mean())
print("Median Loop Length:", pdf['Loop_Length'].median())

# Top 10 domains with the most Smotifs
top_domains = pdf['Domain'].value_counts().head(10)
print("\nTop 10 domains with the most Smotifs:")
print(top_domains)