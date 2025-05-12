from query_data_predictor.sdss_json_importer import JsonDataImporter
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import os
import datetime

DATA_PATH = Path(__file__).parent.parent / "data" / "sdss_joined_sample.json"
REPORT_DIR = Path(__file__).parent.parent / "reports"
REPORT_IMAGES_DIR = REPORT_DIR / "images"

# Create report directories if they don't exist
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(REPORT_IMAGES_DIR, exist_ok=True)

# Generate a timestamp for unique filenames
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
report_filename = f"query_statistics_report_{timestamp}.html"

def save_figure(fig, filename):
    """Save a matplotlib figure to the report images directory"""
    filepath = REPORT_IMAGES_DIR / filename
    fig.savefig(filepath, bbox_inches='tight')
    return filepath.relative_to(REPORT_DIR)

def generate_html_report(stats_summary, additional_stats, image_paths):
    """Generate an HTML report with statistics and images"""
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Query Statistics Report - {timestamp}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
            h1, h2 {{ color: #333; }}
            .stats-container {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
            pre {{ background-color: #f0f0f0; padding: 10px; border-radius: 5px; overflow-x: auto; }}
            img {{ max-width: 100%; margin: 20px 0; border: 1px solid #ddd; }}
        </style>
    </head>
    <body>
        <h1>Query Statistics Report</h1>
        <p>Generated on: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        
        <h2>Basic Statistics</h2>
        <div class="stats-container">
            <pre>{stats_summary}</pre>
        </div>
        
        <h2>Additional Metrics</h2>
        <div class="stats-container">
            <pre>{additional_stats}</pre>
        </div>
        
        <h2>Visualizations</h2>
    """
    
    for title, path in image_paths:
        html += f"""
        <div>
            <h3>{title}</h3>
            <img src="{path}" alt="{title}">
        </div>
        """
    
    html += """
    </body>
    </html>
    """
    return html

# Main code
loader = JsonDataImporter(DATA_PATH)
sessions = loader.get_sessions()

stats = {}
for session in sessions:
    queries = loader.get_queries_for_session(session)
    stats[session] = len(queries)
df = pd.DataFrame.from_dict(stats, orient='index', columns=['query_count'])

# Generate statistics
stats_summary = df['query_count'].describe()
print(stats_summary)

# Additional metrics
additional_stats = f"""
Median queries per session: {df['query_count'].median()}
Mode queries per session: {df['query_count'].mode().values[0]}
Range of queries: {df['query_count'].min()} to {df['query_count'].max()}
Total number of queries across all sessions: {df['query_count'].sum()}
"""
print(additional_stats)

# Keep track of generated images
image_paths = []

# Box plot to show distribution and outliers
plt.figure(figsize=(10, 6))
sns.boxplot(x=df['query_count'])
plt.title('Box Plot of Queries per Session')
plt.xlabel('Number of Queries')
boxplot_path = save_figure(plt.gcf(), f"boxplot_{timestamp}.png")
image_paths.append(("Box Plot of Queries per Session", boxplot_path))
plt.close()

# Create categories for query counts
def categorize_count(count):
    if count <= 5:
        return '1-5'
    elif count <= 10:
        return '6-10'
    elif count <= 20:
        return '11-20'
    elif count <= 50:
        return '21-50'
    else:
        return '50+'

df['count_category'] = df['query_count'].apply(categorize_count)

# Count plot of sessions by category
plt.figure(figsize=(12, 6))
ax = sns.countplot(data=df, x='count_category', order=['1-5', '6-10', '11-20', '21-50', '50+'])
plt.title('Number of Sessions by Query Count Category')
plt.xlabel('Query Count Category')
plt.ylabel('Number of Sessions')

# Add count annotations on top of the bars
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width()/2., height + 0.5, 
            '{:d}'.format(int(height)), 
            ha="center", va="bottom")

countplot_path = save_figure(plt.gcf(), f"countplot_{timestamp}.png")
image_paths.append(("Number of Sessions by Query Count Category", countplot_path))
plt.close()

# Top 10 sessions with highest query counts
top_sessions = df.sort_values('query_count', ascending=False).head(10)

plt.figure(figsize=(12, 6))
sns.barplot(data=top_sessions.reset_index(), x='index', y='query_count')
plt.title('Top 10 Sessions by Query Count')
plt.xlabel('Session ID')
plt.ylabel('Number of Queries')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
barplot_path = save_figure(plt.gcf(), f"barplot_{timestamp}.png")
image_paths.append(("Top 10 Sessions by Query Count", barplot_path))
plt.close()

# Generate and save the HTML report
html_report = generate_html_report(stats_summary, additional_stats, image_paths)
report_path = REPORT_DIR / report_filename
with open(report_path, 'w') as f:
    f.write(html_report)

print(f"\nReport generated successfully at: {report_path}")