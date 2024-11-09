import pandas as pd

# Load the sentiment analysis results
file_path = './data/vader_sentiment_results.csv'
data = pd.read_csv(file_path)

# Optionally, inspect the dataset
print(data.head())

# Generate additional data for visualization
# You could, for example, create some aggregation statistics or categorical grouping

# Count of positive, negative, and neutral sentiments
sentiment_counts = data['Sentiment'].value_counts().reset_index()
sentiment_counts.columns = ['Sentiment', 'Count']

# Calculate the average sentiment score for each category
average_scores = data.groupby('Sentiment')['VADER_Compound_Score'].mean().reset_index()

# You can merge these dataframes to create a more informative CSV
sentiment_summary = pd.merge(sentiment_counts, average_scores, on='Sentiment')

# Save this summary to a new CSV for Visualization
output_path = './data/sentiment_summary.csv'
sentiment_summary.to_csv(output_path, index=False)

print(f"Sentiment summary saved to {output_path}")
