import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import download

# Download VADER lexicon if it's not already installed
download('vader_lexicon')

# Load the dataset
file_path = './data/cleandata.csv'
data = pd.read_csv(file_path)

# Initialize VADER sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Apply VADER sentiment analysis to each tweet in the Cleaned_Tweets column
def get_vader_sentiment(text):
    sentiment_scores = sid.polarity_scores(text)
    return sentiment_scores['compound']

# Calculate sentiment score and classify as Positive, Negative, or Neutral
data['VADER_Compound_Score'] = data['Cleaned_Tweets'].apply(get_vader_sentiment)
data['Sentiment'] = data['VADER_Compound_Score'].apply(lambda score: 'Positive' if score > 0.05 
                                                        else 'Negative' if score < -0.05 else 'Neutral')

# Save the results to a CSV file for use in RapidMiner
output_path = './data/vader_sentiment_results.csv'
data.to_csv(output_path, index=False)

print(f"Sentiment analysis complete. Results saved to {output_path}")
