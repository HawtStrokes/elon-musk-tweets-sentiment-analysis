import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend to avoid Tkinter
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_sentiment_summary(file_path):
    """
    Visualizes sentiment summary data from the CSV as bar charts for sentiment count and sentiment score distribution.

    Parameters:
    file_path (str): Path to the sentiment summary CSV file.
    """
    try:
        # Read the CSV file
        data = pd.read_csv(file_path)

        # Ensure that the CSV has the expected columns
        if 'Sentiment' not in data.columns or 'Count' not in data.columns or 'VADER_Compound_Score' not in data.columns:
            raise ValueError("CSV file is missing expected columns ('Sentiment', 'Count', 'VADER_Compound_Score')")

        # Set the plot style for better appearance
        sns.set(style="whitegrid")

        # Create a figure with subplots
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Bar plot for sentiment counts with corrected 'hue' parameter
        sns.barplot(x='Sentiment', y='Count', data=data, hue='Sentiment', palette="viridis", ax=axes[0], legend=False)
        axes[0].set_title("Sentiment Counts")
        axes[0].set_xlabel("Sentiment")
        axes[0].set_ylabel("Count")

        # Bar plot for VADER Compound Scores by Sentiment with corrected 'hue' parameter
        sns.barplot(x='Sentiment', y='VADER_Compound_Score', data=data, hue='Sentiment', palette="coolwarm", ax=axes[1], legend=False)
        axes[1].set_title("VADER Compound Scores by Sentiment")
        axes[1].set_xlabel("Sentiment")
        axes[1].set_ylabel("VADER Compound Score")

        # Adjust layout to prevent overlapping
        plt.tight_layout()

        # Save the plot as a PNG file
        plt.savefig('./results/sentiment_visualization.png')

        print("Plot saved as 'sentiment_visualization.png'.")

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except ValueError as e:
        print(f"Error: {e}")

# Example of usage: Read and visualize sentiment summary
if __name__ == "__main__":
    # Define the file path
    file_path = './data/sentiment_summary.csv'
    
    # Visualize the sentiment summary data
    visualize_sentiment_summary(file_path)
