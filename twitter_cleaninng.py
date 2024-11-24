import pandas as pd
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os

# File paths
input_file = r"C:\Users\saric\OneDrive\Desktop\Youtube_Comments\Twitter_tweet_data.csv"
output_cleaned_file = r"C:\Users\saric\OneDrive\Desktop\tokenized\last_cleaned_twitter_data.csv"
output_tokenized_file = r"C:\Users\saric\OneDrive\Desktop\tokenized\last_tokenized_cleaned_twitter_data.csv"
output_sentiment_file = r"C:\Users\saric\OneDrive\Desktop\sentiment_results\sentiment_last_cleaned_twitter_data.csv"

# List of stock tickers in scope
stock_tickers_in_scope = ["AMZN", "AAPL", "META", "MSFT", "TSLA", "NIO", "PG", "TSM"]

# Function to clean text
def clean_text(text):
    """
    Clean text by removing URLs, special characters, and converting to lowercase.
    """
    if isinstance(text, str):
        text = re.sub(r'http\S+', '', text)  # Remove URLs
        text = re.sub(r'[^A-Za-z0-9\s]', '', text)  # Remove special characters
        text = text.lower()  # Convert to lowercase
    return text

# Function to tokenize text
def tokenize_text(text):
    """
    Tokenize text by splitting into words.
    """
    if isinstance(text, str):
        return " ".join(re.findall(r'\b\w+\b', text))  # Tokenize and join words
    return text

# Perform sentiment analysis
def perform_sentiment_analysis(df, text_column):
    """
    Perform sentiment analysis using VADER and add sentiment scores to the DataFrame.
    """
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = df[text_column].apply(
        lambda x: analyzer.polarity_scores(x)['compound'] if isinstance(x, str) else 0
    )
    df['Sentiment'] = sentiment_scores
    return df

# Preprocess the Twitter dataset
def preprocess_twitter_data(input_file, cleaned_file, tokenized_file, sentiment_file):
    print(f"Processing Twitter dataset: {input_file}")
    
    # Read dataset
    df = pd.read_csv(input_file)
    
    # Ensure required columns exist
    if 'Tweet' not in df.columns or 'Date' not in df.columns or 'Stock Name' not in df.columns:
        raise ValueError("Columns 'Tweet', 'Date', and 'Stock Name' are required in the dataset.")
    
    # Filter by stock tickers in scope
    df = df[df['Stock Name'].isin(stock_tickers_in_scope)]
    
    # Check if there are no relevant tweets for the selected companies
    if df.empty:
        print("No relevant tweets found for the specified stock tickers.")
        return
    
    # Clean text
    df['Tweet'] = df['Tweet'].apply(clean_text)
    
    # Drop rows with missing or empty tweets
    df.dropna(subset=['Tweet'], inplace=True)
    df = df[df['Tweet'].str.strip() != ""]
    
    # Standardize dates
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.strftime('%Y-%m-%d')
    df.dropna(subset=['Date'], inplace=True)  # Remove rows with invalid dates
    
    # Save cleaned dataset
    df.to_csv(cleaned_file, index=False, encoding='utf-8')
    print(f"Cleaned file saved: {cleaned_file}")
    
    # Tokenize text
    df['Tweet'] = df['Tweet'].apply(tokenize_text)
    
    # Save tokenized dataset
    df.to_csv(tokenized_file, index=False, encoding='utf-8')
    print(f"Tokenized file saved: {tokenized_file}")
    
    # Perform sentiment analysis
    df = perform_sentiment_analysis(df, text_column='Tweet')
    
    # Save dataset with sentiment scores
    df.to_csv(sentiment_file, index=False, encoding='utf-8')
    print(f"Sentiment analysis completed. Sentiment file saved: {sentiment_file}")

# Run the preprocessing and sentiment analysis
preprocess_twitter_data(input_file, output_cleaned_file, output_tokenized_file, output_sentiment_file)
