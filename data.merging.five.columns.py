import pandas as pd

# Define file paths for the YouTube and Twitter datasets
youtube_paths = [
    r"C:\Users\saric\OneDrive\Desktop\updated_merged_youtube_comments\merged_Microsoft_sentiment_stock.csv",
    r"C:\Users\saric\OneDrive\Desktop\updated_merged_youtube_comments\merged_Meta_sentiment_stock.csv",
    r"C:\Users\saric\OneDrive\Desktop\updated_merged_youtube_comments\merged_Apple_sentiment_stock.csv",
    r"C:\Users\saric\OneDrive\Desktop\updated_merged_youtube_comments\merged_Amazon_sentiment_stock.csv",
    r"C:\Users\saric\OneDrive\Desktop\updated_merged_youtube_comments\merged_NIO_sentiment_stock.csv",
    r"C:\Users\saric\OneDrive\Desktop\updated_merged_youtube_comments\merged_PG_sentiment_stock.csv",
    r"C:\Users\saric\OneDrive\Desktop\updated_merged_youtube_comments\merged_Tesla_sentiment_stock.csv",
    r"C:\Users\saric\OneDrive\Desktop\updated_merged_youtube_comments\merged_TSMC_sentiment_stock.csv"
]

twitter_path = r"C:\Users\saric\OneDrive\Desktop\updated_merged_youtube_comments\merged_twitter_sentiment_stock.csv"

# Function to load and select necessary columns from YouTube files
def load_youtube_data(youtube_paths):
    youtube_data = []
    for path in youtube_paths:
        df = pd.read_csv(path)
        # Select only necessary columns for the analysis
        df = df[['date', 'text', 'stock_price', 'volume', 'Stock Name', 'compound_score']]
        # Rename 'compound_score' to 'sentiment_score' for consistency
        df.rename(columns={'compound_score': 'sentiment_score'}, inplace=True)
        youtube_data.append(df)
    
    # Combine all YouTube data into one DataFrame
    youtube_merged = pd.concat(youtube_data, ignore_index=True)
    return youtube_merged

# Load the Twitter data
def load_twitter_data(twitter_path):
    twitter_df = pd.read_csv(twitter_path)
    # Select only necessary columns for the analysis
    twitter_df = twitter_df[['Date', 'Tweet', 'Stock Name', 'Sentiment', 'Stock Price', 'Stock Volume']]
    # Rename columns to match the YouTube columns
    twitter_df.rename(columns={'Tweet': 'text', 'Sentiment': 'sentiment_score', 'Stock Price': 'stock_price', 'Stock Volume': 'volume'}, inplace=True)
    twitter_df['date'] = twitter_df['Date']  # Add a 'date' column
    return twitter_df[['date', 'text', 'stock_price', 'volume', 'Stock Name', 'sentiment_score']]

# Load and merge all datasets
youtube_data = load_youtube_data(youtube_paths)
twitter_data = load_twitter_data(twitter_path)

# Combine YouTube and Twitter data
final_merged_data = pd.concat([youtube_data, twitter_data], ignore_index=True)

# Save the final merged dataset
final_merged_data.to_csv(r"C:\Users\saric\OneDrive\Desktop\five_columns_merged_final_sentiment_analysis_data.csv", index=False)

print("Merging complete and file saved.")
