import pandas as pd

# File paths for both datasets
youtube_files = [
    r"C:\Users\saric\OneDrive\Desktop\updated_merged_youtube_comments\merged_Amazon_sentiment_stock.csv",
    r"C:\Users\saric\OneDrive\Desktop\updated_merged_youtube_comments\merged_Apple_sentiment_stock.csv",
    r"C:\Users\saric\OneDrive\Desktop\updated_merged_youtube_comments\merged_Meta_sentiment_stock.csv",
    r"C:\Users\saric\OneDrive\Desktop\updated_merged_youtube_comments\merged_Microsoft_sentiment_stock.csv",
    r"C:\Users\saric\OneDrive\Desktop\updated_merged_youtube_comments\merged_NIO_sentiment_stock.csv",
    r"C:\Users\saric\OneDrive\Desktop\updated_merged_youtube_comments\merged_PG_sentiment_stock.csv",
    r"C:\Users\saric\OneDrive\Desktop\updated_merged_youtube_comments\merged_Tesla_sentiment_stock.csv",
    r"C:\Users\saric\OneDrive\Desktop\updated_merged_youtube_comments\merged_TSMC_sentiment_stock.csv"
]

twitter_file = r"C:\Users\saric\OneDrive\Desktop\updated_merged_youtube_comments\merged_twitter_sentiment_stock.csv"

# Function to extract the relevant columns from the datasets
def extract_columns(file_path, source="youtube"):
    """
    Extract relevant columns (text, sentiment score, stock price, stock ticker).
    """
    # Read dataset
    df = pd.read_csv(file_path)
    
    if source == "youtube":
        # YouTube specific columns
        df_filtered = df[['text', 'compound_score', 'stock_price', 'Stock Name']].copy()
        df_filtered.columns = ['text', 'sentiment_score', 'stock_price', 'Stock Name']
    else:
        # Twitter specific columns
        df_filtered = df[['Tweet', 'Sentiment', 'Stock Price', 'Stock Name']].copy()
        df_filtered.columns = ['text', 'sentiment_score', 'stock_price', 'Stock Name']
    
    return df_filtered

# Extract relevant columns from both YouTube and Twitter datasets
youtube_data = pd.concat([extract_columns(file, "youtube") for file in youtube_files], ignore_index=True)
twitter_data = extract_columns(twitter_file, "twitter")

# Merge both datasets into a final dataframe
final_data = pd.concat([youtube_data, twitter_data], ignore_index=True)

# Save the final merged data
final_data.to_csv(r"C:\Users\saric\OneDrive\Desktop\final_merged_data.csv", index=False)

print("Final merged data saved to: C:\\Users\\saric\\OneDrive\\Desktop\\final_merged_data.csv")
