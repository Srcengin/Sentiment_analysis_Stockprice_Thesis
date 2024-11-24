import pandas as pd
from datetime import timedelta

# File paths
twitter_file = r"C:\Users\saric\OneDrive\Desktop\sentiment_results\sentiment_last_cleaned_twitter_data.csv"
stock_price_folder = r"C:\Users\saric\OneDrive\Desktop\StockPrice-Dataset"
output_file = r"C:\Users\saric\OneDrive\Desktop\merged_results\merged_twitter_sentiment_stock.csv"

# Stock price files for companies
stock_price_files = {
    "AMZN": "Amazon_Stock_Price_5_years.csv",
    "AAPL": "Apple_Stock_Price_5_years.csv",
    "META": "Meta_Stock_Price_5_years.csv",
    "MSFT": "Microsoft_Stock_Price_5_years.csv",
    "TSLA": "Tesla_Stock_Price_5_years.csv",
    "NIO": "Nio_Inc_Stock_Price_5_years.csv",
    "PG": "Protector_and_Gamble_Stock_Price_5_years.csv",
    "TSM": "TSM_Stock_Price_5_years.csv"
}

# Function to find closest stock price date
def get_closest_date(stock_dates, target_date):
    """
    Get the closest available stock price date for a given target date.
    """
    try:
        target_date = pd.to_datetime(target_date)
        stock_dates = pd.to_datetime(stock_dates)
        closest_date = stock_dates[stock_dates <= target_date].max()
        return closest_date.strftime('%Y-%m-%d') if pd.notna(closest_date) else None
    except Exception as e:
        print(f"Error finding closest date for {target_date}: {e}")
        return None

# Merge Twitter sentiment with stock prices
def merge_twitter_stock(twitter_file, stock_price_folder, stock_price_files, output_file):
    print(f"Processing Twitter file: {twitter_file}")
    twitter_df = pd.read_csv(twitter_file)
    
    # Ensure required columns
    if 'Stock Name' not in twitter_df.columns or 'Date' not in twitter_df.columns:
        raise ValueError("Twitter dataset must include 'Stock Name' and 'Date' columns.")
    
    # Initialize list for merged data
    merged_data = []

    for ticker, stock_file in stock_price_files.items():
        print(f"Processing stock price data for {ticker}...")
        stock_file_path = f"{stock_price_folder}/{stock_file}"
        
        # Read stock price data
        stock_df = pd.read_csv(stock_file_path)
        stock_df['Date'] = pd.to_datetime(stock_df['Date'], format='%m/%d/%Y')  # Convert stock dates to datetime
        stock_df['Close/Last'] = stock_df['Close/Last'].replace('[\$,]', '', regex=True).astype(float)  # Remove $ and convert to float
        
        # Filter Twitter data for the current ticker
        ticker_data = twitter_df[twitter_df['Stock Name'] == ticker]
        if ticker_data.empty:
            print(f"No Twitter data found for {ticker}. Skipping...")
            continue
        
        # Match stock price for each tweet date
        for _, row in ticker_data.iterrows():
            comment_date = row['Date']
            stock_price_date = get_closest_date(stock_df['Date'], comment_date)
            
            if stock_price_date:
                stock_row = stock_df[stock_df['Date'] == stock_price_date].iloc[0]
                merged_data.append({
                    **row,
                    "Stock Price Date": stock_price_date,
                    "Stock Price": stock_row['Close/Last'],
                    "Stock Volume": stock_row['Volume']
                })
            else:
                print(f"No stock price available for {ticker} on or before {comment_date}.")
    
    # Create a merged DataFrame
    merged_df = pd.DataFrame(merged_data)
    merged_df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"Merged file saved: {output_file}")

# Run the merging function
merge_twitter_stock(twitter_file, stock_price_folder, stock_price_files, output_file)
