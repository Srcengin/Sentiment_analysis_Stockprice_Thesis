import pandas as pd
import os

# Define the paths for the sentiment files and stock price files
sentiment_folder = r"C:\Users\saric\OneDrive\Desktop\sentiment_results"
stock_price_folder = r"C:\Users\saric\OneDrive\Desktop\StockPrice-Dataset"
output_folder = r"C:\Users\saric\OneDrive\Desktop\merged_results"

# Define the list of companies and their respective stock tickers
companies = {
    "Amazon": "AMZN",
    "Apple": "AAPL",
    "Meta": "META",
    "Microsoft": "MSFT",
    "Tesla": "TSLA",
    "Nio": "NIO",
    "PG": "PG",
    "TSMC": "TSM"
}

# Function to merge sentiment data with stock prices
def merge_sentiment_with_stock(sentiment_file, stock_file, company_ticker):
    print(f"Processing data for {company_ticker}...")

    # Load the sentiment data
    sentiment_df = pd.read_csv(sentiment_file)
    sentiment_df['date'] = pd.to_datetime(sentiment_df['date'], errors='coerce', dayfirst=True)

    # Load the stock price data
    stock_df = pd.read_csv(stock_file)
    stock_df['Date'] = pd.to_datetime(stock_df['Date'], errors='coerce', dayfirst=True)
    stock_df = stock_df[['Date', 'Close/Last', 'Volume']]  # Keep only the necessary columns

    # Merge sentiment with stock prices (based on the closest date)
    sentiment_df = pd.merge_asof(sentiment_df.sort_values('date'),
                                  stock_df.sort_values('Date'),
                                  left_on='date',
                                  right_on='Date',
                                  direction='nearest')

    # Save the merged data
    output_file = os.path.join(output_folder, f"merged_{company_ticker}_sentiment_stock.csv")
    sentiment_df.to_csv(output_file, index=False)
    print(f"Merged file saved: {output_file}")

# Loop through each company and process its data
for company, ticker in companies.items():
    # Define the paths for the sentiment and stock price files
    sentiment_file = os.path.join(sentiment_folder, f"sentiment_last_cleaned_{company.lower()}_stock_all_comments_with_replies.csv")
    stock_file = os.path.join(stock_price_folder, f"{company}_Stock_Price_5_years.csv")
    
    # Call the function to merge the data
    merge_sentiment_with_stock(sentiment_file, stock_file, ticker)
