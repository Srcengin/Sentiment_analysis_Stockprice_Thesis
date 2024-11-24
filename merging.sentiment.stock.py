import pandas as pd
import os

# Paths
sentiment_folder = r"C:\Users\saric\OneDrive\Desktop\sentiment_results"
stock_price_folder = r"C:\Users\saric\OneDrive\Desktop\StockPrice-Dataset"
output_folder = r"C:\Users\saric\OneDrive\Desktop\merged_results"

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Mapping of companies to files
companies = {
    "Amazon": "sentiment_last_cleaned_amazon_stock_all_comments_with_replies.csv",
    "Apple": "sentiment_last_cleaned_apple_stock_all_comments.csv",
    "Meta": "sentiment_last_cleaned_meta_stock_all_comments_with_replies.csv",
    "Microsoft": "sentiment_last_cleaned_microsoft_stock_all_comments_with_replies.csv",
    "Nio": "sentiment_last_cleaned_nio_stock_all_comments_with_replies.csv",
    "PG": "sentiment_last_cleaned_pg_stock_all_comments_with_replies.csv",
    "Tesla": "sentiment_last_cleaned_tesla_stock_all_comments_with_replies.csv",
    "TSMC": "sentiment_last_cleaned_tsmc_stock_all_comments_with_replies.csv",
    "Twitter": "sentiment_last_cleaned_twitter_data.csv",  # Twitter handled separately
}

stock_files = {
    "Amazon": "Amazon_Stock_Price_5_years.csv",
    "Apple": "Apple_Stock_Price_5_years.csv",
    "Meta": "Meta_Stock_Price_5_years.csv",
    "Microsoft": "Microsoft_Stock_Price_5_years.csv",
    "Nio": "Nio_Inc_Stock_Price_5_years.csv",
    "PG": "Protector_and_Gamble_Stock_Price_5_years.csv",
    "Tesla": "Tesla_Stock_Price_5_years.csv",
    "TSMC": "TSM_Stock_Price_5_years.csv",
}

# Helper function to clean and standardize date format
def standardize_date(df, date_column, format='%Y-%m-%d'):
    df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
    df.dropna(subset=[date_column], inplace=True)  # Drop rows with invalid dates
    df[date_column] = df[date_column].dt.strftime(format)  # Convert to desired format
    return df

# Merge sentiment and stock prices
for company, sentiment_file in companies.items():
    if company == "Twitter":
        continue  # Skip Twitter for now, handled separately

    sentiment_path = os.path.join(sentiment_folder, sentiment_file)
    stock_path = os.path.join(stock_price_folder, stock_files[company])

    try:
        # Load data
        sentiment_df = pd.read_csv(sentiment_path)
        stock_df = pd.read_csv(stock_path)

        # Standardize dates
        sentiment_df = standardize_date(sentiment_df, "date")
        stock_df = standardize_date(stock_df, "Date")

        # Merge on date
        merged_df = pd.merge(
            sentiment_df,
            stock_df[['Date', 'Close/Last']],  # Include only the 'Close/Last' column
            left_on="date",
            right_on="Date",
            how="left"  # Use left join to ensure all sentiment data is retained
        )

        # Drop unnecessary 'Date' column from stock data
        merged_df.drop(columns=["Date"], inplace=True)

        # Save the merged file
        output_file = os.path.join(output_folder, f"merged_{company}_sentiment_stock.csv")
        merged_df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"Merged file saved for {company}: {output_file}")

    except Exception as e:
        print(f"Error processing {company}: {e}")

# Handle Twitter dataset
try:
    twitter_sentiment_path = os.path.join(sentiment_folder, companies["Twitter"])
    twitter_sentiment_df = pd.read_csv(twitter_sentiment_path)

    # Standardize date format in Twitter sentiment data
    twitter_sentiment_df = standardize_date(twitter_sentiment_df, "date")

    # Save cleaned Twitter sentiment file for further use
    twitter_output_file = os.path.join(output_folder, "merged_twitter_sentiment_stock.csv")
    twitter_sentiment_df.to_csv(twitter_output_file, index=False, encoding='utf-8')
    print(f"Cleaned Twitter sentiment file saved: {twitter_output_file}")

except Exception as e:
    print(f"Error processing Twitter sentiment data: {e}")
