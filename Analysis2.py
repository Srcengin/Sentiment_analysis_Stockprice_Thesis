import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
import matplotlib.pyplot as plt
import os
import shutil
import seaborn as sns

# Define paths
data_folder = 'data'
output_folder = 'plots'
event_data_file = os.path.join(data_folder, 'Company_Event_analysis_data.csv')

# Stock symbols mapped to company names
stock_symbols = {
    'Tesla': 'TSLA',
    'Amazon': 'AMZN',
    'Meta': 'META',
    'TSM': 'TSM',
    'NIO': 'NIO',
    'PG': 'PG',
    'Microsoft': 'MSFT',
    'Apple': 'AAPL'
}

# Ensure the output folder exists and clear it
if os.path.exists(output_folder):
    shutil.rmtree(output_folder)
os.makedirs(output_folder, exist_ok=True)

# Load FinBERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained("yiyanghkust/finbert-tone")
model = BertForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
finbert_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)


# Utility function to standardize date columns
def standardize_date_column(df, column_name):
    if column_name in df.columns:
        try:
            print(f"Standardizing date column: {column_name}")
            df[column_name] = pd.to_datetime(df[column_name], errors='coerce', utc=True)
            df[column_name] = df[column_name].dt.tz_localize(None)  # Remove timezone information
            df[column_name] = df[column_name].dt.strftime('%Y-%m-%d')  # Standardize to YYYY-MM-DD
            df[column_name] = pd.to_datetime(df[column_name], format='%Y-%m-%d')  # Ensure correct format
        except Exception as e:
            print(f"Error parsing dates in column '{column_name}': {e}")
    else:
        print(f"Column '{column_name}' not found in DataFrame. Available columns: {df.columns.tolist()}")
    return df


# Perform sentiment analysis using FinBERT
def compute_sentiment(df, text_column):
    if text_column in df.columns:
        print(f"Computing sentiment for column '{text_column}' using FinBERT...")
        sentiments = []
        for text in df[text_column].astype(str):  # Ensure text is a string
            if pd.notna(text) and text.strip():
                sentiment = finbert_pipeline(text[:512])  # Limit to 512 tokens
                sentiments.append(sentiment[0]['label'])  # Extract the label (Positive, Neutral, Negative)
            else:
                sentiments.append("Neutral")  # Default sentiment for empty text
        df['Sentiment'] = sentiments
        df['Sentiment_Score'] = df['Sentiment'].map({
            'positive': 1,
            'neutral': 0,
            'negative': -1
        })  # Map sentiment to numerical values
    else:
        print(f"Text column '{text_column}' not found in DataFrame. Available columns: {df.columns.tolist()}")
        df['Sentiment'] = 'Neutral'
        df['Sentiment_Score'] = 0  # Default sentiment if column missing
    return df


# Apply smoothing to sentiment using a rolling mean
def smooth_sentiment(df, window=7):
    """Apply rolling mean for sentiment data to smooth out the fluctuations."""
    df['Smoothed_Sentiment'] = df['Sentiment_Score'].rolling(window=window, min_periods=1).mean()
    return df


# Fetch data for a specific company
def fetch_company_data(company_name):
    print(f"\nFetching data for {company_name}...")
    company_files = {
        "youtube": f"{company_name}_youtube_comments.csv",
        "twitter": "Twitter_tweet_data.csv",
        "stock_price": f"{company_name}_Stock_Price.csv"
    }
    company_data = {}

    # Load YouTube data
    youtube_path = os.path.join(data_folder, company_files["youtube"])
    if os.path.exists(youtube_path):
        youtube_df = pd.read_csv(youtube_path)
        print(f"YouTube data for {company_name}: {youtube_path}")
        print(youtube_df.head())  # Show first few rows
        youtube_df = standardize_date_column(youtube_df, 'date')
        youtube_df = compute_sentiment(youtube_df, 'comments')
        youtube_df = smooth_sentiment(youtube_df, window=7)  # Smooth sentiment
        youtube_df.rename(columns={'date': 'Date'}, inplace=True)
        numeric_cols = youtube_df.select_dtypes(include='number')  # Filter numeric columns
        youtube_df = youtube_df.groupby('Date')[
            numeric_cols.columns].mean().reset_index()  # Aggregate only numeric columns
        company_data['youtube'] = youtube_df
        print(f"Loaded YouTube data for {company_name}. Rows: {len(youtube_df)}")
    else:
        print(f"YouTube data not found for {company_name}: {youtube_path}")

    # Load Twitter data
    twitter_path = os.path.join(data_folder, company_files["twitter"])
    if os.path.exists(twitter_path):
        twitter_df = pd.read_csv(twitter_path)
        print(f"Twitter data for {company_name}: {twitter_path}")
        print(twitter_df.head())  # Show first few rows
        stock_symbol = stock_symbols.get(company_name)
        if stock_symbol:
            twitter_df = twitter_df[twitter_df['Stock Name'].str.contains(stock_symbol, case=False, na=False)]
            twitter_df = standardize_date_column(twitter_df, 'Date')
            twitter_df = compute_sentiment(twitter_df, 'Tweet')
            twitter_df = smooth_sentiment(twitter_df, window=7)  # Smooth sentiment
            numeric_cols = twitter_df.select_dtypes(include='number')  # Filter numeric columns
            twitter_df = twitter_df.groupby('Date')[
                numeric_cols.columns].mean().reset_index()  # Aggregate only numeric columns
            company_data['twitter'] = twitter_df
            print(f"Loaded Twitter data for {company_name}. Rows: {len(twitter_df)}")
        else:
            print(f"Stock symbol for {company_name} not found.")
    else:
        print(f"Twitter data not found: {twitter_path}")

    # Load stock price data
    stock_path = os.path.join(data_folder, company_files["stock_price"])
    if os.path.exists(stock_path):
        stock_df = pd.read_csv(stock_path)
        print(f"Stock Price data for {company_name}: {stock_path}")
        print(stock_df.head())  # Show first few rows
        stock_df = standardize_date_column(stock_df, 'Date')
        stock_df['Close'] = pd.to_numeric(stock_df['Close'], errors='coerce')
        stock_df = stock_df.dropna(subset=['Date', 'Close']).set_index('Date').sort_index()
        company_data['stock_price'] = stock_df
        print(f"Loaded Stock Price data for {company_name}. Rows: {len(stock_df)}")
    else:
        print(f"Stock Price data not found for {company_name}: {stock_path}")

    return company_data


# Main execution
if __name__ == "__main__":
    print(f"Event data file path: {event_data_file}")
    print(f"Data folder: {data_folder}")
    print(f"Output folder: {output_folder}")

    print("Loading event data...")
    try:
        event_df = pd.read_csv(event_data_file)
        print("Event data loaded successfully!")
    except Exception as e:
        print(f"Error loading event data: {e}")
        exit()

    print("Standardizing event data...")
    event_df = standardize_date_column(event_df, 'Event Date')
    event_df.rename(columns={'Event Date': 'Date'}, inplace=True)
    event_df = event_df.dropna(subset=['Date'])
    print(f"Event data standardized. Number of events: {len(event_df)}")

    for company_name, group in event_df.groupby('CompanyTitle'):
        print(f"\nProcessing company: {company_name}")
        event_dates = group['Date'].tolist()
        company_data = fetch_company_data(company_name)

        if not company_data:
            print(f"No data available for {company_name}. Skipping.")
            continue

        print(f"Data fetched for {company_name}:")
        print(f"- YouTube data rows: {len(company_data.get('youtube', []))}")
        print(f"- Twitter data rows: {len(company_data.get('twitter', []))}")
        print(f"- Stock price data rows: {len(company_data.get('stock_price', []))}")
