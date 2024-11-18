import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
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

# Initialize sentiment analyzer
analyzer = SentimentIntensityAnalyzer()


# Utility function to standardize date columns
def standardize_date_column(df, column_name):
    if column_name in df.columns:
        try:
            df[column_name] = pd.to_datetime(df[column_name], errors='coerce', utc=True)
            df[column_name] = df[column_name].dt.tz_localize(None)  # Remove timezone information
            df[column_name] = df[column_name].dt.strftime('%Y-%m-%d')  # Standardize to YYYY-MM-DD
            df[column_name] = pd.to_datetime(df[column_name], format='%Y-%m-%d')  # Ensure correct format
        except Exception as e:
            print(f"Error parsing dates in column '{column_name}': {e}")
    else:
        print(f"Column '{column_name}' not found in DataFrame. Available columns: {df.columns.tolist()}")
    return df


# Perform sentiment analysis
def compute_sentiment(df, text_column):
    if text_column in df.columns:
        print(f"Computing sentiment for column '{text_column}'...")
        df['Sentiment'] = df[text_column].apply(
            lambda x: analyzer.polarity_scores(str(x))['compound'] if pd.notna(x) else 0)
    else:
        print(f"Text column '{text_column}' not found in DataFrame. Available columns: {df.columns.tolist()}")
        df['Sentiment'] = 0  # Default sentiment if column missing
    return df


# Apply smoothing to sentiment using a rolling mean
def smooth_sentiment(df, window=7):
    """Apply rolling mean for sentiment data to smooth out the fluctuations."""
    df['Smoothed_Sentiment'] = df['Sentiment'].rolling(window=window, min_periods=1).mean()
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
        print(f"YouTube data not found for {company_name}")

    # Load Twitter data
    twitter_path = os.path.join(data_folder, company_files["twitter"])
    if os.path.exists(twitter_path):
        twitter_df = pd.read_csv(twitter_path)
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
        print("Twitter data not found.")

    # Load stock price data
    stock_path = os.path.join(data_folder, company_files["stock_price"])
    if os.path.exists(stock_path):
        stock_df = pd.read_csv(stock_path)
        stock_df = standardize_date_column(stock_df, 'Date')
        stock_df['Close'] = pd.to_numeric(stock_df['Close'], errors='coerce')
        stock_df = stock_df.dropna(subset=['Date', 'Close']).set_index('Date').sort_index()
        company_data['stock_price'] = stock_df
        print(f"Loaded Stock Price data for {company_name}. Rows: {len(stock_df)}")
    else:
        print(f"Stock Price data not found for {company_name}")

    return company_data


# Plot sentiment and stock price separately for a company
def plot_sentiment_and_stock_price(company_name, company_data, event_dates):
    # Prepare separate data for sentiment and stock price plots
    sentiment_data = []
    stock_data = []

    for event_date in event_dates:
        print(f"\nProcessing data for event date: {event_date}")  # Log event date being processed

        # Define the window around the event
        window_start = event_date - pd.Timedelta(days=10)
        window_end = event_date + pd.Timedelta(days=10)
        print(f"{window_start}-{window_end}")

        # Prepare data for sentiment plot
        twitter_sentiment_window = company_data.get('twitter', pd.DataFrame()).set_index('Date')[
                                       'Smoothed_Sentiment'].loc[window_start:window_end]
        youtube_sentiment_window = company_data.get('youtube', pd.DataFrame()).set_index('Date')[
                                       'Smoothed_Sentiment'].loc[window_start:window_end]

        # Convert dates to "Days from Event"
        twitter_sentiment_window.index = (twitter_sentiment_window.index - event_date).days
        youtube_sentiment_window.index = (youtube_sentiment_window.index - event_date).days

        sentiment_data.append(twitter_sentiment_window)
        sentiment_data.append(youtube_sentiment_window)

        # Prepare data for stock price plot
        stock_price_window = company_data.get('stock_price', pd.DataFrame()).loc[window_start:window_end, 'Close']
        stock_price_window.index = (
                    stock_price_window.index - event_date).days  # Convert stock dates to "Days from Event"

        stock_data.append(stock_price_window)

    # Plot Sentiment Data
    plt.figure(figsize=(12, 8))
    for sentiment in sentiment_data:
        if not sentiment.empty:
            plt.plot(sentiment.index, sentiment, label=f"Sentiment Trend", linestyle='--')

    plt.axvline(0, color='black', linestyle='--', label="Event Day")
    plt.title(f"Smoothed Sentiment Trends for {company_name}")
    plt.xlabel("Days from Event")
    plt.ylabel("Smoothed Sentiment Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    sentiment_plot_file = os.path.join(output_folder, f"{company_name}_Sentiment.png")
    plt.savefig(sentiment_plot_file)
    plt.close()
    print(f"Sentiment plot saved: {sentiment_plot_file}")

    # Plot Stock Price Data
    plt.figure(figsize=(12, 8))
    for stock in stock_data:
        if not stock.empty:
            plt.plot(stock.index, stock, label=f"Stock Price Trend", color='blue')

    plt.axvline(0, color='black', linestyle='--', label="Event Day")
    plt.title(f"Stock Price Trends for {company_name}")
    plt.xlabel("Days from Event")
    plt.ylabel("Stock Price ($)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    stock_price_plot_file = os.path.join(output_folder, f"{company_name}_Stock_Price.png")
    plt.savefig(stock_price_plot_file)
    plt.close()
    print(f"Stock price plot saved: {stock_price_plot_file}")


# Plot sentiment and stock price scatter plot and correlation heatmap
def plot_scatter_and_heatmap(company_name, company_data, event_dates):
    # Combine sentiment and stock price data for correlation and scatter plot
    combined_df = pd.DataFrame()

    for event_date in event_dates:
        print(f"\nCombining data for event date: {event_date}")

        # Define the window around the event
        window_start = event_date - pd.Timedelta(days=10)
        window_end = event_date + pd.Timedelta(days=10)

        # Fetch sentiment data
        twitter_sentiment_window = company_data.get('twitter', pd.DataFrame()).set_index('Date')[
                                       'Smoothed_Sentiment'].loc[window_start:window_end]
        youtube_sentiment_window = company_data.get('youtube', pd.DataFrame()).set_index('Date')[
                                       'Smoothed_Sentiment'].loc[window_start:window_end]

        # Fetch stock price data
        stock_price_window = company_data.get('stock_price', pd.DataFrame()).loc[window_start:window_end, 'Close']

        # Create a temporary DataFrame for combined data
        temp_df = pd.DataFrame({
            'Date': stock_price_window.index,
            'Stock Price': stock_price_window.values,
            'Twitter Sentiment': twitter_sentiment_window.reindex(stock_price_window.index, fill_value=0).values,
            'YouTube Sentiment': youtube_sentiment_window.reindex(stock_price_window.index, fill_value=0).values
        })

        combined_df = pd.concat([combined_df, temp_df])

    # Drop missing or invalid values
    combined_df.dropna(inplace=True)

    # Scatter Plot: Sentiment vs Stock Price
    plt.figure(figsize=(12, 8))
    # Twitter Sentiment Scatter and Regression Line
    sns.scatterplot(data=combined_df, x='Stock Price', y='Twitter Sentiment', label="Twitter Sentiment",
                    color='orange', s=100)  # Adjust point size
    sns.regplot(data=combined_df, x='Stock Price', y='Twitter Sentiment', scatter=False, color='orange',
                label="Twitter Correlation", ci=None)  # Add regression line without scatter points

    # YouTube Sentiment Scatter and Regression Line
    sns.scatterplot(data=combined_df, x='Stock Price', y='YouTube Sentiment', label="YouTube Sentiment",
                    color='green', s=100)  # Adjust point size
    sns.regplot(data=combined_df, x='Stock Price', y='YouTube Sentiment', scatter=False, color='green',
                label="YouTube Correlation", ci=None)  # Add regression line without scatter point
    plt.title(f"Stock Price vs Sentiment for {company_name}")
    plt.xlabel("Stock Price ($)")
    plt.ylabel("Sentiment Score")
    plt.legend()
    plt.grid(True)
    scatter_plot_file = os.path.join(output_folder, f"{company_name}_Scatter_Plot.png")
    plt.savefig(scatter_plot_file)
    plt.close()
    print(f"Scatter plot saved: {scatter_plot_file}")

    # Correlation Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(combined_df[['Stock Price', 'Twitter Sentiment', 'YouTube Sentiment']].corr(), annot=True,
                cmap="coolwarm", fmt=".2f")
    plt.title(f"Correlation Matrix for {company_name}")
    heatmap_file = os.path.join(output_folder, f"{company_name}_Correlation_Matrix.png")
    plt.savefig(heatmap_file)
    plt.close()
    print(f"Correlation heatmap saved: {heatmap_file}")


# Main execution
if __name__ == "__main__":
    event_df = pd.read_csv(event_data_file)
    event_df = standardize_date_column(event_df, 'Event Date')
    event_df.rename(columns={'Event Date': 'Date'}, inplace=True)
    event_df = event_df.dropna(subset=['Date'])

    for company_name, group in event_df.groupby('CompanyTitle'):
        event_dates = group['Date'].tolist()
        company_data = fetch_company_data(company_name)

        # Skip if no data is available
        if not company_data:
            print(f"No data available for {company_name}. Skipping.")
            continue

        # Plot sentiment and stock price separately
        plot_sentiment_and_stock_price(company_name, company_data, event_dates)

        # Plot scatter plot and heatmap
        plot_scatter_and_heatmap(company_name, company_data, event_dates)