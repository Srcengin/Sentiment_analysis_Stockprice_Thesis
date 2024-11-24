import pandas as pd
import os
import re

# Define input and output directories
youtube_files = [
    r"C:\Users\saric\OneDrive\Desktop\Youtube_Comments\amazon_stock_all_comments_with_replies.csv",
    r"C:\Users\saric\OneDrive\Desktop\Youtube_Comments\apple_stock_all_comments.csv",
    r"C:\Users\saric\OneDrive\Desktop\Youtube_Comments\meta_stock_all_comments_with_replies.csv",
    r"C:\Users\saric\OneDrive\Desktop\Youtube_Comments\microsoft_stock_all_comments_with_replies.csv",
    r"C:\Users\saric\OneDrive\Desktop\Youtube_Comments\nio_stock_all_comments_with_replies.csv",
    r"C:\Users\saric\OneDrive\Desktop\Youtube_Comments\pg_stock_all_comments_with_replies.csv",
    r"C:\Users\saric\OneDrive\Desktop\Youtube_Comments\tesla_stock_all_comments_with_replies.csv",
    r"C:\Users\saric\OneDrive\Desktop\Youtube_Comments\tsmc_stock_all_comments_with_replies.csv"
]

twitter_file = r"C:\Users\saric\OneDrive\Desktop\Youtube_Comments\Filtered_Twitter_data.csv"
output_folder = r"C:\Users\saric\OneDrive\Desktop\Youtube_Comments"

# Function to clean text
def clean_text(text):
    """
    Clean text by removing URLs, special characters, and converting to lowercase.
    """
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    return text

# Function to preprocess dataset
def preprocess_dataset(file_path, text_column, date_column=None, is_youtube=False):
    """
    Preprocess a dataset by cleaning text, removing duplicates, and handling missing values.
    """
    print(f"Processing: {file_path}")
    df = pd.read_csv(file_path)

    # Drop duplicates and missing values
    df.drop_duplicates(subset=[text_column], inplace=True)
    df.dropna(subset=[text_column], inplace=True)

    # Clean text data
    df[text_column] = df[text_column].apply(clean_text)

    # Standardize and clean dates if date column is provided
    if date_column:
        try:
            df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
            df.dropna(subset=[date_column], inplace=True)
            df[date_column] = df[date_column].dt.strftime('%Y-%m-%d')  # Standardize date format
        except Exception as e:
            print(f"Error processing date column '{date_column}': {e}")

    # If processing YouTube data, ensure "is_reply" column exists
    if is_youtube and 'is_reply' not in df.columns:
        df['is_reply'] = False  # Default value if column is missing

    # Save the cleaned dataset
    output_file = os.path.join(output_folder, f"cleaned_{os.path.basename(file_path)}")
    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"Cleaned file saved: {output_file}")
    return df

# Process YouTube files
for youtube_file in youtube_files:
    preprocess_dataset(youtube_file, text_column='text', date_column='date', is_youtube=True)

# Process Twitter dataset
preprocess_dataset(twitter_file, text_column='Tweet', date_column='Date', is_youtube=False)
