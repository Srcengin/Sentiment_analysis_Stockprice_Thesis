import pandas as pd
import re
from nltk.tokenize import word_tokenize

# Define the path to the input file and output folder
input_file = r"C:\Users\saric\OneDrive\Desktop\tokenized\last_cleaned_twitter_data.csv"
output_folder = r"C:\Users\saric\OneDrive\Desktop\tokenized"
output_file = f"{output_folder}\\last_tokenized_cleaned_twitter_data.csv"

# Load the cleaned dataset
df = pd.read_csv(input_file)

# Ensure 'text' column exists
if 'text' in df.columns:
    # Tokenize text column
    def tokenize_text(text):
        if pd.isna(text):  # Handle missing values
            return ""
        try:
            return " ".join(word_tokenize(text.lower()))
        except Exception as e:
            print(f"Error tokenizing text: {e}")
            return ""

    # Apply tokenization
    print("Tokenizing text column...")
    df['text'] = df['text'].apply(tokenize_text)

    # Save the tokenized dataset
    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"Tokenized file saved: {output_file}")
else:
    print(f"Text column 'text' not found in {input_file}. Please check the file.")
