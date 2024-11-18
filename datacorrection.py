import pandas as pd
import os
from dateutil.parser import parse

# Define the folder containing your CSV files
data_folder = 'E:\\PycharmProjects\\stock files'  # Folder with your original CSV files

# Function to clean the entire file by removing "$" and fixing the "Date" column
def clean_and_fix_csv(file_path):
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)

        # Remove "$" symbols from all columns
        df = df.applymap(lambda x: str(x).replace('$', '') if isinstance(x, str) else x)
        print(f"Removed '$' symbols in {file_path}")

        # Fix the "Date" column to 'dd-mm-yyyy' format if it exists
        if 'Date' in df.columns:
            # Convert each value in the 'Date' column to the correct format
            def standardize_date(date):
                try:
                    # Parse the date and reformat to 'dd-mm-yyyy'
                    return parse(date, dayfirst=False, fuzzy=True).strftime('%d-%m-%Y')
                except (ValueError, TypeError):
                    # Return None for invalid dates
                    return None

            df['Date'] = df['Date'].apply(standardize_date)
            print(f"Processed 'Date' column in {file_path}")

        # Save the cleaned DataFrame back to the original file
        df.to_csv(file_path, index=False)
        print(f"Replaced original file with cleaned data: {file_path}")

    except Exception as e:
        print(f"Error processing {file_path}: {e}")

# Process each CSV file in the data folder
for filename in os.listdir(data_folder):
    if filename.endswith('.csv'):
        file_path = os.path.join(data_folder, filename)
        clean_and_fix_csv(file_path)
