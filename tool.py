import os
import pandas as pd

# Define the directory where the CSV files are located
csv_directory = ''

# Initialize a list to store the DataFrames
dataframes = []

# Variable to track if it is the first file
first_file = True

# Function to read files in chunks
def read_csv_in_chunks(file_path, label, chunk_size=10000):
    chunks = pd.read_csv(file_path, chunksize=chunk_size)
    for chunk in chunks:
        chunk['label'] = label
        yield chunk

# Iterate through all files in the directory
for filename in os.listdir(csv_directory):
    if filename.endswith('.csv'):
        file_path = os.path.join(csv_directory, filename)

        # Identify the label based on the filename
        if 'benign' in filename:
            label = 'benign'
        elif 'mirai' in filename:
            label = 'mirai'
        elif 'gafgyt' in filename:
            label = 'gafgyt'
        else:
            continue  # Skip files that do not match the expected names

        if first_file:
            # Read the first CSV file with the header
            for chunk in read_csv_in_chunks(file_path, label):
                dataframes.append(chunk)
            first_file = False
        else:
            # Read subsequent files ignoring the header
            for chunk in read_csv_in_chunks(file_path, label):
                dataframes.append(chunk)

# Combine all DataFrames into a single DataFrame
complete_dataset = pd.concat(dataframes, ignore_index=True)

# Save the combined DataFrame to a new CSV file
complete_dataset.to_csv('nbaiot_complete.csv', index=False)

print("All CSV files have been combined into 'nbaiot_complete.csv'")
