import pandas as pd
import pyarrow.parquet as pq
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter

# Check if the "punkt" tokenizer data is already downloaded, and download it if needed
if not nltk.data.find("tokenizers/punkt"):
    print("Downloading 'punkt' tokenizer data...")
    nltk.download('punkt')

# Define the paths to the Parquet files
parquet_file_paths = [
    "/home/berkin/Desktop/pythonProject/nlp-samples/dataset/train-00000-of-00002-8d1165eecfd8ca6a.parquet",
    "/home/berkin/Desktop/pythonProject/nlp-samples/dataset/train-00001-of-00002-d36826e9aff76e3d.parquet"
]

# Load the Parquet files into a list of Pandas DataFrames
dataframes = [pq.read_table(parquet_file_path).to_pandas() for parquet_file_path in parquet_file_paths]

# Inspect the structure of the Parquet files and identify the correct column name containing text data
# You should replace 'text_column_name' with the actual column name
text_column_name = 'content'  # Modify this to the correct column name

# Initialize an empty list to store tokenized examples
tokenized_examples = []

# Tokenize the text and perform word frequency analysis
word_frequency = Counter()

for df in dataframes:
    for text in df[text_column_name]:
        # Tokenize the text into words
        words = word_tokenize(text)

        # Update word frequency counts
        word_frequency.update(words)

        # Append the tokenized example to the list
        tokenized_examples.append(words)

# Calculate the total word count
total_word_count = sum(word_frequency.values())

# Print progress and results
print(f"Total word count in the dataset: {total_word_count}")

# Print the most common words and their frequencies
most_common_words = word_frequency.most_common(10)
print("Most common words:")
for word, frequency in most_common_words:
    print(f"{word}: {frequency}")

# Print three examples of tokenized text
print("\nTokenized Examples:")
for i, example in enumerate(tokenized_examples[:3]):
    print(f"Example {i + 1}: {example}")

"""
for i, example in enumerate(tokenized_examples[:3]):
    print(f"Example {i + 1}: {example}")
"""
