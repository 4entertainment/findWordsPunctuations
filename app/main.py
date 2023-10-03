import pandas as pd
import pyarrow.parquet as pq
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
import string

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

# Initialize empty lists to store tokenized examples
tokenized_words = []

# Initialize Counters for word frequency analysis
word_frequency = Counter()

extra_punc = ["''", "``", "==", "==="]
word_string_punctuation = string.punctuation + ''.join(extra_punc)

# Tokenize the text and perform word frequency analysis (excluding punctuation)
for df in dataframes:
    for text in df[text_column_name]:
        # Tokenize the text into words
        words = word_tokenize(text)

        # Filter out punctuation marks from words
        words = [word for word in words if word not in word_string_punctuation]

        # Update word frequency counts
        word_frequency.update(words)

        # Append tokenized words to the list
        tokenized_words.extend(words)

# Calculate the total word count
total_word_count = sum(word_frequency.values())

# Print the total word count
print(f"Total word count in the dataset: {total_word_count}")

# Print the ten most common words and their frequencies
most_common_words = word_frequency.most_common(10)
print("\nMost common words:")
for word, frequency in most_common_words:
    print(f"{word}: {frequency}")

# Initialize empty lists to store tokenized punctuation marks
tokenized_punctuation = []

# Initialize Counters for punctuation frequency analysis
punctuation_frequency = Counter()

# Tokenize the text and perform punctuation frequency analysis
for df in dataframes:
    for text in df[text_column_name]:
        # Tokenize the text into punctuation marks
        punctuation = [char for char in text if char in string.punctuation]

        # Update punctuation frequency counts
        punctuation_frequency.update(punctuation)

        # Append tokenized punctuation marks to the list
        tokenized_punctuation.extend(punctuation)

# Print the most common punctuation marks and their frequencies
most_common_punctuation = punctuation_frequency.most_common(10)
print("\nMost common punctuation marks:")
for char, frequency in most_common_punctuation:
    print(f"{char}: {frequency}")

# Print three examples of tokenized text
print("\nTokenized Examples:")
for i, example in enumerate(tokenized_words[:3]):
    print(f"Example {i + 1}: {example}")

"""
for i, example in enumerate(tokenized_examples[:3]):
    print(f"Example {i + 1}: {example}")
"""
