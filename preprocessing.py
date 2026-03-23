"""
src/preprocessing.py
====================
This module is responsible for ONE thing: reading the raw CSV file and
turning it into a clean pandas DataFrame that the rest of the project can use.

A DataFrame is like an Excel spreadsheet in Python:
  - Each ROW is one book
  - Each COLUMN is one piece of information (title, author, rating, etc.)

WHY DO WE NEED PREPROCESSING?
  Raw data is messy. Books may have:
    - Missing descriptions or authors
    - HTML tags left in the text  (e.g. "<b>Harry Potter</b>")
    - Inconsistent capitalisation  ("Fiction" vs "fiction" vs "FICTION")
    - Extra whitespace or punctuation

  If we fed this dirty text directly to TF-IDF, the model would treat
  "Fiction" and "fiction" as two completely different words — which is wrong.
  Cleaning first ensures consistent, meaningful vectorization.
"""

import re          # re = regular expressions, a tool for pattern-matching in text
import pandas as pd  # pandas handles tabular data (DataFrames)

# These are the column names we expect to find in the CSV.
# We will clean each of them and then combine them into one big text per book.
TEXT_COLS_CANDIDATES = ["title", "authors", "categories", "description"]


def clean_text(x: str) -> str:
    """
    Clean a single piece of text so it is ready for TF-IDF vectorization.

    Steps:
      1. Handle missing values — if the field is NaN (empty), return ""
      2. Lowercase everything  — "Harry" and "harry" should count as the same word
      3. Remove HTML tags      — "<b>text</b>" becomes "text"
      4. Keep only letters, numbers, and spaces  — removes punctuation like ,.!?
      5. Collapse multiple spaces into one        — "hello   world" -> "hello world"

    Example:
      Input:  "<b>Harry Potter</b> & the Chamber!"
      Output: "harry potter  the chamber"
    """
    # Step 1: if the value is NaN (pandas missing value), return empty string
    if pd.isna(x):
        return ""

    # Step 2: convert to string and make everything lowercase
    x = str(x).lower()

    # Step 3: remove HTML tags using a regular expression
    # r"<[^>]+>" matches anything that starts with < and ends with >
    x = re.sub(r"<[^>]+>", " ", x)

    # Step 4: remove anything that is not a letter (a-z), digit (0-9), or space
    x = re.sub(r"[^a-z0-9\s]+", " ", x)

    # Step 5: replace multiple consecutive spaces with a single space, then trim
    x = re.sub(r"\s+", " ", x).strip()

    return x


def load_books(csv_path: str) -> pd.DataFrame:
    """
    Load the books CSV, clean all text columns, and return a ready-to-use DataFrame.

    Parameters:
      csv_path : path to the CSV file, e.g. "data/books.csv"

    Returns:
      df : a pandas DataFrame where:
           - Text columns are cleaned
           - A new "combined_text" column merges title+authors+categories+description
           - A "display_title" column holds the original (uncleaned) title for display
           - Rows with completely empty text are removed

    Why "combined_text"?
      TF-IDF needs ONE text string per book. By combining all text fields,
      we ensure that a search for "rowling fantasy magic" can match books
      whether those words appear in the title, author name, category, or description.
    """
    # Read the CSV file into a DataFrame.
    # pandas automatically detects column names from the first row of the CSV.
    df = pd.read_csv(csv_path)

    # Make sure all expected columns exist — if one is missing, create it as empty.
    # This prevents KeyError crashes when the CSV has a slightly different structure.
    for col in TEXT_COLS_CANDIDATES:
        if col not in df.columns:
            df[col] = ""

    # Apply clean_text() to every cell in each text column.
    # .apply() runs a function on every element of a Series (column).
    for col in TEXT_COLS_CANDIDATES:
        df[col] = df[col].apply(clean_text)

    # Create "combined_text": one long string per book that TF-IDF will vectorize.
    # We concatenate title + authors + categories + description with spaces between.
    # .str.strip() removes any leading/trailing whitespace from the result.
    df["combined_text"] = (
        df["title"] + " " +
        df["authors"] + " " +
        df["categories"] + " " +
        df["description"]
    ).str.strip()

    # "display_title" is what we show to the user in the UI.
    # We use the already-cleaned title column (lowercase is fine for display here).
    # .fillna("") ensures no NaN values slip through to the UI.
    df["display_title"] = df["title"].fillna("").astype(str)

    # Remove rows where combined_text is empty after cleaning.
    # A book with no text at all is useless for recommendations.
    # .reset_index(drop=True) re-numbers the rows from 0 after filtering.
    df = df[df["combined_text"].str.len() > 0].reset_index(drop=True)

    return df
