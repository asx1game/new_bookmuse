"""
src/recommender.py
==================
This module contains the core recommendation logic.

The two key concepts used here are:

1. TF-IDF  (Term Frequency – Inverse Document Frequency)
   ─────────────────────────────────────────────────────
   TF-IDF is a way to convert a text document into a list of numbers (a "vector").

   - TF  (Term Frequency):      how often a word appears in THIS book
   - IDF (Inverse Doc Freq):    how rare the word is ACROSS ALL books

   A word that appears a lot in one book but rarely in others gets a HIGH score.
   Common words like "the", "and", "is" appear everywhere → low score → ignored.
   Unique, meaningful words like "necromancy" or "stoicism" → high score.

   Result: each book becomes a vector of thousands of numbers, one per unique word.

2. Cosine Similarity
   ──────────────────
   Once every book is a vector, we measure "similarity" as the cosine of the
   angle between two vectors.

   - Cosine similarity = 1.0  →  identical direction  →  very similar books
   - Cosine similarity = 0.0  →  perpendicular         →  completely unrelated
   - Cosine similarity = -1.0 →  opposite              →  impossible for text

   Why cosine and not regular distance?
   Because cosine ignores vector length. A long book and a short book about
   the same topic will have very similar direction even if the numbers differ.

HOW A RECOMMENDATION WORKS:
  User selects "Harry Potter" →
    1. Look up row for "Harry Potter" in the TF-IDF matrix
    2. Compute cosine similarity between that row and EVERY other row
    3. Sort all books by similarity score (highest first)
    4. Return the top N — those are the recommendations
"""

from dataclasses import dataclass   # @dataclass creates a clean class with automatic __init__
from typing import Any, Dict, List, Optional  # type hints make the code self-documenting

import numpy as np       # numpy: fast math on arrays and matrices
import pandas as pd      # pandas: DataFrame operations

# sklearn = scikit-learn, the standard Python machine learning library
from sklearn.feature_extraction.text import TfidfVectorizer  # converts text -> TF-IDF matrix
from sklearn.metrics.pairwise import cosine_similarity        # computes cosine similarity


@dataclass
class BookRecommender:
    """
    Stores everything needed to make recommendations:
      - df           : the original DataFrame (so we can return book metadata)
      - vectorizer   : the fitted TF-IDF model (knows the vocabulary)
      - tfidf_matrix : the matrix of shape (num_books x num_unique_words)

    We use @dataclass so Python automatically creates __init__(df, vectorizer, tfidf_matrix).
    """
    df:           pd.DataFrame
    vectorizer:   TfidfVectorizer
    tfidf_matrix: Any   # scipy sparse matrix; "Any" because the exact type is complex

    @classmethod
    def build(cls, df: pd.DataFrame) -> "BookRecommender":
        """
        Build the TF-IDF model from scratch. Called ONCE when the app starts.

        TfidfVectorizer parameters explained:
          stop_words="english"  : ignore very common words (the, is, and...)
          ngram_range=(1, 2)    : use single words AND two-word phrases
                                  e.g. "harry potter" is one feature, not two
          min_df=2              : ignore words that appear in fewer than 2 books
                                  (probably typos or very rare proper nouns)
          max_features=50000    : keep only the 50,000 most important features
                                  (limits memory usage)

        Steps:
          1. vectorizer.fit_transform(texts):
               a. BUILD the vocabulary: scan all books, collect unique words/phrases
               b. TRANSFORM: convert each book's text into a TF-IDF vector
          2. Returns a sparse matrix (most values are 0, stored efficiently)
        """
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            max_features=50000,
        )

        # fit_transform does step (a) + (b) in one call.
        # tfidf_matrix shape: (number_of_books, number_of_unique_words_in_vocabulary)
        tfidf_matrix = vectorizer.fit_transform(df["combined_text"])

        # cls(...) creates a new BookRecommender instance with these three things stored
        return cls(df=df, vectorizer=vectorizer, tfidf_matrix=tfidf_matrix)

    # ── Private helpers ──────────────────────────────────────────────────────

    def _find_title_index(self, title_query: str) -> Optional[int]:
        """
        Find the row index of a book by title (case-insensitive).

        Tries EXACT match first, then falls back to PARTIAL match.
        Returns None if no matching book is found.

        Example:
          _find_title_index("harry potter") might return row index 42
          Then tfidf_matrix[42] is Harry Potter's vector.
        """
        q = (title_query or "").strip().lower()
        if not q:
            return None

        # Get the lowercase version of all display titles
        titles_low = self.df["display_title"].fillna("").astype(str).str.lower()

        # Try exact match first (most precise)
        exact = self.df[titles_low == q]
        if len(exact) > 0:
            return int(exact.index[0])   # return the first exact match

        # Fall back to partial / substring match
        # .str.contains() returns True for every row where the title contains q
        contains = self.df[titles_low.str.contains(q, na=False)]
        if len(contains) == 0:
            return None   # no match found at all
        return int(contains.index[0])   # return the first partial match

    def _rank(
        self,
        query_vec,
        top_n: int,
        exclude_idx: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Compute cosine similarity between query_vec and EVERY book,
        then return the top_n most similar books as a list of dicts.

        Parameters:
          query_vec   : TF-IDF vector of the query (one row of the matrix,
                        or a freshly transformed user query)
          top_n       : how many results to return
          exclude_idx : if set, this book index is excluded from results
                        (used to prevent a book from recommending itself)

        How it works:
          cosine_similarity(query_vec, tfidf_matrix) returns a 1D array
          of similarity scores, one per book in the database.
          np.argsort()[::-1] sorts indices from highest to lowest score.
        """
        # cosine_similarity returns shape (1, num_books) — we flatten to 1D
        sims = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        # If we want to exclude the query book itself, set its score to -1
        # so it never appears in the top results
        if exclude_idx is not None:
            sims[exclude_idx] = -1

        # np.argsort gives indices that would sort the array ascending;
        # [::-1] reverses it to descending; [:top_n] takes only the first top_n
        best_indices = np.argsort(sims)[::-1][: int(top_n)]

        # Build the output list: each item is a dict of all book fields + the score
        results: List[Dict[str, Any]] = []
        for i in best_indices:
            row = self.df.iloc[i].to_dict()    # get all columns for row i as a dict
            row["similarity"] = float(sims[i]) # attach the similarity score
            results.append(row)

        return results

    # ── Public API ───────────────────────────────────────────────────────────

    def recommend_by_title(
        self, title_query: str, top_n: int = 6
    ) -> List[Dict[str, Any]]:
        """
        Given a book title, return the top_n most similar books.

        Steps:
          1. Find the row index of the book in the DataFrame
          2. Look up that book's TF-IDF vector from the matrix
          3. Pass the vector to _rank() to find the most similar books
          4. Exclude the query book itself from results (exclude_idx)

        Returns [] if the title is not found.
        """
        idx = self._find_title_index(title_query)
        if idx is None:
            return []   # book not found in the database

        # tfidf_matrix[idx] is a sparse row vector for the found book
        return self._rank(self.tfidf_matrix[idx], top_n=top_n, exclude_idx=idx)

    def recommend_by_text(
        self, free_text: str, top_n: int = 6
    ) -> List[Dict[str, Any]]:
        """
        Given any free text (user's description or question), return the top_n
        most similar books.

        This works because:
          - We use the SAME vectorizer that was used to build the matrix
          - vectorizer.transform() converts the user's text into a TF-IDF vector
            using the SAME vocabulary and IDF weights
          - So the resulting vector "speaks the same language" as the book vectors
          - Cosine similarity then tells us which books are closest to the query

        This is used in:
          - Tab 2 (Explore by topic) — directly
          - Tab 3 (Ask AI / RAG)     — to retrieve relevant books before calling GPT
          - Agent tools              — search_books and filter_books use this
        """
        text = (free_text or "").strip()
        if not text:
            return []

        # transform() (NOT fit_transform) uses the already-built vocabulary
        # It converts the query into a 1-row sparse matrix
        query_vec = self.vectorizer.transform([text])

        # No exclude_idx here — we are not looking up a specific book,
        # so all books are eligible including those that partially match the query text
        return self._rank(query_vec, top_n=top_n)
