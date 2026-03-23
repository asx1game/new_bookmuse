"""
src/agent_tools.py
==================
This module defines the FOUR TOOLS that the AI agent (GPT) can call.

WHAT IS A "TOOL" IN THIS CONTEXT?
  When we use GPT with "tool calling" (also called "function calling"), we tell GPT:
    "Here are 4 Python functions you can ask us to run. Each one takes some parameters
     and returns data. Whenever you need real data from our database, ask us to run one."

  GPT does NOT run Python itself. It just says:
    "Please call filter_books with genre='Mystery' and min_rating=4.0"

  Our Python code then actually runs the function and sends the result back to GPT.
  GPT reads the result and writes a natural-language answer for the user.

THE FOUR TOOLS:
  1. search_books      : free-text TF-IDF search (most general purpose)
  2. filter_books      : structured filter by genre, rating, year, author
  3. recommend_books   : cosine-similarity recommendations based on a book title
  4. get_book_info     : full details for exactly one book

For each tool there are two things defined here:
  a) A Python function that actually does the work
  b) A JSON schema (in TOOL_SCHEMAS) that describes the tool to GPT
"""

import pandas as pd   # pandas for DataFrame filtering operations

# ── Column name fallbacks ──────────────────────────────────────────────────────
# Different CSVs may use different column names for the same concept.
# We define fallback lists so the code works with multiple dataset formats.
RATING_COLS = ["average_rating", "rating", "ratings"]
DESC_COLS   = ["description", "desc", "summary"]
THUMB_COLS  = ["thumbnail", "thumbnail_url", "image_url", "img_url", "cover"]


# ── Shared private helpers ────────────────────────────────────────────────────

def _pick(row: dict, cols: list, default="") -> str:
    """
    Try each column name in `cols`; return the first value that exists and is not empty.
    Used to handle datasets with inconsistent column naming.
    """
    for c in cols:
        if c in row and pd.notna(row[c]) and str(row[c]).strip():
            return str(row[c])
    return default


def _book_summary(row: dict) -> dict:
    """
    Convert a raw DataFrame row (dict of all columns) into a compact summary dict.
    This is what GPT receives as the tool result — we only include the fields
    GPT needs to write a useful answer. Keeping it small avoids hitting token limits.
    """
    return {
        "title":       row.get("display_title") or row.get("title", ""),
        "authors":     row.get("authors", ""),
        "categories":  row.get("categories", ""),
        "rating":      _pick(row, RATING_COLS, "N/A"),
        "year":        str(row.get("published_year", "")),
        "description": str(_pick(row, DESC_COLS, ""))[:300],  # cap at 300 chars
        "thumbnail":   _pick(row, THUMB_COLS, ""),
    }


# ==============================================================================
# TOOL 1 – search_books
# ==============================================================================

def search_books(
    query: str,
    df: pd.DataFrame,
    recommender,
    top_n: int = 8,
) -> list:
    """
    Free-text search over the entire book catalogue using TF-IDF.

    This is the most general tool — GPT calls it when the user's request
    is a vague theme or topic (e.g. "books about leadership" or "sad romance").

    Parameters:
      query      : any text string describing what the user wants
      df         : the full books DataFrame (not used directly, passed for consistency)
      recommender: the BookRecommender instance (has the TF-IDF vectors)
      top_n      : how many results to return

    Returns:
      list of compact book dicts (one per result)
    """
    results = recommender.recommend_by_text(query, top_n=top_n)
    return [_book_summary(r) for r in results]


# ==============================================================================
# TOOL 2 – filter_books
# ==============================================================================

def filter_books(
    df: pd.DataFrame,
    genre: str       = "",
    min_rating: float = 0.0,
    max_rating: float = 5.0,
    year_from: int   = 0,
    year_to: int     = 9999,
    author: str      = "",
    top_n: int       = 8,
) -> list:
    """
    Filter the catalogue using structured criteria.

    GPT calls this when the user sets specific constraints like:
      "mystery books rated above 4 published after 2010"

    All parameters are optional — GPT only sends the ones relevant to the request.
    We use pandas boolean indexing to filter row by row.

    Parameters:
      df         : the full books DataFrame
      genre      : genre keyword to match (case-insensitive substring)
      min_rating : minimum average rating (0.0 to 5.0)
      max_rating : maximum average rating (0.0 to 5.0)
      year_from  : earliest publication year (inclusive)
      year_to    : latest publication year (inclusive)
      author     : author name keyword (case-insensitive substring)
      top_n      : max number of results to return

    Returns:
      list of compact book dicts sorted by rating (highest first)
    """
    result = df.copy()   # never modify the original DataFrame — always work on a copy

    # Genre filter: keep rows where the categories column contains the genre keyword
    # .fillna("") prevents crashes on missing values
    # case=False makes the match case-insensitive
    if genre:
        result = result[
            result["categories"].fillna("").str.contains(genre, case=False, na=False)
        ]

    # Rating filter: find whichever rating column exists in this dataset
    rating_col = next((c for c in RATING_COLS if c in result.columns), None)
    if rating_col:
        # pd.to_numeric converts the column to numbers; errors="coerce" turns
        # non-numeric values into NaN; .fillna(0) replaces NaN with 0
        numeric_ratings = pd.to_numeric(result[rating_col], errors="coerce").fillna(0)
        result = result[(numeric_ratings >= min_rating) & (numeric_ratings <= max_rating)]

    # Year filter: same pattern — convert to numeric then compare
    if "published_year" in result.columns:
        numeric_years = pd.to_numeric(result["published_year"], errors="coerce").fillna(0)
        result = result[(numeric_years >= year_from) & (numeric_years <= year_to)]

    # Author filter: substring match on the authors column
    if author:
        result = result[
            result["authors"].fillna("").str.contains(author, case=False, na=False)
        ]

    # Sort by rating descending so the best books come first
    if rating_col and rating_col in result.columns:
        result = result.sort_values(rating_col, ascending=False)

    # Return the top_n results as compact dicts
    return [_book_summary(row) for _, row in result.head(top_n).iterrows()]


# ==============================================================================
# TOOL 3 – recommend_books
# ==============================================================================

def recommend_books(
    title: str,
    recommender,
    top_n: int = 6,
) -> list:
    """
    Return the top_n most similar books to a given title using cosine similarity.

    GPT calls this when the user says something like:
      "I loved Dune, what should I read next?"

    Parameters:
      title      : the book title to base recommendations on
      recommender: the BookRecommender instance
      top_n      : how many similar books to return

    Returns:
      list of compact book dicts, sorted by similarity score (highest first)
    """
    results = recommender.recommend_by_title(title, top_n=top_n)
    return [_book_summary(r) for r in results]


# ==============================================================================
# TOOL 4 – get_book_info
# ==============================================================================

def get_book_info(title: str, df: pd.DataFrame) -> dict:
    """
    Return full details for one specific book by title.

    GPT calls this when the user asks:
      "Tell me more about The Alchemist"

    We do a case-insensitive substring match on the title column.
    If multiple books match, we return the first one.

    Parameters:
      title : the book title to look up (partial match is fine)
      df    : the full books DataFrame

    Returns:
      dict with all available book fields,
      or {"error": "..."} if no match is found
    """
    # Decide which column holds the display title
    title_col = "display_title" if "display_title" in df.columns else "title"

    # Build a boolean mask: True for rows where the title contains the query
    mask    = df[title_col].fillna("").str.lower().str.contains(title.lower(), na=False)
    matches = df[mask]

    if matches.empty:
        return {"error": f"No book found matching '{title}'"}

    # Take the first match and convert it to a dict
    row = matches.iloc[0].to_dict()

    return {
        "title":       row.get(title_col, ""),
        "authors":     row.get("authors", ""),
        "categories":  row.get("categories", ""),
        "rating":      _pick(row, RATING_COLS, "N/A"),
        "year":        str(row.get("published_year", "")),
        "pages":       str(row.get("num_pages", "")),
        "description": _pick(row, DESC_COLS, ""),
        "thumbnail":   _pick(row, THUMB_COLS, ""),
    }


# ==============================================================================
# TOOL SCHEMAS
# These JSON objects describe each tool to GPT.
# GPT reads them and decides WHEN and HOW to call each tool.
#
# Structure:
#   "name"         : must exactly match the Python function name
#   "description"  : tells GPT when to use this tool (very important!)
#   "input_schema" : defines the parameters GPT can send
#     "type"       : data type ("string", "number", "integer")
#     "description": explains what the parameter means
#     "default"    : value used if GPT doesn't specify this parameter
# ==============================================================================

TOOL_SCHEMAS = [
    {
        "name": "search_books",
        "description": (
            "Search the book catalogue with a free-text query using TF-IDF similarity. "
            "Use this when the user describes a topic, mood, theme, or genre in natural language "
            "and has not specified hard constraints like rating or year."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What the user is looking for, e.g. 'cooking for beginners' or 'sad romance'",
                },
                "top_n": {
                    "type": "integer",
                    "description": "How many results to return (default 8)",
                    "default": 8,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "filter_books",
        "description": (
            "Filter the book catalogue using structured criteria: genre, rating range, "
            "publication year range, or author name. "
            "Use when the user provides specific constraints like 'only 4-star books' "
            "or 'published after 2010' or 'Mystery genre'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "genre":      {"type": "string",  "description": "Genre or category keyword, e.g. 'Fiction' or 'Mystery'"},
                "min_rating": {"type": "number",  "description": "Minimum average rating (0 to 5)", "default": 0},
                "max_rating": {"type": "number",  "description": "Maximum average rating (0 to 5)", "default": 5},
                "year_from":  {"type": "integer", "description": "Earliest publication year, e.g. 2000", "default": 0},
                "year_to":    {"type": "integer", "description": "Latest publication year, e.g. 2023",   "default": 9999},
                "author":     {"type": "string",  "description": "Author name keyword, e.g. 'Rowling'"},
                "top_n":      {"type": "integer", "description": "Maximum number of results to return",   "default": 8},
            },
            "required": [],   # all parameters are optional
        },
    },
    {
        "name": "recommend_books",
        "description": (
            "Return books that are similar to a book title the user already knows. "
            "Use when the user says 'I liked X, what else should I read?' or "
            "'recommend something similar to X'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "The title of the book to base recommendations on",
                },
                "top_n": {
                    "type": "integer",
                    "description": "Number of similar books to return (default 6)",
                    "default": 6,
                },
            },
            "required": ["title"],
        },
    },
    {
        "name": "get_book_info",
        "description": (
            "Get complete information about one specific book by title. "
            "Use when the user asks 'tell me about X', 'what is X about', "
            "or 'who wrote X'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "The title of the book to look up (partial match is fine)",
                },
            },
            "required": ["title"],
        },
    },
]
