"""
src/__init__.py
===============
This file marks the `src` folder as a Python "package".

Without this file, Python would not allow you to write:
  from src.preprocessing import load_books

With this file present (even if empty), Python knows that `src` is a package
and allows importing from it using dot notation.

The src/ package contains:
  preprocessing.py  : loads and cleans the CSV dataset
  recommender.py    : TF-IDF vectorization + cosine similarity recommendations
  agent_tools.py    : the 4 tools the AI agent can call + their JSON schemas
"""
