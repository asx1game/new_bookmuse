"""
app.py  ─  BookMuse  📚
========================
This is the ENTRY POINT of the entire project.
Running `python app.py` starts a local website on your computer.

HOW IT WORKS (big picture):
  1. We load the book dataset (a CSV file) into memory as a pandas DataFrame
  2. We build a TF-IDF model — this turns every book into a list of numbers
     so that we can measure how "similar" two books are mathematically
  3. We define Python functions for each button/action in the UI
  4. Gradio reads those functions and automatically creates the web interface

THE FOUR TABS:
  Tab 1 – Find similar books  →  pick a title you love → get books like it
  Tab 2 – Explore by topic    →  describe a mood/theme → matching books
  Tab 3 – Ask AI 🤖           →  ask a question in plain English → GPT answers
                                  using our own dataset as the knowledge base (RAG)
  Tab 4 – Agent 🔧            →  select genre, rating, year, mood with buttons
                                  → the agent decides which tools to call → results
"""

# ── Load secret keys from the .env file ───────────────────────────────────────
# The .env file contains OPENAI_API_KEY=sk-...
# We never hardcode keys in source code — that would be a security risk.
# load_dotenv() reads .env and injects everything into os.environ automatically.
from dotenv import load_dotenv
load_dotenv()

# ── Standard Python libraries ─────────────────────────────────────────────────
import json   # converts Python dicts <-> JSON strings  (the AI tools speak JSON)
import os     # lets us read environment variables like os.environ["OPENAI_API_KEY"]

# ── Third-party libraries (installed via pip) ─────────────────────────────────
import openai          # official Python client for GPT (Chat Completions API)
import gradio as gr    # turns Python functions into a web UI with zero HTML/JS
import pandas as pd    # spreadsheet-like data manipulation

# ── Our own modules (src/ folder) ─────────────────────────────────────────────
from src.preprocessing import load_books          # reads & cleans the CSV
from src.recommender   import BookRecommender     # TF-IDF + cosine similarity
from src.agent_tools   import (
    TOOL_SCHEMAS,     # JSON descriptions of the 4 tools (sent to GPT)
    filter_books,     # filters by genre / rating / year / author
    get_book_info,    # returns full details for one specific book
    recommend_books,  # finds books similar to a given title
    search_books,     # free-text TF-IDF search
)


# ==============================================================================
# SECTION 1 – Load data and build the recommender
# This runs ONCE when the app starts. It can take a few seconds.
# ==============================================================================

CSV_PATH = "data/books.csv"

# load_books() reads the CSV, fills missing values, cleans text, and
# creates a "combined_text" column = title + authors + categories + description
df = load_books(CSV_PATH)

# BookRecommender.build():
#   - Takes the combined_text for every book
#   - Applies TF-IDF: converts words -> numbers (higher = more important/unique word)
#   - Stores a big matrix of shape (num_books x num_unique_words) in memory
#   -> After this, comparing two books = computing cosine similarity of their rows
recommender = BookRecommender.build(df)

# Sorted list of all unique titles — used for the dropdown in Tab 1
titles = sorted({
    t for t in df["display_title"].fillna("").astype(str).tolist()
    if t.strip()
})

# Column name fallbacks — different CSVs may call the same thing differently
THUMB_COLS  = ["thumbnail", "thumbnail_url", "image_url", "img_url", "cover"]
RATING_COLS = ["average_rating", "rating", "ratings"]
DESC_COLS   = ["description", "desc", "summary"]

# Unique genres available in the dataset (used for the genre dropdown in Tab 4)
# We split on commas/semicolons because one book can have multiple genres
_all_genres = (
    df["categories"]
    .fillna("")
    .str.split(r"[,;/]")
    .explode()
    .str.strip()
    .str.title()
)
GENRES = ["Any"] + sorted({g for g in _all_genres if len(g) > 2})


# ==============================================================================
# SECTION 2 – Shared helper functions
# Small utilities used by multiple tabs
# ==============================================================================

def pick_first(row, cols, default=""):
    """
    Try each column name in `cols` until we find one that exists and is not empty.
    This handles datasets where the same field has different column names.
    Example: pick_first(row, ["thumbnail", "image_url"]) tries "thumbnail" first.
    """
    for c in cols:
        if c in row and pd.notna(row[c]) and str(row[c]).strip():
            return row[c]
    return default


def shorten(text, n=260):
    """
    Truncate a string to at most n characters.
    Adds "..." if the text was cut. Used so descriptions don't overflow the card.
    """
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    return text if len(text) <= n else text[:n].rstrip() + "..."


def render_cards(results):
    """
    Takes a list of book dicts and returns an HTML string showing one card per book.
    Gradio's gr.HTML() component renders this HTML directly in the browser.

    Each card shows:
      - Book cover image (or a placeholder box if no image is available)
      - Title
      - Match score (cosine similarity — how close it is to the query)
      - Authors, genre, rating
      - Short description excerpt
    """
    if not results:
        return "<div class='no-results'>No results found — try a different search.</div>"

    html = ""
    for r in results:
        title      = r.get("display_title", r.get("title", ""))
        authors    = r.get("authors", "")
        categories = r.get("categories", "")
        rating     = pick_first(r, RATING_COLS, "")
        desc       = pick_first(r, DESC_COLS, "")
        thumb      = pick_first(r, THUMB_COLS, "")
        sim        = r.get("similarity", 0.0)

        # If we have a thumbnail URL, show an <img> tag; otherwise a styled placeholder
        cover = (
            f"<img class='cover' src='{thumb}' onerror=\"this.parentNode.innerHTML='<div class=cover-placeholder>&#128214;</div>'\" />"
            if thumb else
            "<div class='cover-placeholder'>&#128214;</div>"
        )

        # Build the metadata chips (only include fields that exist)
        meta_parts = []
        if authors:      meta_parts.append(f"<span class='meta-item'>&#9998; {authors}</span>")
        if categories:   meta_parts.append(f"<span class='meta-item'>&#127991; {categories}</span>")
        if rating != "": meta_parts.append(f"<span class='meta-item meta-rating'>&#9733; {rating}</span>")
        meta_html = "".join(meta_parts)

        # Format the similarity score as a percentage badge
        sim_pct = f"{sim * 100:.1f}% match" if sim > 0 else ""
        badge   = f"<div class='badge'>{sim_pct}</div>" if sim_pct else ""

        html += f"""
        <div class="card">
          <div class="card-cover">{cover}</div>
          <div class="card-body">
            <div class="card-header">
              <div class="card-title">{title}</div>
              {badge}
            </div>
            <div class="card-meta">{meta_html}</div>
            <div class="card-desc">{shorten(desc)}</div>
          </div>
        </div>
        """
    return html


# ==============================================================================
# SECTION 3 – Tab 1: Find similar books
# ==============================================================================

def ui_by_title(title_query, top_n):
    """
    Called when the user clicks "Find Similar Books" in Tab 1.
    - title_query : the book title chosen from the dropdown
    - top_n       : how many results to show (from the slider)

    Internally:
      1. We look up the chosen book's TF-IDF vector in the matrix
      2. We compute cosine similarity against every other book
      3. We return the top_n highest-scoring books
    """
    results = recommender.recommend_by_title(title_query, top_n=int(top_n))
    return render_cards(results)


# ==============================================================================
# SECTION 4 – Tab 2: Explore by topic
# ==============================================================================

def ui_by_topic(topic_query, top_n):
    """
    Called when the user clicks "Explore Books" in Tab 2.
    - topic_query : free text the user typed (e.g. "dark fantasy with magic")
    - top_n       : how many results to show

    Internally:
      1. The user's text is passed through the SAME TF-IDF vectorizer
         (so it speaks the same "language" as the book vectors)
      2. We compute cosine similarity between the query vector and every book
      3. We return the top_n most similar books
    """
    results = recommender.recommend_by_text(topic_query, top_n=int(top_n))
    return render_cards(results)


# ==============================================================================
# SECTION 5 – Tab 3: Ask AI  (RAG = Retrieval-Augmented Generation)
# ==============================================================================

def build_rag_context(question: str, top_k: int = 12) -> str:
    """
    RAG Step 1 — RETRIEVE the most relevant books for the question.

    We use our TF-IDF recommender to find the top_k books whose text
    is most similar to the user's question. These books become the "context"
    that we pass to GPT — so GPT never has to guess or make up titles.

    Returns a numbered list of books as a plain-text string.
    """
    candidates = recommender.recommend_by_text(question, top_n=top_k)
    if not candidates:
        return "No relevant books found in the database."

    lines = []
    for i, book in enumerate(candidates, 1):
        title   = book.get("display_title", "Unknown")
        authors = book.get("authors", "")
        cats    = book.get("categories", "")
        rating  = pick_first(book, RATING_COLS, "N/A")
        desc    = pick_first(book, DESC_COLS, "")
        lines.append(
            f"{i}. {title} by {authors}"
            + (f" [{cats}]" if cats else "")
            + (f" — Rating: {rating}" if rating != "N/A" else "")
            + (f"\n   {shorten(desc, 200)}" if desc else "")
        )
    return "\n".join(lines)


def ask_ai(question: str) -> str:
    """
    RAG Step 2 — GENERATE: send the retrieved context + user question to GPT.

    Flow:
      user question
        -> retrieve top 12 relevant books from our dataset (build_rag_context)
        -> send [system prompt] + [book list] + [question] to GPT-4o
        -> GPT reads ONLY those books and writes a helpful answer
        -> we display GPT's answer in the text box

    This means GPT is grounded in our real data and cannot invent fake books.
    """
    question = question.strip()
    if not question:
        return "Please type a question first."

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return "OPENAI_API_KEY not found in your .env file."

    context = build_rag_context(question, top_k=12)
    client  = openai.OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=1024,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are BookMuse, a warm and knowledgeable book advisor. "
                    "Use ONLY the books provided in the context below. "
                    "Never invent titles, authors, or ratings. "
                    "If none of the books fit the question well, say so honestly."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    f"Books available:\n{context}\n\n"
                    "Please answer the question based on these books."
                ),
            },
        ],
    )
    return response.choices[0].message.content


# ==============================================================================
# SECTION 6 – Tab 4: Agent  (Tool-Calling / Agentic AI)
# ==============================================================================
#
# HOW THE AGENT WORKS:
#   Instead of typing a message, the user sets parameters using Gradio controls
#   (dropdowns, sliders). When they click Run, we automatically build a
#   natural-language sentence from those parameters and send it to GPT.
#
#   GPT then decides WHICH tool to call and WHAT arguments to pass.
#   We run the tool, send the result back to GPT, and GPT writes a nice answer.
#
#   The 4 tools available to the agent:
#     - search_books     : TF-IDF search by any text query
#     - filter_books     : filter by genre, rating range, year range, author
#     - recommend_books  : "books similar to X" using cosine similarity
#     - get_book_info    : full details for one specific title

# Wrap tool schemas in the OpenAI "function" envelope.
# Our TOOL_SCHEMAS are in a generic format; OpenAI requires {"type":"function", ...}
OPENAI_TOOLS = [
    {"type": "function", "function": schema}
    for schema in TOOL_SCHEMAS
]


def _dispatch_tool(tool_name: str, tool_input: dict) -> str:
    """
    When GPT says "call filter_books with {genre:'Mystery', min_rating:4}",
    this function actually runs the correct Python function and returns the result.

    Returns a JSON string because the OpenAI API requires tool results as JSON.
    """
    if tool_name == "search_books":
        result = search_books(
            query=tool_input["query"],
            df=df,
            recommender=recommender,
            top_n=tool_input.get("top_n", 8),
        )
    elif tool_name == "filter_books":
        result = filter_books(
            df=df,
            genre=tool_input.get("genre", ""),
            min_rating=tool_input.get("min_rating", 0.0),
            max_rating=tool_input.get("max_rating", 5.0),
            year_from=tool_input.get("year_from", 0),
            year_to=tool_input.get("year_to", 9999),
            author=tool_input.get("author", ""),
            top_n=tool_input.get("top_n", 8),
        )
    elif tool_name == "recommend_books":
        result = recommend_books(
            title=tool_input["title"],
            recommender=recommender,
            top_n=tool_input.get("top_n", 6),
        )
    elif tool_name == "get_book_info":
        result = get_book_info(title=tool_input["title"], df=df)
    else:
        result = {"error": f"Unknown tool: {tool_name}"}

    return json.dumps(result, ensure_ascii=False)


def build_agent_message(
    mode: str,
    genre: str,
    min_rating: float,
    year_from: int,
    year_to: int,
    mood: str,
    similar_title: str,
    lookup_title: str,
) -> str:
    """
    Converts the Gradio UI values into a natural-language sentence for GPT.

    This is the key idea: instead of the user typing "Find mystery books
    rated above 4 published after 2000", we BUILD that sentence automatically
    from the dropdown/slider values. GPT reads the sentence and calls the right tool.

    Example outputs:
      "Find me Mystery books rated at least 4.0, published between 2000 and 2020."
      "Recommend books similar to 'Dune'."
      "Give me full details about the book 'The Alchemist'."
    """
    if mode == "Filter by genre / rating / year":
        parts = ["Find me"]
        if genre and genre != "Any":
            parts.append(genre)
        parts.append("books")
        if min_rating > 0:
            parts.append(f"rated at least {min_rating:.1f}")
        if year_from > 1800 or year_to < 2024:
            parts.append(f"published between {year_from} and {year_to}")
        return " ".join(parts) + "."

    elif mode == "Search by mood / theme":
        query = mood.strip() if mood.strip() else "interesting books"
        return f"Find books about: {query}."

    elif mode == "Books similar to a title":
        title = similar_title.strip() if similar_title and similar_title.strip() else "a popular book"
        return f"Recommend books similar to '{title}'."

    elif mode == "Look up a specific book":
        title = lookup_title.strip() if lookup_title and lookup_title.strip() else "an interesting book"
        return f"Give me full details about the book '{title}'."

    return "Show me some great book recommendations."


def run_agent(mode, genre, min_rating, year_from, year_to,
              mood, similar_title, lookup_title, history):
    """
    Main agent function — called when the user clicks "Run Agent".

    Steps:
      1. Build a natural-language message from the UI controls
      2. Add it to the conversation history
      3. Send the full conversation + tool schemas to GPT-4o
      4. If GPT calls a tool -> run the tool -> send result back to GPT -> repeat
      5. When GPT stops calling tools -> display its final text answer
      6. Return the updated history so Gradio shows the conversation as bubbles
    """
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return history + [["(system)", "OPENAI_API_KEY not set in .env file."]]

    user_message = build_agent_message(
        mode, genre, min_rating, year_from, year_to,
        mood, similar_title, lookup_title
    )

    client = openai.OpenAI(api_key=api_key)

    system_prompt = (
        "You are BookMuse, a smart and friendly book recommendation assistant. "
        "You have four tools: search_books, filter_books, recommend_books, get_book_info. "
        "Always call a tool to get real data — never invent book titles, authors, or ratings. "
        "After getting the tool results, summarise the top books in a friendly, readable way. "
        "For each book give: title, author, rating, and one sentence about it."
    )

    # Build the full message list: [system] + [all previous turns] + [new user message]
    messages = [{"role": "system", "content": system_prompt}]
    for user_turn, assistant_turn in history:
        messages.append({"role": "user",      "content": user_turn})
        messages.append({"role": "assistant", "content": assistant_turn})
    messages.append({"role": "user", "content": user_message})

    # Agentic loop: GPT may call multiple tools before giving a final answer.
    # We keep looping until GPT says finish_reason == "stop".
    while True:
        response = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=1024,
            tools=OPENAI_TOOLS,
            messages=messages,
        )

        choice = response.choices[0]

        # GPT is done — return its text answer
        if choice.finish_reason == "stop":
            reply = choice.message.content or ""
            return history + [[user_message, reply]]

        # GPT wants to call one or more tools
        if choice.finish_reason == "tool_calls":
            messages.append(choice.message)   # record GPT's tool-call request

            for tool_call in choice.message.tool_calls:
                tool_input = json.loads(tool_call.function.arguments)
                output     = _dispatch_tool(tool_call.function.name, tool_input)
                # Send the tool result back so GPT can continue reasoning
                messages.append({
                    "role":         "tool",
                    "tool_call_id": tool_call.id,
                    "content":      output,
                })
            # Loop -> GPT reads the result and either calls another tool or stops
        else:
            break

    return history + [["(error)", "Something went wrong. Please try again."]]


# ==============================================================================
# SECTION 7 – CSS  (visual design)
# Inspired by Claude.ai: warm off-white background, coral/orange accents,
# Lora serif for headings, DM Sans for body text, clean card layout
# ==============================================================================

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Lora:wght@400;600;700&family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,400&display=swap');

:root {
  --bg:          #f9f6f1;
  --surface:     #ffffff;
  --surface2:    #f2ede6;
  --border:      #e5ddd3;
  --text:        #1c1917;
  --muted:       #78716c;
  --accent:      #c2622a;
  --accent-lt:   #e07840;
  --accent-bg:   #fdf0e8;
  --shadow-sm:   0 1px 3px rgba(28,25,23,.07);
  --shadow-md:   0 4px 16px rgba(28,25,23,.09);
  --r:           12px;
  --r-sm:        8px;
}

body, .gradio-container {
  background: var(--bg) !important;
  font-family: 'DM Sans', sans-serif !important;
  color: var(--text) !important;
}
.gradio-container {
  max-width: 1080px !important;
  width: 95% !important;
  margin: 0 auto !important;
}

/* ── Header ── */
.bm-header {
  text-align: center;
  padding: 36px 0 24px;
  border-bottom: 1px solid var(--border);
  margin-bottom: 4px;
}
.bm-logo {
  font-family: 'Lora', Georgia, serif;
  font-size: 36px;
  font-weight: 700;
  color: var(--text);
  letter-spacing: -.5px;
}
.bm-logo span { color: var(--accent); }
.bm-tagline {
  font-size: 14px;
  color: var(--muted);
  margin-top: 5px;
}

/* ── Labels ── */
label, .gr-label {
  font-family: 'DM Sans', sans-serif !important;
  font-size: 12px !important;
  font-weight: 600 !important;
  letter-spacing: .04em !important;
  text-transform: uppercase !important;
  color: var(--muted) !important;
}

/* ── Inputs ── */
input, textarea, select {
  background: var(--surface) !important;
  border: 1.5px solid var(--border) !important;
  border-radius: var(--r-sm) !important;
  color: var(--text) !important;
  font-family: 'DM Sans', sans-serif !important;
  font-size: 14px !important;
}
input:focus, textarea:focus {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px rgba(194,98,42,.12) !important;
  outline: none !important;
}

/* ── Primary button ── */
.gr-button-primary, button.primary {
  background: var(--accent) !important;
  color: #fff !important;
  border: none !important;
  border-radius: var(--r-sm) !important;
  font-family: 'DM Sans', sans-serif !important;
  font-weight: 600 !important;
  font-size: 14px !important;
  padding: 10px 22px !important;
  transition: background .18s, transform .1s !important;
  box-shadow: var(--shadow-sm) !important;
  cursor: pointer !important;
}
.gr-button-primary:hover, button.primary:hover {
  background: var(--accent-lt) !important;
}
.gr-button-primary:active, button.primary:active {
  transform: scale(.98) !important;
}

/* ── Secondary button ── */
.gr-button-secondary, button.secondary {
  background: var(--surface) !important;
  color: var(--text) !important;
  border: 1.5px solid var(--border) !important;
  border-radius: var(--r-sm) !important;
  font-family: 'DM Sans', sans-serif !important;
  font-weight: 500 !important;
  font-size: 14px !important;
  cursor: pointer !important;
}
.gr-button-secondary:hover, button.secondary:hover {
  border-color: var(--accent) !important;
  color: var(--accent) !important;
}

/* ── Tab nav ── */
.tab-nav button {
  font-family: 'DM Sans', sans-serif !important;
  font-weight: 500 !important;
  font-size: 14px !important;
  color: var(--muted) !important;
  border-radius: var(--r-sm) !important;
  padding: 8px 14px !important;
}
.tab-nav button.selected {
  color: var(--accent) !important;
  background: var(--accent-bg) !important;
  font-weight: 600 !important;
}

/* ── Tab description ── */
.tab-desc {
  font-size: 13.5px;
  color: var(--muted);
  padding: 10px 0 16px;
  line-height: 1.65;
  border-bottom: 1px solid var(--border);
  margin-bottom: 18px;
}

/* ── Agent mode radio pills ── */
.agent-mode .wrap { display: flex !important; flex-wrap: wrap !important; gap: 8px !important; }
.agent-mode label {
  background: var(--surface) !important;
  border: 1.5px solid var(--border) !important;
  border-radius: 999px !important;
  padding: 7px 18px !important;
  cursor: pointer !important;
  text-transform: none !important;
  font-size: 13px !important;
  font-weight: 500 !important;
  color: var(--text) !important;
  transition: all .18s !important;
  letter-spacing: 0 !important;
}
.agent-mode label:has(input:checked) {
  background: var(--accent-bg) !important;
  border-color: var(--accent) !important;
  color: var(--accent) !important;
  font-weight: 600 !important;
}

/* ── Chatbot ── */
.chatbot, .chatbot > div {
  background: var(--surface) !important;
  border: 1.5px solid var(--border) !important;
  border-radius: var(--r) !important;
  font-family: 'DM Sans', sans-serif !important;
  font-size: 14px !important;
}

/* ── Book cards ── */
.card {
  display: flex;
  gap: 16px;
  padding: 16px;
  margin: 10px 0;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--r);
  box-shadow: var(--shadow-sm);
  transition: box-shadow .2s, transform .18s;
}
.card:hover {
  box-shadow: var(--shadow-md);
  transform: translateY(-2px);
}
.card-cover { flex: 0 0 86px; }
.cover {
  width: 86px; height: 122px;
  object-fit: cover;
  border-radius: 7px;
  box-shadow: var(--shadow-md);
  display: block;
}
.cover-placeholder {
  width: 86px; height: 122px;
  background: var(--surface2);
  border-radius: 7px;
  display: flex; align-items: center; justify-content: center;
  font-size: 28px; color: var(--muted);
  border: 1px solid var(--border);
}
.card-body  { flex: 1; min-width: 0; }
.card-header {
  display: flex; align-items: flex-start;
  justify-content: space-between; gap: 10px;
  margin-bottom: 8px;
}
.card-title {
  font-family: 'Lora', serif;
  font-size: 16px; font-weight: 600;
  color: var(--text); line-height: 1.3;
}
.badge {
  flex-shrink: 0;
  font-size: 11px; font-weight: 600;
  padding: 3px 10px; border-radius: 999px;
  background: var(--accent-bg);
  border: 1px solid rgba(194,98,42,.22);
  color: var(--accent); white-space: nowrap;
}
.card-meta  { display: flex; flex-wrap: wrap; gap: 5px; margin-bottom: 8px; }
.meta-item  {
  font-size: 12px; color: var(--muted);
  background: var(--surface2);
  padding: 3px 9px; border-radius: 999px;
  border: 1px solid var(--border);
}
.meta-rating {
  color: var(--accent);
  border-color: rgba(194,98,42,.2);
  background: var(--accent-bg);
  font-weight: 600;
}
.card-desc  { font-size: 13px; color: var(--muted); line-height: 1.6; }

/* ── No-results ── */
.no-results {
  text-align: center; padding: 40px 20px;
  color: var(--muted); font-size: 15px;
  background: var(--surface);
  border-radius: var(--r);
  border: 1px dashed var(--border);
  margin-top: 16px;
}
"""


# ==============================================================================
# SECTION 8 – Build the Gradio UI
# gr.Blocks lets us arrange components freely using rows and columns.
# ==============================================================================

AGENT_MODES = [
    "Filter by genre / rating / year",
    "Search by mood / theme",
    "Books similar to a title",
    "Look up a specific book",
]

with gr.Blocks(theme=gr.themes.Base(), css=CSS, title="BookMuse") as demo:

    # ── Header ─────────────────────────────────────────────────────────────────
    gr.HTML("""
    <div class="bm-header">
      <div class="bm-logo">Book<span>Muse</span></div>
      <div class="bm-tagline">Content-based book recommendations powered by TF-IDF &amp; GPT</div>
    </div>
    """)

    with gr.Tabs():

        # ── TAB 1: Find similar books ──────────────────────────────────────────
        with gr.Tab("📚 Similar Books"):
            gr.HTML("<div class='tab-desc'>Choose a book you already love. We find its TF-IDF vector and return the books with the highest cosine similarity score.</div>")
            with gr.Row():
                with gr.Column(scale=3):
                    title_in = gr.Dropdown(
                        choices=list(titles),
                        label="Book title",
                        allow_custom_value=True,
                        info="Start typing to filter the list",
                    )
                with gr.Column(scale=1):
                    topn1 = gr.Slider(1, 12, value=6, step=1, label="Number of results")
            btn1 = gr.Button("Find Similar Books", variant="primary")
            out1 = gr.HTML()
            btn1.click(fn=ui_by_title, inputs=[title_in, topn1], outputs=out1)

        # ── TAB 2: Explore by topic ────────────────────────────────────────────
        with gr.Tab("🔭 Explore by Topic"):
            gr.HTML("<div class='tab-desc'>Describe what you are in the mood for. Your text is converted into a TF-IDF vector and matched against every book in the database.</div>")
            with gr.Row():
                with gr.Column(scale=3):
                    topic_in = gr.Textbox(
                        label="Describe your next read",
                        placeholder="e.g. dark fantasy with political intrigue and magic systems...",
                        lines=2,
                    )
                with gr.Column(scale=1):
                    topn2 = gr.Slider(1, 12, value=6, step=1, label="Number of results")
            btn2 = gr.Button("Explore Books", variant="primary")
            out2 = gr.HTML()
            btn2.click(fn=ui_by_topic, inputs=[topic_in, topn2], outputs=out2)

        # ── TAB 3: Ask AI ──────────────────────────────────────────────────────
        with gr.Tab("🤖 Ask AI"):
            gr.HTML("<div class='tab-desc'>Ask anything in natural language. We retrieve the 12 most relevant books from our dataset and send them to GPT-4o as context — the AI can only reference real books from our data.</div>")
            question_in = gr.Textbox(
                label="Your question",
                placeholder="e.g. What are the best books to understand human psychology?",
                lines=3,
            )
            btn3 = gr.Button("Ask", variant="primary")
            out3 = gr.Textbox(label="AI Answer", lines=12, interactive=False)
            btn3.click(fn=ask_ai, inputs=[question_in], outputs=out3)

        # ── TAB 4: Agent ───────────────────────────────────────────────────────
        with gr.Tab("🔧 Agent"):
            gr.HTML("""
            <div class='tab-desc'>
              Select your search criteria using the controls below — no typing needed.
              The agent translates your selections into a query, picks the right tool
              automatically, and returns results from our real dataset.
            </div>
            """)

            # Radio buttons to choose the mode
            # Each mode shows a different set of controls below
            agent_mode = gr.Radio(
                choices=AGENT_MODES,
                value=AGENT_MODES[0],
                label="What do you want to do?",
                elem_classes=["agent-mode"],
            )

            # Panel A — shown when mode = "Filter by genre / rating / year"
            with gr.Group(visible=True) as panel_filter:
                gr.HTML("<div style='font-size:13px;color:var(--muted);padding:2px 0 10px'>Leave any filter at its default value to ignore it.</div>")
                with gr.Row():
                    genre_dd   = gr.Dropdown(choices=GENRES, value="Any", label="Genre")
                    min_rating = gr.Slider(0.0, 5.0, value=0.0, step=0.1, label="Minimum rating (0 = any)")
                with gr.Row():
                    year_from  = gr.Slider(1800, 2024, value=1900, step=1, label="Published from year")
                    year_to    = gr.Slider(1800, 2024, value=2024, step=1, label="Published until year")

            # Panel B — shown when mode = "Search by mood / theme"
            with gr.Group(visible=False) as panel_mood:
                mood_in = gr.Textbox(
                    label="Describe the mood or theme",
                    placeholder="e.g. coming-of-age story set in space with found family...",
                    lines=2,
                )

            # Panel C — shown when mode = "Books similar to a title"
            with gr.Group(visible=False) as panel_similar:
                similar_in = gr.Dropdown(
                    choices=list(titles),
                    label="Base recommendations on this book",
                    allow_custom_value=True,
                    info="Start typing to filter the list",
                )

            # Panel D — shown when mode = "Look up a specific book"
            with gr.Group(visible=False) as panel_lookup:
                lookup_in = gr.Dropdown(
                    choices=list(titles),
                    label="Book to look up",
                    allow_custom_value=True,
                    info="Start typing to filter the list",
                )

            # When the user picks a different mode, hide all panels and show the right one
            def update_panels(mode):
                """
                Returns four gr.update() calls — one per panel.
                Gradio automatically shows/hides each Group based on the visible= value.
                """
                return (
                    gr.update(visible=(mode == AGENT_MODES[0])),  # filter panel
                    gr.update(visible=(mode == AGENT_MODES[1])),  # mood panel
                    gr.update(visible=(mode == AGENT_MODES[2])),  # similar panel
                    gr.update(visible=(mode == AGENT_MODES[3])),  # lookup panel
                )

            agent_mode.change(
                fn=update_panels,
                inputs=[agent_mode],
                outputs=[panel_filter, panel_mood, panel_similar, panel_lookup],
            )

            # Action buttons
            with gr.Row():
                agent_btn   = gr.Button("Run Agent", variant="primary")
                agent_clear = gr.Button("Clear conversation", variant="secondary")

            # Chatbot shows the conversation as speech bubbles
            agent_chat = gr.Chatbot(
                label="Agent conversation",
                height=420,
                elem_classes=["chatbot"],
            )

            # gr.State stores the chat history between clicks.
            # Without State, Gradio would forget previous messages on every click.
            agent_state = gr.State([])

            # Wire up Run Agent button:
            # inputs = all controls + the hidden state
            # outputs = the chatbot display
            agent_btn.click(
                fn=run_agent,
                inputs=[
                    agent_mode, genre_dd, min_rating,
                    year_from, year_to,
                    mood_in, similar_in, lookup_in,
                    agent_state,
                ],
                outputs=[agent_chat],
            ).then(
                # Sync the State after the chatbot updates so the next call
                # includes the full conversation history
                fn=lambda h: h,
                inputs=[agent_chat],
                outputs=[agent_state],
            )

            # Clear button resets both the display and the hidden history
            agent_clear.click(
                fn=lambda: ([], []),
                outputs=[agent_chat, agent_state],
            )


# ── Launch the app ─────────────────────────────────────────────────────────────
# Gradio starts a local web server and opens the browser automatically.
# Set share=True if you want a temporary public URL via ngrok.
demo.launch()
