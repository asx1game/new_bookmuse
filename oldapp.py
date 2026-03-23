"""
app.py  –  BookMuse
===================
Gradio app with four tabs:
  1. Find similar books  (original TF-IDF)
  2. Explore by topic    (original TF-IDF)
  3. Ask AI              (RAG: retrieve context then GPT answers)
  4. Agent               (GPT decides which tools to call, multi-turn chat)
"""
from dotenv import load_dotenv
load_dotenv()
import json
import os

import openai
import gradio as gr
import pandas as pd

from src.preprocessing import load_books
from src.recommender import BookRecommender
from src.agent_tools import (
    TOOL_SCHEMAS,
    filter_books,
    get_book_info,
    recommend_books,
    search_books,
)

# ── Data & recommender ────────────────────────────────────────────────────────

CSV_PATH    = "data/books.csv"
df          = load_books(CSV_PATH)
recommender = BookRecommender.build(df)
titles      = sorted({t for t in df["display_title"].fillna("").astype(str).tolist() if t.strip()})

THUMB_COLS  = ["thumbnail", "thumbnail_url", "image_url", "img_url", "cover"]
RATING_COLS = ["average_rating", "rating", "ratings"]
DESC_COLS   = ["description", "desc", "summary"]


# ── Shared helpers ─────────────────────────────────────────────────────────────

def pick_first(row, cols, default=""):
    for c in cols:
        if c in row and pd.notna(row[c]) and str(row[c]).strip():
            return row[c]
    return default


def shorten(text, n=260):
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    return text if len(text) <= n else text[:n].rstrip() + "..."


def render_cards(results):
    if not results:
        return "<div class='hint'>No results. Try a different title or topic.</div>"
    html = ""
    for r in results:
        title      = r.get("display_title", "")
        authors    = r.get("authors", "")
        categories = r.get("categories", "")
        rating     = pick_first(r, RATING_COLS, "")
        desc       = pick_first(r, DESC_COLS, "")
        thumb      = pick_first(r, THUMB_COLS, "")
        sim        = r.get("similarity", 0.0)

        cover     = f"<img class='cover' src='{thumb}' />" if thumb else "<div class='cover placeholder'></div>"
        meta      = []
        if authors:      meta.append(f"<b>Authors</b>: {authors}")
        if categories:   meta.append(f"<b>Genres</b>: {categories}")
        if rating != "": meta.append(f"<b>Rating</b>: {rating}")
        meta_html = "<br/>".join(meta)

        html += f"""
        <div class="card">
          <div class="left">{cover}</div>
          <div class="right">
            <div class="top">
              <div class="title">{title}</div>
              <div class="badge">Match <b>{sim:.3f}</b></div>
            </div>
            <div class="meta">{meta_html}</div>
            <div class="desc">{shorten(desc)}</div>
          </div>
        </div>"""
    return html


# ── Tab 1 & 2 handlers ────────────────────────────────────────────────────────

def ui_by_title(title_query, top_n):
    return render_cards(recommender.recommend_by_title(title_query, top_n=int(top_n)))


def ui_by_topic(topic_query, top_n):
    return render_cards(recommender.recommend_by_text(topic_query, top_n=int(top_n)))


# ── Tab 3 – Ask AI (RAG) ──────────────────────────────────────────────────────

def build_context(question: str, top_k: int = 12) -> str:
    candidates = recommender.recommend_by_text(question, top_n=top_k)
    if not candidates:
        return "No relevant books found."
    lines = []
    for i, book in enumerate(candidates, 1):
        title   = book.get("display_title", "Unknown")
        authors = book.get("authors", "")
        cats    = book.get("categories", "")
        rating  = pick_first(book, RATING_COLS, "N/A")
        desc    = pick_first(book, DESC_COLS, "")
        lines.append(
            f"{i}. **{title}** by {authors}"
            + (f" [{cats}]" if cats else "")
            + (f" — Rating: {rating}" if rating != "N/A" else "")
            + (f"\n   {shorten(desc, 200)}" if desc else "")
        )
    return "\n".join(lines)


def ask_ai(question: str) -> str:
    question = question.strip()
    if not question:
        return "Please write a question first."
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return "OPENAI_API_KEY not set."

    context = build_context(question, top_k=12)
    client  = openai.OpenAI(api_key=api_key)
    msg = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=1024,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are BookMuse, a friendly book expert. "
                    "Use ONLY the books listed in the context. Do not invent titles."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    f"Relevant books:\n{context}\n\n"
                    "Answer based on these books."
                ),
            },
        ],
    )
    return msg.choices[0].message.content


# ── Tab 4 – Agent ─────────────────────────────────────────────────────────────

# OpenAI tool schemas wrap the same JSON in a "function" envelope
OPENAI_TOOLS = [
    {"type": "function", "function": schema}
    for schema in TOOL_SCHEMAS
]


def _dispatch_tool(tool_name: str, tool_input: dict) -> str:
    """
    Route a tool call from GPT to the correct Python function.
    Returns a JSON string — that is what GPT receives as the tool result.
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


def run_agent(user_message: str, history: list) -> tuple[str, list]:
    """
    Agentic loop — keeps calling GPT until it stops requesting tools.

    Step-by-step:
      1. Build the full conversation (history + new user message)
      2. Call GPT with the 4 tool schemas
      3. If GPT returns finish_reason == 'tool_calls':
           a. Run each requested tool
           b. Append tool results to the conversation
           c. Go back to step 2
      4. When finish_reason == 'stop', return GPT's final text reply
    """
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return "OPENAI_API_KEY not set.", history

    client = openai.OpenAI(api_key=api_key)

    system = (
        "You are BookMuse, an interactive book recommendation assistant. "
        "You have four tools: search_books, filter_books, recommend_books, get_book_info. "
        "Always use tools to fetch real data — never invent book titles or ratings. "
        "Be conversational, concise, and helpful."
    )

    # Rebuild the message list from Gradio history
    messages = [{"role": "system", "content": system}]
    for user_turn, assistant_turn in history:
        messages.append({"role": "user",      "content": user_turn})
        messages.append({"role": "assistant", "content": assistant_turn})
    messages.append({"role": "user", "content": user_message})

    # ── Agentic loop ──────────────────────────────────────────────────────────
    while True:
        response = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=1024,
            tools=OPENAI_TOOLS,
            messages=messages,
        )

        choice = response.choices[0]

        # GPT is done — return the text reply
        if choice.finish_reason == "stop":
            reply = choice.message.content or ""
            history = history + [[user_message, reply]]
            return "", history          # empty string clears the input box

        # GPT wants to call tools
        if choice.finish_reason == "tool_calls":

            # 1. Add GPT's response (with tool_calls) to messages
            messages.append(choice.message)

            # 2. Run every requested tool and collect results
            for tool_call in choice.message.tool_calls:
                tool_input = json.loads(tool_call.function.arguments)
                output     = _dispatch_tool(tool_call.function.name, tool_input)
                messages.append({
                    "role":         "tool",
                    "tool_call_id": tool_call.id,   # must match the request id
                    "content":      output,
                })

            # 3. Loop continues — send tool results back to GPT

        else:
            break   # unexpected finish reason

    return "Something went wrong. Please try again.", history


# ── CSS ───────────────────────────────────────────────────────────────────────

CSS = """
.gradio-container{max-width:1280px !important;width:96% !important;margin:0 auto !important}
:root{--bg:#0b1220;--panel:#111a2e;--text:#e6edf3;--muted:rgba(230,237,243,.75);--border:rgba(255,255,255,.10);--shadow:rgba(0,0,0,.35)}
body,.gradio-container{background:var(--bg) !important}
h1,h2,h3,p,label,span,.prose{color:var(--text) !important}
.card{display:flex;gap:14px;padding:14px;margin:12px 0;border:1px solid var(--border);border-radius:18px;background:var(--panel);box-shadow:0 14px 34px var(--shadow)}
.left{flex:0 0 92px}
.cover{width:92px;height:128px;object-fit:cover;border-radius:14px;border:1px solid rgba(255,255,255,.08)}
.cover.placeholder{background:rgba(255,255,255,.06)}
.right{flex:1}
.top{display:flex;align-items:center;justify-content:space-between;gap:10px}
.title{font-size:18px;font-weight:800;color:#fff;line-height:1.2}
.badge{font-size:12px;padding:6px 10px;border-radius:999px;background:rgba(125,211,252,.14);border:1px solid rgba(125,211,252,.22);color:var(--text);white-space:nowrap}
.meta{margin-top:8px;font-size:13px;color:rgba(230,237,243,.86);line-height:1.45}
.desc{margin-top:10px;font-size:13px;color:var(--muted);line-height:1.55}
.hint{color:var(--muted);padding:10px 0}
button{border-radius:14px !important;font-weight:700 !important}
"""


# ── Gradio UI ─────────────────────────────────────────────────────────────────

with gr.Blocks(theme=gr.themes.Soft(), css=CSS, title="BookMuse") as demo:
    gr.Markdown("# BookMuse\nDiscover your next read in seconds.")

    with gr.Tabs():

        # Tab 1 ────────────────────────────────────────────────────────────────
        with gr.Tab("Find similar books"):
            gr.Markdown("Choose a book you already know and we'll suggest similar picks.")
            with gr.Row():
                title_in = gr.Dropdown(choices=list(titles), label="Book title", allow_custom_value=True)
                topn1    = gr.Slider(1, 12, value=6, step=1, label="Results")
            btn1 = gr.Button("Recommend", variant="primary")
            out1 = gr.HTML()
            btn1.click(ui_by_title, inputs=[title_in, topn1], outputs=out1)

        # Tab 2 ────────────────────────────────────────────────────────────────
        with gr.Tab("Explore by topic"):
            gr.Markdown("Describe what you're in the mood for.")
            with gr.Row():
                topic_in = gr.Textbox(label="Topic", placeholder="Describe your next read...", lines=2)
                topn2    = gr.Slider(1, 12, value=6, step=1, label="Results")
            btn2 = gr.Button("Explore", variant="primary")
            out2 = gr.HTML()
            btn2.click(ui_by_topic, inputs=[topic_in, topn2], outputs=out2)

        # Tab 3 ────────────────────────────────────────────────────────────────
        with gr.Tab("Ask AI 🤖"):
            gr.Markdown(
                "Ask anything in natural language. The AI searches the database and answers.\n\n"
                "_Examples: \"Best books to learn cooking\", \"Thriller with a female detective\"_"
            )
            question_in = gr.Textbox(
                label="Your question",
                placeholder="What books would you recommend to learn cooking?",
                lines=3,
            )
            btn3 = gr.Button("Ask", variant="primary")
            out3 = gr.Textbox(label="Answer", lines=10, interactive=False)
            btn3.click(ask_ai, inputs=[question_in], outputs=out3)

        # Tab 4 – Agent ────────────────────────────────────────────────────────
        with gr.Tab("Agent 🔧"):
            gr.Markdown(
                "Chat with an AI agent that chooses which tools to use automatically.\n\n"
                "**Available tools:** `search_books` · `filter_books` · `recommend_books` · `get_book_info`\n\n"
                "_Try: \"Find mystery books rated above 4 stars after 2005\"_"
            )

            chatbot  = gr.Chatbot(label="BookMuse Agent", height=460)
            chat_in  = gr.Textbox(
                label="Your message",
                placeholder="Ask the agent anything about books...",
                lines=2,
            )
            with gr.Row():
                send_btn  = gr.Button("Send", variant="primary")
                clear_btn = gr.Button("Clear")

            state = gr.State([])   # stores chat history as [[user, assistant], ...]

            send_btn.click(
                fn=run_agent,
                inputs=[chat_in, state],
                outputs=[chat_in, chatbot],
            ).then(
                fn=lambda h: h,
                inputs=[chatbot],
                outputs=[state],
            )

            clear_btn.click(fn=lambda: ([], []), outputs=[chatbot, state])

demo.launch()