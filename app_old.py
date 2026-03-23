import gradio as gr
import pandas as pd

from src.preprocessing import load_books
from src.recommender import BookRecommender

CSV_PATH = "data/books.csv"

df = load_books(CSV_PATH)
recommender = BookRecommender.build(df)

titles = sorted({t for t in df["display_title"].fillna("").astype(str).tolist() if t.strip()})

THUMB_COLS = ["thumbnail", "thumbnail_url", "image_url", "img_url", "cover"]
RATING_COLS = ["average_rating", "rating", "ratings"]
DESC_COLS = ["description", "desc", "summary"]


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
        title = r.get("display_title", "")
        authors = r.get("authors", "")
        categories = r.get("categories", "")
        rating = pick_first(r, RATING_COLS, "")
        desc = pick_first(r, DESC_COLS, "")
        thumb = pick_first(r, THUMB_COLS, "")
        sim = r.get("similarity", 0.0)

        cover = f"<img class='cover' src='{thumb}' />" if thumb else "<div class='cover placeholder'></div>"

        meta = []
        if authors:
            meta.append(f"<b>Authors</b>: {authors}")
        if categories:
            meta.append(f"<b>Genres</b>: {categories}")
        if rating != "":
            meta.append(f"<b>Rating</b>: {rating}")
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
        </div>
        """
    return html


def ui_by_title(title_query, top_n):
    return render_cards(recommender.recommend_by_title(title_query, top_n=int(top_n)))


def ui_by_topic(topic_query, top_n):
    return render_cards(recommender.recommend_by_text(topic_query, top_n=int(top_n)))


CSS = """
.gradio-container { 
  max-width: 1280px !important; 
  width: 96% !important;
  margin: 0 auto !important;
}

:root{
  --bg: #0b1220;
  --panel: #111a2e;
  --text: #e6edf3;
  --muted: rgba(230,237,243,.75);
  --border: rgba(255,255,255,.10);
  --shadow: rgba(0,0,0,.35);
  --accent: #7dd3fc;
}

body, .gradio-container { background: var(--bg) !important; }
h1, h2, h3, p, label, span, .prose { color: var(--text) !important; }

.card{
  display:flex; gap:14px;
  padding:14px; margin:12px 0;
  border:1px solid var(--border);
  border-radius:18px;
  background: var(--panel);
  box-shadow:0 14px 34px var(--shadow);
}

.left{ flex:0 0 92px; }
.cover{
  width:92px; height:128px;
  object-fit:cover;
  border-radius:14px;
  border:1px solid rgba(255,255,255,.08);
  box-shadow:0 10px 24px rgba(0,0,0,.45);
}

.cover.placeholder{
  background: rgba(255,255,255,.06);
}

.right{ flex:1; }
.top{ display:flex; align-items:center; justify-content:space-between; gap:10px; }
.title{ font-size:18px; font-weight:800; color:#fff; line-height:1.2; }

.badge{
  font-size:12px;
  padding:6px 10px;
  border-radius:999px;
  background: rgba(125,211,252,.14);
  border: 1px solid rgba(125,211,252,.22);
  color: var(--text);
  white-space:nowrap;
}

.meta{ margin-top:8px; font-size:13px; color: rgba(230,237,243,.86); line-height:1.45; }
.desc{ margin-top:10px; font-size:13px; color: var(--muted); line-height:1.55; }
.hint{ color: var(--muted); padding: 10px 0; }

button{ border-radius:14px !important; font-weight:700 !important; }
"""


with gr.Blocks(theme=gr.themes.Soft(), css=CSS, title="BookMuse") as demo:
    gr.Markdown(
        """
# BookMuse
Discover your next read in seconds.
"""
    )

    with gr.Tabs():
        with gr.Tab("Find similar books"):
            gr.Markdown("Choose a book you already know and we’ll suggest similar picks.")
            with gr.Row():
                title_in = gr.Dropdown(choices=list(titles), label="Book title", allow_custom_value=True)
                topn1 = gr.Slider(1, 12, value=6, step=1, label="Results")
            btn1 = gr.Button("Recommend", variant="primary")
            out1 = gr.HTML()
            btn1.click(ui_by_title, inputs=[title_in, topn1], outputs=out1)

        with gr.Tab("Explore by topic"):
            gr.Markdown("Describe what you're in the mood for (e.g., “space opera with strong characters”).")
            with gr.Row():
                topic_in = gr.Textbox(label="Topic", placeholder="Describe your next read...", lines=2)
                topn2 = gr.Slider(1, 12, value=6, step=1, label="Results")
            btn2 = gr.Button("Explore", variant="primary")
            out2 = gr.HTML()
            btn2.click(ui_by_topic, inputs=[topic_in, topn2], outputs=out2)

demo.launch()