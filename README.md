---
title: AI Paradigm Comparator
emoji: 🤖
colorFrom: indigo
colorTo: green
sdk: static
app_file: index.html
pinned: false
---

# 🤖 AI Paradigm Comparator

Answers the same query with three AI paradigms (Traditional, Generative and
Agentic) and compares how each one behaves. Pure Python: no API keys, no GPU.

## Files
| File | Purpose |
|---|---|
| `paradigm_comparator.py` | Main program (console output for the report) |
| `AI_Paradigm_Comparator.ipynb` | Same code as a Jupyter notebook, with outputs |
| `app.py` | Streamlit web app for deployment |
| `requirements.txt` | Only `streamlit` |
| `index.html` | Runs `app.py` in the browser via stlite (free static Space) |
| `console_output.txt` | Saved console run of the main program |

## Run
```bash
python paradigm_comparator.py      # console version
streamlit run app.py               # web app at http://localhost:8501
jupyter notebook AI_Paradigm_Comparator.ipynb   # notebook version
```

## Deploy on Hugging Face Spaces (free)
Free Spaces only host **static** sites, so `index.html` uses
[stlite](https://github.com/whitphx/stlite) to run the unchanged Streamlit
`app.py` in the visitor's browser (Pyodide); no server is needed.
1. Go to huggingface.co → **New Space** → choose SDK **Static** (free).
2. Upload `index.html`, `app.py`, `paradigm_comparator.py`, `requirements.txt` and this `README.md`.
3. The Space serves immediately; the first load takes ~20 s while Python loads in the browser.

(Streamlit Community Cloud also works: push the files to a GitHub repo, then
choose **New app** on share.streamlit.io and pick `app.py`.)

## Approach (for the report)
The system uses an AI and machine learning knowledge domain with 9 exact-match
rules and a 16-entry knowledge base, where each entry has searchable tags and a
text fragment. Traditional AI normalizes the query (lower-case, punctuation
removed) and looks it up in the rules dictionary, returning a fixed answer or
"No match found". This makes it precise but brittle, since any rephrasing
fails. Generative AI collects fragments relevant to the query, mixes in random
distractors, and joins 2–3 randomly chosen fragments with connectors. Its
output therefore changes on every run and can drift off-topic, which models
the inconsistency and hallucination risk of generative systems. Agentic AI
runs an iterative loop: it searches the knowledge base with specific keywords,
self-checks whether at least two matches were found, and if not, broadens its
keywords using a synonym map and generic words, switching to fuzzy matching on
the final iteration. Once the evidence is sufficient, it synthesizes an answer
from the top three matches. Four test queries (two rule hits, two misses) and a
comparison table show that only the agentic approach answers all queries
reliably.
