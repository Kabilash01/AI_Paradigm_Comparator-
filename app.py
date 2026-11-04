"""
Streamlit web app for the AI Paradigm Comparator.

Run locally:   streamlit run app.py
Deploy:        Hugging Face Spaces (Streamlit SDK) or Streamlit Community Cloud
"""

import streamlit as st

from paradigm_comparator import (
    KNOWLEDGE_BASE, NO_MATCH, RULES, TEST_QUERIES,
    agentic_ai, generative_ai, run_query, traditional_ai,
)

st.set_page_config(page_title="AI Paradigm Comparator", page_icon="🤖",
                   layout="wide")

st.title("🤖 AI Paradigm Comparator")
st.caption("One query, three AI paradigms: rule-based, generative and "
           f"agentic · {len(RULES)} rules · {len(KNOWLEDGE_BASE)} "
           "knowledge entries · Topic: AI & Machine Learning")

# ---- Query input --------------------------------------------------------
if "query" not in st.session_state:
    st.session_state.query = TEST_QUERIES[0]

st.write("**Try a sample query:**")
cols = st.columns(len(TEST_QUERIES))
for col, sample in zip(cols, TEST_QUERIES):
    if col.button(sample, use_container_width=True):
        st.session_state.query = sample

query = st.text_input("Or type your own question", key="query")

# ---- Three paradigms side by side ---------------------------------------
if query.strip():
    trad_col, gen_col, agent_col = st.columns(3)

    with trad_col:
        st.subheader("📘 Traditional AI")
        answer = traditional_ai(query)
        if answer == NO_MATCH:
            st.error("MISS: No match found")
        else:
            st.success("HIT")
            st.write(answer)
        st.caption("Exact rule lookup: precise but brittle.")

    with gen_col:
        st.subheader("🎲 Generative AI")
        gen = generative_ai(query)
        if gen.on_topic:
            st.success("OK")
        else:
            st.warning("WEAK (off-topic)")
        st.write(gen.answer)
        st.button("🔄 Regenerate")   # any click reruns the app
        st.caption("Random fragment mixing: changes every run.")

    with agent_col:
        st.subheader("🧠 Agentic AI")
        agent = agentic_ai(query, verbose=False)
        label = f"{agent.status} ({agent.iterations} iter)"
        if agent.status == "OK":
            st.success(label)
        elif agent.status == "PARTIAL":
            st.warning(label)
        else:
            st.error("FAIL")
        st.write(agent.answer)
        with st.expander("Show reasoning trace", expanded=True):
            st.code("\n".join(agent.log), language="text")
        st.caption("Search → self-check → broaden → synthesize.")

# ---- Benchmark table ----------------------------------------------------
st.divider()
st.subheader("📊 Comparison on the 4 test queries")
if st.button("Run comparison"):
    rows = [run_query(q, verbose=False) for q in TEST_QUERIES]
    st.table([{"Query": r["query"], "Traditional": r["traditional"],
               "Generative": r["generative"], "Agentic": r["agentic"]}
              for r in rows])
