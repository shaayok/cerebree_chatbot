# chat.py — Autism Support Bot (ChatGPT-style with RAG)
import os
from pathlib import Path
import streamlit as st
import chromadb
from openai import OpenAI
from dotenv import load_dotenv

# -------------------- Load config --------------------
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

CHROMA_PATH = Path(os.getenv("CHROMA_PATH", "chroma_db"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "autism_bot")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

client = OpenAI(api_key=OPENAI_API_KEY)

# -------------------- Helpers --------------------
def compose_context(documents, metadatas, cap=4000) -> str:
    """Format KB chunks into a context string."""
    out, used = [], 0
    for d, m in zip(documents, metadatas):
        src = m.get("file", m.get("source", "kb"))
        seg = f"\n---\nSource: {src}\n---\n{d}"
        if used + len(seg) > cap:
            break
        out.append(seg)
        used += len(seg)
    return "\n".join(out)

def answer_with_openai(messages):
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.3,
        max_tokens=800,
    )
    return resp.choices[0].message.content.strip()

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="Autism Support Bot", page_icon="🧩", layout="wide")

st.markdown(
    """
    # 🧩 Autism Support Bot  
    This tool is designed to **guide parents and caregivers** with general advice, based on expert-informed knowledge.  

    ⚠️ **Disclaimer:** I am *not* a medical professional.  
    This guidance is **not a diagnosis**. If you are worried about your child,  
    please consult a licensed psychologist or pediatrician.  
    """,
    unsafe_allow_html=True,
)

# -------------------- Connect to Chroma --------------------
@st.cache_resource(show_spinner=False)
def get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    return client.get_or_create_collection(name=COLLECTION_NAME)

try:
    collection = get_collection()
    st.caption(f"📚 Knowledge Base connected • {COLLECTION_NAME} • {collection.count()} entries")
except Exception as e:
    st.error("❌ Could not open Chroma collection.")
    st.exception(e)
    st.stop()

# -------------------- Conversation History --------------------
if "history" not in st.session_state:
    st.session_state.history = []

chat = st.container()
for m in st.session_state.history:
    with chat.chat_message(m["role"]):
        st.markdown(m["content"])

# -------------------- User Input --------------------
user_query = st.chat_input("Type your concern or question here…")
if not user_query:
    st.stop()

with chat.chat_message("user"):
    st.markdown(user_query)
st.session_state.history.append({"role": "user", "content": user_query})

# -------------------- Knowledge Base Retrieval --------------------
with st.spinner("Searching knowledge base…"):
    try:
        query_embedding = client.embeddings.create(
            model="text-embedding-3-small",
            input=user_query
        ).data[0].embedding

        q = collection.query(query_embeddings=[query_embedding], n_results=4)
        docs = q.get("documents", [[]])[0]
        metas = q.get("metadatas", [[]])[0]
    except Exception as e:
        docs, metas = [], []
        st.error("⚠️ Retrieval failed.")
        st.exception(e)

context = compose_context(docs, metas)

# -------------------- Build Prompt --------------------
system_prompt = """
You are a compassionate autism support guide for parents.

Rules:
- Always search the knowledge base first.
- If relevant excerpts are found, use them as your primary source.
- If no relevant excerpts are found, fallback to your general knowledge.
- Use the entire conversation history for context.
- Speak gently and validate the parent's concerns.
- Ask at most 1–2 clarifying questions if needed.
- After a few clarifications, provide a conclusion that includes:
  - Child's age (if known)
  - Main area(s) of concern
  - Urgency level (low / medium / high)
  - Recommended next step
- Always add a disclaimer: this is guidance, not a diagnosis.
- Cite sources from the knowledge base if used.
"""

messages = [{"role": "system", "content": system_prompt}]
messages.extend(st.session_state.history)  # full conversation history
messages.append({
    "role": "user",
    "content": f"Parent's Question: {user_query}\n\nRelevant Knowledge Base Excerpts:\n{context if context else 'No relevant excerpts found.'}"
})

# -------------------- OpenAI Call --------------------
answer = answer_with_openai(messages)

# -------------------- Show Assistant Reply --------------------
with chat.chat_message("assistant"):
    st.markdown(answer)
    if docs:
        with st.expander("📖 Sources", expanded=False):
            for d, m in zip(docs, metas):
                src = m.get("file", m.get("source", "kb"))
                st.markdown(f"**{src}**\n\n{d[:800]}{'…' if len(d) > 800 else ''}")

st.session_state.history.append({"role": "assistant", "content": answer})
