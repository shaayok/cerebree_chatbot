# fill_db.py — Load autism knowledge base into Chroma
import os
import hashlib
from pathlib import Path
from dotenv import load_dotenv

import chromadb
from openai import OpenAI
from langchain_community.document_loaders import PyPDFDirectoryLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------- Load env ----------------
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

DATA_PATH = Path("data")
CHROMA_PATH = Path(os.getenv("CHROMA_PATH", "chroma_db"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "autism_bot")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("⚠️ Missing OPENAI_API_KEY in .env")

client = OpenAI()

# ---------------- Init Chroma ----------------
chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

# ---------------- Load all docs ----------------
def load_documents():
    docs = []
    # PDFs
    if (DATA_PATH).exists():
        pdf_loader = PyPDFDirectoryLoader(str(DATA_PATH))
        docs.extend(pdf_loader.load())

        # DOCX
        for docx_file in DATA_PATH.glob("*.docx"):
            loader = Docx2txtLoader(str(docx_file))
            docs.extend(loader.load())

    return docs

raw_documents = load_documents()
if not raw_documents:
    print("⚠️ No documents found in ./data. Add PDFs or DOCX files first.")
    exit()

print(f"📂 Loaded {len(raw_documents)} raw documents.")

# ---------------- Split into chunks ----------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=150, length_function=len
)
chunks = splitter.split_documents(raw_documents)
print(f"✂️ Split into {len(chunks)} chunks.")

# ---------------- Insert into Chroma ----------------
added = 0
for chunk in chunks:
    text = chunk.page_content.strip()
    if not text:
        continue

    # Deduplication via MD5 hash
    uid = hashlib.md5(text.encode("utf-8")).hexdigest()

    # Embed using OpenAI
    embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    ).data[0].embedding

    collection.upsert(
        documents=[text],
        embeddings=[embedding],
        metadatas=[chunk.metadata],
        ids=[uid]
    )
    added += 1

print(f"✅ Successfully added {added} chunks into collection: {COLLECTION_NAME}")
