import os
import json
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm
from uuid import uuid4

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

from pinecone import Pinecone as PineconeClient, ServerlessSpec

# === Load .env ===
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
assert OPENAI_API_KEY and PINECONE_API_KEY, "❌ Missing API Keys"

# === Init Clients ===
embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

index_name = "economic-rag-ocr-index"
pc = PineconeClient(api_key=PINECONE_API_KEY)
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
index = pc.Index(index_name)

# === Load raw documents ===
base_dir = Path("rag_outputs/econ_ocr_only")
raw_documents = []
for doc_dir in tqdm(list(base_dir.iterdir()), desc="📄 Loading OCR Docs"):
    if doc_dir.is_dir():
        content_path = doc_dir / "content.txt"
        meta_path = doc_dir / "meta.json"
        if content_path.exists() and meta_path.exists():
            with open(content_path, "r", encoding="utf-8") as f:
                text = f.read()
            with open(meta_path, "r", encoding="utf-8") as m:
                metadata = json.load(m)
            raw_documents.append(Document(page_content=text, metadata=metadata))

print(f"✅ Loaded {len(raw_documents)} raw documents")

# === Split documents ===
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
split_documents = []
for doc in raw_documents:
    chunks = splitter.split_text(doc.page_content)
    for i, chunk in enumerate(chunks):
        split_documents.append(
            Document(page_content=chunk, metadata={**doc.metadata, "chunk_id": i})
        )

print(f"✂️ Split into {len(split_documents)} chunks")

# === Embed + Upsert in batches ===
batch_size = 50
for i in tqdm(range(0, len(split_documents), batch_size), desc="📤 Upserting to Pinecone"):
    batch = split_documents[i:i + batch_size]
    vectors = []
    for doc in batch:
        vec_id = str(uuid4())
        vector = embedding.embed_query(doc.page_content)
        metadata = {k: ("" if v is None else v) for k, v in doc.metadata.items()}
        metadata["page_content"] = doc.page_content 
        vectors.append((vec_id, vector, metadata))
    index.upsert(vectors=vectors)

print(f"🚀 Completed upserting {len(split_documents)} chunks to index '{index_name}'")