import os
import json
from dotenv import load_dotenv
from pinecone import Pinecone as PineconeClient, ServerlessSpec
from langchain_openai import OpenAIEmbeddings

load_dotenv()

# 🧠 Setup Pinecone client
index_name = "fund-rag-index"
pc = PineconeClient(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(index_name)

# 🔍 Setup embedding model (same one used for upsert)
embedding = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])

def semantic_query(query: str, top_k: int = 5):
    # 1. Embed the query
    query_vector = embedding.embed_query(query)

    # 2. Run semantic search
    resp = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
        include_values=False
    )

    # 3. Process and print results
    matches = resp.get("matches", [])
    print(f"Top {len(matches)} results for query: {query!r}\n")
    for rank, match in enumerate(matches, start=1):
        meta = match.metadata
        score = match.score
        snippet = meta.get("policy_summary", "")[:150]  # ตัวอย่าง snippet
        print(f"{rank}. score: {score:.4f}  |  fund: {meta.get('short_code')}  |  snippet: {snippet}...")
    print()

if __name__ == "__main__":
    while True:
        q = input("Enter search query (or 'exit'): ").strip()
        if q.lower() in ["exit", "quit"]:
            break
        semantic_query(q)