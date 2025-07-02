import os
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone as PineconeClient
from ..classes import ResearchState
from langchain_core.messages import AIMessage

class SearchNode:
    def __init__(self):
        # Embedding model
        self.embedding = OpenAIEmbeddings(
            openai_api_key=os.environ["OPENAI_API_KEY"]
        )

        pc = PineconeClient(api_key=os.environ["PINECONE_API_KEY"])
        self.indexes = {
            "fund": pc.Index("fund-rag-ocr-index"),
             "economy": pc.Index("economic-rag-ocr-index"),
            # "macro": pc.Index("macro-index"),
            # "stock": pc.Index("stock-index"),
        }

    async def run(self, state: ResearchState) -> ResearchState:
        query = state.get("rewritten_query", "")

        namespace = state.get("namespace", "unknown")
        index = self.indexes.get(namespace, self.indexes["fund"])

        # Embed query
        vector = self.embedding.embed_query(query)

        # Semantic search
        response = index.query(
            vector=vector,
            top_k=100,
            include_metadata=True,
            include_values=False
        )

        # Extract documents
        docs = []
        for match in response.get("matches", []):
            meta = match.metadata
            meta["score"] = match.score
            
            docs.append(meta)

        # Save to state
        state["documents"] = docs
        if "messages" not in state:
            state["messages"] = []
        state["messages"].append(
            AIMessage(content=f"🔎 Retrieved {len(docs)} fund documents from vector DB for query: \"{query}\"")
        )
        

        return state