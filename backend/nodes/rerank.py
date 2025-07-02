import os
import cohere
from langchain_core.messages import AIMessage
from ..classes import ResearchState
from rank_bm25 import BM25Okapi

class RerankNode:
    def __init__(self):
        self.client = cohere.Client(os.environ["COHERE_API_KEY"])

    def compose_rerank_input(self, doc):
        return (
            f"page_content: {doc.get('page_content', '')}\n"
        )


    async def run(self, state:ResearchState)->ResearchState:
        query = state.get("rewritten_query", "")
        documents = state.get("documents", [])

        try:
            from nltk.tokenize import word_tokenize
        except LookupError as e:
            raise RuntimeError(
                "❌ Missing NLTK tokenizer 'punkt'. Please run: python -m nltk.downloader punkt"
            ) from e

        # BM25 rerank before Cohere rerank
        tokenized_corpus = [word_tokenize(doc.get("page_content", "")) for doc in documents]
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = word_tokenize(query)
        bm25_scores = bm25.get_scores(tokenized_query)

        for i, score in enumerate(bm25_scores):
            documents[i]["bm25_score"] = score

        # Sort and take top N for semantic rerank
        documents = sorted(documents, key=lambda d: d["bm25_score"], reverse=True)
        top_documents = documents[:50]

        if "messages" not in state:
            state["messages"] = []

        if not documents:
            state["messages"].append(
                AIMessage(content="⚠️ No documents to rerank.")
            )
            return state


        rerank_inputs = [self.compose_rerank_input(doc) for doc in top_documents]

        response = self.client.rerank(
            model="rerank-v3.5",  
            query=query,
            documents=rerank_inputs,
            top_n=10
        )

        reranked = []
        for result in response.results:
            doc = top_documents[result.index]
            doc["rerank_score"] = result.relevance_score
            reranked.append(doc)

        # update state
        state["documents"] = reranked
        if "messages" not in state:
            state["messages"] = []
        state["messages"].append(
            AIMessage(content="✅ Re-ranked top 10 results using Cohere Rerank.")
        )

        return state