from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from ..classes import ResearchState

class RerankSummaryNode:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.2)

    async def run(self, state: ResearchState) -> ResearchState:
        documents = state.get("documents", [])
        query = state.get("rewritten_query", "")

        if not documents:
            state["messages"].append(
                AIMessage(content="⚠️ No documents to summarize after rerank.")
            )
            return state

        namespace = state.get("namespace", "unknown")

        if namespace == "fund":
            context = "\n\n".join(
                f"""
📄 Fund #{i+1}
- AMC: {doc.get('amc_name', '')}
- Fund Code: {doc.get('short_code', '')}
- NAV: {doc.get('nav', '')} (as of {doc.get('nav_date', '')})
- Return (1Y): {doc.get('return_1y', '')}
- Sharpe Ratio (1Y): {doc.get('sharpe_ratio_1y', '')}
- Max Drawdown (1Y): {doc.get('max_drawdown_1y', '')}
- Key Info: {doc.get('page_content', '')}
"""
                for i, doc in enumerate(documents)
            )
        elif namespace == "economy":
            context = "\n\n".join(
                f"""
📄 Article #{i+1}
- Headline: {doc.get('article', '')}
- Last Updated: {doc.get('last_updated', '')}
- Summary: {doc.get('page_content', '')}
"""
                for i, doc in enumerate(documents)
            )
        else:
            context = "\n\n".join(
                f"📄 Document #{i+1}\n{doc.get('page_content', '')}"
                for i, doc in enumerate(documents)
            )

        prompt = f"""
You are a financial assistant AI.

Based on the following documents in the "{namespace}" domain, summarize the key insights in structured form so it can be used to answer the user's query.

User Query: "{query}"

Documents:
{context}
"""

        response = await self.llm.ainvoke(prompt)
        summary_text = response.content.strip()

        source_info = []

        for doc in documents:
            source_entry = {
                "source_name": doc.get("source_name", ""),
                "source_url": doc.get("source_url", ""),
                "article": doc.get("article", ""),
                "source_file": doc.get("source_file", ""),
                "source_type": doc.get("source_type", ""),
                "last_updated": doc.get("last_updated", "")
            }
            source_info.append(source_entry)

        state["documents"] = [{
            "page_content": summary_text,
            "sources": source_info,
            "source": "summary_of_reranked_docs"
        }]

        state["messages"].append(
            AIMessage(content="📝 Summarized reranked documents into condensed context.")
        )

        return state