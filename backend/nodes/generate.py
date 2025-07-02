from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI 
from ..classes import ResearchState

class GenerateNode:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.2)

    async def run(self, state: ResearchState) -> ResearchState:
        """
        Generate a final answer using top reranked documents and the original query.
        """
        query = state.get("rewritten_query", "")
        documents = state.get("documents", [])
        top_docs = documents

        if not top_docs:
            if "messages" not in state:
                state["messages"] = []
            state["messages"].append(
                AIMessage(content="⚠️ No summarized documents available for generation.")
            )
            return state

        summarized_context = top_docs[0].get("page_content", "No summary content available.")

        # 💬 Prompt template
        prompt = f"""
You are a financial assistant AI with access to summarized documents. Your task is to answer the user’s query using the following summary.

Instructions:
- Focus on answering the query using the summarized insights.
- Maintain a professional and clear tone.

User Query: \"{query}\"

Summarized Context:
{summarized_context}
"""
        response = await self.llm.ainvoke(prompt)
        answer = response.content.strip()

        # Update state
        state["answer"] = answer


        if "all_answers" not in state:
            state["all_answers"] = []

        state["all_answers"].append({
            "step": state.get("current_step", 0),
            "intent": state.get("current_intent", ""),
            "rewritten_query": state.get("rewritten_query", ""),
            "answer": answer,
            "sources": top_docs[0].get("sources", []),
        })

        state["current_step"] = state.get("current_step", 0) + 1

        if "messages" not in state:
            state["messages"] = []
        state["messages"].append(AIMessage(content=f"💡 Answer generated: {answer[:600]}..."))

        return state