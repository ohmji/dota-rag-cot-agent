from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from ..classes import ResearchState

class RewriteQueryNode:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)

    async def run(self, state: ResearchState) -> ResearchState:
        """
        Rewrite the query using an LLM to make it clearer and more effective for semantic retrieval.
        """

        cot_plan = state.get("cot_plan", [])    
        cot_query = state.get("cot_query", "")
        current_step = state.get("current_step", 0)


        if current_step >= len(cot_plan):
            state["done"] = True
            return state

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a financial expert assistant helping refine user queries for better retrieval.\n\nYour task is to rewrite the query so it is short, focused, and semantically rich — suitable for vector-based document retrieval.\n\nGuidelines:\n- Keep it one sentence.\n- Preserve the original intent.\n- Include essential financial keywords only.\n- Avoid unnecessary structure or over-specification.\n- Do NOT include bullet points or lists.\n- Do NOT hallucinate data."
            ),
            ("human", "User query: {input}")
        ]).format_messages(input=cot_query)

        response = await self.llm.ainvoke(prompt)
        rewritten_query = response.content.strip()

        # ✅ Update state
        state["rewritten_query"] = rewritten_query
        state.setdefault("rewritten_queries", []).append(rewritten_query)
        state.setdefault("messages", []).append(
            AIMessage(content=f"🔄 Step {current_step + 1} rewritten query: {rewritten_query}")
        )

        return state