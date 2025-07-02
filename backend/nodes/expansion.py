from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from ..classes import ResearchState

class ExpansionNode:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    async def run(self, state: ResearchState) -> ResearchState:
        """
        Expand the rewritten query with relevant contextual detail to improve retrieval precision.
        """
        rewritten_query = state.get("rewritten_query", "")
        intent = state.get("current_intent", "unknown")

        prompt = ChatPromptTemplate.from_messages([
            ("system",
            f"""You are an assistant that expands search queries to improve retrieval accuracy.
Given a rewritten query and an intent '{intent}', enrich it with context, keywords, or clarifications useful for document retrieval in a financial domain.
Avoid inventing facts and keep it concise and precise.

Respond with the expanded query only.
"""),
            ("human", "Rewritten Query:\n{rewritten_query}")
        ]).format_messages(
            rewritten_query=rewritten_query
        )

        response = await self.llm.ainvoke(prompt)
        expanded_query = response.content.strip()

        state["expanded_query"] = expanded_query
        state["messages"].append(AIMessage(content=f"📈 Expanded query: {expanded_query}"))
        return state