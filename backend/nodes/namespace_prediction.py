from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI  
from collections import Counter
from ..classes import ResearchState
class NamespacePredictionNode:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    async def run(self, state:ResearchState)-> ResearchState:
        """
        Predict the namespace using 4-vote ensemble from LLM responses.
        """
        query = state.get("rewritten_query", "")
        candidate_namespaces = ["economy", "fund", "unknown"]

        prompt_template = ChatPromptTemplate.from_messages([
            ("system",
            "You are a classification assistant for English financial queries related to mutual fund analysis, macroeconomic data, company earnings, and stock market updates.\n"
            f"Choose exactly one namespace from: {', '.join(candidate_namespaces)}. Respond with only one word.\n\n"
            "Examples:\n"
            "- 'What are the historical returns of the SSFX fund?' → fund\n"
            "- 'What is DELTA’s Q1 2024 earnings?' → company\n"
            "- 'What is the inflation rate in May 2024?' → macro"),
            ("human", "Query: {input}")
        ])

        votes = []
        for _ in range(4):
            prompt = prompt_template.format_messages(input=query)
            response = await self.llm.ainvoke(prompt)
            vote = response.content.strip().lower()
            if vote in candidate_namespaces:
                votes.append(vote)

        # Aggregate votes
        counted = Counter(votes)
        if counted:
            predicted_namespace = counted.most_common(1)[0][0]
        else:
            predicted_namespace = "unknown"

        # Update state
        state["namespace"] = predicted_namespace
        if "messages" not in state:
            state["messages"] = []
        state["messages"].append(
            AIMessage(content=f"🗳️ Namespace votes: {dict(counted)}\n🔍 Predicted namespace: {predicted_namespace}")
        )

        return state