from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from ..classes import ResearchState

class SummaryNode:
    def __init__(self):
        self.llm = ChatOpenAI(model="o3-mini")

    async def run(self, state: ResearchState) -> ResearchState:
        query = state.get("query", "")
        all_answers = state.get("all_answers", [])
        final_answer = state.get("answer", "")

        # 🧠 Reasoning path
        history = "\n\n".join(
            f"Step {a['step']} [{a['intent']}]:\n"
            f"- Rewritten Query: {a['rewritten_query']}\n"
            f"- Answer: {a['answer']}"
            for a in all_answers
        )

        # ✅ Always use English prompt (regardless of query language)
        prompt = f"""
You are a professional financial advisor.

Client's Original Question:
{query}

Final Recommended Answer:
{final_answer}

Step-by-Step Reasoning Based on Research:
{history}

Based on the reasoning above, write a detailed, professional summary that directly addresses the client's question.
Include the key insights from each step, and explain how each step contributes to the final recommendation. 
Use the same language as the original question.
"""

        # 🔗 Generate final summary
        response = await self.llm.ainvoke(prompt)
        summary = response.content.strip()

        # 🧾 Collect sources
        sources = []
        for answer in all_answers:
            if "sources" in answer:
                sources.extend(answer["sources"])

        # 📋 Remove duplicates based on source_url or source_file
        unique_sources = []
        seen = set()
        for s in sources:
            key = s.get("source_url") or s.get("source_file")
            if key and key not in seen:
                seen.add(key)
                unique_sources.append(s)

        # 📌 Append source list to summary
        if unique_sources:
            source_list = "\n\n📚 Sources:\n" + "\n".join(
                f"- {s.get('source_name') or s.get('source_url') or s.get('article')}"
                for s in unique_sources
            )
            summary += source_list

        state["final_summary"] = summary
        state["messages"].append(AIMessage(content=f"Question is {query} \n\nFinal Summary:\n{summary}"))
        return state