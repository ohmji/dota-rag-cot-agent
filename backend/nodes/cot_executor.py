from langchain_core.messages import AIMessage
from ..classes import ResearchState
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

class CotExecutorNode:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)

    async def run(self, state: ResearchState) -> ResearchState:
        plan = state.get("cot_plan", [])
        current_step = state.get("current_step", 0)

        if current_step >= len(plan):
            state["done"] = True
            return state

        step = plan[current_step]
        state["current_intent"] = step["intent"]
        state.setdefault("messages", []).append(
            AIMessage(content=f"🧭 Step {current_step + 1}: {step['step']} (intent={step['intent']})")
        )

        # CoT reasoning: summarize reasoning context using previous answers + current step
        previous_answers = "\n\n".join(
            f"Step {i + 1} Answer:\n{a['answer']}"
            for i, a in enumerate(state.get("all_answers", []))
        )
        step_instruction = step["step"]
        context = f"{previous_answers}\n\nNext task:\n{step_instruction}" if previous_answers else f"Next task:\n{step_instruction}"

        prompt = ChatPromptTemplate.from_messages([
            ("system", 
             "You are a financial reasoning agent.\n"
             "Your job is to generate a concise and focused reasoning query for the next step of a multi-step financial analysis.\n"
             "Instructions:\n"
             "- Keep it one sentence.\n"
             "- Preserve the intent of the task.\n"
             "- Use essential financial keywords.\n"
             "- Avoid redundant context or structure.\n"
             "- Avoid lists or long formats.\n"
             "- Keep it clear and relevant for document retrieval or summarization."
            ),
            ("human", "Context:\n{context}")
        ]).format_messages(context=context)

        response = await self.llm.ainvoke(prompt)
        state["cot_query"] = response.content.strip()
        
        state["messages"].append(AIMessage(content=f"🔍 CoT Query for Step {current_step + 1}: {state['cot_query']}"))


        return state