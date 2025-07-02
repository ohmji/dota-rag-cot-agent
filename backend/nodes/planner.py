from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import AIMessage
from ..classes import InputState, ResearchState


class CoTPlannerNode:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
        self.parser = JsonOutputParser()  

    async def run(self, state: ResearchState) -> ResearchState:
        query = state.get("query", "")
        format_instructions = self.parser.get_format_instructions()

        prompt_template = ChatPromptTemplate.from_messages([
            (
                "system",
                f"""You are a reasoning planner for financial agents.

        Your task is to decompose a user query into a **concise and minimal** reasoning plan (in JSON array).
        - Each step should be **atomic** (not overly broad or too detailed).
        - Avoid overly fine-grained or redundant steps.
        - Avoid steps that simply rephrase the question.
        - Only include reasoning steps that are strictly **necessary** to answer the query.
        - Max steps: 3 (unless complex question demands more)
        - Each step must include:
        - "step": a short description of the subtask
        - "intent": one of: economy, fund, unknown

        Example:
        User query: "What funds should you buy in the economy right now?"

        {format_instructions}
        """
            ),
            ("human", "{input}")
        ])
        prompt = prompt_template.format_messages(input=query)

        try:
            response = await self.llm.ainvoke(prompt)
            plan = self.parser.parse(response.content)  
            state["cot_plan"] = plan
            state.setdefault("messages", []).append(
                AIMessage(content="🧠 CoT plan: " + str(plan))
            )
            return state

        except Exception as e:
            state["cot_plan"] = [{"step": "fallback reasoning", "intent": "unknown"}]
            state.setdefault("messages", []).append(
                AIMessage(content=f"⚠️ CoT planning failed: {str(e)}")
            )
            return state