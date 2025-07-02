import streamlit as st
import asyncio
from backend.graph import DoTACotGraph
import nest_asyncio
from dotenv import load_dotenv
nest_asyncio.apply()

load_dotenv()

st.set_page_config(page_title="DoTA RAG CoT Agent", layout="wide")
st.title("🧠 DoTA RAG CoT Agent Demo")

query = st.text_input("Enter your research question", placeholder="e.g., What are the risks of PRINCIPAL FI fund?")

run_button = st.button("Run Agent")

if run_button and query:
    graph = DoTACotGraph(query=query)

    placeholder = st.empty()
    output_messages = []

    async def run_graph():
        async for state in graph.run(thread={"thread_id": "ui"}):
            # Flatten sub-states into the main state dictionary
            flat_state = {}
            for key, val in state.items():
                if isinstance(val, dict) and "messages" in val:
                    flat_state["messages"] = val["messages"]
                    flat_state.update(val)
                else:
                    flat_state[key] = val
            state = flat_state

            messages = state.get("messages", [])
            if messages:
                output_messages.clear()
                for m in messages:
                    output_messages.append(m.content)
                placeholder.markdown("\n\n".join(output_messages))

    asyncio.run(run_graph())