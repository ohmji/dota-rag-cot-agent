import logging
from typing import Any, AsyncIterator, Dict

from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, END
from .classes.state import InputState,ResearchState
from .nodes.rewrite_query import RewriteQueryNode
from .nodes.namespace_prediction import NamespacePredictionNode
from .nodes.search import SearchNode
from .nodes.rerank import RerankNode
from .nodes.generate import GenerateNode
from .nodes.planner import CoTPlannerNode
from .nodes.cot_executor import CotExecutorNode
from .nodes.summary import SummaryNode
from .nodes.expansion import ExpansionNode
from .nodes.rerank_summary import RerankSummaryNode
logger = logging.getLogger(__name__)


class DoTACotGraph:
    def __init__(self, query: str = "", job_id=None,current_step: int = 0, done: bool = False):
        self.job_id = job_id
        self.query = query
        self.input_state = InputState(
            query=query,
            job_id=job_id,
            current_step=current_step,
            done=done,
            messages=[SystemMessage(content="🔍 Starting DoTA RAG Cot Research Agent...")]
        )
        self._init_nodes()

    def _init_nodes(self):
        self.planner = CoTPlannerNode()
        self.cot_executor = CotExecutorNode()
        self.rewrite_query = RewriteQueryNode()
        self.predict_namespace = NamespacePredictionNode()
        self.search = SearchNode()
        self.rerank = RerankNode()
        self.rerank_summary = RerankSummaryNode()
        self.generate = GenerateNode()
        self.summary = SummaryNode()
        self.expansion = ExpansionNode()

    def _build_workflow(self):
        self.workflow = StateGraph(InputState, recursion_limit=2000)
        # Initial planner
        self.workflow.add_node("cot_planner", self.planner.run)

        # CotExecutor to decide what to do
        self.workflow.add_node("cot_executor", self.cot_executor.run)
        self.workflow.add_node("summary", self.summary.run)

        # Common execution chain
        self.workflow.add_node("rewrite_query", self.rewrite_query.run)
        self.workflow.add_node("expansion", self.expansion.run)
        self.workflow.add_node("predict_namespace", self.predict_namespace.run)
        self.workflow.add_node("search", self.search.run)
        self.workflow.add_node("rerank", self.rerank.run)
        self.workflow.add_node("rerank_summary", self.rerank_summary.run)
        self.workflow.add_node("generate", self.generate.run)

        # Entry
        self.workflow.set_entry_point("cot_planner")
        self.workflow.add_edge("cot_planner", "cot_executor")

        # Chain
        self.workflow.add_edge("rewrite_query", "expansion")
        self.workflow.add_edge("expansion", "predict_namespace")
        self.workflow.add_edge("predict_namespace", "search")
        self.workflow.add_edge("search", "rerank")
        self.workflow.add_edge("rerank", "rerank_summary")
        self.workflow.add_edge("rerank_summary", "generate")
        
        self.workflow.add_edge("summary", END)


        def should_continue(state: ResearchState) -> str:
            current_step = state.get("current_step", 0)
            plan = state.get("cot_plan", [])
            if current_step >= len(plan):
                return "summary"
            return "cot_executor"

        self.workflow.add_conditional_edges(
            "generate", should_continue, {
                "cot_executor": "cot_executor",
                "summary": "summary",

            }
        )
        
        self.workflow.add_edge("cot_executor", "rewrite_query")


    async def run(self, thread: Dict[str, Any]) -> AsyncIterator[Dict[str, Any]]:
        self._build_workflow()
        
        compiled_graph = self.workflow.compile()
        thread["recursion_limit"] = 100
        async for state in compiled_graph.astream(self.input_state, thread):
            yield state

    def compile(self):
        self._build_workflow()
        return self.workflow.compile()