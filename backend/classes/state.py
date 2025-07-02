from typing import TypedDict, Optional, Required, Dict, List, Any

# Define the input state
class InputState(TypedDict, total=False):
    job_id: str
    query:  str
    current_step: int
    done: bool

class ResearchState(InputState):
    current_step: Required[int]
    done: Optional[int]
    cot_plan: List[Dict[str, Any]]
    current_intent: Optional[str]
    messages: List[Any]
    rewritten_query: Optional[str]
    namespace: Optional[str]
    documents: List[Dict[str, Any]]
    answer: Optional[str]
    all_answers: List[Dict[str, Any]]
    cot_query: Optional[str]
    final_summary: Optional[str]
    