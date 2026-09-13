"""
State Schema for FARM Agentic AI System
=======================================

Defines the NegotiationState TypedDict used by LangGraph to track
the negotiation process between agents.

WoT (Web of Things) Multi-Agent System:
- Planner Agent: Decomposes query into sub-tasks
- Trigger Agent: Selects trigger and makes OFFER
- Action Agent: Selects action and validates compatibility
- Cross-Scorer: Ranks trigger-action pairs
- Verifier: Validates final applet with reflection
"""

from typing import TypedDict, Optional, Literal, List, Dict, Any, Annotated, Tuple
from langgraph.graph import add_messages


class CandidateInfo(TypedDict):
    """
    Information about a trigger or action candidate from RAG.

    Attributes:
        service_name: Name of the service
        category: Service category
        description: Service description
        api_info: Full API schema with ingredients/fields
        score: RAG retrieval score
    """
    service_name: str
    category: str
    description: str
    api_info: Dict[str, Any]
    score: float


class RejectionRecord(TypedDict):
    """
    Record of a rejected trigger-action pair.

    Attributes:
        trigger_idx: Index of the trigger candidate
        action_idx: Index of the action candidate
        reason: Rejection reason from Action Agent
    """
    trigger_idx: int
    action_idx: int
    reason: str


class PairScoreInfo(TypedDict):
    """
    Score information for a trigger-action pair.

    Attributes:
        trigger_idx: Trigger candidate index
        action_idx: Action candidate index
        score: Combined compatibility score
        coverage: Ingredient-field coverage ratio
        rag_score: Combined RAG retrieval score
        reasoning: Explanation of the score
    """
    trigger_idx: int
    action_idx: int
    score: float
    coverage: float
    rag_score: float
    reasoning: str


class ICLExample(TypedDict):
    """
    In-Context Learning example for few-shot prompting.

    Attributes:
        query: Example user query
        trigger_service: Selected trigger service
        action_service: Selected action service
        bindings: Example bindings
        similarity: Similarity score to current query
    """
    query: str
    trigger_service: str
    action_service: str
    bindings: List[Dict[str, Any]]
    similarity: float


class NegotiationState(TypedDict):
    """
    Main state for the negotiation workflow.

    This state is persisted by LangGraph's checkpointer throughout
    the negotiation session, enabling multi-turn interactions.

    Sections:
        Input: User query
        Candidates: Top-K from RAG (loaded once)
        Position: Current pair being negotiated
        Messages: LLM conversation history
        Negotiation: Current OFFER/REQUIREMENTS/bindings
        Result: Final output
    """

    # =========================================================================
    # INPUT
    # =========================================================================
    query: str  # Original user query

    # =========================================================================
    # PLANNER OUTPUT (NEW - Agentic)
    # =========================================================================
    plan: Optional[Dict[str, Any]]           # Planner's decomposition
    trigger_search_query: Optional[str]      # Extracted trigger search query
    action_search_query: Optional[str]       # Extracted action search query
    planner_reasoning: Optional[str]         # Planner's chain-of-thought
    tool_calls: Optional[List[Dict[str, Any]]]  # Tool calls made by Planner

    # =========================================================================
    # CANDIDATES (Short-term memory - loaded once from RAG)
    # =========================================================================
    trigger_candidates: List[CandidateInfo]  # Top-K triggers
    action_candidates: List[CandidateInfo]   # Top-K actions

    # =========================================================================
    # CROSS-SCORER OUTPUT (NEW - Agentic)
    # =========================================================================
    pair_scores: Optional[List[PairScoreInfo]]      # All pair compatibility scores
    priority_queue: Optional[List[Tuple[int, int]]] # Sorted (trigger_idx, action_idx)
    current_pair_idx: int                           # Index into priority_queue

    # =========================================================================
    # ICL MEMORY (NEW - Agentic)
    # =========================================================================
    icl_examples: Optional[List[ICLExample]]  # Few-shot examples for prompts
    attempted_pairs: List[Tuple[int, int]]    # Pairs already tried

    # =========================================================================
    # POSITION (Current negotiation position)
    # =========================================================================
    current_trigger_idx: int   # Index into trigger_candidates (0, 1, 2)
    current_action_idx: int    # Index into action_candidates (0, 1, 2)
    pair_attempt: int          # Which pair we're on (0-8)

    # =========================================================================
    # SELECTOR SCORES (NEW - Multi-candidate scoring)
    # =========================================================================
    trigger_scores: Optional[Dict[int, float]]  # LLM scores for each trigger candidate
    action_scores: Optional[Dict[int, float]]   # LLM scores for each action candidate
    action_scores_trigger_idx: Optional[int]    # Which trigger the action_scores are for (must re-score on change)
    tried_pairs: List[Tuple[int, int]]          # (trigger_idx, action_idx) pairs already tried
    pair_ranking: Optional[List[Tuple[int, int, float]]]  # Sorted pairs by combined RAG score
    llm_overrode_rag: bool                      # True if LLM picked different action than RAG top
    tried_llm_fallback: bool                    # True if we already tried RAG fallback after LLM failed

    # =========================================================================
    # MESSAGES (LLM conversation history)
    # =========================================================================
    messages: Annotated[List[Any], add_messages]  # Full message history

    # =========================================================================
    # NEGOTIATION STATE
    # =========================================================================

    # Trigger Agent output
    current_offer: Optional[Dict[str, Any]]  # Ingredients offered
    trigger_reasoning: Optional[str]         # Why trigger was selected

    # Action Agent output
    current_requirements: Optional[Dict[str, Any]]  # Fields required
    action_reasoning: Optional[str]                 # Why action was selected

    # Negotiation decision
    negotiation_status: Literal[
        "pending",      # Not started
        "negotiating",  # In progress
        "accepted",     # Action Agent accepted
        "rejected",     # Action Agent rejected
        "failed"        # All pairs exhausted
    ]

    # Binding map (if accepted)
    binding_map: Optional[List[Dict[str, Any]]]

    # Rejection history (for learning)
    rejection_history: List[RejectionRecord]

    # =========================================================================
    # VERIFIER OUTPUT
    # =========================================================================
    verifier_score: Optional[float]
    verifier_critique: Optional[str]
    is_executable: Optional[bool]

    # =========================================================================
    # FINAL OUTPUT
    # =========================================================================
    final_applet: Optional[Dict[str, Any]]

    # =========================================================================
    # ERROR HANDLING
    # =========================================================================
    error: Optional[str]
    retry_count: int


def create_initial_state(query: str) -> NegotiationState:
    """
    Create initial state for a new negotiation session.

    Args:
        query: User query describing the desired applet.

    Returns:
        NegotiationState with default values.
    """
    return NegotiationState(
        # Input
        query=query,

        # Planner output (NEW)
        plan=None,
        trigger_search_query=None,
        action_search_query=None,
        planner_reasoning=None,

        # Candidates (to be loaded by load_candidates node)
        trigger_candidates=[],
        action_candidates=[],

        # Cross-scorer output (NEW)
        pair_scores=None,
        priority_queue=None,
        current_pair_idx=0,

        # ICL memory (NEW)
        icl_examples=None,
        attempted_pairs=[],

        # Position
        current_trigger_idx=0,
        current_action_idx=0,
        pair_attempt=0,

        # Selector scores (NEW)
        trigger_scores=None,
        action_scores=None,
        action_scores_trigger_idx=None,
        tried_pairs=[],
        pair_ranking=None,
        llm_overrode_rag=False,
        tried_llm_fallback=False,

        # Messages
        messages=[],

        # Negotiation state
        current_offer=None,
        trigger_reasoning=None,
        current_requirements=None,
        action_reasoning=None,
        negotiation_status="pending",
        binding_map=None,
        rejection_history=[],

        # Verifier output
        verifier_score=None,
        verifier_critique=None,
        is_executable=None,

        # Final output
        final_applet=None,

        # Error handling
        error=None,
        retry_count=0,
    )
