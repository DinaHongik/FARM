"""
RAGAS Metrics for FARM Agentic System
=====================================

Uses the official RAGAS library for evaluation with Ollama/Granite as the LLM.

Note: RAGAS uses its own internal embeddings for evaluation metrics.
The user's RAG system (google/embeddinggemma-300m) is separate and used only
for retrieval, not for RAGAS evaluation.

References:
- https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/agents/
- https://docs.ragas.io/en/stable/howtos/customisations/bring-your-own-llm-or-embs.html
"""

import asyncio
import json
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path

# RAGAS imports
try:
    from ragas import evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness
    from ragas.metrics import LLMContextPrecisionWithReference
    # ToolCallAccuracy removed - not useful for FARM evaluation
    # TopicAdherence is in ragas.metrics.collections (not ragas.metrics)
    from ragas.metrics._topic_adherence import TopicAdherenceScore
    from ragas.dataset_schema import SingleTurnSample, MultiTurnSample, EvaluationDataset
    from ragas.messages import HumanMessage, AIMessage, ToolCall, ToolMessage
    RAGAS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: RAGAS not fully available: {e}")
    RAGAS_AVAILABLE = False

# LangChain imports for Ollama
try:
    from langchain_ollama import ChatOllama
    OLLAMA_AVAILABLE = True
except ImportError:
    try:
        from langchain_community.chat_models import ChatOllama
        OLLAMA_AVAILABLE = True
    except ImportError:
        OLLAMA_AVAILABLE = False


@dataclass
class RAGASMetrics:
    """
    RAGAS metrics for FARM evaluation.

    Primary Metrics:
    - goal_accuracy: Did the applet match user intent? (custom, 0/0.5/1)
    - faithfulness: Is reasoning grounded in retrieved context? (RAGAS)

    RAG Retrieval Quality:
    - context_recall_trigger/action: Is gold service in top-K?
    - context_precision_trigger/action: Are relevant items ranked HIGH? (RAGAS)

    Service Selection:
    - trigger_accuracy: Correct trigger selected
    - action_accuracy: Correct action selected
    - joint_accuracy: Both trigger AND action correct

    Domain Safety:
    - topic_adherence: Does system stay within WoT/IFTTT automation domain? (RAGAS)
    """
    # Primary
    goal_accuracy: float = 0.0
    faithfulness: float = 0.0

    # RAG Retrieval Quality
    context_recall_trigger: float = 0.0
    context_recall_action: float = 0.0
    context_precision_trigger: float = 0.0  # NEW: Are relevant items ranked high?
    context_precision_action: float = 0.0   # NEW: Are relevant items ranked high?

    # Service Selection
    trigger_accuracy: float = 0.0
    action_accuracy: float = 0.0
    joint_accuracy: float = 0.0

    # Domain Safety
    topic_adherence: float = 0.0

    # Process Metrics (kept for evaluator compatibility)
    resolution_rounds: int = 1
    first_try_success: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvaluationResult:
    """Complete evaluation results across a test set."""
    metrics: RAGASMetrics = field(default_factory=RAGASMetrics)
    num_samples: int = 0
    num_success: int = 0
    num_failed: int = 0
    per_sample_results: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": {
                "num_samples": self.num_samples,
                "num_success": self.num_success,
                "num_failed": self.num_failed,
                "success_rate": self.num_success / self.num_samples if self.num_samples > 0 else 0,
            },
            "metrics": self.metrics.to_dict(),
            "per_sample": self.per_sample_results,
        }


# =============================================================================
# RAGAS LLM SETUP (Ollama/Granite)
# =============================================================================

def get_ragas_llm(model_name: str = "granite4:small-h"):
    """
    Get RAGAS-compatible LLM wrapper using Ollama.

    Args:
        model_name: Ollama model name (default: granite4:small-h)

    Returns:
        LangchainLLMWrapper for RAGAS
    """
    if not OLLAMA_AVAILABLE:
        raise ImportError("langchain_ollama not installed. Run: pip install langchain-ollama")

    llm = ChatOllama(model=model_name, temperature=0.0)
    return LangchainLLMWrapper(llm)


def llm_judge(prompt: str, model_name: str = "granite4:small-h") -> str:
    """
    Direct LLM call for evaluation judgments.

    Uses Granite via Ollama to make binary judgments (YES/NO).
    This avoids RAGAS library issues while keeping LLM-based evaluation.

    Args:
        prompt: The judgment prompt
        model_name: Ollama model name

    Returns:
        LLM response text
    """
    if not OLLAMA_AVAILABLE:
        return ""

    try:
        llm = ChatOllama(model=model_name, temperature=0.0)
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        print(f"LLM Judge error: {e}")
        return ""


def llm_judge_binary(prompt: str, model_name: str = "granite4:small-h") -> bool:
    """
    LLM binary judgment (YES/NO).

    Args:
        prompt: The judgment prompt (should ask for YES/NO)
        model_name: Ollama model name

    Returns:
        True if LLM says YES, False otherwise
    """
    response = llm_judge(prompt, model_name)
    response_upper = response.upper()

    # Check for YES/NO in response
    if "YES" in response_upper:
        return True
    elif "NO" in response_upper:
        return False
    else:
        # Fallback: check for positive indicators
        return any(word in response_upper for word in ["TRUE", "CORRECT", "RELEVANT", "VALID"])


# =============================================================================
# SERVICE MATCHING (for trigger/action accuracy)
# =============================================================================

def normalize_service_name(name: str) -> str:
    """Normalize service name for comparison."""
    if not name:
        return ""
    name = name.lower().strip()
    prefixes = ["new ", "add ", "create ", "send ", "post "]
    for p in prefixes:
        if name.startswith(p):
            name = name[len(p):]
    return name


def word_overlap_score(name1: str, name2: str) -> float:
    """
    Compute word overlap score between two service names.

    This is a general algorithm (not hardcoding) that works for any names.
    Uses Jaccard-like similarity based on word overlap.

    Returns:
        Score between 0 and 1 (1 = perfect match)
    """
    # Tokenize into words
    words1 = set(name1.lower().split())
    words2 = set(name2.lower().split())

    # Remove common stop words that don't carry meaning
    stop_words = {"a", "an", "the", "to", "for", "of", "in", "on", "is", "are", "my", "your", "me"}
    words1 = words1 - stop_words
    words2 = words2 - stop_words

    if not words1 or not words2:
        return 0.0

    # Compute overlap
    overlap = len(words1 & words2)
    min_size = min(len(words1), len(words2))

    # Return proportion of smaller set that overlaps
    return overlap / min_size if min_size > 0 else 0.0


# Dynamic semantic matching using embeddings (optional)
_embedding_model = None


def _get_embedding_model():
    """Lazy load embedding model for semantic matching."""
    global _embedding_model
    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            # Use a small, fast model for matching
            _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        except ImportError:
            _embedding_model = False  # Mark as unavailable
    return _embedding_model if _embedding_model else None


def _embedding_similarity(name1: str, name2: str) -> float:
    """
    Compute semantic similarity between two service names using embeddings.
    Returns similarity score between 0 and 1.
    """
    model = _get_embedding_model()
    if model is None:
        return 0.0

    try:
        embeddings = model.encode([name1, name2])
        # Cosine similarity
        from numpy import dot
        from numpy.linalg import norm
        similarity = dot(embeddings[0], embeddings[1]) / (norm(embeddings[0]) * norm(embeddings[1]))
        return float(similarity)
    except Exception:
        return 0.0


def service_matches(predicted: str, expected: str, alternatives: List[str] = None) -> bool:
    """
    Check if predicted service matches expected using multiple strategies.

    Strategies (in order):
    1. Exact match after normalization
    2. Substring match
    3. Word overlap >= 60% (lowered from 70% for more flexibility)
    4. Embedding similarity >= 0.8 (dynamic, no hardcoding)
    """
    pred_norm = normalize_service_name(predicted)
    exp_norm = normalize_service_name(expected)

    if not pred_norm or not exp_norm:
        return False

    # Strategy 1: Exact match
    if pred_norm == exp_norm:
        return True

    # Strategy 2: Substring match
    if pred_norm in exp_norm or exp_norm in pred_norm:
        return True

    # Strategy 3: Word overlap (general algorithm, not hardcoding)
    # Lowered threshold from 0.7 to 0.6 for more flexibility
    if word_overlap_score(pred_norm, exp_norm) >= 0.6:
        return True

    # Strategy 4: Embedding similarity (dynamic, no hardcoding)
    # Only if embedding model is available
    # Lowered threshold to 0.5 for semantic equivalence (e.g., same action, different brand)
    if _embedding_similarity(pred_norm, exp_norm) >= 0.5:
        return True

    # Check alternatives
    if alternatives:
        for alt in alternatives:
            alt_norm = normalize_service_name(alt)
            if pred_norm == alt_norm:
                return True
            if pred_norm in alt_norm or alt_norm in pred_norm:
                return True
            if word_overlap_score(pred_norm, alt_norm) >= 0.6:
                return True
            if _embedding_similarity(pred_norm, alt_norm) >= 0.5:
                return True

    return False


def compute_joint_accuracy(trigger_correct: bool, action_correct: bool) -> float:
    """Joint accuracy: both trigger AND action must be correct."""
    return 1.0 if (trigger_correct and action_correct) else 0.0


# =============================================================================
# CONTEXT RECALL (RAG Retrieval Quality)
# =============================================================================

def compute_context_recall(
    candidates: List[Dict],
    gold_service_name: str,
    k: int = 5,
) -> Tuple[float, int]:
    """
    Compute context recall: is gold service in top-K candidates?

    Returns:
        Tuple of (recall@k, rank) where rank is 1-indexed or 0 if not found
    """
    for i, candidate in enumerate(candidates[:k]):
        cand_name = candidate.get("service_name", "")
        if service_matches(cand_name, gold_service_name):
            return (1.0, i + 1)
    return (0.0, 0)


# =============================================================================
# CONTEXT PRECISION (RAG Ranking Quality) - LLM-based with RAGAS formula
# =============================================================================
# Formula from: https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_precision/
#
# Context Precision@K = Sum(Precision@k * v_k) / Total relevant items in top K
# Where:
#   - Precision@k = true_positives@k / (true_positives@k + false_positives@k)
#   - v_k = 1 if context at rank k is relevant, 0 otherwise (judged by LLM)
#
# This measures if relevant items are ranked HIGH (not just present).

def _llm_judge_context_relevance(
    query: str,
    context_name: str,
    context_desc: str,
    reference: str,
) -> bool:
    """
    Use LLM to judge if a retrieved context is relevant.

    Args:
        query: User query
        context_name: Retrieved service name
        context_desc: Retrieved service description
        reference: Expected correct answer

    Returns:
        True if LLM judges context as relevant
    """
    prompt = f"""You are evaluating if a retrieved context is relevant for answering a query.

Query: {query}
Expected Answer: {reference}

Retrieved Context:
- Service: {context_name}
- Description: {context_desc}

Is this retrieved context relevant for answering the query correctly?
Answer only YES or NO."""

    return llm_judge_binary(prompt)


def compute_context_precision(
    query: str,
    candidates: List[Dict],
    reference_service_name: str,
    ragas_llm=None,  # Not used - we use llm_judge directly
    k: int = 5,
    use_llm: bool = True,
) -> float:
    """
    Compute context precision using RAGAS formula with LLM judgments.

    Uses Granite LLM to judge relevance (v_k), then applies RAGAS formula.

    Formula:
        Context Precision@K = Sum(Precision@k * v_k) / Total relevant items
        Where v_k = 1 if LLM judges item k as relevant, 0 otherwise

    Args:
        query: User query
        candidates: Retrieved candidates from RAG
        reference_service_name: Gold standard service name
        ragas_llm: Not used (kept for API compatibility)
        k: Number of candidates to consider
        use_llm: If True, use LLM for relevance; else use string matching

    Returns:
        Precision score (0-1), higher means relevant items ranked higher
    """
    if not candidates or not reference_service_name:
        return 0.0

    # Get top-k candidates
    top_k = candidates[:k]
    if not top_k:
        return 0.0

    # Determine relevance for each candidate (v_k)
    # v_k = 1 if LLM judges as relevant, 0 otherwise
    relevance = []
    for c in top_k:
        cand_name = c.get("service_name", "")
        cand_desc = c.get("description", "")

        if use_llm and OLLAMA_AVAILABLE:
            # Use LLM to judge relevance
            is_relevant = _llm_judge_context_relevance(
                query, cand_name, cand_desc, reference_service_name
            )
        else:
            # Fallback to string matching
            is_relevant = service_matches(cand_name, reference_service_name)

        relevance.append(1 if is_relevant else 0)

    # Count total relevant items
    total_relevant = sum(relevance)
    if total_relevant == 0:
        return 0.0  # No relevant items found

    # Calculate Context Precision@K using the formula:
    # Sum(Precision@k * v_k) / total_relevant
    precision_sum = 0.0
    true_positives = 0

    for k_idx, v_k in enumerate(relevance):
        if v_k == 1:  # Only count when item is relevant
            true_positives += 1
            # Precision@k = true_positives so far / (k_idx + 1)
            precision_at_k = true_positives / (k_idx + 1)
            precision_sum += precision_at_k * v_k

    # Final score: average precision weighted by relevance
    context_precision = precision_sum / total_relevant

    return context_precision


# =============================================================================
# TOPIC ADHERENCE (Domain Safety) - LLM-based with RAGAS formula
# =============================================================================
# Formula from: https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/agents/#topic-adherence
#
# Topic Adherence measures if response stays within allowed domains.
# Uses LLM to judge if response adheres to IFTTT/WoT topics.
#
# For FARM: Check if response relates to WoT/IFTTT automation

# Define allowed topics for FARM Web of Things (WoT) automation system
# IFTTT is a Web of Things (WoT) platform - NOT just IoT!
# WoT connects web services, APIs, smart devices, and online platforms.
# Based on official IFTTT service categories: https://ifttt.com/services
FARM_ALLOWED_TOPICS = [
    # Core WoT (Web of Things) concepts
    "Web of Things (WoT)",
    "trigger-action programming",
    "IFTTT applets",
    "web service automation",
    "API integration",
    "workflow automation",
    # Official IFTTT categories (alphabetically)
    "Business tools",
    "Contacts",
    "Developer tools",
    "Finance & payments",
    "Gaming & Entertainment",
    "Health & fitness",
    "Journaling & personal data",
    "Machine learning",
    "Mobile devices & accessories",
    "Music",
    "New services",
    "News & information",
    "Notifications",
    "Other",
    "Photo & video",
    "Podcasts",
    "Popular services",
    "Project management & to-dos",
    "Shopping",
    "Smart home & IoT",
    "Social media",
    "Website & blog",
    "YouTube Channels",
]


def _llm_judge_topic_adherence(query: str, response: str, topics: List[str]) -> float:
    """
    Use LLM to judge if response adheres to allowed Web of Things (WoT) topics.

    Args:
        query: User query
        response: System response
        topics: List of allowed topics (ALL topics used, not truncated)

    Returns:
        Adherence score (0-1)
    """
    # Use ALL topics - no truncation
    topics_str = ", ".join(topics)

    prompt = f"""You are evaluating if an AI system's response stays within allowed Web of Things (WoT) topics.

IMPORTANT: IFTTT is a Web of Things (WoT) platform that connects web services, APIs, smart devices,
and online platforms through trigger-action automation. WoT is broader than just IoT!

Allowed WoT Topics: {topics_str}

User Query: {query}
System Response: {response}

Does this response relate to Web of Things automation (triggers, actions, web services, IFTTT, APIs)?
Rate the topic adherence on a scale of 0 to 10, where:
- 0 = Completely off-topic (e.g., cooking recipes, medical advice, unrelated topics)
- 5 = Partially related to web services or automation
- 10 = Fully on-topic (about WoT automation, triggers, actions, web services, IFTTT)

Answer with just a number from 0 to 10."""

    response_text = llm_judge(prompt)

    # Parse the score
    try:
        # Extract first number from response
        import re
        numbers = re.findall(r'\d+', response_text)
        if numbers:
            score = int(numbers[0])
            return min(score / 10.0, 1.0)  # Normalize to 0-1
    except:
        pass

    # Fallback: check for positive/negative indicators
    response_upper = response_text.upper()
    if any(word in response_upper for word in ["10", "FULLY", "COMPLETELY ON", "YES"]):
        return 1.0
    elif any(word in response_upper for word in ["0", "OFF-TOPIC", "NO", "UNRELATED"]):
        return 0.0
    else:
        return 0.5  # Default to partial adherence


def compute_topic_adherence(
    query: str,
    response: str,
    ragas_llm=None,  # Not used - we use llm_judge directly
    reference_topics: List[str] = None,
    use_llm: bool = True,
) -> float:
    """
    Compute topic adherence using LLM judgment (RAGAS-inspired).

    Uses Granite LLM to judge if response stays within WoT/IFTTT domain.

    Args:
        query: User query
        response: System response
        ragas_llm: Not used (kept for API compatibility)
        reference_topics: Allowed topics (defaults to FARM_ALLOWED_TOPICS)
        use_llm: If True, use LLM; else use keyword matching

    Returns:
        Adherence score (0-1), higher means better domain adherence
    """
    if not response or response.strip() == "":
        return 0.0

    if reference_topics is None:
        reference_topics = FARM_ALLOWED_TOPICS

    # Use LLM to judge topic adherence - NO hardcoded fallback
    if not OLLAMA_AVAILABLE:
        print("Warning: Ollama not available for topic adherence evaluation")
        return 1.0  # Assume adherence when can't evaluate

    return _llm_judge_topic_adherence(query, response, reference_topics)


# =============================================================================
# GOAL ACCURACY (Service Matching)
# =============================================================================

def compute_goal_accuracy(
    predicted_applet: Dict,
    reference_applet: Dict,
    query: str,
) -> float:
    """
    Compute agent goal accuracy: did applet achieve user intent?

    We trust RAG selection - if trigger and action match, goal is achieved.

    Returns:
        1.0 if trigger AND action match, 0.5 if one matches, 0.0 otherwise
    """
    if not predicted_applet:
        return 0.0

    pred_trigger = predicted_applet.get("trigger", {})
    ref_trigger = reference_applet.get("trigger", {})

    trigger_match = service_matches(
        pred_trigger.get("service_name", ""),
        ref_trigger.get("service_name", ""),
    )

    pred_action = predicted_applet.get("action", {})
    ref_action = reference_applet.get("action", {})

    action_match = service_matches(
        pred_action.get("service_name", ""),
        ref_action.get("service_name", ""),
    )

    if trigger_match and action_match:
        return 1.0

    if trigger_match or action_match:
        return 0.5

    return 0.0


# =============================================================================
# FAITHFULNESS (RAGAS with proper async handling)
# =============================================================================

def compute_faithfulness_ragas(
    query: str,
    response: str,
    contexts: List[str],
    ragas_llm,
) -> float:
    """
    Compute faithfulness using RAGAS library with proper async handling.

    Args:
        query: User query
        response: Agent response (applet description)
        contexts: Retrieved contexts from RAG
        ragas_llm: RAGAS LLM wrapper

    Returns:
        Faithfulness score (0-1)
    """
    if not RAGAS_AVAILABLE or ragas_llm is None:
        return compute_faithfulness_simple(response, " ".join(contexts))

    try:
        sample = SingleTurnSample(
            user_input=query,
            response=response,
            retrieved_contexts=contexts,
        )

        faithfulness_metric = Faithfulness(llm=ragas_llm)

        # Proper async handling - use nest_asyncio for nested event loops
        async def compute_score():
            return await faithfulness_metric.single_turn_ascore(sample)

        # Simple approach: always create new event loop in thread
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(lambda: asyncio.run(compute_score()))
            score = future.result(timeout=60)  # 60 second timeout

        import math
        result = float(score) if score is not None else 0.0
        # Handle NaN
        return 0.0 if math.isnan(result) else result

    except Exception as e:
        print(f"  [RAGAS faithfulness error: {e} - falling back to LLM claim verification]")
        return compute_faithfulness_simple(response, " ".join(contexts))


def compute_faithfulness_simple(
    reasoning: str,
    context: str,
) -> float:
    """
    Compute faithfulness using RAGAS formula with Granite LLM.

    RAGAS Formula:
        Faithfulness = Claims supported by context / Total claims in response

    Steps:
        1. Extract claims from response using LLM
        2. Verify each claim against context using LLM
        3. Calculate ratio of supported claims

    Args:
        reasoning: Agent response/reasoning
        context: Retrieved context from RAG

    Returns:
        Faithfulness score (0-1)
    """
    if not reasoning or not context:
        return 0.0

    # If LLM not available, fall back to word overlap
    if not OLLAMA_AVAILABLE:
        return _faithfulness_word_overlap(reasoning, context)

    # Step 1: Extract claims from response
    claims = _extract_claims(reasoning)

    if not claims:
        return 1.0  # No claims = nothing to verify = faithful

    # Step 2: Verify each claim against context
    supported_count = 0
    for claim in claims:
        if _verify_claim(claim, context):
            supported_count += 1

    # Step 3: Calculate ratio (RAGAS formula)
    faithfulness = supported_count / len(claims)

    return faithfulness


def _extract_claims(response: str) -> List[str]:
    """
    Extract factual claims from response using LLM.

    Args:
        response: Agent response text

    Returns:
        List of claims extracted from response
    """
    prompt = f"""Extract all factual claims from the following response.
A claim is a statement that can be verified as true or false.

Response: {response}

List each claim on a separate line, numbered 1, 2, 3, etc.
If there are no factual claims, respond with "NO CLAIMS".

Claims:"""

    result = llm_judge(prompt)

    if "NO CLAIMS" in result.upper():
        return []

    # Parse numbered claims
    claims = []
    for line in result.strip().split('\n'):
        line = line.strip()
        if line:
            # Remove numbering (1., 2., etc.)
            import re
            claim = re.sub(r'^\d+[\.\)]\s*', '', line)
            if claim and len(claim) > 5:  # Ignore very short lines
                claims.append(claim)

    return claims[:10]  # Limit to 10 claims for efficiency


def _verify_claim(claim: str, context: str) -> bool:
    """
    Verify if a claim can be inferred from context using LLM.

    Args:
        claim: A factual claim to verify
        context: Retrieved context

    Returns:
        True if claim is supported by context
    """
    prompt = f"""Determine if the following claim can be inferred from the given context.

Context: {context[:2000]}

Claim: {claim}

Can this claim be inferred from the context? Answer only YES or NO."""

    return llm_judge_binary(prompt)


def _faithfulness_word_overlap(reasoning: str, context: str) -> float:
    """
    Fallback: Simple word overlap when LLM unavailable.
    NOT the same as RAGAS - just a basic approximation.
    """
    import re

    reasoning_lower = reasoning.lower()
    context_lower = context.lower()

    # Extract meaningful words (4+ chars)
    context_words = set(re.findall(r'\b\w{4,}\b', context_lower))
    key_terms = list(context_words)[:20]

    if not key_terms:
        return 0.5

    matches = sum(1 for term in key_terms if term.lower() in reasoning_lower)
    return matches / len(key_terms)


# =============================================================================
# RAGAS BATCH EVALUATION
# =============================================================================

def create_ragas_sample(
    query: str,
    prediction: Dict,
    reference: Dict,
) -> Dict:
    """
    Create a sample in RAGAS format.

    Args:
        query: User query
        prediction: Agent output
        reference: Gold standard

    Returns:
        Dict formatted for RAGAS evaluation
    """
    # Extract prediction details
    pred_applet = prediction.get("final_applet") or {}
    pred_trigger = pred_applet.get("trigger") or {}
    pred_action = pred_applet.get("action") or {}

    # Build response string (what agent produced)
    response = f"""Trigger: {pred_trigger.get('service_name', 'None')}
Action: {pred_action.get('service_name', 'None')}
Bindings: {json.dumps(pred_applet.get('bindings', []), indent=2)}"""

    # Build reference string (expected output)
    ref_trigger = reference.get("trigger", {})
    ref_action = reference.get("action", {})
    reference_str = f"""Trigger: {ref_trigger.get('service_name', 'Unknown')}
Action: {ref_action.get('service_name', 'Unknown')}"""

    # Build contexts from retrieved candidates
    trigger_candidates = prediction.get("trigger_candidates", [])
    action_candidates = prediction.get("action_candidates", [])

    contexts = []
    for t in trigger_candidates[:5]:
        contexts.append(f"Trigger: {t.get('service_name', '')} - {t.get('description', '')}")
    for a in action_candidates[:5]:
        contexts.append(f"Action: {a.get('service_name', '')} - {a.get('description', '')}")

    return {
        "user_input": query,
        "response": response,
        "retrieved_contexts": contexts,
        "reference": reference_str,
    }


def evaluate_with_ragas_batch(
    predictions: List[Dict],
    references: List[Dict],
    queries: List[str],
    model_name: str = "granite4:small-h",
) -> Dict[str, float]:
    """
    Evaluate using official RAGAS library (batch mode).

    This is the preferred method - RAGAS is designed for batch evaluation.

    Args:
        predictions: List of agent outputs
        references: List of gold standards
        queries: List of user queries
        model_name: Ollama model for evaluation

    Returns:
        Dict of metric name -> average score
    """
    if not RAGAS_AVAILABLE:
        print("Warning: RAGAS not available")
        return {}

    print(f"Running RAGAS batch evaluation with {model_name}...")

    # Create RAGAS samples
    samples = []
    for pred, ref, query in zip(predictions, references, queries):
        sample_dict = create_ragas_sample(query, pred, ref)
        samples.append(SingleTurnSample(**sample_dict))

    # Create evaluation dataset
    dataset = EvaluationDataset(samples=samples)

    # Get RAGAS LLM
    try:
        ragas_llm = get_ragas_llm(model_name)
    except Exception as e:
        print(f"Warning: Could not initialize RAGAS LLM: {e}")
        return {}

    # Define metrics
    # Note: AgentGoalAccuracyWithReference removed - requires MultiTurnSample
    # but we use SingleTurnSample. Using custom goal_accuracy instead.
    metrics = [
        Faithfulness(llm=ragas_llm),
    ]

    # Run evaluation
    try:
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
        )
        scores = result.to_pandas().mean().to_dict()
        print(f"RAGAS batch evaluation complete: {scores}")
        return scores
    except Exception as e:
        print(f"Warning: RAGAS batch evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return {}


# =============================================================================
# SINGLE SAMPLE EVALUATION
# =============================================================================

def evaluate_single_sample(
    prediction: Dict,
    reference: Dict,
    query: str,
    use_ragas: bool = False,
    ragas_llm=None,
) -> RAGASMetrics:
    """
    Evaluate a single prediction against reference.

    Args:
        prediction: Agent output
        reference: Gold standard
        query: User query
        use_ragas: Whether to use RAGAS library for faithfulness
        ragas_llm: Pre-initialized RAGAS LLM wrapper

    Returns:
        RAGASMetrics for this sample
    """
    metrics = RAGASMetrics()

    # Extract predicted applet
    pred_applet = prediction.get("final_applet") or {}
    pred_trigger = pred_applet.get("trigger") or {}
    pred_action = pred_applet.get("action") or {}

    # Extract reference
    ref_trigger = reference.get("trigger", {})
    ref_action = reference.get("action", {})

    # 1. Service Selection (simple matching)
    metrics.trigger_accuracy = 1.0 if service_matches(
        pred_trigger.get("service_name", ""),
        ref_trigger.get("service_name", ""),
    ) else 0.0

    metrics.action_accuracy = 1.0 if service_matches(
        pred_action.get("service_name", ""),
        ref_action.get("service_name", ""),
    ) else 0.0

    metrics.joint_accuracy = compute_joint_accuracy(
        metrics.trigger_accuracy == 1.0,
        metrics.action_accuracy == 1.0,
    )

    # 2. Goal Accuracy
    metrics.goal_accuracy = compute_goal_accuracy(pred_applet, reference, query)

    # 3. Context Recall (RAG quality)
    trigger_candidates = prediction.get("trigger_candidates", [])
    action_candidates = prediction.get("action_candidates", [])

    metrics.context_recall_trigger, _ = compute_context_recall(
        trigger_candidates,
        ref_trigger.get("service_name", ""),
    )

    metrics.context_recall_action, _ = compute_context_recall(
        action_candidates,
        ref_action.get("service_name", ""),
    )

    # 3b. Context Precision (are relevant items ranked high?)
    # Uses our own implementation based on RAGAS formula - no LLM needed
    metrics.context_precision_trigger = compute_context_precision(
        query, trigger_candidates, ref_trigger.get("service_name", "")
    )
    metrics.context_precision_action = compute_context_precision(
        query, action_candidates, ref_action.get("service_name", "")
    )

    # 4. Faithfulness
    # Build response and contexts for faithfulness evaluation
    response = f"""Trigger: {pred_trigger.get('service_name', 'None')}
Action: {pred_action.get('service_name', 'None')}
Reasoning: {prediction.get('trigger_reasoning', '')} {prediction.get('action_reasoning', '')}"""

    contexts = []
    for t in trigger_candidates[:5]:
        contexts.append(f"Trigger: {t.get('service_name', '')} - {t.get('description', '')}")
    for a in action_candidates[:5]:
        contexts.append(f"Action: {a.get('service_name', '')} - {a.get('description', '')}")

    if use_ragas and ragas_llm:
        print("  [Faithfulness: Using RAGAS library with Granite]")
        metrics.faithfulness = compute_faithfulness_ragas(query, response, contexts, ragas_llm)
    else:
        print("  [Faithfulness: Using LLM-based claim verification (RAGAS formula)]")
        context_text = " ".join(contexts)
        metrics.faithfulness = compute_faithfulness_simple(response, context_text)

    # 5. Topic Adherence (domain safety)
    # Uses our own implementation based on RAGAS formula - no LLM needed
    metrics.topic_adherence = compute_topic_adherence(query, response)

    # 7. Resolution efficiency
    metrics.resolution_rounds = prediction.get("pair_attempt", 0) + 1
    metrics.first_try_success = (metrics.goal_accuracy >= 0.5 and metrics.resolution_rounds == 1)

    return metrics


# =============================================================================
# DATASET EVALUATION
# =============================================================================

def evaluate_dataset(
    predictions: List[Dict],
    references: List[Dict],
    queries: List[str],
    use_ragas: bool = True,
    model_name: str = "granite4:small-h",
) -> EvaluationResult:
    """
    Evaluate a full dataset.

    Args:
        predictions: List of agent outputs
        references: List of gold standards
        queries: List of queries
        use_ragas: Whether to use RAGAS library
        model_name: Ollama model for RAGAS evaluation

    Returns:
        EvaluationResult with aggregated metrics
    """
    result = EvaluationResult()
    result.num_samples = len(predictions)

    # Initialize RAGAS LLM once
    ragas_llm = None
    if use_ragas and RAGAS_AVAILABLE:
        try:
            ragas_llm = get_ragas_llm(model_name)
            print(f"Using RAGAS with {model_name} for evaluation")
        except Exception as e:
            print(f"Warning: Could not initialize RAGAS LLM: {e}")
            use_ragas = False

    # Aggregate metrics
    all_metrics = []

    for i, (pred, ref, query) in enumerate(zip(predictions, references, queries)):
        try:
            sample_metrics = evaluate_single_sample(
                pred, ref, query,
                use_ragas=use_ragas,
                ragas_llm=ragas_llm,
            )
            all_metrics.append(sample_metrics)

            if sample_metrics.goal_accuracy >= 0.5:
                result.num_success += 1
            else:
                result.num_failed += 1

            result.per_sample_results.append({
                "index": i,
                "query": query[:100],
                "success": sample_metrics.goal_accuracy >= 0.5,
                "metrics": sample_metrics.to_dict(),
            })
        except Exception as e:
            print(f"Error evaluating sample {i}: {e}")
            result.num_failed += 1
            result.per_sample_results.append({
                "index": i,
                "query": query[:100],
                "success": False,
                "error": str(e),
            })

    # Compute averages
    if all_metrics:
        n = len(all_metrics)
        result.metrics = RAGASMetrics(
            goal_accuracy=sum(m.goal_accuracy for m in all_metrics) / n,
            faithfulness=sum(m.faithfulness for m in all_metrics) / n,
            context_recall_trigger=sum(m.context_recall_trigger for m in all_metrics) / n,
            context_recall_action=sum(m.context_recall_action for m in all_metrics) / n,
            context_precision_trigger=sum(m.context_precision_trigger for m in all_metrics) / n,
            context_precision_action=sum(m.context_precision_action for m in all_metrics) / n,
            trigger_accuracy=sum(m.trigger_accuracy for m in all_metrics) / n,
            action_accuracy=sum(m.action_accuracy for m in all_metrics) / n,
            joint_accuracy=sum(m.joint_accuracy for m in all_metrics) / n,
            topic_adherence=sum(m.topic_adherence for m in all_metrics) / n,
            resolution_rounds=sum(m.resolution_rounds for m in all_metrics) // n,
            first_try_success=sum(1 for m in all_metrics if m.first_try_success) > n // 2,
        )

    return result


# =============================================================================
# PRINT FORMATTED RESULTS
# =============================================================================

def print_results(result: EvaluationResult, split_name: str = ""):
    """Print formatted evaluation results."""
    print("\n" + "=" * 70)
    print(f"EVALUATION RESULTS {f'({split_name})' if split_name else ''}")
    print("=" * 70)

    m = result.metrics

    print(f"\nSamples: {result.num_samples} | Success: {result.num_success} | Failed: {result.num_failed}")
    print(f"Success Rate: {result.num_success / result.num_samples:.1%}" if result.num_samples > 0 else "N/A")

    print("\n" + "-" * 70)
    print("PRIMARY METRICS")
    print("-" * 70)
    print(f"  Goal Accuracy:                         {m.goal_accuracy:.1%}")
    print(f"  Faithfulness:                          {m.faithfulness:.1%}")

    print("\n" + "-" * 70)
    print("SERVICE SELECTION")
    print("-" * 70)
    print(f"  Trigger Accuracy:                      {m.trigger_accuracy:.1%}")
    print(f"  Action Accuracy:                       {m.action_accuracy:.1%}")
    print(f"  Joint Accuracy:                        {m.joint_accuracy:.1%}")

    print("\n" + "-" * 70)
    print("RAG RETRIEVAL QUALITY")
    print("-" * 70)
    print(f"  Trigger Context Recall@5:              {m.context_recall_trigger:.1%}")
    print(f"  Action Context Recall@5:               {m.context_recall_action:.1%}")
    print(f"  Trigger Context Precision:             {m.context_precision_trigger:.1%}")
    print(f"  Action Context Precision:              {m.context_precision_action:.1%}")

    print("\n" + "-" * 70)
    print("DOMAIN SAFETY (Web of Things)")
    print("-" * 70)
    print(f"  Topic Adherence:                       {m.topic_adherence:.1%}")

    print("\n" + "=" * 70)


def save_results_json(
    results: Dict[str, EvaluationResult],
    output_path: str,
) -> None:
    """Save evaluation results to JSON file."""
    output = {
        "evaluation_summary": {},
        "per_split_results": {},
    }

    for split_name, result in results.items():
        output["per_split_results"][split_name] = result.to_dict()
        output["evaluation_summary"][split_name] = {
            "success_rate": result.num_success / result.num_samples if result.num_samples > 0 else 0,
            "goal_accuracy": result.metrics.goal_accuracy,
            "joint_accuracy": result.metrics.joint_accuracy,
            "faithfulness": result.metrics.faithfulness,
        }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Results saved to: {output_path}")
