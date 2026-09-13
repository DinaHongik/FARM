"""Pytest setup for the wiring tests.

These run on the server working copy so they exercise the real modules, but they
make no LLM calls and use no GPU - everything here is pure routing logic.
"""
import sys
from pathlib import Path

import pytest

FARM_ROOT = Path("/raid/session/aicontents/farm")
if str(FARM_ROOT) not in sys.path:
    sys.path.insert(0, str(FARM_ROOT))


@pytest.fixture
def cfg():
    """The live config singleton, with every wiring flag restored afterwards."""
    from agents.config import config
    saved = {
        k: getattr(config, k, None)
        for k in ("use_scored_pair_order", "enable_quality_gate",
                  "max_quality_retries", "verifier_threshold")
    }
    yield config
    for k, v in saved.items():
        if v is not None:
            setattr(config, k, v)


@pytest.fixture
def base_state():
    """Minimal NegotiationState with 5 trigger and 5 action candidates.

    priority_queue is deliberately NOT in lexicographic order, so a test that
    passes under PAIR_ORDER must fail under the scored order and vice versa.
    """
    return {
        "query": "Turn on the light when the stock price rises",
        "pair_attempt": 0,
        "trigger_candidates": [{"service_name": f"T{i}", "category": "c",
                                "api_info": {}} for i in range(5)],
        "action_candidates": [{"service_name": f"A{i}", "category": "c",
                               "api_info": {}} for i in range(5)],
        "priority_queue": [(3, 2), (0, 4), (1, 1), (2, 0), (4, 3)],
        "negotiation_status": "rejected",
        "current_trigger_idx": 0,
        "current_action_idx": 0,
    }
