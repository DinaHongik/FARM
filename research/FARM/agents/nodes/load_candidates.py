"""
Load Candidates Node
====================

Loads top-K trigger and action candidates from the RAG system.
This node runs once at the start of each negotiation session.
"""

import sys
from pathlib import Path
from typing import Dict, Any

# Add project root to path for RAG import
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.state import NegotiationState
from agents.config import config


# Singleton retriever - load once, reuse for all queries
_retriever = None


def get_retriever():
    """Get or create the singleton FARMRetriever instance."""
    global _retriever
    if _retriever is None:
        from rag.retriever import FARMRetriever
        _retriever = FARMRetriever()
    return _retriever


def load_candidates_node(state: NegotiationState) -> Dict[str, Any]:
    """
    Load top-K candidates from RAG for both triggers and actions.

    This node:
    1. Takes the user query from state
    2. Queries the RAG system for triggers and actions
    3. Stores top-K candidates in state for negotiation

    Args:
        state: Current negotiation state

    Returns:
        State updates with loaded candidates
    """
    query = state["query"]
    top_k = config.top_k_candidates

    # Use full query for both trigger and action retrieval
    # RAG embeddings are trained on full queries - no need for fragile parsing
    # (Previous hardcoded separators like " to " broke queries like "light to green")

    try:
        # Use singleton retriever (loads once, reuses for all queries)
        retriever = get_retriever()

        # Search for triggers (use full query - trigger context is important)
        trigger_results = retriever.search_triggers(
            query=query,
            top_k=top_k,
            score_threshold=0.0  # Get all top-k, filter later if needed
        )

        # Search for actions (use full query - RAG handles semantic matching)
        action_results = retriever.search_actions(
            query=query,
            top_k=top_k,
            score_threshold=0.0
        )

        # Convert to candidate format
        trigger_candidates = [
            {
                "service_name": r.service_name,
                "category": r.category,
                "description": r.description,
                "api_info": r.api_info,
                "score": r.score,
            }
            for r in trigger_results
        ]

        action_candidates = [
            {
                "service_name": r.service_name,
                "category": r.category,
                "description": r.description,
                "api_info": r.api_info,
                "score": r.score,
            }
            for r in action_results
        ]

        # DEBUG: Show RAG retrieval summary
        if config.verbose:
            print(f"\n    [RAG] Retrieved {len(trigger_candidates)} triggers, {len(action_candidates)} actions")
            if trigger_candidates:
                top_t = trigger_candidates[0]
                print(f"    [RAG] Top trigger: {top_t['service_name']} (score={top_t['score']:.3f})")
            if action_candidates:
                top_a = action_candidates[0]
                print(f"    [RAG] Top action: {top_a['service_name']} (score={top_a['score']:.3f})")

        return {
            "trigger_candidates": trigger_candidates,
            "action_candidates": action_candidates,
            "negotiation_status": "negotiating",
        }

    except ImportError as e:
        # RAG not available - return error state
        return {
            "trigger_candidates": [],
            "action_candidates": [],
            "error": f"RAG system not available: {e}",
            "negotiation_status": "failed",
        }
    except Exception as e:
        return {
            "trigger_candidates": [],
            "action_candidates": [],
            "error": f"Failed to load candidates: {e}",
            "negotiation_status": "failed",
        }


def load_candidates_mock(state: NegotiationState) -> Dict[str, Any]:
    """
    Mock loader for testing without RAG.

    Provides sample candidates for development and testing.
    Data format matches real triggers_rag.json and actions_rag.json.

    Args:
        state: Current negotiation state

    Returns:
        State updates with mock candidates
    """
    # Mock trigger candidates (format matches triggers_rag.json)
    trigger_candidates = [
        {
            "service_name": "Darkness detected",
            "category": "Smart home & IoT",
            "description": "This trigger fires when darkness is detected by your sensor.",
            "api_info": {
                "main_description": "This trigger fires when darkness is detected by your sensor.",
                "Trigger fields": {
                    "status": "No fields for this trigger"
                },
                "Ingredients": {
                    "DetectedAt": {
                        "Slug": "DetectedAt",
                        "Filter code": "SmartSensor.darknessDetected.DetectedAt",
                        "Type": "String",
                        "Example": "January 15, 2024 at 06:30PM"
                    },
                    "DeviceName": {
                        "Slug": "DeviceName",
                        "Filter code": "SmartSensor.darknessDetected.DeviceName",
                        "Type": "String",
                        "Example": "Living Room Sensor"
                    },
                    "LuxLevel": {
                        "Slug": "LuxLevel",
                        "Filter code": "SmartSensor.darknessDetected.LuxLevel",
                        "Type": "Number",
                        "Example": "15"
                    }
                }
            },
            "score": 0.95,
        },
        {
            "service_name": "Light level changed",
            "category": "Smart home & IoT",
            "description": "This trigger fires when the light level changes significantly.",
            "api_info": {
                "main_description": "This trigger fires when the light level changes significantly.",
                "Trigger fields": {
                    "Threshold": {
                        "Label": "Threshold",
                        "Slug": "threshold",
                        "Type": "Number"
                    }
                },
                "Ingredients": {
                    "LightLevel": {
                        "Slug": "LightLevel",
                        "Filter code": "SmartSensor.lightChanged.LightLevel",
                        "Type": "Number",
                        "Example": "150"
                    },
                    "ChangedAt": {
                        "Slug": "ChangedAt",
                        "Filter code": "SmartSensor.lightChanged.ChangedAt",
                        "Type": "String",
                        "Example": "January 15, 2024 at 06:30PM"
                    },
                    "PreviousLevel": {
                        "Slug": "PreviousLevel",
                        "Filter code": "SmartSensor.lightChanged.PreviousLevel",
                        "Type": "Number",
                        "Example": "200"
                    },
                    "SensorLocation": {
                        "Slug": "SensorLocation",
                        "Filter code": "SmartSensor.lightChanged.SensorLocation",
                        "Type": "String",
                        "Example": "Kitchen"
                    }
                }
            },
            "score": 0.85,
        },
        {
            "service_name": "New video uploaded",
            "category": "YouTube Channels",
            "description": "This trigger fires when a new video is uploaded to a YouTube channel.",
            "api_info": {
                "main_description": "This trigger fires when a new video is uploaded to a YouTube channel.",
                "Trigger fields": {
                    "status": "No fields for this trigger"
                },
                "Ingredients": {
                    "EntryTitle": {
                        "Slug": "EntryTitle",
                        "Filter code": "YouTube.newVideo.EntryTitle",
                        "Type": "String",
                        "Example": "Amazing Tutorial Part 1"
                    },
                    "EntryUrl": {
                        "Slug": "EntryUrl",
                        "Filter code": "YouTube.newVideo.EntryUrl",
                        "Type": "String",
                        "Example": "https://youtube.com/watch?v=abc123"
                    },
                    "EntryAuthor": {
                        "Slug": "EntryAuthor",
                        "Filter code": "YouTube.newVideo.EntryAuthor",
                        "Type": "String",
                        "Example": "Tech Channel"
                    },
                    "EntryContent": {
                        "Slug": "EntryContent",
                        "Filter code": "YouTube.newVideo.EntryContent",
                        "Type": "String",
                        "Example": "In this video we explore..."
                    },
                    "EntryImageUrl": {
                        "Slug": "EntryImageUrl",
                        "Filter code": "YouTube.newVideo.EntryImageUrl",
                        "Type": "String",
                        "Example": "https://i.ytimg.com/vi/abc123/hqdefault.jpg"
                    },
                    "EntryPublished": {
                        "Slug": "EntryPublished",
                        "Filter code": "YouTube.newVideo.EntryPublished",
                        "Type": "String",
                        "Example": "December 5, 2025 at 10:00AM"
                    }
                }
            },
            "score": 0.75,
        },
    ]

    # Mock action candidates (format matches actions_rag.json)
    action_candidates = [
        {
            "service_name": "Add row to spreadsheet",
            "category": "Popular services",
            "description": "This action will add a single row to the bottom of the first worksheet of a spreadsheet you specify.",
            "api_info": {
                "main_description": "This action will add a single row to the bottom of the first worksheet of a spreadsheet you specify. Note: a new spreadsheet is created after 2000 rows.",
                "Action fields": {
                    "Spreadsheet name": {
                        "Label": "Spreadsheet name",
                        "Helper text": "Will create a new spreadsheet if one with this title doesn't exist",
                        "Slug": "filename",
                        "Required": "true",
                        "Can have default value": "true",
                        "Filter code method": "GoogleSheets.appendToGoogleSpreadsheet.setFilename(string: filename)"
                    },
                    "Formatted row": {
                        "Label": "Formatted row",
                        "Helper text": "Use ||| to separate cells",
                        "Slug": "formatted_row",
                        "Required": "true",
                        "Can have default value": "true",
                        "Filter code method": "GoogleSheets.appendToGoogleSpreadsheet.setFormattedRow(string: formattedRow)"
                    },
                    "Drive folder path": {
                        "Label": "Drive folder path",
                        "Helper text": "Format: some/folder/path (defaults to IFTTT)",
                        "Slug": "path",
                        "Required": "false",
                        "Can have default value": "true",
                        "Filter code method": "GoogleSheets.appendToGoogleSpreadsheet.setPath(string: path)"
                    }
                }
            },
            "score": 0.92,
        },
        {
            "service_name": "Send a notification from the IFTTT app",
            "category": "Popular services",
            "description": "This action will send a notification to your devices from the IFTTT app.",
            "api_info": {
                "main_description": "This action will send a notification to your devices from the IFTTT app.",
                "Action fields": {
                    "Message": {
                        "Label": "Message",
                        "Slug": "message",
                        "Required": "true",
                        "Can have default value": "true",
                        "Filter code method": "IfNotifications.sendNotification.setMessage(string: message)"
                    }
                }
            },
            "score": 0.78,
        },
        {
            "service_name": "Add to weekly email digest",
            "category": "Business tools",
            "description": "This Action will add an item to your weekly digest sent once a week on the day and time you specify.",
            "api_info": {
                "main_description": "This Action will add an item to your weekly digest sent once a week on the day and time you specify.",
                "Action fields": {
                    "Time of day": {
                        "Label": "Time of day",
                        "Slug": "time_of_day",
                        "Required": "true",
                        "Can have default value": "true",
                        "Filter code method": "EmailDigest.sendWeeklyEmail.setTimeOfDay(string: timeOfDay)"
                    },
                    "Day of week": {
                        "Label": "Day of week",
                        "Slug": "day_of_week",
                        "Required": "true",
                        "Can have default value": "true",
                        "Filter code method": "EmailDigest.sendWeeklyEmail.setDayOfWeek(string: dayOfWeek)"
                    },
                    "Title": {
                        "Label": "Title",
                        "Slug": "title",
                        "Required": "true",
                        "Can have default value": "true",
                        "Filter code method": "EmailDigest.sendWeeklyEmail.setTitle(string: title)"
                    },
                    "Message": {
                        "Label": "Message",
                        "Slug": "message",
                        "Required": "false",
                        "Can have default value": "true",
                        "Filter code method": "EmailDigest.sendWeeklyEmail.setMessage(string: message)"
                    },
                    "Item URL": {
                        "Label": "Item URL",
                        "Helper text": "Optional",
                        "Slug": "url",
                        "Required": "false",
                        "Can have default value": "true",
                        "Filter code method": "EmailDigest.sendWeeklyEmail.setUrl(string: url)"
                    }
                }
            },
            "score": 0.65,
        },
    ]

    return {
        "trigger_candidates": trigger_candidates,
        "action_candidates": action_candidates,
        "negotiation_status": "negotiating",
    }


# Global reference data for evaluation mode
_eval_reference_data = None


def set_eval_reference_data(test_data: list):
    """
    Set reference data for evaluation mode.

    This allows the loader to create realistic candidate lists
    from test data when RAG is unavailable.

    Args:
        test_data: List of test cases with trigger/action info
    """
    global _eval_reference_data
    _eval_reference_data = test_data


def load_candidates_from_reference(state: NegotiationState) -> dict:
    """
    Load candidates from reference test data (for evaluation without RAG).

    Creates candidate lists by:
    1. Finding the matching test case by query
    2. Placing the correct trigger/action in the list
    3. Adding distractors from other test cases

    Args:
        state: Current negotiation state

    Returns:
        State updates with candidates from reference data
    """
    global _eval_reference_data
    import random

    if not _eval_reference_data:
        return {
            "trigger_candidates": [],
            "action_candidates": [],
            "error": "No reference data set. Call set_eval_reference_data() first.",
            "negotiation_status": "failed",
        }

    query = state["query"]
    top_k = config.top_k_candidates

    # Find matching test case
    target_case = None
    for case in _eval_reference_data:
        if case.get("query") == query:
            target_case = case
            break

    if not target_case:
        return {
            "trigger_candidates": [],
            "action_candidates": [],
            "error": f"Query not found in reference data: {query[:50]}...",
            "negotiation_status": "failed",
        }

    # Get correct trigger and action
    correct_trigger = target_case.get("trigger", {})
    correct_action = target_case.get("action", {})

    # Collect distractor triggers/actions from other test cases
    distractor_triggers = []
    distractor_actions = []

    for case in _eval_reference_data:
        if case.get("query") != query:
            if case.get("trigger"):
                distractor_triggers.append(case["trigger"])
            if case.get("action"):
                distractor_actions.append(case["action"])

    # Shuffle distractors
    random.seed(42)  # Reproducible for evaluation
    random.shuffle(distractor_triggers)
    random.shuffle(distractor_actions)

    # Build candidate lists with correct answer at random position
    def build_candidates(correct: dict, distractors: list, k: int, seed_offset: int = 0) -> list:
        """Build candidate list with correct item at random position.

        The correct item is placed at a random position (0 to k-1) to simulate
        realistic RAG ranking where correct might not always be #1.
        However, it always gets the highest score for its position.
        """
        # Use top distractors
        selected = distractors[:k-1]

        # Determine position for correct answer (reproducible per query)
        random.seed(42 + seed_offset)
        correct_pos = random.randint(0, min(k-1, len(selected)))

        # Convert to candidate format with simulated RAG scores
        # Score decreases with position, but correct item gets a bonus
        candidates = []
        distractor_idx = 0

        for pos in range(k):
            if pos == correct_pos:
                # Correct answer at this position - give highest score
                candidates.append({
                    "service_name": correct.get("service_name", ""),
                    "category": correct.get("category", ""),
                    "description": correct.get("description", ""),
                    "api_info": correct.get("api_info", {}),
                    "score": 0.95 - (pos * 0.02),  # High score, slight decrease by position
                })
            elif distractor_idx < len(selected):
                # Distractor at this position
                item = selected[distractor_idx]
                candidates.append({
                    "service_name": item.get("service_name", ""),
                    "category": item.get("category", ""),
                    "description": item.get("description", ""),
                    "api_info": item.get("api_info", {}),
                    "score": 0.7 - (pos * 0.05),  # Lower base score
                })
                distractor_idx += 1

        return candidates[:k]

    # Use different seed offsets so trigger/action get different random positions
    trigger_candidates = build_candidates(correct_trigger, distractor_triggers, top_k, seed_offset=0)
    action_candidates = build_candidates(correct_action, distractor_actions, top_k, seed_offset=1000)

    return {
        "trigger_candidates": trigger_candidates,
        "action_candidates": action_candidates,
        "negotiation_status": "negotiating",
    }
