#!/usr/bin/env python3
"""Round6 adapter: pinned Round5 transport with proposer abstention enabled."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


def _load_base() -> Any:
    path = (
        Path(__file__).resolve().parents[2]
        / "20260902T042236Z-farm-round5-selective-verifier"
        / "scripts"
        / "ollama_pair_adapter.py"
    )
    spec = importlib.util.spec_from_file_location("farm_round5_ollama_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load pinned Round5 Ollama adapter")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = _load_base()


class OllamaRound6Chooser(BASE.OllamaPairChooser):
    """Allow ABSTAIN in both proposal and verification phases."""

    @classmethod
    def _prepare_request(cls, request: Any) -> tuple[dict[str, Any], list[str], bool]:
        if not isinstance(request, Mapping):
            raise ValueError("request_invalid")
        BASE._assert_no_references(request)
        phase = request.get("phase")
        query = request.get("query")
        instruction = request.get("instruction")
        evidence_view = request.get("evidence_view")
        allow_abstain = request.get("allow_abstain")
        if phase not in BASE.PHASES:
            raise ValueError("phase_invalid")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query_invalid")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("instruction_invalid")
        if evidence_view not in BASE.EVIDENCE_VIEWS:
            raise ValueError("evidence_view_invalid")
        if not isinstance(allow_abstain, bool):
            raise ValueError("allow_abstain_invalid")
        cards = request.get("cards")
        if not isinstance(cards, Sequence) or isinstance(cards, (str, bytes)):
            raise ValueError("cards_invalid")
        if not 2 <= len(cards) <= 10:
            raise ValueError("card_count_outside_2_to_10")
        public_cards = [cls._public_card(card) for card in cards]
        card_ids = [card["card_id"] for card in public_cards]
        if len(card_ids) != len(set(card_ids)):
            raise ValueError("card_ids_not_unique")
        return {
            "query": query.strip(), "phase": phase, "instruction": instruction.strip(),
            "evidence_view": evidence_view, "cards": public_cards,
            "allow_abstain": allow_abstain,
        }, card_ids, allow_abstain


__all__ = ["OllamaRound6Chooser"]
