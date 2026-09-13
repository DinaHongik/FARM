#!/usr/bin/env python3
"""Pure, conservative complete-pair policy for FARM round five.

The module owns the difficult experimental invariants behind one interface:

* construct a bounded deck of complete trigger/action pairs without gold;
* expose opaque IDs and evidence, never retrieval ranks or catalog URLs;
* make at most one proposer call and two fresh verifier calls;
* change evidence from plain descriptions to schema-bearing catalog text;
* accept an override only after order-swapped verifier agreement; and
* retain retrieval top-1 on every failure, abstention, or disagreement.

Network behavior lives behind the injected :class:`CardChooser` seam.  The
policy is otherwise deterministic and has no dependency on an Ollama SDK.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Protocol, Sequence


Decision = Literal[
    "KEEP_TOP1",
    "ACCEPT_PROPOSAL",
    "CHOOSE_ALT",
    "ABSTAIN",
    "DISAGREE",
    "NOT_RUN",
]


class CardChooser(Protocol):
    """External-model adapter used by the pure policy.

    ``select`` returns ``ok``, one opaque ``choice_id``, exact non-negative
    ``api_attempts`` and ``tool_calls``, numeric ``usage``, and a safe ``error``
    code.  Verification adapters may return the literal ``ABSTAIN``.  Expected
    provider/protocol failures are values with ``ok=False``, not exceptions.
    """

    def select(self, request: dict[str, Any]) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class PairCardPolicy:
    """Frozen bounded policy; model confidence is intentionally absent."""

    card_count: Literal[5, 10]
    verify: bool = True
    candidate_depth: int = 10
    proposer_instruction: str = (
        "Select the one complete trigger-to-action pair that most exactly "
        "implements the request. Candidate IDs and order carry no relevance."
    )
    verifier_instruction: str = (
        "Independently verify the complete pairs against the request and the "
        "new catalog/schema evidence. Select one card only when the evidence "
        "supports it; otherwise abstain. Candidate IDs and order carry no relevance."
    )

    def __post_init__(self) -> None:
        if self.card_count not in {5, 10}:
            raise ValueError("card_count must be exactly 5 or 10")
        if isinstance(self.candidate_depth, bool) or self.candidate_depth != 10:
            raise ValueError("candidate_depth is frozen at 10")
        if not isinstance(self.verify, bool):
            raise TypeError("verify must be boolean")
        for name, value in (
            ("proposer_instruction", self.proposer_instruction),
            ("verifier_instruction", self.verifier_instruction),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")


_REFERENCE_KEYS = {
    "valid_pairs",
    "gold",
    "gold_pair",
    "gold_pairs",
    "gold_trigger_urls",
    "gold_action_urls",
    "gold_channel_pairs",
    "gold_service_pairs",
    "ground_truth",
    "reference",
    "references",
    "reference_answer",
    "expected_pair",
    "observed_pairs",
    "pair_rank",
}


def _reference_paths(value: Any, path: str = "case") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key).lower()
            child_path = f"{path}.{raw_key}"
            if (
                key in _REFERENCE_KEYS
                or key.startswith("gold_")
                or key.startswith("reference_")
                or key.startswith("ground_truth_")
            ):
                found.append(child_path)
            found.extend(_reference_paths(child, child_path))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            found.extend(_reference_paths(child, f"{path}[{index}]"))
    return found


@dataclass(frozen=True)
class _Candidate:
    identity: str
    service_name: str
    function_name: str
    plain_evidence: str
    schema_evidence: str
    rank: int


@dataclass(frozen=True)
class _Card:
    opaque_id: str
    trigger: _Candidate
    action: _Candidate

    @property
    def pair(self) -> dict[str, str]:
        return {
            "trigger_url": self.trigger.identity,
            "action_url": self.action.identity,
        }


def _text(
    item: Mapping[str, Any], keys: Sequence[str], *, fallback: str | None = None
) -> str:
    for key in keys:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    if fallback is not None:
        return fallback
    raise ValueError(f"candidate lacks required field alternatives: {tuple(keys)}")


def _candidate_rows(
    case: Mapping[str, Any], side: str, *, depth: int
) -> list[_Candidate]:
    raw = case.get(f"{side}_candidates")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError(f"case requires {side}_candidates")
    if len(raw) < depth:
        raise ValueError(f"policy requires at least {depth} {side} candidates")
    normalized: list[_Candidate] = []
    for index, value in enumerate(raw):
        if not isinstance(value, Mapping):
            raise TypeError(f"{side} candidate {index} is not an object")
        identity = _text(value, ("url", "id"))
        service_name = _text(
            value,
            ("service_name", "channel_display", "service_display", "channel", "service"),
        )
        function_name = _text(value, ("function_name", "name", "display"))
        plain = _text(value, ("text_plain", "text", "description"))
        schema = _text(value, ("text_schema", "schema"))
        raw_rank = value.get("retrieval_rank", index + 1)
        if isinstance(raw_rank, bool):
            raise TypeError(f"{side} retrieval_rank cannot be boolean")
        try:
            rank = int(raw_rank)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{side} retrieval_rank must be an integer") from exc
        if rank < 1:
            raise ValueError(f"{side} retrieval_rank must be positive")
        normalized.append(
            _Candidate(identity, service_name, function_name, plain, schema, rank)
        )
    normalized.sort(key=lambda item: (item.rank, item.identity))
    selected = normalized[:depth]
    identities = [item.identity for item in selected]
    if len(set(identities)) != len(identities):
        raise ValueError(f"duplicate {side} candidate identity in frozen top-{depth}")
    return selected


def _parse_pair(value: Mapping[str, Any] | str, *, index: int) -> tuple[str, str]:
    if isinstance(value, str):
        pieces = value.split(" || ")
        if len(pieces) != 2 or not all(piece.strip() for piece in pieces):
            raise ValueError(f"pair_ranking[{index}] is not 'trigger || action'")
        return pieces[0].strip(), pieces[1].strip()
    if not isinstance(value, Mapping):
        raise TypeError(f"pair_ranking[{index}] must be an object or string")
    if set(value) != {"trigger_url", "action_url"}:
        raise ValueError(
            f"pair_ranking[{index}] must contain only trigger_url and action_url"
        )
    trigger, action = value["trigger_url"], value["action_url"]
    if not isinstance(trigger, str) or not trigger.strip():
        raise ValueError(f"pair_ranking[{index}] has an invalid trigger_url")
    if not isinstance(action, str) or not action.strip():
        raise ValueError(f"pair_ranking[{index}] has an invalid action_url")
    return trigger.strip(), action.strip()


def _opaque_card_id(case_id: str, trigger: str, action: str) -> str:
    material = "\x00".join(
        ("farm-round5-pair-card-v1", case_id, trigger, action)
    ).encode("utf-8")
    return "pc_" + hashlib.sha256(material).hexdigest()[:24]


def _build_cards(
    case_id: str,
    triggers: Sequence[_Candidate],
    actions: Sequence[_Candidate],
    pair_ranking: Sequence[Mapping[str, Any] | str],
    *,
    card_count: int,
) -> list[_Card]:
    if not isinstance(pair_ranking, Sequence) or isinstance(pair_ranking, (str, bytes)):
        raise TypeError("pair_ranking must be a sequence")
    if not pair_ranking:
        raise ValueError("pair_ranking cannot be empty")
    trigger_by_id = {item.identity: item for item in triggers}
    action_by_id = {item.identity: item for item in actions}
    parsed = [_parse_pair(item, index=index) for index, item in enumerate(pair_ranking)]
    if len(set(parsed)) != len(parsed):
        raise ValueError("pair_ranking contains duplicate complete pairs")
    for trigger, action in parsed:
        if trigger not in trigger_by_id or action not in action_by_id:
            raise ValueError("pair_ranking contains a URL outside the frozen top-10 lattice")

    baseline = (triggers[0].identity, actions[0].identity)
    selected: list[tuple[str, str]] = []
    for pair in (baseline, *parsed):
        if pair not in selected:
            selected.append(pair)
        if len(selected) == card_count:
            break
    if len(selected) != card_count:
        raise ValueError(f"pair_ranking cannot supply {card_count} unique complete cards")
    cards = [
        _Card(
            _opaque_card_id(case_id, trigger, action),
            trigger_by_id[trigger],
            action_by_id[action],
        )
        for trigger, action in selected
    ]
    if len({card.opaque_id for card in cards}) != len(cards):
        raise ValueError("opaque pair-card ID collision")
    return cards


def _public_side(candidate: _Candidate, *, view: Literal["plain", "schema"]) -> dict[str, str]:
    return {
        "service_name": candidate.service_name,
        "function_name": candidate.function_name,
        "evidence": (
            candidate.plain_evidence if view == "plain" else candidate.schema_evidence
        ),
    }


def _public_card(card: _Card, *, view: Literal["plain", "schema"]) -> dict[str, Any]:
    return {
        "card_id": card.opaque_id,
        "trigger": _public_side(card.trigger, view=view),
        "action": _public_side(card.action, view=view),
    }


def _proposer_order(cards: Sequence[_Card], case_id: str) -> list[_Card]:
    """Blind source rank with a reproducible, case-specific permutation."""
    return sorted(
        cards,
        key=lambda card: hashlib.sha256(
            "\x00".join(
                ("farm-round5-proposer-order-v1", case_id, card.opaque_id)
            ).encode("utf-8")
        ).hexdigest(),
    )


def _request(
    *,
    case_id: str,
    query: str,
    phase: Literal["pair_proposal", "pair_verification"],
    instruction: str,
    cards: Sequence[_Card],
    view: Literal["plain", "schema"],
    allow_abstain: bool,
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "query": query,
        "phase": phase,
        "instruction": instruction,
        "evidence_view": view,
        "allow_abstain": allow_abstain,
        "cards": [_public_card(card, view=view) for card in cards],
    }


def _reported_count(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _safe_error(value: Any, fallback: str) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()[:160]
    return fallback


def _invoke(
    chooser: CardChooser,
    request: dict[str, Any],
    *,
    allowed_ids: set[str],
    allow_abstain: bool,
) -> tuple[dict[str, Any], str | None]:
    try:
        raw = chooser.select(request)
    except Exception as exc:
        return (
            {
                "phase": request["phase"],
                "request": request,
                "ok": False,
                "choice_id": None,
                "api_attempts": 0,
                "tool_calls": 0,
                "usage": {},
                "error": f"adapter_exception:{type(exc).__name__}",
                "accounting_complete": False,
            },
            None,
        )
    if not isinstance(raw, Mapping):
        return (
            {
                "phase": request["phase"],
                "request": request,
                "ok": False,
                "choice_id": None,
                "api_attempts": 0,
                "tool_calls": 0,
                "usage": {},
                "error": "adapter_contract:not_mapping",
                "accounting_complete": False,
            },
            None,
        )

    api_attempts = _reported_count(raw.get("api_attempts"))
    tool_calls = _reported_count(raw.get("tool_calls"))
    accounting_complete = api_attempts is not None and tool_calls is not None
    safe_attempts = api_attempts if api_attempts is not None else 0
    safe_tools = tool_calls if tool_calls is not None else 0
    raw_usage = raw.get("usage")
    usage = {
        str(key): value
        for key, value in (raw_usage.items() if isinstance(raw_usage, Mapping) else ())
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and value >= 0
    }
    choice = raw.get("choice_id")
    choice_allowed = isinstance(choice, str) and (
        choice in allowed_ids or (allow_abstain and choice == "ABSTAIN")
    )
    ok = raw.get("ok") is True and accounting_complete and choice_allowed
    if ok:
        error = None
        selected: str | None = choice
    else:
        if raw.get("ok") is True and not accounting_complete:
            fallback = "adapter_contract:missing_accounting"
        elif raw.get("ok") is True and not choice_allowed:
            fallback = "choice_id_outside_request"
        else:
            fallback = "chooser_failed"
        error = _safe_error(raw.get("error"), fallback)
        selected = None
    return (
        {
            "phase": request["phase"],
            "request": request,
            "ok": ok,
            "choice_id": selected,
            "api_attempts": safe_attempts,
            "tool_calls": safe_tools,
            "usage": usage,
            "error": error,
            "accounting_complete": accounting_complete,
        },
        selected,
    )


def _accounting(
    calls: Sequence[Mapping[str, Any]], *, catalog_reads: int, card_count: int
) -> dict[str, Any]:
    usage: dict[str, int | float] = {}
    for call in calls:
        observed = call.get("usage")
        if not isinstance(observed, Mapping):
            continue
        for key, value in observed.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                usage[str(key)] = usage.get(str(key), 0) + value
    verifier_calls = sum(call.get("phase") == "pair_verification" for call in calls)
    return {
        "logical_calls": len(calls),
        "api_attempts": sum(int(call.get("api_attempts", 0)) for call in calls),
        "tool_calls": sum(int(call.get("tool_calls", 0)) for call in calls),
        "catalog_reads": catalog_reads,
        "evidence_presentations": {
            "proposer_pair_cards": card_count,
            "proposer_endpoints": 2 * card_count,
            "verifier_pair_cards": 3 * verifier_calls,
            "verifier_endpoints": 6 * verifier_calls,
            "verifier_unique_schema_catalog_rows": catalog_reads,
        },
        "complete": all(bool(call.get("accounting_complete")) for call in calls),
        "usage": dict(sorted(usage.items())),
    }


def _decision(
    choice: str | None,
    *,
    baseline_id: str,
    proposal_id: str,
    alternative_id: str,
) -> str:
    if choice == baseline_id:
        return "KEEP_TOP1"
    if choice == proposal_id:
        return "ACCEPT_PROPOSAL"
    if choice == alternative_id:
        return "CHOOSE_ALT"
    if choice == "ABSTAIN":
        return "ABSTAIN"
    return "INVALID"


def _result(
    *,
    case_id: str,
    policy: PairCardPolicy,
    cards: Sequence[_Card],
    baseline: _Card,
    proposal: _Card | None,
    alternative: _Card | None,
    final: _Card,
    calls: Sequence[Mapping[str, Any]],
    verifier_decisions: Sequence[str],
    stable_decision: Decision,
    fallback_reason: str | None,
    catalog_reads: int,
) -> dict[str, Any]:
    return {
        "schema_version": "farm_round5_pair_card_trace_v1",
        "case_id": case_id,
        "policy": {
            "card_count": policy.card_count,
            "candidate_depth": policy.candidate_depth,
            "verify": policy.verify,
            "proposer_view": "plain",
            "verifier_view": "schema",
            "verifier_calls": 2 if policy.verify else 0,
            "safe_fallback": "retrieval_top1",
        },
        "baseline_pair": baseline.pair,
        "proposer_pair": proposal.pair if proposal is not None else None,
        "alternative_pair": alternative.pair if alternative is not None else None,
        "final_pair": final.pair,
        "retained_baseline": final.opaque_id == baseline.opaque_id,
        "card_ids": [card.opaque_id for card in cards],
        "candidate_pairs": [card.pair for card in cards],
        "verifier_decisions": list(verifier_decisions),
        "stable_decision": stable_decision,
        "fallback_reason": fallback_reason,
        "calls": [dict(call) for call in calls],
        "accounting": _accounting(
            calls, catalog_reads=catalog_reads, card_count=len(cards)
        ),
    }


def resolve_pair_cards(
    case: Mapping[str, Any],
    pair_ranking: Sequence[Mapping[str, Any] | str],
    policy: PairCardPolicy,
    proposer: CardChooser,
    verifier: CardChooser | None,
) -> dict[str, Any]:
    """Resolve one case through complete cards and conservative verification.

    ``pair_ranking`` must be an inference-only ranking of complete pairs over
    the frozen top-10 lattice.  The retrieval top-1 pair is forced into the
    card deck even if the ranker omitted it.  The proposer sees plain evidence.
    When it proposes an override, two fresh verifier calls see schema evidence
    in opposite presentation orders.  Only identical valid choices override
    top-1; no model confidence or first-call result controls call count.
    """
    contaminated = _reference_paths(case)
    if contaminated:
        raise ValueError(f"inference case contains reference fields: {sorted(contaminated)}")
    if not isinstance(policy, PairCardPolicy):
        raise TypeError("policy must be a PairCardPolicy")
    if policy.verify and verifier is None:
        raise ValueError("verified policy requires a verifier adapter")
    case_id = case.get("group_id", case.get("case_id", case.get("id")))
    query = case.get("query")
    if not isinstance(case_id, str) or not case_id.strip():
        raise ValueError("case requires a non-empty group_id or case_id")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("case requires a non-empty query")

    triggers = _candidate_rows(case, "trigger", depth=policy.candidate_depth)
    actions = _candidate_rows(case, "action", depth=policy.candidate_depth)
    cards = _build_cards(
        case_id,
        triggers,
        actions,
        pair_ranking,
        card_count=policy.card_count,
    )
    baseline = cards[0]
    by_id = {card.opaque_id: card for card in cards}
    proposal_request = _request(
        case_id=case_id,
        query=query,
        phase="pair_proposal",
        instruction=policy.proposer_instruction,
        cards=_proposer_order(cards, case_id),
        view="plain",
        allow_abstain=False,
    )
    proposal_call, proposal_id = _invoke(
        proposer,
        proposal_request,
        allowed_ids=set(by_id),
        allow_abstain=False,
    )
    calls: list[Mapping[str, Any]] = [proposal_call]
    if proposal_id is None:
        return _result(
            case_id=case_id,
            policy=policy,
            cards=cards,
            baseline=baseline,
            proposal=None,
            alternative=None,
            final=baseline,
            calls=calls,
            verifier_decisions=(),
            stable_decision="NOT_RUN",
            fallback_reason="proposer_protocol_failure",
            catalog_reads=0,
        )

    proposal = by_id[proposal_id]
    if proposal.opaque_id == baseline.opaque_id:
        return _result(
            case_id=case_id,
            policy=policy,
            cards=cards,
            baseline=baseline,
            proposal=proposal,
            alternative=None,
            final=baseline,
            calls=calls,
            verifier_decisions=(),
            stable_decision="NOT_RUN",
            fallback_reason=None,
            catalog_reads=0,
        )
    if not policy.verify:
        return _result(
            case_id=case_id,
            policy=policy,
            cards=cards,
            baseline=baseline,
            proposal=proposal,
            alternative=None,
            final=proposal,
            calls=calls,
            verifier_decisions=(),
            stable_decision="NOT_RUN",
            fallback_reason=None,
            catalog_reads=0,
        )

    alternative = next(
        card
        for card in cards
        if card.opaque_id not in {baseline.opaque_id, proposal.opaque_id}
    )
    verification_cards = [baseline, proposal, alternative]
    verifier_decisions: list[str] = []
    verifier_choices: list[str | None] = []
    if verifier is None:  # narrowed by the policy guard above
        raise AssertionError("verified policy lost its verifier adapter")
    # A reversal would leave the proposal in the middle twice. This rotation
    # makes every private role occupy a different position in the fresh call.
    for presented in (verification_cards, [alternative, baseline, proposal]):
        verification_request = _request(
            case_id=case_id,
            query=query,
            phase="pair_verification",
            instruction=policy.verifier_instruction,
            cards=presented,
            view="schema",
            allow_abstain=True,
        )
        call, choice = _invoke(
            verifier,
            verification_request,
            allowed_ids={card.opaque_id for card in verification_cards},
            allow_abstain=True,
        )
        calls.append(call)
        verifier_choices.append(choice)
        verifier_decisions.append(
            _decision(
                choice,
                baseline_id=baseline.opaque_id,
                proposal_id=proposal.opaque_id,
                alternative_id=alternative.opaque_id,
            )
        )

    unique_catalog_rows = {
        (side, candidate.identity)
        for card in verification_cards
        for side, candidate in (("trigger", card.trigger), ("action", card.action))
    }
    catalog_reads = len(unique_catalog_rows)
    first, second = verifier_choices
    if first is None or second is None:
        final = baseline
        stable: Decision = "DISAGREE"
        fallback_reason = "verifier_protocol_failure"
    elif first != second:
        final = baseline
        stable = "DISAGREE"
        fallback_reason = "verifier_disagreement"
    elif first == "ABSTAIN":
        final = baseline
        stable = "ABSTAIN"
        fallback_reason = "verifier_abstained"
    elif first == baseline.opaque_id:
        final = baseline
        stable = "KEEP_TOP1"
        fallback_reason = None
    elif first == proposal.opaque_id:
        final = proposal
        stable = "ACCEPT_PROPOSAL"
        fallback_reason = None
    elif first == alternative.opaque_id:
        final = alternative
        stable = "CHOOSE_ALT"
        fallback_reason = None
    else:  # allowed-ID validation above makes this unreachable
        raise AssertionError("validated verifier choice is unknown")

    return _result(
        case_id=case_id,
        policy=policy,
        cards=cards,
        baseline=baseline,
        proposal=proposal,
        alternative=alternative,
        final=final,
        calls=calls,
        verifier_decisions=verifier_decisions,
        stable_decision=stable,
        fallback_reason=fallback_reason,
        catalog_reads=catalog_reads,
    )


__all__ = ["CardChooser", "Decision", "PairCardPolicy", "resolve_pair_cards"]
