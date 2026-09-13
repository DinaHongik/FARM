"""
Memory Checkpointer for FARM Agentic System
============================================

Provides checkpointer setup for LangGraph state persistence.
Supports both in-memory and SQLite-based persistence.
"""

from typing import Optional
from pathlib import Path

from langgraph.checkpoint.memory import MemorySaver


# Global checkpointer instance
_checkpointer: Optional[MemorySaver] = None


def get_checkpointer(
    persist_path: Optional[str] = None,
    force_new: bool = False
) -> MemorySaver:
    """
    Get or create the checkpointer instance.

    Uses in-memory storage by default. For persistent storage,
    provide a path (future SQLite support).

    Args:
        persist_path: Optional path for persistent storage (not yet implemented)
        force_new: If True, create a new checkpointer even if one exists

    Returns:
        MemorySaver checkpointer instance
    """
    global _checkpointer

    if _checkpointer is None or force_new:
        # Currently using in-memory saver
        # Future: Add SQLite support for persistence across sessions
        _checkpointer = MemorySaver()

    return _checkpointer


def reset_checkpointer():
    """
    Reset the global checkpointer.

    Clears all stored state. Useful for testing.
    """
    global _checkpointer
    _checkpointer = None


def get_thread_state(thread_id: str) -> Optional[dict]:
    """
    Retrieve the state for a specific thread.

    Args:
        thread_id: Thread identifier

    Returns:
        State dictionary or None if not found
    """
    checkpointer = get_checkpointer()

    try:
        config = {"configurable": {"thread_id": thread_id}}
        checkpoint = checkpointer.get(config)
        if checkpoint:
            return checkpoint.get("channel_values", {})
    except Exception:
        pass

    return None


def list_threads() -> list:
    """
    List all thread IDs in the checkpointer.

    Note: This is a simplified implementation. The actual
    MemorySaver doesn't expose a list method, so this returns
    an empty list. Override if using custom storage.

    Returns:
        List of thread IDs
    """
    # MemorySaver doesn't expose stored keys
    # Would need custom implementation for this
    return []


class SessionManager:
    """
    Manages negotiation sessions with memory persistence.

    Provides higher-level session management on top of the
    raw checkpointer.
    """

    def __init__(self):
        """Initialize the session manager."""
        self._sessions = {}

    def create_session(self, session_id: str, query: str) -> dict:
        """
        Create a new negotiation session.

        Args:
            session_id: Unique session identifier
            query: User query for this session

        Returns:
            Session metadata
        """
        session = {
            "session_id": session_id,
            "query": query,
            "status": "created",
            "result": None,
        }
        self._sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> Optional[dict]:
        """
        Get session metadata.

        Args:
            session_id: Session identifier

        Returns:
            Session metadata or None
        """
        return self._sessions.get(session_id)

    def update_session(self, session_id: str, **kwargs):
        """
        Update session metadata.

        Args:
            session_id: Session identifier
            **kwargs: Fields to update
        """
        if session_id in self._sessions:
            self._sessions[session_id].update(kwargs)

    def complete_session(self, session_id: str, result: dict):
        """
        Mark session as complete with result.

        Args:
            session_id: Session identifier
            result: Final negotiation result
        """
        if session_id in self._sessions:
            self._sessions[session_id]["status"] = "completed"
            self._sessions[session_id]["result"] = result

    def list_sessions(self) -> list:
        """
        List all sessions.

        Returns:
            List of session metadata dictionaries
        """
        return list(self._sessions.values())


# Global session manager
_session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    """
    Get the global session manager.

    Returns:
        SessionManager instance
    """
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager
