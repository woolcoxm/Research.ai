"""
Session Persistence Utility
Handles saving and loading research sessions to/from disk
"""
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from core.models import ResearchContext, SearchResult, LLMMessage

# Directory for storing session data
SESSIONS_DIR = Path(__file__).parent.parent / "saved_sessions"
SESSIONS_DIR.mkdir(exist_ok=True)


def _serialize_research_context(context: ResearchContext) -> Dict[str, Any]:
    """Convert ResearchContext to JSON-serializable dict"""
    return {
        "original_query": context.original_query,
        "refined_queries": context.refined_queries,
        "search_results": [
            {
                "title": sr.title,
                "link": sr.link,
                "snippet": sr.snippet,
                "relevance_score": sr.relevance_score
            }
            for sr in context.search_results
        ],
        "key_insights": context.key_insights,
        "source_credibility": context.source_credibility,
        "generated_documents": context.generated_documents
    }


def _deserialize_research_context(data: Dict[str, Any]) -> ResearchContext:
    """Convert dict back to ResearchContext"""
    context = ResearchContext(original_query=data.get("original_query", ""))
    context.refined_queries = data.get("refined_queries", [])
    context.search_results = [
        SearchResult(
            title=sr["title"],
            link=sr["link"],
            snippet=sr["snippet"],
            relevance_score=sr.get("relevance_score", 0.0)
        )
        for sr in data.get("search_results", [])
    ]
    context.key_insights = data.get("key_insights", [])
    context.source_credibility = data.get("source_credibility", {})
    context.generated_documents = data.get("generated_documents", [])
    return context


def _serialize_llm_message(message: LLMMessage) -> Dict[str, Any]:
    """Convert LLMMessage to JSON-serializable dict"""
    return {
        "role": message.role,
        "content": message.content,
        "metadata": message.metadata
    }


def _deserialize_llm_message(data: Dict[str, Any]) -> LLMMessage:
    """Convert dict back to LLMMessage"""
    return LLMMessage(
        role=data["role"],
        content=data["content"],
        metadata=data.get("metadata", {})
    )


def save_session(session_id: str, session_data: Dict[str, Any]) -> bool:
    """
    Save a session to disk
    
    Args:
        session_id: Unique session identifier
        session_data: Dict containing 'context' (ResearchContext) and 'messages' (List[LLMMessage])
    
    Returns:
        bool: True if save succeeded, False otherwise
    """
    try:
        # Create serializable version of session data
        serialized = {
            "session_id": session_id,
            "saved_at": datetime.now().isoformat(),
            "context": _serialize_research_context(session_data["context"]),
            "messages": [_serialize_llm_message(msg) for msg in session_data.get("messages", [])],
            "current_round": session_data.get("current_round", 0),
            "status": session_data.get("status", "in_progress")
        }
        
        # Save to file
        session_file = SESSIONS_DIR / f"{session_id}.json"
        with open(session_file, 'w', encoding='utf-8') as f:
            json.dump(serialized, f, indent=2, ensure_ascii=False)
        
        return True
    except Exception as e:
        print(f"Error saving session {session_id}: {e}")
        return False


def load_session(session_id: str) -> Optional[Dict[str, Any]]:
    """
    Load a session from disk
    
    Args:
        session_id: Unique session identifier
    
    Returns:
        Dict with 'context' and 'messages', or None if load failed
    """
    try:
        session_file = SESSIONS_DIR / f"{session_id}.json"
        
        if not session_file.exists():
            return None
        
        with open(session_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Deserialize back to objects
        return {
            "session_id": data["session_id"],
            "saved_at": data.get("saved_at"),
            "context": _deserialize_research_context(data["context"]),
            "messages": [_deserialize_llm_message(msg) for msg in data.get("messages", [])],
            "current_round": data.get("current_round", 0),
            "status": data.get("status", "in_progress")
        }
    except Exception as e:
        print(f"Error loading session {session_id}: {e}")
        return None


def list_saved_sessions() -> List[Dict[str, str]]:
    """
    List all saved sessions with metadata
    
    Returns:
        List of dicts with session_id, saved_at, and query
    """
    sessions = []
    
    try:
        for session_file in SESSIONS_DIR.glob("*.json"):
            try:
                with open(session_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                sessions.append({
                    "session_id": data.get("session_id", session_file.stem),
                    "saved_at": data.get("saved_at", "Unknown"),
                    "query": data.get("context", {}).get("original_query", "No query"),
                    "status": data.get("status", "unknown")
                })
            except Exception as e:
                print(f"Error reading session file {session_file}: {e}")
                continue
        
        # Sort by saved_at descending (most recent first)
        sessions.sort(key=lambda x: x["saved_at"], reverse=True)
    except Exception as e:
        print(f"Error listing sessions: {e}")
    
    return sessions


def delete_session(session_id: str) -> bool:
    """
    Delete a saved session
    
    Args:
        session_id: Unique session identifier
    
    Returns:
        bool: True if deletion succeeded, False otherwise
    """
    try:
        session_file = SESSIONS_DIR / f"{session_id}.json"
        if session_file.exists():
            session_file.unlink()
            return True
        return False
    except Exception as e:
        print(f"Error deleting session {session_id}: {e}")
        return False


def load_all_sessions() -> Dict[str, Dict[str, Any]]:
    """
    Load all saved sessions into memory
    
    Returns:
        Dict mapping session_id to session data
    """
    all_sessions = {}
    
    for session_file in SESSIONS_DIR.glob("*.json"):
        session_id = session_file.stem
        session_data = load_session(session_id)
        if session_data:
            all_sessions[session_id] = session_data
    
    return all_sessions
