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
    # Serialize search results from both initial and targeted searches
    all_search_results = []
    for sr in (context.initial_searches + context.targeted_searches):
        all_search_results.append({
            "title": sr.title,
            "link": sr.link,
            "snippet": sr.snippet,
            "source": sr.source if hasattr(sr, 'source') else "unknown",
            "relevance_score": sr.relevance_score
        })
    
    return {
        "session_id": context.session_id,
        "user_prompt": context.user_prompt,
        "created_at": context.created_at.isoformat() if hasattr(context.created_at, 'isoformat') else str(context.created_at),
        "updated_at": context.updated_at.isoformat() if hasattr(context.updated_at, 'isoformat') else str(context.updated_at),
        "search_results": all_search_results,
        "key_insights": context.key_insights,
        "technology_references": context.technology_references if hasattr(context, 'technology_references') else {},
        "citation_map": context.citation_map if hasattr(context, 'citation_map') else {},
        "current_stage": context.current_stage.value if hasattr(context.current_stage, 'value') else str(context.current_stage),
        "conversation_round": context.conversation_round,
        "context_maturity": context.context_maturity,
        "quality_gates_passed": context.quality_gates_passed,
        "metadata": context.metadata if hasattr(context, 'metadata') else {}
    }


def _deserialize_research_context(data: Dict[str, Any]) -> ResearchContext:
    """Convert dict back to ResearchContext"""
    from core.models import ConversationStage
    from datetime import datetime
    
    context = ResearchContext()
    context.session_id = data.get("session_id", "")
    context.user_prompt = data.get("user_prompt", "")
    
    # Restore timestamps
    created_at_str = data.get("created_at")
    if created_at_str:
        try:
            context.created_at = datetime.fromisoformat(created_at_str)
        except:
            pass
    
    updated_at_str = data.get("updated_at")
    if updated_at_str:
        try:
            context.updated_at = datetime.fromisoformat(updated_at_str)
        except:
            pass
    
    # Restore search results (store in initial_searches for simplicity)
    search_results = [
        SearchResult(
            title=sr["title"],
            link=sr["link"],
            snippet=sr["snippet"],
            source=sr.get("source", "unknown"),
            relevance_score=sr.get("relevance_score", 0.0)
        )
        for sr in data.get("search_results", [])
    ]
    context.initial_searches = search_results
    context.targeted_searches = []
    
    context.key_insights = data.get("key_insights", [])
    context.technology_references = data.get("technology_references", {})
    context.citation_map = data.get("citation_map", {})
    context.metadata = data.get("metadata", {})
    
    # NOTE: Messages are NOT restored here - they are loaded separately
    # and passed alongside the context in load_session() return value
    
    # Restore stage and progress
    stage_value = data.get("current_stage", "initial_breakdown")
    try:
        context.current_stage = ConversationStage(stage_value)
    except:
        context.current_stage = ConversationStage.INITIAL_BREAKDOWN
    
    context.conversation_round = data.get("conversation_round", 0)
    context.context_maturity = data.get("context_maturity", 0.0)
    context.quality_gates_passed = data.get("quality_gates_passed", [])
    
    return context


def _serialize_llm_message(message: LLMMessage) -> Dict[str, Any]:
    """Convert LLMMessage to JSON-serializable dict with content validation"""
    content = message.content
    
    # Validate content is not truncated/corrupted
    if content and len(content) > 100:
        # Check for common signs of truncation
        if not content.rstrip().endswith(('.', '!', '?', '\n', '`', '"', "'", ')', ']', '}', ':')):
            print(f"[SAVE] WARNING: Message {message.id} content may be truncated (ends with: '{content[-50:]}')")
            # Truncate to last complete sentence
            for end_char in ['.', '!', '?', '\n', ')', ']', '}']:
                last_pos = content.rfind(end_char)
                if last_pos > len(content) * 0.9:  # Within last 10%
                    content = content[:last_pos + 1]
                    print(f"[SAVE] Truncated message to last complete sentence")
                    break
    
    return {
        "id": message.id,
        "content": content,
        "llm_type": message.llm_type.value if hasattr(message, 'llm_type') and message.llm_type and hasattr(message.llm_type, 'value') else 'system',
        "timestamp": message.timestamp.isoformat() if hasattr(message, 'timestamp') else datetime.now().isoformat(),
        "context_references": message.context_references if hasattr(message, 'context_references') else [],
        "confidence_score": message.confidence_score if hasattr(message, 'confidence_score') else 0.0,
        "metadata": message.metadata if message.metadata else {}
    }


def _deserialize_llm_message(data: Dict[str, Any]) -> LLMMessage:
    """Convert dict back to LLMMessage"""
    from core.models import LLMType
    from datetime import datetime
    
    # Reconstruct llm_type
    llm_type_value = data.get("llm_type", "system")
    try:
        llm_type = LLMType(llm_type_value)
    except:
        llm_type = LLMType.SYSTEM
    
    # Reconstruct timestamp
    timestamp_str = data.get("timestamp")
    if timestamp_str:
        try:
            timestamp = datetime.fromisoformat(timestamp_str)
        except:
            timestamp = datetime.now()
    else:
        timestamp = datetime.now()
    
    msg = LLMMessage()
    msg.id = data.get("id", msg.id)
    msg.content = data.get("content", "")
    msg.llm_type = llm_type
    msg.timestamp = timestamp
    msg.context_references = data.get("context_references", [])
    msg.confidence_score = data.get("confidence_score", 0.0)
    msg.metadata = data.get("metadata", {})
    
    return msg


def save_session(session_id: str, session_data: Dict[str, Any]) -> bool:
    """
    Save a session to disk with atomic write and validation
    
    Args:
        session_id: Unique session identifier
        session_data: Dict containing 'context' (ResearchContext) and 'messages' (List[LLMMessage])
    
    Returns:
        bool: True if save succeeded, False otherwise
    """
    try:
        print(f"[SAVE] Attempting to save session {session_id}")
        
        # Check if context exists
        if 'context' not in session_data:
            print(f"[SAVE] ERROR: No 'context' in session_data. Keys: {list(session_data.keys())}")
            return False
        
        # Create serializable version of session data
        print(f"[SAVE] Serializing context...")
        serialized_context = _serialize_research_context(session_data["context"])
        
        print(f"[SAVE] Serializing {len(session_data.get('messages', []))} messages...")
        messages = session_data.get("messages", [])
        
        # Deduplicate messages by ID
        seen_ids = set()
        unique_messages = []
        for msg in messages:
            msg_id = getattr(msg, 'id', None)
            if msg_id and msg_id not in seen_ids:
                seen_ids.add(msg_id)
                unique_messages.append(msg)
        
        if len(unique_messages) < len(messages):
            print(f"[SAVE] WARNING: Deduplicated {len(messages) - len(unique_messages)} duplicate messages")
        
        serialized_messages = [_serialize_llm_message(msg) for msg in unique_messages]
        
        serialized = {
            "session_id": session_id,
            "saved_at": datetime.now().isoformat(),
            "context": serialized_context,
            "messages": serialized_messages,
            "current_round": session_data.get("current_round", 0),
            "status": session_data.get("status", "in_progress")
        }
        
        # Validate JSON serializability before writing
        try:
            json_string = json.dumps(serialized, indent=2, ensure_ascii=False)
            print(f"[SAVE] JSON validated, size: {len(json_string)} bytes")
        except (TypeError, ValueError) as e:
            print(f"[SAVE] ERROR: Data is not JSON serializable: {e}")
            return False
        
        # Atomic write: write to temp file then rename
        session_file = SESSIONS_DIR / f"{session_id}.json"
        temp_file = SESSIONS_DIR / f"{session_id}.json.tmp"
        
        print(f"[SAVE] Writing to temp file: {temp_file}")
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write(json_string)
            f.flush()
            os.fsync(f.fileno())  # Force write to disk
        
        # Validate the written file
        print(f"[SAVE] Validating written file...")
        with open(temp_file, 'r', encoding='utf-8') as f:
            json.load(f)  # This will raise if JSON is invalid
        
        # Atomic rename
        print(f"[SAVE] Atomically renaming to: {session_file}")
        if session_file.exists():
            backup_file = SESSIONS_DIR / f"{session_id}.json.bak"
            if backup_file.exists():
                backup_file.unlink()
            session_file.rename(backup_file)
        temp_file.rename(session_file)
        
        print(f"[SAVE] SUCCESS: Session {session_id} saved to {session_file}")
        return True
    except Exception as e:
        import traceback
        print(f"[SAVE] ERROR saving session {session_id}: {e}")
        print(f"[SAVE] Traceback: {traceback.format_exc()}")
        # Clean up temp file if it exists
        try:
            temp_file = SESSIONS_DIR / f"{session_id}.json.tmp"
            if temp_file.exists():
                temp_file.unlink()
        except:
            pass
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
                    "query": data.get("context", {}).get("user_prompt", "No query"),
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
