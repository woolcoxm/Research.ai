"""
Test script to verify session restore functionality
"""
import sys
sys.path.insert(0, '.')

from utils.session_persistence import load_session

session_id = '688ffd30-ace0-4a9e-9ceb-b40d3132af6c'

print(f"Testing load_session for {session_id}...")
session_data = load_session(session_id)

if session_data:
    print(f"✓ Session loaded successfully")
    print(f"  - Context: {session_data.get('context') is not None}")
    print(f"  - Messages: {len(session_data.get('messages', []))}")
    
    context = session_data.get('context')
    if context:
        print(f"  - Context type: {type(context).__name__}")
        print(f"  - Context.messages: {len(context.messages) if hasattr(context, 'messages') else 'No messages attr'}")
        print(f"  - Context.user_prompt: {context.user_prompt[:50] if hasattr(context, 'user_prompt') else 'No prompt'}")
        print(f"  - Context.current_stage: {context.current_stage if hasattr(context, 'current_stage') else 'No stage'}")
    
    messages = session_data.get('messages', [])
    if messages:
        print(f"\nFirst message:")
        msg = messages[0]
        print(f"  - Type: {type(msg).__name__}")
        print(f"  - llm_type: {msg.llm_type if hasattr(msg, 'llm_type') else 'No llm_type'}")
        print(f"  - content length: {len(msg.content) if hasattr(msg, 'content') else 0}")
        print(f"  - id: {msg.id if hasattr(msg, 'id') else 'No id'}")
else:
    print("✗ Failed to load session")
