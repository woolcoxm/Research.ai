"""
Test script to verify session resume functionality
"""
import json
from pathlib import Path

def test_session_structure():
    """Test that saved sessions have the correct structure"""
    sessions_dir = Path("saved_sessions")
    
    if not sessions_dir.exists():
        print("❌ No saved_sessions directory found")
        return False
    
    session_files = list(sessions_dir.glob("*.json"))
    if not session_files:
        print("❌ No session files found")
        return False
    
    print(f"✓ Found {len(session_files)} session file(s)")
    
    for session_file in session_files:
        print(f"\n📁 Testing {session_file.name}:")
        
        try:
            with open(session_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Check required fields
            required_fields = ['session_id', 'context', 'messages']
            missing = [f for f in required_fields if f not in data]
            
            if missing:
                print(f"  ❌ Missing fields: {missing}")
                return False
            
            print(f"  ✓ Has required fields: {required_fields}")
            
            # Check context structure
            context = data['context']
            context_fields = ['session_id', 'user_prompt', 'current_stage', 'conversation_round']
            missing_context = [f for f in context_fields if f not in context]
            
            if missing_context:
                print(f"  ❌ Context missing fields: {missing_context}")
                return False
            
            print(f"  ✓ Context has required fields")
            print(f"    - Stage: {context['current_stage']}")
            print(f"    - Round: {context['conversation_round']}")
            print(f"    - Prompt: {context['user_prompt'][:50]}...")
            
            # Check messages
            messages = data['messages']
            print(f"  ✓ Has {len(messages)} messages")
            
            if messages:
                first_msg = messages[0]
                msg_fields = ['id', 'content', 'llm_type', 'timestamp']
                missing_msg = [f for f in msg_fields if f not in first_msg]
                
                if missing_msg:
                    print(f"  ❌ Message missing fields: {missing_msg}")
                    return False
                
                print(f"  ✓ Messages have required fields")
                
                # Check for duplicates
                msg_ids = [m['id'] for m in messages]
                if len(msg_ids) != len(set(msg_ids)):
                    print(f"  ⚠️  WARNING: Duplicate message IDs found")
                else:
                    print(f"  ✓ No duplicate message IDs")
            
            # Check search results
            if 'search_results' in context:
                results = context['search_results']
                print(f"  ✓ Has {len(results)} search results")
                
                if results:
                    first_result = results[0]
                    result_fields = ['title', 'link', 'snippet', 'source']
                    missing_result = [f for f in result_fields if f not in first_result]
                    
                    if missing_result:
                        print(f"  ⚠️  Search result missing fields: {missing_result}")
                    else:
                        print(f"  ✓ Search results have all fields")
            
            # Check if session can be resumed
            status = data.get('status', 'unknown')
            can_resume = (status == 'in_progress' and 
                         context['current_stage'] != 'completed')
            
            print(f"  Status: {status}")
            print(f"  {'✓' if can_resume else '✗'} Can be resumed: {can_resume}")
            
        except json.JSONDecodeError as e:
            print(f"  ❌ Invalid JSON: {e}")
            return False
        except Exception as e:
            print(f"  ❌ Error reading session: {e}")
            return False
    
    print("\n" + "="*60)
    print("✅ All session files are valid and properly structured!")
    return True

if __name__ == "__main__":
    success = test_session_structure()
    exit(0 if success else 1)
