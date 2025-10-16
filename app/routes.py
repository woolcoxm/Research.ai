import logging
import json
import uuid
import os
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, session, send_file
from config.settings import Config
from core.conversation_orchestrator import ConversationOrchestrator
from core.models import ResearchContext, ConversationStage
from utils.file_manager import FileManager
from utils.session_persistence import save_session, load_session, list_saved_sessions, delete_session, load_all_sessions

# Configure logging
logging.basicConfig(level=logging.DEBUG)  # Changed to DEBUG for troubleshooting
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = Config.FLASK_SECRET_KEY

# Global session store for real-time updates (in production, use Redis or similar)
global_sessions = {}

# Status tracking for active sessions
active_status = {}

# Load saved sessions on startup
logger.info("Loading saved sessions from disk...")
try:
    loaded_sessions = load_all_sessions()
    # Convert to JSON-safe format for global_sessions
    for sid, sdata in loaded_sessions.items():
        context = sdata.get('context')
        messages = sdata.get('messages', [])
        
        safe_messages = [
            {
                'llm_type': msg.llm_type.value if hasattr(msg, 'llm_type') else 'system',
                'content': msg.content[:200] + '...' if len(msg.content) > 200 else msg.content,
                'timestamp': msg.timestamp.isoformat() if hasattr(msg, 'timestamp') else datetime.now().isoformat()
            }
            for msg in messages[-6:]
        ]
        
        all_messages_safe = [
            {
                'llm_type': msg.llm_type.value if hasattr(msg, 'llm_type') else 'system',
                'content': msg.content,
                'timestamp': msg.timestamp.isoformat() if hasattr(msg, 'timestamp') else datetime.now().isoformat(),
                'id': msg.id if hasattr(msg, 'id') else str(uuid.uuid4())
            }
            for msg in messages
        ]
        
        global_sessions[sid] = {
            'session_id': sid,
            'current_stage': context.current_stage.value if context and hasattr(context, 'current_stage') else 'initial_breakdown',
            'conversation_round': sdata.get('current_round', 0),
            'context_maturity': context.context_maturity if context and hasattr(context, 'context_maturity') else 0.0,
            'quality_gates_passed': context.quality_gates_passed if context and hasattr(context, 'quality_gates_passed') else [],
            'message_count': len(messages),
            'search_count': sdata.get('search_count', 0),
            'latest_messages': safe_messages,
            'all_messages': all_messages_safe,
            'completed': sdata.get('completed', False),
            'failed': sdata.get('failed', False),
            'error': sdata.get('error'),
            'saved_files': None,
            'last_updated': sdata.get('saved_at', datetime.now().isoformat()),
            'user_prompt': context.user_prompt[:100] + '...' if context and hasattr(context, 'user_prompt') and len(context.user_prompt) > 100 else (context.user_prompt if context and hasattr(context, 'user_prompt') else ''),
            'current_round': sdata.get('current_round', 0),
            'status': sdata.get('status', 'in_progress')
        }
    
    logger.info(f"Loaded {len(loaded_sessions)} saved sessions")
except Exception as e:
    logger.error(f"Error loading saved sessions: {e}")

# Session cleanup configuration
MAX_STORED_SESSIONS = 50  # Limit number of stored sessions
SESSION_TIMEOUT_HOURS = 24  # Auto-cleanup sessions older than 24 hours

def update_status(session_id, llm_name, activity, details="", research_context=None):
    """Update the current status for a session"""
    active_status[session_id] = {
        'llm_name': llm_name,
        'activity': activity,
        'details': details,
        'timestamp': datetime.now().isoformat(),
        'is_active': True
    }
    logger.info(f"Status Update [{session_id}]: {llm_name} - {activity} {details[:50] if details else ''}")
    
    # If research_context is provided, update the global session with current stage
    if research_context and session_id in global_sessions:
        global_sessions[session_id]['current_stage'] = research_context.current_stage.value
        global_sessions[session_id]['conversation_round'] = research_context.conversation_round
        global_sessions[session_id]['last_updated'] = datetime.now().isoformat()

def clear_status(session_id):
    """Clear status when session completes"""
    if session_id in active_status:
        active_status[session_id]['is_active'] = False
        active_status[session_id]['activity'] = 'Completed'

def cleanup_old_sessions():
    """Remove old sessions to prevent memory leaks"""
    try:
        current_time = datetime.now()
        cutoff_time = current_time - timedelta(hours=SESSION_TIMEOUT_HOURS)
        
        # Find sessions to remove
        sessions_to_remove = []
        for session_id, session_data in global_sessions.items():
            last_updated = datetime.fromisoformat(session_data.get('last_updated', current_time.isoformat()))
            if last_updated < cutoff_time:
                sessions_to_remove.append(session_id)
        
        # Remove old sessions
        for session_id in sessions_to_remove:
            del global_sessions[session_id]
            if session_id in active_status:
                del active_status[session_id]
            logger.info(f"Cleaned up old session: {session_id}")
        
        # Also enforce max session limit
        if len(global_sessions) > MAX_STORED_SESSIONS:
            # Sort by last_updated and keep only the most recent
            sorted_sessions = sorted(
                global_sessions.items(),
                key=lambda x: x[1].get('last_updated', ''),
                reverse=True
            )
            
            # Remove oldest sessions beyond limit
            for session_id, _ in sorted_sessions[MAX_STORED_SESSIONS:]:
                del global_sessions[session_id]
                if session_id in active_status:
                    del active_status[session_id]
                logger.info(f"Cleaned up excess session: {session_id}")
        
        if sessions_to_remove or len(global_sessions) > MAX_STORED_SESSIONS:
            logger.info(f"Session cleanup: {len(sessions_to_remove)} old, now storing {len(global_sessions)} sessions")
    except Exception as e:
        logger.error(f"Error during session cleanup: {e}")

def _update_global_session(session_id, research_context, completed=False, failed=False, error=None, saved_files=None):
    """Update global session store for real-time tracking - OPTIMIZED"""
    # Get latest messages (last 6 for display) - using list comprehension
    latest_messages = [
        {
            'llm_type': msg.llm_type.value,
            'content': msg.content[:200] + '...' if len(msg.content) > 200 else msg.content,
            'timestamp': msg.timestamp.isoformat()
        }
        for msg in research_context.messages[-6:]
    ]
    
    # Store ALL messages - optimized with list comprehension
    all_messages = [
        {
            'llm_type': msg.llm_type.value,
            'content': msg.content,
            'timestamp': msg.timestamp.isoformat(),
            'id': msg.id
        }
        for msg in research_context.messages
    ]
    
    logger.info(f"Updating global session {session_id}: {len(all_messages)} messages stored")
    
    # Build research metrics - pre-calculate once
    total_searches = len(research_context.initial_searches) + len(research_context.targeted_searches)
    research_metrics = {
        'total_searches': total_searches,
        'key_insights': len(research_context.key_insights),
        'conversation_rounds': research_context.conversation_round,
        'context_maturity': research_context.context_maturity,
        'quality_gates_passed': research_context.quality_gates_passed
    }
    
    global_sessions[session_id] = {
        'session_id': research_context.session_id,
        'current_stage': research_context.current_stage.value,
        'conversation_round': research_context.conversation_round,
        'context_maturity': research_context.context_maturity,
        'quality_gates_passed': research_context.quality_gates_passed,
        'message_count': len(research_context.messages),
        'search_count': len(research_context.initial_searches) + len(research_context.targeted_searches),
        'research_metrics': research_metrics,  # Add structured metrics for completion banner
        'latest_messages': latest_messages,
        'all_messages': all_messages,  # Store all messages for history view
        'completed': completed,
        'failed': failed,
        'error': error,
        'saved_files': saved_files,  # Add saved files for completion banner
        'last_updated': datetime.now().isoformat(),
        'user_prompt': research_context.user_prompt[:100] + '...' if len(research_context.user_prompt) > 100 else research_context.user_prompt,
        'current_round': research_context.conversation_round,
        'status': 'completed' if completed else ('failed' if failed else 'in_progress')
    }
    
    # Auto-save session to disk after each update (with full context and messages)
    try:
        persistence_data = {
            'session_id': research_context.session_id,
            'current_stage': research_context.current_stage.value,
            'conversation_round': research_context.conversation_round,
            'context_maturity': research_context.context_maturity,
            'quality_gates_passed': research_context.quality_gates_passed,
            'message_count': len(research_context.messages),
            'search_count': len(research_context.initial_searches) + len(research_context.targeted_searches),
            'completed': completed,
            'failed': failed,
            'error': error,
            'last_updated': datetime.now().isoformat(),
            'user_prompt': research_context.user_prompt,
            'current_round': research_context.conversation_round,
            'status': 'completed' if completed else ('failed' if failed else 'in_progress'),
            'context': research_context,  # Full context for persistence
            'messages': research_context.messages  # Full messages for persistence
        }
        save_session(session_id, persistence_data)
        logger.info(f"Session {session_id} auto-saved to disk")
    except Exception as e:
        logger.error(f"Failed to auto-save session {session_id}: {e}")

# Initialize components
try:
    orchestrator = ConversationOrchestrator()
    file_manager = FileManager()
    Config.validate_config()
    logger.info("AI Research System initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize system: {e}")
    orchestrator = None
    file_manager = None


@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')


@app.route('/status/<session_id>')
def get_status(session_id):
    """Get current status for a session"""
    status_data = active_status.get(session_id, {
        'llm_name': 'System',
        'activity': 'Idle',
        'details': '',
        'timestamp': datetime.now().isoformat(),
        'is_active': False
    })
    
    session_data = global_sessions.get(session_id, {})
    
    return jsonify({
        'status': status_data,
        'session': session_data
    })


@app.route('/research', methods=['POST'])
def start_research():
    """Start a new research session and run full automation"""
    if orchestrator is None or file_manager is None:
        return jsonify({'error': 'System not properly initialized'}), 500
    
    try:
        # Cleanup old sessions periodically
        if len(global_sessions) > 0 and len(global_sessions) % 10 == 0:
            cleanup_old_sessions()
        
        data = request.get_json()
        user_prompt = data.get('prompt', '').strip()
        
        if not user_prompt:
            return jsonify({'error': 'Prompt is required'}), 400
        
        # Generate session ID
        session_id = str(uuid.uuid4())
        session['current_session'] = session_id
        
        # Start research session
        research_context = orchestrator.start_research_session(user_prompt)
        # Store minimal session data to avoid cookie size limits
        session['research_context'] = {
            'session_id': research_context.session_id,
            'current_stage': research_context.current_stage.value,
            'conversation_round': research_context.conversation_round,
            'context_maturity': research_context.context_maturity
        }
        
        # Initialize global_sessions entry so status endpoint can return stage immediately
        global_sessions[session_id] = {
            'session_id': research_context.session_id,
            'current_stage': research_context.current_stage.value,
            'conversation_round': research_context.conversation_round,
            'context_maturity': research_context.context_maturity,
            'quality_gates_passed': research_context.quality_gates_passed,
            'message_count': 0,
            'search_count': 0,
            'research_metrics': {},
            'latest_messages': [],
            'all_messages': [],
            'completed': False,
            'failed': False,
            'error': None,
            'saved_files': {},
            'last_updated': datetime.now().isoformat(),
            'user_prompt': user_prompt
        }
        
        logger.info(f"Research session started: {session_id} for prompt: {user_prompt}")
        
        # Run the full automation in a background thread
        import threading
        def run_full_workflow(research_context, session_id):
            try:
                # Set up status callback for the orchestrator
                def status_callback(llm_name, activity, details="", ctx=None):
                    update_status(session_id, llm_name, activity, details, research_context=ctx)
                
                orchestrator.set_status_callback(status_callback)
                
                update_status(session_id, "System", "Starting automated research workflow", research_context=research_context)
                logger.info(f"Starting full automation workflow for session: {session_id}")
                
                # Execute all conversation rounds automatically until completion
                max_rounds = 50  # Increased for new 11-stage iterative workflow
                round_num = 0
                while round_num < max_rounds and research_context.current_stage != ConversationStage.COMPLETED:
                    round_num += 1
                    update_status(session_id, "System", f"Executing round {round_num}", research_context=research_context)
                    logger.info(f"Executing round {round_num}, Stage: {research_context.current_stage.value}")
                    research_context = orchestrator.execute_conversation_round(research_context)
                    
                    # Update the global session store for real-time updates
                    _update_global_session(session_id, research_context)
                    
                    # Check if workflow is complete
                    if research_context.current_stage == ConversationStage.COMPLETED:
                        logger.info("Workflow completed - all 11 stages finished")
                        break
                
                # Documents are generated in Stage 9, check if they were saved
                update_status(session_id, "System", "Finalizing documents", research_context=research_context)
                logger.info("Checking for generated documents")
                
                # Get approved documents from metadata
                approved_docs = research_context.metadata.get('approved_documents', [])
                saved_files = {}
                
                if approved_docs:
                    # Save approved documents to files
                    update_status(session_id, "System", f"Saving {len(approved_docs)} documents to files", research_context=research_context)
                    logger.info(f"Saving {len(approved_docs)} approved documents")
                    
                    # Create project directory
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    project_name = file_manager._sanitize_filename(research_context.user_prompt[:50])
                    project_dir = os.path.join(file_manager.devplan_dir, f"{timestamp}_{project_name}")
                    os.makedirs(project_dir, exist_ok=True)
                    
                    for idx, doc in enumerate(approved_docs):
                        title = doc.get('title', f'Document_{idx+1}')
                        content = doc.get('content', '')
                        
                        # Save each document as markdown
                        filename = file_manager._sanitize_filename(title) + '.md'
                        filepath = os.path.join(project_dir, filename)
                        
                        with open(filepath, 'w', encoding='utf-8') as f:
                            f.write(f"# {title}\n\n{content}")
                        
                        saved_files[title] = filepath
                        logger.info(f"Saved document: {title} to {filepath}")
                    
                    update_status(session_id, "System", f"Completed! Generated {len(saved_files)} documents", research_context=research_context)
                else:
                    logger.warning("No approved documents found in metadata")
                    update_status(session_id, "System", "Workflow complete - no documents generated", research_context=research_context)
                
                # Mark automation as completed
                clear_status(session_id)
                _update_global_session(session_id, research_context, completed=True, saved_files=saved_files)
                
            except Exception as e:
                logger.error(f"Automation workflow failed: {e}")
                # Mark session as failed
                _update_global_session(session_id, research_context, failed=True, error=str(e))
        
        # Start the automation in background
        thread = threading.Thread(target=run_full_workflow, args=(research_context, session_id))
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'session_id': session_id,
            'status': 'automation_started',
            'research_context': research_context.to_dict(),
            'message': 'Full automation started. The system will now research, discuss, and generate a development plan automatically.'
        })
        
    except Exception as e:
        logger.error(f"Failed to start research: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/session/<session_id>')
def get_session(session_id):
    """Get session data including results"""
    session_data = global_sessions.get(session_id, {})
    return jsonify(session_data)


@app.route('/conversation/next', methods=['POST'])
def next_conversation_round():
    """Execute next conversation round"""
    if orchestrator is None:
        return jsonify({'error': 'System not properly initialized'}), 500
    
    try:
        # Get current research context from session (minimal data)
        research_context_dict = session.get('research_context')
        if not research_context_dict:
            return jsonify({'error': 'No active research session'}), 400
        
        # For manual conversation rounds, we need to reconstruct from full context
        # This is a limitation - manual mode requires smaller sessions
        return jsonify({'error': 'Manual conversation rounds not supported with current session optimization'}), 400
        
        # Execute next conversation round
        research_context = orchestrator.execute_conversation_round(research_context)
        
        # Update session
        session['research_context'] = research_context.to_dict()
        
        # Prepare response data
        response_data = {
            'session_id': research_context.session_id,
            'current_stage': research_context.current_stage.value,
            'conversation_round': research_context.conversation_round,
            'context_maturity': research_context.context_maturity,
            'quality_gates_passed': research_context.quality_gates_passed,
            'research_context': research_context.to_dict(),
            'latest_messages': []
        }
        
        # Add latest messages
        recent_messages = research_context.messages[-2:]  # Last 2 messages
        for msg in recent_messages:
            response_data['latest_messages'].append({
                'llm_type': msg.llm_type.value,
                'content': msg.content,
                'timestamp': msg.timestamp.isoformat()
            })
        
        logger.info(f"Conversation round {research_context.conversation_round} completed")
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Failed to execute conversation round: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/plan/generate', methods=['POST'])
def generate_plan():
    """Generate final development plan"""
    if orchestrator is None or file_manager is None:
        return jsonify({'error': 'System not properly initialized'}), 500
    
    try:
        # Get current research context from session (minimal data)
        research_context_dict = session.get('research_context')
        if not research_context_dict:
            return jsonify({'error': 'No active research session'}), 400
        
        # For plan generation, we need the full context which isn't stored in session
        return jsonify({'error': 'Manual plan generation not supported with current session optimization'}), 400
        
        # Generate development plan
        development_plan = orchestrator.generate_development_plan(research_context)
        
        # Save plan to file(s) - handle single or multiple documents
        if development_plan.get('multi_document', False):
            saved_files = file_manager.save_multiple_documents(development_plan)
            logger.info(f"Multiple documents saved: {len(saved_files)} files")
            development_plan['saved_files'] = saved_files
        else:
            filepath = file_manager.save_development_plan(development_plan)
            logger.info(f"Single document saved: {filepath}")
            development_plan['saved_files'] = {'Development Plan': filepath}
        
        # Update session
        session['research_context'] = research_context.to_dict()
        session['last_plan'] = development_plan
        
        logger.info(f"Development plan generated and saved: {filepath}")
        
        return jsonify({
            'session_id': research_context.session_id,
            'status': 'plan_generated',
            'development_plan': development_plan,
            'filepath': filepath,
            'message': 'Development plan generated and saved successfully'
        })
        
    except Exception as e:
        logger.error(f"Failed to generate development plan: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/plans', methods=['GET'])
def list_plans():
    """List all development plans (both single and multi-document projects)"""
    if file_manager is None:
        return jsonify({'error': 'System not properly initialized'}), 500
    
    try:
        # Get both legacy single files and new multi-document projects
        legacy_plans = file_manager.list_development_plans()
        multi_doc_projects = file_manager.list_project_documents()
        
        # Combine and sort by generation date
        all_projects = []
        
        # Add legacy plans
        for plan in legacy_plans:
            all_projects.append({
                'type': 'legacy',
                'project_name': plan['project_name'],
                'user_prompt': plan['user_prompt'],
                'generated_at': plan['generated_at'],
                'filename': plan['filename'],
                'file_size': plan['file_size'],
                'feasibility_score': plan['feasibility_score'],
                'documents': [{
                    'title': 'Development Plan',
                    'filename': plan['filename'],
                    'download_url': f"/download/DEVPLAN/{plan['filename']}"
                }]
            })
        
        # Add multi-document projects
        all_projects.extend(multi_doc_projects)
        
        # Sort by generation date (newest first)
        all_projects.sort(key=lambda x: x.get('generated_at', ''), reverse=True)
        
        statistics = file_manager.get_plan_statistics()
        
        return jsonify({
            'projects': all_projects,
            'statistics': statistics,
            'total_projects': len(all_projects),
            'multi_document_projects': len([p for p in all_projects if p.get('type') == 'multi_document']),
            'legacy_projects': len([p for p in all_projects if p.get('type') == 'legacy'])
        })
        
    except Exception as e:
        logger.error(f"Failed to list development plans: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/plans/<filename>', methods=['GET'])
def get_plan(filename):
    """Get a specific development plan"""
    if file_manager is None:
        return jsonify({'error': 'System not properly initialized'}), 500
    
    try:
        plan_data = file_manager.load_development_plan(filename)
        
        if not plan_data:
            return jsonify({'error': 'Plan not found'}), 404
        
        return jsonify(plan_data)
        
    except Exception as e:
        logger.error(f"Failed to get development plan {filename}: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/plans/<filename>/export', methods=['GET'])
def export_plan(filename):
    """Export a development plan to markdown"""
    if file_manager is None:
        return jsonify({'error': 'System not properly initialized'}), 500
    
    try:
        plan_data = file_manager.load_development_plan(filename)
        
        if not plan_data:
            return jsonify({'error': 'Plan not found'}), 404
        
        # Export to markdown
        markdown_path = file_manager.export_plan_to_markdown(plan_data)
        
        return send_file(
            markdown_path,
            as_attachment=True,
            download_name=f"{plan_data.get('project_name', 'plan')}.md",
            mimetype='text/markdown'
        )
        
    except Exception as e:
        logger.error(f"Failed to export development plan {filename}: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/plans/<filename>', methods=['DELETE'])
def delete_plan(filename):
    """Delete a development plan"""
    if file_manager is None:
        return jsonify({'error': 'System not properly initialized'}), 500
    
    try:
        success = file_manager.delete_development_plan(filename)
        
        if not success:
            return jsonify({'error': 'Failed to delete plan'}), 500
        
        return jsonify({'message': 'Plan deleted successfully'})
        
    except Exception as e:
        logger.error(f"Failed to delete development plan {filename}: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/status', methods=['GET'])
def system_status():
    """Get system status and configuration"""
    try:
        status = {
            'system_initialized': orchestrator is not None and file_manager is not None,
            'configuration': {
                'max_conversation_rounds': Config.MAX_CONVERSATION_ROUNDS,
                'max_search_results': Config.MAX_SEARCH_RESULTS,
                'min_context_maturity': Config.MIN_CONTEXT_MATURITY,
                'devplan_directory': Config.DEVPLAN_DIR
            },
            'components': {
                'deepseek': bool(Config.DEEPSEEK_API_KEY),
                'ollama': True,  # We check connection at startup
                'serper': bool(Config.SERPER_API_KEY),
                'file_manager': file_manager is not None
            }
        }
        
        return jsonify(status)
        
    except Exception as e:
        logger.error(f"Failed to get system status: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/session/<session_id>/status', methods=['GET'])
def session_status(session_id):
    """Get current session status for real-time updates"""
    try:
        # Check global session store first (for active automation sessions)
        if session_id in global_sessions:
            session_data = global_sessions[session_id]
            logger.info(f"Retrieved session status from global store: {session_id}")
            return jsonify(session_data)
        
        # Fallback to Flask session data
        session_data = session.get('research_context')
        if not session_data or session_data.get('session_id') != session_id:
            return jsonify({'error': 'No active research session'}), 404
        
        # Return minimal status data for real-time updates
        response_data = {
            'session_id': session_data.get('session_id'),
            'current_stage': session_data.get('current_stage'),
            'conversation_round': session_data.get('conversation_round'),
            'context_maturity': session_data.get('context_maturity'),
            'quality_gates_passed': session_data.get('quality_gates_passed', []),
            'message_count': session_data.get('message_count', 0),
            'search_count': session_data.get('search_count', 0),
            'completed': False,
            'failed': False,
            'error': None,
            'last_updated': datetime.now().isoformat()
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Failed to get session status: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/conversations/<session_id>')
def get_conversations(session_id):
    """Get conversation messages organized by LLM type"""
    try:
        # Get session data from global sessions
        session_data = global_sessions.get(session_id)
        
        if not session_data:
            logger.warning(f"Session not found: {session_id}")
            return jsonify({'error': 'Session not found'}), 404
        
        deepseek_messages = []
        ollama_messages = []
        
        # Use all_messages for full conversation history
        all_messages = session_data.get('all_messages', [])
        logger.info(f"Getting conversations for session {session_id}: {len(all_messages)} total messages")
        
        # Organize by LLM type with numbering
        deepseek_counter = 1
        ollama_counter = 1
        
        for idx, msg in enumerate(all_messages):
            llm_type = msg.get('llm_type', 'unknown')
            logger.debug(f"Message {idx}: llm_type='{llm_type}', content_preview='{msg.get('content', '')[:50]}'")
            
            msg_data = {
                'content': msg.get('content', ''),
                'timestamp': msg.get('timestamp', ''),
                'msg_id': msg.get('id', ''),
                'preview': msg.get('content', '')[:150] + '...' if len(msg.get('content', '')) > 150 else msg.get('content', '')
            }
            
            if llm_type == 'deepseek':
                msg_data['number'] = deepseek_counter
                deepseek_messages.append(msg_data)
                deepseek_counter += 1
            elif llm_type == 'ollama':
                msg_data['number'] = ollama_counter
                ollama_messages.append(msg_data)
                ollama_counter += 1
            else:
                logger.warning(f"Message {idx} has unknown llm_type: '{llm_type}'")
        
        logger.info(f"Returning {len(deepseek_messages)} DeepSeek and {len(ollama_messages)} Ollama messages")
        
        return jsonify({
            'session_id': session_id,
            'user_prompt': session_data.get('user_prompt', ''),
            'deepseek_messages': deepseek_messages,
            'ollama_messages': ollama_messages,
            'total_messages': len(all_messages),
            'current_stage': session_data.get('current_stage', ''),
            'completed': session_data.get('completed', False)
        })
        
    except Exception as e:
        logger.error(f"Failed to get conversations: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/sessions/list', methods=['GET'])
def list_sessions():
    """Get list of all saved sessions"""
    try:
        sessions = list_saved_sessions()
        return jsonify({'sessions': sessions})
    except Exception as e:
        logger.error(f"Failed to list sessions: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/sessions/restore/<session_id>', methods=['POST'])
def restore_session(session_id):
    """Restore a previously saved session"""
    try:
        session_data = load_session(session_id)
        
        if not session_data:
            return jsonify({'error': 'Session not found'}), 404
        
        # Extract context and messages
        context = session_data.get('context')
        messages = session_data.get('messages', [])
        
        logger.info(f"[RESTORE] Loaded session {session_id}: context={context is not None}, messages={len(messages)}")
        
        # CRITICAL: Restore messages to the context object
        # Messages are loaded separately but need to be in context.messages for display
        if context:
            context.messages = messages
            logger.info(f"[RESTORE] Restored {len(messages)} messages to context.messages for session {session_id}")
        else:
            logger.error(f"[RESTORE] No context found for session {session_id}!")
            
        if not messages:
            logger.warning(f"[RESTORE] No messages found for session {session_id}!")
        else:
            logger.info(f"[RESTORE] Have {len(messages)} messages, first message: type={type(messages[0]).__name__}, llm_type={messages[0].llm_type.value if hasattr(messages[0], 'llm_type') else 'NO ATTR'}")
        
        # Create JSON-safe version for global_sessions (without raw objects)
        safe_messages = [
            {
                'llm_type': msg.llm_type.value if hasattr(msg, 'llm_type') else 'system',
                'content': msg.content[:200] + '...' if len(msg.content) > 200 else msg.content,
                'timestamp': msg.timestamp.isoformat() if hasattr(msg, 'timestamp') else datetime.now().isoformat()
            }
            for msg in messages[-6:]  # Last 6 messages
        ]
        
        all_messages_safe = [
            {
                'llm_type': msg.llm_type.value if hasattr(msg, 'llm_type') else 'system',
                'content': msg.content,
                'timestamp': msg.timestamp.isoformat() if hasattr(msg, 'timestamp') else datetime.now().isoformat(),
                'id': msg.id if hasattr(msg, 'id') else str(uuid.uuid4())
            }
            for msg in messages
        ]
        
        logger.info(f"[RESTORE] Created all_messages_safe with {len(all_messages_safe)} messages")
        
        # Update global sessions with JSON-safe data
        search_count = len(context.initial_searches) + len(context.targeted_searches) if context else 0
        
        global_sessions[session_id] = {
            'session_id': session_id,
            'current_stage': context.current_stage.value if context else 'initial_breakdown',
            'conversation_round': context.conversation_round if context else session_data.get('current_round', 0),
            'context_maturity': context.context_maturity if context else 0.0,
            'quality_gates_passed': context.quality_gates_passed if context else [],
            'message_count': len(messages),
            'search_count': search_count,
            'latest_messages': safe_messages,
            'all_messages': all_messages_safe,
            'completed': session_data.get('status') == 'completed',
            'failed': session_data.get('status') == 'failed',
            'error': session_data.get('error'),
            'saved_files': None,
            'last_updated': datetime.now().isoformat(),
            'user_prompt': context.user_prompt if context else 'Unknown',
            'current_round': context.conversation_round if context else session_data.get('current_round', 0),
            'status': session_data.get('status', 'restored')
        }
        
        logger.info(f"[RESTORE] Set global_sessions[{session_id}] with {global_sessions[session_id]['message_count']} messages, all_messages length: {len(global_sessions[session_id]['all_messages'])}")
        
        # CRITICAL: Set status so UI can display the session info immediately
        status_msg = 'Restored session' if session_data.get('status') == 'completed' else 'Session ready to resume'
        update_status(session_id, "System", status_msg, 
                     f"Stage: {context.current_stage.value if context else 'unknown'}, Round: {context.conversation_round if context else 0}",
                     research_context=context)
        
        # Set as current session
        session['current_session'] = session_id
        
        # Store the restored context and messages for potential resumption
        session['research_context'] = {
            'session_id': context.session_id if context else session_id,
            'current_stage': context.current_stage.value if context else 'initial_breakdown',
            'conversation_round': context.conversation_round if context else 0,
            'context_maturity': context.context_maturity if context else 0.0
        }
        
        logger.info(f"Restored session {session_id} at stage {context.current_stage.value if context else 'unknown'}, round {context.conversation_round if context else 0}")
        return jsonify({
            'session_id': session_id,
            'message': 'Session restored successfully',
            'can_resume': session_data.get('status') == 'in_progress' and context and context.current_stage != ConversationStage.COMPLETED,
            'session_data': {
                'session_id': session_id,
                'user_prompt': context.user_prompt if context else 'Unknown',
                'message_count': len(messages),
                'saved_at': session_data.get('saved_at'),
                'status': session_data.get('status', 'in_progress'),
                'current_stage': context.current_stage.value if context else 'unknown',
                'conversation_round': context.conversation_round if context else 0
            }
        })
        
    except Exception as e:
        logger.error(f"Failed to restore session {session_id}: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/sessions/resume/<session_id>', methods=['POST'])
def resume_session(session_id):
    """Resume a restored session from where it left off"""
    if orchestrator is None:
        return jsonify({'error': 'System not properly initialized'}), 500
    
    try:
        # Load the full session data
        session_data = load_session(session_id)
        
        if not session_data:
            return jsonify({'error': 'Session not found'}), 404
        
        context = session_data.get('context')
        messages = session_data.get('messages', [])
        
        if not context:
            return jsonify({'error': 'Invalid session data - no context found'}), 400
        
        # Check if session is already completed
        if context.current_stage == ConversationStage.COMPLETED:
            return jsonify({'error': 'Session is already completed'}), 400
        
        # CRITICAL: Restore messages to the context object
        # The messages are loaded separately but need to be in context.messages
        context.messages = messages
        logger.info(f"Restored {len(messages)} messages to context")
        
        # Update global session with current data - CRITICAL: Include all_messages for UI
        all_messages_safe = [
            {
                'llm_type': msg.llm_type.value if hasattr(msg, 'llm_type') else 'system',
                'content': msg.content,
                'timestamp': msg.timestamp.isoformat() if hasattr(msg, 'timestamp') else datetime.now().isoformat(),
                'id': msg.id if hasattr(msg, 'id') else str(uuid.uuid4())
            }
            for msg in messages
        ]
        
        safe_messages = [
            {
                'llm_type': msg.llm_type.value if hasattr(msg, 'llm_type') else 'system',
                'content': msg.content[:200] + '...' if len(msg.content) > 200 else msg.content,
                'timestamp': msg.timestamp.isoformat() if hasattr(msg, 'timestamp') else datetime.now().isoformat()
            }
            for msg in messages[-6:]  # Last 6 messages
        ]
        
        global_sessions[session_id] = {
            'session_id': session_id,
            'current_stage': context.current_stage.value,
            'conversation_round': context.conversation_round,
            'context_maturity': context.context_maturity,
            'quality_gates_passed': context.quality_gates_passed,
            'message_count': len(messages),
            'search_count': len(context.initial_searches) + len(context.targeted_searches),
            'latest_messages': safe_messages,
            'all_messages': all_messages_safe,
            'completed': False,
            'failed': False,
            'error': None,
            'saved_files': {},
            'last_updated': datetime.now().isoformat(),
            'user_prompt': context.user_prompt,
            'current_round': context.conversation_round,
            'status': 'resuming'
        }
        
        # CRITICAL: Set active status so UI shows activity
        update_status(session_id, "System", f"Resuming from stage {context.current_stage.value}", 
                     f"Round {context.conversation_round}", research_context=context)
        
        logger.info(f"Resuming session {session_id} from stage {context.current_stage.value}, round {context.conversation_round}")
        
        # Run the workflow continuation in a background thread
        import threading
        def continue_workflow(research_context, session_id):
            try:
                # Set up status callback
                def status_callback(llm_name, activity, details="", ctx=None):
                    update_status(session_id, llm_name, activity, details, research_context=ctx)
                
                orchestrator.set_status_callback(status_callback)
                
                update_status(session_id, "System", f"Resuming from stage {research_context.current_stage.value}", research_context=research_context)
                logger.info(f"Continuing automated workflow from stage {research_context.current_stage.value}")
                
                # Continue from current stage
                max_rounds = 50
                while research_context.conversation_round < max_rounds and research_context.current_stage != ConversationStage.COMPLETED:
                    update_status(session_id, "System", f"Executing round {research_context.conversation_round + 1}", research_context=research_context)
                    logger.info(f"Executing round {research_context.conversation_round + 1}, Stage: {research_context.current_stage.value}")
                    research_context = orchestrator.execute_conversation_round(research_context)
                    
                    # Update the global session store
                    _update_global_session(session_id, research_context)
                    
                    if research_context.current_stage == ConversationStage.COMPLETED:
                        logger.info("Workflow completed")
                        break
                
                # Finalize documents if completed
                if research_context.current_stage == ConversationStage.COMPLETED:
                    update_status(session_id, "System", "Finalizing documents", research_context=research_context)
                    
                    approved_docs = research_context.metadata.get('approved_documents', [])
                    if approved_docs:
                        # Save documents (same logic as original workflow)
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        project_name = file_manager._sanitize_filename(research_context.user_prompt[:50])
                        project_dir = os.path.join(file_manager.devplan_dir, f"{timestamp}_{project_name}")
                        os.makedirs(project_dir, exist_ok=True)
                        
                        saved_files = {}
                        for idx, doc in enumerate(approved_docs):
                            title = doc.get('title', f'Document_{idx+1}')
                            content = doc.get('content', '')
                            filename = f"{idx+1:02d}_{file_manager._sanitize_filename(title)}.md"
                            filepath = os.path.join(project_dir, filename)
                            
                            with open(filepath, 'w', encoding='utf-8') as f:
                                f.write(content)
                            
                            saved_files[title] = filepath
                            logger.info(f"Saved document: {filepath}")
                        
                        if session_id in global_sessions:
                            global_sessions[session_id]['saved_files'] = saved_files
                    
                    update_status(session_id, "System", "Research workflow completed!", research_context=research_context)
                    if session_id in global_sessions:
                        global_sessions[session_id]['completed'] = True
                        global_sessions[session_id]['status'] = 'completed'
                
            except Exception as e:
                logger.error(f"Error in continued workflow: {e}")
                import traceback
                traceback.print_exc()
                if session_id in global_sessions:
                    global_sessions[session_id]['failed'] = True
                    global_sessions[session_id]['error'] = str(e)
                    global_sessions[session_id]['status'] = 'failed'
        
        thread = threading.Thread(target=continue_workflow, args=(context, session_id))
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'session_id': session_id,
            'message': f'Session resumed from stage {context.current_stage.value}',
            'current_stage': context.current_stage.value,
            'conversation_round': context.conversation_round
        })
        
    except Exception as e:
        logger.error(f"Failed to resume session {session_id}: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/sessions/delete/<session_id>', methods=['DELETE'])
def delete_saved_session(session_id):
    """Delete a saved session"""
    try:
        success = delete_session(session_id)
        
        if success:
            # Also remove from global_sessions if present
            if session_id in global_sessions:
                del global_sessions[session_id]
            
            return jsonify({'message': 'Session deleted successfully'})
        else:
            return jsonify({'error': 'Session not found'}), 404
            
    except Exception as e:
        logger.error(f"Failed to delete session {session_id}: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/download/<path:filepath>', methods=['GET'])
def download_file(filepath):
    """Download a document file"""
    try:
        import os
        from flask import send_file
        
        # Construct full file path
        full_path = os.path.join(os.getcwd(), filepath)
        
        # Security check - ensure the file is within allowed directories
        allowed_dirs = ['DEVPLAN']
        if not any(filepath.startswith(allowed_dir) for allowed_dir in allowed_dirs):
            return jsonify({'error': 'Access denied'}), 403
        
        # Check if file exists
        if not os.path.exists(full_path):
            return jsonify({'error': 'File not found'}), 404
        
        # Determine download name
        download_name = os.path.basename(full_path)
        
        logger.info(f"Downloading file: {full_path}")
        return send_file(full_path, as_attachment=True, download_name=download_name)
        
    except Exception as e:
        logger.error(f"Failed to download file {filepath}: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/generate-files/<session_id>', methods=['POST'])
def generate_files_from_session(session_id):
    """Generate files from session's approved_documents"""
    try:
        session_data = global_sessions.get(session_id)
        
        if not session_data:
            return jsonify({'error': 'Session not found'}), 404
        
        # Check if files already exist
        if session_data.get('saved_files') and len(session_data.get('saved_files', {})) > 0:
            return jsonify({
                'message': 'Files already exist',
                'files': session_data['saved_files']
            })
        
        # Load full session to get approved_documents
        from utils.session_persistence import load_session
        full_session = load_session(session_id)
        
        if not full_session:
            return jsonify({'error': 'Could not load session data'}), 404
        
        context = full_session.get('context')
        if not context:
            return jsonify({'error': 'No context found in session'}), 404
        
        # Get approved documents from metadata
        approved_docs = context.get('metadata', {}).get('approved_documents', [])
        
        if not approved_docs:
            return jsonify({'error': 'No documents to generate'}), 400
        
        # Create file manager
        file_manager = FileManager()
        
        # Create project directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        user_prompt = context.get('user_prompt', 'Project')[:50]
        project_name = file_manager._sanitize_filename(user_prompt)
        project_dir = os.path.join(file_manager.devplan_dir, f"{timestamp}_{project_name}")
        os.makedirs(project_dir, exist_ok=True)
        
        saved_files = {}
        
        for idx, doc in enumerate(approved_docs):
            title = doc.get('title', f'Document_{idx+1}')
            content = doc.get('content', '')
            
            # Save each document as markdown
            filename = file_manager._sanitize_filename(title) + '.md'
            filepath = os.path.join(project_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"# {title}\n\n{content}")
            
            saved_files[title] = filepath
            logger.info(f"Generated file: {title} at {filepath}")
        
        # Update global session with saved files
        global_sessions[session_id]['saved_files'] = saved_files
        
        logger.info(f"Generated {len(saved_files)} files for session {session_id}")
        
        return jsonify({
            'message': f'Successfully generated {len(saved_files)} files',
            'files': saved_files,
            'project_dir': project_dir
        })
        
    except Exception as e:
        logger.error(f"Failed to generate files for session {session_id}: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/download-all/<session_id>', methods=['GET'])
def download_all_files(session_id):
    """Download all files for a session as a ZIP"""
    try:
        import zipfile
        import io
        
        session_data = global_sessions.get(session_id)
        
        if not session_data:
            return jsonify({'error': 'Session not found'}), 404
        
        saved_files = session_data.get('saved_files', {})
        
        if not saved_files:
            return jsonify({'error': 'No files to download'}), 404
        
        # Create ZIP file in memory
        memory_file = io.BytesIO()
        
        with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
            for title, filepath in saved_files.items():
                if os.path.exists(filepath):
                    # Add file to ZIP with just the filename (no full path)
                    arcname = os.path.basename(filepath)
                    zf.write(filepath, arcname)
        
        memory_file.seek(0)
        
        # Generate download name
        user_prompt = session_data.get('user_prompt', 'project')[:30]
        safe_name = ''.join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in user_prompt)
        download_name = f"{safe_name}_documents.zip"
        
        logger.info(f"Downloading ZIP with {len(saved_files)} files for session {session_id}")
        
        return send_file(
            memory_file,
            mimetype='application/zip',
            as_attachment=True,
            download_name=download_name
        )
        
    except Exception as e:
        logger.error(f"Failed to create ZIP for session {session_id}: {e}")
        return jsonify({'error': str(e)}), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)