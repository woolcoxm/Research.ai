import logging
import json
import uuid
import os
from datetime import datetime
from flask import Flask, render_template, request, jsonify, session, send_file
from config.settings import Config
from core.conversation_orchestrator import ConversationOrchestrator
from core.models import ResearchContext, ConversationStage
from utils.file_manager import FileManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'ai_research_system_secret_key'  # In production, use a proper secret key

# Global session store for real-time updates (in production, use Redis or similar)
global_sessions = {}

# Status tracking for active sessions
active_status = {}

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

def _update_global_session(session_id, research_context, completed=False, failed=False, error=None, saved_files=None):
    """Update global session store for real-time tracking"""
    # Get latest messages (last 6 for display)
    latest_messages = []
    for msg in research_context.messages[-6:]:
        latest_messages.append({
            'llm_type': msg.llm_type.value,
            'content': msg.content[:200] + '...' if len(msg.content) > 200 else msg.content,  # Truncate for UI
            'timestamp': msg.timestamp.isoformat()
        })
    
    # Store ALL messages organized by LLM type for conversation history
    all_messages = []
    for msg in research_context.messages:
        all_messages.append({
            'llm_type': msg.llm_type.value,
            'content': msg.content,
            'timestamp': msg.timestamp.isoformat(),
            'id': msg.id
        })
    
    logger.info(f"Updating global session {session_id}: {len(all_messages)} messages stored")
    
    # Build research metrics
    research_metrics = {
        'total_searches': len(research_context.initial_searches) + len(research_context.targeted_searches),
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
        'user_prompt': research_context.user_prompt[:100] + '...' if len(research_context.user_prompt) > 100 else research_context.user_prompt
    }

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
        
        for msg in all_messages:
            msg_data = {
                'content': msg.get('content', ''),
                'timestamp': msg.get('timestamp', ''),
                'msg_id': msg.get('id', ''),
                'preview': msg.get('content', '')[:150] + '...' if len(msg.get('content', '')) > 150 else msg.get('content', '')
            }
            
            if msg.get('llm_type') == 'deepseek':
                msg_data['number'] = deepseek_counter
                deepseek_messages.append(msg_data)
                deepseek_counter += 1
            elif msg.get('llm_type') == 'ollama':
                msg_data['number'] = ollama_counter
                ollama_messages.append(msg_data)
                ollama_counter += 1
        
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


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)