import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from config.settings import Config

logger = logging.getLogger(__name__)


class FileManager:
    """Manages file operations for development plans"""
    
    def __init__(self):
        self.devplan_dir = Config.DEVPLAN_DIR
        self._ensure_devplan_directory()
    
    def _ensure_devplan_directory(self):
        """Ensure the DEVPLAN directory exists"""
        os.makedirs(self.devplan_dir, exist_ok=True)
        logger.info(f"DEVPLAN directory ensured: {self.devplan_dir}")
    
    def save_development_plan(self, plan_data: Dict[str, Any]) -> str:
        """Save a development plan to file as markdown"""
        try:
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            project_name = self._sanitize_filename(plan_data.get('project_name', 'unknown_project'))
            
            # Create directory for this project
            project_dir = os.path.join(self.devplan_dir, f"{timestamp}_{project_name}")
            os.makedirs(project_dir, exist_ok=True)
            
            saved_files = {}
            
            # Save main development plan as markdown
            main_plan_filename = "development_plan.md"
            main_plan_filepath = os.path.join(project_dir, main_plan_filename)
            
            with open(main_plan_filepath, 'w', encoding='utf-8') as f:
                f.write(plan_data.get('development_plan', ''))
            
            saved_files['Development Plan'] = main_plan_filepath
            logger.info(f"Main development plan saved: {main_plan_filepath}")
            
            # Save research summary
            summary_filepath = os.path.join(project_dir, 'research_summary.md')
            summary_content = self._create_research_summary_markdown(plan_data)
            
            with open(summary_filepath, 'w', encoding='utf-8') as f:
                f.write(summary_content)
            
            saved_files['Research Summary'] = summary_filepath
            logger.info(f"Research summary saved: {summary_filepath}")
            
            logger.info(f"Development plan saved to: {project_dir}")
            return main_plan_filepath
            
        except Exception as e:
            logger.error(f"Failed to save development plan: {e}")
            raise
    
    def load_development_plan(self, filename: str) -> Optional[Dict[str, Any]]:
        """Load a development plan from file"""
        try:
            filepath = os.path.join(self.devplan_dir, filename)
            
            if not os.path.exists(filepath):
                logger.error(f"Development plan file not found: {filepath}")
                return None
            
            with open(filepath, 'r', encoding='utf-8') as f:
                plan_data = json.load(f)
            
            logger.info(f"Development plan loaded: {filename}")
            return plan_data
            
        except Exception as e:
            logger.error(f"Failed to load development plan {filename}: {e}")
            return None
    
    def list_development_plans(self) -> List[Dict[str, Any]]:
        """List all development plans in the DEVPLAN directory"""
        try:
            plans = []
            
            for filename in os.listdir(self.devplan_dir):
                if filename.endswith('.json'):
                    filepath = os.path.join(self.devplan_dir, filename)
                    
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            plan_data = json.load(f)
                        
                        # Extract basic info for listing
                        plan_info = {
                            'filename': filename,
                            'project_name': plan_data.get('project_name', 'Unknown Project'),
                            'user_prompt': plan_data.get('user_prompt', ''),
                            'generated_at': plan_data.get('generated_at', ''),
                            'session_id': plan_data.get('session_id', ''),
                            'file_size': os.path.getsize(filepath),
                            'feasibility_score': plan_data.get('feasibility_assessment', {}).get('feasibility_score', 0.0)
                        }
                        plans.append(plan_info)
                        
                    except Exception as e:
                        logger.warning(f"Failed to read plan file {filename}: {e}")
                        continue
            
            # Sort by generation date (newest first)
            plans.sort(key=lambda x: x.get('generated_at', ''), reverse=True)
            
            logger.info(f"Listed {len(plans)} development plans")
            return plans
            
        except Exception as e:
            logger.error(f"Failed to list development plans: {e}")
            return []
    
    def delete_development_plan(self, filename: str) -> bool:
        """Delete a development plan file"""
        try:
            filepath = os.path.join(self.devplan_dir, filename)
            
            if not os.path.exists(filepath):
                logger.error(f"Development plan file not found: {filepath}")
                return False
            
            os.remove(filepath)
            logger.info(f"Development plan deleted: {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete development plan {filename}: {e}")
            return False
    
    def export_plan_to_markdown(self, plan_data: Dict[str, Any], output_path: Optional[str] = None) -> str:
        """Export a development plan to markdown format"""
        try:
            if output_path is None:
                project_name = self._sanitize_filename(plan_data.get('project_name', 'unknown_project'))
                output_path = os.path.join(self.devplan_dir, f"{project_name}.md")
            
            markdown_content = self._convert_plan_to_markdown(plan_data)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            
            logger.info(f"Development plan exported to markdown: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to export plan to markdown: {e}")
            raise
    
    def _convert_plan_to_markdown(self, plan_data: Dict[str, Any]) -> str:
        """Convert development plan data to markdown format"""
        lines = []
        
        # Header
        lines.append(f"# {plan_data.get('project_name', 'Development Plan')}")
        lines.append("")
        
        # Metadata
        lines.append("## Project Information")
        lines.append(f"- **User Prompt**: {plan_data.get('user_prompt', '')}")
        lines.append(f"- **Generated**: {plan_data.get('generated_at', '')}")
        lines.append(f"- **Session ID**: {plan_data.get('session_id', '')}")
        lines.append("")
        
        # Development Plan Content
        lines.append("## Development Plan")
        development_plan = plan_data.get('development_plan', '')
        lines.append(development_plan)
        lines.append("")
        
        # Feasibility Assessment
        feasibility = plan_data.get('feasibility_assessment', {})
        if feasibility:
            lines.append("## Feasibility Assessment")
            lines.append(f"- **Feasibility Score**: {feasibility.get('feasibility_score', 0.0):.2f}/1.0")
            lines.append("")
            lines.append("### Technical Feedback")
            lines.append(feasibility.get('technical_feedback', ''))
            lines.append("")
            
            risks = feasibility.get('risks_identified', [])
            if risks:
                lines.append("### Identified Risks")
                for risk in risks:
                    lines.append(f"- {risk}")
                lines.append("")
            
            recommendations = feasibility.get('recommendations', [])
            if recommendations:
                lines.append("### Recommendations")
                for rec in recommendations:
                    lines.append(f"- {rec}")
                lines.append("")
        
        # Research Metrics
        metrics = plan_data.get('research_metrics', {})
        if metrics:
            lines.append("## Research Metrics")
            lines.append(f"- **Total Searches**: {metrics.get('total_searches', 0)}")
            lines.append(f"- **Key Insights**: {metrics.get('key_insights', 0)}")
            lines.append(f"- **Conversation Rounds**: {metrics.get('conversation_rounds', 0)}")
            lines.append(f"- **Context Maturity**: {metrics.get('context_maturity', 0.0):.2f}/1.0")
            lines.append(f"- **Quality Gates Passed**: {', '.join(metrics.get('quality_gates_passed', []))}")
            lines.append("")
        
        return "\n".join(lines)
    
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize a string to be safe for use as a filename"""
        # Replace problematic characters including newlines and carriage returns
        sanitized = filename.replace('\n', '_').replace('\r', '_')
        sanitized = sanitized.replace(' ', '_').replace('/', '_').replace('\\', '_')
        sanitized = sanitized.replace(':', '_').replace('*', '_').replace('?', '_')
        sanitized = sanitized.replace('"', '_').replace('<', '_').replace('>', '_')
        sanitized = sanitized.replace('|', '_').replace('\t', '_')
        
        # Remove any remaining control characters
        sanitized = ''.join(c if c.isprintable() or c in ('_', '-') else '_' for c in sanitized)
        
        # Limit length
        if len(sanitized) > 100:
            sanitized = sanitized[:100]
        
        return sanitized
    
    def _create_research_summary_markdown(self, plan_data: Dict[str, Any]) -> str:
        """Create a markdown summary of the research session"""
        lines = []
        
        # Header
        lines.append(f"# Research Summary: {plan_data.get('project_name', 'Unknown Project')}")
        lines.append("")
        lines.append("---")
        lines.append("")
        
        # Session Information
        lines.append("## Session Information")
        lines.append("")
        lines.append(f"- **Session ID**: {plan_data.get('session_id', 'N/A')}")
        lines.append(f"- **Generated**: {plan_data.get('generated_at', 'N/A')}")
        lines.append(f"- **User Prompt**: {plan_data.get('user_prompt', 'N/A')}")
        lines.append("")
        
        # Research Metrics
        metrics = plan_data.get('research_metrics', {})
        if metrics:
            lines.append("## Research Metrics")
            lines.append("")
            lines.append(f"- **Total Searches**: {metrics.get('total_searches', 0)}")
            lines.append(f"- **Key Insights Extracted**: {metrics.get('key_insights', 0)}")
            lines.append(f"- **Conversation Rounds**: {metrics.get('conversation_rounds', 0)}")
            lines.append(f"- **Context Maturity**: {metrics.get('context_maturity', 0.0):.2%}")
            
            quality_gates = metrics.get('quality_gates_passed', [])
            if quality_gates:
                lines.append(f"- **Quality Gates Passed**: {', '.join(quality_gates)}")
            lines.append("")
        
        # Feasibility Assessment
        feasibility = plan_data.get('feasibility_assessment', {})
        if feasibility:
            lines.append("## Feasibility Assessment")
            lines.append("")
            
            feasibility_score = feasibility.get('feasibility_score', 0.0)
            lines.append(f"**Feasibility Score**: {feasibility_score:.2f}/1.0")
            lines.append("")
            
            technical_feedback = feasibility.get('technical_feedback', '')
            if technical_feedback:
                lines.append("### Technical Feedback")
                lines.append("")
                lines.append(technical_feedback)
                lines.append("")
            
            risks = feasibility.get('risks_identified', [])
            if risks:
                lines.append("### Identified Risks")
                lines.append("")
                for risk in risks:
                    lines.append(f"- {risk}")
                lines.append("")
            
            recommendations = feasibility.get('recommendations', [])
            if recommendations:
                lines.append("### Recommendations")
                lines.append("")
                for rec in recommendations:
                    lines.append(f"- {rec}")
                lines.append("")
        
        # Conversation Summary
        conversation_summary = plan_data.get('conversation_summary', '')
        if conversation_summary:
            lines.append("## Conversation Summary")
            lines.append("")
            lines.append(conversation_summary)
            lines.append("")
        
        # Documents Generated
        documents = plan_data.get('documents', [])
        if documents:
            lines.append("## Generated Documents")
            lines.append("")
            for doc in documents:
                lines.append(f"- **{doc.get('title', 'Untitled')}** (`{doc.get('filename', 'unknown.md')}`)")
            lines.append("")
        
        lines.append("---")
        lines.append("")
        lines.append("*This summary was automatically generated by the AI Research System*")
        
        return "\n".join(lines)
    
    def get_plan_statistics(self) -> Dict[str, Any]:
        """Get statistics about stored development plans"""
        try:
            plans = self.list_development_plans()
            
            if not plans:
                return {
                    'total_plans': 0,
                    'average_feasibility': 0.0,
                    'recent_plans': 0
                }
            
            # Calculate statistics
            feasibility_scores = [p.get('feasibility_score', 0.0) for p in plans if p.get('feasibility_score', 0.0) > 0]
            average_feasibility = sum(feasibility_scores) / len(feasibility_scores) if feasibility_scores else 0.0
            
            # Count recent plans (last 7 days)
            week_ago = datetime.now().timestamp() - (7 * 24 * 60 * 60)
            recent_plans = sum(1 for p in plans if self._is_recent(p.get('generated_at', ''), week_ago))
            
            return {
                'total_plans': len(plans),
                'average_feasibility': round(average_feasibility, 2),
                'recent_plans': recent_plans,
                'total_file_size': sum(p.get('file_size', 0) for p in plans)
            }
            
        except Exception as e:
            logger.error(f"Failed to get plan statistics: {e}")
            return {
                'total_plans': 0,
                'average_feasibility': 0.0,
                'recent_plans': 0,
                'total_file_size': 0
            }
    
    def _is_recent(self, timestamp_str: str, cutoff_timestamp: float) -> bool:
        """Check if a timestamp is recent (within cutoff)"""
        try:
            if not timestamp_str:
                return False
            
            plan_timestamp = datetime.fromisoformat(timestamp_str).timestamp()
            return plan_timestamp >= cutoff_timestamp
            
        except (ValueError, TypeError):
            return False
    
    def save_multiple_documents(self, plan_data: Dict[str, Any]) -> Dict[str, str]:
        """Save multiple documents for a single project - documents as .md, metadata as .json"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            project_name = self._sanitize_filename(plan_data.get('project_name', 'unknown_project'))
            
            # Create a subdirectory for this project's documents
            project_dir = os.path.join(self.devplan_dir, f"{timestamp}_{project_name}")
            os.makedirs(project_dir, exist_ok=True)
            
            saved_files = {}
            documents = plan_data.get('documents', [])
            
            # Save each document as a separate markdown file
            for doc in documents:
                filename = doc['filename']
                filepath = os.path.join(project_dir, filename)
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(doc['content'])
                
                saved_files[doc['title']] = filepath
                logger.info(f"Document saved: {filepath}")
            
            # Save the research summary as markdown
            summary_filepath = os.path.join(project_dir, 'research_summary.md')
            summary_content = self._create_research_summary_markdown(plan_data)
            
            with open(summary_filepath, 'w', encoding='utf-8') as f:
                f.write(summary_content)
            
            saved_files['Research Summary'] = summary_filepath
            logger.info(f"Research summary saved: {summary_filepath}")
            
            # Save project metadata as JSON (without document content - just references)
            metadata = {
                'project_name': plan_data.get('project_name', 'Unknown Project'),
                'user_prompt': plan_data.get('user_prompt', ''),
                'generated_at': plan_data.get('generated_at', datetime.now().isoformat()),
                'session_id': plan_data.get('session_id', ''),
                'multi_document': plan_data.get('multi_document', True),
                'documents': [
                    {
                        'title': doc.get('title', 'Untitled'),
                        'filename': doc.get('filename', 'document.md'),
                        'category': doc.get('category', 'general'),
                        'size': len(doc.get('content', ''))
                    }
                    for doc in documents
                ],
                'research_metrics': plan_data.get('research_metrics', {}),
                'saved_files': {title: os.path.basename(path) for title, path in saved_files.items()}
            }
            
            metadata_filepath = os.path.join(project_dir, 'project_plan.json')
            with open(metadata_filepath, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Project metadata saved: {metadata_filepath}")
            logger.info(f"Multiple documents saved in: {project_dir}")
            return saved_files
            
        except Exception as e:
            logger.error(f"Failed to save multiple documents: {e}")
            raise
    
    def get_document_download_url(self, filepath: str, base_url: str = "") -> str:
        """Generate download URL for a document"""
        try:
            # Convert absolute path to relative path for web serving
            relative_path = os.path.relpath(filepath, start=os.getcwd())
            # Convert Windows paths to web paths
            web_path = relative_path.replace('\\', '/')
            return f"{base_url}/download/{web_path}"
        except Exception as e:
            logger.error(f"Failed to generate download URL for {filepath}: {e}")
            return ""
    
    def list_project_documents(self) -> List[Dict[str, Any]]:
        """List all projects with their documents"""
        try:
            projects = []
            
            for item in os.listdir(self.devplan_dir):
                item_path = os.path.join(self.devplan_dir, item)
                
                # Check if it's a directory (multi-document project)
                if os.path.isdir(item_path):
                    project_plan_path = os.path.join(item_path, 'project_plan.json')
                    
                    if os.path.exists(project_plan_path):
                        try:
                            with open(project_plan_path, 'r', encoding='utf-8') as f:
                                plan_data = json.load(f)
                            
                            # List all document files in the directory
                            documents = []
                            for file in os.listdir(item_path):
                                if file.endswith('.md'):
                                    file_path = os.path.join(item_path, file)
                                    file_size = os.path.getsize(file_path)
                                    
                                    # Try to match with document metadata
                                    doc_title = file
                                    for doc in plan_data.get('documents', []):
                                        if doc.get('filename') == file:
                                            doc_title = doc.get('title', file)
                                            break
                                    
                                    documents.append({
                                        'title': doc_title,
                                        'filename': file,
                                        'filepath': file_path,
                                        'size': file_size,
                                        'download_url': self.get_document_download_url(file_path)
                                    })
                            
                            project_info = {
                                'type': 'multi_document',
                                'project_name': plan_data.get('project_name', 'Unknown Project'),
                                'user_prompt': plan_data.get('user_prompt', ''),
                                'generated_at': plan_data.get('generated_at', ''),
                                'session_id': plan_data.get('session_id', ''),
                                'project_directory': item_path,
                                'document_count': len(documents),
                                'documents': documents,
                                'total_size': sum(doc['size'] for doc in documents)
                            }
                            projects.append(project_info)
                            
                        except Exception as e:
                            logger.warning(f"Failed to read project plan {project_plan_path}: {e}")
                            continue
                
                # Also handle single JSON files (legacy format)
                elif item.endswith('.json'):
                    try:
                        filepath = os.path.join(self.devplan_dir, item)
                        with open(filepath, 'r', encoding='utf-8') as f:
                            plan_data = json.load(f)
                        
                        project_info = {
                            'type': 'single_document',
                            'filename': item,
                            'project_name': plan_data.get('project_name', 'Unknown Project'),
                            'user_prompt': plan_data.get('user_prompt', ''),
                            'generated_at': plan_data.get('generated_at', ''),
                            'session_id': plan_data.get('session_id', ''),
                            'file_size': os.path.getsize(filepath),
                            'download_url': self.get_document_download_url(filepath),
                            'feasibility_score': plan_data.get('feasibility_assessment', {}).get('feasibility_score', 0.0)
                        }
                        projects.append(project_info)
                        
                    except Exception as e:
                        logger.warning(f"Failed to read plan file {item}: {e}")
                        continue
            
            # Sort by generation date (newest first)
            projects.sort(key=lambda x: x.get('generated_at', ''), reverse=True)
            
            logger.info(f"Listed {len(projects)} projects")
            return projects
            
        except Exception as e:
            logger.error(f"Failed to list project documents: {e}")
            return []