import logging
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
import concurrent.futures
import re

from config.settings import Config
from core.models import (
    ResearchContext, LLMMessage, LLMType, ConversationStage, 
    SearchResult, QualityLevel
)
from core.deepseek_client import DeepSeekClient
from core.ollama_client import OllamaClient
from core.serper_client import SerperClient
from utils.file_manager import FileManager

logger = logging.getLogger(__name__)


class ConversationOrchestrator:
    """
    STREAMLINED 5-STAGE WORKFLOW
    
    Stage 1: Research Planning (DeepSeek)
    Stage 2: Web Research (Serper)
    Stage 3: Research Analysis (DeepSeek)
    Stage 4: Document Planning (DeepSeek)
    Stage 5: Document Writing (Ollama)
    """
    
    def __init__(self):
        self.deepseek_client = DeepSeekClient()
        self.ollama_client = OllamaClient()
        self.serper_client = SerperClient()
        self.file_manager = FileManager()
        self.max_rounds = 50  # Increased to handle 5 stages + 4 documents
        self.status_callback = None
        self._current_context = None
        
        logger.info("🚀 Conversation Orchestrator initialized with STREAMLINED 5-STAGE WORKFLOW")
    
    def set_status_callback(self, callback):
        """Set the status callback function"""
        self.status_callback = callback
    
    def _update_status(self, llm_name: str, activity: str, details: str = ""):
        """Internal status update that calls the callback if set"""
        if self.status_callback:
            if self._current_context:
                self.status_callback(llm_name, activity, details, self._current_context)
            else:
                self.status_callback(llm_name, activity, details)
        logger.info(f"[{llm_name}] {activity}: {details[:100]}")
    
    def _extract_json_safely(self, content: str) -> Optional[List[str]]:
        """
        Extract JSON array from LLM response
        
        Strategies:
        1. Find ```json ... ```
        2. Find [...] with bracket matching
        3. Parse line by line as fallback
        """
        # Strategy 1: JSON code block
        code_block_match = re.search(r'```json\s*(\[.*?\])\s*```', content, re.DOTALL)
        if code_block_match:
            try:
                return json.loads(code_block_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Strategy 2: Find [...] with proper nesting
        start_idx = content.find('[')
        if start_idx != -1:
            depth = 0
            for i in range(start_idx, len(content)):
                if content[i] == '[':
                    depth += 1
                elif content[i] == ']':
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(content[start_idx:i+1])
                        except json.JSONDecodeError:
                            break
        
        # Strategy 3: Line by line fallback
        lines = [line.strip() for line in content.split('\n')]
        queries = []
        for line in lines:
            # Look for quoted strings or numbered lists
            if line.startswith('"') and line.endswith('"'):
                queries.append(line.strip('"'))
            elif re.match(r'^\d+\.?\s+["\']?(.+)["\']?$', line):
                match = re.match(r'^\d+\.?\s+["\']?(.+?)["\']?\s*$', line)
                if match:
                    queries.append(match.group(1).strip('"\''))
        
        return queries if queries else None
    
    def start_research_session(self, user_prompt: str) -> ResearchContext:
        """Initialize new research session"""
        logger.info(f"✨ Starting STREAMLINED research session for: {user_prompt[:100]}...")
        
        context = ResearchContext(user_prompt=user_prompt)
        context.current_stage = ConversationStage.RESEARCH_PLANNING
        context.conversation_round = 0
        context.metadata['research_queries'] = []
        context.metadata['research_insights'] = []
        context.metadata['document_outlines'] = []
        context.metadata['completed_documents'] = []
        
        self._update_status("System", "Initializing", "Starting 5-stage streamlined workflow")
        
        return context
    
    def execute_conversation_round(self, context: ResearchContext) -> ResearchContext:
        """Execute one stage of the workflow"""
        self._current_context = context
        
        if context.current_stage == ConversationStage.COMPLETED:
            logger.info("✅ Workflow complete")
            return context
        
        if context.conversation_round >= self.max_rounds:
            logger.warning(f"⚠️ Maximum rounds reached ({self.max_rounds}), forcing completion")
            context.current_stage = ConversationStage.COMPLETED
            return context
        
        context.conversation_round += 1
        logger.info(f"🔄 Round {context.conversation_round}, Stage: {context.current_stage.value}")
        
        # Route to appropriate stage handler
        stage_handlers = {
            ConversationStage.RESEARCH_PLANNING: self._stage1_research_planning,
            ConversationStage.WEB_RESEARCH: self._stage2_web_research,
            ConversationStage.RESEARCH_ANALYSIS: self._stage3_research_analysis,
            ConversationStage.DOCUMENT_PLANNING: self._stage4_document_planning,
            ConversationStage.DOCUMENT_WRITING: self._stage5_document_writing,
        }
        
        handler = stage_handlers.get(context.current_stage)
        if handler:
            context = handler(context)
        else:
            logger.error(f"❌ Unknown stage: {context.current_stage}")
            context.current_stage = ConversationStage.COMPLETED
        
        context.update_timestamp()
        return context
    
    # ==================== STAGE 1: RESEARCH PLANNING ====================
    
    def _stage1_research_planning(self, context: ResearchContext) -> ResearchContext:
        """
        DeepSeek analyzes the prompt and creates targeted research queries
        Goal: Generate 15-20 specific, searchable queries
        """
        self._update_status(
            "DeepSeek", 
            "Stage 1/5: Research Planning",
            "Analyzing project and creating research queries..."
        )
        
        logger.info("📋 Stage 1: Research Planning - Creating targeted queries")
        
        prompt = f"""Analyze this project request and create targeted research queries:

PROJECT REQUEST:
{context.user_prompt}

Your task: Create 15-20 specific, searchable research queries that will gather comprehensive technical information.

QUERY CATEGORIES (distribute queries across these):
1. Technology Stack (3-4 queries) - specific frameworks, libraries, tools
2. Architecture Patterns (3-4 queries) - design patterns, system architecture
3. Implementation Guides (4-5 queries) - step-by-step tutorials, code examples
4. Best Practices (2-3 queries) - industry standards, recommendations
5. Common Pitfalls (2-3 queries) - known issues, debugging, troubleshooting
6. Comparisons (2-3 queries) - tool/framework comparisons, alternatives

REQUIREMENTS:
- Each query must be 5-15 words
- Queries must be specific and searchable (not vague)
- Focus on technical implementation details
- Include version numbers where relevant
- Make queries actionable

Return ONLY a valid JSON array of strings:
["query 1", "query 2", "query 3", ...]

Example good queries:
["Next.js 14 App Router complete tutorial", "PostgreSQL vs MongoDB for e-commerce scalability", "Docker multi-stage build Node.js production"]

Example bad queries:
["how to build a website", "best practices", "web development guide"]

Generate the JSON array now:"""

        message = self.deepseek_client.generate_response(
            prompt, 
            max_tokens=2000,
            temperature=0.7
        )
        context.add_message(message)
        
        # Extract queries
        queries = self._extract_json_safely(message.content)
        
        if not queries or len(queries) < 10:
            logger.warning("⚠️ Failed to extract queries, using fallback approach")
            # Fallback: create generic queries
            queries = [
                f"{context.user_prompt} tutorial guide",
                f"{context.user_prompt} best practices",
                f"{context.user_prompt} architecture design",
                f"{context.user_prompt} implementation steps",
                f"{context.user_prompt} code examples",
                f"{context.user_prompt} configuration setup",
                f"{context.user_prompt} testing strategy",
                f"{context.user_prompt} deployment guide",
                f"{context.user_prompt} security considerations",
                f"{context.user_prompt} performance optimization",
                f"{context.user_prompt} common issues",
                f"{context.user_prompt} troubleshooting guide",
                f"{context.user_prompt} comparison alternatives",
                f"{context.user_prompt} production ready",
                f"{context.user_prompt} complete example project"
            ]
        
        context.metadata['research_queries'] = queries
        logger.info(f"✅ Generated {len(queries)} research queries")
        
        self._update_status(
            "DeepSeek",
            "Stage 1/5: Complete",
            f"Created {len(queries)} targeted research queries"
        )
        
        # Move to next stage
        context.current_stage = ConversationStage.WEB_RESEARCH
        return context
    
    # ==================== STAGE 2: WEB RESEARCH ====================
    
    def _stage2_web_research(self, context: ResearchContext) -> ResearchContext:
        """
        Perform parallel web searches for all queries
        Goal: Gather ~100 sources with content
        """
        queries = context.metadata.get('research_queries', [])
        
        self._update_status(
            "Serper",
            "Stage 2/5: Web Research",
            f"Searching {len(queries)} queries in parallel..."
        )
        
        logger.info(f"🔍 Stage 2: Web Research - Searching {len(queries)} queries")
        
        all_results = []
        
        # Perform parallel searches
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_query = {
                executor.submit(self._search_single_query, query, idx): (query, idx)
                for idx, query in enumerate(queries)
            }
            
            for future in concurrent.futures.as_completed(future_to_query):
                query, idx = future_to_query[future]
                try:
                    results = future.result()
                    all_results.extend(results)
                    logger.info(f"✓ Query {idx+1}/{len(queries)}: {len(results)} results")
                    
                    self._update_status(
                        "Serper",
                        f"Stage 2/5: Searching ({idx+1}/{len(queries)})",
                        f"Found {len(all_results)} sources so far..."
                    )
                except Exception as e:
                    logger.error(f"✗ Query {idx+1} failed: {e}")
        
        context.targeted_searches = all_results
        context.metadata['total_sources'] = len(all_results)
        
        logger.info(f"✅ Collected {len(all_results)} total sources")
        
        self._update_status(
            "Serper",
            "Stage 2/5: Complete",
            f"Collected {len(all_results)} sources from {len(queries)} queries"
        )
        
        # Move to next stage
        context.current_stage = ConversationStage.RESEARCH_ANALYSIS
        return context
    
    def _search_single_query(self, query: str, idx: int) -> List[SearchResult]:
        """Perform a single search query"""
        try:
            results = self.serper_client.search(query, num_results=8)
            return results
        except Exception as e:
            logger.error(f"Search failed for query {idx}: {e}")
            return []
    
    # ==================== STAGE 3: RESEARCH ANALYSIS ====================
    
    def _stage3_research_analysis(self, context: ResearchContext) -> ResearchContext:
        """
        DeepSeek analyzes all research and extracts insights
        Goal: Create comprehensive research summary (~15K tokens)
        """
        self._update_status(
            "DeepSeek",
            "Stage 3/5: Research Analysis",
            f"Analyzing {len(context.targeted_searches)} sources..."
        )
        
        logger.info(f"📊 Stage 3: Research Analysis - Analyzing {len(context.targeted_searches)} sources")
        
        # Build research context (fit within 128K context window)
        research_text = self._build_research_text(context.targeted_searches, max_length=100000)
        
        prompt = f"""Analyze these research results and extract comprehensive technical insights:

PROJECT: {context.user_prompt}

RESEARCH RESULTS ({len(context.targeted_searches)} sources):
{research_text}

Create a detailed technical analysis with:

## 1. TECHNOLOGY STACK RECOMMENDATIONS
- Primary stack with justifications
- Alternative options with pros/cons
- Version recommendations
- Compatibility considerations

## 2. ARCHITECTURE PATTERNS
- Recommended architecture (with diagram description)
- Key components and responsibilities
- Data flow and interactions
- Scalability considerations

## 3. IMPLEMENTATION APPROACH
- Phase-by-phase implementation plan
- Key milestones
- Critical paths
- Dependencies

## 4. CODE PATTERNS & EXAMPLES
- Essential code patterns found
- Configuration requirements
- Setup procedures
- Integration approaches

## 5. BEST PRACTICES
- Security recommendations
- Performance optimizations
- Testing strategies
- Error handling patterns

## 6. COMMON PITFALLS & SOLUTIONS
- Known issues from sources
- Debugging approaches
- Troubleshooting tips
- Migration gotchas

## 7. TOOL & LIBRARY MATRIX
Create a comparison table of tools/libraries mentioned

## 8. KEY INSIGHTS
List 10-15 most important technical insights

IMPORTANT: Cite sources throughout using [Source: URL]
Be comprehensive - use up to 8,000 tokens for this analysis."""

        message = self.deepseek_client.generate_response(
            prompt,
            context=research_text[:50000],  # Additional context
            max_tokens=8000,
            temperature=0.3
        )
        context.add_message(message)
        
        context.metadata['research_analysis'] = message.content
        context.metadata['analysis_length'] = len(message.content)
        
        logger.info(f"✅ Research analysis complete ({len(message.content)} chars)")
        
        self._update_status(
            "DeepSeek",
            "Stage 3/5: Complete",
            f"Extracted insights from {len(context.targeted_searches)} sources"
        )
        
        # Move to next stage
        context.current_stage = ConversationStage.DOCUMENT_PLANNING
        return context
    
    def _build_research_text(self, results: List[SearchResult], max_length: int = 100000) -> str:
        """Build research context from search results"""
        texts = []
        current_length = 0
        
        for result in results:
            text = f"Title: {result.title}\nURL: {result.link}\nContent: {result.snippet}\n---\n"
            if current_length + len(text) > max_length:
                break
            texts.append(text)
            current_length += len(text)
        
        return "\n".join(texts)
    
    # ==================== STAGE 4: DOCUMENT PLANNING ====================
    
    def _stage4_document_planning(self, context: ResearchContext) -> ResearchContext:
        """
        DeepSeek creates detailed outlines for documents
        Goal: Create 3-4 detailed document outlines
        """
        self._update_status(
            "DeepSeek",
            "Stage 4/5: Document Planning",
            "Creating document outlines..."
        )
        
        logger.info("📝 Stage 4: Document Planning - Creating outlines")
        
        research_analysis = context.metadata.get('research_analysis', '')
        
        # Define documents to create
        documents = [
            {
                "title": "System Architecture & Setup Guide",
                "focus": "Architecture, technology stack, environment setup, project initialization"
            },
            {
                "title": "Backend Implementation Guide",
                "focus": "Backend code, API, database, authentication, business logic"
            },
            {
                "title": "Frontend Implementation Guide",
                "focus": "Frontend code, UI components, state management, routing"
            },
            {
                "title": "DevOps & Deployment Guide",
                "focus": "Docker, CI/CD, deployment, monitoring, operations"
            }
        ]
        
        outlines = []
        
        for idx, doc in enumerate(documents):
            self._update_status(
                "DeepSeek",
                f"Stage 4/5: Planning ({idx+1}/{len(documents)})",
                f"Creating outline for: {doc['title']}"
            )
            
            outline = self.deepseek_client.create_document_outline(
                doc_number=idx+1,
                doc_type=doc['title'],
                user_prompt=context.user_prompt,
                research_context=research_analysis,
                conversation_summary=research_analysis[:8000]
            )
            
            outlines.append({
                "title": doc['title'],
                "outline": outline.content,
                "focus": doc['focus']
            })
            
            logger.info(f"✓ Outline {idx+1}/4 created ({len(outline.content)} chars)")
        
        context.metadata['document_outlines'] = outlines
        
        logger.info(f"✅ Created {len(outlines)} document outlines")
        
        self._update_status(
            "DeepSeek",
            "Stage 4/5: Complete",
            f"Created {len(outlines)} detailed document outlines"
        )
        
        # Move to next stage
        context.current_stage = ConversationStage.DOCUMENT_WRITING
        context.metadata['current_document_index'] = 0
        return context
    
    # ==================== STAGE 5: DOCUMENT WRITING ====================
    
    def _stage5_document_writing(self, context: ResearchContext) -> ResearchContext:
        """
        Ollama writes comprehensive documents
        Goal: Write all documents (20K-30K words each)
        """
        outlines = context.metadata.get('document_outlines', [])
        current_idx = context.metadata.get('current_document_index', 0)
        completed = context.metadata.get('completed_documents', [])
        
        # Safety check
        if not outlines or len(outlines) == 0:
            logger.error("❌ No document outlines found! Cannot write documents.")
            self._update_status("System", "ERROR", "No document outlines - Stage 4 may have failed")
            context.metadata['error'] = "No document outlines created in Stage 4"
            context.current_stage = ConversationStage.COMPLETED
            return context
        
        if current_idx >= len(outlines):
            logger.info(f"✅ All {len(outlines)} documents written")
            
            # Save final summary
            self._save_session_summary(context, completed)
            
            self._update_status(
                "Ollama",
                "Stage 5/5: Complete",
                f"All {len(completed)} documents written successfully"
            )
            context.current_stage = ConversationStage.COMPLETED
            return context
        
        current_outline = outlines[current_idx]
        
        self._update_status(
            "Ollama",
            f"Stage 5/5: Writing Document {current_idx+1}/{len(outlines)}",
            f"Writing: {current_outline['title']}"
        )
        
        logger.info(f"✍️ Stage 5: Writing document {current_idx+1}/{len(outlines)}: {current_outline['title']}")
        
        research_summary = context.metadata.get('research_analysis', '')
        
        document = self.ollama_client.write_document_from_outline(
            outline=current_outline['outline'],
            research_summary=research_summary,
            doc_number=current_idx+1,
            doc_type=current_outline['title'],
            user_prompt=context.user_prompt
        )
        
        # Generate filename
        filename = f"{current_idx+1:02d}_{current_outline['title'].lower().replace(' ', '_').replace('&', 'and')}.md"
        
        # Save document to disk immediately
        saved_path = self.file_manager.save_document(
            session_id=context.session_id,
            title=current_outline['title'],
            content=document.content
        )
        
        completed.append({
            "title": current_outline['title'],
            "filename": filename,
            "filepath": saved_path,
            "content": document.content,
            "word_count": len(document.content.split()),
            "char_count": len(document.content)
        })
        
        context.metadata['completed_documents'] = completed
        context.metadata['current_document_index'] = current_idx + 1
        
        logger.info(f"✅ Document {current_idx+1} complete ({len(document.content)} chars, {len(document.content.split())} words)")
        logger.info(f"💾 Saved to: {saved_path}")
        
        self._update_status(
            "Ollama",
            f"Stage 5/5: Document {current_idx+1} Complete",
            f"Wrote {len(document.content.split())} words"
        )
        
        # Continue to next document (or complete)
        return context
    
    # ==================== HELPER METHODS ====================
    
    def get_progress_summary(self, context: ResearchContext) -> Dict[str, Any]:
        """Get current progress summary"""
        stage_progress = {
            ConversationStage.RESEARCH_PLANNING: 20,
            ConversationStage.WEB_RESEARCH: 40,
            ConversationStage.RESEARCH_ANALYSIS: 60,
            ConversationStage.DOCUMENT_PLANNING: 70,
            ConversationStage.DOCUMENT_WRITING: 95,
            ConversationStage.COMPLETED: 100
        }
        
        queries = context.metadata.get('research_queries', [])
        sources = context.metadata.get('total_sources', 0)
        outlines = context.metadata.get('document_outlines', [])
        completed_docs = context.metadata.get('completed_documents', [])
        current_doc_idx = context.metadata.get('current_document_index', 0)
        
        return {
            "stage": context.current_stage.value,
            "stage_name": self._get_stage_name(context.current_stage),
            "progress_percent": stage_progress.get(context.current_stage, 0),
            "research_queries": len(queries),
            "sources_collected": sources,
            "documents_planned": len(outlines),
            "documents_completed": len(completed_docs),
            "documents_total": len(outlines),
            "current_document": current_doc_idx + 1 if current_doc_idx < len(outlines) else len(outlines),
            "total_words": sum(doc.get('word_count', 0) for doc in completed_docs)
        }
    
    def _get_stage_name(self, stage: ConversationStage) -> str:
        """Get human-readable stage name"""
        names = {
            ConversationStage.RESEARCH_PLANNING: "Research Planning",
            ConversationStage.WEB_RESEARCH: "Web Research",
            ConversationStage.RESEARCH_ANALYSIS: "Research Analysis",
            ConversationStage.DOCUMENT_PLANNING: "Document Planning",
            ConversationStage.DOCUMENT_WRITING: "Document Writing",
            ConversationStage.COMPLETED: "Completed"
        }
        return names.get(stage, stage.value)
    
    def _save_session_summary(self, context: ResearchContext, completed_docs: List[Dict]) -> None:
        """Save a summary file with all document information"""
        try:
            from datetime import datetime
            
            # Build summary content
            summary_lines = [
                f"# Research Session Summary",
                f"",
                f"**Session ID**: {context.session_id}",
                f"**User Prompt**: {context.user_prompt}",
                f"**Completed**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"**Total Rounds**: {context.conversation_round}",
                f"",
                f"## Research Statistics",
                f"",
                f"- **Research Queries**: {len(context.metadata.get('research_queries', []))}",
                f"- **Sources Collected**: {context.metadata.get('total_sources', 0)}",
                f"- **Documents Generated**: {len(completed_docs)}",
                f"- **Total Words**: {sum(doc.get('word_count', 0) for doc in completed_docs):,}",
                f"- **Total Characters**: {sum(doc.get('char_count', 0) for doc in completed_docs):,}",
                f"",
                f"## Generated Documents",
                f""
            ]
            
            # List each document
            for idx, doc in enumerate(completed_docs, 1):
                summary_lines.extend([
                    f"### {idx}. {doc['title']}",
                    f"",
                    f"- **Filename**: `{doc['filename']}`",
                    f"- **Location**: `{doc.get('filepath', 'N/A')}`",
                    f"- **Word Count**: {doc.get('word_count', 0):,}",
                    f"- **Character Count**: {doc.get('char_count', 0):,}",
                    f""
                ])
            
            summary_content = "\n".join(summary_lines)
            
            # Save summary file
            summary_path = self.file_manager.save_document(
                session_id=context.session_id,
                title="00_SESSION_SUMMARY",
                content=summary_content
            )
            
            logger.info(f"📋 Session summary saved: {summary_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save session summary: {e}")
