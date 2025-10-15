import logging
import json
from typing import List, Dict, Any, Optional
from datetime import datetime

from config.settings import Config
from core.models import (
    ResearchContext, LLMMessage, LLMType, ConversationStage, 
    SearchResult, QualityLevel
)
from core.deepseek_client import DeepSeekClient
from core.ollama_client import OllamaClient
from core.serper_client import SerperClient

logger = logging.getLogger(__name__)


class ConversationOrchestrator:
    """Orchestrates iterative research workflow with DeepSeek as context king"""
    
    def __init__(self):
        self.deepseek_client = DeepSeekClient()
        self.ollama_client = OllamaClient()
        self.serper_client = SerperClient()
        self.max_rounds = 50  # Increased for iterative workflow
        self._current_deepseek_summary = ""
        self._current_ollama_summary = ""
        self._current_context = None  # Store current research context
        self.status_callback = None
        
        logger.info("Conversation orchestrator initialized with new iterative workflow")
    
    def set_status_callback(self, callback):
        """Set the status callback function"""
        self.status_callback = callback
    
    def _update_status(self, llm_name, activity, details=""):
        """Internal status update that calls the callback if set"""
        if self.status_callback:
            # Pass the current context if available
            if self._current_context:
                self.status_callback(llm_name, activity, details, self._current_context)
            else:
                self.status_callback(llm_name, activity, details)
    
    def start_research_session(self, user_prompt: str) -> ResearchContext:
        """Start a new research session with iterative workflow"""
        logger.info(f"Starting iterative research session for: {user_prompt}")
        
        context = ResearchContext(user_prompt=user_prompt)
        context.current_stage = ConversationStage.INITIAL_BREAKDOWN
        context.conversation_round = 0
        
        # Initialize workflow metadata
        context.metadata['key_points'] = []
        context.metadata['research_queries'] = []
        context.metadata['approved_documents'] = []
        context.metadata['pending_documents'] = []
        
        self._update_status("System", "Initializing iterative research workflow")
        logger.info("Iterative research workflow initialized")
        
        return context
    
    def execute_conversation_round(self, context: ResearchContext) -> ResearchContext:
        """Execute one step of the iterative workflow"""
        # Store current context for status updates
        self._current_context = context
        
        if context.current_stage == ConversationStage.COMPLETED:
            logger.info("Workflow complete")
            return context
            
        if context.conversation_round >= self.max_rounds:
            logger.warning(f"Maximum rounds reached ({self.max_rounds}), forcing completion")
            context.current_stage = ConversationStage.COMPLETED
            return context

        context.conversation_round += 1
        logger.info(f"Round {context.conversation_round}, Stage: {context.current_stage.value}")
        
        # Route to appropriate stage handler
        stage_handlers = {
            ConversationStage.INITIAL_BREAKDOWN: self._stage1_initial_breakdown,
            ConversationStage.DISCUSS_BREAKDOWN: self._stage2_discuss_breakdown,
            ConversationStage.RESEARCH: self._stage3_research,
            ConversationStage.ANALYZE_RESEARCH: self._stage4_analyze_research,
            ConversationStage.DISCUSS_FINDINGS: self._stage5_discuss_findings,
            ConversationStage.DEEP_DIVE: self._stage6_deep_dive,
            ConversationStage.COMPILE_INFORMATION: self._stage7_compile_information,
            ConversationStage.DISCUSS_COMPILATION: self._stage8_discuss_compilation,
            ConversationStage.GENERATE_DOCUMENTS: self._stage9_generate_documents,
            ConversationStage.REFINE_DOCUMENTS: self._stage10_refine_documents,
        }
        
        handler = stage_handlers.get(context.current_stage)
        if handler:
            context = handler(context)
        else:
            logger.error(f"Unknown stage: {context.current_stage}")
            context.current_stage = ConversationStage.COMPLETED
        
        # Update research summaries after each round
        self._current_deepseek_summary = self._build_research_summary(context, max_tokens=None)
        self._current_ollama_summary = self._build_research_summary(context, max_tokens=8000)
        
        return context
    
    # ==================== STAGE 1: INITIAL BREAKDOWN ====================
    
    def _stage1_initial_breakdown(self, context: ResearchContext) -> ResearchContext:
        """DeepSeek breaks down prompt into key points"""
        self._update_status("DeepSeek", "Stage 1/11: Analyzing prompt in depth", "Identifying key concepts and requirements")
        logger.info("Stage 1: Initial Breakdown - DeepSeek analyzing")
        
        # DeepSeek performs deep analysis
        breakdown_prompt = f"""Analyze this project prompt in comprehensive detail:

"{context.user_prompt}"

Provide a structured breakdown covering:
1. Core Objective - What is the main goal?
2. Key Technologies - What tech stack is involved?
3. Technical Requirements - What are the must-have features?
4. Architecture Needs - What system components are needed?
5. Implementation Challenges - What are the potential difficulties?
6. Research Priorities - What do we need to learn more about?

For each point, provide 2-3 sentences of analysis. Be thorough and identify 8-12 key discussion points."""

        message = self.deepseek_client.generate_response(breakdown_prompt, max_tokens=4000)
        context.add_message(message)
        
        # Store breakdown for later reference
        context.metadata['initial_breakdown'] = message.content
        context.metadata['breakdown_complete'] = True
        
        # Move to discussion
        context.current_stage = ConversationStage.DISCUSS_BREAKDOWN
        context.metadata['breakdown_discussion_round'] = 0
        
        logger.info("Initial breakdown complete - moving to discussion")
        return context
    
    # ==================== STAGE 2: DISCUSS BREAKDOWN ====================
    
    def _stage2_discuss_breakdown(self, context: ResearchContext) -> ResearchContext:
        """LLMs discuss breakdown and identify research topics"""
        round_num = context.metadata.get('breakdown_discussion_round', 0)
        max_rounds = 4
        
        logger.info(f"Stage 2: Round {round_num}/{max_rounds}")
        
        if round_num >= max_rounds:
            logger.info(f"Breakdown discussion complete after {round_num} rounds")
            context.current_stage = ConversationStage.RESEARCH
            return context
        
        context.metadata['breakdown_discussion_round'] = round_num + 1
        logger.info(f"Stage 2: Incremented to round {round_num + 1}")
        
        if round_num == 0:
            # Ollama reviews breakdown
            self._update_status("Ollama", "Stage 2/11: Reviewing breakdown", f"Discussion round {round_num + 1}/{max_rounds}")
            logger.info("Stage 2: Ollama reviewing breakdown")
            
            breakdown = context.metadata.get('initial_breakdown', '')
            review_prompt = f"""Review this project breakdown and provide critical analysis:

{breakdown[:8000]}

Provide:
1. What's missing or unclear?
2. What assumptions need validation?
3. What additional considerations are needed?
4. Suggest 5-7 specific research topics we should investigate."""

            message = self.ollama_client.generate_response(review_prompt)
            context.add_message(message)
            
            # Show full message in details field
            self._update_status("Ollama", "Stage 2/11: Breakdown Review", message.content)
            logger.info(f"Ollama review complete: {len(message.content)} chars")
            
        elif round_num == 1:
            # DeepSeek refines based on feedback
            self._update_status("DeepSeek", "Stage 2/11: Refining breakdown", f"Discussion round {round_num + 1}/{max_rounds}")
            logger.info("Stage 2: DeepSeek refining breakdown")
            
            ollama_feedback = self._get_latest_message_content(context, LLMType.OLLAMA)
            refine_prompt = f"""Based on this feedback, refine the project breakdown:

{ollama_feedback[:8000]}

Provide an updated breakdown addressing the concerns raised and expanding on unclear areas."""

            message = self.deepseek_client.generate_response(refine_prompt, max_tokens=4000)
            context.add_message(message)
            
            # Show full message in details field
            self._update_status("DeepSeek", "Stage 2/11: Refining Breakdown", message.content)
            logger.info(f"DeepSeek refinement complete: {len(message.content)} chars")
            
        elif round_num == 2:
            # Ollama identifies specific research topics
            self._update_status("Ollama", "Stage 2/11: Identifying research topics", f"Discussion round {round_num + 1}/{max_rounds}")
            logger.info("Stage 2: Ollama identifying research topics")
            
            latest_breakdown = self._get_latest_message_content(context, LLMType.DEEPSEEK)
            research_prompt = f"""Based on this refined breakdown:

{latest_breakdown[:8000]}

Create a list of 10-15 specific research queries we should investigate. Format:
1. [Query about technology/approach]
2. [Query about implementation details]
...

Focus on actionable search terms that will yield useful technical information."""

            message = self.ollama_client.generate_response(research_prompt)
            context.add_message(message)
            
            # Show full message in details field
            self._update_status("Ollama", "Stage 2/11: Research Topics", message.content)
            logger.info(f"Ollama research topics complete: {len(message.content)} chars")
            
        else:
            # DeepSeek finalizes research query list
            self._update_status("DeepSeek", "Stage 2/11: Finalizing research plan", f"Discussion round {round_num + 1}/{max_rounds}")
            logger.info("Stage 2: DeepSeek finalizing research queries")
            
            ollama_queries = self._get_latest_message_content(context, LLMType.OLLAMA)
            finalize_prompt = f"""Review these research queries and finalize the list:

{ollama_queries[:8000]}

Provide the final list of 12-18 research queries, formatted as a JSON array of strings. Ensure queries are:
- Specific and searchable
- Cover all key aspects
- Will yield actionable technical information

Return ONLY a JSON array like: ["query 1", "query 2", ...]"""

            message = self.deepseek_client.generate_response(finalize_prompt, max_tokens=2000)
            context.add_message(message)
            
            # Show full message in details field
            self._update_status("DeepSeek", "Stage 2/11: Research Plan", message.content)
            logger.info(f"DeepSeek finalization complete: {len(message.content)} chars")
            
            # Extract queries from response
            try:
                # Try to find JSON array in response
                content = message.content
                start = content.find('[')
                end = content.rfind(']') + 1
                if start != -1 and end > start:
                    queries = json.loads(content[start:end])
                    context.metadata['research_queries'] = queries
                    logger.info(f"Extracted {len(queries)} research queries from JSON")
                else:
                    # Fallback: split by newlines
                    queries = [line.strip() for line in content.split('\n') if line.strip() and not line.strip().startswith('#')]
                    context.metadata['research_queries'] = queries[:18]
                    logger.warning(f"Fallback extraction: {len(queries)} queries")
                
                # Immediately move to research stage after extracting queries
                if len(queries) > 0:
                    logger.info(f"Stage 2 complete - moving to research with {len(queries)} queries")
                    context.current_stage = ConversationStage.RESEARCH
                    return context
                    
            except Exception as e:
                logger.error(f"Failed to extract queries: {e}")
                # Use generic queries as fallback
                context.metadata['research_queries'] = [
                    f"{context.user_prompt} best practices",
                    f"{context.user_prompt} architecture",
                    f"{context.user_prompt} implementation guide"
                ]
                logger.warning("Using fallback queries, moving to research stage")
                context.current_stage = ConversationStage.RESEARCH
        
        return context
    
    # ==================== STAGE 3: RESEARCH ====================
    
    def _stage3_research(self, context: ResearchContext) -> ResearchContext:
        """Perform web searches on identified topics"""
        queries = context.metadata.get('research_queries', [])
        
        if not queries:
            logger.warning("No research queries found, skipping research")
            context.current_stage = ConversationStage.ANALYZE_RESEARCH
            return context
        
        self._update_status("System", f"Stage 3/11: Executing {len(queries)} research queries", "Gathering comprehensive information")
        logger.info(f"Stage 3: Research - Executing {len(queries)} queries")
        
        all_results = []
        for i, query in enumerate(queries, 1):
            self._update_status("System", f"Stage 3/11: Searching {i}/{len(queries)}", query[:60] + "...")
            try:
                search_results = self.serper_client.search(query)
                all_results.extend(search_results)
                logger.info(f"Query '{query[:50]}...' returned {len(search_results)} results")
            except Exception as e:
                logger.error(f"Search failed for '{query}': {e}")
        
        # Add all results to context
        for result in all_results:
            context.add_search_result(result, is_targeted=False)
        
        logger.info(f"Research complete: {len(all_results)} total results")
        context.metadata['initial_research_complete'] = True
        
        # Move to analysis
        context.current_stage = ConversationStage.ANALYZE_RESEARCH
        return context
    
    # ==================== STAGE 4: ANALYZE RESEARCH ====================
    
    def _stage4_analyze_research(self, context: ResearchContext) -> ResearchContext:
        """DeepSeek analyzes ALL research results using full context"""
        self._update_status("DeepSeek", "Stage 4/11: Analyzing all research results", "Processing 100-200 sources with full context")
        logger.info("Stage 4: Analyze Research - DeepSeek processing all results")
        
        # Build comprehensive research data for DeepSeek
        all_results = context.initial_searches + context.targeted_searches
        research_data = []
        
        for i, result in enumerate(all_results[:150], 1):  # Cap at 150 to stay within context
            research_data.append(f"""
### Source {i}: {result.title}
URL: {result.url}
Content: {result.snippet}
""")
        
        research_text = "\n".join(research_data)
        
        analysis_prompt = f"""Analyze ALL of these research results comprehensively:

{research_text}

Project Context: {context.user_prompt}

Provide a thorough analysis covering:

1. **Key Insights** (15-20 points)
   - Extract the most important technical findings
   - Identify patterns and common themes
   - Note any conflicting information

2. **Technology Analysis**
   - Recommended approaches and technologies
   - Best practices discovered
   - Common pitfalls to avoid

3. **Architecture Patterns**
   - System design patterns found
   - Component relationships
   - Scalability considerations

4. **Implementation Details**
   - Specific technical requirements
   - Code patterns and examples mentioned
   - Integration approaches

5. **Knowledge Gaps**
   - What information is still unclear?
   - What needs deeper investigation?
   - What conflicting information needs resolution?

Be comprehensive - you have access to all research results. Use specific examples and citations."""

        message = self.deepseek_client.generate_response(analysis_prompt, max_tokens=8000)
        context.add_message(message)
        
        # Extract insights from DeepSeek's analysis
        insights = self._extract_insights_from_analysis(message.content)
        context.key_insights.extend(insights)
        
        logger.info(f"Research analysis complete: {len(insights)} insights extracted")
        context.metadata['research_analysis'] = message.content
        
        # Move to findings discussion
        context.current_stage = ConversationStage.DISCUSS_FINDINGS
        context.metadata['findings_discussion_round'] = 0
        
        return context
    
    def _extract_insights_from_analysis(self, analysis: str) -> List[Dict[str, Any]]:
        """Extract structured insights from DeepSeek's analysis"""
        insights = []
        
        # Simple extraction: look for bullet points or numbered items
        lines = analysis.split('\n')
        for line in lines:
            line = line.strip()
            # Look for lines starting with -, *, numbers, or bullet points
            if line and (line[0] in '-*•' or (len(line) > 2 and line[0].isdigit() and line[1] in '.):')):
                # Clean up the line
                clean_line = line.lstrip('-*•0123456789.): ').strip()
                if len(clean_line) > 20:  # Meaningful insights only
                    insights.append({
                        'content': clean_line,
                        'source': 'deepseek_analysis',
                        'timestamp': datetime.now().isoformat()
                    })
        
        return insights[:20]  # Top 20 insights
    
    # ==================== STAGE 5: DISCUSS FINDINGS ====================
    
    def _stage5_discuss_findings(self, context: ResearchContext) -> ResearchContext:
        """LLMs discuss research findings"""
        round_num = context.metadata.get('findings_discussion_round', 0)
        max_rounds = 5
        
        if round_num >= max_rounds:
            logger.info(f"Findings discussion complete after {round_num} rounds")
            context.current_stage = ConversationStage.DEEP_DIVE
            context.metadata['deep_dive_round'] = 0
            return context
        
        context.metadata['findings_discussion_round'] = round_num + 1
        
        if round_num == 0:
            # DeepSeek presents findings
            self._update_status("DeepSeek", "Stage 5/11: Presenting key findings", f"Discussion round {round_num + 1}/{max_rounds}")
            logger.info("Stage 5: DeepSeek presenting findings")
            
            analysis = context.metadata.get('research_analysis', '')
            present_prompt = f"""Based on the comprehensive research analysis, present the TOP 10 key findings that are most critical for this project:

{context.user_prompt}

Focus on actionable insights, technical recommendations, and important considerations."""

            message = self.deepseek_client.generate_response(present_prompt, max_tokens=3000)
            context.add_message(message)
            self._update_status("DeepSeek", "Stage 5/11: Presenting Findings", message.content)
            
        elif round_num % 2 == 1:
            # Ollama critiques and questions
            self._update_status("Ollama", "Stage 5/11: Critiquing findings", f"Discussion round {round_num + 1}/{max_rounds}")
            logger.info("Stage 5: Ollama critiquing")
            
            latest = self._get_latest_message_content(context, LLMType.DEEPSEEK)
            critique_prompt = f"""Review these findings and provide critical analysis:

{latest[:8000]}

Identify:
1. What needs clarification?
2. What seems inconsistent?
3. What additional information is needed?
4. What should we investigate deeper?"""

            message = self.ollama_client.generate_response(critique_prompt)
            context.add_message(message)
            self._update_status("Ollama", "Stage 5/11: Critiquing Findings", message.content)
            
        else:
            # DeepSeek elaborates
            self._update_status("DeepSeek", "Stage 5/11: Elaborating on findings", f"Discussion round {round_num + 1}/{max_rounds}")
            logger.info("Stage 5: DeepSeek elaborating")
            
            critique = self._get_latest_message_content(context, LLMType.OLLAMA)
            elaborate_prompt = f"""Address these questions and concerns:

{critique[:8000]}

Provide detailed clarification using the comprehensive research data available."""

            message = self.deepseek_client.generate_response(elaborate_prompt, max_tokens=3000)
            context.add_message(message)
            self._update_status("DeepSeek", "Stage 5/11: Elaborating", message.content)
        
        return context
    
    # ==================== STAGE 6: DEEP DIVE ====================
    
    def _stage6_deep_dive(self, context: ResearchContext) -> ResearchContext:
        """LLMs do deep dive with dynamic search triggers"""
        round_num = context.metadata.get('deep_dive_round', 0)
        max_rounds = 7
        
        if round_num >= max_rounds:
            logger.info(f"Deep dive complete after {round_num} rounds")
            context.current_stage = ConversationStage.COMPILE_INFORMATION
            return context
        
        context.metadata['deep_dive_round'] = round_num + 1
        
        # Alternate between LLMs, check for search needs
        if round_num % 2 == 0:
            # DeepSeek deep dive
            self._update_status("DeepSeek", "Stage 6/11: Deep dive analysis", f"Round {round_num + 1}/{max_rounds}")
            logger.info("Stage 6: DeepSeek deep dive")
            
            dive_prompt = f"""Perform a deep analysis of the key points for this project:

{context.user_prompt}

Identify any remaining gaps or questions. If you need more information, state: "SEARCH_NEEDED: [query]"

Focus on technical depth and implementation details."""

            message = self.deepseek_client.generate_response(dive_prompt, max_tokens=3000)
            context.add_message(message)
            self._update_status("DeepSeek", "Stage 6/11: Deep Dive Analysis", message.content)
            
            # Check for search triggers
            if "SEARCH_NEEDED:" in message.content:
                self._handle_dynamic_search(context, message.content)
                
        else:
            # Ollama review
            self._update_status("Ollama", "Stage 6/11: Deep dive review", f"Round {round_num + 1}/{max_rounds}")
            logger.info("Stage 6: Ollama review")
            
            latest = self._get_latest_message_content(context, LLMType.DEEPSEEK)
            review_prompt = f"""Review this deep analysis:

{latest[:8000]}

Validate the technical approach and identify any missing pieces. If more research is needed, state: "SEARCH_NEEDED: [query]"."""

            message = self.ollama_client.generate_response(review_prompt)
            context.add_message(message)
            self._update_status("Ollama", "Stage 6/11: Deep Dive Review", message.content)
            
            # Check for search triggers
            if "SEARCH_NEEDED:" in message.content:
                self._handle_dynamic_search(context, message.content)
        
        return context
    
    def _handle_dynamic_search(self, context: ResearchContext, content: str):
        """Handle dynamic search requests from LLMs"""
        import re
        matches = re.findall(r'SEARCH_NEEDED:\s*(.+?)(?:\n|$)', content)
        
        for query in matches[:3]:  # Max 3 per trigger
            query = query.strip()
            logger.info(f"Dynamic search triggered: {query}")
            self._update_status("System", "Performing additional search", query[:60])
            
            try:
                results = self.serper_client.search(query)
                for result in results:
                    context.add_search_result(result, is_targeted=True)
                logger.info(f"Dynamic search found {len(results)} results")
            except Exception as e:
                logger.error(f"Dynamic search failed: {e}")
    
    # ==================== STAGE 7: COMPILE INFORMATION ====================
    
    def _stage7_compile_information(self, context: ResearchContext) -> ResearchContext:
        """DeepSeek compiles all information using maximum context"""
        self._update_status("DeepSeek", "Stage 7/11: Compiling all information", "Using full context (80-120K tokens)")
        logger.info("Stage 7: DeepSeek compiling information")
        
        # Build MASSIVE context summary for DeepSeek
        full_summary = self._build_research_summary(context, max_tokens=None)  # No limits!
        
        compile_prompt = f"""You have access to ALL research data and discussions. Compile a comprehensive master document covering:

PROJECT: {context.user_prompt}

RESEARCH SUMMARY:
{full_summary}

Create a structured compilation with:
1. Executive Summary
2. Technical Architecture Overview  
3. Key Technologies & Approaches
4. Implementation Strategy
5. Critical Considerations
6. Recommended Document Structure (list 1-7 specialized documents needed)

Be comprehensive - use all available context."""

        message = self.deepseek_client.generate_response(compile_prompt, max_tokens=8000)
        context.add_message(message)
        
        context.metadata['master_compilation'] = message.content
        
        # Move to compilation discussion
        context.current_stage = ConversationStage.DISCUSS_COMPILATION
        context.metadata['compilation_discussion_round'] = 0
        
        logger.info("Compilation complete")
        return context
    
    # ==================== STAGE 8: DISCUSS COMPILATION ====================
    
    def _stage8_discuss_compilation(self, context: ResearchContext) -> ResearchContext:
        """LLMs discuss compilation, can trigger more searches"""
        round_num = context.metadata.get('compilation_discussion_round', 0)
        max_rounds = 4
        
        if round_num >= max_rounds:
            logger.info(f"Compilation discussion complete after {round_num} rounds")
            context.current_stage = ConversationStage.GENERATE_DOCUMENTS
            return context
        
        context.metadata['compilation_discussion_round'] = round_num + 1
        
        if round_num == 0:
            # Ollama reviews compilation
            self._update_status("Ollama", "Stage 8/11: Reviewing compilation", f"Round {round_num + 1}/{max_rounds}")
            logger.info("Stage 8: Ollama reviewing compilation")
            
            compilation = context.metadata.get('master_compilation', '')
            review_prompt = f"""Review this master compilation for completeness:

{compilation[:8000]}

Identify:
1. Any gaps in information
2. Areas needing more detail
3. Missing considerations
4. If more research is needed, state: "SEARCH_NEEDED: [query]"."""

            message = self.ollama_client.generate_response(review_prompt)
            context.add_message(message)
            self._update_status("Ollama", "Stage 8/11: Reviewing Compilation", message.content)
            
            # Handle dynamic search
            if "SEARCH_NEEDED:" in message.content:
                self._handle_dynamic_search(context, message.content)
                
        else:
            # Continue discussion
            if round_num % 2 == 1:
                # DeepSeek responds
                self._update_status("DeepSeek", "Stage 8/11: Updating compilation", f"Round {round_num + 1}/{max_rounds}")
                logger.info("Stage 8: DeepSeek updating")
                
                feedback = self._get_latest_message_content(context, LLMType.OLLAMA)
                update_prompt = f"""Address this feedback and update the compilation:

{feedback[:8000]}

Provide an updated version incorporating the new information."""

                message = self.deepseek_client.generate_response(update_prompt, max_tokens=6000)
                context.add_message(message)
                self._update_status("DeepSeek", "Stage 8/11: Updating Compilation", message.content)
                
            else:
                # Ollama confirms
                self._update_status("Ollama", "Stage 8/11: Confirming compilation", f"Round {round_num + 1}/{max_rounds}")
                logger.info("Stage 8: Ollama confirming")
                
                updated = self._get_latest_message_content(context, LLMType.DEEPSEEK)
                confirm_prompt = f"""Review the updated compilation:

{updated[:8000]}

Is this comprehensive and complete? Approve or identify remaining gaps."""

                message = self.ollama_client.generate_response(confirm_prompt)
                context.add_message(message)
                self._update_status("Ollama", "Stage 8/11: Confirming Compilation", message.content)
        
        return context
    
    # ==================== STAGE 9: GENERATE DOCUMENTS ====================
    
    def _stage9_generate_documents(self, context: ResearchContext) -> ResearchContext:
        """DeepSeek generates 1-7 specialized documents"""
        self._update_status("DeepSeek", "Stage 9/11: Generating specialized documents", "Creating 1-7 documents")
        logger.info("Stage 9: Generating documents")
        
        # Use DeepSeek's multi-document generation
        full_summary = self._build_research_summary(context, max_tokens=None)
        conversation_summary = self._build_conversation_summary(context)
        
        # Generate documents
        plan_message = self.deepseek_client.generate_final_plan(
            context.user_prompt,
            full_summary,
            conversation_summary
        )
        
        context.add_message(plan_message)
        
        # Extract documents
        if plan_message.metadata and plan_message.metadata.get('multi_document'):
            documents = plan_message.metadata.get('documents', [])
            context.metadata['pending_documents'] = documents
            logger.info(f"Generated {len(documents)} documents")
        else:
            # Single document
            context.metadata['pending_documents'] = [{
                'title': 'Development Plan',
                'content': plan_message.content
            }]
        
        # Move to refinement
        context.current_stage = ConversationStage.REFINE_DOCUMENTS
        context.metadata['current_doc_index'] = 0
        context.metadata['doc_refinement_round'] = 0
        
        return context
    
    # ==================== STAGE 10: REFINE DOCUMENTS ====================
    
    def _stage10_refine_documents(self, context: ResearchContext) -> ResearchContext:
        """LLMs refine documents iteratively until perfect"""
        pending = context.metadata.get('pending_documents', [])
        approved = context.metadata.get('approved_documents', [])
        current_index = context.metadata.get('current_doc_index', 0)
        round_num = context.metadata.get('doc_refinement_round', 0)
        max_rounds_per_doc = 6
        
        # Check if all documents approved
        if current_index >= len(pending):
            logger.info(f"All {len(approved)} documents approved")
            context.current_stage = ConversationStage.COMPLETED
            return context
        
        current_doc = pending[current_index]
        doc_title = current_doc.get('title', f'Document {current_index + 1}')
        
        # Check if we've refined this doc enough
        if round_num >= max_rounds_per_doc:
            # Force approval and move to next
            logger.info(f"Max rounds reached for '{doc_title}', moving to next")
            approved.append(current_doc)
            context.metadata['approved_documents'] = approved
            context.metadata['current_doc_index'] = current_index + 1
            context.metadata['doc_refinement_round'] = 0
            return context
        
        context.metadata['doc_refinement_round'] = round_num + 1
        
        if round_num == 0:
            # Ollama reviews document
            self._update_status("Ollama", f"Stage 10/11: Reviewing '{doc_title}'", f"Doc {current_index + 1}/{len(pending)}, Round {round_num + 1}/{max_rounds_per_doc}")
            logger.info(f"Stage 10: Ollama reviewing '{doc_title}'")
            
            doc_content = current_doc.get('content', '')
            review_prompt = f"""Review this document for quality and completeness:

DOCUMENT: {doc_title}

{doc_content[:8000]}

Provide:
1. Overall quality assessment
2. Specific improvements needed
3. Missing sections or details
4. If document is ready, state: "APPROVED"
Otherwise, provide detailed feedback."""

            message = self.ollama_client.generate_response(review_prompt)
            context.add_message(message)
            self._update_status("Ollama", f"Stage 10/11: Reviewing '{doc_title}'", message.content)
            
            # Check for approval
            if "APPROVED" in message.content.upper():
                logger.info(f"Document '{doc_title}' approved by Ollama")
                approved.append(current_doc)
                context.metadata['approved_documents'] = approved
                context.metadata['current_doc_index'] = current_index + 1
                context.metadata['doc_refinement_round'] = 0
                
        else:
            # Alternate refinement
            if round_num % 2 == 1:
                # DeepSeek revises
                self._update_status("DeepSeek", f"Stage 10/11: Revising '{doc_title}'", f"Doc {current_index + 1}/{len(pending)}, Round {round_num + 1}/{max_rounds_per_doc}")
                logger.info(f"Stage 10: DeepSeek revising '{doc_title}'")
                
                feedback = self._get_latest_message_content(context, LLMType.OLLAMA)
                revise_prompt = f"""Revise this document based on feedback:

DOCUMENT: {doc_title}
ORIGINAL:
{current_doc.get('content', '')[:6000]}

FEEDBACK:
{feedback[:2000]}

Provide the complete revised document."""

                message = self.deepseek_client.generate_response(revise_prompt, max_tokens=8000)
                context.add_message(message)
                self._update_status("DeepSeek", f"Stage 10/11: Revising '{doc_title}'", message.content)
                
                # Update document
                current_doc['content'] = message.content
                pending[current_index] = current_doc
                context.metadata['pending_documents'] = pending
                
            else:
                # Ollama re-reviews
                self._update_status("Ollama", f"Stage 10/11: Re-reviewing '{doc_title}'", f"Doc {current_index + 1}/{len(pending)}, Round {round_num + 1}/{max_rounds_per_doc}")
                logger.info(f"Stage 10: Ollama re-reviewing '{doc_title}'")
                
                revised = self._get_latest_message_content(context, LLMType.DEEPSEEK)
                review_prompt = f"""Review the revised document:

{revised[:8000]}

Is this now ready? State "APPROVED" if yes, or provide additional feedback."""

                message = self.ollama_client.generate_response(review_prompt)
                context.add_message(message)
                self._update_status("Ollama", f"Stage 10/11: Re-reviewing '{doc_title}'", message.content)
                
                # Check for approval
                if "APPROVED" in message.content.upper():
                    logger.info(f"Document '{doc_title}' approved after revision")
                    approved.append(current_doc)
                    context.metadata['approved_documents'] = approved
                    context.metadata['current_doc_index'] = current_index + 1
                    context.metadata['doc_refinement_round'] = 0
        
        return context
    
    # ==================== Helper Methods ====================
    
    def _get_latest_message_content(self, context: ResearchContext, llm_type: LLMType) -> str:
        """Get the latest message from a specific LLM"""
        for message in reversed(context.messages):
            if message.llm_type == llm_type:
                return message.content
        return ""
    
    def _build_research_summary(self, context: ResearchContext, max_tokens: Optional[int] = None) -> str:
        """Build research summary for LLM context"""
        summary_parts = []
        
        # Overview
        total_searches = len(context.initial_searches) + len(context.targeted_searches)
        summary_parts.append(f"PROJECT: {context.user_prompt}")
        summary_parts.append(f"RESEARCH: {total_searches} searches, {len(context.key_insights)} insights, {len(context.messages)} messages")
        summary_parts.append("")
        
        # Key insights
        if context.key_insights:
            summary_parts.append("KEY INSIGHTS:")
            max_insights = 30 if max_tokens is None else 10
            for insight in context.key_insights[:max_insights]:
                summary_parts.append(f"• {insight['content'][:200]}")
            summary_parts.append("")
        
        # Recent conversation
        if context.messages:
            summary_parts.append("RECENT DISCUSSION:")
            recent = context.messages[-5:] if max_tokens else context.messages[-10:]
            for msg in recent:
                preview = msg.content[:300] if max_tokens else msg.content[:500]
                summary_parts.append(f"[{msg.llm_type.value}]: {preview}...")
        
        full_summary = "\n".join(summary_parts)
        
        # Truncate if needed
        if max_tokens:
            # Rough estimate: 4 chars per token
            max_chars = max_tokens * 4
            if len(full_summary) > max_chars:
                full_summary = full_summary[:max_chars] + "\n..."
        
        return full_summary
    
    def generate_development_plan(self, context: ResearchContext) -> Dict[str, Any]:
        """Generate final documents (called after workflow completes)"""
        logger.info("Generating final development plan...")
        
        # Get approved documents from context
        approved_docs = context.metadata.get('approved_documents', [])
        
        if not approved_docs:
            logger.warning("No approved documents found, generating default")
            # Fallback to old behavior
            return self._generate_default_plan(context)
        
        # Build document package
        research_summary = self._build_research_summary(context)
        conversation_summary = self._build_conversation_summary(context)
        
        development_plan = {
            'project_name': self._extract_project_name(context.user_prompt),
            'user_prompt': context.user_prompt,
            'generated_at': datetime.now().isoformat(),
            'session_id': context.session_id,
            'multi_document': True,
            'documents': approved_docs,
            'research_metrics': {
                'total_searches': len(context.initial_searches) + len(context.targeted_searches),
                'key_insights': len(context.key_insights),
                'conversation_rounds': context.conversation_round,
                'approved_documents': len(approved_docs)
            },
            'conversation_summary': conversation_summary
        }
        
        logger.info(f"Development plan ready with {len(approved_docs)} documents")
        return development_plan
    
    def _generate_default_plan(self, context: ResearchContext) -> Dict[str, Any]:
        """Fallback plan generation"""
        research_summary = self._build_research_summary(context)
        
        plan_message = self.deepseek_client.generate_final_plan(
            context.user_prompt,
            research_summary,
            ""
        )
        
        context.add_message(plan_message)
        
        return {
            'project_name': self._extract_project_name(context.user_prompt),
            'user_prompt': context.user_prompt,
            'generated_at': datetime.now().isoformat(),
            'session_id': context.session_id,
            'development_plan': plan_message.content,
            'multi_document': False
        }
    
    def _build_conversation_summary(self, context: ResearchContext) -> str:
        """Build conversation summary"""
        if not context.messages:
            return "No discussion recorded."
        
        summary_parts = ["CONVERSATION SUMMARY:"]
        summary_parts.append(f"Total rounds: {context.conversation_round}")
        summary_parts.append(f"Messages: {len(context.messages)}")
        summary_parts.append("")
        
        # Key decisions from each stage
        for stage in ConversationStage:
            stage_messages = [m for m in context.messages if stage.value in m.content.lower()[:100]]
            if stage_messages:
                summary_parts.append(f"{stage.value}: {len(stage_messages)} messages")
        
        return "\n".join(summary_parts)
    
    def _extract_project_name(self, prompt: str) -> str:
        """Extract project name from prompt"""
        # Simple extraction: first 3-5 words
        words = prompt.split()[:5]
        name = "_".join(words)
        # Clean up
        name = "".join(c for c in name if c.isalnum() or c == "_")
        return name[:50]
