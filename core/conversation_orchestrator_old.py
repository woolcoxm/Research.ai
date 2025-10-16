import logging
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
import concurrent.futures
from threading import Lock

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
        # Cache for expensive operations
        self._summary_cache = {}
        self._cache_round = 0
        
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
    
    def _extract_json_safely(self, content: str, expected_type: str = "array") -> Optional[Any]:
        """
        Safely extract JSON from LLM response with multiple fallback strategies
        
        Args:
            content: The LLM response content
            expected_type: "array" for lists or "object" for dicts
        
        Returns:
            Parsed JSON or None if all strategies fail
        """
        import re
        
        # Strategy 1: Look for JSON code blocks (```json ... ```)
        code_block_match = re.search(r'```json\s*(\{.*?\}|\[.*?\])\s*```', content, re.DOTALL)
        if code_block_match:
            try:
                parsed = json.loads(code_block_match.group(1))
                logger.info(f"JSON extracted from code block successfully")
                return parsed
            except json.JSONDecodeError as e:
                logger.warning(f"JSON code block invalid: {e}")
        
        # Strategy 2: Find JSON by bracket matching with proper nesting
        if expected_type == "array":
            bracket_start, bracket_end = '[', ']'
        else:
            bracket_start, bracket_end = '{', '}'
        
        start_idx = content.find(bracket_start)
        if start_idx == -1:
            logger.warning(f"No JSON start bracket '{bracket_start}' found")
            return None
        
        # Count brackets to find matching end (handles nesting)
        depth = 0
        for i in range(start_idx, len(content)):
            char = content[i]
            if char == bracket_start:
                depth += 1
            elif char == bracket_end:
                depth -= 1
                if depth == 0:
                    # Found matching closing bracket
                    json_str = content[start_idx:i+1]
                    try:
                        parsed = json.loads(json_str)
                        logger.info(f"JSON extracted via bracket matching successfully")
                        return parsed
                    except json.JSONDecodeError as e:
                        logger.warning(f"Bracket-matched JSON invalid: {e}")
                        # Try to clean common issues
                        json_str_cleaned = re.sub(r',\s*([}\]])', r'\1', json_str)  # Remove trailing commas
                        json_str_cleaned = json_str_cleaned.replace("'", '"')  # Single to double quotes
                        try:
                            parsed = json.loads(json_str_cleaned)
                            logger.info(f"JSON extracted after cleaning successfully")
                            return parsed
                        except json.JSONDecodeError:
                            logger.error(f"Even cleaned JSON failed. First 200 chars: {json_str[:200]}")
                            return None
        
        logger.warning("No matching closing bracket found for JSON")
        return None
    
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
            ConversationStage.WRITE_DOCUMENTS: self._stage_write_documents,
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
        max_rounds = 2  # REDUCED from 4 to 2 for efficiency
        
        logger.info(f"Stage 2: Round {round_num}/{max_rounds}")
        
        # Don't exit early - we need to extract queries from the final round
        context.metadata['breakdown_discussion_round'] = round_num + 1
        logger.info(f"Stage 2: Incremented to round {round_num + 1}")
        
        if round_num == 0:
            # Ollama reviews breakdown and suggests research topics
            self._update_status("Ollama", "Stage 2/11: Reviewing & suggesting research topics", f"Discussion round {round_num + 1}/{max_rounds}")
            logger.info("Stage 2: Ollama reviewing breakdown")
            
            breakdown = context.metadata.get('initial_breakdown', '')
            review_prompt = f"""Review this project breakdown and provide:

{breakdown[:8000]}

1. Critical analysis: What's missing or unclear?
2. Key assumptions that need validation
3. Important considerations
4. **LIST OF 10-15 SPECIFIC RESEARCH QUERIES** to investigate (as numbered list)

Format your research queries clearly so they can be extracted."""

            message = self.ollama_client.generate_response(review_prompt)
            context.add_message(message)
            
            # Show full message in details field
            self._update_status("Ollama", "Stage 2/11: Breakdown Review", message.content)
            logger.info(f"Ollama review complete: {len(message.content)} chars")
            
        else:
            # Round 1: DeepSeek finalizes research query list
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
            
            # Extract queries from response using robust JSON extraction
            try:
                queries = self._extract_json_safely(message.content, expected_type="array")
                
                if queries and isinstance(queries, list) and len(queries) > 0:
                    context.metadata['research_queries'] = queries
                    logger.info(f"Extracted {len(queries)} research queries via robust JSON extraction")
                else:
                    # Fallback: split by newlines
                    logger.warning("JSON extraction returned invalid data, using fallback line-by-line extraction")
                    queries = [line.strip() for line in message.content.split('\n') 
                              if line.strip() and not line.strip().startswith('#') and len(line.strip()) > 10]
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
    
    def _execute_single_search(self, query: str, index: int, total: int) -> tuple:
        """Execute a single search query (for parallel execution)"""
        self._update_status("System", f"Stage 3/11: Searching {index}/{total}", query[:60] + "...")
        try:
            search_results = self.serper_client.search(query)
            logger.info(f"Query '{query[:50]}...' returned {len(search_results)} results")
            return (True, search_results)
        except Exception as e:
            logger.error(f"Search failed for '{query}': {e}")
            return (False, [])
    
    def _stage3_research(self, context: ResearchContext) -> ResearchContext:
        """Perform web searches on identified topics (PARALLELIZED)"""
        queries = context.metadata.get('research_queries', [])
        
        if not queries:
            logger.warning("No research queries found, skipping research")
            context.current_stage = ConversationStage.ANALYZE_RESEARCH
            return context
        
        self._update_status("System", f"Stage 3/11: Executing {len(queries)} research queries", "Gathering comprehensive information (parallel)")
        logger.info(f"Stage 3: Research - Executing {len(queries)} queries in parallel")
        
        # Execute searches in parallel (max 10 concurrent searches - increased for efficiency)
        all_results = []
        max_workers = min(10, len(queries))  # INCREASED from 5 to 10
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all search tasks
            future_to_query = {
                executor.submit(self._execute_single_search, query, i, len(queries)): query 
                for i, query in enumerate(queries, 1)
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_query):
                query = future_to_query[future]
                try:
                    success, search_results = future.result()
                    if success:
                        all_results.extend(search_results)
                except Exception as e:
                    logger.error(f"Search task failed for '{query}': {e}")
        
        # Add all results to context
        for result in all_results:
            context.add_search_result(result, is_targeted=False)
        
        logger.info(f"Research complete: {len(all_results)} total results from {len(queries)} parallel searches")
        context.metadata['initial_research_complete'] = True
        
        # Move to analysis
        context.current_stage = ConversationStage.ANALYZE_RESEARCH
        return context
    
    # ==================== STAGE 4: ANALYZE RESEARCH ====================
    
    def _analyze_research_chunk(self, chunk_data: str, chunk_num: int, total_chunks: int, user_prompt: str) -> str:
        """Analyze a chunk of research results (for parallel execution)"""
        self._update_status(
            "DeepSeek", 
            f"Stage 4/11: Analyzing chunk {chunk_num}/{total_chunks}",
            f"Processing research sources in parallel..."
        )
        
        analysis_prompt = f"""Analyze these research results for the project:

{chunk_data}

Project Context: {user_prompt}

Extract:
1. Key technical insights and findings
2. Recommended technologies and approaches
3. Best practices discovered
4. Common pitfalls to avoid
5. Important patterns or themes

Be concise but thorough. Focus on actionable insights."""

        try:
            message = self.deepseek_client.generate_response(analysis_prompt, max_tokens=4000)
            logger.info(f"Chunk {chunk_num}/{total_chunks} analyzed successfully")
            return message.content
        except Exception as e:
            logger.error(f"Analysis failed for chunk {chunk_num}: {e}")
            return f"Error analyzing chunk {chunk_num}: {str(e)}"
    
    def _stage4_analyze_research(self, context: ResearchContext) -> ResearchContext:
        """DeepSeek analyzes research results using PARALLEL chunked processing"""
        logger.info("Stage 4: Analyze Research - DeepSeek processing with parallel chunks")
        
        # Build comprehensive research data
        all_results = context.initial_searches + context.targeted_searches
        total_sources = min(len(all_results), 150)
        
        # Build research data FAST - use list comprehension instead of loop
        logger.info(f"Building context from {total_sources} sources...")
        research_data = [
            f"### Source {i}: {result.title}\nURL: {result.link}\nContent: {result.snippet}\n"
            for i, result in enumerate(all_results[:150], 1)
        ]
        
        # Split into chunks for parallel processing (30 sources per chunk)
        chunk_size = 30
        chunks = [
            "\n".join(research_data[i:i + chunk_size])
            for i in range(0, len(research_data), chunk_size)
        ]
        
        total_chunks = len(chunks)
        logger.info(f"Split {total_sources} sources into {total_chunks} chunks for parallel analysis")
        
        self._update_status(
            "DeepSeek", 
            f"Stage 4/11: Analyzing {total_sources} sources in {total_chunks} parallel chunks",
            "Processing multiple chunks simultaneously for faster analysis..."
        )
        
        # Process chunks in parallel (max 5 concurrent API calls - increased for efficiency)
        chunk_analyses = []
        max_workers = min(5, total_chunks)  # INCREASED from 3 to 5
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all analysis tasks
            future_to_chunk = {
                executor.submit(self._analyze_research_chunk, chunk, i, total_chunks, context.user_prompt): i 
                for i, chunk in enumerate(chunks, 1)
            }
            
            # Collect results as they complete
            results_dict = {}
            for future in concurrent.futures.as_completed(future_to_chunk):
                chunk_num = future_to_chunk[future]
                try:
                    analysis = future.result()
                    results_dict[chunk_num] = analysis
                    logger.info(f"Collected analysis for chunk {chunk_num}/{total_chunks}")
                except Exception as e:
                    logger.error(f"Failed to get result for chunk {chunk_num}: {e}")
                    results_dict[chunk_num] = f"Error in chunk {chunk_num}"
            
            # Combine results in order
            for i in range(1, total_chunks + 1):
                if i in results_dict:
                    chunk_analyses.append(results_dict[i])
        
        # Synthesize all chunk analyses into final analysis
        self._update_status(
            "DeepSeek", 
            f"Stage 4/11: Synthesizing {total_chunks} chunk analyses",
            "Combining insights from all parallel analyses..."
        )
        
        combined_analysis = "\n\n---\n\n".join(chunk_analyses)
        
        synthesis_prompt = f"""Synthesize these {total_chunks} parallel analyses into one comprehensive analysis:

{combined_analysis[:50000]}

Project Context: {context.user_prompt}

Create a unified analysis covering:
1. **Key Insights** (15-20 most important points)
2. **Technology Analysis** (recommended approaches and best practices)
3. **Architecture Patterns** (system design patterns found)
4. **Implementation Details** (technical requirements and code patterns)
5. **Knowledge Gaps** (what needs deeper investigation)

Remove duplicates and synthesize common themes."""

        final_message = self.deepseek_client.generate_response(synthesis_prompt, max_tokens=8000)
        context.add_message(final_message)
        
        # Update status to show analysis is complete
        self._update_status(
            "DeepSeek", 
            f"Stage 4/11: Analysis complete",
            f"Analyzed {total_sources} sources using {total_chunks} parallel chunks"
        )
        
        # Extract insights from final synthesis
        insights = self._extract_insights_from_analysis(final_message.content)
        context.key_insights.extend(insights)
        
        logger.info(f"Research analysis complete: {len(insights)} insights extracted from {total_sources} sources using parallel processing")
        context.metadata['research_analysis'] = final_message.content
        
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
        max_rounds = 2  # REDUCED from 5 to 2 for efficiency
        
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
        max_rounds = 3  # REDUCED from 7 to 3 for efficiency
        
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
            
            # Get comprehensive research analysis from Stage 4
            research_analysis = context.metadata.get('research_analysis', '')
            # Get recent conversation context (last 10 messages)
            recent_messages = context.messages[-10:] if len(context.messages) > 10 else context.messages
            conversation_summary = "\n\n".join([f"{msg.llm_type.value}: {msg.content[:500]}..." for msg in recent_messages])
            
            dive_prompt = f"""Perform a deep technical analysis based on our research findings.

PROJECT: {context.user_prompt}

RESEARCH ANALYSIS:
{research_analysis[:15000]}

RECENT DISCUSSION:
{conversation_summary[:5000]}

Provide a deep dive focusing on:
1. Technical implementation details
2. Architecture decisions and trade-offs
3. Potential challenges and solutions
4. Best practices and patterns
5. Any gaps requiring more research

If you need additional information on any topic, state: "SEARCH_NEEDED: [specific query]"

Be thorough and technical."""

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
            research_analysis = context.metadata.get('research_analysis', '')
            
            review_prompt = f"""Review and critique this deep technical analysis:

LATEST ANALYSIS:
{latest[:10000]}

RESEARCH CONTEXT:
{research_analysis[:8000]}

Provide critical feedback:
1. What technical details are missing or unclear?
2. Are there better approaches or alternatives?
3. What risks or challenges weren't addressed?
4. What specific implementation details need clarification?
5. Should we investigate any topics deeper?

If more research is needed on specific topics, state: "SEARCH_NEEDED: [specific query]"

Be thorough and challenge assumptions."""

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
        max_rounds = 2  # REDUCED from 4 to 2 for efficiency
        
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
        """OPTIMIZED: DeepSeek creates outlines, Ollama writes documents"""
        self._update_status("DeepSeek", "Stage 9/11: Creating document outlines", "Analyzing research to create detailed outlines")
        logger.info("Stage 9: Creating document outlines (optimized workflow)")
        
        # Get full research context (DeepSeek has 128K context)
        full_summary = self._build_research_summary(context, max_tokens=None)
        conversation_summary = self._build_conversation_summary(context)
        
        # Determine document types to generate
        doc_types = self._determine_document_types(context, full_summary)
        logger.info(f"Generating {len(doc_types)} document types: {doc_types}")
        
        # Create outlines and research summaries for each document
        document_plans = []
        
        for idx, doc_type in enumerate(doc_types):
            # DeepSeek creates detailed outline (within 8K limit)
            self._update_status("DeepSeek", f"Stage 9/11: Creating outline {idx+1}/{len(doc_types)}", f"Document: {doc_type}")
            logger.info(f"Creating outline for document {idx+1}: {doc_type}")
            
            outline_message = self.deepseek_client.create_document_outline(
                doc_number=idx+1,
                doc_type=doc_type,
                user_prompt=context.user_prompt,
                research_context=full_summary,
                conversation_summary=conversation_summary
            )
            context.add_message(outline_message)
            
            # DeepSeek creates targeted research summary (within 8K limit)
            self._update_status("DeepSeek", f"Stage 9/11: Summarizing research {idx+1}/{len(doc_types)}", f"For: {doc_type}")
            logger.info(f"Creating research summary for document {idx+1}: {doc_type}")
            
            research_summary_message = self.deepseek_client.create_research_summary(
                doc_type=doc_type,
                research_context=full_summary,
                user_prompt=context.user_prompt
            )
            context.add_message(research_summary_message)
            
            document_plans.append({
                'doc_number': idx+1,
                'doc_type': doc_type,
                'outline': outline_message.content,
                'research_summary': research_summary_message.content,
                'title': doc_type
            })
        
        # Store document plans for Ollama to write
        context.metadata['document_plans'] = document_plans
        context.metadata['pending_documents'] = []  # Will be populated as Ollama writes
        
        # Move to document writing stage
        context.current_stage = ConversationStage.WRITE_DOCUMENTS
        context.metadata['current_doc_index'] = 0
        
        return context
    
    def _determine_document_types(self, context: ResearchContext, research_summary: str) -> List[str]:
        """Determine what types of documents to generate based on project scope"""
        # For comprehensive projects, generate 4 documents
        # For simpler projects, generate 1-2 documents
        
        total_content = len(research_summary) + len(self._build_conversation_summary(context))
        
        if total_content > 20000:
            # Comprehensive project - all 4 documents
            return [
                "System Architecture & Implementation Guide",
                "Step-by-Step Implementation Guide",
                "Security, Testing & Operations Implementation",
                "API Documentation & Integration Guide"
            ]
        elif total_content > 10000:
            # Medium project - 2 documents
            return [
                "System Architecture & Implementation Guide",
                "Step-by-Step Implementation Guide"
            ]
        else:
            # Simple project - 1 document
            return ["Complete Implementation Guide"]
    
    # ==================== NEW STAGE: WRITE DOCUMENTS ====================
    
    def _stage_write_documents(self, context: ResearchContext) -> ResearchContext:
        """OPTIMIZED: Ollama writes comprehensive documents from outlines"""
        document_plans = context.metadata.get('document_plans', [])
        pending_documents = context.metadata.get('pending_documents', [])
        current_index = context.metadata.get('current_doc_index', 0)
        
        if current_index >= len(document_plans):
            # All documents written, move to refinement
            logger.info(f"All {len(pending_documents)} documents written by Ollama")
            context.current_stage = ConversationStage.REFINE_DOCUMENTS
            context.metadata['current_doc_index'] = 0
            context.metadata['doc_refinement_round'] = 0
            return context
        
        # Get current document plan
        plan = document_plans[current_index]
        doc_type = plan['doc_type']
        doc_number = plan['doc_number']
        
        # Ollama writes the document (32K capacity!)
        self._update_status("Ollama", f"Stage 9b/11: Writing document {doc_number}/{len(document_plans)}", f"Document: {doc_type}")
        logger.info(f"Ollama writing document {doc_number}: {doc_type}")
        
        document_message = self.ollama_client.write_document_from_outline(
            outline=plan['outline'],
            research_summary=plan['research_summary'],
            doc_number=doc_number,
            doc_type=doc_type,
            user_prompt=context.user_prompt
        )
        context.add_message(document_message)
        
        # Store the written document
        pending_documents.append({
            'doc_number': doc_number,
            'title': doc_type,
            'content': document_message.content,
            'outline': plan['outline'],
            'filename': f"{str(doc_number).zfill(2)}_{self._sanitize_filename(doc_type)}.md",
            'category': self._categorize_document(doc_type)
        })
        
        context.metadata['pending_documents'] = pending_documents
        context.metadata['current_doc_index'] = current_index + 1
        
        return context
    
    def _sanitize_filename(self, title: str) -> str:
        """Sanitize title for filename"""
        import re
        # First remove newlines and carriage returns
        sanitized = title.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
        # Remove special characters, replace spaces with underscores
        sanitized = re.sub(r'[^\w\s-]', '', sanitized)
        sanitized = re.sub(r'[-\s]+', '_', sanitized)
        # Remove any remaining control characters
        sanitized = ''.join(c if c.isprintable() or c in ('_', '-') else '_' for c in sanitized)
        return sanitized.lower()[:100]  # Limit length
    
    def _categorize_document(self, doc_type: str) -> str:
        """Categorize document for organization"""
        if 'architecture' in doc_type.lower():
            return 'architecture'
        elif 'implementation' in doc_type.lower() or 'step' in doc_type.lower():
            return 'implementation'
        elif 'security' in doc_type.lower() or 'testing' in doc_type.lower() or 'operations' in doc_type.lower():
            return 'operations'
        elif 'api' in doc_type.lower():
            return 'api'
        else:
            return 'general'
    
    # ==================== STAGE 10: REFINE DOCUMENTS ====================
    
    def _stage10_refine_documents(self, context: ResearchContext) -> ResearchContext:
        """LLMs refine documents iteratively until perfect"""
        pending = context.metadata.get('pending_documents', [])
        approved = context.metadata.get('approved_documents', [])
        current_index = context.metadata.get('current_doc_index', 0)
        round_num = context.metadata.get('doc_refinement_round', 0)
        max_rounds_per_doc = 5  # INCREASED from 3 to 5 for more comprehensive refinement
        
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
            # FIRST: DeepSeek reviews for technical accuracy (NEW OPTIMIZED)
            self._update_status("DeepSeek", f"Stage 10/11: Reviewing '{doc_title}'", f"Doc {current_index + 1}/{len(pending)}, Round {round_num + 1}/{max_rounds_per_doc}")
            logger.info(f"Stage 10: DeepSeek reviewing '{doc_title}' for technical accuracy")
            
            doc_content = current_doc.get('content', '')
            doc_outline = current_doc.get('outline', '')
            full_research = self._build_research_summary(context, max_tokens=None)
            
            # DeepSeek reviews against research context (has 128K context window)
            review_message = self.deepseek_client.review_document_accuracy(
                document_content=doc_content,
                original_outline=doc_outline,
                research_context=full_research,
                doc_title=doc_title
            )
            context.add_message(review_message)
            self._update_status("DeepSeek", f"Stage 10/11: Technical review complete", review_message.content[:500])
            
            # Check for approval
            if "APPROVED" in review_message.content.upper():
                logger.info(f"Document '{doc_title}' approved by DeepSeek")
                approved.append(current_doc)
                context.metadata['approved_documents'] = approved
                context.metadata['current_doc_index'] = current_index + 1
                context.metadata['doc_refinement_round'] = 0
            else:
                # Store review for Ollama to use in revision
                context.metadata['last_review'] = review_message.content
                
        else:
            # SECOND+: Ollama revises based on DeepSeek's feedback (NEW OPTIMIZED)
            if round_num % 2 == 1:
                # Ollama revises (32K capacity!)
                self._update_status("Ollama", f"Stage 10/11: Revising '{doc_title}'", f"Doc {current_index + 1}/{len(pending)}, Round {round_num + 1}/{max_rounds_per_doc}")
                logger.info(f"Stage 10: Ollama revising '{doc_title}' with 32K capacity")
                
                review_feedback = context.metadata.get('last_review', 'Expand with more detail')
                doc_outline = current_doc.get('outline', '')
                
                # Ollama revises with full 32K capacity
                revision_message = self.ollama_client.revise_document(
                    original_document=current_doc.get('content', ''),
                    review_feedback=review_feedback,
                    outline=doc_outline,
                    doc_type=doc_title
                )
                context.add_message(revision_message)
                self._update_status("Ollama", f"Stage 10/11: Revision complete", f"{len(revision_message.content)} chars generated")
                
                # Update document with revision
                current_doc['content'] = revision_message.content
                pending[current_index] = current_doc
                context.metadata['pending_documents'] = pending
                
            else:
                # DeepSeek re-reviews for accuracy
                self._update_status("DeepSeek", f"Stage 10/11: Re-reviewing '{doc_title}'", f"Doc {current_index + 1}/{len(pending)}, Round {round_num + 1}/{max_rounds_per_doc}")
                logger.info(f"Stage 10: DeepSeek re-reviewing '{doc_title}'")
                
                doc_outline = current_doc.get('outline', '')
                full_research = self._build_research_summary(context, max_tokens=None)
                
                # DeepSeek reviews revised document
                review_message = self.deepseek_client.review_document_accuracy(
                    document_content=current_doc.get('content', ''),
                    original_outline=doc_outline,
                    research_context=full_research,
                    doc_title=doc_title
                )
                context.add_message(review_message)
                self._update_status("DeepSeek", f"Stage 10/11: Re-review complete", review_message.content)
                
                # Store review feedback for next revision
                context.metadata['last_review'] = review_message.content
                
                # Check for approval
                if "APPROVED" in review_message.content.upper() or "TECHNICALLY_ACCURATE" in review_message.content.upper():
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
        """Build research summary for LLM context - OPTIMIZED with caching"""
        # Create cache key based on message count and max_tokens
        cache_key = f"{len(context.messages)}_{max_tokens}_{len(context.key_insights)}"
        
        # Return cached summary if available
        if cache_key in self._summary_cache:
            return self._summary_cache[cache_key]
        
        # Pre-calculate sizes once
        total_searches = len(context.initial_searches) + len(context.targeted_searches)
        insights_count = len(context.key_insights)
        messages_count = len(context.messages)
        
        # Use list comprehension for faster building
        summary_parts = [
            f"PROJECT: {context.user_prompt}",
            f"RESEARCH: {total_searches} searches, {insights_count} insights, {messages_count} messages",
            ""
        ]
        
        # Key insights - optimized with list comprehension
        if context.key_insights:
            summary_parts.append("KEY INSIGHTS:")
            max_insights = 30 if max_tokens is None else 10
            summary_parts.extend([f"• {insight['content'][:200]}" for insight in context.key_insights[:max_insights]])
            summary_parts.append("")
        
        # Recent conversation - optimized with list comprehension
        if context.messages:
            summary_parts.append("RECENT DISCUSSION:")
            recent = context.messages[-5:] if max_tokens else context.messages[-10:]
            char_limit = 300 if max_tokens else 500
            summary_parts.extend([f"[{msg.llm_type.value}]: {msg.content[:char_limit]}..." for msg in recent])
        
        full_summary = "\n".join(summary_parts)
        
        # Truncate if needed
        if max_tokens:
            max_chars = max_tokens * 4  # Rough estimate: 4 chars per token
            if len(full_summary) > max_chars:
                full_summary = full_summary[:max_chars] + "\n..."
        
        # Cache the result (limit cache size to last 10 entries)
        self._summary_cache[cache_key] = full_summary
        if len(self._summary_cache) > 10:
            # Remove oldest entry (first item)
            oldest_key = next(iter(self._summary_cache))
            del self._summary_cache[oldest_key]
        
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
