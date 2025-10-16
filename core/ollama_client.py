import requests
import json
import logging
from typing import List, Dict, Any, Optional
from config.settings import Config
from core.models import LLMMessage, LLMType

logger = logging.getLogger(__name__)


class OllamaClient:
    """Client for interacting with local Ollama instance"""
    
    def __init__(self):
        self.base_url = Config.OLLAMA_BASE_URL
        self.model = Config.OLLAMA_MODEL
        self.timeout = Config.OLLAMA_TIMEOUT  # Use dedicated Ollama timeout (5 minutes default)
        
        # Verify Ollama is running and model is available
        self._verify_connection()
    
    def _verify_connection(self):
        """Verify that Ollama is running and the model is available"""
        try:
            # Check if Ollama is running
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            response.raise_for_status()
            
            # Check if our model is available
            models_data = response.json()
            available_models = [model['name'] for model in models_data.get('models', [])]
            
            if self.model not in available_models:
                logger.warning(f"Model {self.model} not found in available models: {available_models}")
                logger.info("You may need to pull the model using: ollama pull qwen3-coder:latest")
            
            logger.info(f"Ollama connection verified, model: {self.model}")
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to connect to Ollama at {self.base_url}: {e}")
            raise ConnectionError(f"Ollama is not running or not accessible at {self.base_url}")
    
    def _make_request(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make API request to Ollama"""
        url = f"{self.base_url}/api/{endpoint}"
        
        try:
            response = requests.post(
                url,
                json=data,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama API request failed: {e}")
            raise
    
    def generate_response(self, 
                         prompt: str, 
                         context: Optional[str] = None,
                         temperature: float = None,
                         max_tokens: int = None) -> LLMMessage:
        """Generate a response from Ollama"""
        
        # Use config defaults if not specified
        if temperature is None:
            temperature = Config.OLLAMA_DEFAULT_TEMPERATURE
        if max_tokens is None:
            max_tokens = Config.OLLAMA_DEFAULT_MAX_TOKENS
        
        # Build the system prompt for CONCISE, focused responses
        system_prompt = f"""You are Ollama, an expert software architect and technical reviewer.

CRITICAL RESPONSE GUIDELINES (FOLLOW STRICTLY):
- Be CONCISE and FOCUSED - quality over quantity
- Limit responses to 800-1200 words maximum
- Use bullet points and structured formats for clarity
- Provide specific, actionable feedback only
- Keep code examples minimal (< 20 lines) - use pseudocode when possible
- ALWAYS complete your thoughts - never stop mid-sentence
- Your response has a STRICT limit of {max_tokens} tokens - use them wisely

Response Structure (stick to this format):
1. **Key Assessment** (2-3 sentences): What's good/bad about the proposal
2. **Specific Issues** (bullet list, 3-5 items): Concrete problems or concerns
3. **Recommendations** (bullet list, 3-5 items): Specific changes needed
4. **Decision**: State "APPROVED" or "NEEDS REVISION: [specific reason]"

REMEMBER: This is a REVIEW phase. Be critical, concise, and decisive. Don't repeat information."""
        if context:
            system_prompt += f"\n\nResearch context: {context[:200]}..."
        citation_instruction = "\n\nCite sources when needed: [Source: URL]"
        system_prompt += citation_instruction
        
        # ENFORCE hard limit (80% of requested to leave safety buffer)
        enforced_limit = int(max_tokens * 0.8)
        max_chars = enforced_limit * 4  # Rough estimate: 1 token ≈ 4 characters
        
        request_data = {
            "model": self.model,
            "prompt": prompt,
            "system": system_prompt,
            "stream": True,  # CRITICAL: Enable streaming for real-time control
            "options": {
                "temperature": temperature,
                "num_predict": enforced_limit,  # Enforced limit, not just suggestion
                "num_ctx": 32768,  # Large context window
                "top_k": 40,
                "top_p": 0.9,
                "repeat_penalty": 1.2,  # Increased to discourage repetition
                "stop": ["</response>", "---END---", "\n\nIn conclusion"]  # Add stop sequences
            }
        }
        
        try:
            # Log request
            logger.info(f"[OLLAMA] Generating response with max_tokens={max_tokens}, enforced_limit={enforced_limit}, max_chars={max_chars}")

            # Stream the response and enforce limits in real-time
            url = f"{self.base_url}/api/generate"
            response = requests.post(url, json=request_data, timeout=self.timeout, stream=True)
            response.raise_for_status()
            
            content = ""
            token_count = 0
            
            # Stream and collect response with hard limits
            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line)
                        if chunk.get("response"):
                            content += chunk.get("response")
                            token_count += 1
                        
                        # HARD STOP if limits exceeded
                        if len(content) >= max_chars or token_count >= enforced_limit:
                            logger.warning(f"[OLLAMA] Stopping at {token_count} tokens / {len(content)} chars (limit reached)")
                            break
                        
                        if chunk.get("done"):
                            logger.info(f"[OLLAMA] Response completed naturally at {token_count} tokens")
                            break
                    except json.JSONDecodeError:
                        continue  # Skip malformed lines
            
            # Validate response is not empty
            if not content.strip():
                raise ValueError("Empty response from Ollama")
            
            # Validate completeness and fix if needed
            if not self._is_response_complete(content):
                logger.warning(f"[OLLAMA] Response incomplete, attempting to finish last sentence")
                content = self._complete_last_sentence(content)
            
            # Create LLM message
            message = LLMMessage(
                llm_type=LLMType.OLLAMA,
                content=content,
                confidence_score=0.8  # Default confidence
            )
            
            logger.info(f"[OLLAMA] Response generated: {len(content)} characters, max_tokens: {max_tokens}")
            return message
            
        except Exception as e:
            logger.error(f"Failed to generate Ollama response: {e}")
            # Return a fallback message
            return LLMMessage(
                llm_type=LLMType.OLLAMA,
                content=f"I encountered an error while processing your request: {str(e)}. Please check if Ollama is running.",
                confidence_score=0.0
            )
    
    def review_deepseek_analysis(self, 
                               user_prompt: str,
                               research_context: str,
                               deepseek_analysis: str) -> LLMMessage:
        """Review and provide comprehensive implementation feedback on DeepSeek's analysis"""
        
        prompt = f"""You are Ollama, a practical implementation expert discussing: {user_prompt}

DeepSeek just provided this comprehensive analysis:
{deepseek_analysis}

RESPONSE LIMIT: I have exactly 24576 tokens for this response. I need to provide thorough, practical implementation guidance.

As the implementation expert, I need to provide comprehensive practical guidance covering:

**IMPLEMENTATION FEASIBILITY ANALYSIS:**
- Review each of DeepSeek's recommendations for real-world practicality
- Identify potential implementation bottlenecks or complexity issues
- Suggest practical alternatives where the proposed approach might be overly complex

**TECHNICAL IMPLEMENTATION DETAILS:**
- Provide specific implementation strategies for the most critical components
- Detail the actual development workflow and setup procedures
- Cover practical considerations for development environment configuration
- Address common pitfalls and how to avoid them during implementation

**RESOURCE & TIMELINE REALITIES:**
- Assess the realistic development timeline based on team size and complexity
- Identify resource requirements (both human and infrastructure)
- Suggest prioritization for MVP vs full feature implementation
- Recommend phasing strategies for complex features

**OPERATIONAL CONSIDERATIONS:**
- Address deployment complexity and operational overhead
- Cover monitoring, maintenance, and troubleshooting procedures
- Discuss performance optimization from a practical standpoint
- Address scalability concerns and when they become relevant

**RISK MITIGATION:**
- Identify technical risks and provide mitigation strategies
- Suggest fallback approaches for high-risk components
- Recommend testing strategies to validate critical functionality
- Address security implementation from a practical perspective

DeepSeek, your analysis is comprehensive, but I want to make sure we're considering the practical realities of actually building and maintaining this system. What's your take on the most challenging aspects to implement?

PROVIDE DETAILED PRACTICAL GUIDANCE - this should give implementers specific direction on how to actually build this."""
        
        return self.generate_response(prompt, research_context, max_tokens=Config.OLLAMA_REVIEW_MAX_TOKENS)
    
    def refine_technical_approach(self,
                                user_prompt: str,
                                research_context: str,
                                deepseek_refinement: str,
                                previous_concerns: str) -> LLMMessage:
        """Refine technical approach based on DeepSeek's refinement"""
        
        prompt = f"""
        Based on the user's prompt, research context, DeepSeek's refined analysis, and your previous concerns, 
        provide updated technical recommendations:

        RESPONSE LIMIT: You have exactly 24576 tokens for this response. Plan accordingly.

        USER PROMPT: {user_prompt}

        RESEARCH CONTEXT: {research_context}

        DEEPSEEK'S REFINED ANALYSIS: {deepseek_refinement}

        YOUR PREVIOUS CONCERNS: {previous_concerns}

        Please (stay within 24576 tokens):
        1. Address how DeepSeek's refinement resolves your previous concerns
        2. Provide updated implementation recommendations
        3. Identify any remaining technical challenges
        4. Suggest specific technologies, frameworks, and tools
        5. Provide code examples or implementation patterns where relevant

        Focus on building a technically sound implementation plan within the token limit.
        """
        
        return self.generate_response(prompt, research_context, max_tokens=Config.OLLAMA_REVIEW_MAX_TOKENS)
    
    def continue_discussion(self, user_prompt: str, research_context: str, deepseek_response: str) -> LLMMessage:
        """Continue the discussion based on DeepSeek's latest points"""
        logger.info("Ollama continuing discussion")
        
        # Keep more context for comprehensive responses
        if len(deepseek_response) > 2000:
            deepseek_response = deepseek_response[:2000] + "... [response continues...]"
        
        prompt = f"""You are Ollama, a technical implementation expert having an in-depth conversation with DeepSeek about: {user_prompt}

DeepSeek's latest analysis:
{deepseek_response}

COMPREHENSIVE RESPONSE REQUIRED:
You must provide a DETAILED technical response of at least 1500-2000 words covering:

**1. IMPLEMENTATION ANALYSIS (400-500 words):**
- Analyze DeepSeek's technical recommendations for practical feasibility
- Identify specific implementation challenges and complexity points
- Suggest concrete alternatives or modifications for better implementability
- Address real-world development constraints and timeline considerations

**2. TECHNICAL DEEP DIVE (500-600 words):**
- Provide detailed implementation strategies for the most critical components
- Include specific code patterns, architectural decisions, and technology choices
- Cover development environment setup and configuration requirements
- Discuss integration patterns and data flow implementations

**3. PRACTICAL CONSIDERATIONS (400-500 words):**
- Address deployment, monitoring, and operational requirements
- Cover testing strategies and quality assurance approaches
- Discuss performance optimization and scalability planning
- Identify potential technical debt and maintenance considerations

**4. SPECIFIC RECOMMENDATIONS (200-300 words):**
- Provide concrete next steps for implementation
- Suggest specific tools, frameworks, and libraries
- Recommend development methodology and team structure
- Pose strategic questions for further technical discussion

Ensure each section provides unique, actionable technical insights. Build upon DeepSeek's analysis with practical implementation expertise."""

        return self.generate_response(prompt, research_context, max_tokens=Config.OLLAMA_DISCUSSION_MAX_TOKENS)
    
    # ==================== NEW: OPTIMIZED DOCUMENT WORKFLOW ====================
    
    def write_document_from_outline(self,
                                    outline: str,
                                    research_summary: str,
                                    doc_number: int,
                                    doc_type: str,
                                    user_prompt: str) -> LLMMessage:
        """Write COMPREHENSIVE, PRODUCTION-READY document from DeepSeek's outline (64K capacity!)"""
        
        prompt = f"""You are an expert technical writer creating PRODUCTION-READY, ENTERPRISE-LEVEL documentation that covers the COMPLETE SOFTWARE DEVELOPMENT LIFECYCLE.

PROJECT: {user_prompt}
DOCUMENT #{doc_number}: {doc_type}

DETAILED OUTLINE (follow EXACTLY and EXPAND MASSIVELY):
{outline}

RESEARCH SUMMARY (use for technical accuracy and examples):
{research_summary}

🎯 YOUR MISSION:
Write a MASSIVE, COMPREHENSIVE IMPLEMENTATION GUIDE that a development team can use to build a COMPLETE PRODUCTION SYSTEM from scratch. This is NOT a tutorial - this is ENTERPRISE DOCUMENTATION.

📋 CRITICAL REQUIREMENTS (ALL MANDATORY):

1. **DOCUMENT LENGTH**: 20,000-30,000+ words minimum
   - This should take 30-60 minutes to read
   - If you're not using 60,000+ tokens, you're not detailed enough

2. **CODE VOLUME**: 3,000-5,000+ lines of actual code
   - COMPLETE implementations, not snippets
   - NO placeholders like "..." or "// rest of code"
   - Every file must be production-ready and runnable
   - Include 50-100+ complete code examples

3. **CONFIGURATION FILES**: 30-50+ complete configuration files
   - package.json/requirements.txt with ALL 50+ dependencies
   - Complete tsconfig.json, .eslintrc, .prettierrc
   - Complete docker-compose.yml with all services
   - Complete CI/CD pipeline files (300+ lines)
   - Complete .env.example with 50-100 variables
   - Every configuration file must be COMPLETE and copy-pasteable

4. **DEVELOPMENT LIFECYCLE COVERAGE** (ALL phases required):
   
   **PHASE 1: Requirements & Planning**
   - 30-50 functional requirements with acceptance criteria
   - 20-30 user stories with detailed scenarios
   - Complete system architecture diagrams (textual/Mermaid)
   - Technology stack justification (500+ words)
   - Project timeline with milestones
   
   **PHASE 2: Environment Setup** (Day-by-day, Week 1)
   - Complete OS setup (Windows/Mac/Linux) - all commands
   - Install 20+ tools with exact versions and commands
   - IDE setup with 30+ extensions listed
   - Database installation and configuration (complete)
   - Docker, Redis, message queues setup (complete)
   - Project initialization with every command explained
   
   **PHASE 3: Database Design** (Week 1-2)
   - Complete SQL schema for 15-30 tables
   - Every table with ALL columns, constraints, indices
   - Complete migration files (3-5 migrations, 500+ lines total)
   - Seed data scripts (200+ lines)
   - Complete ORM models for ALL entities (2,000-3,000 lines)
   
   **PHASE 4: Backend Implementation** (Week 2-4)
   - Complete server setup (300+ lines)
   - Authentication system (JWT + OAuth) - COMPLETE (800-1,000 lines)
   - Authorization & RBAC - COMPLETE (500+ lines)
   - ALL API endpoints for ALL resources (15-30 resources)
   - For EACH resource: GET, POST, PUT, DELETE, PATCH endpoints
   - Complete service layer for ALL business logic (2,000-3,000 lines)
   - Complete validation schemas for ALL endpoints (1,000+ lines)
   - Complete error handling middleware (300+ lines)
   
   **PHASE 5: Frontend Implementation** (Week 3-5)
   - Complete framework setup with all configurations
   - 30-50 reusable components (COMPLETE implementations)
   - 20-30 pages/views (COMPLETE implementations)
   - Complete state management setup (Redux/MobX/Context)
   - Complete API client with all methods (1,000+ lines)
   - Complete routing with protected routes (300+ lines)
   - Forms with validation (10-15 complete forms)
   
   **PHASE 6: Testing Implementation** (Week 5-6)
   - Unit tests for EVERY component/service (3,000-5,000 lines)
   - Integration tests for ALL API endpoints (2,000-3,000 lines)
   - E2E tests for ALL user flows (1,000-2,000 lines)
   - Test configurations (Jest, Pytest, Cypress - complete)
   - Mock data and fixtures (500+ lines)
   
   **PHASE 7: Security Implementation** (Throughout)
   - Complete security middleware (500+ lines)
   - Input validation EVERYWHERE (examples)
   - SQL injection prevention (examples)
   - XSS prevention (examples)
   - CSRF protection (implementation)
   - Rate limiting (complete config)
   - Data encryption (implementation)
   - Security headers (complete Helmet.js config)
   
   **PHASE 8: DevOps & Deployment** (Week 6-7)
   - Complete Dockerfiles for ALL services (500+ lines total)
   - Complete docker-compose.yml (300+ lines)
   - Complete CI/CD pipelines (GitHub Actions, 400+ lines)
   - Complete deployment scripts (300+ lines)
   - Complete Kubernetes manifests OR Terraform (500+ lines)
   - Complete monitoring setup (Prometheus/Grafana configs)
   - Complete logging setup (Winston/Pino configurations)
   
   **PHASE 9: Operations & Maintenance** (Ongoing)
   - 20-30 operational runbooks (200+ lines each)
   - 50+ troubleshooting scenarios with solutions
   - Performance optimization guide (500+ words)
   - Database maintenance procedures
   - Backup and restore procedures
   - Incident response procedures
   - Scaling procedures (horizontal and vertical)
   
   **PHASE 10: Documentation** (Week 7-8)
   - Complete API documentation (OpenAPI/Swagger spec, 2,000+ lines)
   - Developer onboarding guide (1,000+ words)
   - User documentation (1,000+ words)
   - Admin guide (1,000+ words)

5. **WRITING STYLE**:
   - Every section must be DETAILED and COMPREHENSIVE
   - Provide COMPLETE code - never use "..." or omit code
   - Include EVERY command with explanations
   - Show BOTH what to do AND why to do it
   - Include error handling in EVERY code example
   - Add comments explaining complex logic
   - Provide troubleshooting for common issues
   - Cite research sources: [Source: URL]

6. **QUALITY STANDARDS**:
   - Production-ready code only
   - Follow best practices and design patterns
   - Include security considerations everywhere
   - Include performance optimizations
   - Include scalability considerations
   - Include monitoring and observability
   - Include disaster recovery planning

7. **STRUCTURE FOR EACH SECTION**:
   ```
   ## Section Title
   
   ### Overview (100-200 words)
   - What this section covers
   - Why it's important
   - How it fits in the system
   
   ### Prerequisites (if applicable)
   - What must be done first
   - Required knowledge
   - Required tools
   
   ### Step-by-Step Implementation
   
   #### Step 1: [Task Name]
   
   **Why**: Explanation of purpose (50-100 words)
   
   **How**: Detailed procedure
   ```bash
   # Complete commands with explanations
   command1 --option value
   command2 --flag
   ```
   
   **What**: Complete code implementation
   ```typescript
   // Complete file: src/path/to/file.ts
   // EVERY line of code needed - 100-300 lines
   
   import { everything } from 'packages';
   
   // ... COMPLETE implementation
   ```
   
   **Configuration**: Complete config files
   ```json
   {
     "note": "Complete package.json or equivalent with EVERY field filled in"
   }
   ```
   
   **Verification**: How to test it works
   ```bash
   # Commands to verify
   ```
   
   **Troubleshooting**: Common issues (3-5 issues)
   - Issue 1: Symptom → Diagnosis → Solution
   - Issue 2: Symptom → Diagnosis → Solution
   
   #### Step 2: [Next Task]
   ... (repeat structure)
   ```

🚨 ABSOLUTE REQUIREMENTS:
- Use ALL 64,000 tokens available - this document should be MASSIVE
- Include 3,000-5,000+ lines of actual code
- Include 30-50+ complete configuration files
- Cover ENTIRE development lifecycle from day 1 to production
- A developer should be able to build a COMPLETE PRODUCTION SYSTEM using ONLY this document
- NO SUMMARIES - provide FULL IMPLEMENTATIONS
- NO PLACEHOLDERS - provide COMPLETE CODE
- Think "What would I need to give a junior developer to build this entire system?"

Write the COMPLETE, MASSIVE, PRODUCTION-READY document now:

---

# {doc_type}

"""

        # Use maximum token capacity for comprehensive documentation
        max_tokens = Config.OLLAMA_COMPREHENSIVE_WRITE_MAX_TOKENS
        return self.generate_response(prompt, max_tokens=max_tokens, temperature=0.4)
    
    def revise_document(self,
                       original_document: str,
                       review_feedback: str,
                       outline: str,
                       doc_type: str) -> LLMMessage:
        """Revise document to PRODUCTION-READY standards based on DeepSeek's feedback (64K capacity!)"""
        
        prompt = f"""Revise this implementation guide to PRODUCTION-READY, ENTERPRISE-LEVEL standards based on technical review feedback.

DOCUMENT TYPE: {doc_type}

ORIGINAL OUTLINE:
{outline[:5000]}

ORIGINAL DOCUMENT (first part):
{original_document[:15000]}

REVIEW FEEDBACK (from technical reviewer):
{review_feedback}

🎯 YOUR MISSION:
Create the MASSIVELY IMPROVED, PRODUCTION-READY version that addresses ALL feedback and expands to enterprise documentation standards.

📋 REVISION REQUIREMENTS (ALL MANDATORY):

1. **FIX ALL ISSUES IDENTIFIED IN FEEDBACK**:
   - Correct every technical inaccuracy mentioned
   - Add every missing section, file, or example mentioned
   - Complete every incomplete code example
   - Add missing configuration files in FULL
   - Fix or add missing citations [Source: URL]
   - Address every specific concern raised

2. **EXPAND TO PRODUCTION STANDARDS**:
   - Target: 20,000-30,000+ words (use ALL 64,000 tokens)
   - Include 3,000-5,000+ lines of code
   - Include 30-50+ complete configuration files
   - Add comprehensive implementations for ALL phases
   - Expand thin sections to full detail

3. **ENHANCE CODE COMPLETENESS**:
   - Replace ANY code snippets with COMPLETE files
   - Remove ALL placeholders ("...", "// rest of code")
   - Add error handling to EVERY function
   - Add input validation EVERYWHERE
   - Include logging in every important function
   - Add comments explaining complex logic
   - Every code example should be 100-500+ lines

4. **ADD MISSING LIFECYCLE PHASES** (if not comprehensive):
   - Requirements & Planning (if missing)
   - Environment Setup (day-by-day, week 1)
   - Database Design (complete schemas, 15-30 tables)
   - Backend Implementation (complete APIs, 2,000-3,000 lines)
   - Frontend Implementation (complete components, 2,000-3,000 lines)
   - Testing (complete test suites, 3,000-5,000 lines)
   - Security (complete implementations, 800-1,000 lines)
   - DevOps & Deployment (complete pipelines, 1,000+ lines)
   - Operations & Maintenance (20-30 runbooks)
   - Documentation (OpenAPI specs, user guides)

5. **ADD COMPREHENSIVE EXAMPLES**:
   - 50-100+ complete code examples
   - 30-50+ complete configuration files
   - 20-30 operational runbooks
   - 50+ troubleshooting scenarios with solutions
   - Complete setup commands for all environments
   - Complete deployment procedures
   - Complete monitoring and logging setup

6. **MAINTAIN WHAT'S GOOD**:
   - Keep same overall structure and organization
   - Preserve all existing good content and examples
   - Keep existing citations and references
   - Maintain technical accuracy where correct

7. **QUALITY ENHANCEMENTS**:
   - Add production-ready error handling
   - Add security best practices throughout
   - Add performance optimization tips
   - Add scalability considerations
   - Add monitoring and observability
   - Add disaster recovery procedures
   - Add testing strategies for each component

8. **VERIFICATION SECTIONS**:
   - Add "How to verify this works" after each major step
   - Add "Testing this component" for each implementation
   - Add "Common issues" (3-5 per section)
   - Add "Performance benchmarks" where relevant

🚨 CRITICAL STANDARDS:
- This is NOT a minor revision - EXPAND MASSIVELY
- Use ALL 64,000 tokens available
- A junior developer should be able to build a COMPLETE PRODUCTION SYSTEM using ONLY this document
- Include EVERYTHING needed: requirements → design → implementation → testing → deployment → operations
- NO SUMMARIES - provide FULL IMPLEMENTATIONS
- NO PLACEHOLDERS - provide COMPLETE CODE
- Every configuration file must be COMPLETE and copy-pasteable
- Think "What documentation would I want if building this for a Fortune 500 company?"

Write the COMPLETE, MASSIVELY IMPROVED, PRODUCTION-READY document now:

---

# {doc_type}

"""

        # Use maximum token capacity for comprehensive documentation
        max_tokens = Config.OLLAMA_COMPREHENSIVE_WRITE_MAX_TOKENS
        return self.generate_response(prompt, max_tokens=max_tokens, temperature=0.4)
    
    def validate_implementation_feasibility(self, 
                                          development_plan: str) -> Dict[str, Any]:
        """Validate the technical feasibility of a development plan"""
        
        prompt = f"""
        Evaluate the technical feasibility of the following development plan:

        DEVELOPMENT PLAN:
        {development_plan}

        Please assess:
        1. Technical viability of the proposed architecture
        2. Realism of implementation timelines
        3. Resource requirements and availability
        4. Potential technical risks and mitigation strategies
        5. Code complexity and maintainability considerations

        Provide a feasibility score (0.0 to 1.0) and detailed feedback.
        """
        
        response = self.generate_response(prompt, temperature=0.3, max_tokens=Config.OLLAMA_VALIDATION_MAX_TOKENS)
        
        # Parse the response to extract feasibility assessment
        # This is a simplified implementation
        feasibility_score = 0.7  # Default score
        
        # Try to extract a score from the response
        import re
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", response.content)
        if numbers:
            try:
                feasibility_score = min(1.0, max(0.0, float(numbers[0])))
            except (ValueError, IndexError):
                pass
        
        return {
            'feasibility_score': feasibility_score,
            'technical_feedback': response.content,
            'risks_identified': self._extract_risks(response.content),
            'recommendations': self._extract_recommendations(response.content)
        }
    
    def _extract_risks(self, content: str) -> List[str]:
        """Extract identified risks from technical feedback"""
        # Simple keyword-based extraction - could be enhanced with more sophisticated NLP
        risk_keywords = ['risk', 'challenge', 'difficulty', 'complex', 'bottleneck', 'limitation']
        sentences = content.split('.')
        
        risks = []
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in risk_keywords):
                risks.append(sentence.strip())
        
        return risks[:5]  # Return top 5 risks
    
    def _extract_recommendations(self, content: str) -> List[str]:
        """Extract recommendations from technical feedback"""
        # Simple keyword-based extraction
        recommendation_keywords = ['recommend', 'suggest', 'should', 'consider', 'implement']
        sentences = content.split('.')
        
        recommendations = []
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in recommendation_keywords):
                recommendations.append(sentence.strip())
        
        return recommendations[:5]  # Return top 5 recommendations
    
    def generate_code_examples(self, technology: str, use_case: str) -> str:
        """Generate code examples for specific technology and use case"""
        
        prompt = f"""
        Generate practical code examples for using {technology} in the context of: {use_case}

        Please provide:
        1. Installation/setup instructions if needed
        2. Basic usage examples
        3. Best practices and common patterns
        4. Integration considerations
        5. Potential pitfalls to avoid

        Focus on providing actionable, runnable code examples.
        """
        
        response = self.generate_response(prompt, temperature=0.5, max_tokens=Config.OLLAMA_VALIDATION_MAX_TOKENS)
        return response.content
    def _is_response_complete(self, content: str) -> bool:
        """
        Check if response ends properly and appears complete
        
        Returns:
            True if response looks complete, False otherwise
        """
        if not content or len(content) < 10:
            return False
        
        # Check last 100 chars for proper ending
        ending = content[-100:].strip()
        
        # Should end with sentence punctuation or code block
        valid_endings = ('.', '!', '?', '\n', '`', '"', "'", ')', ']', '}')
        has_valid_ending = ending.endswith(valid_endings)
        
        if not has_valid_ending:
            logger.debug(f"Response ending invalid: '{ending[-30:]}'")
            return False
        
        # Check for common incomplete patterns
        incomplete_patterns = ['in order to', 'for example', 'such as', 'as follows:', 'will be', 'should be', 'this is', 'which means', 'because of']
        ending_lower = ending.lower()
        for pattern in incomplete_patterns:
            if ending_lower.endswith(pattern):
                logger.debug(f"Response ends with incomplete pattern: '{pattern}'")
                return False
        
        # Check for words longer than 25 characters (likely corrupted)
        words = ending.split()
        if words and len(words[-1]) > 25:
            logger.debug(f"Response ends with suspiciously long word: '{words[-1]}'")
            return False
        return True

    def _complete_last_sentence(self, content: str) -> str:
        """
        Try to complete an incomplete sentence by finding last complete sentence
        """
        for punct in ['. ', '.\n', '! ', '!\n', '? ', '?\n', '\n\n']:
            last_pos = content.rfind(punct)
            if last_pos > len(content) * 0.85:
                truncated = content[:last_pos + 1].rstrip()
                logger.info(f"Truncated incomplete response at position {last_pos}")
                return truncated
        logger.warning("Could not find good truncation point, adding ellipsis")
        return content.rstrip() + "..."
