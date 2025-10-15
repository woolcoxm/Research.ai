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
        
        # Build the system prompt for comprehensive responses
        system_prompt = f"""You are Ollama, an expert software architect and technical documentation specialist having a discussion with DeepSeek about software project planning and architecture.

CRITICAL INSTRUCTIONS:
- PRIORITIZE comprehensive documentation, architectural decisions, and planning over code implementation
- Focus on WHY and WHAT rather than detailed HOW (save code for later implementation)
- Provide DETAILED architectural analysis and documentation (1500-2000 words minimum)
- NEVER stop mid-sentence or mid-thought
- Include code ONLY when necessary to illustrate critical architectural concepts (keep it minimal - pseudocode or small examples)
- Each response should focus on design patterns, system architecture, and implementation strategy
- Continue discussing until you've provided substantial architectural and planning value
- NEVER repeat the same content - each paragraph must add new insights

Your response should be structured with:
1. Architectural analysis and design decisions
2. System design and component interaction documentation
3. Technology choices and trade-offs (explain WHY, not implement HOW)
4. Minimal pseudocode or conceptual examples ONLY when absolutely needed for clarity
5. Implementation considerations and next planning steps

REMEMBER: This is the PLANNING phase - focus on documenting WHAT to build and WHY, not writing production code.
You have {max_tokens} tokens available - use them for architecture documentation and planning."""
        if context:
            system_prompt += f"\n\nResearch context: {context[:200]}..."
        citation_instruction = "\n\nCite sources when needed: [Source: URL]"
        system_prompt += citation_instruction
        
        request_data = {
            "model": self.model,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,  # Use configurable token limit
                "num_ctx": 32768,  # Large context window
                "top_k": 40,
                "top_p": 0.9,
                "repeat_penalty": 1.1,
                "stop": []  # No stop sequences - let it complete fully
            }
        }
        
        try:
            # Log request size
            try:
                import json as _json
                logger.info(f"Ollama request size: {len(_json.dumps(request_data))} bytes, max_tokens: {max_tokens}")
            except Exception:
                pass

            response_data = self._make_request("generate", request_data)
            
            # Extract the response content
            content = response_data.get("response", "")
            
            # Create LLM message
            message = LLMMessage(
                llm_type=LLMType.OLLAMA,
                content=content,
                confidence_score=0.8  # Default confidence
            )
            try:
                logger.info(f"Ollama response generated: {len(content)} characters, max_tokens: {max_tokens}")
            except Exception:
                pass
            
            logger.info(f"Ollama response generated: {len(content)} characters")
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