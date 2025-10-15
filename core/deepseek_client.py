import requests
import json
import logging
from typing import List, Dict, Any, Optional
from config.settings import Config
from core.models import LLMMessage, LLMType

logger = logging.getLogger(__name__)


class DeepSeekClient:
    """Client for interacting with DeepSeek API"""
    
    def __init__(self):
        self.api_key = Config.DEEPSEEK_API_KEY
        self.base_url = Config.DEEPSEEK_BASE_URL
        self.timeout = Config.REQUEST_TIMEOUT
        
        if not self.api_key:
            raise ValueError("DeepSeek API key not configured")
    
    def _make_request(self, endpoint: str, data: Dict[str, Any], retry_count: int = 3) -> Dict[str, Any]:
        """Make API request to DeepSeek with retry logic"""
        url = f"{self.base_url}/{endpoint}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        last_error = None
        for attempt in range(retry_count):
            try:
                logger.info(f"DeepSeek API request attempt {attempt + 1}/{retry_count}")
                response = requests.post(
                    url,
                    headers=headers,
                    json=data,
                    timeout=self.timeout
                )
                response.raise_for_status()
                return response.json()
            except requests.exceptions.Timeout as e:
                last_error = e
                logger.warning(f"DeepSeek API timeout on attempt {attempt + 1}/{retry_count}: {e}")
                if attempt < retry_count - 1:
                    logger.info(f"Retrying in 5 seconds...")
                    import time
                    time.sleep(5)
            except requests.exceptions.RequestException as e:
                last_error = e
                logger.error(f"DeepSeek API request failed on attempt {attempt + 1}/{retry_count}: {e}")
                # Don't retry on non-timeout errors (like 4xx status codes)
                if attempt < retry_count - 1 and (isinstance(e, requests.exceptions.ConnectionError) or 
                                                   (hasattr(e.response, 'status_code') and e.response.status_code >= 500)):
                    logger.info(f"Retrying in 5 seconds...")
                    import time
                    time.sleep(5)
                else:
                    raise
        
        logger.error(f"DeepSeek API request failed after {retry_count} attempts")
        raise last_error
    
    def generate_response(self, 
                         prompt: str, 
                         context: Optional[str] = None,
                         temperature: float = None,
                         max_tokens: int = None) -> LLMMessage:
        """Generate a response from DeepSeek"""
        
        # Use config defaults if not specified
        if temperature is None:
            temperature = Config.DEEPSEEK_DEFAULT_TEMPERATURE
        if max_tokens is None:
            max_tokens = Config.DEEPSEEK_DEFAULT_MAX_TOKENS
        
        # Build the messages array
        messages = []
        
        if context:
            messages.append({
                "role": "system",
                "content": (
                    f"You are an expert AI researcher and senior software architect specializing in design documentation and architectural planning. Use the following research context to inform your analysis:\n\n{context}\n\n"
                    "When answering, do the following:\n"
                    "1) Reference specific search results or key insights provided in context and cite sources in [Source: URL] format.\n"
                    "2) PRIORITIZE architectural documentation: system design explanations, textual architecture diagrams, design patterns, and technology justifications.\n"
                    "3) Include code ONLY as minimal examples to illustrate key concepts (prefer pseudocode or small snippets).\n"
                    "4) When making recommendations, explain trade-offs, design decisions, and WHY certain approaches are chosen.\n"
                    "5) If the context lacks information needed for a thorough answer, explicitly say what additional searches or details are required.\n"
                    "6) Focus on PLANNING documentation - what to build, why, and how components interact - NOT production implementation code.\n"
                    "Provide long-form architectural and design documentation suitable for engineering teams in the planning phase."
                )
            })
        else:
            messages.append({
                "role": "system",
                "content": "You are an expert AI researcher and software architect specializing in design documentation. Provide detailed architectural analysis, design decisions, and planning documentation. Minimize code - focus on design."
            })
        
        messages.append({
            "role": "user",
            "content": prompt
        })
        
        # Add explicit instruction to cite sources from the context when referencing research
        citation_instruction = "\n\nWhen referencing research or facts, cite the source using the format: [Source: URL]. If multiple sources support a claim, list them."
        if context:
            messages[0]['content'] += citation_instruction

        request_data = {
            "model": Config.DEEPSEEK_MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        
        try:
            response_data = self._make_request("chat/completions", request_data)
            
            # Extract the response content
            content = response_data["choices"][0]["message"]["content"]
            # Log prompt and response sizes for debugging
            try:
                prompt_len = len(json.dumps(request_data))
                resp_len = len(content)
                logger.info(f"DeepSeek request size: {prompt_len} bytes, response size: {resp_len} chars, max_tokens: {max_tokens}")
            except Exception:
                pass
            
            # Create LLM message
            message = LLMMessage(
                llm_type=LLMType.DEEPSEEK,
                content=content,
                confidence_score=0.8  # Default confidence
            )
            
            logger.info(f"DeepSeek response generated: {len(content)} characters")
            return message
            
        except Exception as e:
            logger.error(f"Failed to generate DeepSeek response: {e}")
            # Return a fallback message
            return LLMMessage(
                llm_type=LLMType.DEEPSEEK,
                content=f"I encountered an error while processing your request: {str(e)}. Please try again.",
                confidence_score=0.0
            )
    
    def analyze_research_context(self, 
                               user_prompt: str,
                               research_context: str) -> LLMMessage:
        """Analyze research context and provide comprehensive implementation foundation"""
        
        prompt = f"""Hi! I'm DeepSeek. I need to create a comprehensive architectural design document and implementation plan for: {user_prompt}

Research findings:
{research_context}

IMPORTANT: Focus on ARCHITECTURE, DESIGN, and PLANNING - not detailed code implementation. Provide comprehensive documentation that explains WHAT to build and WHY, with minimal code examples.

**COMPREHENSIVE ARCHITECTURAL DESIGN DOCUMENT:**

**1. System Architecture & Design Philosophy**
- Overall system architecture and component breakdown (use diagrams descriptions, NOT code)
- Architectural patterns and design principles
- Technology stack recommendations with justification (explain WHY, not HOW to implement)
- High-level component interaction flows

**2. Data Architecture & API Design**
- Database design philosophy and data modeling approach (schema concepts, not SQL code)
- API design patterns and endpoint structure (REST/GraphQL concepts, not implementations)
- Data flow documentation and state management strategy

**3. Technical Strategy & Technology Choices**
- Frontend architecture approach and framework rationale
- Backend services design and microservices/monolith decision
- Authentication/authorization strategy (concepts, not auth code)
- Integration patterns and third-party service decisions

**4. Infrastructure & Operational Design**
- Deployment architecture and environment strategy
- CI/CD pipeline approach
- Monitoring, logging, and observability strategy
- Disaster recovery and backup considerations

**5. Security, Performance & Scalability Planning**
- Security architecture and threat model
- Performance optimization strategies
- Scalability approach and growth planning
- Error handling philosophy and fault tolerance

**6. Development & Maintenance Plan**
- Project structure and organization principles
- Testing strategy (types of tests, not test code)
- Documentation requirements and standards
- Development workflow and team collaboration approach

Ollama, I want to make sure this architectural plan covers everything needed. What are your thoughts on the design decisions and implementation approach?

FOCUS ON PLANNING DOCUMENTATION - save detailed code for the actual development phase."""
        
        return self.generate_response(prompt, research_context, max_tokens=Config.DEEPSEEK_STAGE5_MAX_TOKENS)
    
    def refine_analysis(self,
                       user_prompt: str,
                       research_context: str,
                       ollama_analysis: str) -> LLMMessage:
        """Refine analysis based on Ollama's input"""
        
        prompt = f"""
        Based on the user's prompt, research context, and Ollama's technical analysis, please refine your recommendations:

        USER PROMPT: {user_prompt}

        RESEARCH CONTEXT: {research_context}

        OLLAMA'S TECHNICAL ANALYSIS: {ollama_analysis}

        Please:
        1. Address any technical concerns raised by Ollama
        2. Update your recommendations based on implementation feasibility
        3. Identify any remaining knowledge gaps that need research
        4. Propose a refined architecture and technology stack

        Focus on building consensus and addressing technical constraints.
        """
        
        return self.generate_response(prompt, research_context, max_tokens=Config.DEEPSEEK_STAGE2_MAX_TOKENS)
    
    def continue_discussion(self, user_prompt: str, research_context: str, ollama_response: str) -> LLMMessage:
        """Continue the discussion based on Ollama's latest points"""
        logger.info("DeepSeek continuing discussion")
        
        # Truncate long responses for context
        if len(ollama_response) > 1000:
            ollama_response = ollama_response[:1000] + "... [truncated for discussion]"
        
        prompt = f"""You are DeepSeek having a conversation with Ollama about: {user_prompt}

Ollama just said:
{ollama_response}

Continue this technical discussion by exploring ONE of these design areas in depth:

**Architecture & Design Patterns:**
- System architecture (microservices vs monolith, layers, separation of concerns)
- Design patterns (MVC, Repository, Factory, Observer, etc.)
- Component interactions and dependencies

**Data & Security:**
- Database design (schema, relationships, indexing, migrations)
- Data flow and state management
- Security architecture (authentication, authorization, data protection)
- Input validation and sanitization strategies

**Performance & Scalability:**
- Performance optimization techniques
- Caching strategies (Redis, CDN, application-level)
- Load balancing and horizontal scaling
- Database optimization and query performance

**Development & Operations:**
- Testing strategy (unit, integration, E2E testing)
- CI/CD pipeline design
- Monitoring, logging, and error handling
- Development workflow and tooling

Pick ONE area that needs more discussion based on Ollama's response. Be specific and technical, but keep it conversational (3-4 paragraphs).

Research context: {research_context[:500]}...
"""
        
        return self.generate_response(prompt, research_context, max_tokens=Config.DEEPSEEK_STAGE5_MAX_TOKENS)
    
    def generate_multiple_documents(self,
                                  user_prompt: str,
                                  research_context: str,
                                  conversation_summary: str) -> List[Dict[str, str]]:
        """Generate multiple specialized documents for comprehensive implementation"""
        
        documents = []
        
        # Reduce max tokens per document to avoid timeouts (6000 instead of 8000)
        doc_max_tokens = 6000
        logger.info(f"Generating multiple documents with {doc_max_tokens} max tokens each")
        
        try:
            # Document 1: System Architecture & Technical Specifications
            logger.info("Generating Document 1: System Architecture")
            arch_prompt = f"""Create a COMPREHENSIVE SYSTEM ARCHITECTURE & DESIGN DOCUMENT for: {user_prompt}

RESEARCH CONTEXT: {research_context}
TECHNICAL DISCUSSION: {conversation_summary}

# System Architecture & Technical Specifications

## 1. SYSTEM OVERVIEW & DESIGN PHILOSOPHY
- High-level architecture diagram (textual/ASCII description - NO code)
- Core components and their responsibilities (conceptual design)
- Data flow and communication patterns (architectural patterns)
- Integration points and external dependencies

## 2. DETAILED ARCHITECTURAL DESIGN
- Frontend architecture approach (component strategy, state management philosophy)
- Backend architecture pattern (services design, API structure)
- Database design strategy (schema concepts, relationship patterns)
- API design patterns (RESTful/GraphQL approach, authentication strategy)
- Caching architecture and data flow philosophy

## 3. TECHNOLOGY STACK JUSTIFICATION
- Frontend: Framework choice and WHY (trade-offs, benefits)
- Backend: Technology selection rationale and alternatives considered
- Infrastructure: Hosting, scaling, and deployment strategy
- DevOps: CI/CD approach and tooling philosophy
- Security: Authentication approach and security architecture

## 4. DATA MODELS & SCHEMA DESIGN
- Conceptual data model and entity relationships
- Database design patterns and normalization strategy
- Indexing strategy rationale
- Data validation approach
- Schema evolution and migration strategy

IMPORTANT: Focus on DESIGN and ARCHITECTURE documentation. Include minimal code (pseudocode/small examples ONLY).
This is a planning document explaining WHAT and WHY, not a code implementation guide."""

            arch_doc = self.generate_response(arch_prompt, research_context, temperature=0.3, max_tokens=doc_max_tokens)
            documents.append({
                "title": "System Architecture & Technical Specifications",
                "filename": "01_system_architecture.md",
                "content": arch_doc.content,
                "category": "architecture"
            })
            logger.info(f"Document 1 generated successfully ({len(arch_doc.content)} chars)")
        except Exception as e:
            logger.error(f"Failed to generate System Architecture document: {e}")
            documents.append({
                "title": "System Architecture & Technical Specifications",
                "filename": "01_system_architecture.md",
                "content": f"# System Architecture\n\n*Error generating document: {str(e)}*\n\nPlease retry or check the logs for details.",
                "category": "architecture"
            })

        try:
            # Document 2: Implementation Strategy & Development Roadmap
            logger.info("Generating Document 2: Implementation Guide")
            impl_prompt = f"""Create a DETAILED IMPLEMENTATION STRATEGY & DEVELOPMENT ROADMAP for: {user_prompt}

RESEARCH CONTEXT: {research_context}
TECHNICAL DISCUSSION: {conversation_summary}

# Implementation Strategy & Development Roadmap

## 1. DEVELOPMENT PHASES & MILESTONES
- Phase 1 (Weeks 1-3): Infrastructure setup and foundational architecture
- Phase 2 (Weeks 4-6): Core functionality and business logic development
- Phase 3 (Weeks 7-9): User interface and experience implementation
- Phase 4 (Weeks 10-12): Integration, testing, and optimization
- Phase 5 (Weeks 13-15): Deployment preparation and documentation

## 2. IMPLEMENTATION APPROACH & METHODOLOGY
- Development methodology (Agile/Scrum approach)
- Sprint planning and task breakdown strategy
- Feature prioritization and MVP definition
- Risk mitigation and contingency planning
- Team structure and role definitions

## 3. DEVELOPMENT ENVIRONMENT SETUP STRATEGY
- Environment requirements and tooling needs
- Project structure and organization principles
- Configuration management approach
- Version control strategy and branching model
- Development workflow and collaboration tools

## 4. CODING STANDARDS & QUALITY GUIDELINES
- Code organization philosophy and folder structure rationale
- Naming conventions and style guide principles
- Code review processes and quality assurance approach
- Git workflow and collaboration strategy
- Documentation standards and knowledge management

## 5. TECHNICAL DEBT & MAINTENANCE STRATEGY
- Technical debt identification and management
- Refactoring strategy and timing
- Performance monitoring and optimization approach
- Long-term maintenance and upgrade planning

IMPORTANT: Focus on STRATEGY and PLANNING. Provide setup concepts and approaches, NOT detailed code.
Save implementation details for the actual development phase."""

            impl_doc = self.generate_response(impl_prompt, research_context, temperature=0.3, max_tokens=doc_max_tokens)
            documents.append({
                "title": "Implementation Guide & Development Plan",
                "filename": "02_implementation_guide.md",
                "content": impl_doc.content,
                "category": "implementation"
            })
            logger.info(f"Document 2 generated successfully ({len(impl_doc.content)} chars)")
        except Exception as e:
            logger.error(f"Failed to generate Implementation Guide: {e}")
            documents.append({
                "title": "Implementation Guide & Development Plan",
                "filename": "02_implementation_guide.md",
                "content": f"# Implementation Guide\n\n*Error generating document: {str(e)}*\n\nPlease retry or check the logs for details.",
                "category": "implementation"
            })

        try:
            # Document 3: Security, Testing & Operations
            logger.info("Generating Document 3: Security, Testing & Operations")
            ops_prompt = f"""Create a COMPREHENSIVE SECURITY, TESTING & OPERATIONS GUIDE for: {user_prompt}

RESEARCH CONTEXT: {research_context}
TECHNICAL DISCUSSION: {conversation_summary}

# Security, Testing & Operations Guide

## 1. SECURITY IMPLEMENTATION
- Authentication and authorization implementation
- Data validation and sanitization strategies
- Security headers and CORS configuration
- Encryption for data at rest and in transit
- Security audit checklist and penetration testing

## 2. COMPREHENSIVE TESTING STRATEGY
- Unit testing setup with specific frameworks
- Integration testing for APIs and databases
- End-to-end testing for user workflows
- Performance testing and load testing procedures
- Security testing and vulnerability scanning

## 3. DEPLOYMENT & OPERATIONS
- Infrastructure requirements and specifications
- CI/CD pipeline configuration with specific tools
- Environment setup (dev, staging, production)
- Monitoring, logging, and alerting implementation
- Backup and disaster recovery procedures

## 4. PERFORMANCE & SCALABILITY
- Performance optimization techniques
- Caching strategies with Redis configuration
- Database optimization and query performance
- Load balancing and auto-scaling setup
- Performance monitoring and alerting thresholds

## 5. MAINTENANCE & TROUBLESHOOTING
- Operational runbooks and procedures
- Common issues and troubleshooting guides
- Update and maintenance procedures
- Technical debt management strategies

Include specific configurations, monitoring setup, and operational procedures."""

            ops_doc = self.generate_response(ops_prompt, research_context, temperature=0.3, max_tokens=doc_max_tokens)
            documents.append({
                "title": "Security, Testing & Operations Guide",
                "filename": "03_security_testing_ops.md",
                "content": ops_doc.content,
                "category": "operations"
            })
            logger.info(f"Document 3 generated successfully ({len(ops_doc.content)} chars)")
        except Exception as e:
            logger.error(f"Failed to generate Security & Operations Guide: {e}")
            documents.append({
                "title": "Security, Testing & Operations Guide",
                "filename": "03_security_testing_ops.md",
                "content": f"# Security, Testing & Operations\n\n*Error generating document: {str(e)}*\n\nPlease retry or check the logs for details.",
                "category": "operations"
            })

        # Document 4: API Documentation & Integration Guide (if content is substantial enough)
        if len(conversation_summary) > 20000:  # Only create if there's substantial technical discussion
            try:
                logger.info("Generating Document 4: API Documentation")
                api_prompt = f"""Create an API DOCUMENTATION & INTEGRATION GUIDE for: {user_prompt}

RESEARCH CONTEXT: {research_context}
TECHNICAL DISCUSSION: {conversation_summary}

# API Documentation & Integration Guide

## 1. API SPECIFICATIONS
- Complete API endpoint documentation
- Request/response schemas with examples
- Authentication and authorization requirements
- Error handling and status codes
- Rate limiting and usage policies

## 2. INTEGRATION PATTERNS
- Third-party service integrations
- Webhook implementation and handling
- Event-driven architecture patterns
- Message queue implementation
- External API consumption strategies

## 3. DATA CONTRACTS & SCHEMAS
- JSON schema definitions
- Data validation rules and patterns
- API versioning strategy
- Backward compatibility considerations
- Migration strategies for schema changes

## 4. DEVELOPER RESOURCES
- SDK/client library specifications
- Code examples in multiple languages
- Postman collections and test suites
- Integration testing strategies
- Troubleshooting common integration issues

Include complete API specifications, code examples, and integration patterns."""

                api_doc = self.generate_response(api_prompt, research_context, temperature=0.3, max_tokens=doc_max_tokens)
                documents.append({
                    "title": "API Documentation & Integration Guide",
                    "filename": "04_api_documentation.md",
                    "content": api_doc.content,
                    "category": "api"
                })
                logger.info(f"Document 4 generated successfully ({len(api_doc.content)} chars)")
            except Exception as e:
                logger.error(f"Failed to generate API Documentation: {e}")
                documents.append({
                    "title": "API Documentation & Integration Guide",
                    "filename": "04_api_documentation.md",
                    "content": f"# API Documentation\n\n*Error generating document: {str(e)}*\n\nPlease retry or check the logs for details.",
                    "category": "api"
                })

        logger.info(f"Generated {len(documents)} specialized documents")
        return documents

    def generate_final_plan(self,
                          user_prompt: str,
                          research_context: str,
                          conversation_summary: str) -> LLMMessage:
        """Generate final development plan - now creates multiple documents"""
        
        # Check if we have enough content for multiple documents
        total_content_length = len(research_context) + len(conversation_summary)
        
        if total_content_length > 15000:  # Substantial content - create multiple documents
            logger.info("Generating multiple specialized documents due to comprehensive content")
            
            # Generate multiple documents
            documents = self.generate_multiple_documents(user_prompt, research_context, conversation_summary)
            
            # Create a summary document that references all the specialized documents
            summary_content = f"""# Complete Implementation Documentation Suite

Generated {len(documents)} comprehensive documents for: **{user_prompt}**

## Document Overview:

"""
            for i, doc in enumerate(documents, 1):
                summary_content += f"### {i}. {doc['title']} ({doc['filename']})\n"
                summary_content += f"**Category:** {doc['category'].title()}\n"
                summary_content += f"**Content Length:** {len(doc['content']):,} characters\n\n"

            summary_content += """## Implementation Workflow:

1. **Start with System Architecture** - Review the technical specifications and understand the overall system design
2. **Follow the Implementation Guide** - Use the step-by-step development plan and setup instructions  
3. **Implement Security & Testing** - Follow the security guidelines and set up comprehensive testing
4. **Use API Documentation** - Integrate with external services and implement API endpoints

Each document is designed to be comprehensive and actionable. Download each document for detailed implementation guidance.

## Next Steps:
1. Download all documents
2. Review the system architecture first
3. Set up your development environment using the implementation guide
4. Follow the phased development approach outlined in the documents

*These documents provide everything needed to build a production-ready application from start to finish.*"""

            # Store the documents for download (this will be handled by the file manager)
            return LLMMessage(
                llm_type=LLMType.DEEPSEEK,
                content=summary_content,
                confidence_score=0.9,
                metadata={"documents": documents, "multi_document": True}
            )
        
        else:
            # Fall back to single comprehensive document for smaller projects
            logger.info("Generating single comprehensive document due to limited content")
            
            prompt = f"""
            Create a COMPREHENSIVE, PRODUCTION-READY architectural design document and development plan for: {user_prompt}

            RESEARCH FINDINGS: {research_context}
            TECHNICAL DISCUSSION: {conversation_summary}

            Generate a complete planning and design document covering:
            - System architecture and design philosophy (diagrams and explanations, NOT code)
            - Technology stack with detailed justification and trade-offs
            - Implementation roadmap with phases and milestones
            - Component design and interaction patterns (conceptual, NOT implementations)
            - Security, testing, and deployment strategies
            - API design patterns and integration approaches (NOT actual API code)
            - Operations, monitoring, and maintenance planning

            IMPORTANT: Focus on WHAT to build and WHY. Include minimal code examples ONLY for critical concepts.
            This is planning documentation, not a code repository. Make it comprehensive enough for a team to understand 
            the architecture and plan development, but save detailed implementation for the development phase."""
            
            return self.generate_response(prompt, research_context, temperature=0.3, max_tokens=Config.DEEPSEEK_STAGE9_MAX_TOKENS)
    
    def validate_quality(self, content: str, criteria: List[str]) -> Dict[str, Any]:
        """Validate the quality of generated content against specific criteria"""
        
        criteria_text = "\n".join([f"- {criterion}" for criterion in criteria])
        
        prompt = f"""
        Please evaluate the following content against these quality criteria:

        CONTENT TO EVALUATE:
        {content}

        QUALITY CRITERIA:
        {criteria_text}

        For each criterion, provide:
        - A score from 0.0 to 1.0
        - Specific feedback on strengths and weaknesses
        - Suggestions for improvement

        Return your evaluation in a structured format.
        """
        
        response = self.generate_response(prompt)
        
        # Parse the response to extract quality scores
        # This is a simplified implementation - in practice you'd want more sophisticated parsing
        quality_scores = {}
        
        # Extract scores from response (this would be more sophisticated in production)
        lines = response.content.split('\n')
        for line in lines:
            if 'score' in line.lower() and ':' in line:
                parts = line.split(':')
                if len(parts) >= 2:
                    criterion = parts[0].strip().lower()
                    score_text = parts[1].strip()
                    try:
                        # Extract first number from the score text
                        import re
                        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", score_text)
                        if numbers:
                            quality_scores[criterion] = float(numbers[0])
                    except (ValueError, IndexError):
                        continue
        
        return {
            'overall_score': sum(quality_scores.values()) / len(quality_scores) if quality_scores else 0.0,
            'criterion_scores': quality_scores,
            'feedback': response.content
        }
    
    def generate_research_plan(self, user_prompt: str) -> Dict[str, Any]:
        """Generate comprehensive research queries focused on complete implementation"""
        
        research_prompt = f"""You are a research expert. For the following project request: "{user_prompt}"

Generate 8-12 comprehensive search queries that would gather ALL information needed for complete implementation from start to finish. Focus on:

**CORE IMPLEMENTATION RESEARCH:**
1. Technology stack and framework recommendations with specific versions
2. System architecture patterns and design principles
3. Database design and data modeling best practices
4. API design patterns and implementation examples

**DETAILED IMPLEMENTATION GUIDANCE:**
5. Step-by-step implementation tutorials and code examples
6. Production deployment strategies and infrastructure setup
7. Security implementation and authentication patterns
8. Testing strategies and quality assurance approaches

**OPERATIONAL & MAINTENANCE RESEARCH:**
9. Performance optimization and monitoring solutions
10. Error handling and fault tolerance patterns
11. Scalability considerations and load balancing
12. DevOps practices and CI/CD pipeline setup

**PRACTICAL CONSIDERATIONS:**
13. Common pitfalls and troubleshooting guides
14. Resource requirements and timeline estimation
15. Team structure and development workflow
16. Documentation and maintenance strategies

Return your response as a JSON object with this structure:
{{
    "queries": [
        "specific technical implementation query 1",
        "specific deployment and setup query 2",
        "specific architecture pattern query 3",
        ...
    ],
    "research_focus": "comprehensive implementation and deployment guidance"
}}

Make each query specific, technical, and focused on actionable implementation details. Include queries for setup guides, code examples, production deployment, and operational procedures."""

        response = self.generate_response(research_prompt, max_tokens=Config.DEEPSEEK_STAGE4_MAX_TOKENS)
        
        try:
            # Try to parse JSON from the response
            import json
            import re
            
            content = response.content
            # Look for JSON object in the response
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                research_plan = json.loads(json_str)
                logger.info(f"Successfully parsed research plan with {len(research_plan.get('queries', []))} queries")
                return research_plan
            else:
                logger.warning("No JSON found in research plan response, using fallback")
                
        except Exception as e:
            logger.error(f"Failed to parse research plan JSON: {e}")
        
        # Fallback: extract queries from text
        lines = response.content.split('\n')
        queries = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#') and len(line) > 10:
                # Remove common prefixes and clean up
                line = re.sub(r'^[\d\.\-\*\+\>\s]*', '', line)
                line = line.strip('"\'')
                if line and len(line) > 10:
                    queries.append(line)
        
        return {
            "queries": queries[:8],  # Limit to 8 queries
            "research_focus": "Technical implementation and best practices"
        }
    
    def extract_research_insights(self, user_prompt: str, search_results: List) -> List[Dict[str, Any]]:
        """Extract key insights from search results using LLM analysis"""
        
        # Prepare search results summary
        results_summary = ""
        for i, result in enumerate(search_results[:10], 1):  # Limit to top 10 results
            title = getattr(result, 'title', 'Unknown')
            snippet = getattr(result, 'snippet', 'No description')
            results_summary += f"{i}. {title}\n   {snippet}\n\n"
        
        insight_prompt = f"""Analyze these search results for the request: "{user_prompt}"

SEARCH RESULTS:
{results_summary}

Extract 5-8 key insights that would be most valuable for understanding and implementing this request. Focus on:

1. Technical architecture patterns
2. Implementation strategies  
3. Tool and framework recommendations
4. Performance considerations
5. Common challenges and solutions
6. Best practices and patterns

Return insights as a JSON array:
[
    {{
        "insight": "specific technical insight",
        "source": "which search result(s) this came from",
        "relevance": "why this matters for the request"
    }},
    ...
]

Make insights actionable and technically specific."""

        response = self.generate_response(insight_prompt, max_tokens=Config.DEEPSEEK_STAGE4_MAX_TOKENS)
        
        try:
            # Try to parse JSON array from response
            import json
            import re
            
            content = response.content
            # Look for JSON array in the response
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                insights_data = json.loads(json_str)
                
                # Convert to the expected format
                insights = []
                for item in insights_data:
                    if isinstance(item, dict) and 'insight' in item:
                        insights.append({
                            'content': item['insight'],
                            'source': item.get('source', 'search results'),
                            'relevance_score': 0.8,
                            'type': 'technical_insight'
                        })
                
                logger.info(f"Successfully extracted {len(insights)} insights from search results")
                return insights
                
        except Exception as e:
            logger.error(f"Failed to parse insights JSON: {e}")
        
        # Fallback: extract insights from text
        lines = response.content.split('\n')
        insights = []
        current_insight = ""
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                # Remove list prefixes and clean up
                line = re.sub(r'^[\d\.\-\*\+\>\s]*', '', line)
                line = line.strip('"\'')
                
                if len(line) > 20:  # Reasonable insight length
                    insights.append({
                        'content': line,
                        'source': 'research analysis',
                        'relevance_score': 0.7,
                        'type': 'extracted_insight'
                    })
        
        return insights[:8]  # Limit to 8 insights