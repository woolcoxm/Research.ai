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
                    f"You are an expert AI researcher, senior software architect, and implementation specialist. Use the following research context to inform your analysis:\n\n{context}\n\n"
                    "When answering, provide COMPREHENSIVE IMPLEMENTATION GUIDANCE:\n"
                    "1) Reference specific search results or key insights provided in context and cite sources in [Source: URL] format.\n"
                    "2) Provide BOTH architecture AND detailed implementation guidance: system design, code examples, configuration files, and setup procedures.\n"
                    "3) Include DETAILED CODE EXAMPLES with complete implementations - not just pseudocode or snippets. Show actual working code.\n"
                    "4) When making recommendations, explain trade-offs, design decisions, WHY certain approaches are chosen, AND HOW to implement them.\n"
                    "5) Include step-by-step implementation procedures, configuration examples, and troubleshooting guidance.\n"
                    "6) Provide PRODUCTION-READY implementation guidance - what to build, why, how to build it, and how to deploy it.\n"
                    "7) Include complete file structures, configuration files, dependency lists, and deployment scripts.\n"
                    "8) Add code comments, error handling examples, and best practices for each implementation.\n"
                    "Provide comprehensive documentation suitable for developers to take a project from start to finish."
                )
            })
        else:
            messages.append({
                "role": "system",
                "content": "You are an expert AI researcher, software architect, and implementation specialist. Provide detailed architectural analysis, design decisions, AND comprehensive implementation guidance with complete code examples, configurations, and step-by-step procedures."
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
    
    # ==================== NEW: OPTIMIZED DOCUMENT WORKFLOW ====================
    
    def create_document_outline(self, 
                               doc_number: int,
                               doc_type: str,
                               user_prompt: str,
                               research_context: str,
                               conversation_summary: str) -> LLMMessage:
        """Create COMPREHENSIVE outline for PRODUCTION-READY documentation (8K detailed outline)"""
        
        # Build a clean prompt without embedded code to avoid Python syntax issues
        prompt = f"""Create a MASSIVELY DETAILED OUTLINE for Document #{doc_number}: {doc_type}

PROJECT: {user_prompt}

RESEARCH CONTEXT (100+ sources, 128K tokens analyzed):
{research_context[:15000]}

TECHNICAL DISCUSSION:
{conversation_summary[:8000]}


YOUR MISSION: Create a COMPREHENSIVE, PRODUCTION-LEVEL outline that will guide another LLM to write a 20,000-30,000+ word, ENTERPRISE-GRADE implementation guide covering the COMPLETE SOFTWARE DEVELOPMENT LIFECYCLE.

OUTLINE STRUCTURE (Use ALL 8,000 tokens):

# {doc_type}

## DOCUMENT OVERVIEW
**Scope**: Complete production system development from requirements to operations
**Target Length**: 20,000-30,000+ words
**Target Code**: 3,000-5,000+ lines
**Configuration Files**: 30-50+ complete files
**Development Timeline**: 6-8 weeks with detailed breakdown

## PHASE 1: REQUIREMENTS & PLANNING (Week 1)
**Target**: 2,000-3,000 words | Code: 0 lines | Duration: Week 1

### 1.1 Requirements Gathering & Documentation
**Key Points to Cover** (500-800 words):
- 30-50 functional requirements with detailed acceptance criteria
- 20-30 non-functional requirements (performance, security, scalability)
- 20-30 user stories in "As a [role], I want [feature], so that [benefit]" format
- 10-15 use case diagrams (textual/Mermaid descriptions)
- Complete business rules documentation
- Stakeholder analysis and communication plan

**Deliverables to Include**:
- Complete requirements specification document template
- User story mapping examples
- Acceptance criteria checklist
- Requirements traceability matrix

**Research Citations from Context**:
[Pull 5-10 specific facts from research about requirements gathering best practices]

### 1.2 System Architecture Design
**Key Points to Cover** (800-1,200 words):
- High-level architecture diagrams (textual/ASCII/Mermaid)
- Component breakdown (15-30 components with responsibilities)
- Data flow diagrams with complete request/response cycles
- Integration architecture (external APIs, services, dependencies)
- Scalability design (load balancing, caching, database sharding)
- Security architecture (authentication, authorization, encryption)
- Technology stack decisions with detailed justifications

**Deliverables to Include**:
- Architecture Decision Records (ADRs) template with 5-10 examples
- Complete technology comparison matrix
- System context diagrams
- Component interaction diagrams

**Code Examples Needed**: None (planning phase)

### 1.3 Project Structure & File Organization
**Key Points to Cover** (400-600 words):
- Complete folder structure (show 100+ files and folders)
- Frontend structure (20-30 component files, 20-30 pages)
- Backend structure (15-20 services, 15-20 controllers, 15-30 models)
- Test structure (unit, integration, e2e folders)
- Configuration folder structure
- Documentation folder structure

**Deliverables to Include**:
```
Complete ASCII tree structure showing:
/project-root
  /docs (5-10 files)
  /frontend
    /src
      /components (list 20-30 components)
      /pages (list 20-30 pages)
      /services (list 10-15 services)
      ... (complete structure)
  /backend
    /src
      /controllers (list 15-20 controllers)
      /services (list 15-20 services)
      /models (list 15-30 models)
      ... (complete structure)
  /tests (complete test structure)
  /infrastructure (Docker, K8s, Terraform)
  /scripts (build, deploy, migrate)
```

## PHASE 2: ENVIRONMENT SETUP & INITIALIZATION (Week 1, Days 1-5)
**Target**: 3,000-4,000 words | Code: 500-1,000 lines | Duration: 5 days

### 2.1 Day 1: Development Machine Setup
**Key Points to Cover** (600-900 words):
- Operating system setup (Windows/Mac/Linux)
- IDE/Editor installation and configuration (VS Code with 30+ extensions)
- Programming language/runtime installation (exact versions)
- Package manager setup and configuration
- Database installation (PostgreSQL/MySQL/MongoDB)
- Cache server installation (Redis, Memcached)
- Message queue setup (RabbitMQ, Kafka) if needed
- Docker Desktop installation and configuration
- Git configuration (user, SSH keys, GPG signing)

**Complete Commands Needed** (200-300 lines):
```bash
# EVERY installation command for Windows/Mac/Linux
# Node.js installation
choco install nodejs-lts --version=18.17.0  # Windows
brew install node@18  # Mac
# ... complete commands for ALL tools

# VS Code extensions (list 30+)
code --install-extension dbaeumer.vscode-eslint
code --install-extension esbenp.prettier-vscode
# ... ALL extensions

# PostgreSQL setup
# ... complete installation and initialization

# Redis setup  
# ... complete installation and configuration

# Docker setup
# ... complete installation

# Git configuration
git config --global user.name "Your Name"
# ... ALL git configurations
```

**Configuration Files Needed** (300+ lines total):
1. VS Code settings.json (100+ lines with ALL settings)
2. .gitconfig (30+ lines with aliases and settings)
3. Terminal profile configurations
4. Environment PATH setup

**Verification Steps**:
- Commands to verify each tool is installed
- Version check commands
- Connectivity tests

**Troubleshooting** (list 10-15 common issues):
- "Node not found" → Solution
- "PostgreSQL won't start" → Solution
- ... (10-15 issues with complete solutions)

### 2.2 Day 2: Project Initialization & Configuration
**Key Points to Cover** (800-1,200 words):
- Project folder creation (complete mkdir commands)
- Git repository initialization
- Package manager initialization (npm/pip/maven)
- Dependency installation (list ALL 50-100 dependencies)
- Configuration file creation (10-15 config files)
- Environment variable setup
- Initial commit and push

**Complete Commands Needed** (300-500 lines):
```bash
# Create project structure
mkdir -p backend/src/controllers backend/src/services backend/src/models backend/src/middleware
# ... EVERY folder creation command (50+ folders)

# Initialize package managers
npm init -y
# ... complete initialization

# Install ALL dependencies
npm install express@4.18.2 cors@2.8.5 helmet@7.0.0 ...
# List ALL 50-100 dependencies with exact versions
```

**Configuration Files Needed - COMPLETE CONTENTS** (1,000-2,000 lines total):
1. **package.json** - COMPLETE with all scripts, dependencies (200-300 lines)
2. **tsconfig.json** or equivalent - COMPLETE configuration (100+ lines)
3. **.env.example** - ALL 50-100 environment variables with descriptions
4. **.eslintrc.js** - COMPLETE linting rules (100+ lines)
5. **.prettierrc** - Complete code formatting config
6. **docker-compose.yml** - ALL services configured (200-300 lines)
7. **Dockerfile** - Multi-stage production build (80-120 lines)
8. **.dockerignore** - Complete ignore patterns
9. **.gitignore** - ALL patterns for Node/Python/Java
10. **README.md** - Complete project documentation (500+ words)
11. **jest.config.js** or equivalent - Complete test configuration
12. **.editorconfig** - Editor consistency
13. **renovate.json** or dependabot.yml - Dependency management
14. **LICENSE** - Complete license file
15. **CONTRIBUTING.md** - Contribution guidelines

For EACH config file, specify:
- Complete file contents (no placeholders!)
- Purpose of each section
- Customization points

### 2.3 Days 3-5: Database Design & Implementation
[Continue with similar extreme detail for Days 3-5]

## PHASE 3: DATABASE IMPLEMENTATION (Week 2)
**Target**: 3,000-4,000 words | Code: 2,000-3,000 lines | Duration: Week 2

### 3.1 Complete Database Schema Design
**Key Points to Cover** (800-1,200 words):
- Entity relationship design (15-30 tables)
- Normalization strategy (3NF minimum)
- Indexing strategy (20-50+ indices)
- Constraint design (foreign keys, check constraints, unique)
- Data types and size considerations
- Migration strategy
- Seed data strategy

**Complete SQL Schema Needed** (1,500-2,500 lines):
```sql
-- Create database
CREATE DATABASE myapp_production;

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- Users table - COMPLETE
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    first_name VARCHAR(100) NOT NULL,
    last_name VARCHAR(100) NOT NULL,
    ... [EVERY column - 15-25 columns]
);

-- Create ALL indices
CREATE INDEX idx_users_email ON users(email) WHERE deleted_at IS NULL;
CREATE INDEX idx_users_role ON users(role);
... [10-20 indices per major table]

-- Repeat COMPLETE structure for ALL 15-30 tables:
-- users, user_sessions, posts, comments, categories,
-- likes, notifications, activity_logs, file_uploads,
-- orders, products, inventory, invoices, payments,
-- messages, conversations, etc.
[EVERY table with EVERY column, constraint, index]
```

**Migration Files Needed** (500-1,000 lines):
1. 001_initial_schema.sql - Complete initial tables
2. 002_add_indexes.sql - All performance indexes
3. 003_add_triggers.sql - Audit and automation triggers
4. 004_seed_reference_data.sql - Reference data
5. 005_add_full_text_search.sql - Search optimization

### 3.2 ORM Models Implementation
**Key Points to Cover** (600-800 words):
- Model design patterns
- Relationship definitions (one-to-many, many-to-many)
- Validation at model level
- Virtual fields and computed properties
- Model methods and helpers
- Serialization and deserialization

**Complete Model Code Needed** (1,500-2,500 lines):
```
COMPLETE User model (200-400 lines) with:
- Entity decorators and table mapping
- All field definitions with proper types
- All relationships (one-to-many, many-to-many)
- All methods (setPassword, verifyPassword, etc.)
- Complete validation logic
- Serialization methods

Repeat COMPLETE models for ALL 15-30 entities:
UserSession, Post, Comment, Category, Like, Notification, 
ActivityLog, FileUpload, Order, Product, etc.
Each model should be 100-400 lines, fully implemented
```

## PHASE 4: BACKEND IMPLEMENTATION (Weeks 2-4)
**Target**: 5,000-8,000 words | Code: 10,000-15,000 lines | Duration: 2-3 weeks

[Continue with same extreme detail level for:
- Server setup and configuration
- Middleware stack
- Authentication system (complete JWT + OAuth implementations)
- Authorization and RBAC
- API endpoints for ALL resources
- Service layer implementations
- Validation schemas
- Error handling
- Logging and monitoring setup]

## PHASE 5: FRONTEND IMPLEMENTATION (Weeks 3-5)
**Target**: 5,000-8,000 words | Code: 10,000-15,000 lines | Duration: 2-3 weeks

[Continue with extreme detail for:
- Framework setup
- Component library (30-50 components)
- Pages and views (20-30 pages)
- State management
- API client implementation
- Form handling
- Routing and navigation]

## PHASE 6: TESTING IMPLEMENTATION (Week 5-6)
**Target**: 3,000-5,000 words | Code: 5,000-10,000 lines | Duration: 1-2 weeks

[Extreme detail for all testing phases]

## PHASE 7: SECURITY IMPLEMENTATION (Throughout)
**Target**: 2,000-3,000 words | Code: 1,000-2,000 lines

[Complete security implementations]

## PHASE 8: DEVOPS & DEPLOYMENT (Week 6-7)
**Target**: 3,000-5,000 words | Code: 2,000-3,000 lines | Duration: 1-2 weeks

[Complete DevOps setup]

## PHASE 9: OPERATIONS & MAINTENANCE (Ongoing)
**Target**: 2,000-3,000 words | Code: 1,000-2,000 lines

[Operational procedures and runbooks]

## PHASE 10: DOCUMENTATION & HANDOFF (Week 7-8)
**Target**: 1,500-2,500 words | Code: 1,000-2,000 lines (OpenAPI specs)

[Complete documentation]

## 📊 OUTLINE METADATA
**Total Phases**: 10
**Total Sections**: 40-60
**Total Subsections**: 100-150
**Expected Final Document**:
- Word Count: 20,000-30,000+ words
- Code Lines: 3,000-5,000+ lines
- Configuration Files: 30-50+ complete files
- Code Examples: 50-100+ complete examples
- Troubleshooting Scenarios: 50+ issues with solutions

**Key Technologies from Research**:
[List 20-30 technologies identified in research]

**Critical Success Factors**:
- Complete, copy-pasteable code throughout
- Production-ready implementations only
- Full development lifecycle coverage
- Operational procedures included
- Comprehensive troubleshooting

🚨 CRITICAL: This outline must be SO DETAILED that an LLM can write a 20,000-30,000 word production-ready guide by following it. Include specific facts, exact code requirements, complete file lists, and comprehensive coverage of the entire development lifecycle from day 1 to production operations."""

        return self.generate_response(prompt, research_context, temperature=0.3, max_tokens=Config.DEEPSEEK_OUTLINE_MAX_TOKENS)
    
    def create_research_summary(self,
                               doc_type: str,
                               research_context: str,
                               user_prompt: str) -> LLMMessage:
        """Compress 128K research into 2-3K summary relevant to specific document"""
        
        prompt = f"""From the extensive research below, extract ONLY information relevant to: {doc_type}

PROJECT: {user_prompt}

FULL RESEARCH (100+ sources):
{research_context[:20000]}

YOUR TASK: Create a 2,000-3,000 token summary containing ONLY what's needed for writing this specific document type.

EXTRACT:
1. **Key Technical Facts**: Specific facts, best practices, recommendations
2. **Code Examples**: Relevant code patterns and examples
3. **Tool Recommendations**: Specific tools, libraries, frameworks mentioned
4. **Common Pitfalls**: Issues to avoid, debugging tips
5. **Configuration Details**: Setup steps, configuration examples
6. **Performance Tips**: Optimization recommendations
7. **Security Considerations**: Security best practices
8. **Source URLs**: Keep URLs for citations

OMIT:
- General/irrelevant information
- Duplicate information
- Off-topic content

OUTPUT FORMAT:
## Technical Facts
- Fact 1 [Source: URL]
- Fact 2 [Source: URL]

## Code Patterns
- Pattern 1: [description]
- Pattern 2: [description]

## Tools & Libraries
- Tool 1: [why and how to use]
- Tool 2: [why and how to use]

## Best Practices
- Practice 1: [description]
- Practice 2: [description]

This summary will be given to Ollama (24K context) for document writing."""

        return self.generate_response(prompt, research_context, temperature=0.2, max_tokens=3000)
    
    def review_document_accuracy(self,
                                document_content: str,
                                original_outline: str,
                                research_context: str,
                                doc_title: str) -> LLMMessage:
        """Review Ollama's document against research for technical accuracy"""
        
        prompt = f"""Review this implementation guide for TECHNICAL ACCURACY and COMPLETENESS.

DOCUMENT: {doc_title}

ORIGINAL OUTLINE:
{original_outline[:3000]}

DOCUMENT CONTENT (written by Ollama):
{document_content[:6000]}

RESEARCH CONTEXT (for verification):
{research_context[:10000]}

YOUR REVIEW CHECKLIST:
1. **Outline Adherence**: Did it cover all sections from outline?
2. **Technical Accuracy**: Are technical facts correct per research?
3. **Code Quality**: Are code examples correct and complete?
4. **Citations**: Are sources properly cited?
5. **Completeness**: Any missing subsections or examples?
6. **Configuration Files**: Are all needed configs included in full?

PROVIDE:
1. **Overall Assessment**: APPROVED or NEEDS REVISION
2. **Technical Corrections**: List any incorrect facts/code
3. **Missing Content**: What sections/examples are incomplete
4. **Citation Issues**: Missing or incorrect source citations
5. **Specific Improvements**: 3-5 concrete additions/fixes needed

Keep response under 5,000 tokens. Be specific and actionable."""

        return self.generate_response(prompt, research_context, temperature=0.3, max_tokens=5000)
    
    def generate_multiple_documents(self,
                                  user_prompt: str,
                                  research_context: str,
                                  conversation_summary: str) -> List[Dict[str, str]]:
        """Generate multiple specialized documents for comprehensive implementation"""
        
        documents = []
        
        # MASSIVE increase in max tokens for COMPREHENSIVE, PRODUCTION-READY documentation
        # These documents must be complete enough to build entire systems from scratch
        doc_max_tokens = Config.DEEPSEEK_COMPREHENSIVE_DOC_MAX_TOKENS  # 64K tokens per document
        logger.info(f"Generating comprehensive production-ready documents with {doc_max_tokens} max tokens each")
        
        try:
            # Document 1: System Architecture & Implementation Guide
            logger.info("Generating Document 1: System Architecture & Implementation")
            # Simple placeholder prompts - details moved to Ollama
            arch_prompt = 'Create comprehensive system architecture guide for: ' + user_prompt

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
