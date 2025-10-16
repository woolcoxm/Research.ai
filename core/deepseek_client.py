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
    
    def generate_multiple_documents(self,
                                  user_prompt: str,
                                  research_context: str,
                                  conversation_summary: str) -> List[Dict[str, str]]:
        """Generate multiple specialized documents for comprehensive implementation"""
        
        documents = []
        
        # Increase max tokens for comprehensive implementation guides
        doc_max_tokens = 20000  # Increased from 6000 to allow detailed implementation guidance
        logger.info(f"Generating multiple documents with {doc_max_tokens} max tokens each")
        
        try:
            # Document 1: System Architecture & Implementation Guide
            logger.info("Generating Document 1: System Architecture & Implementation")
            arch_prompt = f"""Create a COMPREHENSIVE SYSTEM ARCHITECTURE & IMPLEMENTATION GUIDE for: {user_prompt}

RESEARCH CONTEXT: {research_context}
TECHNICAL DISCUSSION: {conversation_summary}

# System Architecture & Implementation Guide

## 1. SYSTEM OVERVIEW & DESIGN PHILOSOPHY
- High-level architecture diagram (textual/ASCII description)
- Core components and their responsibilities with implementation approach
- Data flow and communication patterns with code examples
- Integration points and external dependencies with configuration examples
- **Include example project structure with file organization**

## 2. DETAILED ARCHITECTURAL IMPLEMENTATION
- Frontend architecture with complete component examples and state management code
- Backend architecture with service implementations, API endpoints, and routing code
- Database implementation with complete schema definitions, migrations, and seed data
- API implementation with request/response examples, validation code, and error handling
- Caching implementation with Redis/Memcached configuration and code examples
- **Include complete configuration files (package.json, requirements.txt, docker-compose.yml)**

## 3. TECHNOLOGY STACK WITH SETUP INSTRUCTIONS
- Frontend: Framework setup, dependencies, and starter code with explanations
- Backend: Technology installation, project initialization, and configuration files
- Infrastructure: Docker setup, cloud deployment configurations, and scripts
- DevOps: CI/CD pipeline configuration files (GitHub Actions, GitLab CI, Jenkins)
- Security: Authentication implementation with JWT/OAuth code examples
- **Include step-by-step setup procedures and commands**

## 4. DATA MODELS & SCHEMA IMPLEMENTATION
- Complete database schema with CREATE TABLE statements and migration files
- ORM model definitions (Django, SQLAlchemy, Prisma, etc.) with relationships
- Indexing implementation with CREATE INDEX statements and query optimization
- Data validation with Joi/Pydantic/validator code examples
- Schema migration procedures with example migration files
- **Include seed data and test data examples**

## 5. CODE ORGANIZATION & FILE STRUCTURE
- Complete project directory structure with explanations
- Module/package organization patterns
- Configuration management (environment variables, config files)
- Static assets organization (CSS, JS, images)
- **Include example files for each major component**

IMPORTANT: This is a COMPREHENSIVE IMPLEMENTATION GUIDE. Include detailed code examples, complete configuration files, and step-by-step setup procedures. Developers should be able to follow this document to build the entire system from scratch."""

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
            # Document 2: Step-by-Step Implementation Guide
            logger.info("Generating Document 2: Step-by-Step Implementation Guide")
            impl_prompt = f"""Create a DETAILED STEP-BY-STEP IMPLEMENTATION GUIDE for: {user_prompt}

RESEARCH CONTEXT: {research_context}
TECHNICAL DISCUSSION: {conversation_summary}

# Step-by-Step Implementation Guide

## 1. ENVIRONMENT SETUP (Day 1)
### Development Machine Setup
- **Install required tools**: Node.js/Python/Ruby (include version numbers and installation commands)
- **IDE/Editor setup**: VS Code extensions, settings.json configuration
- **Database installation**: PostgreSQL/MySQL/MongoDB setup commands and configuration
- **Docker setup**: Installation and basic configuration
- **Include complete commands for each step**

### Project Initialization
- **Create project structure**: mkdir commands and folder organization
- **Initialize package manager**: npm init, pip setup, bundle init with configurations
- **Set up Git repository**: git init, .gitignore template, initial commit
- **Configure linting and formatting**: ESLint/Prettier/Black configuration files
- **Include all configuration file contents**

## 2. BACKEND IMPLEMENTATION (Weeks 1-4)
### Database Setup
- **Schema creation**: Complete SQL/NoSQL schema definitions with CREATE statements
- **Migration setup**: Migration framework installation and first migration file
- **Seed data**: Test data scripts and population procedures
- **Connection configuration**: Database connection code with error handling
- **Include complete migration files and seed scripts**

### API Development
- **Server setup**: Express/Flask/Rails server initialization code
- **Routing configuration**: Complete route definitions with middleware
- **Controller implementation**: CRUD operations with full code examples
- **Authentication**: JWT/OAuth implementation with complete auth middleware code
- **Error handling**: Centralized error handling middleware with examples
- **Include complete API endpoint implementations**

### Business Logic
- **Service layer**: Business logic classes with full implementations
- **Data validation**: Input validation with Joi/Pydantic/validator examples
- **Database operations**: Repository pattern implementations with queries
- **Transaction handling**: Transaction management code examples
- **Include complete service class implementations**

## 3. FRONTEND IMPLEMENTATION (Weeks 3-6)
### UI Framework Setup
- **React/Vue/Angular setup**: Create-react-app or equivalent with configurations
- **State management**: Redux/Vuex/NgRx setup with store configuration
- **Router configuration**: React-router/Vue-router setup with route definitions
- **API client**: Axios/Fetch configuration with interceptors
- **Include complete setup files and configurations**

### Component Development
- **Component structure**: Reusable component examples with props/state
- **Form handling**: Complete form components with validation
- **Data fetching**: API integration with loading/error states
- **Authentication UI**: Login/signup components with JWT handling
- **Include 5-10 complete component implementations**

### Styling
- **CSS framework setup**: Tailwind/Bootstrap/Material-UI configuration
- **Theme configuration**: Custom theme setup and variables
- **Responsive design**: Media query examples and mobile-first approach
- **Include complete styling configuration and examples**

## 4. TESTING IMPLEMENTATION (Throughout)
### Unit Testing
- **Test framework setup**: Jest/Pytest/RSpec configuration
- **Test structure**: Complete test file examples for each component type
- **Mock data**: Test fixtures and mock implementations
- **Coverage setup**: Coverage reporting configuration
- **Include 10+ complete test examples**

### Integration Testing
- **API testing**: Supertest/Postman/HTTPie examples
- **Database testing**: Test database setup and teardown
- **End-to-end testing**: Cypress/Selenium setup and test examples
- **Include complete integration test suites**

## 5. DEPLOYMENT & OPERATIONS (Weeks 7-8)
### Docker Configuration
- **Dockerfile**: Complete Dockerfile for each service
- **Docker Compose**: Full docker-compose.yml with all services
- **Environment variables**: .env template and configuration
- **Include complete Docker setup**

### CI/CD Pipeline
- **GitHub Actions/GitLab CI**: Complete pipeline configuration files
- **Build scripts**: npm scripts, Makefile, or automation scripts
- **Deployment scripts**: Deploy-to-production procedures
- **Include complete CI/CD configurations**

### Production Setup
- **Cloud deployment**: AWS/GCP/Azure deployment procedures and terraform/CloudFormation
- **Domain and SSL**: DNS configuration and Let's Encrypt setup
- **Monitoring**: New Relic/Datadog setup and alerting configuration
- **Backups**: Automated backup scripts and restoration procedures
- **Include step-by-step deployment commands**

## 6. TROUBLESHOOTING & COMMON ISSUES
- **Database connection errors**: Solutions with code examples
- **CORS issues**: Configuration fixes with examples
- **Authentication problems**: Debugging procedures and fixes
- **Performance bottlenecks**: Optimization techniques with before/after code
- **Include 10-15 common issues with solutions**

IMPORTANT: This should be a COMPLETE TUTORIAL that a developer can follow step-by-step to build the entire system. Include every command, every configuration file, and every code example needed."""

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
            # Document 3: Security, Testing & Operations Implementation
            logger.info("Generating Document 3: Security, Testing & Operations Implementation")
            ops_prompt = f"""Create a COMPREHENSIVE SECURITY, TESTING & OPERATIONS IMPLEMENTATION GUIDE for: {user_prompt}

RESEARCH CONTEXT: {research_context}
TECHNICAL DISCUSSION: {conversation_summary}

# Security, Testing & Operations Implementation Guide

## 1. SECURITY IMPLEMENTATION WITH CODE
### Authentication System
- **JWT Implementation**: Complete authentication middleware with token generation/verification
- **OAuth 2.0 Setup**: Full OAuth flow implementation with provider integration
- **Password Security**: Bcrypt/Argon2 implementation with salting and hashing code
- **Session Management**: Session store setup with Redis and security configurations
- **Include complete authentication module code (200-300 lines)**

### Authorization & Access Control
- **RBAC Implementation**: Role-based access control with complete middleware
- **Permission System**: Permission checking decorators and guards with examples
- **API Key Management**: API key generation, storage, and validation code
- **Include complete authorization system (150-200 lines)**

### Security Configurations
- **Security Headers**: Helmet.js/Flask-Talisman configuration with all headers
- **CORS Setup**: Complete CORS configuration with whitelist management
- **Rate Limiting**: Express-rate-limit/Flask-Limiter with Redis store
- **SQL Injection Prevention**: Parameterized queries and ORM usage examples
- **XSS Protection**: Input sanitization and output encoding code
- **CSRF Protection**: CSRF token implementation with validation
- **Include 10+ security configuration files**

### Encryption Implementation
- **Data at Rest**: Database encryption setup with AWS KMS/Azure Key Vault
- **Data in Transit**: TLS/SSL configuration for HTTPS enforcement
- **File Encryption**: File upload encryption with crypto libraries
- **Environment Variables**: Secure secrets management with HashiCorp Vault
- **Include complete encryption utilities (100-150 lines)**

## 2. COMPREHENSIVE TESTING WITH EXAMPLES
### Unit Testing Suite
- **Test Framework Setup**: Jest/Pytest/RSpec configuration files
- **Test Structure**: 15-20 complete unit test examples for:
  - Controller tests with mocked dependencies
  - Service layer tests with test data
  - Utility function tests with edge cases
  - Database model tests with fixtures
- **Mocking Examples**: Complete mock implementations for external services
- **Test Coverage**: Istanbul/Coverage.py configuration for 80%+ coverage
- **Include 500-1000 lines of test code examples**

### Integration Testing
- **API Integration Tests**: Complete test suite for all endpoints
  - Authentication flow tests (login, logout, token refresh)
  - CRUD operation tests for each resource
  - Error handling tests (400, 401, 403, 404, 500)
  - File upload/download tests
- **Database Integration Tests**: Transaction tests, rollback tests, migration tests
- **Third-party Integration Tests**: Mock external APIs with nock/responses
- **Include 300-500 lines of integration test code**

### End-to-End Testing
- **Cypress/Selenium Setup**: Complete E2E test configuration
- **User Flow Tests**: 10-15 complete test scenarios:
  - User registration and login flow
  - Main application workflows
  - Error recovery scenarios
  - Mobile responsive tests
- **Visual Regression Testing**: Percy/BackstopJS setup and configuration
- **Include complete E2E test suite (400-600 lines)**

### Performance & Load Testing
- **Load Testing Scripts**: K6/Locust scripts for API endpoints
- **Performance Benchmarks**: Baseline performance metrics and thresholds
- **Database Query Profiling**: EXPLAIN query analysis and optimization
- **Frontend Performance**: Lighthouse CI configuration and thresholds
- **Include complete load testing scripts (200-300 lines)**

## 3. DEPLOYMENT & OPERATIONS IMPLEMENTATIONS
### Docker & Container Setup
- **Complete Dockerfiles**: Production-ready Dockerfiles for all services (frontend, backend, database)
- **Docker Compose**: Full docker-compose.yml with networking, volumes, health checks
- **Multi-stage Builds**: Optimized build process for smaller images
- **Docker Networking**: Service discovery and inter-container communication
- **Include all Docker files (300-400 lines total)**

### CI/CD Pipeline Implementation
- **GitHub Actions**: Complete workflow files for:
  - Pull request checks (lint, test, build)
  - Automated deployment to staging/production
  - Security scanning (Snyk, Trivy)
  - Performance testing
- **GitLab CI/Jenkins**: Alternative pipeline configurations
- **Build Scripts**: Package.json scripts, Makefiles, deployment scripts
- **Include complete CI/CD configurations (400-500 lines)**

### Cloud Infrastructure
- **Kubernetes Manifests**: Complete K8s deployments, services, ingress, configmaps
- **Terraform/Pulumi**: Infrastructure as Code for AWS/GCP/Azure
- **Helm Charts**: Complete Helm chart for application deployment
- **Cloud Functions**: Serverless function examples for background tasks
- **Include complete infrastructure code (500-800 lines)**

### Monitoring & Logging
- **Application Logging**: Winston/Structlog setup with log levels and formatting
- **Centralized Logging**: ELK Stack/Loki configuration with log aggregation
- **Application Monitoring**: Prometheus metrics endpoints and custom metrics
- **Distributed Tracing**: Jaeger/Zipkin integration with trace context
- **Alerting Rules**: Prometheus AlertManager rules for critical issues
- **Dashboard Configuration**: Grafana dashboards JSON export
- **Include complete monitoring setup (300-500 lines)**

### Database Operations
- **Backup Scripts**: Automated database backup scripts (pg_dump, mongodump)
- **Restore Procedures**: Complete restoration procedures with commands
- **Migration Scripts**: Database migration files with rollback support
- **Replication Setup**: Primary-replica configuration for PostgreSQL/MySQL
- **Include complete database operational scripts (200-300 lines)**

## 4. PERFORMANCE OPTIMIZATION IMPLEMENTATIONS
### Caching Implementation
- **Redis Setup**: Complete Redis configuration and connection pooling
- **Cache Strategies**: Cache-aside, write-through implementations with code
- **Cache Invalidation**: Cache invalidation patterns and implementations
- **API Response Caching**: HTTP caching headers and CDN configuration
- **Include complete caching layer (200-300 lines)**

### Database Optimization
- **Index Creation**: Optimal index definitions with analysis
- **Query Optimization**: Before/after examples of slow queries
- **Connection Pooling**: PgBouncer/connection pool configuration
- **N+1 Query Solutions**: Eager loading and join optimization examples
- **Include 20+ optimization examples**

### Application Performance
- **Code Profiling**: Profiling setup and hotspot identification
- **Async/Await Optimization**: Concurrent request handling examples
- **Memory Leak Detection**: Memory profiling and leak prevention
- **Bundle Optimization**: Webpack/Vite configuration for smaller bundles
- **Include optimization code examples (300-400 lines)**

## 5. OPERATIONAL RUNBOOKS & PROCEDURES
### Production Runbooks
- **Deployment Procedure**: Step-by-step deployment commands with rollback
- **Incident Response**: Incident handling procedures with commands
- **Database Maintenance**: Vacuum, analyze, reindex procedures
- **Certificate Renewal**: SSL certificate renewal with Let's Encrypt
- **Scaling Procedures**: Horizontal and vertical scaling commands
- **Include 10-15 complete runbooks**

### Troubleshooting Guide
- **Common Issues**: 25+ issues with symptoms, diagnosis, and solutions
  - "500 Internal Server Error" - debugging steps with log analysis
  - "Database connection refused" - connection troubleshooting
  - "High memory usage" - memory profiling and optimization
  - "Slow API responses" - performance profiling steps
- **Debug Commands**: Complete command-line debugging toolkit
- **Log Analysis**: Log parsing commands and patterns to search for
- **Include complete troubleshooting playbook**

IMPORTANT: This should be a PRODUCTION-READY operations guide with every security configuration, test file, deployment script, and monitoring setup needed to run the system in production. Include complete, runnable code for everything."""

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
                logger.info("Generating Document 4: API Documentation & Integration Implementation")
                api_prompt = """Create a COMPLETE API DOCUMENTATION & INTEGRATION IMPLEMENTATION GUIDE for: {prompt}

RESEARCH CONTEXT: {context}
TECHNICAL DISCUSSION: {discussion}

RESEARCH CONTEXT: {research_context}
TECHNICAL DISCUSSION: {conversation_summary}

# API Documentation & Integration Implementation Guide

## 1. COMPLETE API ENDPOINT SPECIFICATIONS
For EVERY API endpoint, provide:

### Authentication Endpoints
**POST /api/auth/register**
- **Description**: User registration endpoint
- **Request Headers**: Content-Type: application/json
- **Request Body Schema**:
  ```json
  {{
    "email": "string (required, email format)",
    "password": "string (required, min 8 chars)",
    "name": "string (required)"
  }}
  ```
- **Success Response (201)**:
  ```json
  {{
    "user": {{"id": 123, "email": "user@example.com", "name": "John Doe"}},
    "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
  }}
  ```
- **Error Responses**:
  - 400: {{"error": "Email already exists"}}
  - 422: {{"error": "Invalid email format"}}
- **cURL Example**:
  ```bash
  curl -X POST https://api.example.com/auth/register \
    -H "Content-Type: application/json" \
    -d '{{"email":"user@example.com","password":"secure123","name":"John Doe"}}'
  ```
- **JavaScript Example**:
  ```javascript
  const response = await fetch('/api/auth/register', {{
    method: 'POST',
    headers: {{'Content-Type': 'application/json'}},
    body: JSON.stringify({{email, password, name}})
  }});
  const data = await response.json();
  ```
- **Python Example**:
  ```python
  response = requests.post('https://api.example.com/auth/register',
    json={{"email": "user@example.com", "password": "secure123", "name": "John Doe"}})
  data = response.json()
  ```

**Repeat this detailed format for ALL endpoints** (15-30 endpoints total):
- POST /api/auth/login
- POST /api/auth/logout
- POST /api/auth/refresh
- GET /api/users/me
- PUT /api/users/:id
- DELETE /api/users/:id
- GET /api/resources (with pagination, filtering, sorting)
- POST /api/resources
- GET /api/resources/:id
- PUT /api/resources/:id
- PATCH /api/resources/:id
- DELETE /api/resources/:id
- Plus any domain-specific endpoints

## 2. API CLIENT IMPLEMENTATIONS
### Complete JavaScript/TypeScript SDK
```typescript
// Complete 200-300 line SDK implementation
class ApiClient {{
  private baseUrl: string;
  private token: string | null = null;

  constructor(baseUrl: string) {{
    this.baseUrl = baseUrl;
  }}

  async register(email: string, password: string, name: string) {{
    // Complete implementation with error handling
  }}

  async login(email: string, password: string) {{
    // Complete implementation
  }}

  async getResource(id: number) {{
    // Complete implementation
  }}

  // Include ALL methods for ALL endpoints
}}
```

### Complete Python SDK
```python
# Complete 200-300 line Python client
class ApiClient:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.token = None
    
    def register(self, email: str, password: str, name: str) -> dict:
        # Complete implementation
        pass
    
    # Include ALL methods for ALL endpoints
```

### Complete Go SDK
```go
// Complete 200-300 line Go client
package apiclient

type Client struct {{
    BaseURL string
    Token   string
}}

func NewClient(baseURL string) *Client {{
    // Complete implementation
}}

// Include ALL methods for ALL endpoints
```

## 3. THIRD-PARTY INTEGRATIONS WITH CODE
### Stripe Payment Integration
- **Setup Code**: Complete Stripe SDK initialization
- **Payment Intent Creation**: Full implementation with error handling
- **Webhook Handler**: Complete webhook endpoint with signature verification
- **Test Mode Setup**: Test card numbers and testing procedures
- **Include 300-500 lines of integration code**

### AWS S3 File Upload Integration
- **S3 Client Setup**: Complete AWS SDK configuration
- **Pre-signed URL Generation**: Full implementation for secure uploads
- **Direct Upload Code**: Client-side and server-side upload examples
- **File Processing**: Image resizing, format conversion examples
- **Include 200-400 lines of integration code**

### SendGrid/Mailgun Email Integration
- **Email Client Setup**: Complete SDK initialization
- **Template System**: Email template examples (HTML + text)
- **Transactional Emails**: Order confirmation, password reset implementations
- **Bulk Email**: Newsletter sending with batching
- **Include 150-250 lines of email integration code**

### Twilio SMS/WhatsApp Integration
- **Twilio Setup**: Complete client initialization
- **SMS Sending**: SMS notification implementations
- **Two-Factor Authentication**: Implementation with Twilio Verify
- **WhatsApp Business**: WhatsApp message templates and sending
- **Include 150-250 lines of Twilio integration code**

### Redis Pub/Sub for Real-time Features
- **Redis Setup**: Pub/sub client configuration
- **Event Publishing**: Event emission code for different events
- **Event Subscription**: Event listener implementations
- **WebSocket Integration**: Socket.io with Redis adapter
- **Include 200-300 lines of pub/sub code**

## 4. DATA SCHEMAS & VALIDATION
### Complete JSON Schemas
```json
// User Schema
{{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {{
    "id": {{"type": "integer"}},
    "email": {{"type": "string", "format": "email"}},
    "name": {{"type": "string", "minLength": 1, "maxLength": 100}},
    "createdAt": {{"type": "string", "format": "date-time"}}
  }},
  "required": ["id", "email", "name", "createdAt"]
}}

// Include schemas for ALL data models (10-20 schemas)
```

### Validation Implementations
- **Joi Validation (Node.js)**: Complete validation schemas for all endpoints
- **Pydantic Models (Python)**: Complete Pydantic models with validators
- **Zod Schemas (TypeScript)**: Complete type-safe validation schemas
- **Include 200-400 lines of validation code**

## 5. OPENAPI/SWAGGER SPECIFICATION
```yaml
# Complete 500-1000 line OpenAPI 3.0 specification
openapi: 3.0.0
info:
  title: Project API
  version: 1.0.0
  description: Complete API documentation

servers:
  - url: https://api.example.com/v1
    description: Production server
  - url: http://localhost:3000/v1
    description: Development server

components:
  securitySchemes:
    BearerAuth:
      type: http
      scheme: bearer
      bearerFormat: JWT
  
  schemas:
    # Include ALL schemas
  
paths:
  # Include ALL endpoints with complete specs
  /auth/register:
    post:
      summary: Register new user
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/RegisterRequest'
      responses:
        '201':
          description: User created successfully
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/AuthResponse'
  
  # Complete specification for ALL endpoints
```

## 6. POSTMAN COLLECTION & TESTING
- **Complete Postman Collection**: JSON export with all endpoints and examples
- **Environment Variables**: Dev, staging, production environment configs
- **Pre-request Scripts**: Auth token refresh scripts
- **Test Scripts**: Automated test assertions for each endpoint
- **Include complete Postman collection (500-800 lines JSON)**

## 7. WEBHOOK IMPLEMENTATIONS
### Webhook Endpoint
```javascript
// Complete webhook receiver implementation
app.post('/webhooks/stripe', async (req, res) => {{
  const sig = req.headers['stripe-signature'];
  // Complete implementation with:
  // - Signature verification
  // - Event handling for all event types
  // - Idempotency handling
  // - Error handling and retry logic
  // 150-250 lines
}});
```

### Webhook Sending (Outbound)
- **Webhook Registration API**: CRUD endpoints for webhook subscriptions
- **Event Queue**: Message queue implementation for reliable delivery
- **Retry Logic**: Exponential backoff retry implementation
- **Webhook Security**: HMAC signature generation code
- **Include complete webhook system (400-600 lines)**

## 8. RATE LIMITING & QUOTAS
- **Rate Limit Implementation**: Complete rate limiting middleware with Redis
- **Quota Tracking**: API usage tracking and billing code
- **Rate Limit Headers**: X-RateLimit-* header implementation
- **Quota Exceeded Handling**: 429 response handling examples
- **Include complete rate limiting system (200-300 lines)**

## 9. API VERSIONING IMPLEMENTATION
- **URL Versioning**: /v1, /v2 routing implementation
- **Header Versioning**: Accept-Version header handling
- **Deprecation Strategy**: Sunset header and migration guides
- **Version Migration**: Code examples for breaking changes
- **Include versioning system (150-250 lines)**

## 10. ERROR HANDLING & DEBUGGING
- **Error Response Format**: Standardized error response structure
- **Error Codes**: Complete list of application error codes
- **Debug Mode**: Request ID tracking and debug logging
- **API Health Check**: /health endpoint implementation
- **Include error handling utilities (200-300 lines)**

IMPORTANT: This should be a COMPLETE, PRODUCTION-READY API documentation that includes every single endpoint specification, complete SDK implementations in 3+ languages, full integration code for all third-party services, and everything a developer needs to integrate with the API."""
                
                # Format the template with actual values
                api_prompt = api_prompt.format(prompt=user_prompt, context=research_context, discussion=conversation_summary)

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