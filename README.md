# AI Research System

An advanced automated research system where DeepSeek API (128K context) and local Ollama (qwen3-coder) collaborate through an 11-stage iterative workflow to conduct comprehensive research, analyze 100-200 sources, and generate specialized documentation.

## ✨ Features

- � **Dual LLM Collaboration**: DeepSeek (deep analysis) and Ollama (critical review) work together through structured discussions
- 🔍 **Comprehensive Research**: Automated web searches analyzing 100-200 sources using Serper.dev API
- 📊 **11-Stage Iterative Workflow**: Structured pipeline maximizing DeepSeek's 128K context window
- 🎨 **Modern Web Interface**: Dark-themed UI with real-time activity feed and progress tracking
- 💬 **Live LLM Conversations**: See every message exchanged between the LLMs in real-time
- � **Detailed Progress Tracking**: Never wonder what's happening - continuous status updates
- 📝 **Multi-Document Generation**: Produces 1-7 specialized markdown documents tailored to your research
- 🔄 **Iterative Refinement**: LLMs review and improve documents until they meet quality standards

## 🏗️ Architecture

### 11-Stage Research Pipeline

The system uses a sophisticated workflow designed to maximize the capabilities of DeepSeek's 128K context window:

#### **Stage 1: Initial Breakdown**
DeepSeek performs deep analysis of your research prompt, identifying:
- Core objectives and goals
- Key technologies involved
- Technical requirements
- Architecture needs
- Implementation challenges
- Research priorities (8-12 key discussion points)

#### **Stage 2: Discuss Breakdown** (4 rounds)
DeepSeek and Ollama collaborate to refine the breakdown and agree on research strategy:
- Round 1: Ollama reviews and critiques the breakdown
- Round 2: DeepSeek refines based on feedback
- Round 3: Ollama proposes 10-15 specific research queries
- Round 4: DeepSeek finalizes 12-18 research queries as JSON array

#### **Stage 3: Execute Research**
System executes all research queries in parallel:
- Performs 12-18 web searches via Serper API
- Gathers 100-200 source documents
- Stores all results with URLs and snippets

#### **Stage 4: Analyze Research**
DeepSeek analyzes ALL research results using full context (up to 150 sources):
- Extracts 15-20 key insights
- Identifies technology recommendations
- Documents architecture patterns
- Notes implementation details
- Identifies knowledge gaps

#### **Stage 5: Discuss Findings** (5 rounds)
LLMs extract and validate insights through structured discussion:
- DeepSeek presents top 10 findings
- Ollama critiques and questions
- DeepSeek elaborates with evidence
- Iterative refinement of understanding

#### **Stage 6: Deep Dive** (7 rounds + dynamic searches)
Detailed technical analysis with ability to trigger additional research:
- LLMs alternate deep analysis and critical review
- Can request additional searches when gaps found
- Focus on implementation details and technical depth

#### **Stage 7: Compile Information**
DeepSeek creates master compilation using full 128K context:
- Synthesizes ALL research and discussions
- Comprehensive technical analysis
- Architecture recommendations
- Implementation roadmap

#### **Stage 8: Discuss Compilation** (4 rounds)
LLMs validate completeness and trigger final searches if needed:
- Ollama reviews compilation for gaps
- DeepSeek updates based on feedback
- Additional searches if information missing
- Final validation

#### **Stage 9: Generate Documents**
DeepSeek creates 1-7 specialized markdown documents:
- Tailored to research topic
- Each document serves specific purpose
- Comprehensive and well-structured
- Citations and references included

#### **Stage 10: Refine Documents**
LLMs iteratively improve each document (max 6 rounds per doc):
- Ollama reviews each document for quality
- DeepSeek revises based on feedback
- Process repeats until Ollama approves
- Ensures high-quality, polished output

#### **Stage 11: Completed**
Research finished - documents ready for download

### Key Architecture Decisions

- **DeepSeek as Context King**: With 128K context, DeepSeek handles analysis, synthesis, and document generation
- **Ollama as Critic**: 32K context used for critical review, validation, and identifying gaps
- **Iterative Discussions**: Multiple rounds ensure thorough exploration of topics
- **Dynamic Search Triggers**: LLMs can request additional research when needed
- **Document-Focused Output**: Practical, usable documentation rather than raw data

## Quick Start

### Prerequisites

- Python 3.8+
- Ollama installed and running with `qwen3-coder:latest` model
- DeepSeek API key
- Serper.dev API key

### Installation

1. **Clone or create the project structure**:
   ```bash
   # The system creates the following structure:
   ai_research_system/
   ├── app/              # Flask web application
   ├── core/             # Core business logic
   ├── config/           # Configuration management
   ├── utils/            # Utility functions
   ├── DEVPLAN/          # Generated development plans
   └── requirements.txt  # Python dependencies
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

4. **Start the application**:
   ```bash
   python run.py
   ```

5. **Access the web interface**:
   Open http://localhost:5000 in your browser

## Configuration

### Environment Variables

Create a `.env` file with the following variables:

```env
# DeepSeek API Configuration
DEEPSEEK_API_KEY=your_deepseek_api_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1

# Serper.dev API Configuration
SERPER_API_KEY=your_serper_api_key_here

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen3-coder:latest

# File Storage
DEVPLAN_DIR=DEVPLAN

# Application Settings
MAX_CONVERSATION_ROUNDS=4
MAX_SEARCH_RESULTS=5
REQUEST_TIMEOUT=30

# Quality Settings
MIN_CONTEXT_MATURITY=0.8
MIN_PLAN_QUALITY=0.7
```

### Ollama Setup

Ensure Ollama is running and the qwen3-coder model is available:

```bash
# Start Ollama service
ollama serve

# Pull the model (if not already available)
ollama pull qwen3-coder:latest

# Verify the model is available
ollama list
```

## Usage

### Web Interface

1. **Start Research**: Enter your project idea in the prompt field
2. **Monitor Progress**: Watch the conversation between DeepSeek and Ollama
3. **Review Research**: See web search results and key insights
4. **Generate Plan**: Create and save development plans
5. **Export Plans**: Download plans as Markdown files

### API Endpoints

The system provides a REST API for programmatic access:

- `POST /research` - Start a new research session
- `POST /conversation/next` - Execute next conversation round
- `POST /plan/generate` - Generate development plan
- `GET /plans` - List all development plans
- `GET /plans/{filename}` - Get specific plan
- `GET /plans/{filename}/export` - Export plan as Markdown

### Example Workflow

```javascript
// 1. Start research session
const response = await fetch('/research', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({prompt: 'Build a task management app'})
});

// 2. Execute conversation rounds
for (let i = 0; i < 4; i++) {
    await fetch('/conversation/next', {method: 'POST'});
}

// 3. Generate development plan
const plan = await fetch('/plan/generate', {method: 'POST'});
```

## Development Plans

Development plans are stored in the `DEVPLAN/` directory and include:

- Project overview and objectives
- Technology stack with justification
- System architecture
- Implementation roadmap
- Resource requirements
- Risk assessment
- Research metrics

### Plan Format

```json
{
  "project_name": "Generated from prompt",
  "user_prompt": "Original user input",
  "development_plan": "Detailed plan content",
  "feasibility_assessment": {
    "feasibility_score": 0.85,
    "technical_feedback": "...",
    "risks_identified": ["..."],
    "recommendations": ["..."]
  },
  "research_metrics": {
    "total_searches": 15,
    "key_insights": 8,
    "conversation_rounds": 4,
    "context_maturity": 0.92,
    "quality_gates_passed": ["initial_research", "llm_consensus", ...]
  }
}
```

## Quality Management

The system implements comprehensive quality management:

### Quality Gates
1. **Initial Research Completion** - Comprehensive baseline research
2. **LLM Consensus on Architecture** - Technical approach validation
3. **Implementation Feasibility** - Technical viability confirmation
4. **Research Validation** - Key decisions backed by research
5. **Final Plan Quality** - Plan meets all quality criteria

### Context Evolution
- **Round 1**: Foundation building with initial research
- **Round 2**: Refinement and gap identification
- **Round 3**: Targeted research integration
- **Round 4**: Consensus building and finalization

## Troubleshooting

### Common Issues

1. **Ollama Connection Failed**
   - Ensure Ollama service is running: `ollama serve`
   - Verify model is available: `ollama list`
   - Check OLLAMA_BASE_URL in configuration

2. **API Key Errors**
   - Verify DeepSeek and Serper API keys in .env file
   - Check API key permissions and quotas

3. **File Permission Issues**
   - Ensure write permissions for DEVPLAN directory
   - Check disk space availability

4. **Web Interface Not Loading**
   - Verify Flask application is running on port 5000
   - Check firewall settings

### Logs

Application logs are written to `ai_research_system.log` in the project root directory.

## Contributing

This system is designed to be extensible. Key extension points:

- **New LLM Integrations**: Add clients in `core/` directory
- **Additional Research Sources**: Extend `SerperClient` class
- **Custom Plan Formats**: Modify `FileManager` class
- **Quality Metrics**: Extend quality assessment in `ConversationOrchestrator`

## License

This project is for educational and research purposes.