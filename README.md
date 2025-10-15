# AI Research System

A collaborative AI research system where DeepSeek API and local Ollama (qwen3-coder) work together to generate, refine ideas, and create development plans with web research capabilities.

## Features

- 🤖 **Dual LLM Collaboration**: DeepSeek and Ollama work together to analyze and refine ideas
- 🔍 **Web Research Integration**: Automated web searches using Serper.dev for comprehensive research
- 📋 **Development Plan Generation**: Structured development plans stored in DEVPLAN folder
- 🌐 **Web-based Interface**: User-friendly web interface (no command-line tools needed)
- 📊 **Quality Management**: Automated quality gates and context evolution tracking
- 💾 **File Management**: Export plans as JSON or Markdown

## Architecture

The system follows a multi-phase workflow:

1. **Initial Research**: Web searches for domain information and trends
2. **LLM Collaboration**: DeepSeek and Ollama discuss and refine ideas
3. **Targeted Research**: Additional searches based on identified gaps
4. **Plan Generation**: Comprehensive development plan creation
5. **Quality Validation**: Feasibility assessment and quality checks

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