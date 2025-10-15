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

## 🎨 User Interface

The system features a modern dark-themed web interface with comprehensive real-time visibility into the research process.

### Navigation Sections

The sidebar provides 4 main navigation sections:

#### 1. 🚀 Start Research
- Enter your research prompt (e.g., "Build a web scraping system with Python")
- Start button initiates the 11-stage workflow
- Session management and research initiation

#### 2. 📊 Progress Tracker
**Metric Cards** display real-time statistics:
- Total Rounds: Number of LLM discussion rounds completed
- Web Searches: Count of research queries executed
- Research Sources: Number of sources analyzed (typically 100-200)
- Generated Docs: Count of markdown documents created

**Stage Pipeline** shows all 11 stages with visual indicators:
- ⚪ Not Started (gray): Stage not yet reached
- 🔵 In Progress (pulsing blue): Currently executing
- ✅ Completed (green): Stage finished

#### 3. 📡 Live Activity Feed
Real-time stream of ALL system activity:
- Every status update from the workflow
- LLM actions ("Analyzing research results...", "Reviewing document...")
- Stage transitions ("Moving to Stage 5: Discuss Findings")
- Search operations ("Executing 12 web searches...")
- Timestamps for each activity
- Expandable/collapsible for space management
- Auto-scrolls to show latest activity

**Never wonder "is it stuck?"** - The activity feed provides continuous visibility into what's happening.

#### 4. 💬 LLM Conversations
Side-by-side conversation panels showing full message history:

**DeepSeek Panel** (left):
- All messages sent by DeepSeek
- Analysis results, research findings, document generations
- Scrollable with 300px max height per message

**Ollama Panel** (right):
- All messages sent by Ollama
- Critical reviews, feedback, questions
- Scrollable with 300px max height per message

### Real-Time Updates

The interface polls for updates automatically:
- **Status Updates**: Every 2 seconds (current stage, metrics, activities)
- **Conversation Updates**: Every 3 seconds (new LLM messages)
- No page refresh needed - everything updates live
- Smooth animations and transitions for state changes

### Completion Experience

When research finishes:
- Animated completion banner appears
- Download buttons for each generated document
- Success metrics displayed
- Option to start new research

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

## ⚙️ Configuration

The system is **fully configurable through environment variables**. All settings are loaded from a `.env` file - no hardcoded values!

### Quick Configuration

1. **Copy the example environment file**:
   ```bash
   cp .env.example .env
   ```

2. **Edit `.env` and add your API keys** (required):
   ```env
   DEEPSEEK_API_KEY=your_deepseek_api_key_here
   SERPER_API_KEY=your_serper_api_key_here
   ```

3. **Customize other settings** (optional - defaults work well)

### Complete Configuration Reference

#### 🔑 Required API Keys

```env
# Get from: https://platform.deepseek.com/
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxx

# Get from: https://serper.dev/
SERPER_API_KEY=xxxxxxxxxxxxx
```

#### 🤖 LLM Provider Settings

**DeepSeek Configuration:**
```env
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_MODEL=deepseek-chat              # Model to use
DEEPSEEK_CONTEXT_WINDOW=128000            # 128K tokens
DEEPSEEK_DEFAULT_TEMPERATURE=0.7          # Creativity (0.0-1.0)
DEEPSEEK_DEFAULT_MAX_TOKENS=4096          # Default response length
```

**Ollama Configuration (Easy to Change!):**
```env
OLLAMA_BASE_URL=http://localhost:11434    # Change if Ollama runs elsewhere
OLLAMA_MODEL=qwen3-coder:latest           # 🎯 Change this to use different model!
OLLAMA_CONTEXT_WINDOW=32768               # Adjust based on your model
OLLAMA_DEFAULT_TEMPERATURE=0.7            # Creativity (0.0-1.0)
OLLAMA_DEFAULT_MAX_TOKENS=32768           # Response length limit
```

**Want to use a different Ollama model?** Just change `OLLAMA_MODEL`:
- `llama3.1:latest` - Meta's Llama 3.1
- `mistral:latest` - Mistral AI
- `codellama:latest` - Code-specialized Llama
- `deepseek-coder:latest` - Alternative DeepSeek model
- Any other model from [ollama.com/library](https://ollama.com/library)

#### 🌐 Flask Web Server Settings

```env
FLASK_HOST=0.0.0.0                        # 0.0.0.0 = all interfaces, 127.0.0.1 = localhost only
FLASK_PORT=5000                           # Change if port 5000 is in use
FLASK_DEBUG=True                          # Set to False in production
FLASK_SECRET_KEY=your_secret_key_here     # Change in production!
```

#### 🔬 Research Workflow Settings

```env
MAX_CONVERSATION_ROUNDS=50                # Safety limit (workflow auto-completes)
MAX_SEARCH_RESULTS=15                     # Results per search query
REQUEST_TIMEOUT=120                       # API timeout in seconds
```

#### 🎛️ Advanced Token Limits (Per Stage)

Fine-tune response lengths for each workflow stage:

```env
# DeepSeek token limits
DEEPSEEK_STAGE1_MAX_TOKENS=8000          # Initial breakdown
DEEPSEEK_STAGE2_MAX_TOKENS=2000          # Focused queries  
DEEPSEEK_STAGE4_MAX_TOKENS=8000          # Research analysis
DEEPSEEK_STAGE5_MAX_TOKENS=4000          # Discussions
DEEPSEEK_STAGE7_MAX_TOKENS=8000          # Master compilation
DEEPSEEK_STAGE9_MAX_TOKENS=8000          # Document generation

# Ollama token limits
OLLAMA_REVIEW_MAX_TOKENS=24576           # Critical reviews
OLLAMA_DISCUSSION_MAX_TOKENS=32768       # Deep discussions
OLLAMA_VALIDATION_MAX_TOKENS=8192        # Quick validations
```

#### 📁 File Storage

```env
DEVPLAN_DIR=DEVPLAN                      # Legacy planning directory
OUTPUT_DIR=.                             # Where documents are saved
```

#### 📊 Logging

```env
LOG_LEVEL=INFO                           # DEBUG, INFO, WARNING, ERROR
LOG_FILE=ai_research_system.log          # Log file name
```

### Configuration Tips

**For Faster Research:**
- Reduce `MAX_SEARCH_RESULTS` to 10
- Lower token limits by 25%
- Use a smaller Ollama model

**For Deeper Research:**
- Increase `MAX_SEARCH_RESULTS` to 20
- Use maximum token limits
- Increase `REQUEST_TIMEOUT` to 180

**For Cost Optimization:**
- Lower all DeepSeek token limits by 50%
- Reduce `MAX_SEARCH_RESULTS` to 10
- Use smaller context windows

**For Different Ollama Models:**
1. Pull the model: `ollama pull model-name:tag`
2. Update `.env`: `OLLAMA_MODEL=model-name:tag`
3. Adjust `OLLAMA_CONTEXT_WINDOW` based on model specs
4. Restart the application: `python run.py`

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

## 📖 Usage Guide

### Starting a Research Session

1. Navigate to http://localhost:5000
2. Enter your research prompt in the input field (be specific!)
3. Click "Start Research"
4. Monitor progress through the 4 interface sections

### Example Prompts

**Good prompts are specific and actionable**:
- ✅ "Build a Python web scraping system that handles JavaScript-rendered content and stores data in PostgreSQL"
- ✅ "Design a microservices architecture for a real-time chat application using Node.js and Redis"
- ✅ "Create a machine learning pipeline for sentiment analysis of customer reviews using transformers"

**Poor prompts are vague**:
- ❌ "Make a website"
- ❌ "AI project"
- ❌ "Something with data"

### Monitoring the Workflow

**Progress Tracker Section**:
- Watch the stage pipeline progress through all 11 stages
- Monitor metrics: rounds, searches, sources, documents
- Current stage indicator shows exactly where the system is

**Activity Feed Section**:
- See every action as it happens
- Status updates like "Analyzing 150 research sources..."
- Stage transitions like "Stage 6/11: Deep Dive (Round 3/7)"
- Search operations like "Executing additional search: 'Python async web scraping'"

**Conversations Section**:
- Read full LLM discussion history
- Understand the reasoning and analysis
- See how the LLMs collaborate and refine ideas

### Typical Workflow Timeline

- **Stages 1-2** (5-10 min): Initial analysis and research planning
- **Stage 3** (1-2 min): Web searches execute in parallel
- **Stages 4-5** (10-15 min): Research analysis and insight extraction
- **Stage 6** (15-20 min): Deep technical dive with additional searches
- **Stages 7-8** (10-15 min): Compilation and validation
- **Stages 9-10** (10-20 min): Document generation and refinement
- **Total**: 50-80 minutes for comprehensive research

### Generated Documents

Documents are saved to the workspace root with timestamped names:
```
research_output_YYYYMMDD_HHMMSS_DocumentTitle.md
```

Typical document types:
- Technical Architecture Document
- Implementation Guide
- Technology Comparison Report
- Best Practices Guide
- API Integration Guide
- Deployment Strategy
- Testing Strategy

### API Endpoints

The system provides a REST API for programmatic access:

- `POST /start_research` - Start a new research session (body: `{"user_prompt": "your prompt"}`)
- `GET /status/<session_id>` - Get current status and metrics for a research session
- `GET /conversations/<session_id>` - Get all LLM conversation messages
- `GET /download/<filename>` - Download a generated markdown document

### Programmatic Usage

```javascript
// 1. Start research session
const response = await fetch('/start_research', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({user_prompt: 'Build a real-time collaboration system'})
});
const data = await response.json();
const sessionId = data.session_id;

// 2. Poll for status updates
const statusInterval = setInterval(async () => {
    const status = await fetch(`/status/${sessionId}`).then(r => r.json());
    console.log('Stage:', status.current_stage);
    console.log('Rounds:', status.total_rounds);
    console.log('Sources:', status.research_sources);
    
    if (status.current_stage === 'COMPLETED') {
        clearInterval(statusInterval);
        console.log('Documents:', status.documents);
    }
}, 2000);

// 3. Monitor LLM conversations
const convInterval = setInterval(async () => {
    const convs = await fetch(`/conversations/${sessionId}`).then(r => r.json());
    console.log('DeepSeek messages:', convs.deepseek?.length || 0);
    console.log('Ollama messages:', convs.ollama?.length || 0);
}, 3000);
```

## 📄 Generated Documents

Documents are automatically saved to the workspace root directory with timestamped filenames:

### File Naming Convention
```
research_output_YYYYMMDD_HHMMSS_DocumentTitle.md
```

Example:
```
research_output_20250115_143022_TechnicalArchitectureDocument.md
research_output_20250115_143022_ImplementationGuide.md
research_output_20250115_143022_TechnologyComparisonReport.md
```

### Document Quality

Each document is:
- ✅ **Reviewed by both LLMs**: Ollama critiques, DeepSeek revises
- ✅ **Citation-backed**: Claims supported by research sources
- ✅ **Comprehensive**: Typically 2000-5000 words per document
- ✅ **Properly structured**: Clear sections, headings, code examples
- ✅ **Actionable**: Practical guidance, not just theory

### Document Types by Research Topic

The system intelligently generates relevant document types:

**Web Development Projects**:
- Frontend Architecture Document
- Backend API Design
- Database Schema Guide
- Deployment Strategy

**Machine Learning Projects**:
- Model Selection Report
- Data Pipeline Architecture
- Training Strategy Guide
- Inference Optimization

**System Design Projects**:
- Technical Architecture
- Scalability Analysis
- Technology Stack Comparison
- Implementation Roadmap

## 🔧 Technical Details

### Token Management

**DeepSeek (128K context)**:
- Stage 1: 8000 tokens max (deep analysis)
- Stage 2-3: 2000 tokens (focused queries)
- Stage 4: 8000 tokens (analyze 100-200 sources)
- Stage 5-6: 4000 tokens (detailed discussion)
- Stage 7: 8000 tokens (master compilation)
- Stage 9: 8000 tokens per document (generation)

**Ollama (32K context)**:
- All stages: 2000 tokens (critical review and validation)

### Research Capabilities

**Search Volume**:
- Initial research: 12-18 queries executed in parallel
- Deep dive: 2-8 additional targeted searches
- Total: 100-200 source documents analyzed

**Source Processing**:
- Each search returns ~8-10 results
- DeepSeek analyzes full context of all sources simultaneously
- Citations preserved for document generation

### Performance Characteristics

**Resource Usage**:
- Memory: ~500MB for Flask app + LLM client overhead
- Network: 2-5MB total API calls per research session
- Storage: 50-500KB per generated document

**API Rate Limits**:
- DeepSeek: Typically 60 requests/minute (enough for workflow)
- Serper: 2500 searches/month on free tier
- Ollama: No rate limits (local)

## ⚠️ Troubleshooting

### Common Issues

**1. "Workflow seems stuck at a stage"**
- ✅ Check the **Activity Feed** - it shows every action in real-time
- ✅ Stage 4, 6, 7, 9 can take 10-20 minutes (large context processing)
- ✅ If activity feed shows no updates for 5+ minutes, check logs

**2. "Ollama Connection Failed"**
```bash
# Ensure Ollama service is running
ollama serve

# Verify model is available
ollama list | grep qwen3-coder

# Check Ollama is accessible
curl http://localhost:11434/api/version
```

**3. "DeepSeek API Errors"**
- Verify API key in `.env` file
- Check API quota/billing status at DeepSeek dashboard
- Ensure `DEEPSEEK_BASE_URL=https://api.deepseek.com/v1`

**4. "Serper API Errors"**
- Verify API key in `.env` file
- Check remaining search quota at serper.dev
- Free tier: 2500 searches/month

**5. "Research produces generic documents"**
- ✅ Make your prompt more specific and technical
- ✅ Include technology names, constraints, requirements
- ❌ Avoid vague prompts like "build an app"

**6. "Web Interface Not Loading"**
```bash
# Check Flask is running
ps aux | grep python | grep run.py

# Verify port 5000 is available
netstat -an | grep 5000

# Check firewall allows localhost:5000
```

**7. "Documents Not Saving"**
- Check write permissions in workspace directory
- Ensure sufficient disk space (need ~10MB free)
- Verify no file locks on existing documents

### Debug Mode

To enable detailed logging, modify `config/settings.py`:
```python
LOG_LEVEL = 'DEBUG'
```

Then monitor logs in real-time:
```bash
# Windows PowerShell
Get-Content ai_research_system.log -Wait -Tail 50

# View full log
cat ai_research_system.log
```

### Getting Help

If issues persist:
1. Check the **Activity Feed** for error messages
2. Review `ai_research_system.log` for detailed error traces
3. Verify all API keys are valid and have quota remaining
4. Ensure Ollama is running with `qwen3-coder:latest` available

## 🚀 Performance Tips

### Faster Research Sessions

1. **Use specific prompts**: More focused research = fewer rounds
2. **Preload Ollama model**: Run `ollama run qwen3-coder:latest` before starting
3. **Stable internet**: Research phase downloads 100-200 web pages
4. **Close activity feed**: Reduces browser memory usage during long sessions

### Cost Optimization

**DeepSeek API**:
- Typical session: 150K-300K input tokens, 20K-40K output tokens
- Cost per session: ~$0.10-0.25 USD (varies by pricing)

**Serper API**:
- Typical session: 15-25 searches
- Free tier: 2500 searches/month = ~100-150 research sessions

## 🛠️ Extending the System

Key extension points for developers:

**Add New LLM Providers**:
1. Create client in `core/` (e.g., `anthropic_client.py`)
2. Implement `generate()` method matching interface
3. Update `conversation_orchestrator.py` to use new client

**Modify Workflow Stages**:
1. Edit `core/models.py` - add new `ConversationStage` enum value
2. Update `core/conversation_orchestrator.py` - add stage logic
3. Update `app/templates/index.html` - add stage to pipeline display

**Custom Document Types**:
1. Modify Stage 9 in `conversation_orchestrator.py`
2. Update document type instructions in DeepSeek prompt
3. Adjust document approval logic in Stage 10

**Additional Research Sources**:
1. Extend `core/serper_client.py` with new methods
2. Add API configuration to `config/settings.py`
3. Integrate in Stage 3 or Stage 6 of workflow

## 📝 Project Structure

```
AgentChat/
├── app/
│   ├── routes.py           # Flask routes, session management
│   ├── static/js/app.js    # Frontend JavaScript
│   └── templates/index.html # Web interface
├── core/
│   ├── conversation_orchestrator.py  # 11-stage workflow logic
│   ├── deepseek_client.py           # DeepSeek API integration
│   ├── ollama_client.py             # Ollama local integration
│   ├── serper_client.py             # Web search integration
│   └── models.py                    # Data models and enums
├── config/
│   └── settings.py         # Configuration management
├── utils/
│   └── file_manager.py     # Document saving utilities
├── requirements.txt        # Python dependencies
├── run.py                  # Application entry point
└── .env                    # API keys and configuration
```

## 📜 License

This project is for educational and research purposes.