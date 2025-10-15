import json
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum


class LLMType(Enum):
    DEEPSEEK = "deepseek"
    OLLAMA = "ollama"


class ConversationStage(Enum):
    # New iterative workflow
    INITIAL_BREAKDOWN = "initial_breakdown"           # 1. DeepSeek breaks prompt into key points
    DISCUSS_BREAKDOWN = "discuss_breakdown"           # 2. LLMs discuss key points & what to research
    RESEARCH = "research"                             # 3. Perform searches on topics LLMs identified
    ANALYZE_RESEARCH = "analyze_research"             # 4. DeepSeek breaks down research findings
    DISCUSS_FINDINGS = "discuss_findings"             # 5. LLMs discuss findings & extract key points
    DEEP_DIVE = "deep_dive"                           # 6. LLMs discuss key points, more searches if needed
    COMPILE_INFORMATION = "compile_information"       # 7. DeepSeek compiles all information
    DISCUSS_COMPILATION = "discuss_compilation"       # 8. LLMs discuss compilation, more searches if needed
    GENERATE_DOCUMENTS = "generate_documents"         # 9. DeepSeek creates 1-7 documents
    REFINE_DOCUMENTS = "refine_documents"             # 10. LLMs discuss & refine documents until perfect
    COMPLETED = "completed"                           # 11. Documents ready for download


class QualityLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class SearchResult:
    """Represents a single web search result"""
    title: str
    link: str
    snippet: str
    source: str
    relevance_score: float = 0.0
    confidence_score: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class LLMMessage:
    """Represents a message from an LLM"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    llm_type: LLMType = None
    content: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    context_references: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    quality_issues: List[str] = field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)


@dataclass
class ResearchContext:
    """Main research context object that persists throughout conversation"""
    
    # Core identifiers
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_prompt: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    # Research data
    initial_searches: List[SearchResult] = field(default_factory=list)
    targeted_searches: List[SearchResult] = field(default_factory=list)
    key_insights: List[Dict[str, Any]] = field(default_factory=list)
    technology_references: Dict[str, List[str]] = field(default_factory=dict)
    citation_map: Dict[str, List[str]] = field(default_factory=dict)
    search_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Conversation data
    messages: List[LLMMessage] = field(default_factory=list)
    current_stage: ConversationStage = ConversationStage.INITIAL_BREAKDOWN  # Updated for new workflow
    conversation_round: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)  # For storing workflow state
    
    # Quality metrics
    context_maturity: float = 0.0
    research_coverage: float = 0.0
    decision_confidence: float = 0.0
    implementation_clarity: float = 0.0
    risk_assessment: float = 0.0
    
    # Quality gates
    quality_gates_passed: List[str] = field(default_factory=list)
    quality_issues: List[Dict[str, Any]] = field(default_factory=list)
    
    def update_timestamp(self):
        """Update the updated_at timestamp"""
        self.updated_at = datetime.now()
    
    def add_message(self, message: LLMMessage):
        """Add a message to the conversation"""
        self.messages.append(message)
        self.update_timestamp()
    
    def add_search_result(self, result: SearchResult, is_targeted: bool = False):
        """Add a search result to the appropriate list"""
        if is_targeted:
            self.targeted_searches.append(result)
        else:
            self.initial_searches.append(result)
        self.update_timestamp()
    
    def calculate_context_maturity(self) -> float:
        """Calculate overall context maturity score"""
        scores = []
        
        # Research coverage (30%)
        if self.initial_searches or self.targeted_searches:
            coverage_score = min(1.0, len(self.initial_searches + self.targeted_searches) / 10)
            scores.append(coverage_score * 0.3)
        
        # Conversation depth (30%)
        conversation_score = min(1.0, len(self.messages) / 20)
        scores.append(conversation_score * 0.3)
        
        # Quality gate progress (20%)
        gate_score = len(self.quality_gates_passed) / 5  # 5 total gates
        scores.append(gate_score * 0.2)
        
        # Decision confidence (20%)
        scores.append(self.decision_confidence * 0.2)
        
        self.context_maturity = sum(scores)
        return self.context_maturity
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'session_id': self.session_id,
            'user_prompt': self.user_prompt,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'current_stage': self.current_stage.value,
            'conversation_round': self.conversation_round,
            'context_maturity': self.context_maturity,
            'quality_gates_passed': self.quality_gates_passed,
            'message_count': len(self.messages),
            'search_count': len(self.initial_searches) + len(self.targeted_searches)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ResearchContext':
        """Create ResearchContext from dictionary"""
        context = cls(
            session_id=data.get('session_id', str(uuid.uuid4())),
            user_prompt=data.get('user_prompt', ''),
            created_at=datetime.fromisoformat(data.get('created_at', datetime.now().isoformat())),
            updated_at=datetime.fromisoformat(data.get('updated_at', datetime.now().isoformat())),
            current_stage=ConversationStage(data.get('current_stage', 'initial_research')),
            conversation_round=data.get('conversation_round', 0),
            context_maturity=data.get('context_maturity', 0.0),
            quality_gates_passed=data.get('quality_gates_passed', [])
        )
        return context