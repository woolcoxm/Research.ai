import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Application configuration"""
    
    # API Keys
    DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
    SERPER_API_KEY = os.getenv('SERPER_API_KEY')
    
    # LLM Settings
    OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
    OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'qwen3-coder:latest')
    DEEPSEEK_BASE_URL = os.getenv('DEEPSEEK_BASE_URL', 'https://api.deepseek.com/v1')
    
    # Context Window Settings
    DEEPSEEK_CONTEXT_WINDOW = int(os.getenv('DEEPSEEK_CONTEXT_WINDOW', '128000'))  # 128k tokens
    OLLAMA_CONTEXT_WINDOW = int(os.getenv('OLLAMA_CONTEXT_WINDOW', '24000'))      # 24k tokens
    
    # File Storage
    DEVPLAN_DIR = os.getenv('DEVPLAN_DIR', 'DEVPLAN')
    
    # Application Settings
    MAX_CONVERSATION_ROUNDS = int(os.getenv('MAX_CONVERSATION_ROUNDS', '25'))  # Increased from 8 to 25
    MAX_SEARCH_RESULTS = int(os.getenv('MAX_SEARCH_RESULTS', '15'))  # Increased from 8 to 15 for more comprehensive research
    REQUEST_TIMEOUT = int(os.getenv('REQUEST_TIMEOUT', '30'))
    
    # Quality Settings
    MIN_CONTEXT_MATURITY = float(os.getenv('MIN_CONTEXT_MATURITY', '0.8'))
    MIN_PLAN_QUALITY = float(os.getenv('MIN_PLAN_QUALITY', '0.7'))
    
    @classmethod
    def validate_config(cls):
        """Validate that all required configuration is present"""
        missing = []
        
        if not cls.DEEPSEEK_API_KEY:
            missing.append('DEEPSEEK_API_KEY')
        if not cls.SERPER_API_KEY:
            missing.append('SERPER_API_KEY')
            
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
        
        # Create DEVPLAN directory if it doesn't exist
        os.makedirs(cls.DEVPLAN_DIR, exist_ok=True)
        
        return True