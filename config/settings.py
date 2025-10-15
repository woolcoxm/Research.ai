import os
import logging
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Application configuration - All settings loaded from environment variables"""
    
    # ============================================================================
    # API Keys (REQUIRED)
    # ============================================================================
    DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
    SERPER_API_KEY = os.getenv('SERPER_API_KEY')
    
    # ============================================================================
    # DeepSeek Settings
    # ============================================================================
    DEEPSEEK_BASE_URL = os.getenv('DEEPSEEK_BASE_URL', 'https://api.deepseek.com/v1')
    DEEPSEEK_MODEL = os.getenv('DEEPSEEK_MODEL', 'deepseek-chat')
    DEEPSEEK_CONTEXT_WINDOW = int(os.getenv('DEEPSEEK_CONTEXT_WINDOW', '128000'))
    DEEPSEEK_DEFAULT_TEMPERATURE = float(os.getenv('DEEPSEEK_DEFAULT_TEMPERATURE', '0.7'))
    DEEPSEEK_DEFAULT_MAX_TOKENS = int(os.getenv('DEEPSEEK_DEFAULT_MAX_TOKENS', '4096'))
    
    # DeepSeek token limits per stage
    DEEPSEEK_STAGE1_MAX_TOKENS = int(os.getenv('DEEPSEEK_STAGE1_MAX_TOKENS', '8000'))
    DEEPSEEK_STAGE2_MAX_TOKENS = int(os.getenv('DEEPSEEK_STAGE2_MAX_TOKENS', '2000'))
    DEEPSEEK_STAGE4_MAX_TOKENS = int(os.getenv('DEEPSEEK_STAGE4_MAX_TOKENS', '8000'))
    DEEPSEEK_STAGE5_MAX_TOKENS = int(os.getenv('DEEPSEEK_STAGE5_MAX_TOKENS', '4000'))
    DEEPSEEK_STAGE7_MAX_TOKENS = int(os.getenv('DEEPSEEK_STAGE7_MAX_TOKENS', '8000'))
    DEEPSEEK_STAGE9_MAX_TOKENS = int(os.getenv('DEEPSEEK_STAGE9_MAX_TOKENS', '8000'))
    
    # ============================================================================
    # Ollama Settings
    # ============================================================================
    OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
    OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'qwen3-coder:latest')
    OLLAMA_CONTEXT_WINDOW = int(os.getenv('OLLAMA_CONTEXT_WINDOW', '32768'))
    OLLAMA_DEFAULT_TEMPERATURE = float(os.getenv('OLLAMA_DEFAULT_TEMPERATURE', '0.7'))
    OLLAMA_DEFAULT_MAX_TOKENS = int(os.getenv('OLLAMA_DEFAULT_MAX_TOKENS', '32768'))
    
    # Ollama token limits for different operations
    OLLAMA_REVIEW_MAX_TOKENS = int(os.getenv('OLLAMA_REVIEW_MAX_TOKENS', '24576'))
    OLLAMA_DISCUSSION_MAX_TOKENS = int(os.getenv('OLLAMA_DISCUSSION_MAX_TOKENS', '32768'))
    OLLAMA_VALIDATION_MAX_TOKENS = int(os.getenv('OLLAMA_VALIDATION_MAX_TOKENS', '8192'))
    
    # ============================================================================
    # Flask Application Settings
    # ============================================================================
    FLASK_HOST = os.getenv('FLASK_HOST', '0.0.0.0')
    FLASK_PORT = int(os.getenv('FLASK_PORT', '5000'))
    FLASK_DEBUG = os.getenv('FLASK_DEBUG', 'True').lower() in ('true', '1', 'yes')
    FLASK_SECRET_KEY = os.getenv('FLASK_SECRET_KEY', 'dev_secret_key_change_in_production')
    
    # ============================================================================
    # Research Workflow Settings
    # ============================================================================
    MAX_CONVERSATION_ROUNDS = int(os.getenv('MAX_CONVERSATION_ROUNDS', '50'))
    MAX_SEARCH_RESULTS = int(os.getenv('MAX_SEARCH_RESULTS', '15'))
    REQUEST_TIMEOUT = int(os.getenv('REQUEST_TIMEOUT', '120'))
    
    # ============================================================================
    # File Storage
    # ============================================================================
    DEVPLAN_DIR = os.getenv('DEVPLAN_DIR', 'DEVPLAN')
    OUTPUT_DIR = os.getenv('OUTPUT_DIR', '.')
    
    # ============================================================================
    # Quality Settings
    # ============================================================================
    MIN_CONTEXT_MATURITY = float(os.getenv('MIN_CONTEXT_MATURITY', '0.8'))
    MIN_PLAN_QUALITY = float(os.getenv('MIN_PLAN_QUALITY', '0.7'))
    
    # ============================================================================
    # Logging Settings
    # ============================================================================
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = os.getenv('LOG_FILE', 'ai_research_system.log')
    
    @classmethod
    def validate_config(cls):
        """Validate that all required configuration is present"""
        missing = []
        
        if not cls.DEEPSEEK_API_KEY:
            missing.append('DEEPSEEK_API_KEY')
        if not cls.SERPER_API_KEY:
            missing.append('SERPER_API_KEY')
            
        if missing:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing)}\n"
                f"Please create a .env file based on .env.example and add your API keys."
            )
        
        # Create directories if they don't exist
        os.makedirs(cls.DEVPLAN_DIR, exist_ok=True)
        if cls.OUTPUT_DIR != '.':
            os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        
        return True
    
    @classmethod
    def print_config_summary(cls):
        """Print a summary of current configuration (for debugging)"""
        logger = logging.getLogger(__name__)
        logger.info("=" * 70)
        logger.info("AI Research System Configuration")
        logger.info("=" * 70)
        logger.info(f"DeepSeek Model: {cls.DEEPSEEK_MODEL}")
        logger.info(f"DeepSeek Context: {cls.DEEPSEEK_CONTEXT_WINDOW:,} tokens")
        logger.info(f"Ollama Model: {cls.OLLAMA_MODEL}")
        logger.info(f"Ollama Context: {cls.OLLAMA_CONTEXT_WINDOW:,} tokens")
        logger.info(f"Ollama URL: {cls.OLLAMA_BASE_URL}")
        logger.info(f"Flask: {cls.FLASK_HOST}:{cls.FLASK_PORT} (debug={cls.FLASK_DEBUG})")
        logger.info(f"Max Rounds: {cls.MAX_CONVERSATION_ROUNDS}")
        logger.info(f"Max Search Results: {cls.MAX_SEARCH_RESULTS}")
        logger.info(f"Request Timeout: {cls.REQUEST_TIMEOUT}s")
        logger.info(f"Output Directory: {cls.OUTPUT_DIR}")
        logger.info("=" * 70)