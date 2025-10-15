#!/usr/bin/env python3
"""
AI Research System - Main Application Runner
"""

import os
import sys
import logging
from app.routes import app
from config.settings import Config

# Configure logging from environment
log_level = getattr(logging, Config.LOG_LEVEL.upper(), logging.INFO)
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(Config.LOG_FILE)
    ]
)

logger = logging.getLogger(__name__)


def main():
    """Main application entry point"""
    try:
        logger.info("Starting AI Research System...")
        
        # Check if we're in the correct directory
        if not os.path.exists('config/settings.py'):
            logger.error("Please run this script from the project root directory")
            sys.exit(1)
        
        # Validate configuration
        Config.validate_config()
        
        # Print configuration summary
        Config.print_config_summary()
        
        # Run the Flask application
        logger.info(f"AI Research System is running on http://{Config.FLASK_HOST}:{Config.FLASK_PORT}")
        logger.info("Press Ctrl+C to stop the application")
        
        app.run(debug=Config.FLASK_DEBUG, host=Config.FLASK_HOST, port=Config.FLASK_PORT)
        
    except KeyboardInterrupt:
        logger.info("Shutting down AI Research System...")
    except Exception as e:
        logger.error(f"Failed to start AI Research System: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()