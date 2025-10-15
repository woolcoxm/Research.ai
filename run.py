#!/usr/bin/env python3
"""
AI Research System - Main Application Runner
"""

import os
import sys
import logging
from app.routes import app

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('ai_research_system.log')
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
        
        # Run the Flask application
        logger.info("AI Research System is running on http://localhost:5000")
        logger.info("Press Ctrl+C to stop the application")
        
        app.run(debug=True, host='0.0.0.0', port=5000)
        
    except KeyboardInterrupt:
        logger.info("Shutting down AI Research System...")
    except Exception as e:
        logger.error(f"Failed to start AI Research System: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()