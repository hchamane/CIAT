"""
=================================================================
Cultural Impact Assessment Tool (CIAT) - Application Entry Point
=================================================================

This script serves as the main entry point for the CIAT application.
It provides command-line interface functionality for running the web server
and potentially other command-line features.

Inspired by and adapted from:
    - https://github.com/levan92/logging-example
    - https://docs.python.org/3/library/argparse.html
    - https://flask-script.readthedocs.io/en/latest/
    - https://flask.palletsprojects.com/en/2.3.x/cli/

Author: Hainadine Chamane
Version: 1.0.0
Date: February 2025
"""

import os
import argparse
import logging

from ciat import create_app

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_web():
    """
    Start the Flask web interface.
    
    This function creates the Flask application using the application factory
    and runs the development server. In production, this should be replaced
    with a WSGI server like Gunicorn or uWSGI.
    """
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000)


def main():
    """
    Parse command-line arguments and execute appropriate command.
    
    This function sets up the command-line argument parser and executes
    the appropriate function based on the command provided.
    """
    parser = argparse.ArgumentParser(
        description='Cultural Impact Assessment Tool (CIAT)'
    )
    subparsers = parser.add_subparsers(dest='command')
    
    # Add web command
    web_parser = subparsers.add_parser('web', help='Run the web interface')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute appropriate command
    if args.command == 'web':
        logger.info("Starting web server...")
        run_web()
    else:
        parser.print_help()


if __name__ == '__main__':
    main()

