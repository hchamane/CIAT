"""
=======================================
Cultural Impact Assessment Tool (CIAT)
=======================================

A predictive model and assessment tool for determining the extent of 
cultural impact on international project management success.

Based on the research framework developed by Hainadine Chamane, drawing on:
    - Cultural variables identified in Fog's (2022) cross-cultural study
    - Research on factors influencing international project success by 
      Dumitrașcu-Băldău, Dumitrașcu, and Dobrotă (2021)

References and Inspirations:  
    1. Flask Application Factory - https://flask.palletsprojects.com/en/2.3.x/patterns/appfactories/  
    2. Efficient File Upload - https://transloadit.com/devtips/efficient-flask-file-uploads-a-step-by-step-guide/  
    3. Flask-WTF CSRF Protection - https://flask-wtf.readthedocs.io/en/0.15.x/csrf/  
    4. Flask Logging Techniques - https://github.com/levan92/logging-example  
    5. Flask Tutorial Example App - https://github.com/pallets/flask/blob/main/examples/tutorial/flaskr/__init__.py  
    6. Python Flask Project Structure - https://codingnomads.com/python-flask-app-configuration-project-structure  
    7. Secure Application - https://gist.github.com/mikefromit/19e2bb6dfe05b9851d7dd34103bba522  
    8. GPT Playground Integration Example - https://github.com/marketplace/models/azure-openai/gpt-4o/playground

Author: Hainadine Chamane  
Version: 1.0.0  
Date: February 2025  
"""

import os
import logging
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_wtf.csrf import CSRFProtect

# ===========================================
# Logging Configuration
# ===========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("ciat_app.log"),
        logging.StreamHandler()
    ],
)
logger = logging.getLogger(__name__)

# Flask Extensions Initialisation
db = SQLAlchemy()
csrf = CSRFProtect()


def create_app():
    """
    Create and configure the Flask application using the application factory pattern.
    
    This function sets up the Flask app, database, security features, and registers blueprints. 
    It ensures necessary directories are created and configures important settings for both 
    development and production use.

    Returns:
        Flask: The configured Flask application instance
    """
    # Application Initialisation
    app = Flask(__name__)
    app.debug = True  # NOTE: Disable app.debug=True in production!

    # Ensure Directories Exist (Uploads/Database)
    uploads_dir = os.path.join(app.root_path, "uploads")
    os.makedirs(uploads_dir, exist_ok=True)

    db_path = os.path.join(app.root_path, "instance", "ciat.db")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    # Application Configuration (Keys/Paths/Security)
    app.config["SECRET_KEY"] = "C3l£cE3(y(rfS*7X"  # WARNING: Replace this key with an environment variable in production!
    app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{db_path}"
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    app.config["UPLOAD_FOLDER"] = uploads_dir

    # Session security-related configurations (ensure security improvements in production)
    app.config["SESSION_COOKIE_SECURE"] = False  # Set to True in production with HTTPS
    app.config["SESSION_COOKIE_HTTPONLY"] = True
    app.config["PERMANENT_SESSION_LIFETIME"] = 3600  # Session timeout (1 hour)

    # Initialise Extensions
    db.init_app(app)
    csrf.init_app(app)

    # Database Creation
    with app.app_context():
        try:
            db.create_all()
            logger.info(f"Database created successfully at: {db_path}")
        except Exception as db_error:
            logger.error(f"Failed to create the database: {str(db_error)}")

    # Blueprint Registration
    try:
        from .web_interface import web_app, init_app  # Assumes 'web_interface' module exists
        app.register_blueprint(web_app)
        init_app(app)
        logger.info("Web interface registered and initialised successfully.")
    except ImportError as ie:
        logger.error(f"Failed to register web interface: {str(ie)}")
    except Exception as e:
        logger.error(f"An unexpected error occurred while initialising the app: {str(e)}")

    return app


# Reminder for Production
if __name__ == '__main__':
    logger.warning("This script is intended for use as a library, not as a standalone program.")