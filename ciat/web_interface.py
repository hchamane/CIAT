"""
=======================================================
Cultural Impact Assessment Tool (CIAT) - Web Interface
=======================================================

Flask web application for the Cultural Impact Assessment Tool (CIAT).
Provides a user-friendly interface for assessing cultural impact on 
international project management success.

Inspired by and adapted from:
    - https://gist.github.com/mikefromit/19e2bb6dfe05b9851d7dd34103bba522
    - https://github.com/behai-nguyen/app-demo/blob/main/src/app_demo/utils/context_processor.py
    - https://github.com/pallets-eco/flask-wtf/blob/main/docs/form.rst
    - https://github.com/cerickson/flask-matplotlib-tutorial/blob/master/flaskplotlib/views.py
    - https://blog.pamelafox.org/2023/03/rendering-matplotlib-charts-in-flask.html
    - https://medium.com/%40amelie_yeh/data-visualization-on-the-web-with-flask-11a3b1f7a476
    - https://python-adv-web-apps.readthedocs.io/en/latest/flask3.html
    - https://hackersandslackers.com/flask-application-factory/

Author: Hainadine Chamane
Version: 1.0.0
Date: February 2025
"""

import os
import sys
import io
import base64
import json
import traceback
import logging
from pathlib import Path
from datetime import datetime
from flask_wtf import FlaskForm
from wtforms.fields import FileField
from werkzeug.utils import secure_filename
from flask import (
    Blueprint, render_template, request, jsonify, current_app,
    redirect, url_for, flash, send_file, session
)
from wtforms import (
    StringField, SelectField, IntegerField, SubmitField,
    FloatField, SelectMultipleField
)
from wtforms.validators import (
    DataRequired, NumberRange, Optional,
    Length, ValidationError
)

from flask_wtf.file import FileField as FlaskFileField, FileAllowed, FileRequired
from .forms import TrainModelForm, ProjectAssessmentForm, UploadForm, CompareCountriesForm
from ciat.data_processor import CulturalDataProcessor

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')  # Use non-interactive backend # https://matplotlib.org/stable/users/explain/figure/backends.html

# Configure logging
# https://github.com/if-ai/ComfyUI-IF_Trellis/blob/main/__init__.py
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ciat_web.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import CIAT modules
# https://flask.palletsprojects.com/en/stable/patterns/appfactories/
# https://favtutor.com/blogs/import-from-parent-directory-python
try:
    from .cultural_impact_model import CulturalImpactModel
    from .data_processor import CulturalDataProcessor
except ImportError:
    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from ciat.cultural_impact_model import CulturalImpactModel
        from ciat.data_processor import CulturalDataProcessor
    except ImportError:
        logger.critical("Failed to import CIAT modules. Please check your installation.")
        raise

# Create a Blueprint
# https://gist.github.com/leongjinqwen/9a5d51e5bbb20eecf4ee65f139b10cd6
web_app = Blueprint(
    'web_app', __name__,
    template_folder='templates',
    static_folder='static'
)

# Initialise global variables
# https://github.com/lingthio/Flask-User-starter-app/blob/master/app/__init__.py
data_dir = 'data'  # Placeholder, will be updated in init_app
model_path = None  # Placeholder, will be updated in init_app
upload_folder = 'uploads'  # Placeholder, will be updated in init_app
data_processor = None
model = None
hofstede_data = None
survey_insights = None


# Functions for plotting: 
# https://matplotlib.org/stable/gallery/lines_bars_and_markers/categorical_variables.html?utm_source=chatgpt.com
# https://github.com/Arnav7418/AllGraph.py
def safe_barh(ax, y, width, **kwargs):
    """
    Safely create a horizontal bar chart without parameter conflicts.
    
    Parameters:
        ax: matplotlib axes object
        y: positions for the bars (must be numeric array)
        width: width/length of bars
        **kwargs: additional keyword arguments for barh # https://github.com/Arnav7418/AllGraph.py
    
    Returns:
        The bar container object
    """
    # Ensure y is numeric
    if isinstance(y, (list, tuple)) and any(isinstance(item, str) for item in y):
        y_pos = np.arange(len(y))
        # Save original labels for later
        y_labels = y
        bars = ax.barh(y_pos, width, **kwargs)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(y_labels)
    else:
        bars = ax.barh(y, width, **kwargs)

    return bars


def safe_bar(ax, x, height, **kwargs):
    """
    Safely create a vertical bar chart without parameter conflicts.
    
    Parameters:
        ax: matplotlib axes object
        x: positions for the bars (must be numeric array)
        height: height of bars
        **kwargs: additional keyword arguments for bar
    
    Returns:
        The bar container object
    """
    # Ensure x is numeric
    if isinstance(x, (list, tuple)) and any(isinstance(item, str) for item in x):
        x_pos = np.arange(len(x))
        # Save original labels for later
        x_labels = x
        bars = ax.bar(x_pos, height, **kwargs)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, rotation=45, ha='right')
    else:
        bars = ax.bar(x, height, **kwargs)

    return bars


# Web processor for template utility functions
# https://github.com/eriktaubeneck/flask-patterns/blob/master/presentation.md
# https://github.com/pallets/flask/blob/main/docs/templating.rst
# https://flask-docs-ja.readthedocs.io/en/latest/extensiondev/
# https://hackersandslackers.com/flask-application-factory/
# https://github.com/miguelgrinberg/APIFairy/blob/main/docs/guide.rst
# https://blog.miguelgrinberg.com/post/handling-file-uploads-with-flask
# https://github.com/bigscience-workshop/promptsource/blob/main/promptsource/templates.py
@web_app.context_processor
def utility_processor():
    """
    Provide utility functions to templates.
    
    This context processor makes utility functions available to all templates
    within the blueprint.
    
    Returns:
        dict: Dictionary of utility functions
    """
    def format_date(date=None):
        """Format a date as YYYY-MM-DD."""
        if date is None:
            date = datetime.now()
        return date.strftime('%Y-%m-%d')

    return dict(
        format_date=format_date,
        now=datetime.now
    )


def inject_utilities():
    return {'now': datetime.now, 'format_date': lambda d=None: (d or datetime.now()).strftime('%Y-%m-%d')}
    

def init_app(app):
    global data_processor, model, survey_insights
    paths = prepare_directories(app.root_path)
    prepare_static_assets(app.root_path)
    data_processor = CulturalDataProcessor(data_dir=paths['data_dir'])
    model = load_or_train_model(paths['model_path'], data_processor)
    survey_insights = data_processor.get_default_survey_insights()
    
    
def load_training_data_from_csv(file_path):
    """
    Load training data from CSV and return features (X) and labels (y).

    Args:
        file_path (str): Path to the training data CSV file

    Returns:
        Tuple[pd.DataFrame, pd.Series]: (X, y) or (None, None)
    """
    expected_columns = [
        'power_distance', 'individualism', 'masculinity', 'uncertainty_avoidance',
        'technical_requirements', 'stakeholder_count', 'team_size',
        'language_barriers', 'communication_barriers', 'project_success'
    ]
    try:
        logger.info(f"Attempting to read file: {file_path}")
        df = pd.read_csv(file_path)
        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Columns found: {list(df.columns)}")
        
        missing = [col for col in expected_columns if col not in df.columns]
        if missing:
            logger.error(f"Missing columns in uploaded training data: {missing}")
            return None, None

        X = df[expected_columns[:-1]]
        y = df['project_success']

        if X.empty or y.empty:
            logger.error("Training data X or y is empty")
            return None, None

        logger.info(f"Training data successfully extracted. X shape: {X.shape}, y length: {len(y)}")
        return X, y
    except Exception as e:
        logger.error(f"Failed to load training data: {e}")
        return None, None
        
        
def init_app(app):
    """
    Initialise the application with proper paths and components.
    
    This function sets up the application paths, initialises the data processor
    and model, loads Hofstede cultural dimensions data, and creates default
    templates and static files if they don't exist.
    
    Args:
        app (Flask): The Flask application instance
    """
    global data_dir, upload_folder, data_processor, model, hofstede_data, survey_insights, model_path

    # Update paths to use app's root path instead of hardcoded paths
    base_dir = app.root_path
    data_dir = os.path.join(base_dir, 'data')
    model_path = os.path.join(data_dir, 'ciat_model.joblib')
    upload_folder = os.path.join(base_dir, 'uploads')

    # Create directories if they don't exist
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'assessments'), exist_ok=True)
    os.makedirs(upload_folder, exist_ok=True)

    logger.info(f"Base directory: {base_dir}")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Model path: {model_path}")
    logger.info(f"Upload folder: {upload_folder}")

    # Check if data directory exists
    logger.info(f"Data directory exists: {os.path.exists(data_dir)}")

    # Initialise data processor and model
    data_processor = CulturalDataProcessor(data_dir=data_dir)
    model = CulturalImpactModel()

    # Load Hofstede data
    hofstede_data = data_processor.load_hofstede_data()

    # Attempt to load pre-trained model
    try:
        if os.path.exists(model_path):
            model = CulturalImpactModel.load_model(model_path)
            logger.info(f"Loaded pre-trained CIAT model from {model_path}")
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")
    except Exception as e:
        logger.warning(f"Could not load pre-trained model: {str(e)}")
        logger.warning("Generating sample data and training a new model")

    try:
        # Generate sample project data
        project_data = data_processor._create_example_project_data(
            os.path.join(data_dir, 'project_data.csv')
        )

        # Skip the problematic data preparation and model training
        logger.info("Skipping model training due to data preparation issues")
        logger.info("Using a blank model instead")
        model = CulturalImpactModel()

        # Use a simple synthetic dataset for model initialisation
        # Create a simple synthetic dataset with known good structure
        synthetic_X = pd.DataFrame({
            'power_distance': np.random.randint(30, 90, 100),
            'individualism': np.random.randint(20, 80, 100),
            'masculinity': np.random.randint(30, 70, 100),
            'uncertainty_avoidance': np.random.randint(40, 90, 100),
            'technical_requirements': np.random.randint(1, 6, 100),
            'stakeholder_count': np.random.randint(5, 50, 100),
            'team_size': np.random.randint(3, 30, 100),
            'language_barriers': np.random.randint(1, 6, 100),
            'communication_barriers': np.random.randint(1, 6, 100)
        })

        synthetic_y = pd.Series(np.random.binomial(1, 0.6, 100))

        # Train on synthetic data
        try:
            logger.info("Training model on synthetic data")
            model.train(synthetic_X, synthetic_y)
            model.save_model(model_path)
            logger.info("Model trained and saved successfully with synthetic data")
        except Exception as synth_error:
            logger.error(f"Error training with synthetic data: {str(synth_error)}")
            model = CulturalImpactModel()  # Fallback to blank model

    except Exception as train_error:
        logger.error(f"Error training model: {str(train_error)}")
        # Fallback to a blank model
        model = CulturalImpactModel()

    # Load survey insights
    survey_insights = data_processor.get_default_survey_insights()

    # Add context processor for datetime
    @app.context_processor
    def inject_now():
        """context"""
        return {'now': datetime.now()}

    # Create static/css directory and copy custom theme
    static_css_dir = os.path.join(app.root_path, 'static', 'css')
    os.makedirs(static_css_dir, exist_ok=True)

    # Create custom-theme.css if it doesn't exist
    custom_theme_path = os.path.join(static_css_dir, 'custom-theme.css')
    if not os.path.exists(custom_theme_path):
        try:
            with open(os.path.join(os.path.dirname(__file__), 'static', 'css', 'custom-theme.css'), 'r') as src_file:
                with open(custom_theme_path, 'w') as dest_file:
                    dest_file.write(src_file.read())
            logger.info(f"Created custom theme CSS: {custom_theme_path}")
        except Exception as e:
            logger.warning(f"Could not create custom theme CSS: {str(e)}")

    # Check if templates directory exists, if not create it
    templates_dir = os.path.join(app.root_path, 'templates')
    if not os.path.exists(templates_dir):
        os.makedirs(templates_dir)
        logger.info(f"Created templates directory: {templates_dir}")

    # Check if static directory exists, if not create it
    static_dir = os.path.join(app.root_path, 'static')
    if not os.path.exists(static_dir):
        os.makedirs(static_dir)
        # Create subdirectories
        os.makedirs(os.path.join(static_dir, 'css'), exist_ok=True)
        os.makedirs(os.path.join(static_dir, 'js'), exist_ok=True)
        os.makedirs(os.path.join(static_dir, 'images'), exist_ok=True)
        logger.info(f"Created static directories in: {static_dir}")

    # Create default templates, CSS, and JS if needed
    create_default_templates(templates_dir)
    create_default_css(static_dir)
    create_default_js(static_dir)

# Main dashboard route
# https://www.digitalocean.com/community/tutorials/how-to-make-a-web-application-using-flask-in-python-3
# https://www.restack.io/p/flask-answer-interactive-data-dashboards-cat-ai
# https://github.com/flask-dashboard/Flask-MonitoringDashboard/blob/master/flask_monitoringdashboard/templates/fmd_base.html
# https://github.com/TomasBeuzen/machine-learning-tutorials/blob/master/ml-deploy-model/deploy-with-flask.ipynb
# https://blog.miguelgrinberg.com/post/handling-file-uploads-with-flask?utm_source=chatgpt.com
# https://flask.palletsprojects.com/en/stable/patterns/fileuploads/?utm_source=chatgpt.com
# https://sentry.io/answers/serve-static-files-flask/
# https://matplotlib.org/stable/gallery/lines_bars_and_markers/bar_label_demo.html
# https://plotly.com/python/radar-chart/
# https://flask-wtf.readthedocs.io/en/0.15.x/form/
# https://github.com/hayleyking/examples-forms/blob/master/examples-forms.py
@web_app.route('/')
def index():
    """
    Render the main dashboard page.
    
    This route displays the main dashboard with summary statistics,
    survey insights, and visualisations of regional focus and
    complexity factors.
    
    Returns:
        str: Rendered HTML template
    """
    global model, survey_insights

    # Create summary statistics from survey data
    stats = {
        "response_count": int(sum(survey_insights.get("experience_levels", {}).values()) * 100) if "experience_levels" in survey_insights else 14,
        "top_region": max(survey_insights["regions"].items(), key=lambda x: x[1])[0] if survey_insights["regions"] else "Unknown",
        "top_complexity_factor": max(survey_insights["complexity_factors"].items(), key=lambda x: x[1])[0] if survey_insights["complexity_factors"] else "Unknown",
        "top_communication_challenge": max(survey_insights["communication_challenges"].items(), key=lambda x: x[1])[0] if survey_insights["communication_challenges"] else "Unknown",
        "project_experience": survey_insights.get("experience_levels", {
            "1-5 years": "42.86%",
            "5-10 years": "7.14%",
            "10-15 years": "28.57%",
            "15+ years": "21.43%"
        })
    }

    # Check if model is trained
    model_trained = model.model is not None

    # Generate dashboard visualisations
    plots = {}

    # Regional focus chart
    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        regions = list(survey_insights["regions"].keys())
        values = list(survey_insights["regions"].values())
        colours = plt.cm.viridis(np.linspace(0.2, 0.8, len(regions)))

        # Use numeric positions instead of string labels
        x_pos = np.arange(len(regions))
        bars = ax.bar(x_pos, [v * 100 for v in values], color=colours)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(regions, rotation=45, ha='right')
        ax.set_ylabel('Percentage of Projects (%)')
        ax.set_title('Regional Focus in Survey Responses')
        ax.set_ylim(0, 100)

        # Add percentage labels
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

        plt.tight_layout()

        # Save to base64 for embedding in HTML
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plots['regions'] = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close(fig)
    except Exception as e:
        logger.error(f"Error creating regional focus chart: {str(e)}")

    # Complexity factors chart
    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        factors = list(survey_insights["complexity_factors"].keys())
        values = list(survey_insights["complexity_factors"].values())
        colours = plt.cm.viridis(np.linspace(0.2, 0.8, len(factors)))

        # Use numeric positions instead of string labels
        y_pos = np.arange(len(factors))
        bars = ax.barh(y_pos, [v * 100 for v in values], color=colours)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(factors)
        ax.set_xlabel('Percentage of Responses (%)')
        ax.set_title('Project Complexity Factors')
        ax.set_xlim(0, 100)

        # Add percentage labels
        for bar in bars:
            width = bar.get_width()
            ax.annotate(f'{width:.1f}%',
                        xy=(width, bar.get_y() + bar.get_height() / 2),
                        xytext=(3, 0),
                        textcoords="offset points",
                        ha='left', va='center')

        plt.tight_layout()

        # Save to base64 for embedding in HTML
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plots['complexity_factors'] = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close(fig)
    except Exception as e:
        logger.error(f"Error creating complexity factors chart: {str(e)}")

    return render_template('index.html', stats=stats, survey=survey_insights, plots=plots,
                           model_trained=model_trained)

def generate_survey_visualisations():
    """
    Generate visualisations for survey insights.
    
    Returns:
        dict: Dictionary containing base64-encoded plot images
    """
    plots = {}
    
    # Generate regional focus visualisation
    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Data from survey (15 respondents)
        regions = ['Europe', 'Africa', 'North America', 'Asia Pacific', 'South America', 'Middle East']
        values = [60, 53.33, 13.33, 13.33, 0, 0]  # Percentages
        
        # Create colors using viridis colormap
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(regions)))
        
        # Horizontal bar chart for better readability
        y_pos = np.arange(len(regions))
        bars = ax.barh(y_pos, values, color=colors)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(regions)
        ax.set_xlabel('Percentage of Projects (%)')
        ax.set_title('Regional Focus in Survey Responses')
        ax.set_xlim(0, 100)
        
        # Add percentage labels
        for bar in bars:
            width = bar.get_width()
            ax.annotate(f'{width:.1f}%',
                        xy=(width, bar.get_y() + bar.get_height() / 2),
                        xytext=(3, 0),
                        textcoords="offset points",
                        ha='left', va='center')
        
        plt.tight_layout()
        
        # Save to base64 for embedding in HTML
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100)
        buffer.seek(0)
        plots['regions'] = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close(fig)
    except Exception as e:
        logger.error(f"Error creating regional focus chart: {str(e)}")
    
    # Generate complexity factors visualisation
    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Data from survey
        factors = ['Technical Requirements', 'Number of Stakeholders', 'Regulatory Requirements', 'Geographic Distribution']
        values = [60, 53.33, 46.67, 26.67]  # Percentages
        
        # Use a different colormap
        colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(factors)))
        
        # Create vertical bar chart
        x_pos = np.arange(len(factors))
        bars = ax.bar(x_pos, values, color=colors)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(factors, rotation=45, ha='right')
        ax.set_ylabel('Percentage of Responses (%)')
        ax.set_title('Project Complexity Factors')
        ax.set_ylim(0, 100)
        
        # Add percentage labels
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save to base64 for embedding in HTML
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100)
        buffer.seek(0)
        plots['complexity_factors'] = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close(fig)
    except Exception as e:
        logger.error(f"Error creating complexity factors chart: {str(e)}")
    
    # Generate experience levels visualisation (pie chart)
    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Data from survey
        experience_labels = ['1-5 years', '10-15 years', '15+ years', '5-10 years']
        experience_values = [40, 26.67, 20, 13.33]  # Percentages
        
        # Use a different colormap
        colors = plt.cm.Greens(np.linspace(0.4, 0.8, len(experience_labels)))
        
        # Create pie chart
        wedges, texts, autotexts = ax.pie(
            experience_values, 
            labels=experience_labels, 
            autopct='%1.1f%%',
            startangle=90,
            colors=colors
        )
        
        # Equal aspect ratio ensures that pie is drawn as a circle
        ax.axis('equal')
        ax.set_title('Experience Level Distribution')
        
        # Enhance text visibility
        for text in texts:
            text.set_fontsize(10)
        for autotext in autotexts:
            autotext.set_fontsize(10)
            autotext.set_fontweight('bold')
        
        plt.tight_layout()
        
        # Save to base64 for embedding in HTML
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100)
        buffer.seek(0)
        plots['experience_levels'] = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close(fig)
    except Exception as e:
        logger.error(f"Error creating experience levels chart: {str(e)}")
    
    # Generate communication challenges visualisation
    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Data from survey
        challenges = ['Technical Barriers', 'Time Zone Coordination', 'Documentation Standards',
                      'Meeting Formats', 'Other']
        values = [35.71, 28.57, 21.43, 7.14, 7.14]  # Percentages
        
        # Use a different colormap
        colors = plt.cm.magma(np.linspace(0.2, 0.8, len(challenges)))
        
        # Horizontal bar chart
        y_pos = np.arange(len(challenges))
        bars = ax.barh(y_pos, values, color=colors)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(challenges)
        ax.set_xlabel('Percentage of Responses (%)')
        ax.set_title('Communication Challenges')
        ax.set_xlim(0, 100)
        
        # Add percentage labels
        for bar in bars:
            width = bar.get_width()
            ax.annotate(f'{width:.1f}%',
                        xy=(width, bar.get_y() + bar.get_height() / 2),
                        xytext=(3, 0),
                        textcoords="offset points",
                        ha='left', va='center')
        
        plt.tight_layout()
        
        # Save to base64 for embedding in HTML
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100)
        buffer.seek(0)
        plots['communication_challenges'] = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close(fig)
    except Exception as e:
        logger.error(f"Error creating communication challenges chart: {str(e)}")
    
    # Generate industry sectors visualisation
    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Data from survey
        sectors = ['Technology', 'Other Sectors', 'Finance', 'Manufacturing', 'Healthcare']
        values = [46.67, 33.33, 20, 6.67, 0]  # Percentages
        
        # Use a different colormap
        colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(sectors)))
        
        # Create vertical bar chart
        x_pos = np.arange(len(sectors))
        bars = ax.bar(x_pos, values, color=colors)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(sectors, rotation=45, ha='right')
        ax.set_ylabel('Percentage of Responses (%)')
        ax.set_title('Industry Sectors')
        ax.set_ylim(0, 100)
        
        # Add percentage labels
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save to base64 for embedding in HTML
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100)
        buffer.seek(0)
        plots['industry_sectors'] = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close(fig)
    except Exception as e:
        logger.error(f"Error creating industry sectors chart: {str(e)}")
    
    return plots
    
@web_app.route('/survey_insights')
def survey_insights_view():
    """
    Display survey insights.
    
    This route shows detailed survey insights, including regional distribution,
    complexity factors, and communication challenges.
    
    Returns:
        str: Rendered HTML template
    """
    # Generate all visualisations
    plots = generate_survey_visualisations()
    
    # Set survey data for the template
    survey_data = {
        "response_count": 15,
        "top_region": "Europe (60%)",
        "top_complexity": "Technical Requirements (60%)",
        "top_communication": "Technical Barriers (35.71%)",
        "experience_levels": {
            "1-5 years": 40.0,
            "5-10 years": 13.33,
            "10-15 years": 26.67,
            "15+ years": 20.0
        },
        "regions": {
            "Europe": 0.6000,
            "Africa": 0.5333,
            "Asia Pacific": 0.1333,
            "North America": 0.1333,
            "South America": 0,
            "Middle East": 0
        },
        "complexity_factors": {
            "Technical Requirements": 0.6000,
            "Number of Stakeholders": 0.5333,
            "Regulatory Requirements": 0.4667,
            "Geographic Distribution": 0.2667
        },
        "communication_challenges": {
            "Technical Barriers": 0.3571,
            "Time Zone Coordination": 0.2857,
            "Documentation Standards": 0.2143,
            "Meeting Formats": 0.0714,
            "Other": 0.0714
        },
        "industry_sectors": {
            "Technology": 0.4667,
            "Other Sectors": 0.3333,
            "Finance": 0.2000,
            "Manufacturing": 0.0667,
            "Healthcare": 0
        }
    }
    
    return render_template('survey_insights.html', survey=survey_data, plots=plots)

@web_app.route('/assess', methods=['GET', 'POST'])
def assess_project():
    """
    Handle project assessment requests.
    
    GET: Display the assessment form
    POST: Process form submission and display results
    
    Returns:
        str: Rendered HTML template
    """
    global model, hofstede_data, survey_insights, data_processor

    # Create form instance
    form = ProjectAssessmentForm()

    # Define the list of valid countries
    valid_countries = [
        'Canada', 'China', 'Egypt', 'France', 'Germany',
        'India', 'Italy', 'Japan', 'Kenya', 'Mozambique',
        'Nigeria', 'Portugal', 'South Africa', 'Spain',
        'United Kingdom', 'United States'
    ]

    # Dynamically set the choices for countries
    form.countries.choices = [(country, country) for country in valid_countries]

    if request.method == 'GET':
        return render_template('assess.html', form=form, survey=survey_insights)

    elif request.method == 'POST':
        # Log the entire request for debugging
        logger.info(f"Assess POST request received")
        logger.info(f"Form data: {request.form}")

        # Validate CSRF token explicitly
        if not form.csrf_token.validate(form):
            logger.error("CSRF token validation failed")
            flash("Invalid form submission. Please try again.", "danger")
            return render_template('assess.html', form=form, survey=survey_insights)

        if form.validate_on_submit():
            try:
                # Extract form data
                project_data = {
                    'project_name': form.project_name.data,
                    'project_type': form.project_type.data,
                    'industry_sector': form.industry_sector.data,
                    'primary_region': form.primary_region.data,
                    'project_complexity': form.project_complexity.data,
                    'technical_requirements': form.technical_requirements.data,
                    'stakeholder_count': form.stakeholder_count.data,
                    'team_size': form.team_size.data,
                    'project_duration': form.project_duration.data,
                    'team_diversity': form.team_diversity.data,
                    'virtual_team_ratio': form.virtual_team_ratio.data,
                    'language_barriers': form.language_barriers.data,
                    'communication_barriers': form.communication_barriers.data,
                    'prior_collaboration': form.prior_collaboration.data
                }

                # Get selected countries (from form submission)
                selected_countries = request.form.getlist('countries')

                # EXTRA VALIDATION: Limit number of countries
                if len(selected_countries) > 5:
                    flash("Please select no more than 5 countries.", "warning")
                    return render_template('assess.html', form=form, survey=survey_insights)

                # DETAILED LOGGING
                logger.info("Project Assessment Request")
                logger.info(f"Project Name: {project_data['project_name']}")
                logger.info(f"Selected Countries: {selected_countries}")
                logger.info(f"Primary Region: {project_data['primary_region']}")

                # Check if countries were selected
                if not selected_countries:
                    flash("Please select at least one country for analysis.", "danger")
                    return render_template('assess.html', form=form, survey=survey_insights)

                # Validate selected countries
                invalid_countries = set(selected_countries) - set(valid_countries)
                if invalid_countries:
                    flash(f"Invalid countries selected: {', '.join(invalid_countries)}", "danger")
                    return render_template('assess.html', form=form, survey=survey_insights)

                # Calculate cultural dimension averages
                cultural_dims = ['power_distance', 'individualism', 'masculinity', 'uncertainty_avoidance',
                                 'long_term_orientation', 'indulgence']

                for dim in cultural_dims:
                    values = [hofstede_data.loc[country, dim] for country in selected_countries if country in hofstede_data.index]
                    # IMPROVEMENT: Add a check to prevent division by zero
                    project_data[dim] = sum(values) / len(values) if values else 0

                # Create DataFrame for prediction
                df = pd.DataFrame([project_data])

                # Default fallback values
                success_prob = 0.65
                risk_factors = {
                    'communication_barriers': 0.25,
                    'technical_complexity': 0.20,
                    'stakeholder_count': 0.15,
                    'uncertainty_avoidance': 0.10,
                    'power_distance': 0.10
                }
                comm_impact = 0.4
                regional_impact = {
                    project_data['primary_region']: {
                        'focus_value': survey_insights['regions'].get(project_data['primary_region'], 0.1),
                        'experience_level': 'Medium',
                        'risk_level': 'Medium'
                    }
                }
                recommendations = [
                    "Improve communication protocols and establish clear communication channels.",
                    "Provide cultural awareness training and establish cross-cultural team-building activities.",
                    "Enhance technical documentation and establish clear technical requirements."
                ]

                # Attempt model prediction if possible
                if hasattr(model, 'model') and model.model is not None:
                    try:
                        success_prob = model.predict(df)[0]
                        risk_factors = model.identify_risk_factors(df)
                        comm_impact = model.calculate_communication_impact(df)
                        regional_impact = model.assess_regional_impact([project_data['primary_region']])
                        recommendations = model.generate_recommendations(df, risk_factors, success_prob)
                    except Exception as e:
                        logger.error(f"Model prediction failed: {str(e)}")
                        flash("Model prediction encountered an issue. Using simulated results.", "warning")

                # Generate plots
                try:
                    plots = create_visualisations(
                        project_data,
                        selected_countries,
                        risk_factors,
                        success_prob,
                        comm_impact,
                        regional_impact
                    )
                except Exception as plot_error:
                    logger.error(f"Error creating visualisations: {str(plot_error)}")
                    plots = create_placeholder_visualisations()

                return render_template('results.html', project=project_data,
                                       countries=selected_countries,
                                       success_prob=success_prob,
                                       risk_factors=risk_factors,
                                       comm_impact=comm_impact,
                                       regional_impact=regional_impact,
                                       recommendations=recommendations,
                                       survey=survey_insights,
                                       hofstede_data=hofstede_data,
                                       plots=plots
                                       )

            except Exception as e:
                logger.error(f"Error processing assessment: {str(e)}")
                flash(f"Error processing assessment: {str(e)}", "danger")
                return redirect(url_for('web_app.assess_project'))

        else:
            # Form validation failed
            for field, errors in form.errors.items():
                for error in errors:
                    flash(f"Error in {getattr(form, field).label.text}: {error}", "danger")

            return render_template('assess.html', form=form, survey=survey_insights)


@web_app.route('/upload', methods=['GET', 'POST'])
def upload_data():
    """
    Handle data upload requests.
    
    GET: Display the upload form
    POST: Process file upload and redirect to appropriate page
    
    Returns:
        str: Rendered HTML template or redirect response
    """
    global data_processor, hofstede_data, survey_insights, data_dir, upload_folder

    # Use the UploadForm from forms.py
    form = UploadForm()

    if request.method == 'GET':
        return render_template('upload.html', form=form)

    elif request.method == 'POST':
        try:
            # Log request details
            logger.info(f"Upload request received. Method: {request.method}")
            logger.info(f"Form data: {request.form}")
            logger.info(f"Files: {request.files}")

            # Validate form submission
            if not form.validate_on_submit():
                # Log form validation errors
                logger.warning(f"Form validation failed: {form.errors}")
                for field, errors in form.errors.items():
                    for error in errors:
                        flash(f"Error in {getattr(form, field).label.text}: {error}", "danger")
                return render_template('upload.html', form=form)

            # Check if file is provided
            if 'file' not in request.files:
                logger.warning("No file in request")
                flash("No file selected. Please select a file to upload.", "danger")
                return render_template('upload.html', form=form)

            file = request.files['file']

            # Check if filename is empty
            if file.filename == '':
                logger.warning("Empty filename")
                flash("No file selected. Please select a file to upload.", "danger")
                return render_template('upload.html', form=form)

            # Determine upload type
            upload_type = request.form.get('upload_type', 'generic')
            append_data = request.form.get('append_data') == 'on'

            # Secure filename and save
            filename = secure_filename(file.filename)
            filepath = os.path.join(upload_folder, filename)
            file.save(filepath)

            logger.info(f"File saved: {filepath}")

            # Process the uploaded file
            try:
                df = data_processor.parse_uploaded_csv(filepath)

                if df.empty:
                    logger.error("Parsed DataFrame is empty")
                    flash("Could not parse uploaded file. Please check the file format.", "danger")
                    return render_template('upload.html', form=form)

                # Log dataframe details
                logger.info(f"Parsed DataFrame shape: {df.shape}")
                logger.info(f"Parsed DataFrame columns: {list(df.columns)}")

                # Process based on upload type
                if upload_type == 'survey':
                    survey_path = os.path.join(data_dir, 'survey_results.csv')

                    # Check if the uploaded file has expected columns for survey data
                    expected_survey_columns = ['complexity_factors', 'communication_barriers', 'regions']
                    actual_columns = set(df.columns.str.lower())

                    if not any(col.lower() in actual_columns for col in expected_survey_columns):
                        logger.warning(f"Uploaded file doesn't appear to be survey data. Columns: {list(df.columns)}")
                        flash("This file doesn't appear to be survey data. Please upload a valid survey results file.", "danger")
                        return render_template('upload.html', form=form)

                    if append_data and os.path.exists(survey_path):
                        existing_df = pd.read_csv(survey_path)
                        combined_df = pd.concat([existing_df, df], ignore_index=True)
                        combined_df.to_csv(survey_path, index=False)
                        flash("Survey data appended to existing dataset.", "success")
                    else:
                        df.to_csv(survey_path, index=False)
                        flash("Survey data uploaded successfully.", "success")

                    # Update survey insights
                    survey_insights = data_processor.process_survey_data(survey_path)

                    return redirect(url_for('web_app.survey_insights_view'))

                # Add other upload type handling
                if upload_type == 'training':
                    training_path = os.path.join(data_dir, 'project_data.csv')

                    replace_existing = request.form.get('replace_existing') == 'on'

                    if replace_existing or not os.path.exists(training_path):
                        df.to_csv(training_path, index=False)
                        flash("Training data saved successfully.", "success")
                    else:
                        existing_df = pd.read_csv(training_path)
                        combined_df = pd.concat([existing_df, df], ignore_index=True)
                        combined_df.to_csv(training_path, index=False)
                        flash("Training data appended to existing dataset.", "success")

                flash("File uploaded successfully.", "success")
                return redirect(url_for('web_app.index'))

            except Exception as parse_error:
                # Log detailed error information
                logger.error(f"CSV parsing error: {str(parse_error)}")
                logger.error(traceback.format_exc())  # Add full traceback
                flash(f"Error processing CSV: {str(parse_error)}", "danger")
                return render_template('upload.html', form=form)

        except Exception as e:
            # Log unexpected errors
            logger.error(f"Unexpected upload error: {str(e)}")
            logger.error(traceback.format_exc())  # Add full traceback
            flash(f"Error uploading file: {str(e)}", "danger")
            return render_template('upload.html', form=form)


def load_survey_data():
    """
    Load survey data from data file or return default values if file doesn't exist.
    
    Returns:
        dict: Survey data dictionary with default values if necessary
    """
    global data_processor, data_dir

    try:
        survey_path = os.path.join(data_dir, 'survey_results.csv')
        if os.path.exists(survey_path):
            return data_processor.process_survey_data(survey_path)
        else:
            # Return default structure with empty dictionaries and minimal data
            return {
                "complexity_factors": {"Technical Requirements": 60.0, "Stakeholders": 50.0},
                "region_data": {"Europe": 60.0, "Africa": 53.33},
                "communication_data": {"Language Barriers": 38.46, "Technical Jargon": 30.77},
                "industry_data": {"Technology": 4.667, "Manufacturing": 6.67},
                "experience_levels": {"1-5 years": 0.4000, "5-10 years": 1.333, "10-15 years": 2.267, "15+ years": 0.2000},
                "regions": {"Europe": 0.6000, "Africa": 0.5000, "Asia Pacific": 1.333, "North America": 1.333}
            }
    except Exception as e:
        logger.error(f"Error loading survey data: {str(e)}")
        # Fallback to minimal structure
        return {
            "complexity_factors": {},
            "region_data": {},
            "communication_data": {},
            "industry_data": {},
            "experience_levels": {},
            "regions": {}
        }

@web_app.route('/train', methods=['GET', 'POST'])
def train_model():
    """
    Handle model training requests.
    
    GET: Display the training form
    POST: Process form submission and train model
    
    Returns:
        str: Rendered HTML template or redirect response
    """
    global model, data_processor, model_path, data_dir, upload_folder, survey_insights

    try:
        # Explicitly import the form to ensure it's the correct version
        from .forms import TrainModelForm

        # Create form instance
        form = TrainModelForm()

        # Check if model is currently trained
        model_trained = hasattr(model, 'model') and model.model is not None

        # Handle GET request
        if request.method == 'GET':
            return render_template('train.html',
                                   form=form,
                                   model_trained=model_trained,
                                   survey=survey_insights)

        # Handle POST request
        elif request.method == 'POST':
            # Validate form submission
            if not form.validate_on_submit():
                # Log form validation errors
                logger.warning("Form validation failed")
                for field, errors in form.errors.items():
                    for error in errors:
                        logger.warning(f"Error in {field}: {error}")
                        flash(f"Error in {getattr(form, field).label.text}: {error}", "danger")

                # Re-render the form with error messages
                return render_template('train.html',
                                       form=form,
                                       model_trained=model_trained,
                                       survey=survey_insights)

            try:
                # Initialise custom dataset path
                custom_data_path = None

                # Check if user uploaded a custom dataset
                if form.dataset.data:
                    file = form.dataset.data
                    filename = secure_filename(file.filename)
                    custom_data_path = os.path.join(upload_folder, filename)

                    # Safely save uploaded file
                    try:
                        file.save(custom_data_path)
                        logger.info(f"Custom dataset saved to {custom_data_path}")
                    except Exception as file_save_error:
                        logger.error(f"Error saving uploaded file: {str(file_save_error)}")
                        flash("Could not save uploaded file. Please try again.", "danger")
                        return render_template('train.html',
                                               form=form,
                                               model_trained=model_trained,
                                               survey=survey_insights)

                # Determine which dataset to use
                if custom_data_path and os.path.exists(custom_data_path):
                    project_data = pd.read_csv(custom_data_path)
                    logger.info(f"Using custom dataset from {custom_data_path}")
                else:
                    project_data = data_processor.load_project_data()
                    logger.info("Using default project data")

                # Validate project data
                if project_data.empty:
                    flash("No training data available. Please upload project data first.", "danger")
                    return redirect(url_for('web_app.upload_data'))

                # Validate data if requested
                if form.validate_data.data:
                    # Add your data validation logic here
                    # For example, check required columns, data types, etc.
                    logger.info("Data validation enabled")

                # Prepare data for training
                X, y = data_processor.prepare_model_data(project_data)

                if X.empty or len(y) == 0:
                    flash("Could not prepare training data. Please check your dataset.", "danger")
                    return render_template('train.html',
                                           form=form,
                                           model_trained=model_trained,
                                           survey=survey_insights)

                # Collect model training parameters
                model_params = {
                    'model_type': form.model_type.data,
                    'n_estimators': form.n_estimators.data,
                    'learning_rate': form.learning_rate.data,
                    'max_depth': form.max_depth.data,
                    'test_size': form.test_size.data,
                    'cross_validation': form.cross_validation.data
                }

                # Train the modelusing selected dataset
                if custom_data_path:
                    X, y = load_training_data_from_csv(custom_data_path)
                else:
                    X, y = data_processor.prepare_model_data(project_data)

                if X is None or y is None or X.empty or len(y) == 0:
                    flash("Could not prepare training data. Please check your dataset.", "danger")
                    return render_template("train.html", form=form)

                model.train(X, y)

                # Save the trained model
                model_path = os.path.join(data_dir, 'ciat_model.joblib')
                model.save_model(model_path)

                # Success message
                flash("Model trained successfully!", "success")
                return redirect(url_for('web_app.index'))

            except Exception as training_error:
                # Detailed logging for training errors
                logger.error(f"Error training model: {str(training_error)}")
                import traceback
                logger.error(traceback.format_exc())

                # User-friendly error message
                flash(f"Error training model: {str(training_error)}", "danger")
                return render_template('train.html',
                                       form=form,
                                       model_trained=model_trained,
                                       survey=survey_insights)

    except Exception as unexpected_error:
        # Catch any unexpected errors in the entire route
        logger.critical(f"Unexpected error in train_model route: {str(unexpected_error)}")
        import traceback
        logger.critical(traceback.format_exc())

        # User-friendly error handling
        flash("An unexpected error occurred. Please try again.", "danger")
        return redirect(url_for('web_app.index'))


@web_app.route('/compare', methods=['GET', 'POST'])
def compare_countries():
    """
    Compare cultural dimensions between countries.
    
    GET: Display the comparison form
    POST: Process form submission and display comparison
    
    Returns:
        str: Rendered HTML template or redirect response
    """
    global hofstede_data, model, survey_insights

    # Create form instance
    form = CompareCountriesForm()

    # Ensure Mozambique is in the dataset
    if 'Mozambique' not in hofstede_data.index:
        # Add placeholder values for Mozambique
        # These values are approximated - replace with actual values if available
        mozambique_values = {
            'power_distance': 85,  # High power distance
            'individualism': 20,   # Collectivist
            'masculinity': 40,     # Relatively balanced
            'uncertainty_avoidance': 60,  # Medium-high
            'long_term_orientation': 30,  # Short-term oriented
            'indulgence': 38       # Restrained
        }

        # Add to the Hofstede data
        try:
            hofstede_data.loc['Mozambique'] = mozambique_values
            logger.info("Added Mozambique to Hofstede data")
        except Exception as e:
            logger.error(f"Error adding Mozambique to Hofstede data: {str(e)}")

    # Dynamically set the choices for countries (with filter for string types)
    countries = [country for country in sorted(hofstede_data.index.tolist())
                 if isinstance(country, str) and len(country) > 1]
    form.countries.choices = [(country, country) for country in countries]

    if request.method == 'GET':
        return render_template('compare.html', form=form, countries=countries, survey=survey_insights)

    elif request.method == 'POST':
        if form.validate_on_submit():
            # Get selected countries
            selected_countries = form.countries.data

            if not selected_countries or len(selected_countries) < 2:
                flash("Please select at least two countries to compare.", "danger")
                return redirect(url_for('web_app.compare_countries'))

            # Initialise visualisation variables
            radar_plot = None
            distance_plot = None
            heatmap_plot = None
            distances = []

            # Create radar chart
            try:
                fig = model.plot_cultural_dimensions(selected_countries, hofstede_data)

                # Save radar chart to base64 for display
                buffer = io.BytesIO()
                fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
                buffer.seek(0)
                radar_plot = base64.b64encode(buffer.read()).decode('utf-8')
                plt.close(fig)
            except Exception as e:
                logger.error(f"Error creating radar chart: {str(e)}")
                flash(f"Warning: Could not create radar chart visualisation.", "warning")

            # Calculate cultural distances between country pairs
            try:
                for i in range(len(selected_countries)):
                    for j in range(i+1, len(selected_countries)):
                        country1 = selected_countries[i]
                        country2 = selected_countries[j]
                        try:
                            distance = model.calculate_cultural_distance(country1, country2, hofstede_data)
                            distances.append({
                                'country1': country1,
                                'country2': country2,
                                'distance': distance
                            })
                        except Exception as e:
                            logger.error(f"Error calculating distance between {country1} and {country2}: {str(e)}")

                # Sort distances by value (descending)
                distances.sort(key=lambda x: x['distance'], reverse=True)
            except Exception as e:
                logger.error(f"Error processing cultural distances: {str(e)}")
                flash(f"Warning: Could not calculate cultural distances.", "warning")

            # Create bar chart of cultural distances if we have distances
            try:
                if distances:
                    fig, ax = plt.subplots(figsize=(10, 6))

                    # Create labels for each country pair
                    labels = [f"{d['country1']} vs {d['country2']}" for d in distances]
                    values = [d['distance'] for d in distances]

                    # Create horizontal bar chart with numeric positions
                    y_pos = np.arange(len(labels))
                    bars = ax.barh(y_pos, values)
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels(labels)

                    # Add colour gradient
                    for i, bar in enumerate(bars):
                        bar.set_color(plt.cm.viridis(i / len(distances)))

                    ax.set_xlabel('Cultural Distance')
                    ax.set_title('Cultural Distance Between Country Pairs')

                    # Add value labels
                    for i, v in enumerate(values):
                        ax.text(v + 0.1, i, f"{v:.2f}", va='center')

                    plt.tight_layout()

                    # Save to base64 for display
                    buffer = io.BytesIO()
                    fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
                    buffer.seek(0)
                    distance_plot = base64.b64encode(buffer.read()).decode('utf-8')
                    plt.close(fig)
                else:
                    distance_plot = None
            except Exception as e:
                logger.error(f"Error creating distance plot: {str(e)}")
                flash(f"Warning: Could not create distance plot visualisation.", "warning")

            # Create heatmap of cultural dimensions
            try:
                dim_data = hofstede_data.loc[selected_countries, :]

                fig, ax = plt.subplots(figsize=(12, 8))
                sns.heatmap(dim_data, annot=True, fmt=".1f", cmap="viridis", linewidths=.5, ax=ax)
                plt.title('Cultural Dimensions by Country')
                plt.tight_layout()

                # Save to base64 for display
                buffer = io.BytesIO()
                fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
                buffer.seek(0)
                heatmap_plot = base64.b64encode(buffer.read()).decode('utf-8')
                plt.close(fig)
            except Exception as e:
                logger.error(f"Error creating heatmap: {str(e)}")
                flash(f"Warning: Could not create heatmap visualisation.", "warning")

            # Get regional insights
            regional_insights = {}
            try:
                for country in selected_countries:
                    # Map country to region (simplified approach)
                    region_map = {
                        'United Kingdom': 'Europe', 'Germany': 'Europe', 'France': 'Europe', 'Italy': 'Europe', 'Spain': 'Europe',
                        'South Africa': 'Africa', 'Nigeria': 'Africa', 'Kenya': 'Africa', 'Morocco': 'Africa', 'Egypt': 'Africa',
                        'Mozambique': 'Africa',
                        'Japan': 'Asia Pacific', 'China': 'Asia Pacific', 'India': 'Asia Pacific',
                        'United States': 'North America', 'Canada': 'North America',
                        'Brazil': 'South America', 'Argentina': 'South America', 'Colombia': 'South America',
                        'Saudi Arabia': 'Middle East', 'United Arab Emirates': 'Middle East'
                    }

                    region = region_map.get(country, 'Other')

                    regional_insights[country] = {
                        'region': region,
                        'survey_focus': survey_insights['regions'].get(region, 0) * 100  # Convert to percentage
                    }
            except Exception as e:
                logger.error(f"Error creating regional insights: {str(e)}")
                flash(f"Warning: Could not process regional insights.", "warning")

            # Render the template with all data
            return render_template('comparison_results.html', countries=selected_countries,
                                   radar_plot=radar_plot,
                                   distance_plot=distance_plot,
                                   heatmap_plot=heatmap_plot,
                                   distances=distances,
                                   hofstede_data=hofstede_data,
                                   regional_insights=regional_insights,
                                   survey=survey_insights)

        else:
            # Form validation failed
            for field, errors in form.errors.items():
                for error in errors:
                    flash(f"Error in {getattr(form, field).label.text}: {error}", "danger")

            return redirect(url_for('web_app.compare_countries'))


@web_app.route('/download/<filename>')
def download_file(filename):
    """
    Download a file from the data directory.
    
    Parameters:
        filename (str): Name of the file to download
        
    Returns:
        Flask response: File download response
    """
    global data_dir

    try:
        filepath = os.path.join(data_dir, filename)

        if os.path.exists(filepath):
            return send_file(filepath, as_attachment=True)
        else:
            flash(f"File {filename} not found.", "danger")
            return redirect(url_for('web_app.index'))
    except Exception as e:
        logger.error(f"Error downloading file: {str(e)}")
        flash(f"Error downloading file: {str(e)}", "danger")
        return redirect(url_for('web_app.index'))
    
# Helper functions for visualisations

def create_visualisations(project_data, countries, risk_factors, success_prob, comm_impact, regional_impact):
    """
    Create visualisations for assessment results.
    
    This function generates various visualisations for assessment results,
    including success probability gauge, risk factors bar chart, cultural
    dimensions radar chart, communication impact visualisation, and
    regional focus chart.
    
    Args:
        project_data (dict): Project details
        countries (list): List of countries involved in the project
        risk_factors (dict): Dictionary of risk factors and their importance
        success_prob (float): Probability of project success
        comm_impact (float): Communication impact score
        regional_impact (dict): Dictionary of regional impact assessments
        
    Returns:
        dict: Dictionary of base64-encoded plot images
    """
    global survey_insights, hofstede_data

    plots = {}

    try:
        # Success probability gauge
        fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={'projection': 'polar'})
        theta = np.linspace(0, 1.8 * np.pi, 100)
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_thetamin(0)
        ax.set_thetamax(180)

        # Create gauge colours (red to green)
        cmap = plt.cm.RdYlGn
        colours = cmap(np.linspace(0, 1, 100))

        # Draw gauge background
        for i in range(99):
            ax.barh(0, np.pi/50, left=theta[i], color=colours[i])

        # Draw needle
        angle = np.pi * success_prob
        ax.plot([0, angle], [0, 0.8], 'k-', lw=3)
        ax.plot([0, 0], [0, 0], 'k-', lw=1)

        # Add success probability text
        ax.text(np.pi/2, 0.4, f"{success_prob*100:.1f}%", ha='center', va='center', fontsize=20, fontweight='bold')

        # Add labels
        ax.text(0, 1, "Low", ha='center', va='center', fontsize=12)
        ax.text(np.pi/2, 1, "Medium", ha='center', va='center', fontsize=12)
        ax.text(np.pi, 1, "High", ha='center', va='center', fontsize=12)

        # Remove unnecessary elements
        ax.set_yticks([])
        ax.set_xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
        ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
        ax.spines['polar'].set_visible(False)

        ax.set_title('Project Success Probability', pad=20)

        # Save to base64 for display
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        plots['success_gauge'] = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close(fig)

        # Risk factors bar chart
        fig, ax = plt.subplots(figsize=(10, 6))

        # Get top 8 risk factors
        top_risks = list(risk_factors.items())[:8]
        labels = [str(item[0]).replace('_', ' ').title() for item in top_risks]
        values = [item[1] for item in top_risks]

        # Create horizontal bar chart
        y_pos = np.arange(len(labels))
        bars = ax.barh(y_pos, values)

        # Add colour gradient
        for i, bar in enumerate(bars):
            bar.set_color(plt.cm.plasma(i / len(labels)))

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()  # Labels read top-to-bottom
        ax.set_xlabel('Importance Score')
        ax.set_title('Top Risk Factors')

        # Add value labels
        for i, v in enumerate(values):
            ax.text(v + 0.01, i, f"{v:.3f}", va='center')

        plt.tight_layout()

        # Save to base64 for display
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        plots['risk_factors'] = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close(fig)

        # Cultural dimensions radar chart
        if countries:
            try:
                global model
                fig = model.plot_cultural_dimensions(countries, hofstede_data)

                # Save to base64 for display
                buffer = io.BytesIO()
                fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
                buffer.seek(0)
                plots['cultural_dimensions'] = base64.b64encode(buffer.read()).decode('utf-8')
                plt.close(fig)
            except Exception as e:
                logger.error(f"Error creating cultural dimensions chart: {str(e)}")

        # Communication impact visualisation
        fig, ax = plt.subplots(figsize=(8, 6))

        # Create a horizontal progress bar style visualisation
        ax.barh(0, 1, height=0.3, color='lightgray')
        ax.barh(0, comm_impact, height=0.3, color=plt.cm.RdYlGn_r(comm_impact))

        # Add labels
        ax.text(comm_impact, 0, f"{comm_impact:.2f}",
                ha='center', va='center', fontsize=12,
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))

        # Add category labels
        ax.text(0.1, 0.5, "Low Impact", ha='center', va='center', fontsize=10)
        ax.text(0.5, 0.5, "Medium Impact", ha='center', va='center', fontsize=10)
        ax.text(0.9, 0.5, "High Impact", ha='center', va='center', fontsize=10)

        # Remove axes
        ax.set_yticks([])
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)

        ax.set_title('Communication Barriers Impact Score')
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, 1)

        # Save to base64 for display
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        plots['communication_impact'] = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close(fig)

        # Regional focus visualisation
        if regional_impact:
            fig, ax = plt.subplots(figsize=(8, 6))

            regions = list(survey_insights['regions'].keys())
            values = [survey_insights['regions'].get(region, 0) * 100 for region in regions]

            # Highlight the primary region
            primary_region = list(regional_impact.keys())[0]
            colors = ['lightgray' if region != primary_region else 'steelblue' for region in regions]

            # Use numeric positions instead of string labels
            x_pos = np.arange(len(regions))
            ax.bar(x_pos, values, color=colors)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(regions, rotation=45, ha='right')
            ax.set_ylabel('Survey Focus (%)')
            ax.set_title('Regional Experience Level')

            # Add horizontal line for average
            avg = sum(values) / len(values)
            ax.axhline(y=avg, color='red', linestyle='--', label=f'Average: {avg:.1f}%')

            plt.legend()
            plt.tight_layout()

            # Save to base64 for display
            buffer = io.BytesIO()
            fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            plots['regional_focus'] = base64.b64encode(buffer.read()).decode('utf-8')
            plt.close(fig)

        return plots

    except Exception as e:
        logger.error(f"Error creating visualisations: {str(e)}")
        return create_placeholder_visualisations()

# https://github.com/rvarun7777/Deep_Learning/blob/master/Improving%20Deep%20Neural%20Networks_Hyperparameter%20tuning_%20Regularization/Week%203/Tensorflow%20Tutorial.py
# https://github.com/isl-org/Open3D/blob/main/examples/python/visualization/all_widgets.py
# https://github.com/thormeier/generate-placeholder-image
def create_placeholder_visualisations():
    """
    Create placeholder visualisations if the actual ones fail.
    
    This function generates simple placeholder images to display
    when visualisation creation fails for any reason.
    
    Returns:
        dict: Dictionary of base64-encoded placeholder images
    """
    plots = {}

    # Create a simple placeholder image
    def create_placeholder_image():
        """create"""
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "Visualisation unavailable",
                ha='center', va='center', fontsize=14)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        buffer = io.BytesIO()
        fig.savefig(buffer, format='png')
        buffer.seek(0)
        img_data = base64.b64encode(buffer.read()).decode()
        plt.close(fig)

        return img_data

    # Create placeholder visualisations
    plots['success_gauge'] = create_placeholder_image()
    plots['risk_factors'] = create_placeholder_image()
    plots['cultural_dimensions'] = create_placeholder_image()
    plots['communication_impact'] = create_placeholder_image()
    plots['regional_focus'] = create_placeholder_image()

    return plots

# https://python-adv-web-apps.readthedocs.io/en/latest/flask3.html
def create_default_templates(templates_dir):
    """
    Create default HTML templates if they don't exist.
    
    This function generates simple placeholder templates for the application
    if they don't already exist. These are minimal templates that can be
    customised and expanded as needed.
    
    Args:
        templates_dir (str): Directory where templates should be stored
    """
    # Create templates directory if it doesn't exist
    if not os.path.exists(templates_dir):
        os.makedirs(templates_dir)

    # Define template file paths
    templates = {
        'index.html': os.path.join(templates_dir, 'index.html'),
        'assess.html': os.path.join(templates_dir, 'assess.html'),
        'results.html': os.path.join(templates_dir, 'results.html'),
        'compare.html': os.path.join(templates_dir, 'compare.html'),
        'comparison_results.html': os.path.join(templates_dir, 'comparison_results.html'),
        'train.html': os.path.join(templates_dir, 'train.html'),
        'upload.html': os.path.join(templates_dir, 'upload.html'),
        'survey_insights.html': os.path.join(templates_dir, 'survey_insights.html'),
        'error.html': os.path.join(templates_dir, 'error.html')
    }

    # Create basic templates for each path if they don't exist
    for name, path in templates.items():
        if not os.path.exists(path):
            with open(path, 'w') as f:
                f.write(f"<!DOCTYPE html>\n<html>\n<head>\n<title>{name[:-5].title()} - CIAT</title>\n</head>\n<body>\n<h1>{name[:-5].title()}</h1>\n<p>Template for {name}</p>\n</body>\n</html>")
                logger.info(f"Created placeholder template: {path}")

def create_default_css(static_dir):
    """
    Create default CSS files if they don't exist.
    
    This function generates a basic CSS file for styling the application
    if it doesn't already exist.
    
    Args:
        static_dir (str): Directory where static files should be stored
    """
    css_dir = os.path.join(static_dir, 'css')

    # Create CSS directory if it doesn't exist
    if not os.path.exists(css_dir):
        os.makedirs(css_dir)

    # Create main.css if it doesn't exist
    main_css = os.path.join(css_dir, 'main.css')
    if not os.path.exists(main_css):
        with open(main_css, 'w') as f:
            f.write("""/* Cultural Impact Assessment Tool (CIAT) - Main CSS */

/* Rest of CSS on /ciat/static/css */
""")
        logger.info(f"Created default CSS: {main_css}")


def create_default_js(static_dir):
    """
    Create default JavaScript files if they don't exist.
    
    This function generates a basic JavaScript file for the application
    if it doesn't already exist. The JavaScript handles simple interactions
    like auto-closing flash messages.
    
    Args:
        static_dir (str): Directory where static files should be stored
    """
    js_dir = os.path.join(static_dir, 'js')

    # Create JS directory if it doesn't exist
    if not os.path.exists(js_dir):
        os.makedirs(js_dir)

    # Create main.js if it doesn't exist
    main_js = os.path.join(js_dir, 'main.js')
    if not os.path.exists(main_js):
        with open(main_js, 'w') as f:
            f.write("""// Cultural Impact Assessment Tool (CIAT) - Main JavaScript
            
/* Rest of JS in /ciat/static/js */
});
""")
        logger.info(f"Created default JavaScript: {main_js}")
        