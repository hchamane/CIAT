"""
===============================================
Cultural Impact Assessment Tool (CIAT) - Forms
===============================================

This module defines the WTForms classes used for data collection and validation
in the CIAT web interface. These forms handle project assessments, model training,
file uploads, and country comparisons.

Inspired by and adapted from:
    - https://python-adv-web-apps.readthedocs.io/en/latest/flask_forms.html
      https://wtforms.readthedocs.io/en/3.0.x/validators/
      https://flask-wtf.readthedocs.io/en/1.0.x/form/#file-uploads

Author: Hainadine Chamane
Version: 1.0.0
Date: February 2025
"""

from flask_wtf import FlaskForm
from flask_wtf.file import FileField as FlaskFileField, FileAllowed, FileRequired

from wtforms import (
    StringField, SelectField, IntegerField, SubmitField, 
    FloatField, SelectMultipleField, BooleanField, FileField
)
from wtforms.validators import (
    DataRequired, NumberRange, Optional, 
    Length, ValidationError
)


class ProjectAssessmentForm(FlaskForm):
    """
    Form for collecting project assessment data.
    
    This form captures all necessary information to evaluate the cultural impact
    on a project, including project details, complexity factors, team composition,
    and communication-related metrics.
    """
    # Project Information
    project_name = StringField(
        'Project Name',
        validators=[
            DataRequired(message="Project name is required"),
            Length(min=2, max=100, message="Project name must be between 2 and 100 characters")
        ]
    )
    
    project_type = SelectField(
        'Project Type',
        choices=[
            ('software_development', 'Software Development'),
            ('infrastructure', 'Infrastructure'),
            ('business_transformation', 'Business Transformation'),
            ('research', 'Research & Development'),
            ('marketing', 'Marketing & Events'),
            ('other', 'Other')
        ],
        validators=[DataRequired(message="Project type is required")]
    )
    
    industry_sector = SelectField(
        'Industry Sector',
        choices=[
            ('technology', 'Technology'),
            ('manufacturing', 'Manufacturing'),
            ('finance', 'Finance'),
            ('healthcare', 'Healthcare'),
            ('energy', 'Energy'),
            ('retail', 'Retail'),
            ('government', 'Government'),
            ('education', 'Education'),
            ('other', 'Other')
        ],
        validators=[DataRequired(message="Industry sector is required")]
    )
    
    primary_region = SelectField(
        'Primary Region',
        choices=[
            ('Europe', 'Europe'),
            ('Africa', 'Africa'),
            ('Asia Pacific', 'Asia Pacific'),
            ('North America', 'North America'),
            ('South America', 'South America'),
            ('Middle East', 'Middle East')
        ],
        validators=[DataRequired(message="Primary region is required")]
    )
    
    countries = SelectMultipleField(
        'Countries Involved',
        choices=[
            ('Canada', 'Canada'), ('China', 'China'), ('Egypt', 'Egypt'),
            ('France', 'France'), ('Germany', 'Germany'),
            ('India', 'India'), ('Italy', 'Italy'), ('Japan', 'Japan'),
            ('Kenya', 'Kenya'), ('Mozambique', 'Mozambique'),
            ('Nigeria', 'Nigeria'), ('Portugal', 'Portugal'),
            ('South Africa', 'South Africa'), ('Spain', 'Spain'),
            ('United Kingdom', 'United Kingdom'), ('United States', 'United States')
        ],
        validators=[Optional()]
    )
    
    # Project Complexity
    project_complexity = IntegerField(
        'Project Complexity (1-5)',
        validators=[
            DataRequired(message="Project complexity is required"),
            NumberRange(min=1, max=5, message="Project complexity must be between 1 and 5")
        ],
        default=3
    )
    
    technical_requirements = IntegerField(
        'Technical Requirements Complexity (1-5)',
        validators=[
            DataRequired(message="Technical requirements complexity is required"),
            NumberRange(min=1, max=5, message="Technical requirements complexity must be between 1 and 5")
        ],
        default=3
    )
    
    stakeholder_count = IntegerField(
        'Number of Stakeholders',
        validators=[
            DataRequired(message="Stakeholder count is required"),
            NumberRange(min=1, message="At least one stakeholder is required")
        ],
        default=10
    )
    
    # Team Composition
    team_size = IntegerField(
        'Team Size',
        validators=[
            DataRequired(message="Team size is required"),
            NumberRange(min=1, message="Team size must be at least 1")
        ],
        default=5
    )
    
    project_duration = IntegerField(
        'Project Duration (months)',
        validators=[
            DataRequired(message="Project duration is required"),
            NumberRange(min=1, message="Project duration must be at least 1 month")
        ],
        default=6
    )
    
    team_diversity = SelectField(
        'Team Cultural Diversity',
        choices=[
            ('low', 'Low - Mostly same culture'),
            ('medium', 'Medium - Some diversity'),
            ('high', 'High - Very diverse team')
        ],
        validators=[DataRequired(message="Team diversity is required")],
        default='low'
    )
    
    # Communication & Collaboration
    virtual_team_ratio = IntegerField(
        'Virtual Team Ratio (%)',
        validators=[
            DataRequired(message="Virtual team ratio is required"),
            NumberRange(min=0, max=100, message="Virtual team ratio must be between 0 and 100")
        ],
        default=50
    )
    
    language_barriers = IntegerField(
        'Language Barriers (1-5)',
        validators=[
            DataRequired(message="Language barriers level is required"),
            NumberRange(min=1, max=5, message="Language barriers must be between 1 and 5")
        ],
        default=2
    )
    
    communication_barriers = IntegerField(
        'Communication Barriers (1-5)',
        validators=[
            DataRequired(message="Communication barriers level is required"),
            NumberRange(min=1, max=5, message="Communication barriers must be between 1 and 5")
        ],
        default=2
    )
    
    prior_collaboration = IntegerField(
        'Prior Collaboration Level (1-5)',
        validators=[
            DataRequired(message="Prior collaboration level is required"),
            NumberRange(min=1, max=5, message="Prior collaboration level must be between 1 and 5")
        ],
        default=3
    )
    
    submit = SubmitField('Assess Project')


class TrainModelForm(FlaskForm):
    """
    Form for model training configuration.
    
    This form allows users to upload custom datasets and configure model
    hyperparameters for training the CIAT predictive model.
    """
    # Basic options
    dataset = FileField(
        'Custom Dataset (Optional)',
        validators=[FileAllowed(['csv'], 'CSV files only!')]
    )
    
    validate_data = BooleanField('Validate Dataset', default=True)
    show_advanced = BooleanField('Show Advanced Options', default=False)
    
    # Advanced options
    model_type = SelectField(
        'Model Type',
        choices=[
            ('gradient_boosting', 'Gradient Boosting (Default)'),
            ('random_forest', 'Random Forest')
        ],
        default='gradient_boosting'
    )
    
    cross_validation = BooleanField('Use Cross-Validation', default=False)
    
    n_estimators = IntegerField(
        'Number of Estimators',
        validators=[Optional(), NumberRange(min=50, max=500)],
        default=100,
        description="Number of trees in the ensemble"
    )
    
    learning_rate = FloatField(
        'Learning Rate',
        validators=[Optional(), NumberRange(min=0.01, max=0.5)],
        default=0.1,
        description="Controls how much each tree contributes to the final prediction"
    )
    
    max_depth = IntegerField(
        'Max Tree Depth',
        validators=[Optional(), NumberRange(min=1, max=10)],
        default=3,
        description="Maximum depth of each tree"
    )
    
    test_size = FloatField(
        'Test Set Size',
        validators=[Optional(), NumberRange(min=0.1, max=0.4)],
        default=0.2,
        description="Percentage of data used for validation"
    )
    
    submit = SubmitField('Train Model')

    def get_model_params(self):
        """
        Get model parameters if advanced options are selected.
        
        Returns:
            dict: Dictionary of model training parameters
        """
        return {
            'model_type': self.model_type.data,
            'n_estimators': self.n_estimators.data,
            'learning_rate': self.learning_rate.data,
            'max_depth': self.max_depth.data,
            'test_size': self.test_size.data
        }


class UploadForm(FlaskForm):
    """
    Form for file uploads.
    
    This form handles CSV file uploads for surveys, training data, and
    other data sources used by the CIAT application.
    """
    file = FileField(
        'Upload CSV File',
        validators=[
            FileRequired(message="Please select a file to upload."),
            FileAllowed(['csv'], message="Only CSV files are allowed!")
        ]
    )
    submit = SubmitField('Upload')


class CompareCountriesForm(FlaskForm):
    """
    Form for comparing cultural dimensions between countries.
    
    This form allows users to select multiple countries for comparison
    of their Hofstede cultural dimension values.
    """
    countries = SelectMultipleField('Countries to Compare')
    submit = SubmitField('Compare')
