"""
==============================================================
Cultural Impact Assessment Tool (CIAT) - Data Processor
==============================================================

Module for loading and processing data for the Cultural Impact Assessment Tool.
This module handles Hofstede cultural dimensions data, project data,
and survey results for use in the predictive model.

References:
    - Kim, H.G., Gaur, A.S., and Mukherjee, D. (2020) 'Added cultural distance and ownership in 
        cross-border acquisitions.' Cross Cultural & Strategic Management, 27(3), 487-510.
        https://doi.org/10.1108/ccsm-01-2020-0003
    - Dumitrașcu-Băldău, I., Dumitrașcu, D.-D. and Dobrotă, G. (2021) 'Predictive model for the 
        factors influencing international project success: A data mining approach,' Sustainability, 
        13(7), p. 3819. https://doi.org/10.3390/su13073819.
    - Fog, A. (2022) 'Two-Dimensional models of cultural differences: Statistical and theoretical 
        analysis,' Cross-Cultural Research, 57(2-3), pp. 115-165. https://doi.org/10.1177/10693971221135703.
    - Fiorio, C. (2023) 'Unveiling the Nexus of Cultural Dimensions, Skill, and Rival Behavior: A Codeforces.com 
        API Data Analysis. How Do Cultural Dimensions and Skill Shape the Impact of Rival Behavior on Performance? 
        Available at: https://research-api.cbs.dk/ws/portalfiles/portal/98729666/1590003_Master_Thesis.pdf.
        
Inspired by and adapted from:
    - Preub, B. (2021) 'Natural Language Processing -to Analyze Corporate Culture.' Available at: 
    - https://repository.ubn.ru.nl/bitstream/handle/2066/237332/237332.pdf?sequence=1.
    - https://github.com/lgp171188/flask-form-validation/blob/master/app.py
    - https://github.com/jovicigor/DataPreprocessor/blob/develop/datapreprocessor/engine.py
    - https://github.com/MS20190155/Measuring-Corporate-Culture-Using-Machine-Learning/blob/master/culture/culture_dictionary.py
    - https://github.com/levan92/logging-example
    - https://github.com/seita-f/Python-Data-Processing-App/blob/main/dataProcessor.py
    - https://github.com/mlabonne/llm-course
    - https://colab.research.google.com/github/jazoza/cultural-data-analysis/blob/main/05_CDA_compare_visualize.ipynb

Author: Hainadine Chamane
Version: 1.0.0
Date: February 2025
"""

# Standard library imports
import os
import csv
import json
import logging
import re
from pathlib import Path

# Data analysis imports
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ciat_data_processor.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class CulturalDataProcessor:
    """
    Data processor class for handling cultural dimensions and project management data.
    
    This class provides methods for loading and processing Hofstede cultural dimensions data,
    project data, and survey results for use in the CIAT predictive model.
    
    Attributes:
        data_dir (str): Directory containing data files
        hofstede_path (str): Path to Hofstede dimensions data
        project_data_path (str): Path to project data
        survey_data_path (str): Path to survey results data
    """
    
    def __init__(self, data_dir="data"):
        """
        Initialise the data processor with paths to data resources.
        
        Args:
            data_dir (str): Directory containing data files
        """
        self.data_dir = data_dir
        
        # Create data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Default file paths
        self.hofstede_path = os.path.join(self.data_dir, "hofstede_dimensions.csv")
        self.project_data_path = os.path.join(self.data_dir, "project_data.csv")
        self.survey_data_path = os.path.join(self.data_dir, "survey_results.csv")
        
        logger.info(f"Initialised CulturalDataProcessor with data directory: {data_dir}")
    
    def load_hofstede_data(self, filepath=None):
        """
        Load Hofstede cultural dimensions data.
        
        This method loads Hofstede's six cultural dimensions (Hofstede, 2011) from a CSV file.
        If the file doesn't exist, it creates example data with values for various countries.
        
        Args:
            filepath (str, optional): Path to Hofstede data CSV file
            
        Returns:
            pandas.DataFrame: Dataframe containing Hofstede dimensions by country
        """
        if filepath is None:
            filepath = self.hofstede_path
        
        logger.info(f"Loading Hofstede data from {filepath}")
        
        try:
            # Check if file exists
            if os.path.exists(filepath):
                hofstede_data = pd.read_csv(filepath, index_col=0)
                logger.info(f"Loaded Hofstede data with {len(hofstede_data)} countries")
                return hofstede_data
            else:
                logger.warning(f"Hofstede data file not found: {filepath}")
                # Create example data based on Hofstede's dimensions
                return self._create_example_hofstede_data(filepath)
        except Exception as e:
            logger.error(f"Error loading Hofstede data: {str(e)}")
            # Return example data as fallback
            return self._create_example_hofstede_data(filepath)
    
    def _create_example_hofstede_data(self, filepath):
        """
        Create example Hofstede cultural dimensions data.
        
        This method generates sample data based on Hofstede's cultural dimensions 
        research (Hofstede, 2011). Values are approximations of the actual 
        dimensions for selected countries across regions.
        
        Args:
            filepath (str): Path to save example data
            
        Returns:
            pandas.DataFrame: Example Hofstede dimensions data
        """
        logger.info("Creating example Hofstede dimensions data")
        
        # Example data based on Hofstede's dimensions
        example_data = {
            # European countries
            'United Kingdom': [35, 89, 66, 35, 51, 69],
            'Germany': [35, 67, 66, 65, 83, 40],
            'France': [68, 71, 43, 86, 63, 48],
            'Italy': [50, 76, 70, 75, 61, 30],
            'Spain': [57, 51, 42, 86, 48, 44],
            
            # African countries
            'South Africa': [49, 65, 63, 49, 34, 63],
            'Nigeria': [80, 30, 60, 55, 13, 84],
            'Kenya': [70, 25, 60, 50, 30, 40],
            'Morocco': [70, 46, 53, 68, 14, 25],
            'Egypt': [70, 25, 45, 80, 7, 4],
            
            # Asian countries
            'Japan': [54, 46, 95, 92, 88, 42],
            'China': [80, 20, 66, 30, 87, 24],
            'India': [77, 48, 56, 40, 51, 26],
            
            # North American countries
            'United States': [40, 91, 62, 46, 26, 68],
            'Canada': [39, 80, 52, 48, 36, 68],
            
            # South American countries
            'Brazil': [69, 38, 49, 76, 44, 59],
            'Argentina': [49, 46, 56, 86, 20, 62],
            'Colombia': [67, 13, 64, 80, 13, 83],
            
            # Middle Eastern countries
            'Saudi Arabia': [95, 25, 60, 80, 36, 52],
            'United Arab Emirates': [90, 25, 50, 80, 36, 32]
        }
        
        # Convert to DataFrame
        dimensions = [
            'power_distance', 'individualism', 'masculinity',
            'uncertainty_avoidance', 'long_term_orientation', 'indulgence'
        ]
        
        hofstede_data = pd.DataFrame.from_dict(
            example_data, orient='index', columns=dimensions
        )
        
        # Save example data
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        hofstede_data.to_csv(filepath)
        
        logger.info(f"Created and saved example Hofstede data to {filepath}")
        
        return hofstede_data
    
    def load_project_data(self, filepath=None):
        """
        Load historical project data for training the CIAT model.
        
        This method loads project data with features and success metrics from a CSV file.
        If the file doesn't exist, it creates synthetic example data for model training.
        
        Args:
            filepath (str, optional): Path to project data CSV file
            
        Returns:
            pandas.DataFrame: Dataframe containing project data with success metrics
        """
        if filepath is None:
            filepath = self.project_data_path
        
        logger.info(f"Loading project data from {filepath}")
        
        try:
            # Check if file exists
            if os.path.exists(filepath):
                project_data = pd.read_csv(filepath)
                logger.info(f"Loaded project data with {len(project_data)} projects")
                return project_data
            else:
                logger.warning(f"Project data file not found: {filepath}")
                # Create example data as fallback
                return self._create_example_project_data(filepath)
        except Exception as e:
            logger.error(f"Error loading project data: {str(e)}")
            # Return example data as fallback
            return self._create_example_project_data(filepath)
    
    def _create_example_project_data(self, filepath):
        """
        Create example project data for model training.
        
        This method generates synthetic project data with relevant features and
        success metrics. The data includes cultural dimensions, project characteristics,
        and outcome variables for model training.
        
        Args:
            filepath (str): Path to save example data
            
        Returns:
            pandas.DataFrame: Example project data
        """
        logger.info("Creating example project data")
        
        # Number of example projects
        n_projects = 100  # Increased from 50 to ensure robust data generation
        
        # Generate random project data
        np.random.seed(42)  # For reproducibility
        
        # Generate dataset
        data = {
            'project_id': [f'PROJ-{i:03d}' for i in range(1, n_projects + 1)],
            'project_name': [f'Project {i}' for i in range(1, n_projects + 1)],
            'primary_region': np.random.choice(['Europe', 'Africa', 'Asia Pacific', 'North America', 'South America', 'Middle East'], n_projects),
            'project_complexity': np.random.randint(1, 6, n_projects),
            'technical_requirements': np.random.randint(1, 6, n_projects),
            'stakeholder_count': np.random.randint(5, 51, n_projects),
            'team_size': np.random.randint(3, 31, n_projects),
            'project_duration': np.random.randint(3, 37, n_projects),
            'virtual_team_ratio': np.random.randint(0, 101, n_projects),
            'language_barriers': np.random.randint(1, 6, n_projects),
            'communication_barriers': np.random.randint(1, 6, n_projects),
            'prior_collaboration': np.random.randint(1, 6, n_projects),
            'team_diversity': np.random.choice(['Low', 'Medium', 'High'], n_projects),
            'industry_sector': np.random.choice(['Technology', 'Manufacturing', 'Finance', 'Healthcare', 'Other'], n_projects)
        }
        
        # Add cultural dimensions
        hofstede_data = self.load_hofstede_data()
        countries = hofstede_data.index.tolist()
        
        # Assign random countries
        data['primary_country'] = np.random.choice(countries, n_projects)
        
        # Add cultural dimensions from Hofstede data
        for dim in hofstede_data.columns:
            data[dim] = [hofstede_data.loc[country, dim] for country in data['primary_country']]
        
        # Success probability calculation
        base_prob = np.random.rand(n_projects) * 0.3 + 0.5  # Base 0.5-0.8 probability
        
        # Creating multiple factors affecting success
        complexity_effect = (data['project_complexity'] - 3) * -0.05
        tech_effect = (data['technical_requirements'] - 3) * -0.03
        comm_effect = (data['communication_barriers'] - 3) * -0.07
        collab_effect = (data['prior_collaboration'] - 3) * 0.04
        team_effect = (data['team_size'] - 15) * -0.005
        
        # Calculate final success probability
        success_prob = base_prob + complexity_effect + tech_effect + comm_effect + collab_effect + team_effect
        success_prob = np.clip(success_prob, 0.01, 0.99)
        
        # Ensure balanced classes
        data['project_success'] = np.random.binomial(1, success_prob)
        
        # Ensure a reasonable balance of success and failure
        target_balance = {0: 0.4, 1: 0.6}  # Aim for 40% failures, 60% successes
        current_counts = pd.Series(data['project_success']).value_counts(normalize=True)
        
        logger.info(f"Initial project_success distribution: {current_counts.to_dict()}")
        
        # Adjust for target distribution if necessary
        for class_val, target_proportion in target_balance.items():
            current_proportion = current_counts.get(class_val, 0)
            if abs(current_proportion - target_proportion) > 0.1:
                # Identify indices to modify
                class_indices = np.where(data['project_success'] == (1 - class_val))[0]
                
                # Calculate how many to change
                change_count = int(len(data['project_success']) * (target_proportion - current_proportion))
                
                # Randomly select indices to change
                change_indices = np.random.choice(class_indices, abs(change_count), replace=False)
                
                # Change those indices
                for idx in change_indices:
                    data['project_success'][idx] = class_val
        
        # Verify final distribution
        final_distribution = pd.Series(data['project_success']).value_counts(normalize=True)
        logger.info(f"Final project_success distribution: {final_distribution.to_dict()}")
        logger.info(f"Example data columns: {list(data.keys())}")
        logger.info(f"Project success column exists: {'project_success' in data}")
        logger.info(f"Project success distribution: {pd.Series(data['project_success']).value_counts().to_dict()}")

        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Additional performance/satisfaction columns
        df['schedule_variance'] = np.random.normal(0, 0.15, n_projects) + (1 - df['project_success'] * 0.3)
        df['budget_performance'] = np.random.normal(0.8, 0.1, n_projects) * df['project_success'] + 0.2
        df['stakeholder_satisfaction'] = np.random.normal(0.7, 0.15, n_projects) * df['project_success'] + 0.3
        df['quality_metrics'] = np.random.normal(0.75, 0.1, n_projects) * df['project_success'] + 0.25
        df['team_performance'] = np.random.normal(0.8, 0.1, n_projects) * (df['project_success'] * 0.8 + 0.2)
        
        # Clip metrics to valid ranges
        metrics = ['schedule_variance', 'budget_performance', 'stakeholder_satisfaction', 'quality_metrics', 'team_performance']
        for metric in metrics:
            df[metric] = np.clip(df[metric], 0, 1)
        
        # Save example data
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False)
        
        logger.info(f"Created and saved example project data to {filepath}")
        
        return df
    
    def load_survey_data(self, filepath=None):
        """
        Load survey data containing insights on international project management.
        
        This method loads survey data that provides context for model training and
        interpretation. If the file doesn't exist, it returns default survey insights.
        
        Args:
            filepath (str, optional): Path to survey data CSV file
            
        Returns:
            dict: Processed survey insights
        """
        if filepath is None:
            filepath = self.survey_data_path
        
        logger.info(f"Loading survey data from {filepath}")
        
        try:
            # Check if file exists
            if os.path.exists(filepath):
                survey_insights = self.process_survey_data(filepath)
                return survey_insights
            else:
                logger.warning(f"Survey data file not found: {filepath}")
                # Return default insights if file not found
                return self.get_default_survey_insights()
        except Exception as e:
            logger.error(f"Error loading survey data: {str(e)}")
            # Return default insights if loading fails
            return self.get_default_survey_insights()
    
    def process_survey_data(self, filepath):
        """
        Process survey data to extract key insights.
        
        This method parses and processes survey data from CSV format, handling
        various encodings and formats. It extracts key insights about regional
        distribution, complexity factors, and communication challenges.
        
        Args:
            filepath (str): Path to survey data CSV file
            
        Returns:
            dict: Processed survey insights
        """
        try:
            # Try reading with different encodings and parameters
            encodings = ['utf-8', 'latin-1', 'utf-16']
            for encoding in encodings:
                try:
                    df = pd.read_csv(filepath, encoding=encoding, low_memory=False)
                    break
                except Exception:
                    continue
            
            # If no encoding worked, raise an exception
            if 'df' not in locals():
                raise ValueError("Could not parse CSV with any encoding")
            
            # Process the parsed DataFrame
            insights = self.extract_survey_insights(df)
            return insights

        except Exception as e:
            logger.error(f"Error processing survey data: {str(e)}")
            return self.get_default_survey_insights()
    
    def extract_survey_insights(self, survey_df):
        """
        Extract insights from survey data.
        
        This method analyses the survey DataFrame to extract key insights about
        regional distribution, complexity factors, communication challenges, and
        industry sectors relevant to international project management.
        
        Args:
            survey_df (pandas.DataFrame): Survey data
            
        Returns:
            dict: Extracted insights
        """
        try:
            # Initialise insights dictionary
            insights = {}
            
            # Extract regional distribution if available
            if 'region' in survey_df.columns:
                region_counts = survey_df['region'].value_counts(normalize=True)
                insights['regions'] = region_counts.to_dict()
            else:
                insights['regions'] = {}
                
            # Fill in any missing insights with defaults
            default_insights = self.get_default_survey_insights() 
            for key, value in default_insights.items():
                if key not in insights:
                    insights[key] = value

            return insights
            
        except Exception as e:
            logger.error(f"Error extracting survey insights: {str(e)}")
            return self.get_default_survey_insights()
    
    def get_default_survey_insights(self):
        """
        Return default survey insights if data can't be loaded or processed.
        
        This method provides fallback survey data based on preliminary research
        findings. These values are used when actual survey data is unavailable.
        
        Returns:
            dict: Default survey insights
        """
        return {
            "regions": {
                "Europe": 0.5714,
                "Africa": 0.5000,
                "Asia Pacific": 0.0714,
                "North America": 0.0714,
                "South America": 0.0000,
                "Middle East": 0.0000
            },
            "complexity_factors": {
                "Technical requirements": 0.5714,
                "Number of stakeholders": 0.5000,
                "Regulatory requirements": 0.4286,
                "Geographic distribution": 0.2143
            },
            "communication_challenges": {
                "Technical barriers": 0.3846,
                "Time zone coordination": 0.2308,
                "Documentation standards": 0.2308,
                "Meeting formats": 0.0714
            },
            "industry_sectors": {
                "Technology": 0.4286,
                "Finance": 0.2143,
                "Manufacturing": 0.0714,
                "Other": 0.3571
            },
            "experience_levels": {
                "1-5 years": 0.4000,
                "5-10 years": 0.1333,
                "10-15 years": 0.2667,
                "15+ years": 0.2000
            }
        }
    
    def prepare_model_data(self, X=None, y=None, include_features=None, target='project_success'):
        """
        Prepare data for model training by selecting relevant features.
        
        This method processes input data for model training, handling feature selection,
        data validation, and target variable preparation. It ensures that the data is in
        the correct format for training the predictive model.
        
        Args:
            X (pandas.DataFrame, optional): Feature data (if None, load from project_data)
            y (pandas.Series, optional): Target variable (if None, extract from project_data)
            include_features (list, optional): Specific features to include
            target (str): Target variable name
            
        Returns:
            tuple: (X, y) where X is feature data and y is the target variable
        """
        logger.info("Preparing data for model training")
        
        try:
            # If X is not provided, load project data
            if X is None:
                project_data = self.load_project_data()
                
                # DEBUGGING: Print details about loaded data
                logger.info(f"Loaded project data shape: {project_data.shape}")
                logger.info(f"Loaded project data columns: {project_data.columns.tolist()}")
                
                # Check if project_data is valid
                if project_data is None or len(project_data) == 0:
                    logger.error("Project data is empty or None")
                    return pd.DataFrame(), pd.Series(dtype='int')
                
                # Check if target column exists
                if target not in project_data.columns:
                    logger.warning(f"Target variable '{target}' not found in project data. Creating default target.")
                    # Create a balanced target column for demonstration
                    np.random.seed(42)  # For reproducibility
                    project_data[target] = np.random.binomial(1, 0.6, len(project_data))
                    logger.info(f"Created default target variable '{target}' with {project_data[target].sum()} positive samples")
                
                # DEBUGGING: Print target column details
                logger.info(f"Target column '{target}' exists: {target in project_data.columns}")
                if target in project_data.columns:
                    logger.info(f"Target column values: {project_data[target].value_counts().to_dict()}")
                
                # Add default feature sets (make sure these match what's in your data)
                cultural_dimensions = [
                    'power_distance', 'individualism', 'masculinity',
                    'uncertainty_avoidance', 'long_term_orientation', 'indulgence'
                ]
                
                project_factors = [
                    'project_complexity', 'technical_requirements', 'stakeholder_count',
                    'team_size', 'project_duration', 'virtual_team_ratio',
                    'language_barriers', 'communication_barriers', 'prior_collaboration'
                ]
                
                categorical_features = [
                    'team_diversity', 'industry_sector', 'primary_region'
                ]
                
                # Determine which features to include
                if include_features is None:
                    include_features = []
                    
                    # Add existing columns from predefined sets
                    for col_list in [cultural_dimensions, project_factors, categorical_features]:
                        include_features.extend([col for col in col_list if col in project_data.columns])
                
                # Log the selected features
                logger.info(f"Selected features: {include_features}")
                
                # Make sure we have some features
                if len(include_features) == 0:
                    logger.error("No valid features found")
                    include_features = [col for col in project_data.columns if col != target]
                    logger.info(f"Using all non-target columns as features: {include_features}")
                
                # Extract X and y
                X = project_data[include_features]
                y = project_data[target]
                
                # DEBUGGING: Print extracted X and y
                logger.info(f"Extracted X shape: {X.shape}")
                logger.info(f"Extracted y shape: {y.shape if hasattr(y, 'shape') else len(y)}")
                logger.info(f"Extracted y values: {y.value_counts().to_dict()}")
            
            # If X and y are provided directly, validate them
            if X is not None and y is not None:
                logger.info(f"Using provided X and y: X shape = {X.shape}, y length = {len(y)}")
            
            # Ensure X and y are valid
            if X is None or y is None:
                logger.error("X or y is None after preparation")
                return pd.DataFrame(), pd.Series(dtype='int')
            
            if len(X) == 0 or len(y) == 0:
                logger.error("X or y is empty after preparation")
                return pd.DataFrame(), pd.Series(dtype='int')
            
            # Validate data types and convert if necessary
            # Coerce to binary (0/1) values for classification
            if pd.api.types.is_bool_dtype(y):
                y = y.astype(int)
                logger.info("Converted boolean target to int")
            elif pd.api.types.is_object_dtype(y):
                unique_values = y.unique()
                if len(unique_values) == 2:
                    # Map unique values to 0 and 1
                    mapping = {unique_values[0]: 0, unique_values[1]: 1}
                    y = y.map(mapping)
                    logger.info(f"Mapped target values using: {mapping}")
            
            # Ensure binary classification
            unique_targets = y.unique()
            logger.info(f"Final target unique values: {unique_targets}")
            
            if len(unique_targets) != 2:
                logger.error(f"Target must have exactly 2 unique values. Current values: {unique_targets}")
                
                # Fix the target if needed
                if len(unique_targets) < 2:
                    logger.warning("Creating binary target since current target has fewer than 2 classes")
                    y = pd.Series(np.random.binomial(1, 0.6, len(X)))
                    logger.info(f"Created binary target with distribution: {y.value_counts().to_dict()}")
            
            logger.info(f"Prepared model data with {len(X)} samples, {len(X.columns)} features")
            logger.info(f"Target distribution: {y.value_counts().to_dict()}")
            
            return X, y
        
        except Exception as e:
            logger.error(f"Error preparing model data: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Return empty dataframe and series rather than None
            return pd.DataFrame(), pd.Series(dtype='int')
    
    def save_project_assessment(self, project_data, assessment_results, filepath=None):
        """
        Save project assessment results to a file.
        
        This method serialises project assessment results to a JSON file for
        future reference and analysis. It includes both project data and
        assessment results.
        
        Args:
            project_data (pandas.DataFrame): Project details
            assessment_results (dict): Assessment results
            filepath (str, optional): Path to save assessment results
            
        Returns:
            str: Path to saved assessment file
        """
        if filepath is None:
            # Generate default filepath
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            project_name = project_data.get('project_name', [f'project_{timestamp}'])[0]
            project_name = project_name.replace(' ', '_').lower()
            filepath = os.path.join(self.data_dir, 'assessments', f'{project_name}_{timestamp}.json')
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Combine project data and assessment results
        assessment = {
            'project_data': project_data.to_dict(orient='records')[0],
            'assessment_results': assessment_results,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(assessment, f, indent=2)
        
        logger.info(f"Assessment saved to {filepath}")
        
        return filepath
    
    def load_project_assessment(self, filepath):
        """
        Load project assessment results from a file.
        
        This method loads previously saved project assessment results from a JSON file.
        
        Args:
            filepath (str): Path to assessment file
            
        Returns:
            dict: Assessment data or None if loading fails
        """
        try:
            with open(filepath, 'r') as f:
                assessment = json.load(f)
            
            logger.info(f"Assessment loaded from {filepath}")
            
            return assessment
        except Exception as e:
            logger.error(f"Error loading assessment: {str(e)}")
            return None
    
    def parse_uploaded_csv(self, filepath):
        """
        Enhanced CSV parsing method with detailed error handling.
        
        This method handles parsing of CSV files with robust error handling,
        including support for different encodings, column name cleaning, and
        data validation.
        
        Args:
            filepath (str): Path to the CSV file to be parsed
            
        Returns:
            pandas.DataFrame: Parsed CSV data or empty DataFrame if parsing fails
        """
        try:
            # List of different encodings to try
            encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
            
            for encoding in encodings:
                try:
                    # Try reading with different parameters
                    df = pd.read_csv(
                        filepath, 
                        encoding=encoding,
                        low_memory=False,  # Handle mixed data types
                        dtype=str,         # Read all columns as strings initially
                        na_filter=False,   # Prevent converting empty strings to NaN
                        skipinitialspace=True  # Skip spaces after delimiter
                    )
                    
                    # Basic validation
                    if df.empty:
                        logger.warning(f"Empty DataFrame when parsing with {encoding} encoding")
                        continue
                    
                    # Clean column names
                    df.columns = [self._clean_column_name(col) for col in df.columns]
                    
                    # Log successful parsing
                    logger.info(f"Successfully parsed CSV with {encoding} encoding")
                    logger.info(f"Columns: {list(df.columns)}")
                    logger.info(f"Shape: {df.shape}")
                    
                    return df
                
                except Exception as encoding_error:
                    logger.warning(f"Failed to parse with {encoding} encoding: {str(encoding_error)}")
            
            # If no encoding worked
            logger.error("Could not parse CSV file with any encoding")
            return pd.DataFrame()
        
        except Exception as e:
            logger.error(f"Unexpected error parsing CSV: {str(e)}")
            return pd.DataFrame()

    def _clean_column_name(self, col):
        """
        Clean column names by lowercasing, removing special characters, and replacing spaces.
        
        This method ensures consistent column naming across datasets by
        standardising column names to a common format.
        
        Args:
            col: Column name to clean
            
        Returns:
            str: Cleaned column name
        """
        try:
            # Convert to string to handle non-string inputs
            col = str(col)
            
            # Convert to lowercase
            cleaned = col.lower().strip()
            
            # Remove special characters
            cleaned = re.sub(r'[^\w\s]', '', cleaned)
            
            # Replace spaces and multiple spaces with single underscore
            cleaned = re.sub(r'\s+', '_', cleaned)
            
            # Truncate very long column names
            cleaned = cleaned[:50]
            
            return cleaned
        except Exception as e:
            logger.error(f"Error cleaning column name '{col}': {str(e)}")
            return f"column_{hash(col)}"
            