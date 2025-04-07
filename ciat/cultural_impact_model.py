"""
==============================================
Cultural Impact Assessment Tool (CIAT) - Model
==============================================

A predictive model for determining the extent of cultural impact on international 
project management success, based on cultural variables identified in Fog's (2022) 
cross-cultural study and Dumitrașcu-Băldău, Dumitrașcu and Dobrotă's (2021) research.

This implementation is informed by primary survey data showing that 60% of respondents 
manage projects in Europe and 53.33% in Africa, with technical requirements (57.14%) being 
the primary complexity factor.

References:
    - Da Cunha, H.C., Farrel, C., Floriani, D.E., Andersson, S. and Amal, M (2022) 'Toward a more in-depth measurement 
        of cultural distance: A re-evaluation of the underlying assumptions,' International Journal of Cross-Cultural Management, 
        22(1), pp. 157–188. https://doi.org/10.1177/14705958221089192. 
    - Dinçer, M.A.M., Yıldırım, M. and Dil, E. (2023) 'As an emerging market Turkish culture's quest to be 
        positioned on Meyer's cultural map,' Review of International Business and Strategy, 34(1), pp. 126–151. 
        https://doi.org/10.1108/ribs-03-2023-0023. 
    - Dumitrașcu-Băldău, I., Dumitrașcu, D.-D. and Dobrotă, G. (2021) 'Predictive model for the factors 
        influencing international project success: A data mining approach,' Sustainability, 13(7), p. 3819. 
        https://doi.org/10.3390/su13073819. 
    - Fog, A. (2022) 'Two-Dimensional models of cultural differences: Statistical and theoretical analysis,' 
        Cross-Cultural Research, 57(2–3), pp. 115–165. https://doi.org/10.1177/10693971221135703. 
    - Hofstede, G. (2011) 'Dimensionalizing Cultures: The Hofstede Model in context,' Online Readings in 
        Psychology and Culture, 2(1). https://doi.org/10.9707/2307-0919.1014.
    - Iqbal, Z. and Ergenecosar, G.T. (2024) 'Predictive Analysis of Cross-Cultural Issues in Global Software 
        Development Using AI Techniques,' set-science.com, pp. 49–51. https://doi.org/10.36287/setsci.21.9.049.
    - Kim, H.G., Gaur, A.S. and Mukherjee, D. (2020) 'Added cultural distance and ownership in cross-border acquisitions,'
        Cross-Cultural & Strategic Management, 27(3), pp. 487–510. https://doi.org/10.1108/ccsm-01-2020-0003. 
    - Kogut, B. and Singh, H. (1988). The Effect of National Culture on the Choice of Entry Mode. 
        Journal of International Business Studies, 19(3), pp.411–432. http://dx.doi.org/10.1057/palgrave.jibs.8490394.
    - Kuhn, M. (n.d.). 5 Model Training and Tuning | The caret Package. [online] topepo.github.io. 
        Available at: https://topepo.github.io/caret/model-training-and-tuning.html 
    - Masoud, R., Liu, Z., Ferianc, M., Treleaven, P. and Rodrigues, M. (2025). Cultural Alignment in Large Language Models:
        An Explanatory Analysis Based on Hofstede's Cultural Dimensions. [online] pp.8474–8503. Available at: 
        https://aclanthology.org/2025.coling-main.567.pdf.
    - Trompenaars, F. and Hampden-Turner, C. (2021) 'Riding the Waves of Culture: Understanding Diversity 
        in Global Business', 4th edition, Nicholas Brealey Publishing.
    - House, R.J., et al. (2004) 'Culture, Leadership, and Organizations: The GLOBE Study of 62 Societies', 
        Sage Publications.
        
Inspired by and adapted from:
    - https://github.com/reasonablecow/cultural_dimensions/blob/main/notebook.ipynb
    - https://github.com/streamlit/demo-culture-map/tree/main/culture_map
    - https://github.com/rishabhRsinghvi/CrowdFunding-Campaign-Success-Prediction/blob/main/Modeling.ipynb
    - https://github.com/scikit-learn/scikit-learn
    - https://github.com/Starignus/AppliedML_Python_Coursera/blob/master/Module_4.ipynb
    - https://github.com/mikensubuga/ml-trauma-triage/blob/main/models.ipynb
    - https://github.com/pb111/Data-Science-Portfolio-in-Python/blob/master/Predict_Customer_Churn_with_Python.ipynb
    - https://github.com/christinebuckler/snippets/blob/master/data_preprocess.py
    - https://github.com/mahmoudyusof/DataCamp/blob/main/machine-learning-with-tree-based-models-in-python/02-the-bias-variance-tradeoff.ipynb
    - https://github.com/jzhou60/DataCamp/tree/master/Supervised%20Learning%20with%20scikit-learn/2%20-%20Regression
    - https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/model_selection/_validation.py
    - https://github.com/ratloop/MatchOutcomeAI/blob/main/model_comparison/gradient_boosting.ipynb
    - https://github.com/asharifara/data-preprocessing
    - https://github.com/shaadclt/Data-Preprocessing-Pipeline
    - https://github.com/mlabonne/llm-course/tree/main
    - https://github.com/dmlc/xgboost/tree/master/demo/guide-python
    - https://github.com/koaning/scikit-lego
    - https://github.com/microsoft/responsible-ai-toolbox
    - https://github.com/scikit-learn-contrib/imbalanced-learn
    - https://github.com/databricks/koalas

Refinements with the help of:
    - https://github.com/marketplace/models/azure-openai/gpt-4o/playground
        
Author: Hainadine Chamane
Version: 1.0.0
Date: February 2025
"""

import numpy as np
import pandas as pd
import os
import logging
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
import joblib

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CulturalImpactModel:
    """
    Predictive model for assessing cultural impact on international project success.
    
    This class implements a machine learning model that predicts project success probability 
    based on cultural aspects and project features, using the theoretical frameworks developed 
    by Fog (2022) and Dumitrașcu-Băldău et al. (2021).

    The approach combines Hofstede's six cultural dimensions (Hofstede, 2011) with project-specific
    characteristics discovered via primary research to give a thorough assessment of how cultural 
    influences affect project results.

    References:
    - Hofstede, G. (2011). Dimensionalizing cultures: The Hofstede model in context.
      Online Readings in Psychology and Culture, 2(1), 2307-0919.
    - Dinçer, M.A.M., Yıldırım, M. and Dil, E. (2023) 'As an emerging market Turkish culture's quest to be 
        positioned on Meyer's cultural map,' Review of International Business and Strategy, 34(1), pp. 126–151.
    - https://github.com/ratloop/MatchOutcomeAI/blob/main/model_comparison/gradient_boosting.ipynb
    - https://github.com/MoinDalvs/Gradient_Boosting_Algorithms_From_Scratch?tab=readme-ov-file
    - https://github.com/marketplace/models/azure-openai/gpt-4o/playground
    - https://github.com/scikit-learn-contrib/imbalanced-learn
    """
    def __init__(self):
        """
        Initialise the Cultural Impact Model with default parameters.
        
        Sets up the model pipeline and defines cultural dimensions based on
        established theoretical frameworks (Hofstede, 2011) and survey results.
        """
        self.model = None
        self.preprocessor = None
        
        # Hofstede's cultural dimensions as per Fog (2022) and Hofstede (2011)
        self.cultural_dimensions = [
            'power_distance',            # Power Distance Index (PDI)
            'individualism',             # Individualism vs. Collectivism (IDV)
            'masculinity',               # Masculinity vs. Femininity (MAS)
            'uncertainty_avoidance',     # Uncertainty Avoidance Index (UAI)
            'long_term_orientation',     # Long-term vs. Short-term Orientation (LTO)
            'indulgence'                 # Indulgence vs. Restraint (IVR)
        ]
        
        # Project success indicators from Dumitrașcu-Băldău et al. (2021)
        self.success_indicators = [
            'schedule_variance',         # Timeline
            'budget_performance',        # Financial performance
            'stakeholder_satisfaction',  # Stakeholder satisfaction ratings
            'quality_metrics',           # Quality outcomes
            'team_performance'           # Team effectiveness
        ]
        
        # Additional factors from primary survey data
        # Survey showed technical requirements (60%) and stakeholders (53.33%) as main complexity factors
        self.project_factors = [
            'project_complexity',        # Complexity level of the project
            'technical_requirements',    # Technical complexity (60% of respondents)
            'stakeholder_count',         # Number of stakeholders (53.33% of respondents)
            'team_size',                 # Number of team members
            'project_duration',          # Expected duration in months
            'virtual_team_ratio',        # Percentage of virtual teamwork
            'language_barriers',         # Level of language barriers (1-5)
            'communication_barriers',    # Technical communication barriers (38.46% of respondents)
            'prior_collaboration'        # Previous experience working together (1-5)
        ]
        
        # Regional focus based on survey data (Europe 60%, Africa 53.33%)
        self.regional_focus = {
            'europe': 0.6000,            # 60% of respondents manage projects in Europe
            'africa': 0.5333,            # 53.33% of respondents manage projects in Africa
            'asia_pacific': 0.1333,      # 13.33% of respondents manage projects in Asia-Pacific
            'north_america': 0.1333,     # 13.33% of respondents manage projects in North America
            'south_america': 0.0000,     # No respondents manage projects in South America
            'middle_east': 0.0000        # No respondents manage projects in Middle East
        }
        
        logger.info("CulturalImpactModel initialised")
    
    def preprocess_data(self, X):
        """
        Preprocess input data for model training for cross-cultural data preparation.
        
        This method creates a preprocessing pipeline that handles both numerical
        and categorical features. Numerical features are standardised using
        StandardScaler, while categorical features are transformed using OneHotEncoder.
        The method also validates input data for consistency and completeness.
        
        Args:
            x: Input data containing features for prediction
            
        Returns:
            Preprocessed feature data ready for model training or prediction
            
        Raises:
            ValueError: If input data is empty, None, or contains invalid values
        
        References:
           - https://github.com/mlabonne/llm-course/tree/main
           - https://github.com/asharifara/data-preprocessing
           - https://github.com/shaadclt/Data-Preprocessing-Pipeline
           - https://github.com/jakobrunge/tigramite/blob/master/tigramite/data_processing.py
           - https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/compose/_column_transformer.py
           - https://github.com/marketplace/models/azure-openai/gpt-4o/playground
        """
        if X is None or len(X) == 0:
            logger.error("Input data is empty")
            raise ValueError("Input data cannot be empty")
            
        logger.info("Preprocessing data with shape: %s", X.shape)
        
        # Validate the input data types
        self.validate_input_data(X)
        
        # Identify numerical and categorical features
        numerical_features = []
        categorical_features = []
        
        # Check which columns are present in the dataframe
        for col in self.cultural_dimensions:
            if col in X.columns:
                numerical_features.append(col)
                
        for col in self.project_factors:
            if col in X.columns:
                numerical_features.append(col)
        
        # Add any categorical features if present
        potential_categorical = [
            'team_diversity', 'industry_sector', 'project_type', 'primary_region'
        ]
        
        for col in potential_categorical:
            if col in X.columns:
                categorical_features.append(col)
        
        logger.info("Numerical features: %s", numerical_features)
        logger.info("Categorical features: %s", categorical_features)
        
        # Create preprocessing pipeline
        transformers = []
        
        if numerical_features:
            numerical_transformer = Pipeline(steps=[
                ('scaler', StandardScaler())
            ])
            transformers.append(('num', numerical_transformer, numerical_features))
        
        if categorical_features:
            categorical_transformer = Pipeline(steps=[
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            transformers.append(('cat', categorical_transformer, categorical_features))
        
        # Handle the case where no transformers are available
        if not transformers:
            logger.error("No valid features found for preprocessing")
            raise ValueError("No valid features found for preprocessing")
            
        self.preprocessor = ColumnTransformer(transformers=transformers)
        
        try:
            # Fit and transform the data
            preprocessed_X = self.preprocessor.fit_transform(X)
            logger.info("Data preprocessing complete. Output shape: %s", preprocessed_X.shape)
            return preprocessed_X
        except Exception as e:
            logger.error(f"Error during preprocessing: {str(e)}")
            raise
    
    def validate_input_data(self, X):
        """
        Validate input data types and values following best practices for cross-cultural
        data integrity.
        
        This internal method checks for NaN values, inappropriate data types,
        and other potential issues in the input data. It performs in-place 
        fixes where possible, such as filling NaN values with appropriate defaults.
        
        Args:
            x: Input data to validate
            
        Raises:
            ValueError: If the data contains sequence-type values or other incompatible data types
                       
        References: 
            - https://github.com/shaadclt/Data-Preprocessing-Pipeline
            - https://pandas.pydata.org/docs/user_guide/missing_data.html
            - https://github.com/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/03.04-Missing-Values.ipynb
            - https://github.com/Hari-prasaanth/Checklist/blob/main/Secure-Coding-Practices/Secure-Coding-Practices.md
            - https://github.com/pandas-dev/pandas/blob/master/pandas/core/dtypes/missing.py
            - https://github.com/marketplace/models/azure-openai/gpt-4o/playground
        """
        for col in X.columns:
            # Check for NaN values
            if X[col].isna().any():
                logger.warning(f"Column {col} contains NaN values. Filling with appropriate values.")
                
                # Fill NaN values based on column type
                if X[col].dtype.kind in 'iuf':  # integer, unsigned int, or float
                    X[col].fillna(X[col].mean() if len(X[col].dropna()) > 0 else 0, inplace=True)
                else:
                    X[col].fillna(X[col].mode()[0] if len(X[col].dropna()) > 0 else "unknown", inplace=True)
            
            # Check for list-type values
            if any(isinstance(val, (list, tuple, dict, set)) for val in X[col].dropna()):
                logger.error(f"Column {col} contains sequence-type values (lists, tuples, etc.)")
                raise ValueError(f"Column {col} contains sequence-type values which cannot be processed")
    
    def train(self, X, y, **kwargs):
        """
        Train the predictive model on historical project data using methodology from
        Kuhn (n.d.) and Hosni (2022).
        
        This method implements the training process following Kuhn's (no date) applied 
        Model Training and Tuning approach, and includes data validation, preprocessing,
        model training, evaluation, and cross-validation.
        
        Args:
            x: Feature data for training
            y: Target variable (project success indicator)
            **kwargs: Additional parameters for model configuration:
                - model_type (str): 'gradient_boosting' or 'random_forest'
                - n_estimators (int): Number of trees
                - learning_rate (float): Learning rate for gradient boosting
                - max_depth (int): Maximum tree depth
                - test_size (float): Proportion of data to use for testing
                - cross_validation (bool): Whether to use cross-validation

        Returns:
            The trained model instance (self)
            
        Raises:
            ValueError: If training data is invalid or incompatible
            
        References:
            - https://github.com/Arnav7418/AllGraph.py
            - https://github.com/Starignus/AppliedML_Python_Coursera/blob/master/Module_4.ipynb
            - https://github.com/imdeepmind/imdeepmind.github.io/blob/main/docs/programming-languages/python/args-kwargs.md
            - https://github.com/WillKoehrsen/machine-learning-project-walkthrough/blob/master/Machine%20Learning%20Project%20Part%202.ipynb
            - https://github.com/dr-mushtaq/Machine-Learning/blob/master/Supervised_(Classification)_ML_Model_Training_and_Evulation_.ipynb
            - https://github.com/scikit-learn-contrib/imbalanced-learn/blob/master/examples/ensemble/plot_bagging_classifier.py
            - https://github.com/marketplace/models/azure-openai/gpt-4o/playground
            - https://scikit-learn.org/stable/modules/cross_validation.html
        """
        # Validate inputs
        if X is None or y is None:
            logger.error("Training data or target is None")
            raise ValueError("Training data and target cannot be None")
            
        if len(X) == 0 or len(y) == 0:
            logger.error("Training data or target is empty")
            raise ValueError("Training data and target cannot be empty")
            
        if len(X) != len(y):
            logger.error(f"Length mismatch: X has {len(X)} samples, y has {len(y)} samples")
            raise ValueError("X and y must have the same number of samples")
            
        logger.info("Training model with %d samples", len(X))
        
        # Extract model parameters from kwargs
        model_type = kwargs.get('model_type', 'gradient_boosting')
        n_estimators = kwargs.get('n_estimators', 100)
        learning_rate = kwargs.get('learning_rate', 0.1)
        max_depth = kwargs.get('max_depth', 3)
        test_size = kwargs.get('test_size', 0.2)
        use_cv = kwargs.get('cross_validation', False)
        
        # Convert y to numpy array if it is a series or dataframe
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y = y.values
            
        # Ensure y is proper format
        if hasattr(y, 'shape') and len(y.shape) > 1 and y.shape[1] > 1:
            logger.warning("Target y has more than one column, using the first column only")
            y = y[:, 0]
            
        # Check for valid values in y
        unique_y = np.unique(y)
        logger.info(f"Unique values in target: {unique_y}")
        
        # For classification tasks, ensuring at least two classes
        if len(unique_y) < 2:
            logger.error(f"Target variable has only {len(unique_y)} unique value(s), need at least 2 for classification")
            raise ValueError(f"Target variable must have at least 2 unique values for classification")
            
        try:
            # Preprocess the input data
            X_processed = self.preprocess_data(X)
            
            # Confirm X_processed is valid
            if X_processed is None:
                logger.error("Preprocessing returned None")
                raise ValueError("Preprocessing failed")
                
            logger.info(f"Preprocessed data shape: {X_processed.shape}")
            
            # Split data into training and validation sets
            X_train, X_val, y_train, y_val = train_test_split(
                X_processed, y, test_size=test_size, random_state=42
            )
            
            # Select model type
            if model_type == 'random_forest':
                self.model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=42
                )
                logger.info("Using Random Forest Classifier")
                
            else:  # Default to gradient boosting
                self.model = GradientBoostingClassifier(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    random_state=42
                )
                logger.info("Using Gradient Boosting Classifier")
            
            # Train the model
            try:
                self.model.fit(X_train, y_train)
            except Exception as e:
                logger.error(f"Error during model fitting: {str(e)}")
                logger.error(f"X_train shape: {X_train.shape}, y_train shape: {np.array(y_train).shape}")
                logger.error(f"X_train type: {type(X_train)}, y_train type: {type(y_train)}")
                raise
            
            # Evaluate the model
            val_score = self.model.score(X_val, y_val)
            y_pred = self.model.predict(X_val)
            
            # Calculate additional metrics
            precision = precision_score(y_val, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_val, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)
            
            logger.info(f"Validation accuracy: {val_score:.4f}")
            logger.info(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
            
            # Cross-validation
            if use_cv:
                # Determine appropriate number of folds based on data size and class distribution
                unique_classes, counts = np.unique(y, return_counts=True)
                min_class_count = np.min(counts)
                n_splits = min(5, min_class_count)  # Use at most 5 folds, but no more than samples in smallest class
                
                logger.info(f"Using {n_splits}-fold cross-validation based on class distribution")
                cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
                cv_scores = cross_val_score(self.model, X_processed, y, cv=cv)
                logger.info(f"Cross-validation scores: {cv_scores}")
                logger.info(f"Mean CV score: {cv_scores.mean():.4f}, Std: {cv_scores.std():.4f}")
            else:
                # Basic cross-validation with adaptive number of folds
                unique_classes, counts = np.unique(y, return_counts=True)
                min_class_count = np.min(counts)
                n_splits = min(5, min_class_count)  # Use at most 5 folds, but no more than samples in smallest class
                
                cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
                cv_scores = cross_val_score(self.model, X_processed, y, cv=cv)
                logger.info(f"Cross-validation scores: {cv_scores}")
                logger.info(f"Mean CV score: {cv_scores.mean():.4f}")
            
            return self
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise
    
    def predict(self, X):
        """
        Predict project success probability based on cultural and project factors.
        
        Args:
            x: Project data for prediction
            
        Returns:
            Array of success probabilities for each project
            
        Raises:
            ValueError: If the model has not been trained
            
        References:
            - https://github.com/dr-mushtaq/Machine-Learning/blob/master/Supervised_(Classification)_ML_Model_Training_and_Evulation_.ipynb
            - https://github.com/marketplace/models/azure-openai/gpt-4o/playground
            - https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/ensemble/_forest.py
        """
        if self.preprocessor is None or self.model is None:
            raise ValueError("Model has not been trained. Call train() first.")
        
        logger.info("Making predictions for %d instances", len(X))
        
        try:
            # Preprocess the input data
            X_processed = self.preprocessor.transform(X)
            
            # Make predictions
            return self.model.predict_proba(X_processed)[:, 1]
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise
    
    def calculate_cultural_distance(self, country1, country2, hofstede_data):
        """
        Calculate cultural distance between two countries using Hofstede dimensions.
        
        Implements the formula from Kogut and Singh (1988) as recommended by
        Kim, Gaur and Mukherjee (2020), for measuring cultural distance.
        Extends the approach with insights from Da Cunha et al. (2022) Toward a more in-depth measurement 
        of cultural distance and Beugelsdijk et al.'s (2018) meta-analysis of cultural distance effects.
        
        Args:
            country1: First country name
            country2: Second country name
            hofstede_data: DataFrame containing Hofstede dimensions by country
            
        Returns:
            Cultural distance value
            
        Raises:
            ValueError: If country data is not available or contains missing values
            
        References:
            - Kogut, B. and Singh, H. (1988). The Effect of National Culture on the Choice of Entry Mode. 
              Journal of International Business Studies, 19(3), pp.411–432.
            - Da Cunha, H.C., Farrel, C., Floriani, D.E., Andersson, S. and Amal, M (2022) 'Toward a more in-depth measurement 
              of cultural distance: A re-evaluation of the underlying assumptions,' International Journal of Cross-Cultural Management, 
              22(1), pp. 157–188.
            - https://github.com/marketplace/models/azure-openai/gpt-4o/playground
            - https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/metrics/pairwise.py
        """
       
        # Identify which countries are missing from the dataset
        missing = [c for c in [country1, country2] if c not in hofstede_data.index]
        if missing:
            raise ValueError(f"Hofstede data not available for: {', '.join(missing)}")
        
        # Extract cultural dimension values for both countries
        c1_values = hofstede_data.loc[country1, self.cultural_dimensions].values
        c2_values = hofstede_data.loc[country2, self.cultural_dimensions].values
        
        # Checks for missing (NaN) values in either country's Hofstede scores
        if np.isnan(c1_values).any() or np.isnan(c2_values).any():
            raise ValueError("Missing Hofstede values for one of the countries.")
            
        # Creates a copy of variances to safely modify (prevent divide-by-zero errors)
        variances = hofstede_data[self.cultural_dimensions].var().copy()
        variances[variances == 0] = np.nan

        # Calculate squared differences normalised by variance
        # https://github.com/dstansby/notebooks/blob/main/Calculating%20variance.ipynb
        # https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-normalize-or-standardize-a-dataset-in-python.md
        # https://github.com/marketplace/models/azure-openai/gpt-4o/playground
        squared_diffs = ((c1_values - c2_values) ** 2) / variances.values

        # Cultural distance formula (Kogut & Singh, 1988)
        cultural_distance = np.sqrt(np.nansum(squared_diffs))

        return cultural_distance
    
    def identify_risk_factors(self, project_data):
        """
        Identify specific cultural risk factors for a given project.
        
        Uses the trained model to determine which cultural factors
        contribute most significantly to potential project risks.
        
        Args:
            project_data: Project details for risk assessment
            
        Returns:
            Sorted dictionary of risk factors and their importance scores
            
        Raises:
            ValueError: If the model has not been trained
        
        References:
            - https://github.com/shap/shap
            - https://github.com/marketplace/models/azure-openai/gpt-4o/playground
            - https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/ensemble/_forest.py
            - https://github.com/slundberg/shap/blob/master/notebooks/feature_selection/credit_card_fraud_feature_selection.ipynb
        """
        if self.model is None:
            raise ValueError("Model has not been trained. Call train() first.")
        
        # Get feature names after preprocessing
        feature_names = []
        
        # For gradient boosting, we can use feature_importances_
        importances = self.model.feature_importances_
        
        # Get feature names from preprocessor
        try:
            if hasattr(self.preprocessor, 'get_feature_names_out'):
                feature_names = self.preprocessor.get_feature_names_out()
            else:
                # Fallback to generic feature names
                feature_names = [f'feature_{i}' for i in range(len(importances))]
                
        except Exception as e:
            logger.warning(f"Could not get feature names: {str(e)}")
            feature_names = [f'feature_{i}' for i in range(len(importances))]
        
        # Create a dictionary of feature importances
        risk_factors = {
            feature: importance 
            for feature, importance in zip(feature_names, importances)
        }
        
        # Sort by importance (descending)
        sorted_risk_factors = dict(
            sorted(risk_factors.items(), key=lambda x: x[1], reverse=True)
        )
        
        return sorted_risk_factors
    
    def calculate_communication_impact(self, project_data):
        """
        Calculate the impact of communication barriers on project success.
        
        Based on survey data showing communication barriers as a key challenge (38.46%).
        This method calculates a weighted score of communication-related factors
        to determine their overall impact on project success.
        
        Args:
            project_data: Project data containing communication factors
            
        Returns:
            Communication impact score (0-1, where higher means greater impact)
        
        References:
            - https://github.com/marketplace/models/azure-openai/gpt-4o/playground
            - https://github.com/pmservice/wml-sample-models/blob/master/cplex/customer-satisfaction/customer-satisfaction.ipynb
            - https://github.com/rwightman/pytorch-image-models/blob/master/timm/utils/metrics.py
        """
        # Extract communication-related factors
        comm_factors = [
            'language_barriers',
            'communication_barriers',
            'virtual_team_ratio',
            'team_size'
        ]
        
        # Check if all required factors are present
        missing_factors = [factor for factor in comm_factors if factor not in project_data.columns]
        if missing_factors:
            logger.warning("Missing communication factors: %s", missing_factors)
            # Use available factors
            comm_factors = [factor for factor in comm_factors if factor in project_data.columns]
            
        if not comm_factors:
            logger.error("No communication factors available in project data")
            return 0.5  # Return neutral impact if no data available
        
        # Calculate weighted communication impact score
        # Weights based on survey results
        weights = {
            'language_barriers': 0.25,
            'communication_barriers': 0.40,  # Highest weight based on survey (38.46%)
            'virtual_team_ratio': 0.20,
            'team_size': 0.15
        }
        
        # Normalise factors to 0-1 scale
        # https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-normalize-or-standardize-a-dataset-in-python.md
        # https://github.com/marketplace/models/azure-openai/gpt-4o/playground
        normalized_data = project_data[comm_factors].copy()
        for factor in comm_factors:
            max_val = project_data[factor].max()
            if max_val > 0:
                normalized_data[factor] = project_data[factor] / max_val
        
        # Calculate weighted score using available factors
        # https://www.geeksforgeeks.org/how-to-calculate-weighted-average-in-pandas/?utm_source=chatgpt.com
        # https://github.com/marketplace/models/azure-openai/gpt-4o/playground
        weighted_sum = 0
        weights_sum = 0
        
        for factor in comm_factors:
            # Use .iloc ensuring to get a scalar value, not a pandas Series
            # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.iloc.html
            # https://github.com/marketplace/models/azure-openai/gpt-4o/playground
            if len(normalized_data) > 0:
                factor_value = normalized_data[factor].iloc[0] if hasattr(normalized_data[factor], 'iloc') else normalized_data[factor][0]
                weighted_sum += factor_value * weights[factor]
                weights_sum += weights[factor]
        
        # Convert to 0-1 scale where 1 means highest negative impact
        if weights_sum > 0:
            impact_score = weighted_sum / weights_sum
        else:
            impact_score = 0.5  # Neutral impact
        
        return impact_score
    
    def assess_regional_impact(self, regions):
        """
        Assess the impact of geographical regions on project success.
        
        Based on survey data showing Europe (57.14%) and Africa (50%) as primary regions.
        This method evaluates regional experience levels and associated risks based on
        survey data about regional project management experience.
        
        Args:
            regions: List of region names to assess
            
        Returns:
            Dictionary of regional impact assessments including experience and risk levels
            
        References:
            - https://github.com/marketplace/models/azure-openai/gpt-4o/playground
            - https://github.com/cran/rworldmap/blob/master/R/mapCountryData.R
            - https://github.com/geodatasource/country-borders
            - https://github.com/python-visualization/folium/blob/master/examples/world-map.ipynb
        """
        # Regional factors based on survey data
        impact_scores = {}
        
        for region in regions:
            if region is None:
                logger.warning("None value in regions list, skipping")
                continue
                
            region_lower = region.lower()
            if region_lower in self.regional_focus:
                # Higher focus value = more survey respondents = more available experience
                impact_scores[region] = {
                    'focus_value': self.regional_focus[region_lower],
                    'experience_level': self._map_focus_to_experience(self.regional_focus[region_lower]),
                    'risk_level': self._map_focus_to_risk(self.regional_focus[region_lower])
                }
            else:
                impact_scores[region] = {
                    'focus_value': 0.0,
                    'experience_level': 'Unknown',
                    'risk_level': 'High'
                }
        
        return impact_scores
    
    def _map_focus_to_experience(self, focus_value):
        """
        Map regional focus to experience level.
        
        Args:
            focus_value: Regional focus value from survey data
            
        Returns:
            Experience level category (High, Medium, Low, None)
            
        References:
            - https://github.com/marketplace/models/azure-openai/gpt-4o/playground
            - https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/preprocessing/_discretization.py
        """
        if focus_value > 0.4:
            return 'High'
        elif focus_value > 0.2:
            return 'Medium'
        elif focus_value > 0:
            return 'Low'
        else:
            return 'None'
    
    def _map_focus_to_risk(self, focus_value):
        """
        Map regional focus to risk level (inverse relationship).
        
        Args:
            focus_value (float): Regional focus value from survey data
            
        Returns:
            str: Risk level category (Low, Medium, High, Very High)
        """
        if focus_value > 0.4:
            return 'Low'
        elif focus_value > 0.2:
            return 'Medium'
        elif focus_value > 0:
            return 'High'
        else:
            return 'Very High'
    
    def save_model(self, filepath):
        """
        Save the trained model to a file.
        
        This method serialises the entire model, including preprocessing pipeline
        and all associated data, using joblib for efficient storage.
        
        Args:
            filepath (str): Path where the model should be saved
            
        Raises:
            ValueError: If there is no trained model to save
        """
        if self.model is None:
            raise ValueError("No trained model to save")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'preprocessor': self.preprocessor,
            'cultural_dimensions': self.cultural_dimensions,
            'success_indicators': self.success_indicators,
            'project_factors': self.project_factors,
            'regional_focus': self.regional_focus
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath):
        """
        Load a trained model from a file.
        
        This class method creates a new model instance and populates it with
        data from a previously saved model file.
        
        Args:
            filepath (str): Path to the saved model file
            
        Returns:
            CulturalImpactModel: Loaded model instance
            
        Raises:
            Exception: If the model cannot be loaded
        """
        try:
            model_data = joblib.load(filepath)
            
            instance = cls()
            instance.model = model_data['model']
            instance.preprocessor = model_data['preprocessor']
            instance.cultural_dimensions = model_data['cultural_dimensions']
            instance.success_indicators = model_data['success_indicators']
            instance.project_factors = model_data['project_factors']
            if 'regional_focus' in model_data:
                instance.regional_focus = model_data['regional_focus']
            
            logger.info(f"Model loaded from {filepath}")
            return instance
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def plot_cultural_dimensions(self, countries, hofstede_data, figsize=(10, 10)):
        """
        Create a radar chart visualising cultural dimensions for selected countries.
        
        This visualisation method creates a radar chart comparing the Hofstede
        cultural dimensions across multiple countries, allowing for easy identification
        of cultural similarities and differences.
        
        Args:
            countries: List of country names to include in the visualisation
            hofstede_data: DataFrame containing Hofstede dimensions
            figsize: Figure size (width, height) in inches
            
        Returns:
            The created figure object
            
        Raises:
            ValueError: If a requested country is not in the dataset
            
        References:
            - https://github.com/marketplace/models/azure-openai/gpt-4o/playground
            - https://github.com/mwaskom/seaborn/blob/master/seaborn/objects.py
            - https://github.com/plotly/plotly.py/blob/master/packages/python/plotly/plotly/graph_objs/scatterpolar/__init__.py
        """
        # Ensure all countries are in the dataset
        for country in countries:
            if country not in hofstede_data.index:
                raise ValueError(f"Country {country} not found in Hofstede data")
        
        # Extract cultural dimensions for selected countries
        country_data = hofstede_data.loc[countries, self.cultural_dimensions]
        
        # Create a radar chart
        categories = self.cultural_dimensions
        N = len(categories)
        
        # Create angles for each dimension
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
        
        # Draw one axis per variable and add labels
        plt.xticks(angles[:-1], categories, size=12)
        
        # Draw ylabels
        ax.set_rlabel_position(0)
        plt.yticks([20, 40, 60, 80, 100], ["20", "40", "60", "80", "100"], size=10)
        plt.ylim(0, 100)
        
        # Plot each country with a different colour
        colours = cm.tab10(np.linspace(0, 1, len(countries)))
        
        for i, country in enumerate(countries):
            values = hofstede_data.loc[country, categories].values.flatten().tolist()
            values += values[:1]  # Close the loop
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=country, color=colours[i])
            ax.fill(angles, values, alpha=0.1, color=colours[i])
        
        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.title("Cultural Dimensions Comparison", size=15, pad=20)
        
        return fig
    
    def plot_risk_factors(self, risk_factors, top_n=10, figsize=(12, 8)):
        """
        Create a bar chart visualising the top risk factors.
        
        This method creates a horizontal bar chart showing the most important
        risk factors identified by the model, with colour gradients indicating
        the relative importance of each factor.
        
        Args:
            risk_factors: Dictionary of risk factors and their importance scores
            top_n: Number of top factors to include
            figsize: Figure size (width, height) in inches
            
        Returns:
            The created figure object
            
        References:
            - https://github.com/marketplace/models/azure-openai/gpt-4o/playground
            - https://github.com/mwaskom/seaborn/blob/master/seaborn/categorical.py
            - https://github.com/mpltools/mpltools/blob/master/mpltools/color.py
        """
        # Get top N risk factors
        top_factors = list(risk_factors.items())[:top_n]
        factor_names = [item[0] for item in top_factors]
        importance_scores = [item[1] for item in top_factors]
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot horizontal bar chart
        y_pos = np.arange(len(factor_names))
        bars = ax.barh(y_pos, importance_scores, align='center')
        
        # Add colour gradient based on importance
        for i, bar in enumerate(bars):
            bar.set_color(plt.cm.viridis(importance_scores[i] / max(importance_scores)))
        
        # Add labels and formatting
        ax.set_yticks(y_pos)
        ax.set_yticklabels(factor_names)
        ax.invert_yaxis()  # labels read top-to-bottom
        ax.set_xlabel('Importance Score')
        ax.set_title('Top Risk Factors for Project Success')
        
        # Add value labels to bars
        for i, v in enumerate(importance_scores):
            ax.text(v + 0.01, i, f"{v:.3f}", va='center')
        
        plt.tight_layout()
        return fig
    
    def calculate_success_probability(self, project_data):
        """
        Calculate overall success probability based on all factors.
        
        Args:
            project_data (pandas.DataFrame): Project data for prediction
            
        Returns:
            float: Probability of project success (0-1)
            
        Raises:
            ValueError: If the model has not been trained
        """
        if self.model is None:
            raise ValueError("Model has not been trained. Call train() first.")
        
        # Make prediction
        success_prob = self.predict(project_data)[0]
        
        return success_prob
    
    def generate_recommendations(self, project_data, risk_factors, success_prob):
        """
        Generate recommendations for improving project success.
        
        This method creates tailored recommendations based on identified risk
        factors, success probability, and project characteristics. The recommendations
        focus on mitigating cultural impact risks and improving project outcomes.
        
        Args:
            project_data: Project details
            risk_factors: Dictionary of risk factors and importance scores
            success_prob: Predicted success probability
            
        Returns:
            List of recommendations for improving project success
            
        References:
            - https://github.com/slundberg/shap
            - https://github.com/TeamHG-Memex/eli5
            - https://github.com/interpretml/interpret
            - https://github.com/marketplace/models/azure-openai/gpt-4o/playground
            - https://github.com/TeamHG-Memex/eli5/blob/master/eli5/explain.py
            - https://github.com/slundberg/shap/blob/master/shap/explainers/tree.py
            - https://github.com/interpretml/interpret/blob/master/python/interpret-core/interpret/glassbox/ebm/ebm.py
        """
        recommendations = []
        
        # Get top 5 risk factors
        top_risks = list(risk_factors.items())[:min(5, len(risk_factors))]
        
        # Generate recommendations based on top risks
        for factor_name, importance in top_risks:
            factor_name_str = str(factor_name).lower()
            
            if 'communication' in factor_name_str:
                recommendations.append(
                    "Improve communication channels and establish clear communication protocols."
                )
            elif 'cultural' in factor_name_str or 'distance' in factor_name_str:
                recommendations.append(
                    "Provide cultural awareness training and establish cross-cultural team-building activities."
                )
            elif 'technical' in factor_name_str:
                recommendations.append(
                    "Enhance technical documentation and establish clear technical requirements."
                )
            elif 'stakeholder' in factor_name_str:
                recommendations.append(
                    "Implement robust stakeholder management plan with regular engagement."
                )
            elif 'team' in factor_name_str:
                recommendations.append(
                    "Focus on team cohesion through virtual team-building and regular check-ins."
                )
        
        # Add general recommendations based on success probability
        if success_prob < 0.5:
            recommendations.append(
                "Review and revise project plan to address cultural impact factors."
            )
            recommendations.append(
                "Consider bringing in cultural experts or consultants for high-risk areas."
            )
        elif success_prob < 0.7:
            recommendations.append(
                "Implement regular monitoring of cultural impact factors throughout the project."
            )
        
        # Remove duplicates while preserving order
        unique_recommendations = []
        for rec in recommendations:
            if rec not in unique_recommendations:
                unique_recommendations.append(rec)
        
        return unique_recommendations
