"""
=================================================================================
Generate a sample training dataset for the Cultural Impact Assessment Tool (CIAT)
=================================================================================

This script generates synthetic project data for training and testing the CIAT
predictive model. The data includes project characteristics, cultural dimensions,
and success outcomes, reflecting the theoretical frameworks established in the
cultural impact assessment literature.

References:
    - Fog, A. (2022). Two-Dimensional models of cultural differences: Statistical and theoretical 
      analysis. Cross-Cultural Research, 57(2-3), 115-165. https://doi.org/10.1177/10693971221135703
    - Dumitrașcu-Băldău, I., Dumitrașcu, D.D., & Dobrotă, G. (2021). Predictive model for the factors 
      influencing international project success: A data mining approach. Sustainability, 13(7), 3819. 
      https://doi.org/10.3390/su13073819
    - Hofstede, G. (2011). Dimensionalizing cultures: The Hofstede model in context. Online Readings 
      in Psychology and Culture, 2(1). https://doi.org/10.9707/2307-0919.1014
    - Kogut, B., & Singh, H. (1988). The effect of national culture on the choice of entry mode. 
      Journal of International Business Studies, 19(3), 411-432.
    - Kim, H.G., Gaur, A.S., & Mukherjee, D. (2020). Added cultural distance and ownership in 
      cross-border acquisitions. Cross Cultural & Strategic Management, 27(3), 487-510.
      
Inspired by and adapted from:
    - https://numpy.org/doc/2.2/reference/random/parallel.html
    - https://github.com/trenton3983/DataCamp/blob/master/python_data_science_toolbox_1.py
    - https://github.com/dzianissokalau/data_generator/blob/main/data_generator.py
    - https://github.com/theodi/synthetic-data-tutorial/blob/master/DataSynthesizer/DataGenerator.py

Author: Hainadine Chamane
Version: 1.0.0
Date: February 2025
"""

import pandas as pd
import numpy as np
import csv
import os
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_sample_training_data(num_projects=50, output_file="sample_training_data.csv"):
    """
    Create a sample CSV file with training data for the CIAT model
    
    This function generates synthetic project data that reflects the cultural dimensions
    and project factors identified by Hofstede (2011), Fog (2022), and 
    Dumitrașcu-Băldău et al. (2021) as significant predictors of international 
    project success. The relationship between variables is modeled based on 
    empirical findings from cross-cultural project management research.
    
    Parameters:
        num_projects (int): Number of projects to generate
        output_file (str): Path to save the CSV file
        
    Returns:
        pandas.DataFrame: The generated sample training data
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    logger.info(f"Generating sample data for {num_projects} projects")
    
    # Create basic project data
    data = {
        'project_id': [f'PROJ-{i+1:03d}' for i in range(num_projects)],
        'project_name': [f'Project {i+1}' for i in range(num_projects)],
        
        # Regional distribution based on survey data (57.14% Europe, 50% Africa)
        'primary_region': np.random.choice(
            ['Europe', 'Africa', 'Asia Pacific', 'North America', 'South America', 'Middle East'], 
            num_projects,
            p=[0.30, 0.25, 0.15, 0.15, 0.10, 0.05]  # Weighted distribution
        ),
        
        # Project types based on research framework
        'project_type': np.random.choice(
            ['IT', 'Engineering', 'Construction', 'Consulting', 'Research', 'Other'], 
            num_projects
        ),
        
        # Industry sectors distribution informed by survey data
        'industry_sector': np.random.choice(
            ['Technology', 'Finance', 'Healthcare', 'Education', 'Manufacturing', 'Energy', 'Other'], 
            num_projects,
            p=[0.30, 0.15, 0.10, 0.10, 0.15, 0.10, 0.10]  # Weighted distribution
        ),
        
        # Project complexity factors (Dumitrașcu-Băldău et al. 2021)
        'project_complexity': np.random.randint(1, 6, num_projects),  # Scale 1-5
        'technical_requirements': np.random.randint(1, 6, num_projects),  # Scale 1-5 (57.14% primary factor)
        'stakeholder_count': np.random.randint(3, 51, num_projects),  # 3-50 stakeholders (50% complexity factor)
        
        # Team composition factors
        'team_size': np.random.randint(2, 31, num_projects),  # 2-30 team members
        'project_duration': np.random.randint(1, 37, num_projects),  # 1-36 months
        'team_diversity': np.random.choice(['low', 'medium', 'high'], num_projects),
        'virtual_team_ratio': np.random.randint(0, 101, num_projects),  # 0-100%
        
        # Communication factors (Identified as key challenges in survey data)
        'language_barriers': np.random.randint(1, 6, num_projects),  # Scale 1-5
        'communication_barriers': np.random.randint(1, 6, num_projects),  # Scale 1-5 (38.46% of respondents)
        'prior_collaboration': np.random.randint(1, 6, num_projects),  # Scale 1-5
    }
    
    # Add Hofstede cultural dimensions for each project
    # Values based on Hofstede's (2011)
    logger.info("Adding Hofstede cultural dimensions")
    
    # Power Distance Index (PDI)
    data['power_distance'] = np.random.randint(20, 101, num_projects)  # 20-100
    
    # Individualism vs. Collectivism (IDV)
    data['individualism'] = np.random.randint(10, 91, num_projects)    # 10-90
    
    # Masculinity vs. Femininity (MAS)
    data['masculinity'] = np.random.randint(10, 101, num_projects)     # 10-100
    
    # Uncertainty Avoidance Index (UAI)
    data['uncertainty_avoidance'] = np.random.randint(20, 101, num_projects)  # 20-100
    
    # Long-Term vs. Short-Term Orientation (LTO)
    data['long_term_orientation'] = np.random.randint(10, 101, num_projects)  # 10-100
    
    # Indulgence vs. Restraint (IVR)
    data['indulgence'] = np.random.randint(20, 101, num_projects)      # 20-100
    
    # Create success factors based on theoretical frameworks and empirical research
    # These factors reflect the relationships identified in Dumitrașcu-Băldău et al. (2021)
    # and Fog's (2022) cross-cultural analysis
    
    logger.info("Calculating success factors based on theoretical frameworks")
    success_factors = {
        # Project complexity factors (negative impact on success)
        'complexity_factor': (6 - data['project_complexity']) * 0.05,  # Lower complexity = higher success
        'tech_factor': (6 - data['technical_requirements']) * 0.03,    # Lower tech requirements = higher success
        
        # Team composition factors
        'team_size_factor': np.where(data['team_size'] > 20, -0.05, 0.03),  # Smaller teams = higher success
        'duration_factor': np.where(data['project_duration'] > 24, -0.1, 0.05),  # Shorter projects = higher success
        
        # Communication factors (identified as critical in survey)
        'communication_factor': (6 - data['communication_barriers']) * 0.07,  # Lower barriers = higher success
        'language_factor': (6 - data['language_barriers']) * 0.05,     # Lower barriers = higher success
        'prior_collab_factor': data['prior_collaboration'] * 0.05,     # More prior collaboration = higher success
        
        # Cultural dimension factors (based on Hofstede framework)
        # Power Distance - High power distance can create communication challenges
        'pdi_factor': np.where(data['power_distance'] > 70, -0.05, 0.02),
        
        # Individualism - Very low individualism can impact decision making in int'l projects
        'idv_factor': np.where(data['individualism'] < 30, -0.03, 0.03),
        
        # Uncertainty Avoidance - High UAI can impact projects with high uncertainty
        'uai_factor': np.where(
            data['uncertainty_avoidance'] > 80,
            np.where(data['project_complexity'] > 3, -0.07, -0.02),
            0.01
        ),
        
        # Team diversity factor - Kim et al. (2020) suggests diversity impact depends on collaboration
        'diversity_factor': np.where(
            data['team_diversity'] == 'high', 
            np.where(data['prior_collaboration'] >= 3, 0.1, -0.1),  # High diversity beneficial only with good collaboration
            0.05  # Low/medium diversity generally positive
        ),
        
        # Regional experience factor - Based on survey data, more experience with Europe and Africa
        'regional_factor': np.where(
            np.isin(data['primary_region'], ['Europe', 'Africa']), 
            0.05,  # More experience with these regions
            -0.03  # Less experience with other regions
        ),
    }
    
    # Calculate base success probability
    logger.info("Calculating project success probabilities")
    base_prob = np.random.uniform(0.4, 0.6, num_projects)  # Random base probability between 0.4-0.6
    
    # Apply all success factors
    success_prob = base_prob.copy()
    for factor_name, factor in success_factors.items():
        success_prob += factor
    
    # Ensure probability is between 0 and 1
    success_prob = np.clip(success_prob, 0.1, 0.9)
    
    # Generate binary success outcome (1 = success, 0 = failure)
    data['project_success'] = np.random.binomial(1, success_prob)
    
    # Add performance metrics for successful projects (influenced by success probability)
    logger.info("Adding performance metrics")
    data['schedule_variance'] = np.random.normal(0, 0.15, num_projects) + (1 - data['project_success'] * 0.3)
    data['budget_performance'] = np.random.normal(0.8, 0.1, num_projects) * data['project_success'] + 0.2
    data['stakeholder_satisfaction'] = np.random.normal(0.7, 0.15, num_projects) * data['project_success'] + 0.3
    data['quality_metrics'] = np.random.normal(0.75, 0.1, num_projects) * data['project_success'] + 0.25
    data['team_performance'] = np.random.normal(0.8, 0.1, num_projects) * (data['project_success'] * 0.8 + 0.2)
    
    # Clip metrics to valid ranges (0-1)
    metrics = ['schedule_variance', 'budget_performance', 'stakeholder_satisfaction', 'quality_metrics', 'team_performance']
    for metric in metrics:
        data[metric] = np.clip(data[metric], 0, 1)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Save to CSV
    df.to_csv(output_file, index=False)
    logger.info(f"Sample training data saved to: {output_file}")
    logger.info(f"Success rate: {df['project_success'].mean():.2%}")
    
    return df

def generate_hofstede_data(output_file="hofstede_dimensions.csv"):
    """
    Generate a CSV file with Hofstede's cultural dimensions for various countries
    
    Based on Hofstede's (2011) framework of six cultural dimensions:
    - Power Distance (PDI)
    - Individualism vs. Collectivism (IDV)
    - Masculinity vs. Femininity (MAS)
    - Uncertainty Avoidance (UAI)
    - Long-Term vs. Short-Term Orientation (LTO)
    - Indulgence vs. Restraint (IVR)
    
    Parameters:
        output_file (str): Path to save the CSV file
    
    Returns:
        pandas.DataFrame: The generated Hofstede dimensions data
    """
    logger.info("Generating Hofstede dimensions data")
    
    # Real Hofstede dimension values for selected countries
    # Values are approximations based on Hofstede's research
    hofstede_data = {
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
        'Mozambique': [85, 20, 40, 60, 30, 38],  # Estimated values
        
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
    
    # Define column names
    dimensions = [
        'power_distance', 'individualism', 'masculinity',
        'uncertainty_avoidance', 'long_term_orientation', 'indulgence'
    ]
    
    # Create DataFrame
    df = pd.DataFrame.from_dict(
        hofstede_data, orient='index', columns=dimensions
    )
    
    # Save to CSV
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    df.to_csv(output_file)
    logger.info(f"Hofstede dimensions data saved to: {output_file}")
    
    return df

def generate_survey_data(output_file="survey_results.csv", num_responses=15):
    """
    Generate synthetic survey results based on the CIAT research findings
    
    This function creates sample survey data that mimics the distribution of responses
    from the actual survey referenced in the CIAT research, where 57.14% of respondents
    manage projects in Europe and 50% in Africa, with technical requirements (57.14%)
    being the primary complexity factor.
    
    Parameters:
        output_file (str): Path to save the CSV file
        num_responses (int): Number of survey responses to generate
        
    Returns:
        pandas.DataFrame: The generated survey data
    """
    logger.info(f"Generating survey data with {num_responses} responses")
    
    # Set random seed for reproducibility
    np.random.seed(43)  # Different seed from sample_training_data
    
    # Generate synthetic survey responses
    data = []
    
    for i in range(num_responses):
        # Region selection (multiple choice)
        regions = np.random.choice(
            ['Europe', 'Africa', 'Asia Pacific', 'North America', 'South America', 'Middle East'],
            size=np.random.randint(1, 3),  # Each respondent manages projects in 1-2 regions
            replace=False,
            p=[0.35, 0.30, 0.10, 0.10, 0.05, 0.10]  # Based on survey findings
        )
        
        # Complexity factors (multiple choice)
        complexity_factors = np.random.choice(
            ['Technical Requirements', 'Number of Stakeholders', 'Regulatory Requirements', 'Geographic Distribution'],
            size=np.random.randint(1, 3),  # Each respondent selects 1-2 factors
            replace=False,
            p=[0.35, 0.30, 0.25, 0.10]  # Based on survey findings
        )
        
        # Communication challenges (multiple choice)
        communication_challenges = np.random.choice(
            ['Technical Barriers', 'Time Zone Coordination', 'Documentation Standards', 'Meeting Formats', 'Other'],
            size=np.random.randint(1, 3),  # Each respondent selects 1-2 challenges
            replace=False,
            p=[0.30, 0.25, 0.20, 0.15, 0.10]  # Based on survey findings
        )
        
        # Industry sector (single choice)
        industry_sector = np.random.choice(
            ['Technology', 'Finance', 'Manufacturing', 'Healthcare', 'Education', 'Other'],
            p=[0.40, 0.20, 0.10, 0.05, 0.05, 0.20]  # Based on survey findings
        )
        
        # Experience level (single choice)
        experience_level = np.random.choice(
            ['1-5 years', '5-10 years', '10-15 years', '15+ years'],
            p=[0.40, 0.15, 0.25, 0.20]  # Based on survey findings
        )
        
        # Create response record
        response = {
            'response_id': f'RESP-{i+1:03d}',
            'regions': ';'.join(regions),
            'complexity_factors': ';'.join(complexity_factors),
            'communication_challenges': ';'.join(communication_challenges),
            'industry_sector': industry_sector,
            'experience_level': experience_level,
            'virtual_team_percentage': np.random.randint(0, 101),  # 0-100%
            'international_projects_per_year': np.random.randint(1, 11),  # 1-10 projects
        }
        
        data.append(response)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Save to CSV
    df.to_csv(output_file, index=False)
    logger.info(f"Survey data saved to: {output_file}")
    
    return df


if __name__ == "__main__":
    # Create output directory if it doesn't exist
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    
    # Generate Hofstede dimensions data
    hofstede_file = output_dir / "hofstede_dimensions.csv"
    hofstede_df = generate_hofstede_data(output_file=hofstede_file)
    
    # Generate sample training data
    training_file = output_dir / "sample_training_data.csv"
    training_df = create_sample_training_data(num_projects=50, output_file=training_file)
    
    # Generate synthetic survey data
    survey_file = output_dir / "survey_results.csv"
    survey_df = generate_survey_data(output_file=survey_file, num_responses=15)
    
    # Display summary of training data
    print("\nTraining Dataset Summary:")
    print(f"Total projects: {len(training_df)}")
    print(f"Successful projects: {training_df['project_success'].sum()} ({training_df['project_success'].mean():.2%})")
    print(f"Failed projects: {len(training_df) - training_df['project_success'].sum()} ({1 - training_df['project_success'].mean():.2%})")
    
    # Display regional distribution in survey data
    if 'regions' in survey_df.columns:
        print("\nSurvey Regional Distribution:")
        region_counts = {}
        for regions_str in survey_df['regions']:
            for region in regions_str.split(';'):
                region_counts[region] = region_counts.get(region, 0) + 1
        
        for region, count in sorted(region_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(survey_df)) * 100
            print(f"- {region}: {percentage:.1f}%")
    
    # Display column names for reference
    print("\nTraining data column names:")
    for col in training_df.columns:
        print(f"- {col}")
