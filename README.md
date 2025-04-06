# Cultural Impact Assessment Tool (CIAT)

## Overview
The **Cultural Impact Assessment Tool (CIAT)** is a predictive model and assessment tool for determining the extent of cultural impact on international project management success. Based on the research framework developed by Hainadine Chamane, it draws on cultural variables identified in Fog's (2022) cross-cultural study and Dumitrașcu-Băldău, Dumitrașcu and Dobrotă's (2021) research on factors influencing international project success.


## Features
- **Cultural Distance Analysis**: Calculates and visualises cultural distances between countries
- **Country Comparison**: Interactive tool to compare cultural dimensions between countries
- **Data Visualisation**: Visual representations of cultural dimensions, risk factors, and regional insights
- **Machine Learning Model**: Predicts project success probabilities based on cultural dimensions and project-related factors
- **Recommendation Engine**: Provides customized recommendations to mitigate cultural risks
- **Risk Factor Analysis**: Identifies key risk factors that could impact project success
- **Success Probability Prediction**: Estimate overall project success probability
- **Survey Integration**: Uses research-based survey data to improve predictions
- **Web Interface**: User-friendly Flask web application with interactive visualisations

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)
- Virtual environment (recommended)

### Setup
1. Clone the repository:
```bash
git clone https://github.com/hchamane/CIAT.git
cd cultural-impact-tool
```

1. Create and activate a virtual environment:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

1. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Application
Start the web interface:
```bash
python run.py web
```
The application will be available at http://localhost:5000

### Web Interface Features

The web interface provides the following functionality:

1. **Dashboard**: Overview of survey insights and project statistics
2. **Assess Project**: Evaluate cultural impact on project success
3. **Compare Country**: Compare cultural dimensions between countries
4. **Train Model**: Train the predictive model with custom datasets
5. **Upload Data**: Upload survey data, project data, and other datasets
6. **Survey Insights**: Analysis of survey responses from 15 project management

## Project Structure
```
CIAT/
│
├── ciat/                       
│   ├── __init__.py
│   ├── cultural_impact_model.py
│   ├── data_processor.py
│   ├── forms.py
│   ├── models.py
│   ├── web_interface.py
│   │
│   ├── data/                   
│   │	 ├── assessments/
│   │	 │    └── test_20250306_213248.json
│   │	 ├── hofstede_dimensions.csv  
│   │  	 ├── project_data.csv    
│   │  	 ├── survey_results.csv
│   │  	 ├── SurveyHero-Report.csv   
│   │  	 ├── TrainingData.csv 
│   │  	 ├── TrainingData_Test.csv
│   │  	 └── ciat_model.joblib 
│   │
│   ├── static/
│   │	 ├── css
│   │	 ├── fonts
│   │ 	 ├── images
│   │	 └── js
│   │
│   ├── templates/
│   │	 ├── assess.html
│   │	 ├── base.html
│   │	 ├── compare.html
│   │	 ├── comparison_results.html
│   │	 ├── error.html
│   │	 ├── index.html
│   │	 ├── results.html
│   │	 ├── survey_insights.html
│   │	 ├── train.html
│   │	 └── upload.html         
│   │      
│   ├── uploads/
│   │
│   ├── utils/
│   │   ├── generate_sample_data.py
│   │   ├── logger.py
│   │   ├── init_app_parts.py
│   │   └── visualisation_utils.py
│   │
├── README.md 
├── requirements.txt 
├── run.py 
├── setup.py 
└── strucuture.txt
```

## Usage Guide

### Project Assessment
1. Navigate to the "Assess" page
2. Fill in the project details including countries involved
3. Submit the form to receive a comprehensive cultural impact assessment
4. Review the success probability, risk factors, and recommendations

### Country Comparison
1. Navigate to the "Compare" page
2. Select countries to compare
3. View visualizations of cultural dimensions and cultural distances
4. Identify potential areas of cultural conflict or synergy

### Model Training
1. Navigate to the "Train" page
2. Upload custom project data (optional) or use the default dataset
3. Train the model to improve prediction accuracy
4. View model performance metrics

## Theoretical Framework
CIAT is based on established cultural frameworks:
- **Hofstede's Cultural Dimensions**: Power distance, individualism, masculinity, uncertainty avoidance, long-term orientation, and indulgence
- **Fog's (2022) Cross-Cultural Study**: Methodology for measuring cultural impact on projects
- **Dumitrașcu-Băldău, Dumitrașcu and Dobrotă's (2021) Research**: Factors influencing international project success
- **Kogut & Singh (1988)**: Cultural distance formula
- **Kim, Gaur & Mukherjee (2020)**: Cultural distance in cross-border 

## Research Context

The implementation is informed by primary survey data showing that 57.14% of respondents manage projects in Europe and 50% in Africa, with technical requirements (57.14%) being the primary complexity factor. The tool provides insights into how these cultural factors influence project outcomes and offers recommendations to mitigate risks.

### Common Issues
- **ModuleNotFoundError**: Ensure you've installed all dependencies with `pip install -r requirements.txt`
- **Database Errors**: Make sure your database path is accessible and the application has write permissions
- **Visualisation Issues**: Ensure matplotlib and seaborn are properly installed

### Data Requirements
- For custom model training, upload a CSV with project features and a binary success indicator
- For cultural profiles, ensure your data matches Hofstede's dimension format

## Future Enhancements
- API integration for direct access from project management tools
- Additional cultural frameworks beyond Hofstede
- Enhanced visualisation options
- Machine learning model improvements

## License
Copyright <2025> <HAINADINE CHAMANE>

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


## Contact
- Author: Hainadine Chamane
- Email: hchamane@outlook.com
- Repository: https://github.com/hchamane/CIAT

## Acknowledgments
- Hofstede Insights for cultural dimensions data
- Survey participants for providing valuable insights
- Open-source libraries: Flask, scikit-learn, pandas, matplotlib
