import React, { useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';

const SurveyInsights = () => {
  // Survey data with 15 respondents
  const [activeTab, setActiveTab] = useState('regions');
  
  // Data from survey_results.csv with 15 respondents
  const surveyData = {
    respondents: 15,
    topRegion: "Europe (60%)",
    topComplexity: "Technical Requirements (60%)",
    topCommunication: "Technical Barriers (35.71%)",
    regionData: [
      { name: 'Europe', value: 60 },
      { name: 'Africa', value: 53.33 },
      { name: 'North America', value: 13.33 },
      { name: 'Asia Pacific', value: 13.33 },
      { name: 'South America', value: 0 },
      { name: 'Middle East', value: 0 }
    ],
    complexityFactors: [
      { name: 'Technical Requirements', value: 60 },
      { name: 'Number of Stakeholders', value: 53.33 },
      { name: 'Regulatory Requirements', value: 46.67 },
      { name: 'Geographic Distribution', value: 26.67 }
    ],
    experienceLevels: [
      { name: '1-5 years', value: 40 },
      { name: '5-10 years', value: 13.33 },
      { name: '10-15 years', value: 26.67 },
      { name: '15+ years', value: 20 }
    ],
    communicationChallenges: [
      { name: 'Technical Barriers', value: 35.71 },
      { name: 'Time Zone Coordination', value: 28.57 },
      { name: 'Documentation Standards', value: 21.43 },
      { name: 'Meeting Formats', value: 7.14 },
      { name: 'Other', value: 7.14 }
    ],
    industrySectors: [
      { name: 'Technology', value: 46.67 },
      { name: 'Other Sectors', value: 33.33 },
      { name: 'Finance', value: 20 },
      { name: 'Manufacturing', value: 6.67 },
      { name: 'Healthcare', value: 0 }
    ]
  };

  // Color palettes
  const COLORS_REGIONS = ['#13547a', '#80d0c7', '#2E86C1', '#48C9B0', '#D4AC0D', '#CB4335'];
  const COLORS_COMPLEXITY = ['#8884d8', '#83a6ed', '#8dd1e1', '#82ca9d'];
  const COLORS_EXPERIENCE = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042'];
  const COLORS_COMMUNICATION = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA5A5', '#98D8C8'];
  const COLORS_SECTORS = ['#845EC2', '#D65DB1', '#FF6F91', '#FF9671', '#FFC75F'];

  // Function to render the active chart
  const renderChart = () => {
    switch(activeTab) {
      case 'regions':
        return (
          <div className="chart-container">
            <h5 className="chart-title">Regional Focus in Survey Responses</h5>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={surveyData.regionData.filter(item => item.value > 0)}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis domain={[0, 100]} />
                <Tooltip formatter={(value) => [`${value.toFixed(1)}%`, 'Percentage']} />
                <Legend />
                <Bar dataKey="value" name="Percentage of Projects" radius={[5, 5, 0, 0]}>
                  {surveyData.regionData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS_REGIONS[index % COLORS_REGIONS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        );
      
      case 'complexity':
        return (
          <div className="chart-container">
            <h5 className="chart-title">Project Complexity Factors</h5>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart layout="vertical" data={surveyData.complexityFactors}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" domain={[0, 100]} />
                <YAxis dataKey="name" type="category" width={150} />
                <Tooltip formatter={(value) => [`${value.toFixed(1)}%`, 'Percentage']} />
                <Legend />
                <Bar dataKey="value" name="Percentage of Responses" radius={[0, 5, 5, 0]}>
                  {surveyData.complexityFactors.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS_COMPLEXITY[index % COLORS_COMPLEXITY.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        );
      
      case 'experience':
        return (
          <div className="chart-container">
            <h5 className="chart-title">Experience Level Distribution</h5>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={surveyData.experienceLevels}
                  cx="50%"
                  cy="50%"
                  labelLine={true}
                  label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(1)}%`}
                  outerRadius={100}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {surveyData.experienceLevels.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS_EXPERIENCE[index % COLORS_EXPERIENCE.length]} />
                  ))}
                </Pie>
                <Tooltip formatter={(value) => [`${value.toFixed(1)}%`, 'Percentage']} />
              </PieChart>
            </ResponsiveContainer>
          </div>
        );
      
      case 'communication':
        return (
          <div className="chart-container">
            <h5 className="chart-title">Communication Challenges</h5>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart layout="vertical" data={surveyData.communicationChallenges}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" domain={[0, 100]} />
                <YAxis dataKey="name" type="category" width={150} />
                <Tooltip formatter={(value) => [`${value.toFixed(1)}%`, 'Percentage']} />
                <Legend />
                <Bar dataKey="value" name="Percentage of Responses" radius={[0, 5, 5, 0]}>
                  {surveyData.communicationChallenges.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS_COMMUNICATION[index % COLORS_COMMUNICATION.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        );
      
      case 'sectors':
        return (
          <div className="chart-container">
            <h5 className="chart-title">Industry Sectors</h5>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={surveyData.industrySectors.filter(item => item.value > 0)}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis domain={[0, 100]} />
                <Tooltip formatter={(value) => [`${value.toFixed(1)}%`, 'Percentage']} />
                <Legend />
                <Bar dataKey="value" name="Percentage of Responses" radius={[5, 5, 0, 0]}>
                  {surveyData.industrySectors.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS_SECTORS[index % COLORS_SECTORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        );
      
      default:
        return null;
    }
  };

  return (
    <div className="survey-insights-container bg-white rounded-4 shadow-sm">
      <div className="card h-100">
        <div className="card-header bg-info text-white">
          <h2 className="h5 mb-0">Survey Insights</h2>
        </div>
        <div className="card-body">
          <div className="row mb-4">
            <div className="col-md-4">
              <div className="stat-card bg-gradient rounded p-3 text-center mb-3 shadow-sm" style={{ background: 'linear-gradient(15deg, #13547a 0%, #80d0c7 100%)' }}>
                <div className="stat-label text-white">Survey Respondents</div>
                <div className="stat-value display-5 text-white fw-bold">{surveyData.respondents}</div>
              </div>
            </div>
            <div className="col-md-4">
              <div className="stat-card bg-gradient rounded p-3 text-center mb-3 shadow-sm" style={{ background: 'linear-gradient(15deg, #FF6B6B 0%, #FFE66D 100%)' }}>
                <div className="stat-label text-white">Top Region</div>
                <div className="stat-value h5 text-white fw-bold">Europe</div>
                <div className="stat-percentage text-white">60%</div>
              </div>
            </div>
            <div className="col-md-4">
              <div className="stat-card bg-gradient rounded p-3 text-center mb-3 shadow-sm" style={{ background: 'linear-gradient(15deg, #845EC2 0%, #D65DB1 100%)' }}>
                <div className="stat-label text-white">Top Complexity Factor</div>
                <div className="stat-value h5 text-white fw-bold">Technical Requirements</div>
                <div className="stat-percentage text-white">60%</div>
              </div>
            </div>
          </div>
          
          <div className="nav-tabs-container mb-3">
            <ul className="nav nav-tabs">
              <li className="nav-item">
                <button 
                  className={`nav-link ${activeTab === 'regions' ? 'active' : ''}`} 
                  onClick={() => setActiveTab('regions')}
                >
                  Regional Focus
                </button>
              </li>
              <li className="nav-item">
                <button 
                  className={`nav-link ${activeTab === 'complexity' ? 'active' : ''}`} 
                  onClick={() => setActiveTab('complexity')}
                >
                  Complexity Factors
                </button>
              </li>
              <li className="nav-item">
                <button 
                  className={`nav-link ${activeTab === 'experience' ? 'active' : ''}`} 
                  onClick={() => setActiveTab('experience')}
                >
                  Experience Levels
                </button>
              </li>
              <li className="nav-item">
                <button 
                  className={`nav-link ${activeTab === 'communication' ? 'active' : ''}`} 
                  onClick={() => setActiveTab('communication')}
                >
                  Communication
                </button>
              </li>
              <li className="nav-item">
                <button 
                  className={`nav-link ${activeTab === 'sectors' ? 'active' : ''}`} 
                  onClick={() => setActiveTab('sectors')}
                >
                  Sectors
                </button>
              </li>
            </ul>
          </div>
          
          {renderChart()}
          
          <div className="text-center mt-3">
            <a href="/survey_insights" className="btn btn-outline-primary">View All Survey Insights</a>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SurveyInsights;