import React, { useState, useEffect } from 'react';
import axios from 'axios';
import Header from './components/Header';
import FilterSection from './components/FilterSection';
import ChartsSection from './components/ChartsSection';
import ReportSection from './components/ReportSection';
import ChatSection from './components/ChatSection';
import DataTableSection from './components/DataTableSection';
import LoadingSpinner from './components/LoadingSpinner';
import './App.css';

const API_BASE_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8001';

function App() {
  const [metadata, setMetadata] = useState(null);
  const [filteredData, setFilteredData] = useState(null);
  const [chartsData, setChartsData] = useState(null);
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [currentFilters, setCurrentFilters] = useState(null);

  // Initialize metadata on component mount
  useEffect(() => {
    loadMetadata();
  }, []);

  const loadMetadata = async () => {
    try {
      setLoading(true);
      const response = await axios.get(`${API_BASE_URL}/api/metadata`);
      setMetadata(response.data);
      setError(null);
    } catch (err) {
      console.error('Error loading metadata:', err);
      setError('Failed to load metadata. Please check if the server is running.');
    } finally {
      setLoading(false);
    }
  };

  const handleFiltersApplied = async (filters) => {
    try {
      setLoading(true);
      setError(null);
      setCurrentFilters(filters);
      setReport(null); // Clear previous report

      // Filter data
      const filterResponse = await axios.post(`${API_BASE_URL}/api/filter-data`, filters);
      setFilteredData(filterResponse.data);

      // Get charts data
      const chartsResponse = await axios.get(`${API_BASE_URL}/api/charts-data`, {
        params: filters
      });
      setChartsData(chartsResponse.data);

    } catch (err) {
      console.error('Error applying filters:', err);
      setError('Failed to apply filters. Please try again.');
      setFilteredData(null);
      setChartsData(null);
    } finally {
      setLoading(false);
    }
  };

  const handleGenerateReport = async () => {
    if (!currentFilters) {
      setError('Please apply filters first');
      return;
    }

    try {
      setLoading(true);
      setError(null);

      const response = await axios.post(`${API_BASE_URL}/api/generate-report`, currentFilters);
      setReport(response.data);

    } catch (err) {
      console.error('Error generating report:', err);
      setError('Failed to generate report. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <Header />
      
      <main className="container mx-auto px-4 py-6 space-y-8">
        {/* Introduction */}
        <div className="bg-white rounded-lg shadow-sm p-6">
          <h1 className="text-2xl font-bold text-gray-800 mb-4">
            Field Intelligence Report Generation Tool 📊
          </h1>
          <p className="text-gray-600 leading-relaxed">
            Welcome to the Field Intelligence Report Generation Tool, your go-to solution for 
            categorizing and analyzing field data with ease. This tool leverages cutting-edge AI 
            to process and classify insights from diverse inputs, helping you turn raw information 
            into actionable reports. Whether you're streamlining workflows, enhancing decision-making, 
            or exploring patterns in field intelligence, this tool is designed to simplify and 
            elevate your data analysis experience.
          </p>
        </div>

        {/* Error Display */}
        {error && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-4">
            <div className="flex">
              <div className="text-red-800">{error}</div>
            </div>
          </div>
        )}

        {/* Loading Spinner */}
        {loading && <LoadingSpinner />}

        {/* Filter Section */}
        {metadata && (
          <FilterSection 
            metadata={metadata} 
            onFiltersApplied={handleFiltersApplied}
            loading={loading}
          />
        )}

        {/* Charts Section */}
        {chartsData && filteredData && (
          <ChartsSection 
            data={chartsData} 
            dataCount={filteredData.count}
          />
        )}

        {/* Generate Report Button */}
        {filteredData && filteredData.count > 0 && (
          <div className="bg-white rounded-lg shadow-sm p-6">
            <div className="flex justify-between items-center">
              <div>
                <h3 className="text-lg font-semibold text-gray-800">
                  Generate Report
                </h3>
                <p className="text-sm text-gray-600">
                  {filteredData.count > 1200 
                    ? `Dataset exceeds limit (${filteredData.count} rows). Combined generation will be used.`
                    : `Ready to generate report with ${filteredData.count} data points.`
                  }
                </p>
              </div>
              <button
                onClick={handleGenerateReport}
                disabled={loading}
                className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                {loading ? 'Generating...' : 'Generate Report'}
              </button>
            </div>
          </div>
        )}

        {/* Report Section */}
        {report && (
          <ReportSection report={report} />
        )}

        {/* Chat Section */}
        {report && (
          <ChatSection 
            report={report}
            data={filteredData}
          />
        )}

        {/* Data Table Section */}
        {filteredData && (
          <DataTableSection data={filteredData} />
        )}
      </main>
    </div>
  );
}

export default App;