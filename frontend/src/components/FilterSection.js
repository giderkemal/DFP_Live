import React, { useState, useEffect } from 'react';
import { Calendar, Filter } from 'lucide-react';
import DatePicker from 'react-datepicker';
import "react-datepicker/dist/react-datepicker.css";

const FilterSection = ({ metadata, onFiltersApplied, loading }) => {
  const [filters, setFilters] = useState({
    date_range: ['', ''],
    feedback_class: [],
    form_type: 'All',
    region: [],
    market: [],
    location: [],
    tmo: [],
    brand: [],
    product_category: [],
    pmi_product: [],
    switch_from: [],
    switch_to: []
  });

  const [startDate, setStartDate] = useState(null);
  const [endDate, setEndDate] = useState(null);

  // Initialize date range from metadata
  useEffect(() => {
    if (metadata?.DATE) {
      const start = new Date(metadata.DATE.min);
      const end = new Date(metadata.DATE.max);
      setStartDate(start);
      setEndDate(end);
      setFilters(prev => ({
        ...prev,
        date_range: [metadata.DATE.min, metadata.DATE.max]
      }));
    }
  }, [metadata]);

  const handleDateChange = (dates) => {
    const [start, end] = dates;
    setStartDate(start);
    setEndDate(end);
    
    if (start && end) {
      setFilters(prev => ({
        ...prev,
        date_range: [
          start.toISOString().split('T')[0],
          end.toISOString().split('T')[0]
        ]
      }));
    }
  };

  const handleMultiSelectChange = (field, value) => {
    setFilters(prev => ({
      ...prev,
      [field]: prev[field].includes(value)
        ? prev[field].filter(item => item !== value)
        : [...prev[field], value]
    }));
  };

  const handleSingleSelectChange = (field, value) => {
    setFilters(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const getFilteredMetadata = () => {
    // Return filtered metadata based on form type selection
    // This would need to implement the same logic as the original Streamlit app
    return metadata;
  };

  const getFormTypeFilters = (formType) => {
    const formTypeFilters = {
      'All': ['region', 'market', 'location', 'tmo', 'brand'],
      'CROSS_CATEGORY': ['region', 'market', 'location', 'product_category'],
      'CONSUMER_FEEDBACK': ['region', 'market', 'location', 'product_category', 'pmi_product'],
      'TOBACCO_CATEGORY': ['region', 'market', 'location', 'tmo', 'brand', 'product_category'],
      'BRAND_SOURCING': ['region', 'market', 'location', 'switch_from', 'switch_to'],
      'INFRA_MAINTENANCE': ['region', 'market', 'location']
    };
    
    return formTypeFilters[formType] || formTypeFilters['All'];
  };

  const renderMultiSelect = (field, label, options) => {
    if (!options || options.length === 0) return null;
    
    return (
      <div className="space-y-2">
        <label className="block text-sm font-medium text-gray-700">{label}</label>
        <div className="max-h-32 overflow-y-auto border border-gray-300 rounded-md p-2">
          {options.map((option) => (
            <label key={option} className="flex items-center space-x-2 text-sm">
              <input
                type="checkbox"
                checked={filters[field].includes(option)}
                onChange={() => handleMultiSelectChange(field, option)}
                className="rounded border-gray-300 text-blue-600 shadow-sm focus:border-blue-300 focus:ring focus:ring-blue-200 focus:ring-opacity-50"
              />
              <span>{option}</span>
            </label>
          ))}
        </div>
        {filters[field].length > 0 && (
          <div className="text-xs text-gray-500">
            Selected: {filters[field].length} items
          </div>
        )}
      </div>
    );
  };

  const getFieldMapping = () => ({
    region: { key: 'VP_REGION_NAME', label: 'Region' },
    market: { key: 'DF_MARKET_NAME', label: 'Market' },
    location: { key: 'LOCATION_NAME', label: 'Location' },
    tmo: { key: 'TMO_NAME', label: 'TMO' },
    brand: { key: 'BRAND_NAME', label: 'Brand' },
    product_category: { key: 'PRODUCT_CATEGORY_NAME', label: 'Product Category' },
    pmi_product: { key: 'PMI_PRODUCT_NAME', label: 'PMI Product' },
    switch_from: { key: 'BRAND_NAME_FROM', label: 'Switch From' },
    switch_to: { key: 'BRAND_NAME_TO', label: 'Switch To' }
  });

  const handleApplyFilters = () => {
    onFiltersApplied(filters);
  };

  if (!metadata) return null;

  const availableFilters = getFormTypeFilters(filters.form_type);
  const fieldMapping = getFieldMapping();
  const filteredMetadata = getFilteredMetadata();

  return (
    <div className="bg-white rounded-lg shadow-sm p-6">
      <div className="flex items-center space-x-2 mb-6">
        <Filter className="h-5 w-5 text-gray-600" />
        <h2 className="text-lg font-semibold text-gray-800">Filters</h2>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
        {/* Date Range */}
        <div className="space-y-2">
          <label className="block text-sm font-medium text-gray-700 flex items-center space-x-2">
            <Calendar className="h-4 w-4" />
            <span>Date Range</span>
          </label>
          <DatePicker
            selectsRange={true}
            startDate={startDate}
            endDate={endDate}
            onChange={handleDateChange}
            className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
            dateFormat="yyyy-MM-dd"
            placeholderText="Select date range"
          />
        </div>

        {/* Feedback Class */}
        <div className="space-y-2">
          <label className="block text-sm font-medium text-gray-700">Feedback Class</label>
          <div className="max-h-32 overflow-y-auto border border-gray-300 rounded-md p-2">
            {metadata.CLASS?.map((option) => (
              <label key={option} className="flex items-center space-x-2 text-sm">
                <input
                  type="checkbox"
                  checked={filters.feedback_class.includes(option)}
                  onChange={() => handleMultiSelectChange('feedback_class', option)}
                  className="rounded border-gray-300 text-blue-600 shadow-sm focus:border-blue-300 focus:ring focus:ring-blue-200 focus:ring-opacity-50"
                />
                <span>{option}</span>
              </label>
            ))}
          </div>
        </div>

        {/* Form Type */}
        <div className="space-y-2">
          <label className="block text-sm font-medium text-gray-700">Form Type</label>
          <select
            value={filters.form_type}
            onChange={(e) => handleSingleSelectChange('form_type', e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
          >
            <option value="All">All</option>
            {metadata.FORM_TYPE?.map((option) => (
              <option key={option} value={option}>{option}</option>
            ))}
          </select>
        </div>
      </div>

      {/* Dynamic Filters based on Form Type */}
      <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-4 gap-6 mb-6">
        {availableFilters.map((filterKey) => {
          const mapping = fieldMapping[filterKey];
          if (!mapping || !filteredMetadata[mapping.key]) return null;

          return (
            <div key={filterKey}>
              {renderMultiSelect(filterKey, mapping.label, filteredMetadata[mapping.key])}
            </div>
          );
        })}
      </div>

      {/* Apply Button */}
      <div className="flex justify-center">
        <button
          onClick={handleApplyFilters}
          disabled={loading}
          className="px-8 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors font-medium"
        >
          {loading ? 'Applying...' : 'Apply Filters'}
        </button>
      </div>
    </div>
  );
};

export default FilterSection;