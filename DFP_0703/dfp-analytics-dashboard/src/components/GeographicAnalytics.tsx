import React from 'react'
import type { FilterState } from './Dashboard'

interface GeographicAnalyticsProps {
  filters: FilterState
}

const GeographicAnalytics: React.FC<GeographicAnalyticsProps> = ({ filters }) => {
  return (
    <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
      <h3 className="text-lg font-medium text-gray-900 mb-4">Geographic Analytics</h3>
      <p className="text-gray-600">Advanced geographic visualizations coming soon...</p>
    </div>
  )
}

export default GeographicAnalytics 