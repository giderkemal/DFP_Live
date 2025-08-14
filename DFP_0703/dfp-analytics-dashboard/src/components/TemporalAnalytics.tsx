import React from 'react'
import type { FilterState } from './Dashboard'

interface TemporalAnalyticsProps {
  filters: FilterState
}

const TemporalAnalytics: React.FC<TemporalAnalyticsProps> = ({ filters }) => {
  return (
    <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
      <h3 className="text-lg font-medium text-gray-900 mb-4">Temporal Analytics</h3>
      <p className="text-gray-600">Time-based analytics and trends coming soon...</p>
    </div>
  )
}

export default TemporalAnalytics 