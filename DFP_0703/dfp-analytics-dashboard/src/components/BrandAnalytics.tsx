import React from 'react'
import type { FilterState } from './Dashboard'

const BrandAnalytics: React.FC<{ filters: FilterState }> = ({ filters }) => {
  return (
    <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
      <h3 className="text-lg font-medium text-gray-900 mb-4">Brand Analytics</h3>
      <p className="text-gray-600">Brand and TMO analytics coming soon...</p>
    </div>
  )
}

export default BrandAnalytics 