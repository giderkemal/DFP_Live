import React from 'react'
import { useQuery } from '@tanstack/react-query'
import { fetchDataSummary } from '../services/api'
import type { FilterState } from './Dashboard'

interface OverviewMetricsProps {
  filters: FilterState
}

const OverviewMetrics: React.FC<OverviewMetricsProps> = ({ filters }) => {
  const { data: summary, isLoading, error } = useQuery({
    queryKey: ['dataSummary'],
    queryFn: fetchDataSummary,
  })

  if (isLoading) {
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {[...Array(4)].map((_, i) => (
          <div key={i} className="bg-white p-6 rounded-lg shadow animate-pulse">
            <div className="h-4 bg-gray-200 rounded w-3/4 mb-2"></div>
            <div className="h-8 bg-gray-200 rounded w-1/2"></div>
          </div>
        ))}
      </div>
    )
  }

  if (error) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-4">
        <p className="text-red-600">Error loading data summary</p>
      </div>
    )
  }

  if (!summary) return null

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
          <div className="text-sm font-medium text-gray-500">Total Records</div>
          <div className="text-2xl font-bold text-gray-900">
            {summary.total_records.toLocaleString()}
          </div>
        </div>
        
        <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
          <div className="text-sm font-medium text-gray-500">Regions</div>
          <div className="text-2xl font-bold text-gray-900">
            {summary.unique_counts.regions}
          </div>
        </div>
        
        <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
          <div className="text-sm font-medium text-gray-500">Markets</div>
          <div className="text-2xl font-bold text-gray-900">
            {summary.unique_counts.markets}
          </div>
        </div>
        
        <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
          <div className="text-sm font-medium text-gray-500">Users</div>
          <div className="text-2xl font-bold text-gray-900">
            {summary.unique_counts.users}
          </div>
        </div>
      </div>

      <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
        <h3 className="text-lg font-medium text-gray-900 mb-4">Data Overview</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h4 className="font-medium text-gray-700 mb-2">Date Range</h4>
            <p className="text-gray-600">
              {summary.date_range.start && summary.date_range.end
                ? `${new Date(summary.date_range.start).toLocaleDateString()} - ${new Date(summary.date_range.end).toLocaleDateString()}`
                : 'No date range available'}
            </p>
          </div>
          <div>
            <h4 className="font-medium text-gray-700 mb-2">Data Quality</h4>
            <div className="space-y-1 text-sm text-gray-600">
              <div>Form Types: {summary.unique_counts.form_types}</div>
              <div>Locations: {summary.unique_counts.locations}</div>
              <div>Total Columns: {summary.columns.length}</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default OverviewMetrics 