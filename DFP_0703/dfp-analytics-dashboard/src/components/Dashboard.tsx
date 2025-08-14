import React, { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { BarChart3, Globe, Calendar, Users, Package, Filter } from 'lucide-react'
import Header from './Header'
import FilterPanel from './FilterPanel'
import OverviewMetrics from './OverviewMetrics'
import GeographicAnalytics from './GeographicAnalytics'
import TemporalAnalytics from './TemporalAnalytics'
import BrandAnalytics from './BrandAnalytics'
import FormAnalytics from './FormAnalytics'
import UserAnalytics from './UserAnalytics'
import DataTable from './DataTable'
import { fetchDataSummary } from '../services/api'

export interface FilterState {
  region?: string
  market?: string
  form_type?: string
  tmo?: string
  start_date?: string
  end_date?: string
  search?: string
}

const Dashboard: React.FC = () => {
  const [activeTab, setActiveTab] = useState('overview')
  const [filters, setFilters] = useState<FilterState>({})
  const [showFilters, setShowFilters] = useState(false)

  const { data: summary, isLoading } = useQuery({
    queryKey: ['dataSummary'],
    queryFn: fetchDataSummary,
  })

  const tabs = [
    { id: 'overview', label: 'Overview', icon: BarChart3 },
    { id: 'geographic', label: 'Geographic', icon: Globe },
    { id: 'temporal', label: 'Temporal', icon: Calendar },
    { id: 'brands', label: 'Brands & TMO', icon: Package },
    { id: 'forms', label: 'Form Types', icon: BarChart3 },
    { id: 'users', label: 'Users', icon: Users },
    { id: 'data', label: 'Data Table', icon: Filter },
  ]

  const renderTabContent = () => {
    switch (activeTab) {
      case 'overview':
        return <OverviewMetrics filters={filters} />
      case 'geographic':
        return <GeographicAnalytics filters={filters} />
      case 'temporal':
        return <TemporalAnalytics filters={filters} />
      case 'brands':
        return <BrandAnalytics filters={filters} />
      case 'forms':
        return <FormAnalytics filters={filters} />
      case 'users':
        return <UserAnalytics filters={filters} />
      case 'data':
        return <DataTable filters={filters} />
      default:
        return <OverviewMetrics filters={filters} />
    }
  }

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading DFP Analytics Dashboard...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <Header summary={summary} />
      
      <div className="flex">
        {/* Sidebar with Filters */}
        <div className={`${showFilters ? 'w-80' : 'w-0'} transition-all duration-300 overflow-hidden bg-white shadow-lg`}>
          <FilterPanel filters={filters} onFiltersChange={setFilters} />
        </div>

        {/* Main Content */}
        <div className="flex-1">
          {/* Filter Toggle */}
          <div className="bg-white border-b border-gray-200 px-6 py-4">
            <button
              onClick={() => setShowFilters(!showFilters)}
              className="flex items-center space-x-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
            >
              <Filter className="h-4 w-4" />
              <span>{showFilters ? 'Hide Filters' : 'Show Filters'}</span>
            </button>
          </div>

          {/* Tab Navigation */}
          <div className="bg-white border-b border-gray-200">
            <nav className="flex space-x-8 px-6">
              {tabs.map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center space-x-2 py-4 px-1 border-b-2 font-medium text-sm transition-colors ${
                    activeTab === tab.id
                      ? 'border-blue-500 text-blue-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  }`}
                >
                  <tab.icon className="h-4 w-4" />
                  <span>{tab.label}</span>
                </button>
              ))}
            </nav>
          </div>

          {/* Tab Content */}
          <div className="p-6">
            {renderTabContent()}
          </div>
        </div>
      </div>
    </div>
  )
}

export default Dashboard 