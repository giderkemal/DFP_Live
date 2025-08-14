import React from 'react'
import { BarChart3 } from 'lucide-react'
import type { DataSummary } from '../services/api'

interface HeaderProps {
  summary?: DataSummary
}

const Header: React.FC<HeaderProps> = ({ summary }) => {
  return (
    <header className="bg-white shadow-sm border-b border-gray-200">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          <div className="flex items-center">
            <BarChart3 className="h-8 w-8 text-blue-600" />
            <h1 className="ml-3 text-2xl font-bold text-gray-900">
              DFP Analytics Dashboard
            </h1>
          </div>
          
          {summary && (
            <div className="flex items-center space-x-6 text-sm text-gray-600">
              <div className="text-right">
                <div className="font-medium text-gray-900">
                  {summary.total_records.toLocaleString()} Records
                </div>
                <div className="text-xs">
                  {summary.unique_counts.regions} Regions • {summary.unique_counts.markets} Markets
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </header>
  )
}

export default Header 