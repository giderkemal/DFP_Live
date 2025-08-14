import React from 'react';
import { BarChart3 } from 'lucide-react';

const Header = () => {
  return (
    <header className="bg-white shadow-sm border-b">
      <div className="container mx-auto px-4 py-4">
        <div className="flex items-center space-x-3">
          <div className="flex items-center space-x-2">
            <BarChart3 className="h-8 w-8 text-blue-600" />
            <div>
              <h1 className="text-xl font-bold text-gray-800">
                Field Intelligence
              </h1>
              <p className="text-sm text-gray-500">
                Report Generation Tool
              </p>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;