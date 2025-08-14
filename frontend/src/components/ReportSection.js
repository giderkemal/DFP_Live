import React, { useState } from 'react';
import { FileText, Eye, EyeOff } from 'lucide-react';

const ReportSection = ({ report }) => {
  const [showRawResponse, setShowRawResponse] = useState(false);

  if (!report) return null;

  const formatReport = (reportText) => {
    if (!reportText) return '';
    
    // Convert line breaks to HTML
    return reportText
      .replace(/\n\n/g, '</p><p>')
      .replace(/\n/g, '<br>')
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      .replace(/\*(.*?)\*/g, '<em>$1</em>')
      .replace(/(\d+\.\s)/g, '<br><strong>$1</strong>')
      .replace(/\[Row_ID:(\d+)\]/g, '<span class="bg-blue-100 text-blue-800 px-2 py-1 rounded text-xs font-mono">[Row_ID:$1]</span>');
  };

  return (
    <div className="bg-white rounded-lg shadow-sm p-6">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-2">
          <FileText className="h-5 w-5 text-gray-600" />
          <h2 className="text-lg font-semibold text-gray-800">Generated Report</h2>
        </div>
        <div className="flex items-center space-x-2">
          <span className="text-sm text-gray-600">Data points: {report.data_count}</span>
          {report.examples_id && (
            <span className="text-sm text-blue-600">
              {report.examples_id.length} examples cited
            </span>
          )}
        </div>
      </div>

      {/* Report Content */}
      <div className="bg-gray-50 rounded-lg p-6 mb-6">
        <div 
          className="report-content prose max-w-none text-gray-800 leading-relaxed"
          dangerouslySetInnerHTML={{ 
            __html: `<p>${formatReport(report.report)}</p>`
          }}
        />
      </div>

      {/* Examples ID Section */}
      {report.examples_id && report.examples_id.length > 0 && (
        <div className="mb-6">
          <h3 className="text-md font-semibold text-gray-800 mb-3">
            Cited Examples ({report.examples_id.length})
          </h3>
          <div className="flex flex-wrap gap-2">
            {report.examples_id.map((id, index) => (
              <span 
                key={index}
                className="bg-blue-100 text-blue-800 px-3 py-1 rounded-full text-xs font-mono"
              >
                Row_ID: {id}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Raw Response Toggle */}
      <div className="border-t pt-4">
        <button
          onClick={() => setShowRawResponse(!showRawResponse)}
          className="flex items-center space-x-2 text-sm text-gray-600 hover:text-gray-800 transition-colors"
        >
          {showRawResponse ? (
            <>
              <EyeOff className="h-4 w-4" />
              <span>Hide Raw Response</span>
            </>
          ) : (
            <>
              <Eye className="h-4 w-4" />
              <span>Show Raw Response</span>
            </>
          )}
        </button>
        
        {showRawResponse && (
          <div className="mt-4 bg-gray-900 text-green-400 p-4 rounded-lg overflow-x-auto">
            <pre className="text-xs whitespace-pre-wrap font-mono">
              {report.raw_response}
            </pre>
          </div>
        )}
      </div>
    </div>
  );
};

export default ReportSection;