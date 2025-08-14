import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, PieChart, Pie, Cell, LineChart, Line, ResponsiveContainer } from 'recharts';
import { TrendingUp } from 'lucide-react';

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884d8', '#82ca9d', '#ffc658', '#ff7c7c'];

const ChartsSection = ({ data, dataCount }) => {
  if (!data || !data.charts) return null;

  const { charts } = data;

  const renderPieChart = (chartData, title, dataKey = 'value') => {
    if (!chartData || !chartData.labels || !chartData.values) return null;

    const pieData = chartData.labels.map((label, index) => ({
      name: label,
      value: chartData.values[index],
      fill: COLORS[index % COLORS.length]
    }));

    return (
      <div className="bg-white rounded-lg shadow-sm p-6">
        <h3 className="text-lg font-semibold text-gray-800 mb-4">{title}</h3>
        <ResponsiveContainer width="100%" height={300}>
          <PieChart>
            <Pie
              data={pieData}
              cx="50%"
              cy="50%"
              labelLine={false}
              label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
              outerRadius={80}
              fill="#8884d8"
              dataKey="value"
            >
              {pieData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.fill} />
              ))}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </div>
    );
  };

  const renderBarChart = (chartData, title, maxItems = 10) => {
    if (!chartData || !chartData.labels || !chartData.values) return null;

    const barData = chartData.labels.slice(0, maxItems).map((label, index) => ({
      name: label.length > 20 ? `${label.substring(0, 20)}...` : label,
      value: chartData.values[index],
      fullName: label
    }));

    return (
      <div className="bg-white rounded-lg shadow-sm p-6">
        <h3 className="text-lg font-semibold text-gray-800 mb-4">{title}</h3>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={barData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis 
              dataKey="name" 
              angle={-45}
              textAnchor="end"
              height={80}
              fontSize={12}
            />
            <YAxis />
            <Tooltip 
              formatter={(value, name, props) => [value, props.payload.fullName]}
            />
            <Bar dataKey="value" fill="#0088FE" />
          </BarChart>
        </ResponsiveContainer>
      </div>
    );
  };

  const renderLineChart = (chartData, title) => {
    if (!chartData || !chartData.labels || !chartData.values) return null;

    const lineData = chartData.labels.map((label, index) => ({
      name: label,
      value: chartData.values[index]
    }));

    return (
      <div className="bg-white rounded-lg shadow-sm p-6">
        <h3 className="text-lg font-semibold text-gray-800 mb-4">{title}</h3>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={lineData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" />
            <YAxis />
            <Tooltip />
            <Line type="monotone" dataKey="value" stroke="#00C49F" strokeWidth={2} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    );
  };

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow-sm p-6">
        <div className="flex items-center space-x-2 mb-4">
          <TrendingUp className="h-5 w-5 text-gray-600" />
          <h2 className="text-lg font-semibold text-gray-800">Data Visualization</h2>
        </div>
        <div className="text-sm text-gray-600 mb-6">
          Showing insights from {dataCount} data points
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Class Distribution */}
        {charts.class_distribution && renderPieChart(
          charts.class_distribution, 
          "Feedback Class Distribution"
        )}

        {/* Location Distribution */}
        {charts.location_distribution && renderBarChart(
          charts.location_distribution, 
          "Top Locations by Feedback Count",
          8
        )}
      </div>

      {/* Monthly Trends - Full Width */}
      {charts.monthly_trends && (
        <div className="w-full">
          {renderLineChart(charts.monthly_trends, "Monthly Feedback Trends")}
        </div>
      )}
    </div>
  );
};

export default ChartsSection;