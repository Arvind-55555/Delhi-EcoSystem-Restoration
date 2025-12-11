import React from 'react';
import { Activity, Wind, Droplets, Thermometer, AlertTriangle, CheckCircle2, TrendingUp, TrendingDown } from 'lucide-react';
import { getAQIColor, getAQICategory, formatNumber } from '../utils/helpers';

export const MetricCard = ({ title, value, unit, icon: Icon, trend, color = 'blue' }) => {
  const colorClasses = {
    blue: 'text-blue-600 bg-blue-100',
    green: 'text-green-600 bg-green-100',
    yellow: 'text-yellow-600 bg-yellow-100',
    red: 'text-red-600 bg-red-100',
    purple: 'text-purple-600 bg-purple-100',
  };

  return (
    <div className="card">
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <p className="text-sm font-medium text-gray-600 mb-1">{title}</p>
          <div className="flex items-baseline gap-2">
            <span className="text-3xl font-bold text-gray-900">{formatNumber(value)}</span>
            {unit && <span className="text-sm text-gray-500">{unit}</span>}
          </div>
          {trend && (
            <div className={`flex items-center gap-1 mt-2 text-sm ${trend > 0 ? 'text-red-600' : 'text-green-600'}`}>
              {trend > 0 ? <TrendingUp size={16} /> : <TrendingDown size={16} />}
              <span>{Math.abs(trend)}% from yesterday</span>
            </div>
          )}
        </div>
        {Icon && (
          <div className={`p-3 rounded-lg ${colorClasses[color]}`}>
            <Icon size={24} />
          </div>
        )}
      </div>
    </div>
  );
};

export const AQICard = ({ aqi, pm25 }) => {
  const category = getAQICategory(aqi);
  const colorClass = getAQIColor(aqi);

  return (
    <div className="card">
      <div className="flex items-start justify-between mb-4">
        <div>
          <h3 className="text-lg font-semibold text-gray-900">Air Quality Index</h3>
          <p className="text-sm text-gray-500 mt-1">Real-time AQI monitoring</p>
        </div>
        <Activity className="text-gray-400" size={24} />
      </div>
      
      <div className="flex items-end gap-4 mb-4">
        <div className="text-5xl font-bold text-gray-900">{Math.round(aqi)}</div>
        <div className={`px-3 py-1 rounded-full font-medium text-sm ${colorClass}`}>
          {category}
        </div>
      </div>

      <div className="w-full h-2 bg-gray-200 rounded-full overflow-hidden mb-4">
        <div 
          className="h-full bg-gradient-to-r from-green-500 via-yellow-500 via-orange-500 via-red-500 to-purple-600 transition-all" 
          style={{ width: `${Math.min((aqi / 500) * 100, 100)}%` }}
        />
      </div>

      <div className="grid grid-cols-2 gap-4 pt-4 border-t border-gray-200">
        <div>
          <p className="text-sm text-gray-500">PM2.5</p>
          <p className="text-xl font-semibold text-gray-900">{formatNumber(pm25)} µg/m³</p>
        </div>
        <div>
          <p className="text-sm text-gray-500">Status</p>
          <p className="text-xl font-semibold text-gray-900">{category}</p>
        </div>
      </div>
    </div>
  );
};

export const HealthScoreCard = ({ score, breakdown }) => {
  const getScoreColor = (score) => {
    if (score >= 75) return 'text-green-600';
    if (score >= 60) return 'text-blue-600';
    if (score >= 45) return 'text-yellow-600';
    if (score >= 30) return 'text-orange-600';
    return 'text-red-600';
  };

  const getScoreLabel = (score) => {
    if (score >= 75) return 'Excellent';
    if (score >= 60) return 'Good';
    if (score >= 45) return 'Moderate';
    if (score >= 30) return 'Poor';
    return 'Critical';
  };

  return (
    <div className="card">
      <h3 className="text-lg font-semibold text-gray-900 mb-4">Ecosystem Health Score</h3>
      
      <div className="flex items-center justify-center mb-6">
        <div className="relative w-40 h-40">
          <svg className="transform -rotate-90 w-40 h-40">
            <circle
              cx="80"
              cy="80"
              r="70"
              stroke="#e5e7eb"
              strokeWidth="12"
              fill="none"
            />
            <circle
              cx="80"
              cy="80"
              r="70"
              stroke={score >= 60 ? '#22c55e' : score >= 45 ? '#f59e0b' : '#ef4444'}
              strokeWidth="12"
              fill="none"
              strokeDasharray={`${(score / 100) * 440} 440`}
              strokeLinecap="round"
            />
          </svg>
          <div className="absolute inset-0 flex flex-col items-center justify-center">
            <span className={`text-4xl font-bold ${getScoreColor(score)}`}>
              {Math.round(score)}
            </span>
            <span className="text-sm text-gray-500">/ 100</span>
          </div>
        </div>
      </div>

      <div className="text-center mb-4">
        <span className={`px-4 py-2 rounded-full font-medium ${getScoreColor(score)} bg-opacity-10`}>
          {getScoreLabel(score)}
        </span>
      </div>

      {breakdown && (
        <div className="space-y-3 pt-4 border-t border-gray-200">
          {Object.entries(breakdown).map(([key, value]) => (
            <div key={key} className="flex items-center justify-between">
              <span className="text-sm text-gray-600 capitalize">{key.replace(/_/g, ' ')}</span>
              <span className="text-sm font-semibold text-gray-900">{formatNumber(value)}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export const AlertCard = ({ type = 'warning', title, message, action }) => {
  const styles = {
    warning: {
      bg: 'bg-yellow-50',
      border: 'border-yellow-200',
      icon: 'text-yellow-600',
      Icon: AlertTriangle,
    },
    success: {
      bg: 'bg-green-50',
      border: 'border-green-200',
      icon: 'text-green-600',
      Icon: CheckCircle2,
    },
    danger: {
      bg: 'bg-red-50',
      border: 'border-red-200',
      icon: 'text-red-600',
      Icon: AlertTriangle,
    },
  };

  const style = styles[type];
  const Icon = style.Icon;

  return (
    <div className={`${style.bg} ${style.border} border rounded-lg p-4`}>
      <div className="flex gap-3">
        <Icon className={style.icon} size={20} />
        <div className="flex-1">
          <h4 className="font-semibold text-gray-900 mb-1">{title}</h4>
          <p className="text-sm text-gray-700">{message}</p>
          {action && (
            <button className="mt-2 text-sm font-medium text-primary-600 hover:text-primary-700">
              {action}
            </button>
          )}
        </div>
      </div>
    </div>
  );
};
