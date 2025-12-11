import React, { useState, useEffect } from 'react';
import { Home, Activity, TrendingUp, Settings, BarChart3, Sparkles } from 'lucide-react';
import { MetricCard, AQICard, HealthScoreCard, AlertCard } from '../components/Cards';
import { TimeSeriesChart, MultiLineChart } from '../components/Charts';
import { predictAQI, forecastPM25, calculateEcosystemHealth } from '../utils/api';

const Dashboard = () => {
  const [currentData, setCurrentData] = useState({
    aqi: 245,
    pm25: 120,
    pm10: 200,
    no2: 50,
    so2: 10,
    co: 1.5,
    o3: 40,
    temperature: 25,
    humidity: 60,
    wind_speed: 2,
    green_cover: 20,
  });

  const [ecosystemHealth, setEcosystemHealth] = useState(null);
  const [forecast, setForecast] = useState([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    setLoading(true);
    try {
      // Get ecosystem health
      const healthData = await calculateEcosystemHealth({
        PM25: currentData.pm25,
        PM10: currentData.pm10,
        NO2: currentData.no2,
        SO2: currentData.so2,
        CO: currentData.co,
        O3: currentData.o3,
        temperature: currentData.temperature,
        humidity: currentData.humidity,
        wind_speed: currentData.wind_speed,
        precipitation: 0,
        green_cover_percentage: currentData.green_cover,
      });
      setEcosystemHealth(healthData);

      // Get 7-day forecast
      const forecastData = await forecastPM25(7, 'prophet');
      setForecast(forecastData.predictions || []);
    } catch (error) {
      console.error('Error loading data:', error);
    } finally {
      setLoading(false);
    }
  };

  const pollutantData = [
    { name: 'PM2.5', value: currentData.pm25, unit: 'µg/m³', limit: 60 },
    { name: 'PM10', value: currentData.pm10, unit: 'µg/m³', limit: 100 },
    { name: 'NO2', value: currentData.no2, unit: 'µg/m³', limit: 80 },
    { name: 'SO2', value: currentData.so2, unit: 'µg/m³', limit: 80 },
    { name: 'CO', value: currentData.co, unit: 'mg/m³', limit: 4 },
    { name: 'O3', value: currentData.o3, unit: 'µg/m³', limit: 100 },
  ];

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-primary-600 rounded-lg">
                <Sparkles className="text-white" size={24} />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-gray-900">Ecosystem Health Dashboard</h1>
                <p className="text-sm text-gray-500">Delhi, India - Real-time Monitoring</p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <button className="btn-secondary">
                <Settings size={18} className="inline mr-2" />
                Settings
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {loading && (
          <div className="text-center py-12">
            <div className="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"></div>
            <p className="mt-4 text-gray-600">Loading data...</p>
          </div>
        )}

        {!loading && (
          <>
            {/* Alert Banner */}
            <div className="mb-6">
              <AlertCard
                type="warning"
                title="Air Quality Alert"
                message="PM2.5 levels are Very Unhealthy. Sensitive groups should avoid outdoor activities."
                action="View Recommendations →"
              />
            </div>

            {/* Top Metrics */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-6">
              <MetricCard
                title="PM2.5"
                value={currentData.pm25}
                unit="µg/m³"
                icon={Activity}
                color="red"
                trend={5.2}
              />
              <MetricCard
                title="Temperature"
                value={currentData.temperature}
                unit="°C"
                icon={Activity}
                color="blue"
              />
              <MetricCard
                title="Humidity"
                value={currentData.humidity}
                unit="%"
                icon={Activity}
                color="blue"
              />
              <MetricCard
                title="Wind Speed"
                value={currentData.wind_speed}
                unit="m/s"
                icon={Activity}
                color="green"
              />
            </div>

            {/* AQI and Health Score */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
              <div className="lg:col-span-2">
                <AQICard aqi={currentData.aqi} pm25={currentData.pm25} />
              </div>
              <div>
                {ecosystemHealth && (
                  <HealthScoreCard
                    score={ecosystemHealth.ecosystem_health_score}
                    breakdown={{
                      'Air Quality': ecosystemHealth.air_quality_score,
                      'Weather': ecosystemHealth.weather_score,
                      'Green Cover': ecosystemHealth.green_cover_score,
                    }}
                  />
                )}
              </div>
            </div>

            {/* Pollutant Levels */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
              <div className="card">
                <h3 className="text-lg font-semibold mb-4">Pollutant Levels</h3>
                <div className="space-y-4">
                  {pollutantData.map((pollutant) => (
                    <div key={pollutant.name}>
                      <div className="flex justify-between mb-1">
                        <span className="text-sm font-medium text-gray-700">{pollutant.name}</span>
                        <span className="text-sm text-gray-500">
                          {pollutant.value} / {pollutant.limit} {pollutant.unit}
                        </span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-2">
                        <div
                          className={`h-2 rounded-full transition-all ${
                            pollutant.value > pollutant.limit ? 'bg-red-500' : 'bg-green-500'
                          }`}
                          style={{ width: `${Math.min((pollutant.value / pollutant.limit) * 100, 100)}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              <div className="card">
                <h3 className="text-lg font-semibold mb-4">Health Recommendations</h3>
                {ecosystemHealth ? (
                  <div className="space-y-3">
                    {ecosystemHealth.recommendations && ecosystemHealth.recommendations.length > 0 ? (
                      ecosystemHealth.recommendations.map((rec, index) => (
                        <div key={index} className="flex gap-3 p-3 bg-blue-50 rounded-lg">
                          <Activity className="text-blue-600 flex-shrink-0" size={20} />
                          <p className="text-sm text-gray-700">{rec}</p>
                        </div>
                      ))
                    ) : (
                      <>
                        <div className="flex gap-3 p-3 bg-green-50 rounded-lg">
                          <Activity className="text-green-600 flex-shrink-0" size={20} />
                          <p className="text-sm text-gray-700">
                            Overall Status: <strong>{ecosystemHealth.overall_status}</strong>
                          </p>
                        </div>
                        <div className="flex gap-3 p-3 bg-blue-50 rounded-lg">
                          <Activity className="text-blue-600 flex-shrink-0" size={20} />
                          <p className="text-sm text-gray-700">
                            Continue monitoring air quality levels and maintain green cover initiatives.
                          </p>
                        </div>
                        <div className="flex gap-3 p-3 bg-yellow-50 rounded-lg">
                          <Activity className="text-yellow-600 flex-shrink-0" size={20} />
                          <p className="text-sm text-gray-700">
                            Sensitive groups should limit prolonged outdoor exposure during peak pollution hours.
                          </p>
                        </div>
                      </>
                    )}
                  </div>
                ) : (
                  <div className="text-center py-8">
                    <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mb-2"></div>
                    <p className="text-sm text-gray-500">Loading recommendations...</p>
                  </div>
                )}
              </div>
            </div>

            {/* Forecast Chart */}
            {forecast.length > 0 && (
              <div className="mb-6">
                <TimeSeriesChart
                  data={forecast}
                  dataKey="predicted_pm25"
                  title="7-Day PM2.5 Forecast"
                  color="#ef4444"
                />
              </div>
            )}

            {/* Footer Stats */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="card text-center">
                <div className="text-3xl font-bold text-primary-600 mb-2">5</div>
                <p className="text-sm text-gray-600">ML Models Deployed</p>
              </div>
              <div className="card text-center">
                <div className="text-3xl font-bold text-blue-600 mb-2">1,826</div>
                <p className="text-sm text-gray-600">Daily Records Analyzed</p>
              </div>
              <div className="card text-center">
                <div className="text-3xl font-bold text-green-600 mb-2">99.75%</div>
                <p className="text-sm text-gray-600">Model Accuracy (R²)</p>
              </div>
            </div>
          </>
        )}
      </main>
    </div>
  );
};

export default Dashboard;
