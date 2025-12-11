import React, { useState, useEffect } from 'react';
import { TrendingUp, DollarSign, Clock, Leaf, Target, ArrowRight } from 'lucide-react';
import { getRestorationRecommendation, getRestorationScenarios } from '../utils/api';
import { formatCurrency } from '../utils/helpers';

const RestorationPlanner = () => {
  const [budget, setBudget] = useState(500);
  const [timeline, setTimeline] = useState(5);
  const [priority, setPriority] = useState('balanced');
  const [recommendation, setRecommendation] = useState(null);
  const [scenarios, setScenarios] = useState(null);
  const [loading, setLoading] = useState(false);

  const getRecommendation = async () => {
    setLoading(true);
    try {
      const data = await getRestorationRecommendation(budget, timeline, priority);
      console.log('Recommendation received:', data); // Debug log
      setRecommendation(data);
    } catch (error) {
      console.error('Error getting recommendation:', error);
      alert('Failed to get recommendation. Please check if the API server is running.');
    } finally {
      setLoading(false);
    }
  };

  const loadScenarios = async () => {
    try {
      const data = await getRestorationScenarios();
      setScenarios(data);
    } catch (error) {
      console.error('Error loading scenarios:', error);
    }
  };

  useEffect(() => {
    loadScenarios();
  }, []);

  const priorityOptions = [
    { value: 'balanced', label: 'Balanced', icon: Target },
    { value: 'air_quality', label: 'Best Air Quality', icon: Leaf },
    { value: 'cost', label: 'Lowest Cost', icon: DollarSign },
    { value: 'time', label: 'Fastest', icon: Clock },
  ];

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">Restoration Scenario Planner</h1>
          <p className="text-gray-600">
            Design optimal ecosystem restoration strategies for Delhi based on budget, timeline, and priorities
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Input Panel */}
          <div className="lg:col-span-1">
            <div className="card sticky top-4">
              <h2 className="text-xl font-semibold mb-6">Configure Scenario</h2>

              {/* Budget */}
              <div className="mb-6">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Budget (₹ Million)
                </label>
                <input
                  type="range"
                  min="10"
                  max="1000"
                  step="10"
                  value={budget}
                  onChange={(e) => setBudget(Number(e.target.value))}
                  className="w-full"
                />
                <div className="flex justify-between mt-2">
                  <span className="text-sm text-gray-500">₹10M</span>
                  <span className="text-lg font-semibold text-primary-600">₹{budget}M</span>
                  <span className="text-sm text-gray-500">₹1,000M</span>
                </div>
              </div>

              {/* Timeline */}
              <div className="mb-6">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Timeline (Years)
                </label>
                <input
                  type="range"
                  min="1"
                  max="15"
                  step="1"
                  value={timeline}
                  onChange={(e) => setTimeline(Number(e.target.value))}
                  className="w-full"
                />
                <div className="flex justify-between mt-2">
                  <span className="text-sm text-gray-500">1 year</span>
                  <span className="text-lg font-semibold text-blue-600">{timeline} years</span>
                  <span className="text-sm text-gray-500">15 years</span>
                </div>
              </div>

              {/* Priority */}
              <div className="mb-6">
                <label className="block text-sm font-medium text-gray-700 mb-3">
                  Optimization Priority
                </label>
                <div className="grid grid-cols-2 gap-2">
                  {priorityOptions.map((option) => {
                    const Icon = option.icon;
                    return (
                      <button
                        key={option.value}
                        onClick={() => setPriority(option.value)}
                        className={`p-3 rounded-lg border-2 transition-all ${
                          priority === option.value
                            ? 'border-primary-500 bg-primary-50'
                            : 'border-gray-200 hover:border-gray-300'
                        }`}
                      >
                        <Icon
                          className={`mx-auto mb-1 ${
                            priority === option.value ? 'text-primary-600' : 'text-gray-400'
                          }`}
                          size={20}
                        />
                        <p
                          className={`text-xs font-medium ${
                            priority === option.value ? 'text-primary-600' : 'text-gray-600'
                          }`}
                        >
                          {option.label}
                        </p>
                      </button>
                    );
                  })}
                </div>
              </div>

              <button
                onClick={getRecommendation}
                disabled={loading}
                className="btn-primary w-full"
              >
                {loading ? 'Calculating...' : 'Get Recommendation'}
              </button>
            </div>
          </div>

          {/* Results Panel */}
          <div className="lg:col-span-2 space-y-6">
            {recommendation && recommendation.recommended_scenario && (
              <>
                {/* Recommended Scenario */}
                <div className="card">
                  <div className="flex items-center gap-2 mb-4">
                    <Target className="text-primary-600" size={24} />
                    <h2 className="text-xl font-semibold">Recommended Scenario</h2>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                    <div className="p-4 bg-blue-50 rounded-lg">
                      <p className="text-sm text-gray-600 mb-1">PM2.5 Target</p>
                      <p className="text-2xl font-bold text-blue-600">
                        {recommendation.recommended_scenario.expected_outcomes.pm25_target}
                      </p>
                    </div>
                    <div className="p-4 bg-green-50 rounded-lg">
                      <p className="text-sm text-gray-600 mb-1">Total Cost</p>
                      <p className="text-2xl font-bold text-green-600">
                        {recommendation.recommended_scenario.expected_outcomes.total_cost}
                      </p>
                    </div>
                    <div className="p-4 bg-purple-50 rounded-lg">
                      <p className="text-sm text-gray-600 mb-1">Timeline</p>
                      <p className="text-2xl font-bold text-purple-600">
                        {recommendation.recommended_scenario.expected_outcomes.implementation_time}
                      </p>
                    </div>
                  </div>

                  <h3 className="text-lg font-semibold mb-3">Interventions</h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                    {Object.entries(recommendation.recommended_scenario.interventions).map(([key, value]) => (
                      <div key={key} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                        <span className="text-sm text-gray-700 capitalize">
                          {key.replace(/_/g, ' ')}
                        </span>
                        <span className="text-sm font-semibold text-gray-900">{value}</span>
                      </div>
                    ))}
                  </div>

                  {recommendation.alternatives_count > 0 && (
                    <div className="mt-4 p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
                      <p className="text-sm text-gray-700">
                        <strong>{recommendation.alternatives_count}</strong> alternative scenarios found
                        matching your constraints
                      </p>
                    </div>
                  )}
                </div>

                {/* Implementation Plan */}
                <div className="card">
                  <h3 className="text-lg font-semibold mb-4">Implementation Roadmap</h3>
                  <div className="space-y-4">
                    {[
                      {
                        phase: 'Phase 1',
                        duration: '0-2 years',
                        tasks: ['Policy framework', 'Pilot projects', 'Community engagement'],
                      },
                      {
                        phase: 'Phase 2',
                        duration: '2-4 years',
                        tasks: ['Scale interventions', 'Infrastructure development', 'Monitoring systems'],
                      },
                      {
                        phase: 'Phase 3',
                        duration: '4+ years',
                        tasks: ['Full deployment', 'Impact assessment', 'Continuous improvement'],
                      },
                    ].map((phase, index) => (
                      <div key={index} className="flex gap-4">
                        <div className="flex flex-col items-center">
                          <div className="w-10 h-10 rounded-full bg-primary-600 text-white flex items-center justify-center font-semibold">
                            {index + 1}
                          </div>
                          {index < 2 && <div className="w-0.5 h-full bg-gray-300 mt-2" />}
                        </div>
                        <div className="flex-1 pb-8">
                          <h4 className="font-semibold text-gray-900 mb-1">{phase.phase}</h4>
                          <p className="text-sm text-gray-500 mb-2">{phase.duration}</p>
                          <ul className="space-y-1">
                            {phase.tasks.map((task, taskIndex) => (
                              <li key={taskIndex} className="text-sm text-gray-700 flex items-center gap-2">
                                <ArrowRight size={14} className="text-primary-600" />
                                {task}
                              </li>
                            ))}
                          </ul>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </>
            )}

            {!recommendation && !loading && (
              <div className="card">
                <div className="text-center py-12">
                  <Target className="mx-auto mb-4 text-gray-400" size={48} />
                  <h2 className="text-xl font-semibold mb-2">Get Started</h2>
                  <p className="text-gray-600 mb-6">
                    Configure your budget, timeline, and priority on the left,<br />
                    then click "Get Recommendation" to see optimized restoration scenarios.
                  </p>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4 max-w-2xl mx-auto">
                    <div className="p-4 bg-blue-50 rounded-lg">
                      <div className="text-2xl font-bold text-blue-600 mb-1">100+</div>
                      <p className="text-sm text-gray-600">Optimized Scenarios</p>
                    </div>
                    <div className="p-4 bg-green-50 rounded-lg">
                      <div className="text-2xl font-bold text-green-600 mb-1">33%</div>
                      <p className="text-sm text-gray-600">Max PM2.5 Reduction</p>
                    </div>
                    <div className="p-4 bg-purple-50 rounded-lg">
                      <div className="text-2xl font-bold text-purple-600 mb-1">AI</div>
                      <p className="text-sm text-gray-600">Multi-Objective Optimizer</p>
                    </div>
                  </div>
                </div>
              </div>
            )}
            
            {loading && (
              <div className="card">
                <div className="text-center py-12">
                  <div className="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600 mb-4"></div>
                  <p className="text-gray-600">Analyzing 100+ scenarios to find the best match...</p>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default RestorationPlanner;
