import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const healthCheck = async () => {
  const response = await api.get('/health');
  return response.data;
};

export const predictAQI = async (data) => {
  const response = await api.post('/predict/aqi', data);
  return response.data;
};

export const forecastPM25 = async (daysAhead = 7, modelType = 'prophet') => {
  const response = await api.post('/forecast/pm25', {
    days_ahead: daysAhead,
    model_type: modelType,
  });
  return response.data;
};

export const calculateEcosystemHealth = async (data) => {
  const response = await api.post('/ecosystem/health', data);
  return response.data;
};

export const getRestorationScenarios = async () => {
  const response = await api.get('/restoration/scenarios');
  return response.data;
};

export const getRestorationRecommendation = async (budget, timeline, priority) => {
  const response = await api.get('/restoration/recommend', {
    params: { budget, timeline, priority },
  });
  return response.data;
};

export const getModelsInfo = async () => {
  const response = await api.get('/models/info');
  return response.data;
};

export default api;
