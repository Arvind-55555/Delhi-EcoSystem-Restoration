export const getAQIColor = (aqi) => {
  if (aqi <= 50) return 'text-green-600 bg-green-100';
  if (aqi <= 100) return 'text-yellow-600 bg-yellow-100';
  if (aqi <= 150) return 'text-orange-600 bg-orange-100';
  if (aqi <= 200) return 'text-red-600 bg-red-100';
  if (aqi <= 300) return 'text-purple-600 bg-purple-100';
  return 'text-red-900 bg-red-200';
};

export const getAQICategory = (aqi) => {
  if (aqi <= 50) return 'Good';
  if (aqi <= 100) return 'Moderate';
  if (aqi <= 150) return 'Unhealthy for Sensitive Groups';
  if (aqi <= 200) return 'Unhealthy';
  if (aqi <= 300) return 'Very Unhealthy';
  return 'Hazardous';
};

export const getHealthScoreColor = (score) => {
  if (score >= 75) return 'text-green-600 bg-green-100';
  if (score >= 60) return 'text-blue-600 bg-blue-100';
  if (score >= 45) return 'text-yellow-600 bg-yellow-100';
  if (score >= 30) return 'text-orange-600 bg-orange-100';
  return 'text-red-600 bg-red-100';
};

export const getHealthScoreLabel = (score) => {
  if (score >= 75) return 'Excellent';
  if (score >= 60) return 'Good';
  if (score >= 45) return 'Moderate';
  if (score >= 30) return 'Poor';
  return 'Critical';
};

export const formatNumber = (num, decimals = 2) => {
  return Number(num).toFixed(decimals);
};

export const formatCurrency = (amount) => {
  return `₹${amount.toFixed(1)}M`;
};

export const formatDate = (dateString) => {
  return new Date(dateString).toLocaleDateString('en-IN', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
  });
};
