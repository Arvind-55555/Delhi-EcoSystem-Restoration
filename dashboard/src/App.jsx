import React from 'react';
import { BrowserRouter as Router, Routes, Route, NavLink } from 'react-router-dom';
import { Home, BarChart3, Sparkles } from 'lucide-react';
import Dashboard from './pages/Dashboard';
import RestorationPlanner from './pages/RestorationPlanner';
import './styles/index.css';

function App() {
  const navItems = [
    { path: '/', label: 'Dashboard', icon: Home },
    { path: '/planner', label: 'Restoration Planner', icon: Sparkles },
  ];

  return (
    <Router>
      <div className="flex h-screen bg-gray-50">
        {/* Sidebar */}
        <aside className="w-64 bg-white border-r border-gray-200">
          <div className="p-6">
            <div className="flex items-center gap-3 mb-8">
              <div className="p-2 bg-primary-600 rounded-lg">
                <BarChart3 className="text-white" size={24} />
              </div>
              <div>
                <h1 className="text-lg font-bold text-gray-900">EcoHealth</h1>
                <p className="text-xs text-gray-500">Delhi ML Platform</p>
              </div>
            </div>

            <nav className="space-y-2">
              {navItems.map((item) => {
                const Icon = item.icon;
                return (
                  <NavLink
                    key={item.path}
                    to={item.path}
                    className={({ isActive }) =>
                      `flex items-center gap-3 px-4 py-3 rounded-lg transition-colors ${
                        isActive
                          ? 'bg-primary-50 text-primary-700 font-medium'
                          : 'text-gray-600 hover:bg-gray-50'
                      }`
                    }
                  >
                    <Icon size={20} />
                    <span>{item.label}</span>
                  </NavLink>
                );
              })}
            </nav>
          </div>
        </aside>

        {/* Main Content */}
        <main className="flex-1 overflow-y-auto">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/planner" element={<RestorationPlanner />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;
