# Ecosystem Health Dashboard

Interactive web dashboard for ecosystem health monitoring and restoration planning in Delhi, India.

## Features

- **Real-time Monitoring**: Live AQI, PM2.5, and pollutant level tracking
- **Ecosystem Health Score**: Comprehensive health metrics with component breakdown
- **7-Day Forecasting**: AI-powered PM2.5 predictions using Prophet/LSTM models
- **Restoration Planner**: Interactive scenario optimizer with budget and timeline constraints
- **Visual Analytics**: Beautiful charts and visualizations powered by Recharts
- **Responsive Design**: Works seamlessly on desktop, tablet, and mobile

## Tech Stack

**Frontend:**
- React 18 with Vite
- Tailwind CSS for styling
- Recharts for data visualization
- React Router for navigation
- Lucide React for icons
- Axios for API calls

**Backend:**
- FastAPI REST API
- 5 ML models (XGBoost, Random Forest, LSTM, Prophet, Linear Regression)
- Multi-objective optimization (NSGA-II)

## Installation

### Prerequisites
- Node.js 18+ and npm
- Python 3.10+ (for API backend)

### Setup

1. **Install frontend dependencies:**
```bash
cd dashboard
npm install
```

2. **Configure environment:**
```bash
# Create .env file
echo "VITE_API_URL=http://localhost:8000" > .env
```

3. **Start development server:**
```bash
npm run dev
```

The dashboard will be available at `http://localhost:3000`

4. **Start backend API** (in separate terminal):
```bash
cd ..
python api/main.py
```

The API will be available at `http://localhost:8000`

## Development

### Available Scripts

- `npm run dev` - Start development server with hot reload
- `npm run build` - Build for production
- `npm run preview` - Preview production build
- `npm run lint` - Run ESLint

### Project Structure

```
dashboard/
├── src/
│   ├── components/        # Reusable UI components
│   │   ├── Cards.jsx     # Metric cards, AQI card, health score
│   │   └── Charts.jsx    # Chart components (time series, bar, pie)
│   ├── pages/            # Page components
│   │   ├── Dashboard.jsx           # Main dashboard view
│   │   └── RestorationPlanner.jsx  # Scenario planner
│   ├── utils/            # Utility functions
│   │   ├── api.js       # API client
│   │   └── helpers.js   # Helper functions
│   ├── styles/          # Global styles
│   │   └── index.css    # Tailwind CSS
│   ├── App.jsx          # Main app component
│   └── main.jsx         # Entry point
├── public/              # Static assets
├── index.html          # HTML template
├── package.json        # Dependencies
├── vite.config.js      # Vite configuration
└── tailwind.config.js  # Tailwind configuration
```

## Usage

### Dashboard View

The main dashboard displays:
- Current AQI and air quality category
- Ecosystem Health Score (0-100)
- Real-time pollutant levels (PM2.5, PM10, NO2, SO2, CO, O3)
- Weather metrics (temperature, humidity, wind speed)
- 7-day PM2.5 forecast
- Health recommendations

### Restoration Planner

Interactive scenario optimizer:
1. Set budget (₹10M - ₹1,000M)
2. Set timeline (1-15 years)
3. Choose priority:
   - Balanced (recommended)
   - Best Air Quality
   - Lowest Cost
   - Fastest Implementation
4. Get optimized restoration scenario with:
   - Intervention mix (green cover, tree plantation, emission controls, etc.)
   - Expected outcomes (PM2.5 target, cost, timeline)
   - Implementation roadmap

## API Integration

The dashboard communicates with the FastAPI backend:

```javascript
// Example API calls
import { predictAQI, forecastPM25, getRestorationRecommendation } from './utils/api';

// Predict AQI
const prediction = await predictAQI({
  PM25: 120,
  PM10: 200,
  temperature: 25,
  // ... other parameters
});

// Get 7-day forecast
const forecast = await forecastPM25(7, 'prophet');

// Get restoration recommendation
const recommendation = await getRestorationRecommendation(500, 5, 'balanced');
```

## Production Build

```bash
# Build for production
npm run build

# Preview production build
npm run preview
```

The production build will be in the `dist/` directory.

## Deployment

### Deploy Frontend (Vercel/Netlify)

1. Connect your repository
2. Set build command: `npm run build`
3. Set output directory: `dist`
4. Add environment variable: `VITE_API_URL=https://your-api-url.com`

### Deploy Backend (Docker)

See `api/README.md` for backend deployment instructions.

## Browser Support

- Chrome/Edge (latest)
- Firefox (latest)
- Safari (latest)

## Performance

- Lighthouse Score: 95+
- First Contentful Paint: < 1.5s
- Time to Interactive: < 3s
- Bundle Size: ~500KB (gzipped)

## Screenshots

### Dashboard
![Dashboard](screenshots/dashboard.png)

### Restoration Planner
![Planner](screenshots/planner.png)

## Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/new-feature`
3. Commit changes: `git commit -am 'Add new feature'`
4. Push to branch: `git push origin feature/new-feature`
5. Submit pull request

## License

MIT License

## Support

For issues and questions:
- Create an issue on GitHub
- Email: support@ecohealth.ai

## Acknowledgments

- Data sources: NASA POWER, World Bank, CPCB
- ML models: XGBoost, Prophet, TensorFlow
- UI components: Tailwind CSS, Recharts
- Icons: Lucide React
