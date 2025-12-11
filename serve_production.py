"""
Production server for Ecosystem Health Dashboard
Serves built frontend and proxies API requests
"""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
import uvicorn

# Import the API app
import sys
sys.path.append(str(Path(__file__).parent))
from api.main import app as api_app

# Paths
BASE_DIR = Path(__file__).parent
DIST_DIR = BASE_DIR / 'dashboard' / 'dist'

# Create main app
app = FastAPI(title="Ecosystem Health Platform")

# Mount API routes under /api prefix
app.mount("/api", api_app)

# Serve static files
if DIST_DIR.exists():
    app.mount("/assets", StaticFiles(directory=DIST_DIR / "assets"), name="assets")
    
    @app.get("/")
    async def serve_dashboard():
        """Serve the dashboard index.html"""
        return FileResponse(DIST_DIR / "index.html")
    
    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        """Serve SPA - return index.html for all routes"""
        file_path = DIST_DIR / full_path
        if file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(DIST_DIR / "index.html")
else:
    @app.get("/")
    async def no_build():
        return {
            "error": "Dashboard not built",
            "message": "Run 'python deploy_dashboard.py --mode build' first"
        }

if __name__ == "__main__":
    print("=" * 70)
    print("ECOSYSTEM HEALTH PLATFORM - PRODUCTION SERVER")
    print("=" * 70)
    print("\nServer starting...")
    print("\n📊 Dashboard:  http://localhost:8000")
    print("🔌 API:        http://localhost:8000/api")
    print("📚 API Docs:   http://localhost:8000/api/docs")
    print("\n" + "=" * 70)
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
