"""
Deployment script for Ecosystem Health Dashboard
Builds frontend and starts backend server
"""

import subprocess
import sys
import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
DASHBOARD_DIR = BASE_DIR / 'dashboard'
API_DIR = BASE_DIR / 'api'


def check_dependencies():
    """Check if required dependencies are installed"""
    logger.info("Checking dependencies...")
    
    # Check Node.js
    try:
        result = subprocess.run(['node', '--version'], capture_output=True, text=True)
        logger.info(f"✓ Node.js version: {result.stdout.strip()}")
    except FileNotFoundError:
        logger.error("✗ Node.js not found. Please install Node.js 18+")
        return False
    
    # Check npm
    try:
        result = subprocess.run(['npm', '--version'], capture_output=True, text=True)
        logger.info(f"✓ npm version: {result.stdout.strip()}")
    except FileNotFoundError:
        logger.error("✗ npm not found. Please install npm")
        return False
    
    # Check Python
    logger.info(f"✓ Python version: {sys.version.split()[0]}")
    
    return True


def install_frontend_dependencies():
    """Install frontend dependencies"""
    logger.info("\n" + "=" * 70)
    logger.info("INSTALLING FRONTEND DEPENDENCIES")
    logger.info("=" * 70)
    
    os.chdir(DASHBOARD_DIR)
    
    try:
        subprocess.run(['npm', 'install'], check=True)
        logger.info("✓ Frontend dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ Failed to install frontend dependencies: {e}")
        return False
    finally:
        os.chdir(BASE_DIR)


def build_frontend():
    """Build frontend for production"""
    logger.info("\n" + "=" * 70)
    logger.info("BUILDING FRONTEND")
    logger.info("=" * 70)
    
    os.chdir(DASHBOARD_DIR)
    
    try:
        subprocess.run(['npm', 'run', 'build'], check=True)
        logger.info("✓ Frontend built successfully")
        
        # Check if build directory exists
        dist_dir = DASHBOARD_DIR / 'dist'
        if dist_dir.exists():
            logger.info(f"✓ Build output: {dist_dir}")
            return True
        else:
            logger.error("✗ Build directory not found")
            return False
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ Frontend build failed: {e}")
        return False
    finally:
        os.chdir(BASE_DIR)


def start_development_servers():
    """Start both frontend and backend development servers"""
    logger.info("\n" + "=" * 70)
    logger.info("STARTING DEVELOPMENT SERVERS")
    logger.info("=" * 70)
    
    import threading
    
    def start_backend():
        logger.info("Starting backend API server on http://localhost:8000")
        os.chdir(BASE_DIR)
        subprocess.run([sys.executable, 'api/main.py'])
    
    def start_frontend():
        logger.info("Starting frontend dev server on http://localhost:3000")
        os.chdir(DASHBOARD_DIR)
        subprocess.run(['npm', 'run', 'dev'])
    
    # Start backend in separate thread
    backend_thread = threading.Thread(target=start_backend, daemon=True)
    backend_thread.start()
    
    # Wait a bit for backend to start
    import time
    time.sleep(3)
    
    # Start frontend in main thread
    start_frontend()


def start_production_server():
    """Start production server with built frontend"""
    logger.info("\n" + "=" * 70)
    logger.info("STARTING PRODUCTION SERVER")
    logger.info("=" * 70)
    
    logger.info("Server will be available at:")
    logger.info("  - API: http://localhost:8000")
    logger.info("  - Dashboard: http://localhost:8000/dashboard")
    logger.info("  - API Docs: http://localhost:8000/docs")
    
    os.chdir(BASE_DIR)
    subprocess.run([sys.executable, 'api/main.py'])


def main():
    """Main deployment workflow"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Deploy Ecosystem Health Dashboard')
    parser.add_argument('--mode', choices=['dev', 'build', 'prod'], default='dev',
                        help='Deployment mode: dev (development servers), build (build only), prod (production)')
    parser.add_argument('--skip-install', action='store_true',
                        help='Skip npm install step')
    
    args = parser.parse_args()
    
    logger.info("=" * 70)
    logger.info("ECOSYSTEM HEALTH DASHBOARD DEPLOYMENT")
    logger.info("=" * 70)
    
    # Check dependencies
    if not check_dependencies():
        logger.error("Dependency check failed. Exiting.")
        sys.exit(1)
    
    # Install dependencies
    if not args.skip_install:
        if not install_frontend_dependencies():
            logger.error("Failed to install dependencies. Exiting.")
            sys.exit(1)
    
    if args.mode == 'dev':
        # Development mode - run both servers
        logger.info("\nMode: DEVELOPMENT")
        start_development_servers()
    
    elif args.mode == 'build':
        # Build mode - only build frontend
        logger.info("\nMode: BUILD ONLY")
        if build_frontend():
            logger.info("\n✓ Build complete!")
            logger.info(f"Build output: {DASHBOARD_DIR / 'dist'}")
        else:
            sys.exit(1)
    
    elif args.mode == 'prod':
        # Production mode - build and serve
        logger.info("\nMode: PRODUCTION")
        if build_frontend():
            start_production_server()
        else:
            sys.exit(1)


if __name__ == "__main__":
    main()
