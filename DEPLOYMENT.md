# HALT-NN Free Deployment Guide

## Option 1: Render.com (Recommended - Free Tier)

### Setup
1. Create account at https://render.com
2. Connect GitHub repository
3. Deploy as Web Service

### render.yaml
```yaml
services:
  - type: web
    name: halt-nn-api
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn api:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: PYTHON_VERSION
        value: 3.11
```

## Option 2: Railway.app (Free Tier)

### Procfile
```
web: uvicorn api:app --host 0.0.0.0 --port $PORT
```

## Option 3: Local Network Deployment

### Start server accessible on local network:
```bash
python -m uvicorn api:app --host 0.0.0.0 --port 8000
```

Access from any device on same WiFi:
- http://YOUR_IP:8000

## Docker Deployment (Any Platform)

### Build and run:
```bash
docker build -t halt-nn .
docker run -p 8000:8000 halt-nn
```
