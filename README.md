# ML-Based Ground/Soil Risk Prediction and Monitoring System

An academic decision-support prototype that accepts urban road segment measurements (pavement condition, soil properties, hydrological data, infrastructure state) and classifies collapse susceptibility into four risk levels (**Low**, **Moderate**, **High**, **Critical**).

> [!CAUTION]
> This system is an academic decision-support prototype built on a synthetic dataset. It must **not** be presented as a replacement for professional geotechnical engineering assessment.

---

## 🏗 Architecture & Stack

- **Frontend**: Vue 3 + TypeScript + Vite + Vue Router + Pinia
- **Backend**: FastAPI (Python 3.11+) + Pydantic V2 + SQLAlchemy ORM
- **Database**: PostgreSQL 15 (with Alembic migrations)
- **ML Pipeline**: Scikit-Learn + XGBoost + Custom Adapters & Preprocessors
- **Containerization**: Docker & Docker Compose (Multi-stage builds + Nginx)

---

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.11+
- Node.js 20+
- Docker Desktop (optional, for containerized run)

### 2. Environment Setup

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

### 3. Running with Docker Compose (Recommended)

```bash
docker compose up --build
```

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/api/docs
- **Health Check**: http://localhost:8000/api/health

---

## 🧪 Local Development & Testing

### Backend & ML Tests

```bash
# Run test suite
python -m pytest

# Run FastAPI dev server directly
cd backend
uvicorn app.main:app --reload --port 8000
```

### Frontend Development

```bash
cd frontend
npm install
npm run dev
```

---

## 📚 Project Documentation

- [PRD.md](PRD.md) — Product Requirements Document
- [ARCHITECTURE.md](ARCHITECTURE.md) — System Architecture Specifications
- [TODO.md](TODO.md) — Implementation Phases & Task Tracking
- [AGENTS.md](AGENTS.md) — Development Rules & Conventions
