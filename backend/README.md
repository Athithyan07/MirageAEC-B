# MirageAEC Backend API Gateway

High-Performance Generative 3D Architecture & Spatial Synthesis API for **MirageAEC**.

Built with **FastAPI**, **SQLAlchemy 2.0 (Async)**, **SQLite / PostgreSQL**, and **Shap-E** generative 3D pipelines.

---

## ⚡ Quick Start (Local)

### 1. Create Virtual Environment & Install Dependencies
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Run API Server
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

- **Interactive API Documentation:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **OpenAPI Schema:** [http://localhost:8000/openapi.json](http://localhost:8000/openapi.json)

---

## 🚀 Render Deployment Guide

### Deploying to Render (Free Web Service):
1. Create a **New Web Service** on [Render.com](https://render.com)
2. Connect your repository: `Athithyan07/MirageAEC-B`
3. Configure the following settings:
   - **Environment:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. Set Environment Variables (optional):
   - `DATABASE_URL`: `sqlite+aiosqlite:///./mirage_aec.db` (or Render PostgreSQL connection URL)
   - `SECRET_KEY`: *[Your secret JWT key]*
   - `CORS_ORIGINS`: `*` (or your frontend URL `https://mirageaec.onrender.com`)

---

## 📁 Project Architecture

```
backend/
├── main.py                  # Application entrypoint & middleware configuration
├── config.py                # Environment & application settings
├── database.py              # Async SQLAlchemy engine & session factory
├── models.py                # Database ORM models (User, Model3D, GenerationJob, Contact)
├── schemas.py               # Pydantic v2 validation schemas
├── requirements.txt         # Production dependencies
├── Procfile                 # Process deployment definition
├── routers/                 # Modular API endpoints
│   ├── auth.py              # User authentication & token verification
│   ├── generation.py        # 3D prompt synthesis & task queues
│   ├── models.py            # 3D assets metadata & query operations
│   ├── sketchfab.py         # Sketchfab 3D model search & proxy integration
│   ├── contact.py           # Inquiries & enterprise lead intake
│   └── system.py            # System health & diagnostic monitors
└── services/                # Business logic & AI pipelines
    ├── auth_service.py      # JWT encoding, decoding & password hashing
    ├── gen_service.py       # Generation state management
    ├── llm_service.py       # LLM prompt enhancement & design specs
    └── shap_e_service.py    # Generative 3D model synthesis (Shap-E)
```
