# MLDevOps Engineer Portfolio

Welcome! I'm Kashad Turner-Warren, and this is my personal portfolio built through Udacity's **Machine Learning DevOps Engineer Nanodegree Program**. This program strengthened my ability to build scalable, production-ready machine learning systems and pipelines. It combines software engineering, DevOps, and machine learning into real-world projects that demonstrate my capabilities and aspirations as a next-generation ML/MLOps engineer.

---

## 🌟 Program Focus & Goals

This program sharpened my ability to:
- Deploy ML models in production environments without relying solely on cloud tools like SageMaker.
- Automate retraining pipelines using CI/CD and CT (Continuous Training) workflows.
- Monitor live model performance and detect drift for retraining or alerting.
- Engineer reproducible pipelines using MLflow, FastAPI, GitHub Actions, and more.
- Understand MLOps workflows from experimentation to scalable deployment and maintenance.

I'm passionate about solving real-world problems in cybersecurity and automation using AI. These projects reflect my ability to merge data science, infrastructure, and DevOps into secure, scalable systems.

---

## 📦 Featured Projects

### 1. NYC Airbnb Price Prediction Pipeline
**Status:** ✅ Completed  
**GitHub:** [Repo](https://github.com/krillavilla/Build-ML-Pipeline_for_Short-Term_Rental_Prices)  
**W&B Dashboard:** [Project Link](https://wandb.ai/build-ml-pipeline-for-short-term-rental-prices/nyc_airbnb)

**Highlights:**
- Built a reproducible ML pipeline from scratch
- Tracked experiments, metrics, and models using MLflow and W&B
- Automated training, testing, and packaging via Hydra + Python
- Integrated CI/CD workflows and weekly data updates

**Tools:** MLflow, Weights & Biases, scikit-learn, Hydra, pytest, pandas, GitHub Actions

---

### 2. Customer Churn Prediction System
**Status:** ✅ Completed  
**GitHub:** [Repo](https://github.com/krillavilla/MLDevOps_Program/tree/main/Predict_Customer_Churn)

**Highlights:**
- Refactored an exploratory notebook into a clean, modular Python package
- Implemented logging, unit testing, and PEP8 formatting
- Added both CLI and interactive functionality
- Built for reusability and maintainability in production

**Tools:** Python, scikit-learn, PyLint, AutoPEP8, pytest, logging

---

### 3. ML Model FastAPI Deployment on Heroku
**Status:** ✅ Completed  
**GitHub:** [Repo](https://github.com/krillavilla/Deploying_MLmodel_CloudApp_using_FastAPI)

**Highlights:**
- Developed an ML inference API using FastAPI
- Built a CI/CD pipeline with GitHub Actions
- Version-controlled code and data using DVC
- Deployed and tested the API live on Heroku with Swagger UI docs

**Tools:** FastAPI, GitHub Actions, DVC, pytest, Heroku, Pydantic, Swagger

---

### 4. Dynamic Risk Assessment System (Capstone)
**Status:** 🚧 In Progress  
**GitHub:** [Repo](https://github.com/krillavilla/Dynamic_Risk_Assesment_System)

**Highlights:**
- Developed ingestion, training, scoring, and deployment scripts
- Built a FastAPI-based service to expose API endpoints
- Monitors model drift and automates retraining via CronJobs
- Emphasizes clean modular design and automation

**Workspace Layout:**
- `/sourcedata/` - raw input data for model training
- `/ingesteddata/` - deduplicated data used for training
- `/models/` - stored pickle models
- `/production_deployment/` - final deployed assets
- `/testdata/` - used to validate and evaluate models

**Scripts Included:**
- `training.py`, `scoring.py`, `deployment.py`, `ingestion.py`
- `diagnostics.py`, `reporting.py`, `app.py`, `wsgi.py`, `apicalls.py`, `fullprocess.py`

**Optional Kubernetes Deployment:**
- `deployment.yaml`, `service.yaml`, `cronjob.yaml`, `configmap.yaml`
- Automate deployment via `kubectl` and Docker images
- Hourly retraining scheduled with `CronJob`
- K8s setup supports scalability and API monitoring

**Tools:** Docker, Kubernetes, CronJob, FastAPI, Python, scikit-learn

---

## 🧠 Skills & Tech Stack

### MLOps & DevOps
- CI/CD Pipelines with GitHub Actions
- MLflow & Weights & Biases for tracking
- Experiment tracking & model versioning
- DVC for data and model reproducibility
- Containerization (Docker)
- Kubernetes & CronJobs (for optional deployment)
- Heroku cloud deployment

### ML Engineering
- Feature engineering and EDA
- Model retraining and drift detection
- ML pipeline automation and orchestration
- Model testing and validation

### Software Engineering
- PEP8 + Clean Code Practices
- Logging and Debugging
- Modular Design Patterns
- Version Control with Git/GitHub
- Automated Testing (pytest)

---

## 📁 Getting Started
Each project in this portfolio is maintained as a Git submodule. To clone the portfolio and its projects:

```bash
# Clone the main repository
git clone https://github.com/krillavilla/MLDevOps_Program.git

# Initialize and update submodules
cd MLDevOps_Program
git submodule init
git submodule update
```

---

## 🔮 Career Vision
My goal is to bridge the worlds of **machine learning, DevOps, and cybersecurity**. I want to engineer intelligent systems that are not only smart but secure, scalable, and reliable. I’m excited about opportunities in:
- AI for Cybersecurity (Adversarial ML, Risk Assessment)
- Red Team Automation
- ML System Architecture & Deployment

If you're working on something in this space or think we should collaborate—let's connect!

---

## 📄 License
This project is licensed under the MIT License. See the LICENSE file for details.

