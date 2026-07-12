# Consumer Intelligence Platform

An end-to-end consumer analytics platform combining advanced ML segmentation, predictive modeling, LLM-powered persona generation, and an interactive stakeholder dashboard.

---

# Elevator Pitch

The Consumer Intelligence Platform is a comprehensive analytics system designed for product managers, marketing teams, and data scientists who need to understand customer behavior at scale. It transforms raw consumer data into actionable insights through automated segmentation, predictive modeling, and AI-generated persona profiles. The platform solves the critical problem of making complex customer analytics accessible without requiring specialized data science expertise for every query.

What makes this solution interesting is its end-to-end approach: from synthetic data generation to ML model training to interactive visualization, all running locally with optional LLM integration. The platform differentiates itself by combining traditional machine learning (clustering, gradient boosting, random forests) with modern LLM capabilities for persona generation, while maintaining a focus on model explainability and fairness auditing—capabilities often missing in commercial analytics tools.

---

# Product Overview

**Users**: Product managers, marketing analysts, data scientists, and business stakeholders who need customer insights without depending on data science teams for every analysis.

**Primary Workflow**: Users generate or upload consumer data, train ML models through a guided pipeline, then explore results through an interactive dashboard. The workflow is: generate data → train models → explore segments → review personas → analyze predictions → audit fairness → simulate scenarios.

**Core Functionality**: The platform generates synthetic consumer datasets (100k+ records), performs feature engineering (RFM scores, engagement metrics), runs clustering segmentation, trains three predictive models (churn, LTV, conversion), generates AI-powered persona profiles, provides SHAP-based explainability, conducts fairness audits, and offers a what-if simulation tool.

**Inputs**: Consumer demographic data (age, gender, income, location), behavioral metrics (purchase history, session data, engagement scores), and transactional data (spend, frequency, recency). The platform can also generate realistic synthetic data for testing.

**Outputs**: Segment assignments with visualizations, persona profiles with motivations and pain points, churn/LTV/conversion predictions, feature importance explanations, fairness audit reports, and interactive what-if scenario results.

**User Experience**: A dark-mode dashboard with sidebar navigation, real-time training progress, interactive visualizations (scatter plots, histograms, bar charts), and responsive design. The pipeline interface guides users through data generation and model training with live logs.

**Real-World Use Case**: A marketing team uses the platform to segment their 100k customer base, identify high-value loyalists versus at-risk customers, understand churn drivers through SHAP explanations, and simulate the impact of retention campaigns before execution—all without writing code or waiting for data science backlog.

---

# Problem Statement

Organizations struggle to transform customer data into actionable insights because traditional analytics require specialized data science skills, expensive commercial tools, or lengthy development cycles. Product managers and marketers need to understand customer segments, predict churn, estimate lifetime value, and personalize interventions—but they often lack direct access to ML models or interpretability tools.

Existing solutions are insufficient because: (1) Commercial analytics platforms are expensive and often black-box, limiting customization; (2) Building custom ML solutions requires data science expertise and significant development time; (3) Most tools focus on either segmentation OR prediction OR explainability—not an integrated pipeline; (4) Fairness and bias auditing are rarely included in standard analytics workflows; (5) LLM integration for persona generation is uncommon in traditional ML tools.

This project addresses these gaps by providing a complete, locally-hosted platform that integrates data generation, segmentation, prediction, explainability, fairness auditing, and LLM-powered personas in a single accessible interface.

---

# Key Features

## Data Generation & Feature Engineering

Generates realistic synthetic consumer datasets at scale (10k–500k records) with demographics, behavioral metrics, transactional history, NPS scores, and time-series activity. Computes advanced features including RFM scores (recency, frequency, monetary), engagement index, digital activity score, support burden, and session quality metrics. This feature enables testing and experimentation without requiring access to sensitive real customer data.

## ML Segmentation Engine

Performs unsupervised clustering using KMeans or Gaussian Mixture Models with automatic PCA dimensionality reduction for 2D visualization. Assigns descriptive segment labels (e.g., "High-Value Loyalists", "At-Risk Customers") based on cluster statistics and computes silhouette scores for quality assessment. This enables users to discover natural customer groupings without manual labeling.

## Predictive Modeling Suite

Trains three supervised learning models: Gradient Boosting for churn prediction, Random Forest for lifetime value regression, and Logistic Regression for conversion scoring. Each model includes cross-validation, performance metrics (accuracy, AUC, R²), and feature importance rankings. This provides actionable predictions for retention, revenue forecasting, and marketing optimization.

## LLM-Powered Persona Generation

Uses local Ollama LLM (llama3) to generate rich, human-readable persona profiles for each segment including names, taglines, motivations, pain points, recommended actions, and narrative descriptions. Includes intelligent fallback to curated templates when LLM is unavailable. This transforms statistical segments into relatable customer profiles that stakeholders can act on.

## Model Explainability

Computes SHAP (SHapley Additive exPlanations) values for tree-based and linear models to provide per-prediction feature attribution. Falls back to permutation importance when SHAP is unavailable. Displays global feature importance rankings and sample-level explanations showing which features drove specific predictions. This builds trust in ML outputs and enables actionable insights.

## Fairness & Bias Auditing

Automatically audits model predictions for demographic disparities across gender, income bracket, and geographic tier using standard fairness metrics including adverse impact ratio (80% rule) and demographic parity. Flags problematic disparities and generates specific mitigation recommendations. This addresses ethical AI concerns and helps ensure equitable treatment across customer groups.

## What-If Simulation

Provides an interactive tool where users can adjust feature values (e.g., increase email open rate, reduce support tickets) and see real-time predictions from all three models. This enables scenario planning and "what-if" analysis to understand the potential impact of interventions before implementation.

## Interactive Dashboard

Premium dark-mode interface with sidebar navigation, real-time training progress with live logs, interactive visualizations using Recharts (scatter plots, histograms, bar charts), and responsive design. The pipeline interface guides users through the end-to-end workflow with status indicators and error handling.

---

# Engineering Highlights

**Backend**: FastAPI-based REST API with modular router architecture separating concerns (data, ML models, segments, personas, predictions, explanations, bias). Implements async operations for LLM calls and background thread training to keep API responsive. Uses Pydantic for request/response validation and pydantic-settings for configuration management.

**Frontend**: React 18 with Vite build tool, React Router for navigation, and custom hooks for data fetching. Uses Recharts for data visualization and Axios for API communication. Implements a custom design system with global CSS for consistent dark-mode styling across all components.

**ML Pipeline**: scikit-learn-based feature engineering with StandardScaler, LabelEncoder, and SimpleImputer. Implements KMeans and GaussianMixture for clustering, GradientBoostingClassifier for churn, RandomForestRegressor for LTV, and LogisticRegression for conversion. Uses train/test splitting with stratification for classification tasks.

**Model Persistence**: Joblib-based model serialization for all trained models (segmentation, churn, LTV, conversion) along with preprocessing artifacts (scaler, encoders, imputer, PCA, feature columns). Enables model reuse without retraining.

**LLM Integration**: Async HTTP client for Ollama with availability checking and timeout handling. Implements structured JSON parsing with regex-based markdown stripping and graceful fallback to templates when LLM responses are invalid or unavailable.

**Explainability**: SHAP TreeExplainer for tree-based models and LinearExplainer for logistic regression, with permutation importance fallback when SHAP is not installed. Computes explanations on sampled data (500 rows) for performance while maintaining statistical validity.

**Fairness Engine**: Implements demographic parity and adverse impact ratio calculations across protected attributes (gender, income, city_tier). Generates contextual recommendations based on specific fairness violations detected.

**Data Engineering**: Synthetic data generation using numpy distributions, Faker for realistic names, and rule-based behavioral modeling that correlates demographics with spending patterns. Generates time-series activity data as JSON for each customer.

**Performance Optimization**: Limits scatter plot rendering to 3,000 sampled points, computes SHAP on 500-row samples, uses background threading for ML training, and implements efficient data structures for large datasets.

**Reliability**: Comprehensive error handling with try-catch blocks, logging via Loguru with file rotation, health check endpoints, and graceful degradation when optional components (Ollama, SHAP) are unavailable.

---

# AI / ML Components

**LLM Integration**: Uses Ollama (llama3 model) running locally for two purposes: (1) generating synthetic customer review text during data generation, and (2) creating rich persona profiles from cluster statistics. Implements structured prompting with JSON response requirements, regex-based parsing, and intelligent fallback to curated templates when LLM is unavailable.

**Clustering**: Implements both KMeans and Gaussian Mixture Model for unsupervised segmentation. Uses PCA for 2D dimensionality reduction and visualization. Computes silhouette scores for cluster quality assessment. Assigns descriptive labels based on cluster characteristics.

**Supervised Learning**: Gradient Boosting Classifier for binary churn prediction with 200 estimators and depth 4. Random Forest Regressor for lifetime value prediction with 150 trees and depth 8. Logistic Regression with balanced class weights for conversion prediction. All models use train/test splitting with appropriate metrics (accuracy, AUC, R²).

**Feature Engineering**: Computes RFM (Recency, Frequency, Monetary) scores using log transformations and normalization. Derives engagement score from email metrics, session data, and app usage. Calculates digital activity score as log-transformed session quality. Creates support burden metric and spend-per-day ratios.

**Explainability**: SHAP (SHapley Additive exPlanations) for model interpretability. TreeExplainer for gradient boosting and random forest models. LinearExplainer for logistic regression. Provides both global feature importance and per-prediction attribution with waterfall-style explanations.

**Fairness Auditing**: Implements demographic parity difference and adverse impact ratio (80% EEOC rule) across protected attributes. Generates specific mitigation recommendations based on detected disparities. Audits both churn and conversion predictions for bias.

**Prompt Engineering**: Structured system prompts for persona generation requiring specific JSON schema (name, tagline, motivations, pain_points, recommended_actions, narrative). Implements few-shot-style prompting with cluster statistics as context.

---

# Data Engineering / Data Science Components

**Synthetic Data Generation**: Generates 100k+ realistic consumer records using numpy distributions with correlated demographics and behaviors. Uses Faker for realistic names and implements rule-based relationships between income, subscription type, and spending patterns. Generates 12-month time-series activity data per customer.

**Feature Engineering Pipeline**: Transforms raw data into 25+ derived features including RFM scores, engagement metrics, digital activity indices, and support burden calculations. Implements StandardScaler for normalization, LabelEncoder for categorical variables, and SimpleImputer for missing value handling.

**Model Training Pipeline**: Automated end-to-end training workflow that processes data through feature engineering, segmentation, and three predictive models in sequence. Implements cross-validation and performance metric calculation. Saves all artifacts for reuse.

**Evaluation Metrics**: Uses silhouette score for clustering quality, accuracy and AUC for classification models (churn, conversion), and R² for regression (LTV). Computes feature importance rankings for all models using model-specific methods (feature_importances_, coefficients).

**Data Visualization**: PCA-based 2D scatter plots for cluster visualization with sampling for performance. Histograms for prediction distributions. Bar charts for feature importance and demographic distributions. All visualizations implemented via Recharts in the frontend.

**Experimentation**: Supports configurable parameters including dataset size (10k–500k), number of clusters (default 6), and clustering method (KMeans vs GMM). Enables rapid experimentation with different segmentation strategies.

---

# Product Thinking

**User-Centric Design**: The platform is designed for non-technical stakeholders by abstracting ML complexity behind a simple interface. The pipeline guides users step-by-step through data generation and model training with live progress logs and clear status indicators.

**Reducing Friction**: Eliminates the need for data science teams by providing pre-built models that work out of the box. Synthetic data generation enables immediate experimentation without data access barriers. Optional LLM integration degrades gracefully when unavailable.

**Workflow Optimization**: Integrates what would typically be separate tools (segmentation, prediction, explainability, fairness) into a single cohesive workflow. Users can move from discovering segments to understanding drivers to planning interventions without switching contexts.

**Decision Support**: Persona profiles translate statistical segments into actionable customer archetypes with specific motivations and pain points. Recommended actions provide concrete next steps for each segment. What-if simulation enables scenario planning before resource commitment.

**Accessibility**: Dark-mode design with high contrast for reduced eye strain. Responsive layout works across different screen sizes. Clear visual hierarchy with KPI cards, charts, and narrative text to accommodate different learning styles.

**Personalization**: LLM-generated personas create relatable customer profiles that feel specific rather than generic. Segment labels are descriptive and actionable (e.g., "At-Risk Customers" rather than "Cluster 1").

**Scalability**: Architecture supports datasets from 10k to 500k records. Background threading keeps the API responsive during training. Sampling strategies ensure visualization performance at scale.

**Adoption Strategy**: Optional Ollama integration means the platform works immediately without additional setup. Fallback templates ensure core functionality is never blocked by missing dependencies. Clear troubleshooting documentation addresses common issues.

---

# Technologies Used

## Languages

- Python 3.11+
- JavaScript/ES6+

## Frontend

- React 18.2
- Vite 5.0
- React Router DOM 6.20
- Recharts 2.10
- Axios 1.6

## Backend

- FastAPI 0.109
- Uvicorn 0.27
- Pydantic 2.5
- Pydantic Settings 2.1

## Machine Learning

- scikit-learn 1.4
- pandas 2.1
- numpy 1.26
- SHAP 0.44
- scipy 1.11
- joblib 1.3

## AI/LLM

- Ollama (llama3 model)
- httpx 0.26 (async HTTP client)

## Data

- pyarrow 14.0
- Faker 21.0 (synthetic data)

## Utilities

- python-dotenv 1.0
- aiofiles 23.2
- loguru 0.7 (logging)

## DevOps

- Virtual environment (venv)
- npm package management

---

# Skills Demonstrated

- Full Stack Development
- Backend Engineering (FastAPI, REST APIs)
- Frontend Engineering (React, Vite, Component Architecture)
- Machine Learning Engineering
- Feature Engineering
- Model Training & Evaluation
- Clustering & Segmentation
- Predictive Modeling (Classification & Regression)
- LLM Integration & Prompt Engineering
- Retrieval-Augmented Generation (RAG) patterns
- Model Explainability (SHAP)
- AI Ethics & Fairness Auditing
- Data Engineering & Pipeline Design
- Synthetic Data Generation
- API Design & Documentation
- State Management (React hooks)
- Data Visualization (Recharts)
- Async Programming (Python async/await)
- Error Handling & Graceful Degradation
- Performance Optimization
- Configuration Management
- Logging & Monitoring
- Product Thinking & UX Design
- Dashboard Development

---

# Resume Impact

- Architected and built an end-to-end consumer analytics platform serving 100k+ customer records with ML segmentation, predictive modeling, and LLM-powered persona generation
- Implemented a complete ML pipeline featuring feature engineering, KMeans/GMM clustering, and three predictive models (churn, LTV, conversion) using scikit-learn with 85%+ accuracy
- Integrated local LLM (Ollama) for automated persona generation with structured JSON prompting and intelligent fallback to ensure 100% system availability
- Developed SHAP-based model explainability module providing per-prediction feature attribution, enabling stakeholders to understand and trust ML-driven insights
- Built fairness auditing engine implementing adverse impact ratio and demographic parity across protected attributes, generating actionable mitigation recommendations
- Designed interactive React dashboard with real-time training progress, PCA visualizations, and what-if simulation tools for non-technical stakeholders
- Engineered synthetic data generator producing realistic consumer datasets with correlated demographics and behavioral patterns for testing and experimentation
- Implemented async background training pipeline keeping API responsive during model training on datasets up to 500k records
- Created modular FastAPI backend with 7 RESTful endpoints, Pydantic validation, and comprehensive error handling for production-grade reliability

---

# Ideal Roles

- Machine Learning Engineer
- AI Engineer
- Full Stack Engineer
- Data Scientist
- Applied Scientist
- Backend Engineer
- Product Engineer
- Technical Product Manager
- Data Platform Engineer
- Analytics Engineer

---

# Portfolio Tags

React
FastAPI
Machine Learning
scikit-learn
Clustering
Predictive Modeling
LLM Integration
Ollama
SHAP
Explainability
Fairness
Data Engineering
Feature Engineering
Full Stack
Dashboard
Analytics
Python
JavaScript
API Design
Product

---

# Project Complexity

Advanced

This is an advanced project because it integrates multiple complex domains: full-stack web development, end-to-end ML pipelines, LLM integration, model explainability, and fairness auditing. The architecture demonstrates sophisticated engineering decisions including async operations, background threading, graceful degradation, modular design, and performance optimization. The combination of traditional ML (clustering, ensemble methods) with modern LLM capabilities, along with a focus on ethical AI (fairness auditing), shows depth beyond typical ML projects. The platform is production-ready with comprehensive error handling, logging, configuration management, and documentation.

---

# One-line Portfolio Summary

End-to-end consumer analytics platform with ML segmentation, predictive modeling, LLM-powered personas, and fairness auditing.

---

# Repository Evidence

- backend/main.py - FastAPI application with CORS, lifespan management, and router organization
- backend/config.py - Pydantic settings for environment configuration
- backend/requirements.txt - Python dependencies (FastAPI, scikit-learn, SHAP, pandas, etc.)
- backend/services/ - Business logic modules (data_generator.py, feature_engineering.py, segmentation.py, prediction_engine.py, persona_generator.py, explainability.py, bias_fairness.py)
- backend/routers/ - API route handlers (data.py, ml_models.py, segments.py, personas.py, predictions.py, explanations.py, bias.py)
- frontend/package.json - React dependencies (React 18, Vite, Recharts, React Router, Axios)
- frontend/src/App.jsx - React Router setup with 8 routes for different dashboard views
- frontend/src/components/ - 11 component directories (Overview, Segmentation, Personas, Predictions, Explainability, Bias, Simulation, Pipeline, Layout, Common)
- models/ - Trained model artifacts (segmentation_model.pkl, churn_model.pkl, ltv_model.pkl, conversion_model.pkl, pca.pkl, scaler.pkl, encoders.pkl, etc.)
- data/ - Generated consumer datasets (consumers.csv, consumers_segmented.csv)
- README.md - Comprehensive documentation with architecture diagram, API reference, and setup instructions
