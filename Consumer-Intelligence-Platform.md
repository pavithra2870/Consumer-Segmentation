# Consumer Intelligence Platform

An end-to-end consumer analytics platform. It segments customers, predicts churn, conversion and lifetime value, explains the predictions, audits them for fairness, and shows everything in one dashboard. It runs on one synthetic dataset and three public datasets, and compares its models against standard baselines.

---

# Elevator Pitch

**30 seconds.** I built a customer analytics platform that takes customer data and turns it into segments, churn and conversion predictions, lifetime value estimates, explanations and a fairness check, all in one dashboard. It runs on four datasets: one I generated and three public ones (telecom churn, bank marketing and online retail). I benchmarked my models against logistic regression, random forest and LightGBM on every dataset, and they land within about 0.01 AUC of the best baseline.

**2 minutes.** Companies want to know which customers are about to leave, which are likely to buy, and what each group looks like. This project does that end to end. The backend is FastAPI and scikit-learn. The frontend is React. You pick a dataset from a dropdown and every page changes to that dataset: segments, personas, predictions, SHAP explanations, a fairness audit, a what-if simulator and a benchmarks page.

The part I am most proud of is how I checked the data and the results. My first synthetic dataset had a 55% churn rate and an LTV score of 0.95 R², which looked great but was wrong: the LTV was a formula of the input columns. I found that, fixed the generator, and then validated the whole pipeline on real public data. On Telco churn my model scores AUC 0.838, and on Bank Marketing conversion it scores 0.802, which is typical for those datasets. I also found that a column in the bank data (call duration) leaks the answer and pushes AUC from 0.805 to 0.935, so I removed it.

---

# Product Overview

**Users**: Product managers, marketing analysts and data scientists who want customer insights without writing code.

**Primary workflow**: Pick a dataset, train models from the Data Pipeline page, then explore: segments → personas → predictions → explanations → fairness → what-if → benchmarks.

**What it does**:
- Builds a customer table for each dataset (feature engineering).
- Clusters customers into 6 segments with KMeans (Gaussian Mixture also available).
- Trains churn, conversion and value (LTV) models, depending on what each dataset supports.
- Writes persona profiles for each segment (local LLM, with template fallback).
- Explains models with SHAP.
- Audits fairness with the adverse impact ratio (80% rule) and a parity gap.
- Lets users change customer features and see predictions (what-if).
- Benchmarks every model against baselines.

**Datasets** (the dropdown in the sidebar):

| Dataset | Rows | Supports |
|---|---|---|
| Synthetic Consumers (generated) | 100,000 | Churn, conversion, LTV, segmentation, fairness |
| Telco Churn (IBM) | 7,043 | Churn, segmentation, fairness |
| Bank Marketing (UCI) | 45,211 | Conversion, segmentation, fairness |
| Online Retail II (UCI) | 824,364 transactions → 5,281 customers | Churn, LTV (next 90 days), segmentation |

A page shows only what the chosen dataset supports. Online Retail has no age or gender, so it has no Fairness page.

---

# Problem Statement

Teams struggle to turn customer data into decisions. Custom ML needs data scientists, commercial tools are expensive and closed, and most tools do only one thing (segmentation, or prediction, or explanation). Fairness checks and honest model comparison are rarely included.

This project puts segmentation, prediction, explanation, fairness and benchmarking in one place, runs locally, and works on both synthetic and real public data.

---

# Key Features

## Dataset switcher
One dropdown at the top of the sidebar changes the whole dashboard to another dataset. Each dataset has its own prepared data and its own trained models, so they never overwrite each other. The sidebar shows "(not trained)" for datasets that need training.

## Data generation and feature engineering
The synthetic generator creates customers from 5 hidden types (champions, loyal regulars, bargain hunters, at risk, dormant). Churn is set to about 18% and conversion to about 15%, with hidden noise so labels are never perfectly predictable. Feature engineering adds RFM scores (recency, frequency, monetary), an engagement score, a digital activity score, support burden and session quality.

## Public dataset preparation
- **Telco**: one-hot encoding; blank `TotalCharges` set to 0.
- **Bank Marketing**: removed the leaking `duration` column; recoded `pdays`.
- **Online Retail II**: reads the Excel file once and caches it; drops rows with no customer ID; treats cancellations as returns; builds one row per customer using a **time split** (features from before a cutoff date, targets from the 90 days after).

## Segmentation
KMeans or Gaussian Mixture, PCA for the 2D scatter plot, silhouette score for quality. Segment labels on the public datasets are built from each segment's value and churn (or subscription) rate, for example "High-value, low churn".

## Prediction models
- Churn: Gradient Boosting (200 trees, depth 4)
- Conversion: Gradient Boosting (200 trees, depth 3)
- Value (LTV): Random Forest (150 trees, depth 8), trained on the log of the value

## Personas
A local Ollama model (phi4-mini) writes a persona for each segment. If Ollama is offline, built-in template personas are used instead.

## Explainability
SHAP TreeExplainer on a 500-row sample. Shows global feature importance and example customers.

## Fairness audit
For each protected attribute (gender, income, city tier on synthetic; gender, senior citizen, partner on Telco; age group, marital status, education on Bank), the audit flags the highest-scoring share of customers (same size as the real event rate) and compares groups using the adverse impact ratio and a 10-point parity gap.

## What-if simulator
Sliders for the 10 most important features of the selected dataset, with ranges taken from the data. Features you do not touch are set to the dataset median.

## Benchmarks page
For each dataset and task, our model is compared with a majority-class guess, Logistic Regression, Random Forest and LightGBM on the same train/test split. It reports AUC with a 95% range, PR-AUC, F1, accuracy and lift (classification); R² and MAE (value); silhouette, Davies-Bouldin and stability (segmentation).

---

# Engineering Highlights

**Backend**: FastAPI with one router per area (data, training, segments, personas, predictions, explanations, bias, datasets, benchmarks). Pydantic for validation, pydantic-settings for config.

**Dataset switching on the backend**: every request can carry `?dataset=...`. A small middleware stores that choice in a context variable, and the settings object returns the right data and model folder for it. This means the existing services did not need dataset-specific code. A dataset registry (`services/datasets.py`) describes each dataset: which tasks it supports, which columns are features, which column means "value" and "risk", and what to show in segment and persona views.

**Frontend**: React 18, Vite, React Router, Recharts, Axios. A shared dataset context feeds every page. The request layer adds the selected dataset to every API call. Pages remount when the dataset changes so no old data is shown.

**Data-driven pages**: the Predictions, Explainability, Overview, What-If and Fairness pages build themselves from what the backend says the dataset supports, so there is no separate page per dataset.

**Model persistence**: Joblib for models and preprocessing objects, saved per dataset. Metrics, what-if ranges and medians are saved as JSON at training time.

**Benchmark runner**: one shared train/test split per task, bootstrap confidence ranges for AUC, and results written as one JSON file per dataset.

**Reliability**: error handling, Loguru logging, a health endpoint, and graceful fallback when Ollama or SHAP is missing.

**Testing done**: I rendered every page for all four datasets in a headless DOM test to check nothing crashes (36 page and dataset combinations), checked the API for each dataset, and took screenshots from the running app.

---

# AI / ML Components

**Supervised models**: Gradient Boosting classifiers for churn and conversion, Random Forest regressor for value. Value is trained on `log(1 + value)` because it is very skewed.

**Clustering**: KMeans (default) and Gaussian Mixture, 6 clusters by default. Compared with Agglomerative (Ward) and random assignment in the benchmarks.

**Explainability**: SHAP TreeExplainer for all three model types.

**Fairness**: adverse impact ratio (80% rule) and demographic parity gap.

**LLM**: Ollama with phi4-mini writes personas from segment statistics, requiring a strict JSON format, with template fallback if the reply is missing or invalid.

**Evaluation**: AUC, PR-AUC, F1, accuracy, lift at 10% (classification); R² on the log scale and MAE (regression); silhouette, Davies-Bouldin and seed stability (clustering). Every classifier is also compared with the "always guess the majority" baseline.

---

# Data Engineering / Data Science Components

**Synthetic data**: numpy distributions, hidden customer types, calibrated rates, hidden noise, Faker names, 12-month activity as JSON per customer.

**Calibration**: for churn and conversion, the generator finds the intercept (by bisection) that makes the average probability equal the target rate (18% churn, 15% conversion).

**Time-based split (Online Retail)**: cutoff = last transaction date minus 90 days. Features only use earlier data. Churn = no purchase after the cutoff. Value = spend after the cutoff. This stops future information from leaking into the features.

**Leakage handling**: three cases found and handled (see Results section).

**Imbalanced data**: stratified splits, and metrics that suit rare events (AUC, PR-AUC, lift) instead of accuracy alone.

**Metrics storage**: each dataset saves its metrics, what-if ranges and medians when it is trained.

---

# Results (the numbers to remember)

Full tables and screenshots: [results.md](results.md).

## Model results (AUC unless noted; 80/20 split)

| Dataset | Task | Our model | Best baseline | Majority guess |
|---|---|---|---|---|
| Synthetic | Churn | 0.840 | Logistic Regression 0.842 | 0.500 |
| Synthetic | Conversion | 0.807 | Logistic Regression / LightGBM 0.807 | 0.500 |
| Synthetic | LTV (R², log) | 0.609 | LightGBM 0.626 | 0.000 |
| Telco | Churn | 0.838 | Random Forest 0.843 | 0.500 |
| Bank Marketing | Conversion | 0.802 | LightGBM 0.805 | 0.500 |
| Online Retail | Churn (90 days) | 0.802 | Random Forest 0.813 | 0.500 |
| Online Retail | Value (R², log) | 0.371 | LightGBM 0.315 | 0.000 |

On Online Retail value, our model is also best on MAE (549 vs 580 for LightGBM and 694 for the mean).

## Event rates and baselines

| Dataset | Event | Rate | Majority-guess accuracy | Our accuracy |
|---|---|---|---|---|
| Synthetic | Churn | 18.1% | 81.9% | 85.3% |
| Synthetic | Conversion | 14.8% | 85.2% | 86.1% |
| Telco | Churn | 26.5% | 73.5% | 80.0% |
| Bank | Subscribed | 11.7% | 88.3% | 89.6% |
| Retail | No purchase in 90 days | 56.6% | 56.6% | 74.0% |

## Lift in the top 10% of scored customers

| Dataset | Task | Lift |
|---|---|---|
| Synthetic | Churn | 3.7× |
| Telco | Churn | 2.9× |
| Bank | Conversion | 4.5× |
| Retail | Churn | 1.6× |

## Segmentation (KMeans, 6 clusters, benchmark silhouette on a 10,000-row sample)

| Dataset | Silhouette | Davies-Bouldin | Stability (ARI) |
|---|---|---|---|
| Synthetic | 0.119 | 1.82 | 0.983 |
| Telco | 0.319 | 1.24 | 0.817 |
| Bank | 0.240 | 1.29 | 0.867 |
| Retail | 0.287 | 1.06 | 0.766 |

KMeans has the best silhouette on all four datasets. Random assignment scores about 0 everywhere.

## Segment examples (Telco churn)

The Telco segments separate churn clearly: from 4.1% (high-value, low churn) up to 54.4% (low-value, high churn).

## Explainability: top drivers

| Dataset | Model | Top features |
|---|---|---|
| Synthetic | Churn | rfm_score, engagement_score, support_burden, nps_score |
| Synthetic | Conversion | engagement_score, email_click_rate, pages_per_session |
| Synthetic | LTV | subscription type, total_spend, engagement_score |
| Telco | Churn | tenure, two-year contract, fibre-optic internet, TotalCharges |
| Bank | Conversion | previous campaign success, age, day of month, days since last contact |
| Retail | Churn | average days between orders, days since last order, total spend |

## Fairness audit

| Dataset | Checks flagged | Main finding |
|---|---|---|
| Synthetic | 5 of 20 | All on income bracket |
| Telco | 2 of 6 | Senior-citizen and partner status |
| Bank | 7 of 11 | Mostly age group (under-30s subscribe at 23.5% vs 7–8% for 30–59) |
| Retail | Not available | No demographic columns |

---

# Benchmarks

**Method**
- One stratified 80/20 split per task (seed 42), shared by all models, so the comparison is fair.
- Models: majority class, Logistic Regression (scaled), Random Forest (200 trees, depth 12), LightGBM (300 trees), and our model.
- AUC comes with a 95% range from 200 bootstrap resamples of the test set.
- Lift @10%: among the 10% highest-scored customers, how many times more often the event happens than average.

**Reading the results**
- Our model is within about 0.01 AUC of the best baseline on every churn and conversion task.
- The confidence ranges overlap, so "equal" is the honest conclusion, not "better".
- Bank Marketing: Logistic Regression scored 0.772 against 0.802 for Gradient Boosting, so I switched the conversion model from logistic regression to gradient boosting.
- Online Retail value is the one place our model is clearly best (R² 0.371 vs 0.315 for LightGBM).
- On the synthetic data, Logistic Regression ties the boosted models because the labels come from simple rules I wrote.

**Leakage check**: with the call-length column left in, LightGBM on Bank Marketing scores AUC 0.935 instead of 0.805.

---

# How the Data Was Checked and Fixed

| Check | First synthetic version | Fixed version |
|---|---|---|
| Churn rate | 55.4% | 18.1% |
| Conversion rate | 43.9% | 14.8% |
| Churn accuracy vs ceiling | 64.5% vs about 65.1% best possible | AUC 0.840 |
| LTV R² | 0.9495 (formula leak) | 0.609 |
| LTV correlation with total_spend | 0.76 | 0.34 |

How I found the problems: I computed the base rates and the best possible score using the hidden true probabilities (the "oracle"). The model was already at the ceiling, which told me accuracy was capped by the labels, not by the model. I read the LTV formula in the generator and saw it was built from the same columns the model used.

---

# Interview Prep

## The 5 things to say first

1. It is end to end: data, features, segmentation, three kinds of models, explanations, fairness, dashboard, benchmarks.
2. It runs on four datasets, one synthetic and three real public ones, and the whole dashboard switches with a dropdown.
3. I benchmarked against baselines with the same split, and my models are within about 0.01 AUC of the best one. That is an honest result, not a win claim.
4. I found and fixed three data problems: a bad synthetic dataset (55% churn, LTV leak), a leaking column in the bank data, and the need for a time split in the retail data.
5. I know the limits: synthetic scores mostly measure my generator, and the retail churn label is one I defined.

## Number cheat sheet

- Datasets: 100,000 / 7,043 / 45,211 / 824,364 transactions (5,281 customers).
- Churn AUC: Telco 0.838, Synthetic 0.840, Retail 0.802.
- Conversion AUC: Bank 0.802, Synthetic 0.807.
- LTV R² (log): Synthetic 0.609, Retail 0.371.
- Bank leak: 0.805 → 0.935 with `duration`.
- Event rates: churn 18.1% (synthetic), 26.5% (Telco), 56.6% (Retail); conversion 14.8% (synthetic), 11.7% (Bank).
- Silhouette (KMeans k=6): Synthetic 0.119, Telco 0.319, Bank 0.240, Retail 0.287.
- Fairness: 5/20, 2/6, 7/11 flagged on synthetic, Telco, Bank.

## Questions and answers

**1. Walk me through the project.**
Use the 2-minute pitch above. Then offer to go deeper on data validation, benchmarking, or the architecture.

**2. Why four datasets?**
The synthetic data lets me test the full pipeline with known rules. The public datasets show the pipeline works on real data, and give numbers other people have also reported. Each public dataset covers different tasks: Telco for churn, Bank for conversion, Online Retail for segmentation, churn and value from raw transactions.

**3. How did you know your synthetic data was good enough?**
I measured it. I checked the churn rate (55%, too high), computed the best possible accuracy using the hidden true probabilities, and read the LTV formula. That showed the model was already at the ceiling and that LTV was a formula of the inputs. I fixed those, then checked again: churn 18%, conversion 15%, LTV R² 0.61.

**4. Why AUC and not accuracy?**
With rare events accuracy is misleading. On Bank Marketing, "nobody subscribes" is 88.3% accurate. My model gets 89.6%. That looks like a tiny gain. AUC shows the model ranks customers well (0.802 vs 0.5 for guessing), and lift shows the top 10% of scored customers subscribe 4.5 times more often than average.

**5. Your model does not beat LightGBM. Why present it?**
On this kind of table data, boosted trees, random forests and logistic regression usually land close together, and that is what I found. The confidence ranges overlap. What the benchmark proves is that my pipeline is not broken: it matches strong standard models on real data. I would not claim it is better.

**6. What is data leakage? Did you hit it?**
Leakage means the model sees information it would not have at prediction time. I hit it three times. In the bank data, call duration is only known after the call, so I removed it (AUC 0.805 with it removed, 0.935 with it in). In my first synthetic data, LTV was a formula of the spend columns, giving R² 0.95. In the retail data, I used a time split so features come only from before the cutoff.

**7. How did you define churn for Online Retail?**
The dataset has no churn column. I picked a cutoff date 90 days before the last transaction. Features come from before the cutoff. A customer churned if they made no purchase in the 90 days after. 56.6% did not buy, which is high but normal for a shop with many one-time buyers. It is my definition, so I say so.

**8. How do you handle imbalanced classes?**
Stratified splits, so each split has the same event rate. I use AUC, PR-AUC and lift instead of accuracy alone. I did not resample, and the models output probabilities that I rank on. For the fairness audit I flag the top share of customers matching the real event rate, because a 0.5 cutoff would flag almost nobody.

**9. Why did you switch the conversion model from logistic regression?**
In the benchmark on Bank Marketing, logistic regression scored 0.772 AUC and gradient boosting scored 0.802. It also let me use the same SHAP explainer for all three models.

**10. What does the silhouette score tell you, and why is it low on synthetic data?**
It measures how well separated clusters are, from −1 to 1. Real datasets scored 0.24 to 0.32. Synthetic scored 0.12, meaning its customer groups overlap a lot. I say that plainly: the synthetic segments are soft groupings. Also, the dashboard computes silhouette on the first 5,000 rows, while the benchmark uses a random 10,000, so the benchmark number is the more reliable one.

**11. How did you pick 6 clusters?**
It is the default setting. The benchmark sweep from 2 to 8 shows the best silhouette is at 3 (synthetic), 8 (Telco), 8 (Bank) and 2 (Retail), so there is no single right answer. Six is a business-friendly number of groups, not a statistical optimum.

**12. How stable are your clusters?**
I refit KMeans with 3 different random seeds and compared the results with the adjusted Rand index. It was 0.98 on synthetic, 0.82 on Telco, 0.87 on Bank and 0.77 on Retail. Closer to 1 means more stable.

**13. What is lift?**
Take the 10% of customers the model scores highest. Lift is how many times more often the event happens in that group than in the whole population. Bank conversion lift is 4.5×, so calling the top 10% finds 4.5 times more subscribers per call than calling at random.

**14. What do the bootstrap confidence ranges mean?**
I resampled the test set 200 times and recomputed AUC each time. The 2.5th to 97.5th percentile is the range. Telco has only 1,409 test customers, so its range is about ±0.02, and small differences between models there are noise.

**15. Why do you predict log of LTV?**
Value is heavily skewed (a few customers spend a lot). Training on the log makes the model focus on proportional errors instead of the biggest customers. R² is then reported on the log scale, and MAE on the real scale.

**16. Why is Retail value R² only 0.37?**
56.6% of customers spend nothing in the next 90 days, so the target is mostly zeros. Our model is still the best of the four on R² and on MAE.

**17. What does SHAP show?**
How much each feature pushed a prediction up or down compared with the average, for the whole dataset and for single customers. On Telco, the top drivers are tenure, two-year contract and fibre-optic internet.

**18. What did the fairness audit find, and what does it mean?**
It compares flag rates across groups. On Bank Marketing, under-30s are flagged at 23.5% against 7–8% for 30–59-year-olds. That reflects a real pattern in the data (young people subscribe more), so the audit flags it for review. A flag is a prompt to investigate, not proof of unfair treatment.

**19. How does the dataset switch work?**
Each request carries a `dataset` value. A middleware stores it in a context variable, and the settings object returns the data and model folder for that dataset, so the existing services needed no changes. A registry describes each dataset's tasks, feature columns and display columns. The frontend keeps the choice in a context and remounts pages when it changes.

**20. How does the what-if tool work?**
The backend picks the top 10 features by importance and takes slider ranges from the 1st to 99th percentile of the data. Any feature the user does not set is filled with the dataset median.

**21. How do you handle the LLM being unavailable?**
Persona generation checks whether Ollama is reachable and parses the reply as strict JSON. If either fails, it uses built-in template personas. All other features work without Ollama.

**22. What would you do differently?**
I would be careful about what I claim. The synthetic scores mostly test my own generator, and the retail churn label is my own definition. I would also not trust the synthetic clusters as real customer types.

## Things not to claim

- Do not say the models beat the baselines. They tie them within error.
- Do not present synthetic-data scores as real-world performance.
- Do not oversell the personas. A small model (phi4-mini) writes them, so names repeat (four of six were "John Smith") and it sometimes misreads a number. If Ollama is offline they fall back to built-in templates.
- Do not call the retail churn rate a business fact. It is a definition.
- Do not say the segment names on the synthetic data are discovered. They are fixed labels by cluster number.

---

# Product Thinking

**For non-technical users**: pages explain themselves with plain labels (for example "High-value, low churn"). Charts and tables show the numbers a marketer would ask for.

**Honest comparison**: the Benchmarks page shows the baselines next to our model, with ranges, so users can judge it.

**Decision support**: personas and recommended actions turn segments into actions. The what-if tool lets a user test an idea before spending money.

**Graceful behaviour**: pages hide features a dataset cannot support instead of showing empty charts. Ollama being offline does not block anything.

---

# Technologies Used

**Languages**: Python (developed on 3.10), JavaScript

**Frontend**: React 18.2, Vite 5, React Router 6.20, Recharts 2.10, Axios 1.6

**Backend**: FastAPI 0.109, Uvicorn 0.27, Pydantic 2.5, Pydantic Settings 2.1

**Machine learning**: scikit-learn 1.4, LightGBM 4.3, SHAP 0.44, pandas 2.1, numpy 1.26, scipy 1.11, joblib 1.3

**Data**: openpyxl 3.1 (reads the retail Excel file), pyarrow 14, Faker 21

**LLM**: Ollama (phi4-mini), httpx 0.26

**Utilities**: python-dotenv, aiofiles, loguru

---

# Skills Demonstrated

- Full-stack development (React and FastAPI)
- Machine learning engineering: feature engineering, training, evaluation
- Benchmarking against baselines with confidence ranges
- Handling imbalanced data and choosing the right metrics
- Spotting and fixing data leakage
- Time-based train/feature splits
- Clustering and cluster quality (silhouette, Davies-Bouldin, stability)
- Regression on skewed targets (log transform)
- Model explainability (SHAP)
- Fairness auditing (adverse impact ratio, parity gap)
- Synthetic data design and validation
- Working with public datasets (UCI, Kaggle) and their licences
- API design and per-request configuration
- Data visualisation (Recharts)
- LLM integration with fallback
- Honest reporting of results and limits

---

# Resume Impact

- Built an end-to-end consumer analytics platform (FastAPI, React, scikit-learn) with segmentation, churn, conversion and LTV models, SHAP explanations, a fairness audit, and a what-if simulator.
- Made the whole dashboard switch between a synthetic dataset and three public datasets (Telco churn, Bank Marketing, Online Retail II) with one dropdown, using a per-request dataset setting and a dataset registry.
- Benchmarked the models against majority-class, Logistic Regression, Random Forest and LightGBM on a shared split: churn AUC 0.838 on Telco and 0.802 on Online Retail, conversion AUC 0.802 on Bank Marketing, each within about 0.01 of the best baseline.
- Found and fixed data leakage: removed the call-duration column from Bank Marketing (AUC 0.935 → 0.805 honest score), and rebuilt a synthetic generator whose LTV was a formula of its inputs (R² 0.95 → 0.61) and whose churn rate was unrealistic (55% → 18%).
- Built a 90-day time-split churn and spend target from 824k raw retail transactions (5,281 customers), keeping features strictly before the cutoff date.
- Added a fairness audit (80% adverse impact rule) that flagged income gaps in synthetic data and age gaps in Bank Marketing (under-30s subscribe at 23.5% vs 7–8% for 30–59).
- Created a local-LLM persona generator (Ollama) with a template fallback, and tested every dashboard page across all four datasets in a headless DOM test.

---

# Ideal Roles

- Machine Learning Engineer
- Data Scientist
- Applied Scientist
- AI Engineer
- Full Stack Engineer
- Analytics Engineer
- Data Platform Engineer
- Technical Product Manager

---

# Portfolio Tags

React · FastAPI · Machine Learning · scikit-learn · LightGBM · Clustering · Churn Prediction · Benchmarking · Data Leakage · SHAP · Explainability · Fairness · Feature Engineering · Public Datasets · Dashboard · Python · JavaScript · API Design

---

# Project Complexity

Advanced

It combines a full-stack app, a multi-dataset ML pipeline, honest benchmarking, explainability and fairness. The hardest parts were not the models but the data work: validating the synthetic data, finding leakage, building a time-split target from raw transactions, and making one dashboard work across datasets that support different tasks.

---

# One-line Portfolio Summary

End-to-end consumer analytics platform with segmentation, churn, conversion and LTV models, SHAP, fairness auditing, and benchmarks on four datasets, switchable from one dropdown.

---
