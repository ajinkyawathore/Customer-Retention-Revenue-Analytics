# 📊 Customer Retention & Revenue Analytics

> A complete end-to-end data analytics pipeline for predicting customer churn, scoring revenue risk, and surfacing actionable retention insights — built on a real-world SaaS-style dataset.

---

## 🗂️ Project Overview

| Item | Detail |
|---|---|
| **Domain** | SaaS / Subscription Business |
| **Goal** | Predict churn, segment customers by risk, quantify MRR at risk |
| **Dataset** | 7 relational tables — 3 fact + 4 dimension |
| **Models** | Logistic Regression (baseline) · Random Forest (best) |
| **Output** | Risk-scored customer list · Saved model · Cohort heatmap · Feature importance |
| **Stack** | Python · Pandas · Scikit-learn · Seaborn · Matplotlib |

---

## 🏗️ Architecture

```
Raw CSVs (data/raw/)
       │
       ▼
┌─────────────────────┐
│  Step 1             │  Data Ingestion & Cleaning
│  data_cleaning.ipynb│  → date parsing, deduplication, outlier capping
└────────┬────────────┘
         │  data/cleaned/
         ▼
┌─────────────────────┐
│  Step 2             │  Build Master Analytics Table (MAT)
│  build_mat.ipynb    │  → join 7 tables, feature engineering, RFM scoring
└────────┬────────────┘
         │  data/mat/master_analytics_table.csv
         ▼
┌─────────────────────┐
│  Step 3             │  EDA & Cohort Analysis
│  eda_cohort.ipynb   │  → 5 charts: churn by plan/channel, retention heatmap
└────────┬────────────┘
         │  outputs/charts/
         ▼
┌─────────────────────┐
│  Step 4             │  Churn Prediction Model
│  churn_model.ipynb  │  → train/val/test split, pipeline, feature importance
└────────┬────────────┘
         │  outputs/model/
         ▼
   churn_model.pkl  +  at_risk_customers.csv
```

---

## 📁 Repository Structure

```
project/
├── data/
│   ├── raw/                        ← original CSV files (never edited)
│   │   ├── dim_users.csv
│   │   ├── dim_plans.csv
│   │   ├── dim_geography.csv
│   │   ├── dim_acquisition_channel.csv
│   │   ├── fact_subscriptions.csv
│   │   ├── fact_payments.csv
│   │   └── fact_activity_logs.csv
│   ├── cleaned/                    ← cleaned files (output of Step 1)
│   └── mat/                        ← master analytics table (output of Step 2)
│       └── master_analytics_table.csv
│
├── outputs/
│   ├── charts/                     ← all generated visualisations
│   │   ├── 01_churn_overview.png
│   │   ├── 02_cohort_retention_heatmap.png
│   │   ├── 03_rfm_scores.png
│   │   ├── 04_mrr_at_risk.png
│   │   ├── 05_adoption_vs_churn.png
│   │   ├── 06_feature_importance.png
│   │   └── 07_roc_and_confusion.png
│   └── model/
│       ├── churn_model.pkl         ← serialised best model
│       └── at_risk_customers.csv  ← scored customer list (churn prob > 50%)
│
├── step01_data_cleaning.ipynb
├── step02_build_mat.ipynb
├── step03_eda_cohort_analysis.ipynb
├── step04_churn_model.ipynb
├── requirements.txt
└── README.md
```

---

## 📦 Dataset — Table Reference

### Dimension Tables (the "who" and "what")

| Table | Key Column | Description |
|---|---|---|
| `dim_users` | `user_id` | Customer profile — gender, NPS score, signup date, company size |
| `dim_plans` | `plan_id` | Plan tiers (Basic / Standard / Pro / Enterprise) |
| `dim_geography` | `geo_id` | Region, country, city |
| `dim_acquisition_channel` | `channel_id` | How the customer was acquired (Organic, Paid, Referral, etc.) |

### Fact Tables (the "what happened")

| Table | Key Column | Description |
|---|---|---|
| `fact_subscriptions` | `user_id` | One row per customer — plan, MRR, tenure, churn flag |
| `fact_payments` | `payment_id` | All payment transactions — amount, status, refunds, failures |
| `fact_activity_logs` | `activity_id` | Granular event log — logins, features used, session duration, errors |

---

## ⚙️ Step-by-Step Breakdown

### Step 1 — Data Ingestion & Cleaning (`step01_data_cleaning.ipynb`)

**Problems detected and fixed:**

| Table | Issue | Fix Applied |
|---|---|---|
| `dim_users` | `signup_date` stored as text | Converted to `datetime` |
| `dim_users` | Duplicate `user_id` rows | `drop_duplicates()` |
| `dim_users` | Missing `nps_score` values | Filled with column median |
| `fact_subscriptions` | Negative `mrr_usd` values | Clipped to 0 |
| `fact_subscriptions` | Missing `start_date` rows | Dropped (unusable without it) |
| `fact_payments` | Outlier `amount_usd` values | IQR capping (Q1−3×IQR to Q3+3×IQR) |
| `fact_payments` | Duplicate `payment_id` rows | `drop_duplicates()` |
| `fact_activity_logs` | Missing `session_duration_min` | Filled with 0 |
| `fact_activity_logs` | Duplicate `activity_id` rows | `drop_duplicates()` |

**Outputs:** `data/cleaned/*.csv`

---

### Step 2 — Build Master Analytics Table (`step02_build_mat.ipynb`)

**Why a MAT?**
Machine learning requires one row per customer with all information. The raw data spans 7 tables and ~1.7 million activity rows. Step 2 compresses everything into a single wide table.

**Activity aggregation (per user):**

| Feature Created | Aggregation |
|---|---|
| `total_events` | COUNT of all activity records |
| `total_logins` | COUNT where `event_type == 'login'` |
| `distinct_features` | NUNIQUE of `feature_used` |
| `avg_session_dur` | MEAN of `session_duration_min` |
| `last_activity_date` | MAX of `event_date` |
| `total_errors` | SUM of `error_occurred` |
| `mobile_events` | SUM of `is_mobile` |

**Payment aggregation (per user):**

| Feature Created | Aggregation |
|---|---|
| `total_payments` | COUNT of transactions |
| `total_revenue` | SUM of `amount_usd` |
| `avg_payment_amt` | MEAN of `amount_usd` |
| `failed_payments` | SUM of `is_failed` |
| `refunded_payments` | SUM of `is_refunded` |

**Feature engineering (derived columns):**

| New Feature | Formula | Business Meaning |
|---|---|---|
| `days_since_last_login` | snapshot_date − last_activity_date | Recency signal |
| `days_since_last_payment` | snapshot_date − last_payment_date | Payment recency |
| `avg_monthly_spend` | total_revenue ÷ (tenure_days / 30) | Revenue commitment |
| `feature_adoption_score` | distinct_features clipped to 10 | Product depth usage |
| `login_frequency_monthly` | total_logins ÷ active months | Engagement cadence |
| `failed_payment_rate` | failed_payments ÷ total_payments | Billing health signal |
| `events_per_day` | total_events ÷ tenure_days | Overall activity level |
| `cohort_month` | `signup_date` truncated to month | Cohort grouping key |

**RFM Scoring (1–5 per dimension, total 3–15):**

| Dimension | Logic | Score 5 means… |
|---|---|---|
| **Recency** | Days since last login (fewer = better) | Logged in very recently |
| **Frequency** | Total logins (more = better) | Logs in very frequently |
| **Monetary** | Avg monthly spend (higher = better) | High-value customer |

**Risk Segmentation:**

| RFM Total Score | Segment |
|---|---|
| ≥ 11 | 🟢 Low Risk |
| 8–10 | 🟡 Medium Risk |
| < 8 | 🔴 High Risk |

**Output:** `data/mat/master_analytics_table.csv`

---

### Step 3 — EDA & Cohort Analysis (`step03_eda_cohort_analysis.ipynb`)

**Charts produced:**

| # | Chart | Key Question Answered |
|---|---|---|
| 1 | Churn Rate by Plan, Channel & Risk Pie | Which plan/channel loses the most customers? |
| 2 | Cohort Retention Heatmap | At which tenure month do customers drop off? |
| 3 | RFM Score Distributions | How healthy is the customer base overall? |
| 4 | MRR at Risk by Segment | How much revenue is threatened right now? |
| 5 | Feature Adoption: Churned vs Retained | Do power users stay longer? |

**Cohort heatmap interpretation:**
- Rows = signup month (cohort)
- Columns = months since signup (tenure)
- Cell value = % of cohort still active at that tenure
- Red cells = high dropout zones → intervention opportunities

---

### Step 4 — Churn Prediction Model (`step04_churn_model.ipynb`)

**Data split:**

| Set | Size | Purpose |
|---|---|---|
| Training | 70% | Model learns patterns from this |
| Validation | 15% | Tuning and intermediate evaluation |
| Test | 15% | Final evaluation — never seen during training |

**Preprocessing pipeline:**
- Numeric features → `StandardScaler` (mean=0, std=1 normalisation)
- Categorical features → `OneHotEncoder` (text → binary columns)
- Wrapped in `sklearn.Pipeline` to prevent data leakage

**Models trained:**

| Model | Description | Hyperparameters |
|---|---|---|
| Logistic Regression | Linear baseline | `C=0.1`, `class_weight=balanced`, `max_iter=1000` |
| Random Forest ⭐ | Ensemble of 200 decision trees | `n_estimators=200`, `max_depth=10`, `class_weight=balanced` |

**Evaluation metric: AUC-ROC**
- 0.5 = random guessing
- 1.0 = perfect separation
- Target: > 0.85

**Feature Importance (top predictors of churn):**
`days_since_last_login` · `rfm_score` · `failed_payment_rate` · `avg_monthly_spend` · `feature_adoption_score` · `login_frequency_monthly` · `tenure_days`

**Outputs:**
- `outputs/model/churn_model.pkl` — serialised best model
- `outputs/model/at_risk_customers.csv` — all customers with churn probability > 50%, sorted by risk

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/your-username/customer-retention-analytics.git
cd customer-retention-analytics
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add raw data
Place all 7 CSV files into `data/raw/`.

### 4. Run notebooks in order
```
step01_data_cleaning.ipynb      →  cleans raw data
step02_build_mat.ipynb          →  builds the master analytics table
step03_eda_cohort_analysis.ipynb →  generates all EDA charts
step04_churn_model.ipynb        →  trains models, saves at-risk list
```

---

## 🔧 Requirements

```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
matplotlib>=3.7
seaborn>=0.12
```

---

## 💡 Key Business Insights

1. **Feature adoption is the strongest retention signal** — customers who use more product features have significantly lower churn rates. Onboarding campaigns that drive feature discovery directly reduce churn.

2. **RFM score is a reliable early-warning system** — High-Risk (RFM < 8) customers churn at a rate far exceeding Low-Risk customers. Intervening at the RFM "Medium Risk" threshold is more cost-effective than waiting for High Risk.

3. **Payment failure is a leading churn indicator** — A rising `failed_payment_rate` often precedes voluntary cancellation. Automated dunning and retry logic can recover these customers before they leave.

4. **Cohort analysis reveals the critical dropout window** — The retention heatmap pinpoints the exact tenure months where each cohort loses the most customers, enabling time-targeted outreach.

5. **Acquisition channel quality varies significantly** — Churn rate differs by channel, meaning not all growth is equal. High-churn channels inflate customer counts but compress lifetime value.

---

## 📬 Contact

Built by **[Ajinkya Wathore]**
[LinkedIn]((https://www.linkedin.com/in/ajinkya-wathore-a94405245/) · [Email](ajinkyawg@email.com)
